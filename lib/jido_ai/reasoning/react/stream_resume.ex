defmodule Jido.AI.Reasoning.ReAct.StreamResume do
  @moduledoc """
  Rebuilding a turn whose stream died in transport.

  A long provider turn can lose its connection before the turn finishes — a
  proxy or gateway that caps connection lifetime kills the socket mid-stream
  even though the provider is still working. The work itself survives on the
  provider side (an Anthropic code-execution container outlives the request),
  so the turn can be continued instead of failed: replay the assistant content
  the dead stream had already produced and ask the provider to carry on.

  Two rules make that replay legal, both verified against the Anthropic API:

    * Trailing blocks may be OMITTED. Cutting the assistant message short at a
      block boundary is accepted.
    * Blocks may NOT be REORDERED. A thinking block that moves relative to a
      non-text block is rejected ("`thinking` blocks ... cannot be modified").

  So the captured chunk list is truncated at the last completed server-tool
  result block and replayed in arrival order — never re-bucketed by type.
  Everything after that boundary (a half-written text block, a thinking block
  whose signature never arrived) is dropped, and the provider regenerates it.
  """

  alias Jido.AI.Turn
  alias ReqLLM.StreamChunk

  # Every server tool answers under its own result type, all of them ending in
  # `_tool_result`: the three code-execution sub-tools, web search, and whatever
  # the provider adds next. Each arrives as a single complete block, so seeing
  # one in the chunk list is proof the block closed — there is no partial form
  # to guard against.
  @boundary_block_suffix "_tool_result"

  # Matched by name rather than by struct so this module keeps working without a
  # compile-time dependency on the HTTP client's error structs.
  @transport_error_structs [Finch.TransportError, Mint.TransportError]

  # The cause chain is a handful of terms deep; the bound stops the walk from
  # wandering into a large struct that happens to hang off an error.
  @max_cause_depth 6

  @doc """
  Whether a failed stream died from a transport close, the one failure worth
  resuming.

  A closed connection says nothing about the turn — the provider kept working.
  Every other failure (a refusal, a rate limit, a decode error) means the turn
  itself is broken, and repeating it would only repeat the failure. The check
  walks the error's cause chain because the close is wrapped several times over
  by the time it reaches the runner.
  """
  @spec resumable_error?(term()) :: boolean()
  def resumable_error?(error), do: closed_transport?(error, @max_cause_depth)

  @doc """
  Truncates a captured chunk list to its last complete server-tool result.

  Returns `{:ok, chunks, blocks_kept}` with the chunks up to and including that
  block, or `:no_boundary` when the stream died before any result arrived.
  """
  @spec truncate_at_last_tool_result([StreamChunk.t()]) ::
          {:ok, [StreamChunk.t()], pos_integer()} | :no_boundary
  def truncate_at_last_tool_result(chunks) when is_list(chunks) do
    case last_boundary_index(chunks) do
      nil ->
        :no_boundary

      index ->
        kept = Enum.take(chunks, index + 1)
        {:ok, kept, Enum.count(kept, &boundary?/1)}
    end
  end

  def truncate_at_last_tool_result(_chunks), do: :no_boundary

  @doc """
  Builds the partial turn a resume replays from a captured chunk list.

  Returns `{:ok, turn, blocks_kept}`, or `:none` when there is no complete
  result block to cut at — in which case the caller re-requests the turn with
  its context unchanged, which is a plain reconnect.

  The turn is assembled by the same provider response builder the successful
  path uses, so the replayed message has the shape the provider already accepts
  from a paused-turn continuation.
  """
  @spec partial_turn([StreamChunk.t()], ReqLLM.StreamResponse.t(), String.t() | nil) ::
          {:ok, Turn.t(), pos_integer()} | :none
  def partial_turn(chunks, %ReqLLM.StreamResponse{} = stream_response, model_label) do
    with {:ok, kept, blocks_kept} <- truncate_at_last_tool_result(chunks),
         {:ok, response} <- build_response(kept, stream_response) do
      {:ok, Turn.from_response(response, model: model_label), blocks_kept}
    else
      _ -> :none
    end
  end

  def partial_turn(_chunks, _stream_response, _model_label), do: :none

  defp build_response(chunks, %ReqLLM.StreamResponse{model: model, context: context}) do
    builder = ReqLLM.Provider.ResponseBuilder.for_model(model)

    # No metadata: the metadata handle carries the stream's failure, not the
    # usage and finish reason a completed turn would report. The builder fills
    # in the absent fields, and the container id rides the captured chunks.
    builder.build_response(chunks, %{}, context: context, model: model)
  rescue
    _ -> :error
  end

  defp last_boundary_index(chunks) do
    chunks
    |> Enum.with_index()
    |> Enum.reduce(nil, fn {chunk, index}, last ->
      if boundary?(chunk), do: index, else: last
    end)
  end

  defp boundary?(%StreamChunk{type: :meta, metadata: %{provider_block: %{"type" => type}}})
       when is_binary(type),
       do: String.ends_with?(type, @boundary_block_suffix)

  defp boundary?(_chunk), do: false

  defp closed_transport?(_term, depth) when depth < 0, do: false

  defp closed_transport?(%{__struct__: module, reason: :closed}, _depth)
       when module in @transport_error_structs,
       do: true

  defp closed_transport?(%{__struct__: _} = struct, depth) do
    struct |> Map.from_struct() |> any_closed_transport?(depth)
  end

  defp closed_transport?(term, depth) when is_map(term), do: any_closed_transport?(term, depth)

  defp closed_transport?(term, depth) when is_list(term) do
    Enum.any?(term, &closed_transport?(&1, depth - 1))
  end

  defp closed_transport?(term, depth) when is_tuple(term) do
    term |> Tuple.to_list() |> Enum.any?(&closed_transport?(&1, depth - 1))
  end

  defp closed_transport?(_term, _depth), do: false

  defp any_closed_transport?(map, depth) do
    map |> Map.values() |> Enum.any?(&closed_transport?(&1, depth - 1))
  end
end

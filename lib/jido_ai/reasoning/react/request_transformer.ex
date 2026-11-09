defmodule Jido.AI.Reasoning.ReAct.RequestTransformer do
  @moduledoc """
  Behavior for advanced per-turn ReAct request shaping.

  A request transformer can inspect the current runtime state and tool context
  before each LLM turn, then return request overrides.

  This is intended for patterns such as:

  - request-scoped tool gating
  - dynamic structured-output schemas
  - provider-specific `llm_opts` based on tool results
  - custom message projection beyond the default context rendering
  - **per-turn model selection** — returning `model:` in the overrides swaps
    which provider handles the next LLM turn. `llm_opts.provider_options`
    overrides are re-validated against the selected model's provider schema,
    so xAI-only keys (e.g. `xai_api`) can be added on xAI turns without
    breaking Fireworks turns on the same agent.

  The runtime always regenerates `llm_opts[:tools]` from the returned `tools`
  field so the exposed LLM tools and execution registry stay aligned.

  A transformer can also implement `c:transform_tool_results/2` to rewrite a
  tool round's results. The calls of one round run concurrently against one
  state snapshot, so no call can see another's result. The callback runs once
  every call of the round has finished, before any result is reported in a
  `tool_completed` event, appended to the context, or applied as an effect.
  """

  alias Jido.AI.Reasoning.ReAct.{Config, State, ToolSelection}

  @type request :: %{
          required(:messages) => [map()],
          required(:llm_opts) => keyword(),
          required(:tools) => ToolSelection.tools_input(),
          required(:model) => term()
        }

  @type overrides :: %{
          optional(:messages) => [map()],
          optional(:llm_opts) => keyword() | map(),
          optional(:tools) => ToolSelection.tools_input(),
          optional(:model) => term()
        }

  @doc """
  Validate a request transformer module.
  """
  @spec validate(module() | nil) ::
          {:ok, module() | nil}
          | {:error, :invalid_request_transformer}
          | {:error, {:request_transformer_not_loaded, module()}}
          | {:error, {:request_transformer_missing_callback, module()}}
  def validate(nil), do: {:ok, nil}

  def validate(module) when is_atom(module) do
    cond do
      not Code.ensure_loaded?(module) ->
        {:error, {:request_transformer_not_loaded, module}}

      not function_exported?(module, :transform_request, 4) ->
        {:error, {:request_transformer_missing_callback, module}}

      true ->
        {:ok, module}
    end
  end

  def validate(_other), do: {:error, :invalid_request_transformer}

  @doc """
  Fingerprint a validated transformer for checkpoint compatibility.
  """
  @spec fingerprint(module() | nil) :: String.t()
  def fingerprint(nil), do: ""
  def fingerprint(module) when is_atom(module), do: Atom.to_string(module)

  @callback transform_request(request(), State.t(), Config.t(), map()) ::
              {:ok, overrides()} | {:error, term()}

  @doc """
  Rewrite one tool round's results. Gets `{tool_name, result}` for each call
  that finished, in call order, and the tool context the calls ran with. Returns
  the results, in the same order and of the same length.

  A raise, or a list of another length, is logged and the round keeps its
  original results: the tools have already run, so their results must still
  reach the model and the event stream.
  """
  @callback transform_tool_results([{String.t(), term()}], map()) :: [term()]

  @optional_callbacks transform_tool_results: 2
end

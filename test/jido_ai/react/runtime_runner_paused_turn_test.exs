defmodule Jido.AI.Reasoning.ReAct.RuntimeRunnerPausedTurnTest do
  use ExUnit.Case, async: false
  use Mimic

  alias Jido.AI.Context, as: AIContext
  alias Jido.AI.Reasoning.ReAct
  alias Jido.AI.Reasoning.ReAct.{Config, State}
  alias ReqLLM.Message.ContentPart
  alias ReqLLM.StreamChunk

  @call_key {__MODULE__, :llm_calls}

  # Stands in for a req_llm that decodes Anthropic server-tool blocks. The
  # version this repo builds against drops provider blocks on the floor, so a
  # paused turn rebuilt through the real builder would carry no blocks to trim.
  defmodule ForkResponseBuilder do
    @moduledoc false

    def build_response(chunks, metadata, opts) do
      model = Keyword.fetch!(opts, :model)

      {:ok,
       %ReqLLM.Response{
         id: "resp_paused",
         model: model.id,
         context: Keyword.fetch!(opts, :context),
         message: %ReqLLM.Message{role: :assistant, content: content_parts(chunks), metadata: %{}},
         stream?: false,
         finish_reason: Map.get(metadata, :finish_reason),
         provider_meta: Map.get(metadata, :provider_meta, %{})
       }}
    end

    defp content_parts(chunks) do
      Enum.flat_map(chunks, fn
        %StreamChunk{type: :meta, metadata: %{provider_block: block}} ->
          [%ContentPart{type: :provider_block, data: block, metadata: %{provider: :anthropic}}]

        %StreamChunk{type: :content, text: text} when is_binary(text) ->
          [ContentPart.text(text)]

        _chunk ->
          []
      end)
    end
  end

  setup :set_mimic_from_context

  setup do
    :persistent_term.put(@call_key, 0)
    on_exit(fn -> :persistent_term.erase(@call_key) end)
    :ok
  end

  test "a pause_turn continuation replays the assistant content and ends with a user message" do
    parent = self()

    Mimic.stub(ReqLLM.Generation, :stream_text, fn model, messages, opts ->
      call = :persistent_term.get(@call_key) + 1
      :persistent_term.put(@call_key, call)
      send(parent, {:messages, call, messages})
      send(parent, {:opts, call, opts})

      case call do
        1 -> {:ok, paused_response(model)}
        _ -> {:ok, final_response(model)}
      end
    end)

    events = run("build the deck")

    completed = Enum.find(events, &(&1.kind == :request_completed))
    assert completed.data.result == "Finished answer"
    assert :persistent_term.get(@call_key) == 2

    assert_receive {:messages, 2, second}

    assert [%{role: :assistant, content: replayed}, %{role: :user, content: "continue"}] =
             Enum.take(second, -2)

    assert [%ContentPart{type: :text, text: "Working on it"}] = replayed

    assert_receive {:opts, 2, opts}
    assert Keyword.get(opts, :anthropic_container) == "cntr_pause"
  end

  test "a pause_turn continuation drops a server-tool use whose result never arrived" do
    Mimic.stub(ReqLLM.Provider.ResponseBuilder, :for_model, fn _model -> ForkResponseBuilder end)

    stub_stream_text([
      StreamChunk.text("Working on it"),
      tool_use_chunk("srvtoolu_paired", "bash_code_execution"),
      tool_result_chunk("srvtoolu_paired", "bash_code_execution_tool_result"),
      tool_use_chunk("srvtoolu_in_flight", "bash_code_execution")
    ])

    run("build the deck")

    assert_receive {:messages, 2, second}

    assert [%{role: :assistant, content: replayed}, %{role: :user, content: "continue"}] =
             Enum.take(second, -2)

    assert [
             %ContentPart{type: :text, text: "Working on it"},
             %ContentPart{type: :provider_block, data: %{"id" => "srvtoolu_paired"}},
             %ContentPart{type: :provider_block, data: %{"tool_use_id" => "srvtoolu_paired"}}
           ] = replayed
  end

  test "a pause_turn whose only content is an unpaired use repeats the request unchanged" do
    Mimic.stub(ReqLLM.Provider.ResponseBuilder, :for_model, fn _model -> ForkResponseBuilder end)

    stub_stream_text([tool_use_chunk("srvtoolu_in_flight", "bash_code_execution")])

    run("build the deck")

    assert_receive {:messages, 1, first}
    assert_receive {:messages, 2, second}
    assert second == first

    assert_receive {:opts, 2, opts}
    assert Keyword.get(opts, :anthropic_container) == "cntr_pause"
  end

  # Anthropic has stopped with "tool_use" on a turn that used only server
  # tools, so there was no client call for the runner to answer. Running an
  # empty tool round left the assistant message last, and the API rejects a
  # request that ends with one.
  test "a tool_use stop with no client tool call is resumed like a pause" do
    Mimic.stub(ReqLLM.Provider.ResponseBuilder, :for_model, fn _model -> ForkResponseBuilder end)

    stub_stream_text(
      [
        tool_use_chunk("srvtoolu_search", "web_search"),
        tool_result_chunk("srvtoolu_search", "web_search_tool_result"),
        StreamChunk.text("Found it")
      ],
      %{finish_reason: :tool_calls, provider_meta: %{"stop_reason" => "tool_use"}}
    )

    log = ExUnit.CaptureLog.capture_log(fn -> send(self(), {:events, run("look it up")}) end)
    assert_received {:events, events}

    completed = Enum.find(events, &(&1.kind == :request_completed))
    assert completed.data.result == "Finished answer"
    refute Enum.any?(events, &(&1.kind == :tool_started))
    refute log =~ "context ended with an assistant message"

    assert_receive {:messages, 2, second}

    assert [%{role: :assistant, content: replayed}, %{role: :user, content: "continue"}] =
             Enum.take(second, -2)

    assert [
             %ContentPart{type: :provider_block, data: %{"id" => "srvtoolu_search"}},
             %ContentPart{type: :provider_block, data: %{"tool_use_id" => "srvtoolu_search"}},
             %ContentPart{type: :text, text: "Found it"}
           ] = replayed
  end

  # A guard for any path that leaves the model's own message last: the API
  # rejects that request ("does not support assistant message prefill").
  test "a run whose context ends with an assistant message asks with a continuation" do
    stub_stream_text([StreamChunk.text("unused")])

    state =
      State.new("build the deck", nil, request_id: "req_pause", run_id: "run_pause")
      |> then(&%{&1 | context: AIContext.append_assistant(&1.context, "Halfway there")})

    log =
      ExUnit.CaptureLog.capture_log(fn ->
        events =
          ReAct.stream("build the deck", Config.new(%{model: :capable, tools: %{}}), state: state)
          |> Enum.to_list()

        assert Enum.any?(events, &(&1.kind == :request_completed))
      end)

    assert log =~ "context ended with an assistant message"

    assert_receive {:messages, 1, first}

    assert [
             %{role: :user},
             %{role: :assistant},
             %{role: :user, content: "continue"}
           ] = first
  end

  defp stub_stream_text(paused_chunks, metadata \\ nil) do
    parent = self()

    Mimic.stub(ReqLLM.Generation, :stream_text, fn model, messages, opts ->
      call = :persistent_term.get(@call_key) + 1
      :persistent_term.put(@call_key, call)
      send(parent, {:messages, call, messages})
      send(parent, {:opts, call, opts})

      case {call, metadata} do
        {1, nil} -> {:ok, paused_response(model, paused_chunks)}
        {1, metadata} -> {:ok, stream_response(paused_chunks, model, metadata)}
        _ -> {:ok, final_response(model)}
      end
    end)
  end

  defp tool_use_chunk(id, name) do
    StreamChunk.meta(%{
      provider_block: %{"type" => "server_tool_use", "id" => id, "name" => name},
      provider: :anthropic
    })
  end

  defp tool_result_chunk(id, type) do
    StreamChunk.meta(%{
      provider_block: %{"type" => type, "tool_use_id" => id},
      provider: :anthropic
    })
  end

  defp run(query) do
    config = Config.new(%{model: :capable, tools: %{}})

    ReAct.stream(query, config, request_id: "req_pause", run_id: "run_pause") |> Enum.to_list()
  end

  defp paused_response(model, chunks \\ [StreamChunk.text("Working on it")]) do
    stream_response(chunks, model, %{
      finish_reason: :incomplete,
      provider_meta: %{
        "stop_reason" => "pause_turn",
        "container" => %{"id" => "cntr_pause", "expires_at" => "2026-01-01T00:00:00Z"}
      }
    })
  end

  defp final_response(model) do
    stream_response([StreamChunk.text("Finished answer")], model, %{finish_reason: :stop})
  end

  defp stream_response(chunks, model_spec, metadata) do
    {:ok, model} = ReqLLM.model(model_spec)
    {:ok, metadata_handle} = ReqLLM.StreamResponse.MetadataHandle.start_link(fn -> metadata end)

    %ReqLLM.StreamResponse{
      stream: chunks,
      metadata_handle: metadata_handle,
      cancel: fn -> :ok end,
      model: model,
      context: ReqLLM.Context.new([])
    }
  end
end

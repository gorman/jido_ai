defmodule Jido.AI.Reasoning.ReAct.RuntimeRunnerStreamResumeTest do
  use ExUnit.Case, async: false
  use Mimic

  alias Jido.AI.Reasoning.ReAct
  alias Jido.AI.Reasoning.ReAct.Config
  alias ReqLLM.Message.ContentPart
  alias ReqLLM.StreamChunk

  @call_key {__MODULE__, :llm_calls}
  @pid_key {__MODULE__, :test_pid}

  # Stands in for a req_llm that decodes Anthropic server-tool blocks. The
  # version this repo builds against drops provider blocks on the floor, so a
  # rebuild through the real builder would have nothing positional to replay.
  defmodule ForkResponseBuilder do
    @moduledoc false

    def build_response(chunks, _metadata, opts) do
      send(:persistent_term.get({Jido.AI.Reasoning.ReAct.RuntimeRunnerStreamResumeTest, :test_pid}), {:rebuilt, chunks})

      model = Keyword.fetch!(opts, :model)

      {:ok,
       %ReqLLM.Response{
         id: "resp_partial",
         model: model.id,
         context: Keyword.fetch!(opts, :context),
         message: %ReqLLM.Message{role: :assistant, content: provider_block_parts(chunks), metadata: %{}},
         stream?: false,
         provider_meta: %{"container" => %{"id" => "cntr_resumed"}}
       }}
    end

    defp provider_block_parts(chunks) do
      for %StreamChunk{type: :meta, metadata: %{provider_block: block}} <- chunks,
          do: %ContentPart{type: :provider_block, data: block, metadata: %{provider: :anthropic}}
    end
  end

  setup :set_mimic_from_context

  setup do
    :persistent_term.put(@call_key, 0)
    :persistent_term.put(@pid_key, self())

    on_exit(fn ->
      :persistent_term.erase(@call_key)
      :persistent_term.erase(@pid_key)
    end)

    :ok
  end

  test "resumes a closed stream and replays the content up to the last tool result" do
    stub_stream_text(fn model, _opts ->
      dying_response(
        [
          StreamChunk.text("kept "),
          tool_use_chunk("srvtoolu_1"),
          tool_result_chunk("srvtoolu_1"),
          StreamChunk.text("dropped"),
          StreamChunk.thinking("unsigned")
        ],
        model
      )
    end)

    events = run("build the deck")

    assert completed_result(events) == "Second stream answer"
    assert llm_calls() == 2

    assert_receive {:messages, 1, _first}
    assert_receive {:messages, 2, second}

    assistant = Enum.filter(second, &match?(%{role: :assistant}, &1))
    assert [%{content: [%ContentPart{type: :text, text: "kept "}]}] = assistant
    assert %{role: :user, content: "continue"} = List.last(second)
  end

  test "hands the builder the chunks truncated at the last tool result, in arrival order" do
    Mimic.stub(ReqLLM.Provider.ResponseBuilder, :for_model, fn _model -> ForkResponseBuilder end)

    chunks = [
      StreamChunk.thinking("planning"),
      tool_use_chunk("srvtoolu_1"),
      tool_result_chunk("srvtoolu_1"),
      tool_use_chunk("srvtoolu_2"),
      tool_result_chunk("srvtoolu_2"),
      StreamChunk.text("half a sentence")
    ]

    stub_stream_text(fn model, _opts -> dying_response(chunks, model) end)

    events = run("build the deck")

    assert completed_result(events) == "Second stream answer"
    assert_receive {:rebuilt, rebuilt}
    assert rebuilt == Enum.take(chunks, 5)

    assert_receive {:messages, 2, second}
    assistant = Enum.find(second, &match?(%{role: :assistant}, &1))

    assert Enum.map(assistant.content, & &1.data["type"]) == [
             "server_tool_use",
             "code_execution_tool_result",
             "server_tool_use",
             "code_execution_tool_result"
           ]
  end

  test "threads the container id from the partial turn into the resumed request" do
    Mimic.stub(ReqLLM.Provider.ResponseBuilder, :for_model, fn _model -> ForkResponseBuilder end)

    stub_stream_text(fn model, _opts ->
      dying_response([tool_use_chunk("srvtoolu_1"), tool_result_chunk("srvtoolu_1")], model)
    end)

    assert completed_result(run("build the deck")) == "Second stream answer"

    assert_receive {:opts, 2, opts}
    assert Keyword.get(opts, :anthropic_container) == "cntr_resumed"
  end

  test "logs the resume with its ordinal, blocks kept and container" do
    Mimic.stub(ReqLLM.Provider.ResponseBuilder, :for_model, fn _model -> ForkResponseBuilder end)

    stub_stream_text(fn model, _opts ->
      dying_response([tool_use_chunk("srvtoolu_1"), tool_result_chunk("srvtoolu_1")], model)
    end)

    log = ExUnit.CaptureLog.capture_log(fn -> run("build the deck") end)

    assert log =~ "react stream_resume: resume=1"
    assert log =~ "blocks_kept=1"
    assert log =~ "container=true"
  end

  test "re-requests the turn unchanged when no tool result completed" do
    stub_stream_text(fn model, _opts ->
      dying_response([StreamChunk.text("half a thought")], model)
    end)

    assert completed_result(run("build the deck")) == "Second stream answer"

    assert_receive {:messages, 1, first}
    assert_receive {:messages, 2, second}
    assert second == first
  end

  test "fails the run on a stream error that is not a transport close" do
    stub_stream_text(fn model, _opts ->
      dying_response([StreamChunk.text("partial")], model, timeout_error())
    end)

    events = run("build the deck")

    assert llm_calls() == 1
    failed = Enum.find(events, &(&1.kind == :request_failed))
    assert failed.data.error_type == :llm_stream
  end

  test "stops resuming once the cap is spent and fails with the transport error" do
    stub_stream_text(fn model, _opts ->
      dying_response([tool_use_chunk("srvtoolu_1"), tool_result_chunk("srvtoolu_1")], model, closed_error(), :always)
    end)

    events = run("build the deck")

    assert llm_calls() == 3
    failed = Enum.find(events, &(&1.kind == :request_failed))
    assert failed.data.error_type == :llm_stream
    assert %ReqLLM.Error.API.Stream{} = failed.data.error
  end

  test "honours a lower resume cap from the config" do
    stub_stream_text(fn model, _opts ->
      dying_response([tool_use_chunk("srvtoolu_1"), tool_result_chunk("srvtoolu_1")], model, closed_error(), :always)
    end)

    events = run("build the deck", max_stream_resumes: 0)

    assert llm_calls() == 1
    assert Enum.any?(events, &(&1.kind == :request_failed))
  end

  defp run(query, config_opts \\ []) do
    config = Config.new(Map.new([{:model, :capable}, {:tools, %{}} | config_opts]))

    ReAct.stream(query, config, request_id: "req_resume", run_id: "run_resume") |> Enum.to_list()
  end

  defp completed_result(events) do
    case Enum.find(events, &(&1.kind == :request_completed)) do
      nil -> nil
      event -> event.data.result
    end
  end

  defp stub_stream_text(dying_fun) do
    parent = self()

    Mimic.stub(ReqLLM.Generation, :stream_text, fn model, messages, opts ->
      call = llm_calls() + 1
      :persistent_term.put(@call_key, call)
      send(parent, {:messages, call, messages})
      send(parent, {:opts, call, opts})

      dying_fun.(model, opts)
    end)
  end

  # The first stream dies; later ones answer normally unless the test asks for
  # every stream to die.
  defp dying_response(chunks, model, error \\ nil, mode \\ :once) do
    error = error || closed_error()

    if mode == :always or llm_calls() == 1 do
      {:ok, stream_response(dying_stream(chunks, error), model)}
    else
      {:ok, final_response(model)}
    end
  end

  defp final_response(model) do
    stream_response([StreamChunk.text("Second stream answer")], model, %{finish_reason: :stop})
  end

  defp dying_stream(chunks, error) do
    Stream.map(chunks ++ [:die], fn
      :die -> raise error
      chunk -> chunk
    end)
  end

  defp closed_error do
    %ReqLLM.Error.API.Stream{
      reason: "Stream failed: closed",
      cause: %Finch.TransportError{reason: :closed, source: %Mint.TransportError{reason: :closed}}
    }
  end

  defp timeout_error do
    %ReqLLM.Error.API.Stream{
      reason: "Stream failed: timeout",
      cause: %Finch.TransportError{reason: :timeout}
    }
  end

  defp tool_use_chunk(id) do
    StreamChunk.meta(%{
      provider_block: %{"type" => "server_tool_use", "id" => id, "name" => "code_execution"},
      provider: :anthropic
    })
  end

  defp tool_result_chunk(id) do
    StreamChunk.meta(%{
      provider_block: %{"type" => "code_execution_tool_result", "tool_use_id" => id},
      provider: :anthropic
    })
  end

  defp stream_response(chunks, model_spec, metadata \\ %{}) do
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

  defp llm_calls, do: :persistent_term.get(@call_key, 0)
end

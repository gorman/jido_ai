defmodule Jido.AI.Reasoning.ReAct.RuntimeRunnerPausedTurnTest do
  use ExUnit.Case, async: false
  use Mimic

  alias Jido.AI.Reasoning.ReAct
  alias Jido.AI.Reasoning.ReAct.Config
  alias ReqLLM.Message.ContentPart
  alias ReqLLM.StreamChunk

  @call_key {__MODULE__, :llm_calls}

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

  defp run(query) do
    config = Config.new(%{model: :capable, tools: %{}})

    ReAct.stream(query, config, request_id: "req_pause", run_id: "run_pause") |> Enum.to_list()
  end

  defp paused_response(model) do
    stream_response([StreamChunk.text("Working on it")], model, %{
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

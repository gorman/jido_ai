defmodule Jido.AI.Reasoning.ReAct.WorkerStrategyTest do
  use ExUnit.Case, async: false
  use Mimic

  alias Jido.Agent.Strategy.State, as: StratState
  alias Jido.AI.Reasoning.ReAct.Config
  alias Jido.AI.Reasoning.ReAct.Worker.Strategy
  alias ReqLLM.Message.ContentPart

  setup :set_mimic_from_context

  test "starts delegated runtime for multimodal query lists" do
    parent = self()

    Mimic.stub(ReqLLM.Generation, :stream_text, fn model, messages, _opts ->
      send(parent, {:req_llm_messages, messages})
      {:ok, resolved_model} = ReqLLM.model(model)
      {:ok, metadata_handle} = ReqLLM.StreamResponse.MetadataHandle.start_link(fn -> %{finish_reason: :stop} end)

      {:ok,
       %ReqLLM.StreamResponse{
         stream: [ReqLLM.StreamChunk.text("Done")],
         metadata_handle: metadata_handle,
         model: resolved_model,
         context: ReqLLM.Context.new([]),
         cancel: fn -> :ok end
       }}
    end)

    query = [
      ContentPart.text("Summarize this document."),
      ContentPart.file_id("file_123")
    ]

    agent =
      %Jido.Agent{id: "worker-test", name: "worker", state: %{}}
      |> then(fn agent ->
        {agent, []} = Strategy.init(agent, %{})
        agent
      end)

    start =
      %Jido.Instruction{
        action: :react_worker_start,
        params: %{
          request_id: "req_file",
          run_id: "req_file",
          query: query,
          config: Config.new(%{model: :capable, tools: %{}})
        }
      }

    {agent, []} = Strategy.cmd(agent, [start], %{})

    state = StratState.get(agent, %{})
    assert state.status == :running
    assert state.active_request_id == "req_file"

    assert_receive {:req_llm_messages, [%{role: :user, content: ^query}]}, 200
  end

  describe "a runtime that ends without a terminal event" do
    # The runner's stream halts silently when its own wait for the next event
    # runs out. If the worker then said nothing, the parent would stay
    # `awaiting_llm` with the request active, and refuse every later ask as busy.
    setup do
      parent = %Jido.AgentServer.ParentRef{pid: self(), id: "parent", tag: :react_worker}

      agent =
        %Jido.Agent{id: "worker-done-test", name: "worker", state: %{__parent__: parent}}
        |> then(fn agent ->
          {agent, []} = Strategy.init(agent, %{})
          agent
        end)

      running =
        agent
        |> StratState.get(%{})
        |> Map.merge(%{status: :running, active_request_id: "req_quiet", run_id: "req_quiet", seq: 4})

      {:ok, agent: %{agent | state: Map.put(agent.state, StratState.key(), running)}}
    end

    test "is reported to the parent as a failed request", %{agent: agent} do
      done = %Jido.Instruction{action: :react_worker_runtime_done, params: %{request_id: "req_quiet"}}

      {agent, directives} = Strategy.cmd(agent, [done], %{})

      assert [%Jido.AI.Reasoning.ReAct.Worker.EmitEvent{signal: signal}] = directives
      assert %{request_id: "req_quiet", event: %{kind: :request_failed, data: data}} = signal.data
      assert data.error_type == :runtime_ended

      state = StratState.get(agent, %{})
      assert state.status == :idle
      assert state.active_request_id == nil
    end

    test "is not reported again after the run already ended", %{agent: agent} do
      completed = %Jido.Instruction{
        action: :react_worker_runtime_event,
        params: %{
          request_id: "req_quiet",
          event: %{seq: 5, run_id: "req_quiet", request_id: "req_quiet", kind: :request_completed, data: %{}}
        }
      }

      done = %Jido.Instruction{action: :react_worker_runtime_done, params: %{request_id: "req_quiet"}}

      {agent, [_completed_event]} = Strategy.cmd(agent, [completed], %{})
      {_agent, directives} = Strategy.cmd(agent, [done], %{})

      assert directives == []
    end
  end
end

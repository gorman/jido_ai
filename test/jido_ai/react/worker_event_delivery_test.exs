defmodule Jido.AI.Reasoning.ReAct.WorkerEventDeliveryTest do
  use ExUnit.Case, async: false
  use Mimic

  alias Jido.AgentServer.{DirectiveExec, ParentRef}
  alias Jido.AI.Reasoning.ReAct.{Event, Strategy, Worker}
  alias Jido.AI.Request
  alias Jido.AI.Request.Handle
  alias Jido.Tracing.{Context, Trace}

  defmodule ParentAgent do
    use Jido.AI.Agent, name: "ordered_worker_events", tools: []
  end

  setup :set_mimic_from_context

  setup do
    Mimic.copy(Task.Supervisor)

    # Hold any dispatch tasks and run them in reverse order. Event delivery
    # must not depend on the order in which the task supervisor runs them.
    Mimic.stub(Task.Supervisor, :start_child, fn _supervisor, fun ->
      Process.put(:dispatch_tasks, [fun | Process.get(:dispatch_tasks, [])])
      {:ok, self()}
    end)

    :ok
  end

  test "streams worker events in runtime order under reversed task scheduling" do
    request_id = "ordered_request"
    parent_ref = ParentRef.new!(%{pid: self(), id: "parent", tag: :react_worker})
    worker = Worker.Agent.new(state: %{__parent__: parent_ref})
    parent = Request.start_request(ParentAgent.new(), request_id, "test", stream_to: {:pid, self()})

    events = [
      event(1, :request_started, %{query: "test"}),
      event(2, :llm_delta, %{chunk_type: :content, delta: "The Spring "}),
      event(3, :llm_delta, %{chunk_type: :content, delta: "Campaign plan "}),
      event(4, :llm_delta, %{chunk_type: :content, delta: "has one swimlane."}),
      event(5, :llm_completed, %{text: "The Spring Campaign plan has one swimlane.", model: "test:model"}),
      event(6, :request_completed, %{result: "The Spring Campaign plan has one swimlane."})
    ]

    Enum.reduce(events, worker, fn event, worker ->
      params = %{request_id: request_id, event: Map.from_struct(event)}
      input = Jido.Signal.new!("ai.react.worker.runtime.event", params, source: "/test")
      instruction = %Jido.Instruction{action: :react_worker_runtime_event, params: params}
      {worker, directives} = Worker.Strategy.cmd(worker, [instruction], %{})

      state =
        struct(Jido.AgentServer.State,
          id: worker.id,
          agent: worker,
          agent_module: Worker.Agent,
          jido: nil
        )

      Enum.each(directives, &DirectiveExec.exec(&1, input, state))
      worker
    end)

    Enum.each(Process.get(:dispatch_tasks, []), & &1.())

    Enum.reduce(events, parent, fn _, parent ->
      signal = receive_worker_event()
      instruction = %Jido.Instruction{action: :ai_react_worker_event, params: signal.data}
      {parent, directives} = Strategy.cmd(parent, [instruction], %{})

      {:ok, parent, _} =
        ParentAgent.on_after_cmd(parent, {:ai_react_worker_event, signal.data}, directives)

      parent
    end)

    streamed =
      Handle.new(request_id, self(), "test")
      |> Request.Stream.events(stream_event_timeout_ms: 0)
      |> Enum.to_list()

    assert Enum.map(streamed, & &1.seq) == [1, 2, 3, 4, 5, 6]

    assert streamed |> Enum.filter(&(&1.kind == :llm_delta)) |> Enum.map_join(& &1.data.delta) ==
             "The Spring Campaign plan has one swimlane."
  end

  test "a worker without a parent does not emit an event" do
    worker = Worker.Agent.new()
    event = event(1, :request_started, %{query: "test"})

    instruction = %Jido.Instruction{
      action: :react_worker_runtime_event,
      params: %{request_id: event.request_id, event: Map.from_struct(event)}
    }

    assert {_worker, []} = Worker.Strategy.cmd(worker, [instruction], %{})
  end

  test "ordered delivery preserves trace and causation data" do
    input = Jido.Signal.new!("ai.react.worker.runtime.event", %{}, source: "/test")
    {input, trace} = Context.ensure_from_signal(input)
    signal = Jido.Signal.new!("ai.react.worker.event", %{}, source: "/test")
    directive = %Worker.EmitEvent{parent: self(), signal: signal}

    try do
      assert {:ok, %{}} = DirectiveExec.exec(directive, input, %{})
      delivered = receive_worker_event()
      delivered_trace = Trace.get(delivered)
      assert delivered_trace.trace_id == trace.trace_id
      assert delivered_trace.parent_span_id == trace.span_id
      assert delivered_trace.causation_id == input.id
      refute delivered_trace.span_id == trace.span_id
    after
      Context.clear()
    end
  end

  defp receive_worker_event do
    receive do
      {:"$gen_cast", {:signal, %Jido.Signal{type: "ai.react.worker.event"} = signal}} -> signal
      {:signal, %Jido.Signal{type: "ai.react.worker.event"} = signal} -> signal
    after
      1_000 -> flunk("worker event was not delivered")
    end
  end

  defp event(seq, kind, data) do
    Event.new(%{
      seq: seq,
      kind: kind,
      data: data,
      request_id: "ordered_request",
      run_id: "ordered_request",
      iteration: 1
    })
  end
end

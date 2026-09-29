defmodule Jido.AI.Reasoning.ReAct.RuntimeRunnerInFlightTest do
  use ExUnit.Case, async: false
  use Mimic

  alias Jido.AI.Reasoning.ReAct
  alias Jido.AI.Reasoning.ReAct.Config
  alias ReqLLM.StreamChunk

  setup :set_mimic_from_context

  setup do
    on_exit(fn -> :persistent_term.erase({__MODULE__, :stalled}) end)
    :ok
  end

  # What Anthropic's message_start carries, as req_llm's decoder emits it: the
  # prefill is billed from here on, whether or not the stream ever finishes.
  @prefill %{
    input_tokens: 1_200,
    output_tokens: 1,
    total_tokens: 1_201,
    cached_tokens: 50_000,
    cache_read_input_tokens: 50_000,
    cache_creation_input_tokens: 3_000,
    reasoning_tokens: 0
  }

  test "a stream that goes quiet ends in request_failed that names the open call and its prefill" do
    parent = self()

    stub_stream_text(fn model ->
      stream_response([StreamChunk.meta(%{usage: @prefill}), :stall], model, parent)
    end)

    events = run(stream_timeout_ms: 80)

    started = Enum.find(events, &(&1.kind == :llm_started))
    failed = List.last(events)

    assert failed.kind == :request_failed
    assert failed.data.error == :runner_receive_timeout
    assert failed.data.error_type == :timeout
    assert failed.request_id == "req_in_flight"
    assert failed.seq > started.seq
    assert failed.data.last_event_kind == :stream_activity
    assert is_integer(failed.data.ms_since_last_event)

    assert %{llm_call_id: call_id, usage: usage} = failed.data.in_flight
    assert call_id == started.data.call_id
    assert usage.cached_tokens == 50_000
    assert usage.cache_creation_tokens == 3_000
  end

  test "a stream that dies after message_start reports the open call's prefill" do
    parent = self()

    stub_stream_text(fn model ->
      stream_response([StreamChunk.meta(%{usage: @prefill}), :die], model, parent)
    end)

    events = run(max_stream_resumes: 0)

    started = Enum.find(events, &(&1.kind == :llm_started))
    failed = Enum.find(events, &(&1.kind == :request_failed))

    assert failed.data.error_type == :llm_stream
    assert failed.data.in_flight.llm_call_id == started.data.call_id
    assert failed.data.in_flight.usage.cache_creation_tokens == 3_000
  end

  test "a runner that dies mid-call ends in request_failed, not a silent finish" do
    parent = self()

    stub_stream_text(fn model ->
      stream_response([StreamChunk.meta(%{usage: @prefill}), :kill], model, parent)
    end)

    failed = List.last(run())

    assert failed.kind == :request_failed
    assert failed.data.error == :runner_down
    assert failed.data.in_flight.usage.cached_tokens == 50_000
  end

  test "a cancel mid-call reports the open call's prefill" do
    parent = self()

    stub_stream_text(fn model ->
      stream_response([StreamChunk.meta(%{usage: @prefill}), :stall_until_cancel], model, parent)
    end)

    consumer =
      spawn(fn ->
        send(parent, {:events, run(max_stream_resumes: 0) |> Enum.to_list()})
      end)

    assert_receive {:stalled, _coordinator}, 1_000
    send(consumer, {:react_stream_cancel, :user_stop})

    assert_receive {:events, events}, 1_000

    started = Enum.find(events, &(&1.kind == :llm_started))
    cancelled = Enum.find(events, &(&1.kind == :request_cancelled))

    assert cancelled.data.reason == :user_stop
    assert cancelled.data.in_flight.llm_call_id == started.data.call_id
    assert cancelled.data.in_flight.usage.cached_tokens == 50_000
  end

  test "a call that completed is no longer in flight when a later request fails" do
    parent = self()
    :persistent_term.put({__MODULE__, :calls}, 0)
    on_exit(fn -> :persistent_term.erase({__MODULE__, :calls}) end)

    Mimic.stub(ReqLLM.Generation, :stream_text, fn model, _messages, _opts ->
      call = :persistent_term.get({__MODULE__, :calls}) + 1
      :persistent_term.put({__MODULE__, :calls}, call)

      case call do
        1 ->
          {:ok,
           stream_response(
             [StreamChunk.meta(%{usage: @prefill}), StreamChunk.tool_call("missing_tool", %{}, %{id: "tc_1"})],
             model,
             parent,
             %{finish_reason: :tool_calls}
           )}

        _ ->
          {:error, :boom}
      end
    end)

    events = run()

    assert Enum.any?(events, &(&1.kind == :llm_completed))
    failed = Enum.find(events, &(&1.kind == :request_failed))
    assert failed.data.error_type == :llm_request
    assert failed.data.in_flight == nil
  end

  defp run(config_opts \\ []) do
    config = Config.new(Map.new([{:model, :capable}, {:tools, %{}} | config_opts]))

    ReAct.stream("build the deck", config, request_id: "req_in_flight", run_id: "run_in_flight")
    |> Enum.to_list()
  end

  defp stub_stream_text(response_fun) do
    Mimic.stub(ReqLLM.Generation, :stream_text, fn model, _messages, _opts -> {:ok, response_fun.(model)} end)
  end

  # `:stall` blocks the stream until the stream's cancel wakes it; `:die` is a
  # transport close.
  defp stream_response(chunks, model_spec, parent, metadata \\ %{}) do
    {:ok, model} = ReqLLM.model(model_spec)
    {:ok, metadata_handle} = ReqLLM.StreamResponse.MetadataHandle.start_link(fn -> metadata end)

    stream =
      Stream.map(chunks, fn
        :stall ->
          :persistent_term.put({__MODULE__, :stalled}, self())
          send(parent, {:stalled, self()})

          receive do
            :wake -> raise closed_error()
          end

        # A real stream dies as soon as its cancel lands; waking on the runner's
        # own cancel message keeps the test off the cancel-vs-error race.
        :stall_until_cancel ->
          send(parent, {:stalled, self()})

          receive do
            {:react_cancel, _ref, _reason} = cancel ->
              send(self(), cancel)
              raise closed_error()
          end

        :die ->
          raise closed_error()

        :kill ->
          Process.exit(self(), :kill)

        chunk ->
          chunk
      end)

    %ReqLLM.StreamResponse{
      stream: stream,
      metadata_handle: metadata_handle,
      cancel: fn ->
        case :persistent_term.get({__MODULE__, :stalled}, nil) do
          pid when is_pid(pid) -> send(pid, :wake)
          nil -> :ok
        end

        :ok
      end,
      model: model,
      context: ReqLLM.Context.new([])
    }
  end

  defp closed_error do
    %ReqLLM.Error.API.Stream{
      reason: "Stream failed: closed",
      cause: %Finch.TransportError{reason: :closed, source: %Mint.TransportError{reason: :closed}}
    }
  end
end

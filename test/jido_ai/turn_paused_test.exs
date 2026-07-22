defmodule Jido.AI.TurnPausedTest do
  use ExUnit.Case, async: true

  alias Jido.AI.Context, as: AIContext
  alias Jido.AI.Turn
  alias ReqLLM.Message.ContentPart

  @moduletag :unit

  defp paused_response do
    parts = [
      %ContentPart{type: :provider_block, data: %{"type" => "server_tool_use", "id" => "srv_1"}, metadata: %{provider: :anthropic}},
      ContentPart.text("Searching…")
    ]

    %ReqLLM.Response{
      id: "msg_1",
      model: "claude-sonnet-4-6",
      context: %ReqLLM.Context{messages: []},
      message: %ReqLLM.Message{role: :assistant, content: parts, metadata: %{}},
      stream?: false,
      stream: nil,
      usage: %{input_tokens: 1, output_tokens: 1},
      finish_reason: :incomplete,
      provider_meta: %{"stop_reason" => "pause_turn"}
    }
  end

  describe "paused turns" do
    test "from_response carries the raw stop_reason and full content parts" do
      turn = Turn.from_response(paused_response())

      assert turn.stop_reason == "pause_turn"
      assert [%ContentPart{type: :provider_block}, %ContentPart{type: :text}] = turn.content_parts
      assert Turn.paused?(turn)
    end

    test "paused? is false for ordinary completions and other incomplete reasons" do
      refute Turn.paused?(%Turn{finish_reason: :stop, stop_reason: nil})
      refute Turn.paused?(%Turn{finish_reason: :incomplete, stop_reason: nil})
      refute Turn.paused?(%Turn{finish_reason: :stop, stop_reason: "pause_turn"})
    end

    test "from_response carries the code-execution container id" do
      response = paused_response()

      provider_meta =
        Map.put(response.provider_meta, "container", %{
          "id" => "container_abc",
          "expires_at" => "2026-01-01T00:00:00Z"
        })

      turn = Turn.from_response(%{response | provider_meta: provider_meta})

      assert turn.container_id == "container_abc"
    end

    test "container_id is nil when the response reports no container" do
      turn = Turn.from_response(paused_response())

      assert turn.container_id == nil
    end
  end

  describe "args_lost tool calls" do
    defp response_with_tool_call(tool_call) do
      %ReqLLM.Response{
        id: "msg_1",
        model: "claude-sonnet-4-6",
        context: %ReqLLM.Context{messages: []},
        message: %ReqLLM.Message{
          role: :assistant,
          content: [],
          tool_calls: [tool_call],
          metadata: %{}
        },
        stream?: false,
        stream: nil,
        usage: %{input_tokens: 1, output_tokens: 1},
        finish_reason: :tool_calls,
        provider_meta: %{}
      }
    end

    test "a transport-truncated tool call is flagged through normalization" do
      tool_call =
        ReqLLM.ToolCall.new("toolu_1", "propose_moments", "{}")
        |> ReqLLM.ToolCall.put_metadata(%{error: {:args_lost, :missing_fragments}})

      turn = Turn.from_response(response_with_tool_call(tool_call))

      assert [%{name: "propose_moments", args_lost: true}] = turn.tool_calls

      assert %Jido.AI.Reasoning.ReAct.PendingToolCall{args_lost: true} =
               Jido.AI.Reasoning.ReAct.PendingToolCall.from_tool_call(hd(turn.tool_calls))
    end

    test "an intact tool call carries no args_lost flag" do
      tool_call = ReqLLM.ToolCall.new("toolu_2", "list_moments", ~s({"limit":5}))

      turn = Turn.from_response(response_with_tool_call(tool_call))

      assert [normalized] = turn.tool_calls
      refute Map.has_key?(normalized, :args_lost)

      assert %Jido.AI.Reasoning.ReAct.PendingToolCall{args_lost: false} =
               Jido.AI.Reasoning.ReAct.PendingToolCall.from_tool_call(normalized)
    end
  end

  describe "context round-trip of paused assistant content" do
    test "an assistant entry with content_parts projects them verbatim" do
      parts = [
        %ContentPart{type: :provider_block, data: %{"type" => "server_tool_use", "id" => "srv_1"}, metadata: %{provider: :anthropic}},
        %ContentPart{type: :provider_block, data: %{"type" => "web_search_tool_result"}, metadata: %{provider: :anthropic}},
        ContentPart.text("Searching…")
      ]

      messages =
        AIContext.new()
        |> AIContext.append_user("find something")
        |> AIContext.append_assistant("Searching…", nil, content_parts: parts)
        |> AIContext.to_messages()

      assert [%{role: :user}, %{role: :assistant, content: ^parts}] = messages
    end

    test "assistant entries without content_parts keep the plain-text projection" do
      messages =
        AIContext.new()
        |> AIContext.append_assistant("plain answer")
        |> AIContext.to_messages()

      assert [%{role: :assistant, content: "plain answer"}] = messages
    end
  end
end

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

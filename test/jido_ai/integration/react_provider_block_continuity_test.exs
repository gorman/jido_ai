defmodule Jido.AI.Integration.ReactProviderBlockContinuityTest do
  use ExUnit.Case, async: false
  use Mimic

  alias ReqLLM.Message.ContentPart

  defmodule BlockAgent do
    use Jido.AI.Agent,
      name: "provider_block_continuity_agent",
      model: "test:model",
      system_prompt: "Prompt",
      tools: [],
      streaming: false
  end

  @provider_blocks [
    %ContentPart{
      type: :provider_block,
      data: %{"type" => "server_tool_use", "id" => "srvtoolu_1", "name" => "web_search"},
      metadata: %{provider: :anthropic}
    },
    %ContentPart{
      type: :provider_block,
      data: %{
        "type" => "web_search_tool_result",
        "tool_use_id" => "srvtoolu_1",
        "content" => [%{"type" => "web_search_result", "encrypted_content" => "ENCRYPTED", "url" => "https://example.com"}]
      },
      metadata: %{provider: :anthropic}
    }
  ]

  setup :set_mimic_from_context

  setup do
    if is_nil(Process.whereis(Jido)) do
      start_supervised!({Jido, name: Jido})
    end

    :ok
  end

  defp stub_llm(test_pid, first_content) do
    Mimic.stub(ReqLLM.Generation, :generate_text, fn _model, messages, _opts ->
      send(test_pid, {:llm_messages, messages})

      content =
        case Enum.count(messages, &match?(%{role: :user}, &1)) do
          1 -> first_content
          _ -> "Done."
        end

      {:ok, %{message: %{content: content, tool_calls: nil, metadata: %{}}, finish_reason: :stop, usage: %{}}}
    end)
  end

  test "provider blocks from a completed turn are replayed verbatim on the next request" do
    stub_llm(self(), @provider_blocks ++ [ContentPart.text("Found it.")])

    {:ok, pid} = Jido.AgentServer.start_link(agent: BlockAgent)
    on_exit(fn -> if Process.alive?(pid), do: Process.exit(pid, :kill) end)

    assert {:ok, "Found it."} = BlockAgent.ask_sync(pid, "search please", timeout: 5_000)
    assert_receive {:llm_messages, _first_request}, 1_000

    assert {:ok, "Done."} = BlockAgent.ask_sync(pid, "and now?", timeout: 5_000)
    assert_receive {:llm_messages, second_request}, 1_000

    assistant = Enum.find(second_request, &match?(%{role: :assistant}, &1))

    assert %{content: [_ | _] = parts} = assistant
    assert Enum.filter(parts, &match?(%ContentPart{type: :provider_block}, &1)) == @provider_blocks
    assert Enum.any?(parts, &match?(%ContentPart{type: :text, text: "Found it."}, &1))
  end

  test "turns without provider blocks keep the plain text projection" do
    stub_llm(self(), "Just text.")

    {:ok, pid} = Jido.AgentServer.start_link(agent: BlockAgent)
    on_exit(fn -> if Process.alive?(pid), do: Process.exit(pid, :kill) end)

    assert {:ok, "Just text."} = BlockAgent.ask_sync(pid, "hello", timeout: 5_000)
    assert_receive {:llm_messages, _first_request}, 1_000

    assert {:ok, "Done."} = BlockAgent.ask_sync(pid, "again", timeout: 5_000)
    assert_receive {:llm_messages, second_request}, 1_000

    assistant = Enum.find(second_request, &match?(%{role: :assistant}, &1))
    assert %{content: "Just text."} = assistant
  end
end

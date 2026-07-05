defmodule Jido.AI.ReactPauseLoopTest do
  use Jido.AI.TestCase, async: false

  alias Jido.AI.Reasoning.ReAct

  setup do
    if is_nil(Process.whereis(Jido)) do
      start_supervised!({Jido, name: Jido})
    end

    :ok
  end

  test "resumes a paused provider turn with no new user or tool message" do
    expect_react do
      user("search for opal")
      pause()
      answer("Opal is a gem.")
    end

    result =
      ReAct.run("search for opal", %{
        model: :fast,
        tools: [],
        token_secret: "test-secret-that-is-long-enough-123"
      })

    assert_final_answer(result, "Opal is a gem.")
    assert_no_runtime_failure(result)

    # Two LLM calls for one user message: the paused turn and its resume.
    llm_starts = Enum.filter(result.trace, &(&1.kind == :llm_started))
    assert length(llm_starts) == 2

    # The resume added exactly one message — the paused assistant content —
    # and no user or tool message.
    assert Enum.map(llm_starts, & &1.data.message_count) == [1, 2]
  end

  test "a paused turn still counts against max_iterations" do
    expect_react do
      user("loop forever")
      pause()
      pause()
      answer("never reached")
    end

    result =
      ReAct.run("loop forever", %{
        model: :fast,
        tools: [],
        max_iterations: 1,
        token_secret: "test-secret-that-is-long-enough-123"
      })

    # Cut off at the cap rather than looping past it.
    refute Enum.any?(result.trace, &(&1.kind == :llm_started and &1.data[:message_count] == 3))
  end
end

defmodule Jido.AI.Reasoning.ReAct.StreamResumeTest do
  use ExUnit.Case, async: true

  alias Jido.AI.Reasoning.ReAct.Config
  alias Jido.AI.Reasoning.ReAct.StreamResume
  alias ReqLLM.StreamChunk

  defp block(type, id) do
    StreamChunk.meta(%{
      provider_block: %{"type" => type, "tool_use_id" => id},
      provider: :anthropic
    })
  end

  defp tool_use(id) do
    StreamChunk.meta(%{
      provider_block: %{"type" => "server_tool_use", "id" => id, "name" => "code_execution"},
      provider: :anthropic
    })
  end

  defp stream_response(model_spec \\ "anthropic:claude-sonnet-4-5") do
    {:ok, model} = ReqLLM.model(model_spec)
    {:ok, handle} = ReqLLM.StreamResponse.MetadataHandle.start_link(fn -> %{} end)

    %ReqLLM.StreamResponse{
      stream: [],
      metadata_handle: handle,
      cancel: fn -> :ok end,
      model: model,
      context: ReqLLM.Context.new([])
    }
  end

  describe "the resume budget" do
    test "allows two resumes per run by default" do
      assert Config.new(%{model: :capable}).max_stream_resumes == 2
    end

    test "takes an override, and treats an unset override as the default" do
      assert Config.new(%{model: :capable, max_stream_resumes: 5}).max_stream_resumes == 5
      assert Config.new(%{model: :capable, max_stream_resumes: 0}).max_stream_resumes == 0
      assert Config.new(%{model: :capable, max_stream_resumes: nil}).max_stream_resumes == 2
    end
  end

  describe "truncate_at_last_tool_result/1" do
    test "keeps everything through the last code-execution result" do
      chunks = [
        StreamChunk.thinking("planning"),
        tool_use("srvtoolu_1"),
        block("code_execution_tool_result", "srvtoolu_1"),
        StreamChunk.text("halfway"),
        tool_use("srvtoolu_2"),
        block("code_execution_tool_result", "srvtoolu_2"),
        StreamChunk.text("cut off"),
        StreamChunk.thinking("unsigned")
      ]

      assert {:ok, kept, 2} = StreamResume.truncate_at_last_tool_result(chunks)
      assert kept == Enum.take(chunks, 6)
      assert List.last(kept).metadata.provider_block["tool_use_id"] == "srvtoolu_2"
    end

    test "reports no boundary when no result block arrived" do
      chunks = [StreamChunk.thinking("planning"), tool_use("srvtoolu_1"), StreamChunk.text("partial")]

      assert StreamResume.truncate_at_last_tool_result(chunks) == :no_boundary
    end

    test "reports no boundary for an empty chunk list" do
      assert StreamResume.truncate_at_last_tool_result([]) == :no_boundary
    end

    test "accepts every code-execution result type" do
      for type <- ~w(code_execution_tool_result bash_code_execution_tool_result text_editor_code_execution_tool_result) do
        assert {:ok, _kept, 1} = StreamResume.truncate_at_last_tool_result([block(type, "id"), StreamChunk.text("x")])
      end
    end

    test "does not cut at other server-tool results" do
      chunks = [tool_use("srvtoolu_1"), block("web_search_tool_result", "srvtoolu_1")]

      assert StreamResume.truncate_at_last_tool_result(chunks) == :no_boundary
    end
  end

  describe "resumable_error?/1" do
    test "matches a Finch transport close inside a stream error" do
      error = %ReqLLM.Error.API.Stream{
        reason: "Stream failed",
        cause: %Finch.TransportError{reason: :closed}
      }

      assert StreamResume.resumable_error?(error)
    end

    test "matches a Finch transport close wrapping a Mint one" do
      error = %Finch.TransportError{reason: :closed}
      assert StreamResume.resumable_error?({:http_task_failed, error})
    end

    test "matches a bare Mint transport close" do
      assert StreamResume.resumable_error?(%Mint.TransportError{reason: :closed})
    end

    test "matches a close nested in a list" do
      assert StreamResume.resumable_error?({:error, [%Mint.TransportError{reason: :closed}]})
    end

    test "rejects other transport failures" do
      refute StreamResume.resumable_error?(%Finch.TransportError{reason: :timeout})
      refute StreamResume.resumable_error?(%Mint.TransportError{reason: :econnrefused})
    end

    test "rejects unrelated errors" do
      refute StreamResume.resumable_error?(:timeout)
      refute StreamResume.resumable_error?(%RuntimeError{message: "boom"})

      refute StreamResume.resumable_error?(%ReqLLM.Error.API.Stream{
               reason: "Stream failed",
               cause: {:api_error, 429}
             })
    end
  end

  describe "partial_turn/3" do
    # The provider blocks themselves only survive the rebuild on a req_llm that
    # decodes them, so the block-level assertions live in the runner test where
    # the builder is stubbed. Here the text on either side of the boundary
    # stands in for "kept" and "dropped".
    test "builds a turn from the content before the boundary only" do
      chunks = [
        StreamChunk.text("kept"),
        tool_use("srvtoolu_1"),
        block("code_execution_tool_result", "srvtoolu_1"),
        StreamChunk.text(" dropped")
      ]

      assert {:ok, turn, 1} = StreamResume.partial_turn(chunks, stream_response(), "anthropic:test")
      assert turn.text == "kept"
    end

    test "returns :none without a boundary" do
      assert StreamResume.partial_turn([StreamChunk.text("partial")], stream_response(), "m") == :none
    end
  end
end

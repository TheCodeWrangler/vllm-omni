# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for realtime streaming helpers (PR #2581 /v1/realtime path)."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.entrypoints.speech_to_text.realtime.protocol import TranscriptionDelta
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection, _PendingToolCall

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def realtime_conn() -> RealtimeConnection:
    return RealtimeConnection.__new__(RealtimeConnection)


@dataclass
class _FakeModelConfig:
    """Only the field `_async_chunk_enabled` reads."""

    async_chunk: bool = False


@dataclass
class _FakeServing:
    model_config: _FakeModelConfig = field(default_factory=_FakeModelConfig)

    def _is_model_supported(self, model: str | None) -> bool:
        """What upstream's `session.update` handling asks before accepting the event."""
        return model == _MODEL

    # Upstream's `_check_model` builds its refusal with this and then reads
    # `err.error.message` off it, so a fake that omits it turns the `model_not_found`
    # refusal - the one a real client is most likely to hit - into an AttributeError.
    # vLLM's own helper, so the shape cannot drift from what upstream reads.
    create_error_response = staticmethod(create_error_response)


# The model name `_FakeServing` accepts, so upstream takes a `session.update`.
_MODEL = "qwen3-omni"

# The same tool, as the OpenAI Realtime API spells it (flat) and as
# /v1/chat/completions does (nested). Both are accepted; the nested form is what
# gets stored.
_FLAT_WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather in a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}

_TOOL_CALL_TEXT = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Berlin"}}\n</tool_call>'


@pytest.fixture
def tool_call_conn() -> RealtimeConnection:
    conn = RealtimeConnection.__new__(RealtimeConnection)
    conn._tools = None
    conn._tool_choice = "auto"
    conn._tool_choice_not_enforced_warned = False
    conn._turn_tools = None
    # Upstream's own flag, and the one `_handle_session_update` reads to tell
    # whether the session was validated: a fresh connection starts False.
    conn._is_model_validated = False
    conn._tool_result_queue = asyncio.Queue()
    conn._pending_tool_calls = {}
    conn._tool_rounds = 0
    # Tool calling is only supported with async_chunk off, so that is the default
    # here; tests that care about the other mode set it explicitly.
    conn.serving = _FakeServing()
    return conn


def _patch_accepting_base(conn, mocker):
    """Stand in for upstream's `handle_event` with one that ACCEPTS the event.

    Upstream marks a session validated by setting `_is_model_validated`, and that
    flag is the signal `_handle_session_update` reads, so a stand-in that skipped
    it would leave every tool update looking refused. Tests about the gate itself
    drive the REAL base class instead - see
    TestToolStateFollowsUpstreamSessionValidation.
    """

    async def _accept(_event: dict) -> None:
        conn._is_model_validated = True

    return mocker.patch.object(
        VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock, side_effect=_accept
    )


class TestRealtimeConnectionTensorAndPcm:
    def test_tensor_to_numpy_none(self) -> None:
        assert RealtimeConnection._tensor_to_numpy(None) is None

    def test_tensor_to_numpy_1d_numpy(self) -> None:
        arr = np.array([1.0, 2.0], dtype=np.float64)
        out = RealtimeConnection._tensor_to_numpy(arr)
        assert out is not None
        assert out.dtype == np.float32
        assert out.shape == (2,)

    def test_tensor_to_numpy_2d_numpy_flattened(self) -> None:
        arr = np.array([[0.5], [-0.5]], dtype=np.float32)
        out = RealtimeConnection._tensor_to_numpy(arr)
        assert out is not None
        assert out.shape == (2,)

    def test_tensor_to_numpy_torch(self) -> None:
        t = torch.tensor([[0.25, -0.25]], dtype=torch.float32)
        out = RealtimeConnection._tensor_to_numpy(t)
        assert out is not None
        assert out.shape == (2,)
        np.testing.assert_allclose(out, [0.25, -0.25], rtol=1e-5)

    def test_pcm16_b64_roundtrip(self) -> None:
        audio = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        b64 = RealtimeConnection._pcm16_b64(audio)
        raw = base64.b64decode(b64)
        assert len(raw) == 6
        pcm = np.frombuffer(raw, dtype=np.int16)
        assert pcm[0] == 0
        assert pcm[1] == 32767
        assert pcm[2] == -32767


class TestAsyncOmniStreamingParamsValidation:
    def test_accepts_streaming_friendly_params(self) -> None:
        p = SamplingParams(
            n=1,
            stop=[],
            output_kind=RequestOutputKind.DELTA,
        )
        AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_non_sampling_params(self) -> None:
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(object())  # type: ignore[arg-type]

    def test_rejects_n_greater_than_one(self) -> None:
        p = SamplingParams(n=2, stop=[], output_kind=RequestOutputKind.DELTA)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_final_only(self) -> None:
        p = SamplingParams(n=1, stop=[], output_kind=RequestOutputKind.FINAL_ONLY)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_stop_strings(self) -> None:
        p = SamplingParams(n=1, stop=["\n"], output_kind=RequestOutputKind.DELTA)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)


class TestRealtimeConnectionToolCallEventRouting:
    """handle_event's tool-calling additions (session.update.tools capture,
    conversation.item.create routing) - see realtime_tool_calls.py for the
    <tool_call> text parser these events feed."""

    def test_session_update_captures_tools_and_delegates_to_base(self, tool_call_conn, mocker) -> None:
        base_handle_event = _patch_accepting_base(tool_call_conn, mocker)
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        event = {"type": "session.update", "model": "qwen3-omni", "tools": tools}

        asyncio.run(tool_call_conn.handle_event(event))

        assert tool_call_conn._tools == tools
        base_handle_event.assert_awaited_once_with(event)

    def test_session_update_without_tools_leaves_existing_tools_untouched(self, tool_call_conn, mocker) -> None:
        _patch_accepting_base(tool_call_conn, mocker)
        tool_call_conn._tools = [{"type": "function", "function": {"name": "existing"}}]

        asyncio.run(tool_call_conn.handle_event({"type": "session.update", "model": "qwen3-omni"}))

        assert tool_call_conn._tools == [{"type": "function", "function": {"name": "existing"}}]

    def test_conversation_item_create_function_call_output_is_queued(self, tool_call_conn) -> None:
        item = {"type": "function_call_output", "call_id": "call_1", "output": "sunny and 72"}

        asyncio.run(tool_call_conn.handle_event({"type": "conversation.item.create", "item": item}))

        assert tool_call_conn._tool_result_queue.qsize() == 1
        assert tool_call_conn._tool_result_queue.get_nowait() == item

    def test_conversation_item_create_unsupported_item_type_sends_error(self, tool_call_conn, mocker) -> None:
        send_error = mocker.patch.object(tool_call_conn, "send_error", new_callable=mocker.AsyncMock)

        asyncio.run(tool_call_conn.handle_event({"type": "conversation.item.create", "item": {"type": "not_a_thing"}}))

        send_error.assert_awaited_once()
        assert send_error.await_args.args[1] == "unsupported_item"
        assert tool_call_conn._tool_result_queue.empty()

    def test_unrelated_event_types_still_delegate_to_base(self, tool_call_conn, mocker) -> None:
        base_handle_event = _patch_accepting_base(tool_call_conn, mocker)
        event = {"type": "input_audio_buffer.commit", "final": True}

        asyncio.run(tool_call_conn.handle_event(event))

        base_handle_event.assert_awaited_once_with(event)


class TestRenderTokenPromptReattachesAudio:
    """Regression test for a real bug: the tool-call continuation re-submitted
    the engine's POST-expansion `output.prompt_token_ids` as a bare
    TokensPrompt. Those ids still contain the expanded `<|audio_pad|>` run for
    the user's spoken turn, but with no `multi_modal_data` the audio encoder
    output is gone - so the thinker saw placeholder tokens backed by nothing,
    lost the question entirely, and "answered" by emitting further tool calls
    for unrelated cities/items (and even fabricating tool results) instead of
    replying, never terminating. Fixed by splicing onto the PRE-expansion
    prompt and re-attaching its audio on every continuation."""

    def _conn(self, mocker):
        conn = RealtimeConnection.__new__(RealtimeConnection)
        conn.serving = mocker.Mock()
        conn.serving.model_config.is_encoder_decoder = False
        conn.serving.renderer.render_cmpl_async = mocker.AsyncMock(side_effect=lambda prompts: [dict(prompts[0])])
        return conn

    @staticmethod
    def _first(gen):
        async def _run():
            return await anext(gen)

        return asyncio.run(_run())

    def test_audio_is_reattached_to_continuation_prompt(self, mocker) -> None:
        conn = self._conn(mocker)
        audio = {"audio": np.zeros(16000, dtype=np.float32)}

        result = self._first(conn._render_token_prompt([1, 2, 3], audio))

        assert result.prompt["multi_modal_data"] is audio

    def test_no_multi_modal_data_is_a_noop(self, mocker) -> None:
        conn = self._conn(mocker)

        result = self._first(conn._render_token_prompt([1, 2, 3]))

        assert "multi_modal_data" not in result.prompt

    def test_turn_prompt_capture_keeps_unexpanded_ids_and_audio(self, mocker) -> None:
        """_buffer_realtime_audio_with_tools must stash the pre-render prompt so
        the continuation has something audio-bearing to splice onto."""
        conn = self._conn(mocker)
        conn._turn_tools = None
        conn._turn_prompt = None
        audio = {"audio": np.zeros(8000, dtype=np.float32)}
        prompt = TokensPrompt(prompt_token_ids=[10, 11], multi_modal_data=audio)

        async def _fake_buffer(*_args, **_kwargs):
            yield prompt

        conn.serving.model_cls.buffer_realtime_audio = _fake_buffer

        async def _drain():
            return [x async for x in conn._buffer_realtime_audio_with_tools(None, None)]

        asyncio.run(_drain())

        assert conn._turn_prompt["prompt_token_ids"] == [10, 11]
        assert conn._turn_prompt["multi_modal_data"] is audio


class TestCloseAssistantTurnBeforeToolResult:
    """Regression test for a real bug: the tool-result suffix opens with
    `<|im_start|>user`, but the raw generated token ids stop at the tool call
    without the `<|im_end|>` the chat template would emit. Splicing them
    directly produced `</tool_call><|im_start|>user`, leaving the assistant turn
    open. The thinker answered that malformed conversation by re-emitting the
    same tool call, looping until something bounded it - while the reference HF
    path (apply_chat_template, which closes the turn) answered the identical
    prompt correctly. Verified against a live Qwen3-Omni realtime session:
    tool results that previously looped 8+ times now answer in one round, with
    wording matching the reference implementation."""

    class _Tok:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == "<|im_end|>"
            return 151645

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert text == "\n"
            return [198]

    def test_appends_terminator_and_newline(self) -> None:
        out = RealtimeConnection._close_assistant_turn(self._Tok(), [1, 2, 3])
        assert out == [1, 2, 3, 151645, 198]

    def test_adds_only_newline_when_terminator_present(self) -> None:
        out = RealtimeConnection._close_assistant_turn(self._Tok(), [1, 2, 151645])
        assert out == [1, 2, 151645, 198]

    def test_is_idempotent_when_already_closed(self) -> None:
        out = RealtimeConnection._close_assistant_turn(self._Tok(), [1, 151645, 198])
        assert out == [1, 151645, 198]

    def test_does_not_mutate_caller_list(self) -> None:
        original = [1, 2, 3]
        RealtimeConnection._close_assistant_turn(self._Tok(), original)
        assert original == [1, 2, 3]

    def test_unknown_terminator_leaves_splice_untouched(self) -> None:
        class _Bad(self._Tok().__class__):
            def convert_tokens_to_ids(self, token: str) -> int:
                return 0  # == unk_token_id

        out = RealtimeConnection._close_assistant_turn(_Bad(), [1, 2, 3])
        assert out == [1, 2, 3]


class TestToolChainIsBounded:
    """A model that keeps re-emitting tool calls instead of answering must not
    recurse without bound: `_await_tool_results_and_continue` re-enters
    `_run_generation`, so an unbounded chain grows the stack and holds every
    round's result generator open. Observed during bring-up when the spliced
    prompt was malformed (see TestCloseAssistantTurnBeforeToolResult)."""

    def test_exceeding_max_rounds_reports_error_and_stops(self, tool_call_conn, mocker) -> None:
        tool_call_conn._is_connected = True
        tool_call_conn._tool_rounds = RealtimeConnection.MAX_TOOL_ROUNDS
        send_error = mocker.patch.object(tool_call_conn, "send_error", new_callable=mocker.AsyncMock)
        run_generation = mocker.patch.object(tool_call_conn, "_run_generation", new_callable=mocker.AsyncMock)

        asyncio.run(tool_call_conn._await_tool_results_and_continue([1, 2], [3]))

        send_error.assert_awaited_once()
        assert send_error.await_args.args[1] == "tool_call_loop"
        run_generation.assert_not_awaited()

    def test_rounds_below_the_cap_still_continue(self, tool_call_conn, mocker) -> None:
        """The cap must not fire early: with no pending calls the wait loop is
        skipped and generation continues."""
        tool_call_conn._is_connected = True
        tool_call_conn._tool_rounds = 0
        tool_call_conn._turn_prompt = None
        mocker.patch.object(tool_call_conn, "send_error", new_callable=mocker.AsyncMock)
        run_generation = mocker.patch.object(tool_call_conn, "_run_generation", new_callable=mocker.AsyncMock)
        mocker.patch(
            "vllm_omni.entrypoints.openai.realtime_connection.cached_tokenizer_from_config",
            return_value=mocker.Mock(
                encode=lambda text, add_special_tokens=True: [9],
                convert_tokens_to_ids=lambda token: 151645,
                unk_token_id=0,
            ),
        )
        mocker.patch("vllm_omni.entrypoints.openai.realtime_connection.cached_processor_from_config")
        mocker.patch(
            "vllm_omni.entrypoints.openai.realtime_connection.safe_apply_chat_template",
            return_value="<|im_start|>user\nresult<|im_end|>\n<|im_start|>assistant\n",
        )
        tool_call_conn.serving = mocker.Mock()

        asyncio.run(tool_call_conn._await_tool_results_and_continue([1, 2], [3]))

        run_generation.assert_awaited_once()


class TestToolResultWaitReleasesOnDisconnect:
    """A client that vanishes mid-tool-call must not park the generation task
    forever. The wait is bounded so `_is_connected` is re-checked instead of
    blocking indefinitely on an empty queue."""

    def test_wait_returns_when_client_disconnects(self, tool_call_conn, mocker) -> None:
        tool_call_conn._is_connected = False  # client already gone
        tool_call_conn._pending_tool_calls = {0: _PendingToolCall(call_id="call_x", name="get_weather")}
        run_generation = mocker.patch.object(tool_call_conn, "_run_generation", new_callable=mocker.AsyncMock)

        # Returns rather than hanging on the empty queue.
        asyncio.run(asyncio.wait_for(tool_call_conn._await_tool_results_and_continue([1], [2]), timeout=5))

        run_generation.assert_not_awaited()


class TestParallelToolResultsRenderSeparateBlocks:
    """Review #5555: joining parallel tool results into one `role="tool"` message
    produced a single `<tool_response>` holding both outputs, so the model could
    not associate each result with its call. The chat template emits one
    `<tool_response>` per tool message (grouping consecutive tool messages under
    one user turn), so each result must be its own message, in call order.
    """

    @staticmethod
    def _capture_suffix(conn, mocker, pending: dict[int, _PendingToolCall]) -> list[dict[str, str]]:
        """Drive `_await_tool_results_and_continue` and return the message list
        handed to the chat template."""
        conn._is_connected = True
        conn._tool_rounds = 0
        conn._turn_prompt = None
        conn._pending_tool_calls = pending
        conn.serving = mocker.Mock()
        mocker.patch.object(conn, "_run_generation", new_callable=mocker.AsyncMock)
        mocker.patch.object(conn, "send_error", new_callable=mocker.AsyncMock)
        mocker.patch(
            "vllm_omni.entrypoints.openai.realtime_connection.cached_tokenizer_from_config",
            return_value=mocker.Mock(
                encode=lambda text, add_special_tokens=True: [9],
                convert_tokens_to_ids=lambda token: 151645,
                unk_token_id=0,
            ),
        )
        mocker.patch("vllm_omni.entrypoints.openai.realtime_connection.cached_processor_from_config")
        apply_tmpl = mocker.patch(
            "vllm_omni.entrypoints.openai.realtime_connection.safe_apply_chat_template",
            return_value="<|im_start|>user\nr<|im_end|>\n<|im_start|>assistant\n",
        )
        asyncio.run(conn._await_tool_results_and_continue([1, 2], [3]))
        return apply_tmpl.call_args.args[2]

    def test_two_results_become_two_tool_messages_in_call_order(self, tool_call_conn, mocker) -> None:
        pending = {
            0: _PendingToolCall(call_id="call_a", name="get_weather"),
            1: _PendingToolCall(call_id="call_b", name="get_weather"),
        }
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_a", "output": "sunny 72"})
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_b", "output": "rainy 55"})

        messages = self._capture_suffix(tool_call_conn, mocker, pending)

        assert messages == [
            {"role": "tool", "content": "sunny 72"},
            {"role": "tool", "content": "rainy 55"},
        ]

    def test_call_order_is_kept_when_results_arrive_reversed(self, tool_call_conn, mocker) -> None:
        """Arrival order is the client's choice; call order is the model's."""
        pending = {
            0: _PendingToolCall(call_id="call_a", name="get_weather"),
            1: _PendingToolCall(call_id="call_b", name="get_weather"),
        }
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_b", "output": "rainy 55"})
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_a", "output": "sunny 72"})

        messages = self._capture_suffix(tool_call_conn, mocker, pending)

        assert [m["content"] for m in messages] == ["sunny 72", "rainy 55"]

    def test_duplicate_result_for_one_call_does_not_desync_the_wait(self, tool_call_conn, mocker) -> None:
        pending = {0: _PendingToolCall(call_id="call_a", name="get_weather")}
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_a", "output": "first"})
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_a", "output": "second"})

        messages = self._capture_suffix(tool_call_conn, mocker, pending)

        assert messages == [{"role": "tool", "content": "first"}]


class TestToolResultValidation:
    """Review #5555: any dict with `type="function_call_output"` was enqueued, so a
    missing/non-string `call_id` or a non-string `output` was accepted and the turn
    then waited for a result that could never match - with nothing reported to the
    client. Shape problems are protocol errors."""

    @staticmethod
    def _create(item: dict) -> dict:
        return {"type": "conversation.item.create", "item": item}

    def _run(self, conn, mocker, item: dict):
        send_error = mocker.patch.object(conn, "send_error", new_callable=mocker.AsyncMock)
        mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)
        asyncio.run(conn.handle_event(self._create(item)))
        return send_error

    def test_missing_call_id_is_rejected(self, tool_call_conn, mocker) -> None:
        send_error = self._run(tool_call_conn, mocker, {"type": "function_call_output", "output": "x"})
        assert send_error.await_args.args[1] == "invalid_function_call_output"
        assert tool_call_conn._tool_result_queue.empty()

    def test_non_string_call_id_is_rejected(self, tool_call_conn, mocker) -> None:
        send_error = self._run(tool_call_conn, mocker, {"type": "function_call_output", "call_id": 7, "output": "x"})
        assert send_error.await_args.args[1] == "invalid_function_call_output"
        assert tool_call_conn._tool_result_queue.empty()

    def test_empty_call_id_is_rejected(self, tool_call_conn, mocker) -> None:
        """`call_id: str` alone accepts "", which can never match a pending call."""
        send_error = self._run(tool_call_conn, mocker, {"type": "function_call_output", "call_id": "", "output": "x"})
        assert send_error.await_args.args[1] == "invalid_function_call_output"
        assert tool_call_conn._tool_result_queue.empty()

    def test_non_string_output_is_rejected(self, tool_call_conn, mocker) -> None:
        send_error = self._run(
            tool_call_conn, mocker, {"type": "function_call_output", "call_id": "call_a", "output": {"a": 1}}
        )
        assert send_error.await_args.args[1] == "invalid_function_call_output"
        assert tool_call_conn._tool_result_queue.empty()

    def test_well_formed_result_is_enqueued(self, tool_call_conn, mocker) -> None:
        send_error = self._run(
            tool_call_conn, mocker, {"type": "function_call_output", "call_id": "call_a", "output": "sunny"}
        )
        send_error.assert_not_awaited()
        assert tool_call_conn._tool_result_queue.qsize() == 1

    def test_unknown_call_id_is_reported_to_the_client(self, tool_call_conn, mocker) -> None:
        """Previously only logged, so a client typo produced silence."""
        tool_call_conn._is_connected = True
        tool_call_conn._tool_rounds = 0
        tool_call_conn._pending_tool_calls = {0: _PendingToolCall(call_id="call_a", name="get_weather")}
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_TYPO", "output": "sunny"})
        send_error = mocker.patch.object(tool_call_conn, "send_error", new_callable=mocker.AsyncMock)
        mocker.patch.object(tool_call_conn, "_run_generation", new_callable=mocker.AsyncMock)

        async def _drive() -> None:
            task = asyncio.ensure_future(tool_call_conn._await_tool_results_and_continue([1], [2]))
            for _ in range(40):
                await asyncio.sleep(0.02)
                if send_error.await_count:
                    break
            tool_call_conn._is_connected = False
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(_drive())

        assert send_error.await_args.args[1] == "unknown_tool_call_id"


class TestToolsRejectedUnderAsyncChunk:
    """Review #5555: with async_chunk on, the buffer yields one TokensPrompt per
    segment, so a tool-call continuation reattached only the final segment and lost
    the start of the utterance. Aggregating the audio would not be enough - the
    generation loop also never sees one complete thinker turn to scan for a
    <tool_call> block - so tools are refused outright instead.

    The REAL base class runs here, not a stand-in: the refusal has to survive the
    same path upstream's own validation takes, and be reported without any tool
    state having been stored."""

    def _session_update(self, conn, mocker, async_chunk: bool, tools: list[dict]):
        conn.serving = _FakeServing(_FakeModelConfig(async_chunk=async_chunk))
        send_error = mocker.patch.object(conn, "send_error", new_callable=mocker.AsyncMock)
        asyncio.run(conn.handle_event({"type": "session.update", "model": _MODEL, "tools": tools}))
        return send_error

    def test_tools_rejected_when_async_chunk_enabled(self, tool_call_conn, mocker) -> None:
        send_error = self._session_update(tool_call_conn, mocker, async_chunk=True, tools=[_WEATHER_TOOL])
        assert send_error.await_args.args[1] == "tools_require_no_async_chunk"
        assert tool_call_conn._tools is None
        assert tool_call_conn._is_model_validated  # the session itself was still validated

    def test_tools_accepted_when_async_chunk_disabled(self, tool_call_conn, mocker) -> None:
        send_error = self._session_update(tool_call_conn, mocker, async_chunk=False, tools=[_WEATHER_TOOL])
        send_error.assert_not_awaited()
        assert tool_call_conn._tools == [_WEATHER_TOOL]

    def test_empty_tool_list_still_clears_under_async_chunk(self, tool_call_conn, mocker) -> None:
        """Review #5555: `tools: []` is a CLEAR, and it was being refused with
        `tools_require_no_async_chunk` - so the tools the client asked to drop
        stayed on the session, in the one mode where the endpoint cannot run them.
        Clearing needs nothing async_chunk breaks."""
        tool_call_conn._tools = [_WEATHER_TOOL]

        send_error = self._session_update(tool_call_conn, mocker, async_chunk=True, tools=[])

        send_error.assert_not_awaited()
        assert tool_call_conn._tools == []
        assert tool_call_conn._active_tools() is None


class TestSessionUpdateShapes:
    """Review #5555: `tools` was read off the TOP LEVEL of `session.update`, but
    the OpenAI Realtime API nests session fields under a `session` object - and
    `OpenAIBaseModel` allows extra fields, so a spec-shaped client's `session`
    validated cleanly and its tools were silently dropped. Both shapes are
    accepted now (a nested value wins), as are both shapes of a tool item, while
    the event itself must come through untouched, because upstream reads `model`
    off the same dict afterwards."""

    def test_nested_session_tools_are_applied(self, tool_call_conn, mocker) -> None:
        base_handle_event = _patch_accepting_base(tool_call_conn, mocker)
        event = {"type": "session.update", "model": "qwen3-omni", "session": {"tools": [_WEATHER_TOOL]}}

        asyncio.run(tool_call_conn.handle_event(event))

        # A nested definition round-trips: validating one must not rewrite it,
        # these dicts go on to the chat template.
        assert tool_call_conn._tools == [_WEATHER_TOOL]
        base_handle_event.assert_awaited_once_with(event)

    def test_flat_tool_item_is_normalized_to_the_nested_form(self, tool_call_conn, mocker) -> None:
        """The headline bug: the OpenAI Realtime API's tool item is flat, so a
        real Realtime client's tools were never applied - first silently, then as
        a confusing `function` error. The chat template and
        `ChatCompletionToolsParam` want the fields under `function`."""
        _patch_accepting_base(tool_call_conn, mocker)
        event = {"type": "session.update", "model": "m", "session": {"tools": [_FLAT_WEATHER_TOOL]}}

        asyncio.run(tool_call_conn.handle_event(event))

        assert tool_call_conn._tools == [_WEATHER_TOOL]

    def test_null_nested_tools_keeps_the_top_level_list(self, tool_call_conn, mocker) -> None:
        """A `session` object that spells `tools` out as null must not discard a
        real list sent in the same event: nothing was reported, and mid-session
        the old tools stayed in place while the client believed it had replaced
        them."""
        _patch_accepting_base(tool_call_conn, mocker)
        event = {"type": "session.update", "model": "m", "tools": [_WEATHER_TOOL], "session": {"tools": None}}

        asyncio.run(tool_call_conn.handle_event(event))

        assert tool_call_conn._tools == [_WEATHER_TOOL]

    def test_null_nested_tool_choice_keeps_the_top_level_one(self, tool_call_conn, mocker) -> None:
        _patch_accepting_base(tool_call_conn, mocker)
        event = {"type": "session.update", "model": "m", "tool_choice": "none", "session": {"tool_choice": None}}

        asyncio.run(tool_call_conn.handle_event(event))

        assert tool_call_conn._tool_choice == "none"

    def test_non_string_model_is_left_to_upstream(self, tool_call_conn, mocker) -> None:
        """`model` is vLLM's own field on this event: validating it here added a
        second, misleading error to upstream's own complaint about it."""
        send_error = mocker.patch.object(tool_call_conn, "send_error", new_callable=mocker.AsyncMock)
        base_handle_event = _patch_accepting_base(tool_call_conn, mocker)

        asyncio.run(tool_call_conn.handle_event({"type": "session.update", "model": 7}))

        send_error.assert_not_awaited()
        base_handle_event.assert_awaited_once_with({"type": "session.update", "model": 7})

    def test_nested_session_wins_over_a_top_level_copy(self, tool_call_conn, mocker) -> None:
        _patch_accepting_base(tool_call_conn, mocker)
        nested = {"type": "function", "function": {"name": "nested"}}
        event = {
            "type": "session.update",
            "model": "m",
            "tools": [{"type": "function", "function": {"name": "flat"}}],
            "session": {"tools": [nested]},
        }

        asyncio.run(tool_call_conn.handle_event(event))

        assert tool_call_conn._tools == [nested]

    def test_event_is_not_mutated(self, tool_call_conn, mocker) -> None:
        _patch_accepting_base(tool_call_conn, mocker)
        event = {"type": "session.update", "model": "m", "session": {"tools": [_WEATHER_TOOL], "tool_choice": "none"}}
        before = copy.deepcopy(event)

        asyncio.run(tool_call_conn.handle_event(event))

        assert event == before

    def test_nested_model_is_lifted_into_a_copy_for_upstream(self, tool_call_conn, mocker) -> None:
        """Review #5555: the canonical Realtime `session.update` puts `model`
        inside `session`, and upstream reads it off the TOP level - so the whole
        nested-`session` shape was refused as "Missing required field: model" and
        its tools discarded. Lifted for upstream, in a copy: the caller's event
        still goes untouched (see test_event_is_not_mutated)."""
        base_handle_event = _patch_accepting_base(tool_call_conn, mocker)
        event = {"type": "session.update", "session": {"model": _MODEL, "tools": [_WEATHER_TOOL]}}
        before = copy.deepcopy(event)

        asyncio.run(tool_call_conn.handle_event(event))

        base_handle_event.assert_awaited_once_with({**before, "model": _MODEL})
        assert event == before
        assert tool_call_conn._tools == [_WEATHER_TOOL]

    def test_top_level_model_wins_over_a_nested_one(self, tool_call_conn, mocker) -> None:
        """Upstream owns that field; this only fills it in when it is absent."""
        base_handle_event = _patch_accepting_base(tool_call_conn, mocker)
        event = {"type": "session.update", "model": "top", "session": {"model": "nested"}}

        asyncio.run(tool_call_conn.handle_event(event))

        base_handle_event.assert_awaited_once_with(event)

    def test_malformed_tool_definition_is_rejected(self, tool_call_conn, mocker) -> None:
        """A tool with no `function.name` would otherwise blow up inside the chat
        template, mid-generation, as a generic processing error."""
        send_error = mocker.patch.object(tool_call_conn, "send_error", new_callable=mocker.AsyncMock)
        base_handle_event = _patch_accepting_base(tool_call_conn, mocker)

        asyncio.run(
            tool_call_conn.handle_event(
                {"type": "session.update", "model": "m", "tools": [{"type": "function", "function": {}}]}
            )
        )

        assert send_error.await_args.args[1] == "invalid_session_update"
        assert tool_call_conn._tools is None
        # Reported, but the event still reaches upstream: a client that got its
        # tools wrong should not additionally be told, on its next commit, that
        # its model was never validated.
        base_handle_event.assert_awaited_once()


class TestToolChoice:
    """`tool_choice` on `session.update`. `none` has to genuinely disable tool
    calling, `auto` is the default, and `required` or a named function is accepted
    and treated as `required` but cannot be enforced without guided decoding - see
    RealtimeConnection's scope notes. A field the event does not carry keeps its
    current value, so `tools` and `tool_choice` can be set in separate updates."""

    def _update(self, conn, mocker, **fields) -> None:
        _patch_accepting_base(conn, mocker)
        asyncio.run(conn.handle_event({"type": "session.update", "model": "m", **fields}))

    def test_auto_is_the_default_and_keeps_tools_active(self, tool_call_conn, mocker) -> None:
        self._update(tool_call_conn, mocker, tools=[_WEATHER_TOOL])

        assert tool_call_conn._tool_choice == "auto"
        assert tool_call_conn._active_tools() == [_WEATHER_TOOL]

    def test_none_disables_tools_but_keeps_them_on_the_session(self, tool_call_conn, mocker) -> None:
        self._update(tool_call_conn, mocker, tools=[_WEATHER_TOOL], tool_choice="none")

        assert tool_call_conn._tools == [_WEATHER_TOOL]
        assert tool_call_conn._active_tools() is None

    def test_later_auto_re_enables_the_declared_tools(self, tool_call_conn, mocker) -> None:
        self._update(tool_call_conn, mocker, tools=[_WEATHER_TOOL], tool_choice="none")
        self._update(tool_call_conn, mocker, tool_choice="auto")

        assert tool_call_conn._active_tools() == [_WEATHER_TOOL]

    def test_update_without_tool_choice_keeps_the_current_one(self, tool_call_conn, mocker) -> None:
        self._update(tool_call_conn, mocker, tool_choice="none")
        self._update(tool_call_conn, mocker, tools=[_WEATHER_TOOL])

        assert tool_call_conn._tool_choice == "none"
        assert tool_call_conn._active_tools() is None

    def test_required_is_recorded_and_leaves_tools_active(self, tool_call_conn, mocker) -> None:
        self._update(tool_call_conn, mocker, tools=[_WEATHER_TOOL], tool_choice="required")

        assert tool_call_conn._tool_choice == "required"
        assert tool_call_conn._active_tools() == [_WEATHER_TOOL]

    def test_named_function_is_treated_as_required(self, tool_call_conn, mocker) -> None:
        """OpenAI-legal, and rejecting it sank the whole event - taking the `tools`
        alongside it down too, so tool calling silently turned off for the session.
        Honored as far as this path can: like `required`, minus the enforcement.
        The name itself is not kept, because nothing here could act on it."""
        self._update(
            tool_call_conn,
            mocker,
            tools=[_WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        assert tool_call_conn._tool_choice == "required"
        assert tool_call_conn._active_tools() == [_WEATHER_TOOL]

    def test_flat_named_function_is_accepted_too(self, tool_call_conn, mocker) -> None:
        """The Realtime API spells the named form flat, without the wrapper."""
        send_error = mocker.patch.object(tool_call_conn, "send_error", new_callable=mocker.AsyncMock)

        self._update(
            tool_call_conn, mocker, tools=[_WEATHER_TOOL], tool_choice={"type": "function", "name": "get_weather"}
        )

        send_error.assert_not_awaited()
        assert tool_call_conn._tool_choice == "required"
        assert tool_call_conn._active_tools() == [_WEATHER_TOOL]

    def test_nested_tool_choice_is_applied(self, tool_call_conn, mocker) -> None:
        self._update(tool_call_conn, mocker, session={"tool_choice": "none"})

        assert tool_call_conn._tool_choice == "none"

    def test_unknown_tool_choice_is_rejected(self, tool_call_conn, mocker) -> None:
        send_error = mocker.patch.object(tool_call_conn, "send_error", new_callable=mocker.AsyncMock)

        self._update(tool_call_conn, mocker, tools=[_WEATHER_TOOL], tool_choice="always")

        assert send_error.await_args.args[1] == "invalid_session_update"
        # Nothing is half-applied: the whole event is rejected.
        assert tool_call_conn._tool_choice == "auto"
        assert tool_call_conn._tools is None


class TestEmptyToolList:
    """`tools: []` is how an OpenAI client clears its tools, so it has to mean the
    same as never having declared any."""

    def test_empty_list_is_recorded_and_turns_tool_calling_off(self, tool_call_conn, mocker) -> None:
        _patch_accepting_base(tool_call_conn, mocker)

        asyncio.run(tool_call_conn.handle_event({"type": "session.update", "model": "m", "tools": [_WEATHER_TOOL]}))
        asyncio.run(tool_call_conn.handle_event({"type": "session.update", "model": "m", "tools": []}))

        assert tool_call_conn._tools == []
        assert tool_call_conn._active_tools() is None


class TestToolChoiceDowngradeIsLogged:
    """Review #5555: `required` and a named function are accepted but behave like
    `auto`, and nothing said so at runtime - the operator saw a session ask for a
    forced call and get an ordinary one. There is no `session.updated` event on
    this endpoint (nor upstream) to tell the client, so it goes to the log the way
    the other two unenforceable situations in this file do."""

    def _update(self, conn, mocker, **fields) -> None:
        _patch_accepting_base(conn, mocker)
        asyncio.run(conn.handle_event({"type": "session.update", "model": "m", **fields}))

    def test_required_logs_a_warning(self, tool_call_conn, mocker) -> None:
        logger = mocker.patch("vllm_omni.entrypoints.openai.realtime_connection.logger")

        self._update(tool_call_conn, mocker, tools=[_WEATHER_TOOL], tool_choice="required")

        logger.warning.assert_called_once()
        assert "tool_choice" in logger.warning.call_args.args[0]

    def test_named_function_logs_the_name_it_asked_for(self, tool_call_conn, mocker) -> None:
        logger = mocker.patch("vllm_omni.entrypoints.openai.realtime_connection.logger")

        self._update(
            tool_call_conn, mocker, tools=[_WEATHER_TOOL], tool_choice={"type": "function", "name": "get_weather"}
        )

        logger.warning.assert_called_once()
        assert "get_weather" in logger.warning.call_args.args[1]

    def test_warning_is_logged_once_per_session(self, tool_call_conn, mocker) -> None:
        """A client that repeats `tool_choice` on every update must not fill the
        log with it."""
        logger = mocker.patch("vllm_omni.entrypoints.openai.realtime_connection.logger")

        self._update(tool_call_conn, mocker, tool_choice="required")
        self._update(tool_call_conn, mocker, tool_choice="required")
        self._update(tool_call_conn, mocker, tool_choice={"type": "function", "name": "get_weather"})

        logger.warning.assert_called_once()

    def test_enforceable_choices_log_nothing(self, tool_call_conn, mocker) -> None:
        logger = mocker.patch("vllm_omni.entrypoints.openai.realtime_connection.logger")

        self._update(tool_call_conn, mocker, tools=[_WEATHER_TOOL], tool_choice="auto")
        self._update(tool_call_conn, mocker, tool_choice="none")

        logger.warning.assert_not_called()


class TestValidationMessagesDescribeTheClientPayload:
    """Review #5555: the `invalid_session_update` message is the only thing the
    client gets, and it was written in this module's terms - naming the `function`
    key that `_nest_flat_function` ADDS to a flat tool item (so the fix it asked
    for was to a key the client never sent), and naming the pydantic models
    themselves."""

    def _message(self, conn, mocker, event: dict) -> str:
        send_error = mocker.patch.object(conn, "send_error", new_callable=mocker.AsyncMock)
        _patch_accepting_base(conn, mocker)
        asyncio.run(conn.handle_event(event))
        assert send_error.await_args.args[1] == "invalid_session_update"
        return send_error.await_args.args[0]

    def test_flat_tool_item_is_named_flat(self, tool_call_conn, mocker) -> None:
        flat_without_name = {"type": "function", "description": "Get the weather in a city"}

        message = self._message(
            tool_call_conn, mocker, {"type": "session.update", "model": "m", "tools": [flat_without_name]}
        )

        assert message == "tools.0.name: Field required"

    def test_nested_tool_item_is_named_nested(self, tool_call_conn, mocker) -> None:
        """The client that did send `function` is still told about `function`."""
        message = self._message(
            tool_call_conn,
            mocker,
            {"type": "session.update", "model": "m", "tools": [{"type": "function", "function": {}}]},
        )

        assert message == "tools.0.function.name: Field required"

    def test_tool_item_of_the_wrong_type_does_not_name_the_model(self, tool_call_conn, mocker) -> None:
        message = self._message(tool_call_conn, mocker, {"type": "session.update", "model": "m", "tools": ["nope"]})

        assert message == "tools.0: Input should be a valid dictionary"


class TestToolStateFollowsUpstreamSessionValidation:
    """Review #5555: tool state was committed before upstream had looked at the
    event, so a `session.update` upstream refuses still took effect -
    `{"type": "session.update", "tool_choice": "none"}` with no `model` is refused
    upstream and yet disabled tool calling for good, since no later update that
    omits `tool_choice` can undo it.

    The REAL base class runs in every test here, and neither `send_error` nor
    `handle_event` is mocked: the whole point is which of upstream's states the
    tool fields ride on (`_is_model_validated`, which upstream sets when it takes
    a `session.update` and never clears), so the errors written to the socket and
    that flag are the assertions."""

    def _handle(self, conn, mocker, event: dict) -> list[dict]:
        conn.websocket = mocker.Mock(send_text=mocker.AsyncMock())
        asyncio.run(conn.handle_event(event))
        return [json.loads(call.args[0]) for call in conn.websocket.send_text.await_args_list]

    def test_refused_update_leaves_the_current_choice_alone(self, tool_call_conn, mocker) -> None:
        tool_call_conn._tools = [_WEATHER_TOOL]

        # No `model` on a session upstream has never validated: refused with
        # "Missing required field".
        errors = self._handle(tool_call_conn, mocker, {"type": "session.update", "tool_choice": "none"})

        assert [error["code"] for error in errors] == ["invalid_event"]
        assert not tool_call_conn._is_model_validated
        assert tool_call_conn._tool_choice == "auto"
        assert tool_call_conn._active_tools() == [_WEATHER_TOOL]

    def test_refused_update_does_not_apply_its_tools_either(self, tool_call_conn, mocker) -> None:
        errors = self._handle(tool_call_conn, mocker, {"type": "session.update", "tools": [_WEATHER_TOOL]})

        assert [error["code"] for error in errors] == ["invalid_event"]
        assert tool_call_conn._tools is None

    def test_refused_update_reports_upstream_s_error_only(self, tool_call_conn, mocker) -> None:
        """One client-visible error per unvalidated event: the malformed `tools`
        would be a second complaint about an event that has to be resent whole."""
        event = {"type": "session.update", "tools": [{"type": "function", "function": {}}]}

        errors = self._handle(tool_call_conn, mocker, event)

        assert [error["code"] for error in errors] == ["invalid_event"]

    def test_unknown_model_is_refused_and_applies_nothing(self, tool_call_conn, mocker) -> None:
        """The likelier refusal in practice, and the one this gate exists for:
        upstream builds it through `serving.create_error_response`."""
        event = {"type": "session.update", "model": "not-served", "tools": [_WEATHER_TOOL]}

        errors = self._handle(tool_call_conn, mocker, event)

        assert [error["code"] for error in errors] == ["model_not_found"]
        assert not tool_call_conn._is_model_validated
        assert tool_call_conn._tools is None

    def test_accepted_update_still_applies(self, tool_call_conn, mocker) -> None:
        event = {"type": "session.update", "model": _MODEL, "tools": [_WEATHER_TOOL], "tool_choice": "none"}

        errors = self._handle(tool_call_conn, mocker, event)

        assert errors == []
        assert tool_call_conn._is_model_validated  # upstream took the event
        assert tool_call_conn._tools == [_WEATHER_TOOL]
        assert tool_call_conn._tool_choice == "none"

    def test_later_update_without_model_still_applies_its_tools(self, tool_call_conn, mocker) -> None:
        """Regression: a `session.update` that omits `model` stopped updating tools
        at all, on a session whose model upstream had already validated - which is
        every session after its first update, since clients send `model` once. The
        base branch applied those tools. Upstream still answers the missing field
        with its own error; that is upstream's call, not a reason to drop the
        tools."""
        first = {"type": "session.update", "model": _MODEL, "tools": [_WEATHER_TOOL]}
        assert self._handle(tool_call_conn, mocker, first) == []

        errors = self._handle(tool_call_conn, mocker, {"type": "session.update", "tool_choice": "none"})

        assert [error["code"] for error in errors] == ["invalid_event"]  # upstream's, about `model`
        assert tool_call_conn._tool_choice == "none"
        assert tool_call_conn._active_tools() is None

    def test_nested_model_is_accepted_and_its_tools_applied(self, tool_call_conn, mocker) -> None:
        """The spec-conformant shape, end to end through the real base class: the
        whole nested-`session` support is unreachable if this event is refused."""
        event = {"type": "session.update", "session": {"model": _MODEL, "tools": [_FLAT_WEATHER_TOOL]}}

        errors = self._handle(tool_call_conn, mocker, event)

        assert errors == []
        assert tool_call_conn._is_model_validated
        assert tool_call_conn._tools == [_WEATHER_TOOL]
        assert tool_call_conn._active_tools() == [_WEATHER_TOOL]

    def test_async_chunk_refusal_stores_nothing_on_a_validated_session(self, tool_call_conn, mocker) -> None:
        """The refusal has to reach the client and leave no tool state behind, on
        the path where the session IS validated - the one the gate lets through."""
        tool_call_conn.serving = _FakeServing(_FakeModelConfig(async_chunk=True))
        event = {"type": "session.update", "model": _MODEL, "tools": [_WEATHER_TOOL], "tool_choice": "none"}

        errors = self._handle(tool_call_conn, mocker, event)

        assert [error["code"] for error in errors] == ["tools_require_no_async_chunk"]
        assert tool_call_conn._tools is None
        # `tool_choice` is session state and still lands - a later update may bring
        # tools this server can apply.
        assert tool_call_conn._tool_choice == "none"


def _wire_one_generation(conn, mocker, text: str) -> dict:
    """Wire `conn` up for one generation over a single output carrying `text`.

    The fake engine consumes the prompt stream the way the real one does, so what
    the model's `buffer_realtime_audio` was told to declare lands in `declared`.
    """
    declared: dict = {}

    async def _fake_buffer(audio_stream, input_stream, model_config, tools=None):
        declared["tools"] = tools
        yield TokensPrompt(prompt_token_ids=[10, 11])

    async def _outputs(prompt=None, **_kwargs):
        if prompt is not None:
            async for _ in prompt:
                pass
        yield SimpleNamespace(
            stage_id=0,
            outputs=[SimpleNamespace(text=text, token_ids=[7])],
            prompt_token_ids=[1, 2],
            multimodal_output=None,
        )

    conn.connection_id = "ws-test"
    conn._is_connected = True
    conn._turn_prompt = None
    conn.generation_task = None
    conn.audio_queue = asyncio.Queue()
    conn.serving = mocker.Mock()
    conn.serving.model_config.is_encoder_decoder = False
    conn.serving.model_cls.buffer_realtime_audio = _fake_buffer
    conn.serving.renderer.render_cmpl_async = mocker.AsyncMock(side_effect=lambda prompts: [dict(prompts[0])])
    conn.engine = mocker.Mock()
    conn.engine.default_sampling_params_list = []
    conn.engine.generate = lambda **kwargs: _outputs(**kwargs)
    mocker.patch(
        "vllm_omni.entrypoints.openai.realtime_connection.coerce_param_message_types",
        side_effect=lambda params, is_streaming: params,
    )
    return {
        "declared": declared,
        "send": mocker.patch.object(conn, "send", new_callable=mocker.AsyncMock),
        "send_json": mocker.patch.object(conn, "send_json", new_callable=mocker.AsyncMock),
        "send_error": mocker.patch.object(conn, "send_error", new_callable=mocker.AsyncMock),
        "await_results": mocker.patch.object(conn, "_await_tool_results_and_continue", new_callable=mocker.AsyncMock),
    }


def _run_turn(conn) -> None:
    """One user turn, the way an `input_audio_buffer.commit` drives it."""

    async def _turn():
        await conn.start_generation()
        await conn.generation_task

    asyncio.run(_turn())


def _streamed_text(mocks: dict) -> str:
    return "".join(
        call.args[0].delta for call in mocks["send"].await_args_list if isinstance(call.args[0], TranscriptionDelta)
    )


def _sent_events(mocks: dict) -> list[dict]:
    return [call.args[0] for call in mocks["send_json"].await_args_list]


_TOOL_CALL_EVENTS = [
    "response.output_item.added",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
]


class TestBothToolGatesFollowOnePredicate:
    """`_active_tools` decides both halves of the feature at once: whether the
    `<tools>` preamble goes into the rendered prompt (`buffer_realtime_audio`
    renders the plain template without it) and whether the generated text is
    scanned for `<tool_call>`. The review found the two gates reading different
    predicates - `tools: []` declared nothing and still armed the extractor,
    which swallows the turn's audio and then waits for a tool result the client
    was never asked for - so every state is asserted on both at once."""

    def _turn(self, conn, mocker, text: str = _TOOL_CALL_TEXT) -> dict:
        mocks = _wire_one_generation(conn, mocker, text)
        _run_turn(conn)
        mocks["send_error"].assert_not_awaited()  # nothing fell into the generic error path
        return mocks

    def test_auto_declares_the_tools_and_extracts_the_call(self, tool_call_conn, mocker) -> None:
        tool_call_conn._tools = [_WEATHER_TOOL]

        mocks = self._turn(tool_call_conn, mocker)
        sent = _sent_events(mocks)

        assert mocks["declared"]["tools"] == [_WEATHER_TOOL]
        assert _streamed_text(mocks) == ""
        assert [event["type"] for event in sent] == _TOOL_CALL_EVENTS
        mocks["await_results"].assert_awaited_once()
        # The wire payloads come from the event models now; their fields are what
        # the example client and the integration test read.
        assert sent[0]["item"]["name"] == "get_weather"
        assert set(sent[0]["item"]) == {"type", "name", "call_id"}
        assert set(sent[-1]) == {"type", "call_id", "arguments"}
        assert sent[-1]["arguments"] == '{"city": "Berlin"}'
        assert sent[-1]["call_id"] == sent[0]["item"]["call_id"]

    def test_none_declares_nothing_and_streams_the_text(self, tool_call_conn, mocker) -> None:
        tool_call_conn._tools = [_WEATHER_TOOL]
        tool_call_conn._tool_choice = "none"

        mocks = self._turn(tool_call_conn, mocker)

        assert mocks["declared"]["tools"] is None
        # A model that emits the tag unprompted is reported as the text it is.
        assert _streamed_text(mocks) == _TOOL_CALL_TEXT
        assert [event["type"] for event in _sent_events(mocks)] == ["response.audio.done"]
        mocks["await_results"].assert_not_awaited()

    def test_empty_tool_list_declares_nothing_and_streams_the_text(self, tool_call_conn, mocker) -> None:
        tool_call_conn._tools = []

        mocks = self._turn(tool_call_conn, mocker)

        assert mocks["declared"]["tools"] is None
        assert _streamed_text(mocks) == _TOOL_CALL_TEXT
        assert [event["type"] for event in _sent_events(mocks)] == ["response.audio.done"]
        mocks["await_results"].assert_not_awaited()


class TestMidTurnUpdateDoesNotReachTheTurnInFlight:
    """A `session.update` that lands while the client is running a tool must not
    change the turn it lands in. The tool state used to be read per
    `_run_generation` call rather than per turn, so a `tool_choice` flip also
    flipped the tool-call CONTINUATION - whose prompt still carries the `<tools>`
    preamble and the call history, so the extractor and the prompt then disagree
    for the rest of the turn."""

    def test_continuation_keeps_the_tools_the_turn_started_with(self, tool_call_conn, mocker) -> None:
        tool_call_conn._tools = [_WEATHER_TOOL]
        mocks = _wire_one_generation(tool_call_conn, mocker, _TOOL_CALL_TEXT)
        _run_turn(tool_call_conn)
        mocks["await_results"].assert_awaited_once()

        # The client's `session.update` arrives while it is running the tool.
        tool_call_conn._tool_choice = "none"

        # Only the continuation's events matter from here.
        mocks["send"].reset_mock()
        mocks["send_json"].reset_mock()
        asyncio.run(tool_call_conn._run_generation(None, asyncio.Queue()))

        assert _streamed_text(mocks) == ""
        assert [event["type"] for event in _sent_events(mocks)] == _TOOL_CALL_EVENTS

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for realtime streaming helpers (PR #2581 /v1/realtime path)."""

from __future__ import annotations

import asyncio
import base64

import numpy as np
import pytest
import torch
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def realtime_conn() -> RealtimeConnection:
    return RealtimeConnection.__new__(RealtimeConnection)


@pytest.fixture
def tool_call_conn() -> RealtimeConnection:
    conn = RealtimeConnection.__new__(RealtimeConnection)
    conn._tools = None
    conn._speaker = None
    conn._instructions = None
    conn._tool_result_queue = asyncio.Queue()
    conn._pending_tool_calls = {}
    return conn


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

    def _patch_base_handle_event(self, mocker):
        return mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)

    def test_session_update_captures_tools_and_delegates_to_base(self, tool_call_conn, mocker) -> None:
        base_handle_event = self._patch_base_handle_event(mocker)
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        event = {"type": "session.update", "model": "qwen3-omni", "tools": tools}

        asyncio.run(tool_call_conn.handle_event(event))

        assert tool_call_conn._tools == tools
        base_handle_event.assert_awaited_once_with(event)

    def test_session_update_without_tools_leaves_existing_tools_untouched(self, tool_call_conn, mocker) -> None:
        self._patch_base_handle_event(mocker)
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
        assert tool_call_conn._tool_result_queue.empty()

    def test_unrelated_event_types_still_delegate_to_base(self, tool_call_conn, mocker) -> None:
        base_handle_event = self._patch_base_handle_event(mocker)
        event = {"type": "input_audio_buffer.commit", "final": True}

        asyncio.run(tool_call_conn.handle_event(event))

        base_handle_event.assert_awaited_once_with(event)


class TestRealtimeConnectionSpeakerRouting:
    """handle_event's speaker/voice-selection addition to session.update -
    threaded into buffer_realtime_audio's own `speaker` param (see
    Qwen3OmniMoeForConditionalGeneration.buffer_realtime_audio), the same
    `additional_information={"speaker": [...]}` shape /v1/chat/completions
    already uses (serving_chat.py)."""

    def _patch_base_handle_event(self, mocker):
        return mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)

    def test_session_update_captures_voice_field(self, tool_call_conn, mocker) -> None:
        self._patch_base_handle_event(mocker)

        asyncio.run(tool_call_conn.handle_event({"type": "session.update", "model": "qwen3-omni", "voice": "aiden"}))

        assert tool_call_conn._speaker == "aiden"

    def test_session_update_captures_speaker_field(self, tool_call_conn, mocker) -> None:
        self._patch_base_handle_event(mocker)

        asyncio.run(tool_call_conn.handle_event({"type": "session.update", "model": "qwen3-omni", "speaker": "ethan"}))

        assert tool_call_conn._speaker == "ethan"

    def test_session_update_without_voice_or_speaker_leaves_existing_value_untouched(
        self, tool_call_conn, mocker
    ) -> None:
        self._patch_base_handle_event(mocker)
        tool_call_conn._speaker = "aiden"

        asyncio.run(tool_call_conn.handle_event({"type": "session.update", "model": "qwen3-omni"}))

        assert tool_call_conn._speaker == "aiden"

    def test_session_update_captures_instructions(self, tool_call_conn, mocker) -> None:
        self._patch_base_handle_event(mocker)
        event = {
            "type": "session.update",
            "model": "qwen3-omni",
            "instructions": "Only use tools directly relevant to what the user asked.",
        }

        asyncio.run(tool_call_conn.handle_event(event))

        assert tool_call_conn._instructions == "Only use tools directly relevant to what the user asked."

    def test_session_update_without_instructions_leaves_existing_value_untouched(self, tool_call_conn, mocker) -> None:
        self._patch_base_handle_event(mocker)
        tool_call_conn._instructions = "existing"

        asyncio.run(tool_call_conn.handle_event({"type": "session.update", "model": "qwen3-omni"}))

        assert tool_call_conn._instructions == "existing"


class TestRenderPromptPropagatesAdditionalInformation:
    """Regression test for a real bug: BaseRenderer.render_cmpl_async's
    internal pipeline (process_for_engine_async) only carries over fields
    it explicitly knows about, so `additional_information` set on the
    pre-render prompt (e.g. buffer_realtime_audio's `speaker`) was silently
    dropped and never reached the engine - the voice selection feature
    looked correct in isolation but produced no audible effect. Fixed by
    reapplying it to the rendered engine_input, mirroring how
    serving_chat.py._preprocess_chat does it for /v1/chat/completions."""

    def _conn_with_fake_renderer(self, mocker, rendered_engine_input: dict):
        conn = RealtimeConnection.__new__(RealtimeConnection)
        conn.serving = mocker.Mock()
        conn.serving.model_config.is_encoder_decoder = False
        conn.serving.renderer.render_cmpl_async = mocker.AsyncMock(return_value=[rendered_engine_input])
        return conn

    def test_additional_information_survives_render(self, mocker) -> None:
        # render_cmpl_async's fake return simulates the real pipeline: it never
        # echoes back fields it doesn't recognize, so additional_information
        # must NOT be in here even though the input prompt has it.
        rendered = {"prompt_token_ids": [1, 2, 3]}
        conn = self._conn_with_fake_renderer(mocker, rendered)
        prompt = TokensPrompt(prompt_token_ids=[1, 2, 3], additional_information={"speaker": ["aiden"]})

        result = asyncio.run(conn._render_prompt(prompt))

        assert result.prompt["additional_information"] == {"speaker": ["aiden"]}

    def test_no_additional_information_is_a_noop(self, mocker) -> None:
        rendered = {"prompt_token_ids": [1, 2, 3]}
        conn = self._conn_with_fake_renderer(mocker, rendered)
        prompt = TokensPrompt(prompt_token_ids=[1, 2, 3])

        result = asyncio.run(conn._render_prompt(prompt))

        assert "additional_information" not in result.prompt

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the /v1/realtime example client's tool-calling flags.

The client is driven against a scripted fake websocket, so the whole
declare-tools -> receive-call -> return-result -> receive-audio exchange is
covered without a server: the real thing needs 2 GPUs and 30B of weights
(tests/entrypoints/openai_api/test_qwen3_omni_realtime_websocket.py).

Two tests deliberately cross the wire: the scripted server events are built from
the server's own event models, and the client's `session.update` is parsed by the
server's own model. A fake on each side agreeing with itself proves nothing if the
two sides disagree with each other.
"""

import asyncio
import base64
import importlib.util
import json
import sys
import wave
from pathlib import Path

import pytest

from vllm_omni.entrypoints.openai.realtime_protocol import (
    FunctionCallItem,
    OmniSessionUpdate,
    ResponseFunctionCallArgumentsDelta,
    ResponseFunctionCallArgumentsDone,
    ResponseOutputItemAdded,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CLIENT_PATH = Path(__file__).resolve().parents[2] / "examples/online_serving/qwen3_omni/openai_realtime_client.py"

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather in a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_ARGUMENTS = '{"city": "Berlin"}'


def _load_client_module():
    spec = importlib.util.spec_from_file_location("qwen3_omni_openai_realtime_client_test", CLIENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeWebSocket:
    """Replays a scripted list of server events and records what was sent."""

    def __init__(self, events: list[dict]) -> None:
        self._events = list(events)
        self.sent: list[dict] = []

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        if not self._events:
            raise AssertionError("client asked for more events than the script has")
        return json.dumps(self._events.pop(0))

    async def __aenter__(self) -> "_FakeWebSocket":
        return self

    async def __aexit__(self, *_exc) -> None:
        return None


def _patch_connect(client, ws: _FakeWebSocket, mocker) -> None:
    mocker.patch.object(client.websockets, "connect", side_effect=lambda url, **_kwargs: ws)


def _input_wav(tmp_path: Path) -> Path:
    path = tmp_path / "input.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x01" * 1600)
    return path


def _audio_delta() -> dict:
    return {
        "type": "response.audio.delta",
        "audio": base64.b64encode(b"\x00\x01" * 240).decode("utf-8"),
        "format": "pcm16",
        "sample_rate_hz": 24000,
    }


def _tool_call_events() -> list[dict]:
    return [
        {"type": "session.created"},
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "name": "get_weather", "call_id": "call_1"},
        },
        {"type": "response.function_call_arguments.delta", "call_id": "call_1", "delta": _ARGUMENTS},
        {"type": "response.function_call_arguments.done", "call_id": "call_1", "arguments": _ARGUMENTS},
    ]


def _run_client(client, tmp_path: Path, ws: _FakeWebSocket, mocker, **kwargs) -> Path:
    _patch_connect(client, ws, mocker)
    output_wav = tmp_path / "out.wav"
    asyncio.run(
        client.run_client(
            url="ws://fake/v1/realtime",
            model="qwen3-omni",
            input_wav=_input_wav(tmp_path),
            output_wav=output_wav,
            output_text=None,
            chunk_ms=200,
            send_delay_ms=0,
            delta_dump_dir=None,
            **kwargs,
        )
    )
    return output_wav


def _sent_of_type(ws: _FakeWebSocket, event_type: str) -> list[dict]:
    return [event for event in ws.sent if event.get("type") == event_type]


def test_no_tool_flags_keeps_the_plain_session_update(tmp_path, mocker) -> None:
    """Nothing new on the wire unless the caller asks for tools."""
    client = _load_client_module()
    ws = _FakeWebSocket([{"type": "session.created"}, _audio_delta(), {"type": "response.audio.done"}])

    _run_client(client, tmp_path, ws, mocker)

    assert ws.sent[0] == {"type": "session.update", "model": "qwen3-omni"}


def test_tools_are_declared_nested_under_session(tmp_path, mocker) -> None:
    client = _load_client_module()
    ws = _FakeWebSocket([{"type": "session.created"}, _audio_delta(), {"type": "response.audio.done"}])

    _run_client(client, tmp_path, ws, mocker, tools=[_WEATHER_TOOL], tool_choice="auto")

    assert ws.sent[0] == {
        "type": "session.update",
        "model": "qwen3-omni",
        "session": {"tool_choice": "auto", "tools": [_WEATHER_TOOL]},
    }


def test_tool_choice_none_is_sent_without_tools(tmp_path, mocker) -> None:
    """`--tool-choice none` alone is a valid update, so it must not be dropped."""
    client = _load_client_module()
    ws = _FakeWebSocket([{"type": "session.created"}, _audio_delta(), {"type": "response.audio.done"}])

    _run_client(client, tmp_path, ws, mocker, tool_choice="none")

    assert ws.sent[0] == {"type": "session.update", "model": "qwen3-omni", "session": {"tool_choice": "none"}}


def test_tool_result_is_returned_and_the_turn_completes(tmp_path, mocker) -> None:
    """The result must go back from inside the receive loop: the server holds the
    turn open - no transcription.done, no response.audio.done - until every call
    it made has been answered."""
    client = _load_client_module()
    ws = _FakeWebSocket(
        _tool_call_events()
        + [
            {"type": "transcription.delta", "delta": "It is sunny in Berlin"},
            {"type": "transcription.done", "text": "It is sunny in Berlin"},
            _audio_delta(),
            {"type": "response.audio.done"},
        ]
    )

    output_wav = _run_client(
        client, tmp_path, ws, mocker, tools=[_WEATHER_TOOL], tool_output='{"temperature_c": 21}', tool_choice="auto"
    )

    assert _sent_of_type(ws, "conversation.item.create") == [
        {
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": "call_1", "output": '{"temperature_c": 21}'},
        }
    ]
    assert output_wav.exists()


def test_function_call_events_are_printed(tmp_path, mocker, capsys) -> None:
    client = _load_client_module()
    ws = _FakeWebSocket(
        _tool_call_events()
        + [{"type": "transcription.done", "text": ""}, _audio_delta(), {"type": "response.audio.done"}]
    )

    _run_client(client, tmp_path, ws, mocker, tools=[_WEATHER_TOOL], tool_output="{}")

    printed = capsys.readouterr().out
    assert "function call started: get_weather call_id=call_1" in printed
    assert f"function call: get_weather({_ARGUMENTS}) call_id=call_1" in printed
    assert "sent tool result for call_id=call_1: {}" in printed


def test_call_with_no_tool_output_stops_instead_of_hanging(tmp_path, mocker) -> None:
    """Without a result the server would never send another event, so the client
    stops there rather than blocking on recv - and says why."""
    client = _load_client_module()
    ws = _FakeWebSocket(_tool_call_events())

    output_wav = _run_client(client, tmp_path, ws, mocker, tools=[_WEATHER_TOOL])

    assert _sent_of_type(ws, "conversation.item.create") == []
    assert not output_wav.exists()


def test_server_event_models_are_what_the_client_reads(tmp_path, mocker) -> None:
    """The events are the server's own models here, not hand-written dicts: the
    scripted fake above and the server could otherwise drift apart while each
    stayed self-consistent, and the client would still pass its tests."""
    client = _load_client_module()
    ws = _FakeWebSocket(
        [
            {"type": "session.created"},
            ResponseOutputItemAdded(item=FunctionCallItem(name="get_weather", call_id="call_1")).model_dump(),
            ResponseFunctionCallArgumentsDelta(call_id="call_1", delta=_ARGUMENTS).model_dump(),
            ResponseFunctionCallArgumentsDone(call_id="call_1", arguments=_ARGUMENTS).model_dump(),
            {"type": "transcription.done", "text": ""},
            _audio_delta(),
            {"type": "response.audio.done"},
        ]
    )

    _run_client(client, tmp_path, ws, mocker, tools=[_WEATHER_TOOL], tool_output='{"temperature_c": 21}')

    # Name and call_id both came off the models, so the client's keys match them.
    assert _sent_of_type(ws, "conversation.item.create") == [
        {
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": "call_1", "output": '{"temperature_c": 21}'},
        }
    ]


def test_client_session_update_parses_on_the_server(tmp_path, mocker) -> None:
    """The other direction: the nested-`session` update the client sends is fed to
    the server's own model, so the shape the example advertises is the shape the
    endpoint accepts."""
    client = _load_client_module()
    ws = _FakeWebSocket([{"type": "session.created"}, _audio_delta(), {"type": "response.audio.done"}])

    _run_client(client, tmp_path, ws, mocker, tools=[_WEATHER_TOOL], tool_choice="required")

    update = OmniSessionUpdate.model_validate(ws.sent[0])
    assert update.tools is not None
    assert [tool.function.name for tool in update.tools] == ["get_weather"]
    assert update.tool_choice == "required"

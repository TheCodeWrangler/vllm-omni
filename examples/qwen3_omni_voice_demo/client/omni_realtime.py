"""Custom livekit-agents RealtimeModel/RealtimeSession adapter for vLLM-Omni's
`/v1/realtime` websocket, serving Qwen3-Omni-30B-A3B-Instruct.

THROWAWAY LOCAL TEST CODE — not part of services/voice-agent's production
package (that service deliberately dropped its LLM/brain in June 2026).

Why this exists: vllm-omni's `/v1/realtime` endpoint is NOT OpenAI Realtime
API-compatible. It's a narrow, vLLM-native protocol confirmed by reading
vllm_omni 0.24.0 source + its own reference client
(examples/online_serving/qwen3_omni/openai_realtime_client.py):

  client -> server: {"type": "session.update", "model": "..."}
                    {"type": "input_audio_buffer.commit", "final": false}  (starts a turn)
                    {"type": "input_audio_buffer.append", "audio": "<b64 PCM16 mono 16kHz>"}
                    {"type": "input_audio_buffer.commit", "final": true}   (closes input)
  server -> client: {"type": "session.created", ...}
                    {"type": "transcription.delta", "delta": "..."}       (assistant's own reply text,
                    {"type": "transcription.done", "text": "...", ...}     NOT a user-speech transcript —
                                                                            this is an Instruct chat model,
                                                                            not an ASR model)
                    {"type": "response.audio.delta", "audio": "<b64 PCM16>", "sample_rate_hz": 24000}
                    {"type": "response.audio.done", "has_audio": true}
                    {"type": "error", "error": "...", "code": "..."}

Real, hard protocol limitations this adapter works around or surfaces honestly:
  - No server-side VAD/turn-detection: vLLM only reacts to
    input_audio_buffer.commit. This adapter does its own simple RMS-energy VAD
    to decide when the user has stopped talking and finalize a turn.
  - No cancel/interrupt event in the protocol at all. `interrupt()` cancels our
    local reader task (stops feeding LiveKit playback) but the vLLM server will
    keep computing the abandoned turn server-side until it finishes.
  - session.update now supports an `instructions` field (system prompt),
    sent on every turn's connection - see update_instructions(). Still no
    proactive "speak first" without user audio though: `generate_reply()`
    can only finalize whatever's already buffered; with an empty buffer it
    resolves with an empty (silent) generation.
  - Reference client pattern is one turn per websocket connection, so this
    adapter opens a fresh connection per detected utterance rather than
    holding one persistent duplex session for the whole call — EXCEPT for a
    tool-calling round trip (see below), which deliberately keeps that same
    connection open rather than opening a second one.

Tool calling: the server's own tool-call event sequence (see
vllm_omni/entrypoints/openai/realtime_tool_calls.py and
realtime_connection.py on the vllm-omni side) is:

  server -> client: {"type": "response.output_item.added",
                      "item": {"type": "function_call", "name": "...", "call_id": "..."}}
                     {"type": "response.function_call_arguments.delta", "call_id": "...", "delta": "..."}
                     {"type": "response.function_call_arguments.done", "call_id": "...", "arguments": "..."}
  client -> server: {"type": "conversation.item.create",
                      "item": {"type": "function_call_output", "call_id": "...", "output": "..."}}

After emitting `.done` for every call in the turn, the server BLOCKS waiting
for a `function_call_output` per call_id on that same websocket, then
resumes generation itself (no separate "go ahead" event needed) and streams
the model's actual spoken reply as normal transcription/audio events. This
means `auto_tool_reply_generation=True` here — the underlying model, not
livekit-agents, decides when to continue — and this adapter's job is only to
(a) surface FunctionCall items on function_stream and (b) relay the executed
tool's output back over the *same still-open* connection once
update_chat_ctx() delivers it.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from collections.abc import AsyncIterable, AsyncIterator

import numpy as np
import websockets
from livekit import rtc
from livekit.agents.llm import (
    ChatContext,
    FunctionCall,
    FunctionCallOutput,
    FunctionTool,
    RawFunctionTool,
    Tool,
)
from livekit.agents.llm.realtime import (
    GenerationCreatedEvent,
    InputSpeechStartedEvent,
    InputSpeechStoppedEvent,
    MessageGeneration,
    NotGivenOr,
    RealtimeCapabilities,
    RealtimeModel,
    RealtimeSession,
)
from livekit.agents.llm.utils import build_legacy_openai_schema
from livekit.agents.types import NOT_GIVEN
from livekit.agents.llm import ToolChoice

logger = logging.getLogger("omni_realtime")

_TARGET_SAMPLE_RATE = 16000
_SEND_CHUNK_MS = 200
_VAD_RMS_THRESHOLD = 500.0  # int16 scale; tune per mic/room if VAD feels wrong
_MIN_SPEECH_MS = 200
_END_SILENCE_MS = 700
# Cap on tool-call round trips within a single user utterance. The server keeps
# regenerating after each function_call_output, so a checkpoint that always answers
# with another tool call would otherwise loop forever, burning GPU and never speaking.
_MAX_TOOL_ROUNDS = 4
# How many prior turns to replay. Measured against this checkpoint with tools +
# instructions in the prompt: 20 turns is ~1.9k tokens (5.6% of the 32k batch limit,
# ~3% of the 65k context), so context is not the binding constraint - this mainly
# bounds the per-turn re-encode cost, since prefix caching is off for this deploy and
# every history clip is re-encoded on every turn.
_MAX_HISTORY_TURNS = int(os.environ.get("OMNI_MAX_HISTORY_TURNS", "20"))
# When set, each turn writes its exact input (session.update payload, history summary, PCM)
# and the raw text the model generated, so a failing live turn can be replayed verbatim
# offline instead of guessed at. Also switches on the server-side prompt logging.
_DEBUG_DIR = os.environ.get("OMNI_DEBUG_DUMP_DIR") or None


def _tools_to_realtime_schema(tools: list[Tool]) -> list[dict]:
    schemas: list[dict] = []
    for tool in tools:
        if isinstance(tool, RawFunctionTool):
            schemas.append({"type": "function", "function": tool.info.raw_schema})
        elif isinstance(tool, FunctionTool):
            schemas.append(build_legacy_openai_schema(tool))
    return schemas


def _pcm16_frame(pcm_bytes: bytes, sample_rate: int) -> rtc.AudioFrame:
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    frame = rtc.AudioFrame.create(sample_rate=sample_rate, num_channels=1, samples_per_channel=len(samples))
    np.frombuffer(frame.data, dtype=np.int16)[:] = samples
    return frame


def _make_generation_event(
    message_id: str,
    text_queue: asyncio.Queue[str | None],
    audio_queue: asyncio.Queue[rtc.AudioFrame | None],
    function_queue: asyncio.Queue[FunctionCall | None],
    *,
    user_initiated: bool,
) -> GenerationCreatedEvent:
    """One GenerationCreatedEvent = one closed phase of output: either the
    model's spoken/text reply, or a batch of tool calls (never both — the
    server suppresses audio/text for a tool-call turn). livekit-agents'
    function_stream consumer (see voice/generation.py) blocks on `async for
    fnc_call in function_stream` until the stream *closes* before it will
    execute any tool, so every phase's queues must reach their None sentinel
    once that phase's content is fully known, even mid-round-trip."""

    async def text_stream() -> AsyncIterator[str]:
        while True:
            item = await text_queue.get()
            if item is None:
                return
            yield item

    async def audio_stream() -> AsyncIterator[rtc.AudioFrame]:
        while True:
            item = await audio_queue.get()
            if item is None:
                return
            yield item

    async def function_stream() -> AsyncIterator[FunctionCall]:
        while True:
            item = await function_queue.get()
            if item is None:
                return
            yield item

    modalities_future: asyncio.Future[list[str]] = asyncio.get_event_loop().create_future()
    modalities_future.set_result(["text", "audio"])

    async def message_stream() -> AsyncIterator[MessageGeneration]:
        yield MessageGeneration(
            message_id=message_id,
            text_stream=text_stream(),
            audio_stream=audio_stream(),
            modalities=modalities_future,
        )

    return GenerationCreatedEvent(
        message_stream=message_stream(),
        function_stream=function_stream(),
        user_initiated=user_initiated,
        response_id=message_id,
    )


class OmniRealtimeModel(RealtimeModel):
    """RealtimeModel backed by a self-hosted vllm-omni /v1/realtime server."""

    def __init__(self, *, ws_url: str, model: str, voice: str | None = None) -> None:
        super().__init__(
            capabilities=RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,  # this adapter does its own VAD; AgentSession shouldn't run its own
                user_transcription=False,  # vLLM only gives us the assistant's own reply text, not ASR of the user
                # The server auto-continues generation itself once it receives a
                # function_call_output over the websocket — no separate "go ahead" call
                # needed from livekit-agents. See module docstring's tool-calling section.
                auto_tool_reply_generation=True,
                audio_output=True,
                manual_function_calls=False,  # no concept of resuming a call from chat history; each turn is live
                supports_say=False,
            )
        )
        self._ws_url = ws_url
        self._model = model
        self._voice = voice

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "vllm-omni"

    def session(self) -> RealtimeSession:
        return OmniRealtimeSession(self)

    async def aclose(self) -> None:
        pass


class OmniRealtimeSession(RealtimeSession):
    def __init__(self, realtime_model: OmniRealtimeModel) -> None:
        super().__init__(realtime_model)
        self._ws_url = realtime_model._ws_url  # noqa: SLF001 - same module, internal wiring
        self._model = realtime_model._model  # noqa: SLF001
        self._voice = realtime_model._voice  # noqa: SLF001

        self._chat_ctx = ChatContext.empty()
        self._tools: list[Tool] = []
        self._instructions: str | None = None

        self._resampler: rtc.AudioResampler | None = None
        self._resampler_input_rate: int | None = None
        self._logged_first_frame = False
        self._vad_log_count = 0

        self._pcm_chunks: list[bytes] = []
        self._preroll: list[bytes] = []
        self._speaking = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0

        self._current_task: asyncio.Task | None = None

        # Tool-calling round trip: the connection for the in-flight turn (if any),
        # kept open across a function_call_output relay instead of closed/reopened;
        # call_ids already relayed so update_chat_ctx() doesn't resend stale history
        # on a later turn; and a per-call_id Event that update_chat_ctx() sets once
        # it has sent that call's output, so _run_turn's relay-wait can proceed
        # regardless of which side (send vs. wait) happens first.
        # Prior conversation as an ordered message list, replayed on every later turn.
        # User turns carry the original PCM (the endpoint gives us no transcript of
        # the user's speech); everything else carries text. A completed TOOL turn is
        # replayed in full - assistant `<tool_call>`, the tool result, then the spoken
        # answer - because replaying only the answer teaches the model that these
        # questions are answered from knowledge, and it stops calling tools and
        # confabulates instead (verified on the reference HF path too).
        self._history: list[dict] = []
        # Messages accumulated for the turn currently in flight.
        self._turn_messages: list[dict] = []
        self._active_ws: websockets.ClientConnection | None = None
        self._relayed_call_ids: set[str] = set()
        self._call_relayed_events: dict[str, asyncio.Event] = {}

    def _event_for_call(self, call_id: str) -> asyncio.Event:
        event = self._call_relayed_events.get(call_id)
        if event is None:
            event = asyncio.Event()
            self._call_relayed_events[call_id] = event
        return event

    # -- ABC-required properties -------------------------------------------------

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> list[Tool]:
        return self._tools

    # -- audio in / VAD -----------------------------------------------------------

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass  # vllm-omni's /v1/realtime endpoint used here is audio-only; no video support

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if not self._logged_first_frame:
            self._logged_first_frame = True
            logger.info(
                "push_audio: first frame received (sample_rate=%s, channels=%s, samples=%s)",
                frame.sample_rate,
                frame.num_channels,
                frame.samples_per_channel,
            )
        if self._resampler is None or self._resampler_input_rate != frame.sample_rate:
            self._resampler = rtc.AudioResampler(frame.sample_rate, _TARGET_SAMPLE_RATE, num_channels=1)
            self._resampler_input_rate = frame.sample_rate

        for resampled in self._resampler.push(frame):
            self._process_resampled(resampled)

    def _process_resampled(self, frame: rtc.AudioFrame) -> None:
        samples = np.frombuffer(frame.data, dtype=np.int16)
        if samples.size == 0:
            return
        frame_ms = samples.size / _TARGET_SAMPLE_RATE * 1000.0
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        is_speech = rms > _VAD_RMS_THRESHOLD

        self._vad_log_count += 1
        if self._vad_log_count % 25 == 0:  # ~ every couple seconds, not every 10-20ms frame
            logger.info("vad: rms=%.1f threshold=%.1f speaking=%s", rms, _VAD_RMS_THRESHOLD, self._speaking)

        if self._speaking:
            # Committed to this utterance: keep everything, including brief in-utterance
            # pauses, until end-of-speech silence closes it out.
            self._pcm_chunks.append(samples.tobytes())
        else:
            # Not yet committed: only keep a short rolling pre-roll so we don't clip the
            # first syllable, instead of buffering unbounded silence from session start.
            self._preroll.append(samples.tobytes())
            max_preroll_frames = max(int(_MIN_SPEECH_MS / frame_ms), 1) if frame_ms > 0 else 1
            if len(self._preroll) > max_preroll_frames:
                self._preroll = self._preroll[-max_preroll_frames:]

        if is_speech:
            if not self._speaking and self._speech_ms + frame_ms >= _MIN_SPEECH_MS:
                self._speaking = True
                logger.info("VAD: speech started (rms=%.1f)", rms)
                self._pcm_chunks = self._preroll + self._pcm_chunks
                self._preroll = []
                self.emit("input_speech_started", InputSpeechStartedEvent())
            self._speech_ms += frame_ms
            self._silence_ms = 0.0
        elif self._speaking:
            self._silence_ms += frame_ms
            if self._silence_ms >= _END_SILENCE_MS:
                self._finalize_turn_from_vad()
        else:
            self._speech_ms = 0.0  # require continuous qualifying speech, not accumulated blips

    def _reset_input_state(self) -> None:
        self._pcm_chunks = []
        self._preroll = []
        self._speaking = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0

    def _finalize_turn_from_vad(self) -> None:
        if self._current_task is not None and not self._current_task.done():
            logger.info("VAD: end-of-speech detected but a turn is already in flight; dropping this utterance")
            self._reset_input_state()
            return
        pcm = b"".join(self._pcm_chunks)
        logger.info("VAD: speech stopped, finalizing turn (%.2fs of audio)", len(pcm) / 2 / _TARGET_SAMPLE_RATE)
        self._reset_input_state()
        self.emit("input_speech_stopped", InputSpeechStoppedEvent(user_transcription_enabled=False))
        if not pcm:
            return
        event = self._begin_turn(pcm, user_initiated=False)
        self.emit("generation_created", event)

    # -- turn execution -------------------------------------------------------------

    def _begin_turn(self, pcm16_bytes: bytes, *, user_initiated: bool) -> GenerationCreatedEvent:
        message_id = f"omni-{uuid.uuid4().hex[:12]}"
        text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        audio_queue: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
        function_queue: asyncio.Queue[FunctionCall | None] = asyncio.Queue()
        event = _make_generation_event(
            message_id, text_queue, audio_queue, function_queue, user_initiated=user_initiated
        )

        self._current_task = asyncio.create_task(
            self._run_turn(pcm16_bytes, message_id, text_queue, audio_queue, function_queue),
            name=f"omni-turn-{message_id}",
        )
        return event

    async def _run_turn(
        self,
        pcm16_bytes: bytes,
        message_id: str,
        text_queue: asyncio.Queue[str | None],
        audio_queue: asyncio.Queue[rtc.AudioFrame | None],
        function_queue: asyncio.Queue[FunctionCall | None],
    ) -> None:
        """Drive one detected utterance to completion, which may span several
        *phases* over the same websocket connection: an initial phase (plain
        reply, or a batch of tool calls), then — if the model asked for
        tools — one more phase per round trip once update_chat_ctx() relays
        each tool's output back to the server. Each phase gets its own fresh
        queues/GenerationCreatedEvent (see _make_generation_event's docstring
        for why: livekit-agents won't execute a tool until function_stream
        closes, so a single stream can't span the whole round trip)."""
        self._turn_messages = []
        logger.info("omni turn: connecting to %s (model=%s)", self._ws_url, self._model)
        try:
            async with websockets.connect(self._ws_url, max_size=64 * 1024 * 1024) as ws:
                self._active_ws = ws
                # Hardcoded: the model's default speaker (whichever key HF's
                # talker_config.speaker_id lists first - "chelsie" for this
                # checkpoint) is not the desired demo voice.
                update_msg: dict = {"type": "session.update", "model": self._model}
                if self._voice:
                    # Server reads `voice` (or `speaker`) off session.update and threads it to
                    # the talker's speaker-token selection. Omitting it uses the checkpoint's
                    # default, which is just the FIRST key of talker_config.speaker_id
                    # (chelsie here) — not necessarily the one you want, so be explicit.
                    update_msg["voice"] = self._voice
                    logger.info("omni turn: requesting voice=%s", self._voice)
                if self._instructions:
                    update_msg["instructions"] = self._instructions
                tools_payload = _tools_to_realtime_schema(self._tools)
                if tools_payload:
                    update_msg["tools"] = tools_payload
                    logger.info("omni turn: sending %d tool definition(s) in session.update", len(tools_payload))
                await ws.send(json.dumps(update_msg))

                # Replay prior messages so the model can refer back to them, and so a
                # past tool turn still reads as "this needed a tool call".
                replay = self._trim_history()
                for past in replay:
                    if past["role"] == "user":
                        content = [{"type": "input_audio", "audio": base64.b64encode(past["pcm"]).decode("utf-8")}]
                    else:
                        content = [{"type": "text", "text": past["text"]}]
                    await ws.send(
                        json.dumps(
                            {
                                "type": "conversation.item.create",
                                "item": {"type": "message", "role": past["role"], "content": content},
                            }
                        )
                    )
                if replay:
                    logger.info(
                        "omni turn: replayed %d history message(s) (%d prior turn(s))",
                        len(replay),
                        sum(1 for m in replay if m["role"] == "user"),
                    )

                if _DEBUG_DIR:
                    self._dump_turn_input(update_msg, replay, pcm16_bytes)

                await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))

                chunk_bytes = max(_TARGET_SAMPLE_RATE * 2 * _SEND_CHUNK_MS // 1000, 2)
                for i in range(0, len(pcm16_bytes), chunk_bytes):
                    chunk = pcm16_bytes[i : i + chunk_bytes]
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(chunk).decode("utf-8"),
                            }
                        )
                    )
                await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

                reply_parts: list[str] = []
                for round_num in range(1, _MAX_TOOL_ROUNDS + 3):  # one iteration per phase
                    tool_call_ids = await self._run_phase(
                        ws, text_queue, audio_queue, function_queue, reply_sink=reply_parts
                    )
                    if tool_call_ids is None:
                        self._record_turn(pcm16_bytes, reply_parts)
                        return  # phase ended with real content; whole turn is done
                    if not tool_call_ids:
                        return  # error mid-graph; already logged, queues already closed

                    if round_num > _MAX_TOOL_ROUNDS + 1:
                        # Even the wrap-up directive below did not stop it. Give up rather
                        # than let a turn run indefinitely.
                        logger.error(
                            "omni turn: still requesting tools after the wrap-up directive "
                            "(call_ids=%s) — abandoning turn",
                            tool_call_ids,
                        )
                        return

                    if round_num > _MAX_TOOL_ROUNDS:
                        # Budget spent. A broad question ("weather in Spain") legitimately
                        # fans out over several cities, so the fan-out is not the problem -
                        # running out of rounds mid-gather and saying NOTHING is. The tool
                        # result is just text the model reads, so use it to ask for the
                        # summary instead of abandoning the turn in silence.
                        logger.info(
                            "omni turn: tool budget spent after %d round(s); asking for a "
                            "wrap-up instead of abandoning (call_ids=%s)",
                            _MAX_TOOL_ROUNDS,
                            tool_call_ids,
                        )
                        for call_id in tool_call_ids:
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "function_call_output",
                                            "call_id": call_id,
                                            "output": (
                                                "No further tool calls are available. Answer the user "
                                                "now, in one or two short spoken sentences, using only "
                                                "the information already gathered. Do not call any more "
                                                "tools."
                                            ),
                                        },
                                    }
                                )
                            )
                    else:
                        await self._wait_for_tool_relay(tool_call_ids)

                    message_id = f"omni-{uuid.uuid4().hex[:12]}"
                    text_queue = asyncio.Queue()
                    audio_queue = asyncio.Queue()
                    function_queue = asyncio.Queue()
                    logger.info("%s: continuing after tool result(s), starting next phase", message_id)
                    self.emit(
                        "generation_created",
                        _make_generation_event(
                            message_id, text_queue, audio_queue, function_queue, user_initiated=False
                        ),
                    )
        except Exception:
            logger.exception("omni realtime turn failed")
            text_queue.put_nowait(None)
            audio_queue.put_nowait(None)
            function_queue.put_nowait(None)
        finally:
            self._active_ws = None

    def _dump_turn_input(self, update_msg: dict, replay: list[dict], pcm16_bytes: bytes) -> None:
        """Write everything needed to replay this turn offline: the session.update payload
        verbatim, a readable history summary, and the PCM actually sent."""
        try:
            import wave

            stamp = f"{_DEBUG_DIR}/turn-{uuid.uuid4().hex[:8]}"
            payload = {
                "session_update": update_msg,
                "history": [
                    {"role": m["role"], **({"pcm_bytes": len(m["pcm"])} if m["role"] == "user" else {"text": m["text"]})}
                    for m in replay
                ],
            }
            with open(f"{stamp}.json", "w") as f:
                json.dump(payload, f, indent=1)
            with wave.open(f"{stamp}.wav", "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(_TARGET_SAMPLE_RATE)
                w.writeframes(pcm16_bytes)
            self._debug_stamp = stamp
            logger.info("OMNIDBG turn input -> %s.{json,wav}  (%d history msg(s))", stamp, len(replay))
        except Exception:
            logger.exception("OMNIDBG: failed to dump turn input")

    def _record_turn(self, pcm16_bytes: bytes, reply_parts: list[str]) -> None:
        """Commit a completed turn to history: the user's audio, then everything the
        model produced for it - any `<tool_call>`, each tool result, and the spoken
        answer. Turns that produced no answer (abandoned mid tool-loop) are dropped
        rather than left as a dangling half-turn."""
        text = "".join(reply_parts).strip()
        turn = list(self._turn_messages)
        self._turn_messages = []
        if not text:
            return
        turn.append({"role": "assistant", "text": text})
        if _DEBUG_DIR:
            # The text the THINKER wrote, before the talker speaks it and before any ASR
            # round-trip folds words back into glyphs. This is the only faithful view of
            # what the model actually produced.
            logger.info("OMNIDBG generated text (%d chars): %r", len(text), text)
            stamp = getattr(self, "_debug_stamp", None)
            if stamp:
                try:
                    with open(f"{stamp}.generated.txt", "w") as f:
                        f.write(text)
                except Exception:
                    logger.exception("OMNIDBG: failed to write generated text")
        self._history.append({"role": "user", "pcm": pcm16_bytes})
        self._history.extend(turn)
        logger.info(
            "omni turn: recorded turn in history (%d message(s), %d prior turn(s) total)",
            len(turn) + 1,
            sum(1 for m in self._history if m["role"] == "user"),
        )

    def _trim_history(self) -> list[dict]:
        """The last `_MAX_HISTORY_TURNS` user turns and everything that followed them.
        Trimming on message count would risk cutting between a `<tool_call>` and its
        result, which reads as an unanswered call and makes the model retry it."""
        if _MAX_HISTORY_TURNS <= 0:
            return []
        user_idx = [i for i, m in enumerate(self._history) if m["role"] == "user"]
        if len(user_idx) <= _MAX_HISTORY_TURNS:
            return list(self._history)
        return self._history[user_idx[-_MAX_HISTORY_TURNS] :]


    async def _run_phase(
        self,
        ws: websockets.ClientConnection,
        text_queue: asyncio.Queue[str | None],
        audio_queue: asyncio.Queue[rtc.AudioFrame | None],
        function_queue: asyncio.Queue[FunctionCall | None],
        reply_sink: list[str] | None = None,
    ) -> set[str] | None:
        """Read events for one phase. Returns None if the phase ended with
        real content (whole turn done), or the set of call_ids requested if
        it ended with tool calls instead (caller relays results and starts
        the next phase), or an empty set on a protocol error (already
        logged; queues already closed, whole turn done)."""
        pending_call_names: dict[str, str] = {}
        requested_call_ids: set[str] = set()
        reply_parts: list[str] = reply_sink if reply_sink is not None else []
        try:
            while True:
                raw = await ws.recv()
                if isinstance(raw, (bytes, bytearray)):
                    continue
                event = json.loads(raw)
                etype = event.get("type")

                if etype == "transcription.delta":
                    delta = event.get("delta", "")
                    if delta:
                        reply_parts.append(delta)
                        text_queue.put_nowait(delta)
                elif etype == "response.audio.delta":
                    sr = event.get("sample_rate_hz") or 24000
                    audio_b64 = event.get("audio", "")
                    if audio_b64:
                        audio_queue.put_nowait(_pcm16_frame(base64.b64decode(audio_b64), int(sr)))
                elif etype == "response.audio.done":
                    text_queue.put_nowait(None)
                    audio_queue.put_nowait(None)
                    function_queue.put_nowait(None)
                    return None
                elif etype == "transcription.done":
                    continue
                elif etype == "response.output_item.added":
                    item = event.get("item") or {}
                    if item.get("type") == "function_call":
                        call_id, name = item.get("call_id"), item.get("name")
                        if call_id and name:
                            pending_call_names[call_id] = name
                elif etype == "response.function_call_arguments.delta":
                    continue  # only the aggregated .done payload is needed
                elif etype == "response.function_call_arguments.done":
                    call_id = event.get("call_id")
                    name = pending_call_names.get(call_id)
                    if not call_id or name is None:
                        logger.warning("omni turn: function_call_arguments.done for unknown call_id=%s", call_id)
                        continue
                    arguments = event.get("arguments") or "{}"
                    logger.info(
                        "omni turn: tool call requested name=%s call_id=%s arguments=%s", name, call_id, arguments
                    )
                    function_queue.put_nowait(FunctionCall(call_id=call_id, name=name, arguments=arguments))
                    # Keep the call in history exactly as the model emitted it, so a
                    # replay of this turn still shows that a tool was needed.
                    self._turn_messages.append(
                        {
                            "role": "assistant",
                            "text": '<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>' % (name, arguments),
                        }
                    )
                    requested_call_ids.add(call_id)
                    if len(requested_call_ids) >= len(pending_call_names):
                        # Every call announced via output_item.added has now had its
                        # .done event — close this phase so livekit-agents' function_stream
                        # consumer will actually start executing them (see
                        # _make_generation_event's docstring).
                        logger.info(
                            "omni turn: phase closing after %d tool call(s) requested_call_ids=%s",
                            len(requested_call_ids),
                            requested_call_ids,
                        )
                        text_queue.put_nowait(None)
                        audio_queue.put_nowait(None)
                        function_queue.put_nowait(None)
                        return requested_call_ids
                elif etype == "error":
                    logger.error("vllm-omni realtime error: %s", event)
                    text_queue.put_nowait(None)
                    audio_queue.put_nowait(None)
                    function_queue.put_nowait(None)
                    return set()
        except Exception:
            logger.exception("omni realtime phase failed")
            text_queue.put_nowait(None)
            audio_queue.put_nowait(None)
            function_queue.put_nowait(None)
            return set()

    async def _wait_for_tool_relay(self, call_ids: set[str]) -> None:
        for call_id in call_ids:
            await self._event_for_call(call_id).wait()
        for call_id in call_ids:
            self._call_relayed_events.pop(call_id, None)

    # -- RealtimeSession ABC -----------------------------------------------------

    def commit_audio(self) -> None:
        if self._current_task is not None and not self._current_task.done():
            return
        pcm = b"".join(self._pcm_chunks)
        self._reset_input_state()
        if not pcm:
            return
        event = self._begin_turn(pcm, user_initiated=False)
        self.emit("generation_created", event)

    def clear_audio(self) -> None:
        self._reset_input_state()

    def interrupt(self) -> None:
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()
            logger.info("interrupt(): cancelled local turn reader (vLLM keeps computing server-side; no cancel event in the protocol)")

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        fut: asyncio.Future[GenerationCreatedEvent] = asyncio.get_event_loop().create_future()
        pcm = b"".join(self._pcm_chunks)
        self._reset_input_state()
        if pcm:
            event = self._begin_turn(pcm, user_initiated=True)
        else:
            logger.warning(
                "generate_reply() called with no buffered user audio; the omni backend "
                "can only respond to audio it received, it can't originate speech from text "
                "instructions alone — resolving with an empty (silent) generation"
            )
            message_id = f"omni-empty-{uuid.uuid4().hex[:8]}"

            async def _empty_messages() -> AsyncIterator[MessageGeneration]:
                return
                yield  # pragma: no cover

            async def _empty_functions() -> AsyncIterator:
                return
                yield  # pragma: no cover

            event = GenerationCreatedEvent(
                message_stream=_empty_messages(),
                function_stream=_empty_functions(),
                user_initiated=True,
                response_id=message_id,
            )
        self.emit("generation_created", event)
        fut.set_result(event)
        return fut

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list,
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        logger.debug("truncate() is a no-op: vllm-omni's realtime protocol has no message-truncation support")

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        # AgentSession calls this right after appending the executed tool's
        # FunctionCallOutput(s) to chat context — relay each one to the server as
        # conversation.item.create so it can resume the blocked generation. Only new,
        # not-yet-relayed outputs are sent (this session's chat_ctx can be updated for
        # other reasons too, e.g. after our own turns complete).
        new_outputs = [
            item
            for item in chat_ctx.items
            if isinstance(item, FunctionCallOutput) and item.call_id not in self._relayed_call_ids
        ]
        logger.info(
            "update_chat_ctx: called with %d chat_ctx item(s), %d new FunctionCallOutput(s) (already-relayed=%s)",
            len(chat_ctx.items),
            len(new_outputs),
            self._relayed_call_ids,
        )
        self._chat_ctx = chat_ctx
        if not new_outputs:
            return
        ws = self._active_ws
        if ws is None:
            logger.warning(
                "update_chat_ctx: got %d tool result(s) but no active omni connection to relay them on "
                "(call_ids=%s) — the server-side turn has likely already ended/timed out",
                len(new_outputs),
                [o.call_id for o in new_outputs],
            )
            return
        for output in new_outputs:
            self._relayed_call_ids.add(output.call_id)
            logger.info("update_chat_ctx: relaying tool result for call_id=%s", output.call_id)
            self._turn_messages.append({"role": "tool", "text": output.output})
            try:
                await ws.send(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": output.call_id,
                                "output": output.output,
                            },
                        }
                    )
                )
            except Exception:
                logger.exception("update_chat_ctx: failed to relay tool result for call_id=%s", output.call_id)
            finally:
                # Signal _wait_for_tool_relay() regardless of send success/failure -
                # a stuck wait forever is worse than moving on to let the turn error out.
                self._event_for_call(output.call_id).set()

    async def update_instructions(self, instructions: str) -> None:
        self._instructions = instructions

    async def update_tools(self, tools: list[Tool]) -> None:
        self._tools = list(tools)

    def update_options(self, *, tool_choice: NotGivenOr = NOT_GIVEN) -> None:
        pass

    async def aclose(self) -> None:
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()

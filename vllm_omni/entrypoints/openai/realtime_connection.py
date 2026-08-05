from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

import numpy as np
from pydantic import ValidationError
from vllm.engine.protocol import StreamingInput
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.entrypoints.speech_to_text.realtime.protocol import TranscriptionDelta, TranscriptionDone
from vllm.inputs import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.renderers.hf import safe_apply_chat_template
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime_protocol import (
    FunctionCallItem,
    FunctionCallOutputItem,
    OmniNamedToolChoice,
    OmniSessionUpdate,
    ResponseFunctionCallArgumentsDelta,
    ResponseFunctionCallArgumentsDone,
    ResponseOutputItemAdded,
    first_error_message,
    lift_session_fields,
)
from vllm_omni.entrypoints.openai.realtime_tool_calls import ToolCallDelta, ToolCallStreamState, extract_deltas
from vllm_omni.entrypoints.utils import coerce_param_message_types

logger = init_logger(__name__)

# How long to block on a client tool result before re-checking the connection.
# Only bounds the wait between liveness checks, not the total wait: a tool may
# legitimately take a long time, but a client that has gone away must not leave
# the generation task parked forever.
_TOOL_RESULT_POLL_S = 0.5


@dataclass
class _PendingToolCall:
    """A tool call being accumulated from the model's streamed output."""

    call_id: str
    name: str
    arguments: str = ""


class RealtimeConnection(VllmRealtimeConnection):
    """Omni realtime connection with audio-only server events, plus
    OpenAI-Realtime-shaped tool/function calling.

    Reuses upstream vLLM websocket/session lifecycle and customizes
    generation output handling to emit audio deltas and tool-call events.

    Tool-calling protocol (mirrors OpenAI's Realtime API event shapes):
      - client -> server: `session.update` gains optional `tools` (a list of
        function definitions) and `tool_choice` (`"none"`/`"auto"`/`"required"`,
        or a function named as `{"type": "function", "name": ...}`) fields,
        nested under a `session` object as the OpenAI Realtime API puts them or
        flat on the event - see realtime_protocol.py, which owns every accepted
        shape and validates them.
      - server -> client, once the model starts a tool call:
        `response.output_item.added` (item.type="function_call", name, call_id)
        `response.function_call_arguments.delta` (call_id, delta)
        `response.function_call_arguments.done` (call_id, arguments)
      - client -> server, once the tool has run:
        `conversation.item.create` with
        `item = {"type": "function_call_output", "call_id": ..., "output": "..."}`
      - generation then continues automatically with the tool result appended,
        streaming the model's actual spoken reply as normal.

    Audio for a tool-call turn is not forwarded to the client once the
    `<tool_call>` tag has been parsed - the underlying 3-stage pipeline
    (thinker->talker->code2wav) still synthesizes it (there is no clean
    lower-level hook to skip talker/code2wav without changing the shared
    orchestrator - see PR description), the bytes are just dropped. Any audio
    that arrived before the tag was recognized has already been sent; in
    practice the tag appears in the thinker's text well ahead of the
    corresponding synthesized audio.

    A chain of tool calls is bounded by MAX_TOOL_ROUNDS.

    Scope and limitations of the tool-calling path:

    - **Non-duplex only.** This is the half-duplex `/v1/realtime` path. The
      full-duplex runtime under `experimental/fullduplex/` has no tool-calling
      support and shares no code with this.
    - **Requires `async_chunk` disabled.** A `session.update` that DECLARES tools
      is rejected when the server runs in async-chunk mode (`tools: []` still
      clears them, which that mode can do); see `_parse_session_tools` for why
      aggregating the audio instead would not be enough.
    - **`tool_choice` is recorded, not enforced.** `"none"` genuinely disables
      tool calling for the session - no `<tools>` preamble in the prompt and no
      scanning of the generated text (see `_active_tools`) - and so does an
      empty `tools` list, which is how a client clears its tools. `"required"`
      and a named function both behave like `"auto"` (the default): forcing any
      call, let alone a specific one, needs guided decoding, whereas this path
      drives the engine with plain sampling params and parses `<tool_call>` out
      of the text afterwards. Both are accepted rather than refused so a client
      written against the OpenAI API keeps working; a named function is treated
      as `"required"` and the name it asked for is not kept, since nothing here
      could act on it. The downgrade is logged once per session
      (`_warn_tool_choice_not_enforced`) rather than pretended to.
    - **Waits on client liveness.** Once the model has requested a tool, the turn
      blocks until a `function_call_output` arrives for every pending call. There
      is deliberately no deadline, because a slow tool is indistinguishable from
      an absent one; a client that disconnects releases the wait, and malformed
      or unknown results are reported back rather than silently dropped.
    """

    # Upper bound on consecutive tool-call rounds within one user turn.
    MAX_TOOL_ROUNDS = 8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = cast(AsyncOmni, self.serving.engine_client)
        self._realtime_audio_ref: np.ndarray | None = None
        self._tools: list[dict[str, Any]] | None = None
        # `"none"`/`"auto"`/`"required"`, per session, defaulting to `"auto"` as
        # the OpenAI API does; a named function is recorded as `"required"`. Only
        # `"none"` changes behavior - see _active_tools and the class docstring's
        # scope notes.
        self._tool_choice: str = "auto"
        # Whether this session has already been told that its `tool_choice`
        # cannot be enforced - see _warn_tool_choice_not_enforced.
        self._tool_choice_not_enforced_warned = False
        # The tools this turn runs with, latched by start_generation so that a
        # `session.update` arriving mid-turn cannot desynchronize the extractor
        # from the prompt the model is still answering - see _active_tools.
        self._turn_tools: list[dict[str, Any]] | None = None
        # The current turn's prompt as handed to the engine BEFORE multimodal
        # expansion: un-expanded `prompt_token_ids` (one `<|audio_pad|>`) plus the
        # audio in `multi_modal_data`. Tool-call continuations rebuild from this
        # rather than from the engine's post-expansion `output.prompt_token_ids`,
        # because those contain the expanded audio placeholder run with no way to
        # re-attach the audio - see _await_tool_results_and_continue.
        self._turn_prompt: dict[str, Any] | None = None
        # parser-assigned index (per generation) -> the call being accumulated
        self._pending_tool_calls: dict[int, _PendingToolCall] = {}
        self._tool_result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        # Consecutive tool-call rounds in this turn, bounded by MAX_TOOL_ROUNDS.
        self._tool_rounds = 0

    async def handle_event(self, event: dict):
        event_type = event.get("type")
        if event_type == "session.update":
            await self._handle_session_update(event)
        elif event_type == "conversation.item.create":
            item = event.get("item") or {}
            if item.get("type") == "function_call_output":
                await self._enqueue_tool_result(item)
            else:
                await self.send_error(f"Unsupported conversation.item type: {item.get('type')!r}", "unsupported_item")
        else:
            await super().handle_event(event)

    async def _handle_session_update(self, event: dict) -> None:
        """Route a `session.update` past upstream, then apply its tool fields.

        Parsed before `super().handle_event`, applied after it: upstream owns
        `model`, and tool state must not outlive an event whose session was never
        validated. A refused `{"tool_choice": "none"}` used to disable tool
        calling permanently, since no later update that omits `tool_choice` can
        undo it.

        Whether the session is validated is upstream's own
        `_is_model_validated`, which it sets when it takes a `session.update` and
        never clears - so tool state sticks only once a model this server serves
        has been accepted, and after that the session keeps taking tool updates
        that omit `model`, which is what clients send (upstream still answers
        those with its "Missing required field: model" error - its call, and the
        same as on the base branch, where the tools were applied too). The flag
        being sticky is also its limit: once a session is validated, an update
        naming a model upstream refuses still has its tool fields applied. Telling
        those apart would mean re-deciding `model` here, which is the one thing
        this method exists not to do.

        A session that has not been validated yet gets one error per event:
        upstream's refusal is reported by upstream, and a second complaint about
        the tool fields of an event that has to be resent anyway would only add
        noise.
        """
        update, refusal = self._parse_session_tools(event)
        await super().handle_event(self._with_top_level_model(event))
        # Fail open: if a future upstream drops the attribute, tool updates keep
        # working rather than every `session.update` raising in here.
        if not getattr(self, "_is_model_validated", True):
            return  # upstream refused the session and said why; nothing of ours applies
        if refusal is not None:
            await self.send_error(*refusal)
        if update is not None:
            self._apply_session_tools(update)

    @staticmethod
    def _with_top_level_model(event: dict) -> dict:
        """The event as upstream reads it, with `model` at the top level.

        The canonical OpenAI Realtime `session.update` carries `model` inside the
        `session` object, but upstream's `handle_event` reads it off the top
        level - so a spec-shaped event was refused as "Missing required field:
        model" and took its `tools` down with it, which made this endpoint's
        nested-`session` support unreachable for the clients it exists for.

        Lifted into a shallow copy, never into the caller's dict, which the tests
        assert stays as it arrived. A `model` already at the top level always
        wins: that field is upstream's, and this only fills it in.
        """
        session = event.get("session")
        if event.get("model") is not None or not isinstance(session, dict):
            return event
        model = session.get("model")
        if model is None:
            return event
        return {**event, "model": model}

    def _parse_session_tools(self, event: dict) -> tuple[OmniSessionUpdate | None, tuple[str, str] | None]:
        """Validate the tool fields of a `session.update` without applying them.

        Returns the parsed event and the (message, code) to report, if any; the
        caller applies neither until upstream has validated the session itself.
        Shape errors are protocol errors, so they are caught here rather than
        discovered later inside the chat template.
        """
        lifted = lift_session_fields(event)
        try:
            update = OmniSessionUpdate.model_validate(lifted)
        except ValidationError as e:
            # Described against `lifted`, the mapping pydantic validated, so the
            # field path names the keys the client actually sent.
            return None, (first_error_message(e, lifted), "invalid_session_update")
        if update.tools and self._async_chunk_enabled():
            # Refuse rather than half-work. Two independent things break under
            # async_chunk: the buffer yields one TokensPrompt per segment, so a
            # tool-call continuation would reattach only the final segment's
            # audio and lose the start of the utterance; and the generation loop
            # never sees one complete thinker turn to scan for a <tool_call>
            # block. Aggregating the audio would fix only the first, leaving the
            # feature looking supported while still broken -- so the limitation
            # is explicit instead. `tool_choice` is still recorded: it is session
            # state, and a later update may bring tools this server can apply.
            #
            # Only a NON-empty list gets here: `tools: []` is how a client clears
            # its tools, and refusing a clear would leave the previous tools in
            # place - the opposite of what was asked, for a request that needs
            # nothing this mode cannot do.
            update.tools = None
            return update, (
                "Tool calling on /v1/realtime requires async_chunk to be disabled "
                "(serve with --no-async-chunk); tools were not applied.",
                "tools_require_no_async_chunk",
            )
        return update, None

    def _apply_session_tools(self, update: OmniSessionUpdate) -> None:
        """Record the tool state of an accepted `session.update`.

        A field the event does not carry leaves the session's current value
        alone, so `tools` and `tool_choice` can be set in separate updates.
        """
        if update.tools is not None:
            # Plain dicts again for the chat template, in the nested
            # `{"type": "function", "function": {...}}` form whichever shape the
            # client sent. `exclude_none` drops the optional fields it never set,
            # so a nested definition comes back as it arrived - except that
            # ChatCompletionToolsParam's validator copies a tool-level
            # `defer_loading` down into `function`.
            self._tools = [tool.model_dump(exclude_none=True) for tool in update.tools]
        if update.tool_choice is not None:
            if isinstance(update.tool_choice, OmniNamedToolChoice):
                # The closest this path can get: the tools stay active and the
                # request is treated as `"required"`. The name itself is NOT kept
                # - nothing on this path could act on it, and state nothing reads
                # only invites the belief that something does.
                self._tool_choice = "required"
                self._warn_tool_choice_not_enforced(f"the function {update.tool_choice.function.name!r}")
            else:
                self._tool_choice = update.tool_choice
                if update.tool_choice == "required":
                    self._warn_tool_choice_not_enforced("required")

    def _warn_tool_choice_not_enforced(self, requested: str) -> None:
        """Say once per session that a `tool_choice` was recorded, not enforced.

        `"required"` and a named function are accepted so a client written
        against the OpenAI API keeps working, but this path drives the engine with
        plain sampling params, so neither can be forced - see the class
        docstring's scope notes. Told to the operator's log rather than the
        client, because it is not an error and the endpoint sends no
        `session.updated` event to carry it (neither does upstream). Once per
        session: a client that sets it on every `session.update` would otherwise
        fill the log with it.
        """
        if self._tool_choice_not_enforced_warned:
            return
        self._tool_choice_not_enforced_warned = True
        logger.warning(
            "session.update asked for tool_choice=%s; /v1/realtime records it but cannot enforce it "
            '(no guided decoding on this path), so it behaves like "auto" and the model may answer '
            "without calling a tool",
            requested,
        )

    def _active_tools(self) -> list[dict[str, Any]] | None:
        """This session's tools, or None when tool calling is off.

        The single decision behind both halves of the feature: None keeps the
        `<tools>` preamble out of the rendered prompt (`buffer_realtime_audio`
        renders the plain template) and keeps `_run_generation` from scanning the
        generated text for `<tool_call>`. Off means `tool_choice="none"` or no
        tools - and `tools: []` is how a client clears its tools, so it has to
        mean the same as never having declared any, not "declare nothing, then
        parse calls the prompt never offered".

        Declared tools stay on the session either way, so a later
        `session.update` can turn them back on with `"auto"` without resending
        them.
        """
        if self._tool_choice == "none" or not self._tools:
            return None
        return self._tools

    def _async_chunk_enabled(self) -> bool:
        """Whether the server runs in async-chunk mode.

        Read off ``model_config`` the same way ``serving_speech.py`` does
        (``:3232``, ``:3569``).
        """
        return bool(getattr(self.serving.model_config, "async_chunk", False))

    async def _enqueue_tool_result(self, item: dict) -> None:
        """Validate a `function_call_output` before it can influence generation.

        Without this, any dict carrying the right `type` was accepted: a missing
        or non-string `call_id` never matched a pending call, and `output` was
        coerced with `str()`, so a client typo left generation waiting with
        nothing reported back. The shape itself lives in
        `FunctionCallOutputItem`; whether a well-formed `call_id` matches a call
        this turn actually made is answered later against the pending map
        (`unknown_tool_call_id` in `_await_tool_results_and_continue`).
        """
        try:
            FunctionCallOutputItem.model_validate(item)
        except ValidationError as e:
            await self.send_error(first_error_message(e, item), "invalid_function_call_output")
            return
        # The client's own dict is what gets queued: the wait loop reads
        # `call_id`/`output` straight off it, and re-serializing the validated
        # model here would only add a way for the two to drift.
        self._tool_result_queue.put_nowait(item)

    async def start_generation(self):
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        # New user turn: reset the tool-round budget and discard any tool result
        # left over from a previous turn, which would otherwise be consumed as if
        # it answered one of this turn's calls.
        self._tool_rounds = 0
        while not self._tool_result_queue.empty():
            self._tool_result_queue.get_nowait()

        # Decided once for the whole turn, including the continuations after a
        # tool result: the prompt below is rendered with (or without) the <tools>
        # preamble, and every generation in this turn scans the model's text on
        # exactly that basis.
        self._turn_tools = self._active_tools()

        audio_stream = self.audio_stream_generator()
        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        streaming_input_gen = self._buffer_realtime_audio_with_tools(audio_stream, input_stream)
        self.generation_task = asyncio.create_task(self._run_generation(streaming_input_gen, input_stream))

    async def _render_prompt(self, prompt: PromptType) -> StreamingInput:
        model_config = self.serving.model_config
        parsed_prompt = parse_model_prompt(model_config, prompt)
        (engine_input,) = await self.serving.renderer.render_cmpl_async([parsed_prompt])
        return StreamingInput(prompt=engine_input)

    async def _buffer_realtime_audio_with_tools(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
    ) -> AsyncGenerator[StreamingInput, None]:
        """Equivalent to `OpenAIServingRealtime.transcribe_realtime`, but
        threads this turn's tools through to the model's
        `buffer_realtime_audio`. The base class's `transcribe_realtime` has a
        fixed (audio_stream, input_stream, model_config) call signature with no
        seam for extra per-connection state like tools, so this reimplements
        its (short) body directly rather than patching upstream vLLM."""
        stream_input_iter = self.serving.model_cls.buffer_realtime_audio(
            audio_stream, input_stream, self.serving.model_config, tools=self._turn_tools
        )
        async for prompt in stream_input_iter:
            # Remember the pre-expansion prompt so tool-call continuations can
            # re-anchor on the user's audio (see self._turn_prompt).
            if isinstance(prompt, dict):
                self._turn_prompt = dict(prompt)
            yield await self._render_prompt(prompt)

    async def _render_token_prompt(
        self,
        prompt_token_ids: list[int],
        multi_modal_data: dict[str, Any] | None = None,
    ) -> AsyncGenerator[StreamingInput, None]:
        token_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
        # Tool-call continuation: re-attach the turn's audio. Without it the
        # `<|audio_pad|>` placeholder still sits in the token ids but has no
        # encoder output behind it, so the thinker cannot see what the user asked
        # and free-associates unrelated tool calls instead of answering.
        if multi_modal_data:
            token_prompt["multi_modal_data"] = multi_modal_data
        yield await self._render_prompt(token_prompt)

    @staticmethod
    def _tensor_to_numpy(value) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            arr = value
        elif hasattr(value, "detach"):
            arr = value.detach().float().cpu().numpy()
        else:
            try:
                arr = np.asarray(value)
            except Exception:
                return None
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _numpy_audio_prefix_match(prev: np.ndarray, curr: np.ndarray) -> bool:
        n = prev.shape[0]
        if n == 0:
            return True
        if curr.shape[0] < n:
            return False
        return bool(np.allclose(curr[:n], prev, rtol=1e-3, atol=2e-4))

    def _raw_waveform_to_deltas(self, arr: np.ndarray) -> list[np.ndarray]:
        """Convert one streaming PCM f32 chunk into incremental piece(s) for the client.

        Some engine paths emit a growing cumulative waveform each step; others emit
        true per-step deltas. We support both without duplicating audio on the client.
        """
        if arr.size == 0:
            return []
        ref = self._realtime_audio_ref
        if ref is None:
            self._realtime_audio_ref = arr.copy()
            return [arr]
        if self._numpy_audio_prefix_match(ref, arr):
            delta = arr[ref.shape[0] :]
            self._realtime_audio_ref = arr.copy()
            return [delta] if delta.size > 0 else []
        # True per-step delta (not a prefix extension of what we have seen).
        self._realtime_audio_ref = np.concatenate([ref, arr])
        return [arr]

    def _extract_audio_chunks(self, output) -> tuple[list[np.ndarray], int]:
        mm = getattr(output, "multimodal_output", None)
        if mm is None:
            return [], 24000
        # Support both MultimodalPayload and plain dict
        if not isinstance(mm, Mapping):
            return [], 24000

        sr = mm.get("sr") or mm.get("sample_rate") or mm.get("audio_sample_rate") or 24000
        if isinstance(sr, (list, tuple)) and sr:
            sr = sr[-1]
        if hasattr(sr, "item"):
            sr = sr.item()
        sample_rate_hz = int(sr)
        key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
        if key is None:
            return [], sample_rate_hz

        raw_audio = mm.get(key)
        chunks: list[np.ndarray] = []
        if isinstance(raw_audio, (list, tuple)):
            if len(raw_audio) > 0:
                arr = self._tensor_to_numpy(raw_audio[-1])
                if arr is not None and arr.size > 0:
                    chunks.extend(self._raw_waveform_to_deltas(arr))
        else:
            arr = self._tensor_to_numpy(raw_audio)
            if arr is not None and arr.size > 0:
                chunks.extend(self._raw_waveform_to_deltas(arr))
        return chunks, sample_rate_hz

    @staticmethod
    def _pcm16_b64(audio_f32: np.ndarray) -> str:
        clipped = np.clip(audio_f32, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return base64.b64encode(pcm16.tobytes()).decode("utf-8")

    async def _emit_tool_call_deltas(self, tool_deltas: list[ToolCallDelta]) -> None:
        for delta in tool_deltas:
            if delta.name is not None:
                call = _PendingToolCall(call_id=f"call_{uuid4().hex[:24]}", name=delta.name)
                self._pending_tool_calls[delta.index] = call
                await self.send_json(
                    ResponseOutputItemAdded(item=FunctionCallItem(name=call.name, call_id=call.call_id)).model_dump()
                )
            if delta.arguments_delta:
                call = self._pending_tool_calls.get(delta.index)
                if call is None:
                    continue  # shouldn't happen: name delta always precedes argument deltas for the same index
                call.arguments += delta.arguments_delta
                await self.send_json(
                    ResponseFunctionCallArgumentsDelta(call_id=call.call_id, delta=delta.arguments_delta).model_dump()
                )

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ):
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        sent_audio = False
        audio_done_sent = False
        full_text = ""
        prompt_token_ids_len = 0
        completion_tokens_len = 0
        self._realtime_audio_ref = None

        request_prompt_token_ids: list[int] = []
        assistant_token_ids: list[int] = []
        tool_state = ToolCallStreamState()
        self._pending_tool_calls = {}
        # The turn's own decision, not the session's current one: this runs again
        # for a tool-call continuation, whose prompt still carries the <tools>
        # preamble and the call history, so a `session.update` that arrives while
        # the client is running the tool must not stop the extractor scanning it -
        # nor start it scanning a prompt that never declared any tools.
        tools_active = self._turn_tools is not None

        # Coerce cumulative outputs to delta outputs; this ensures
        # we don't emit redundant MM data & drain after emitting.
        sampling_params_list = list(self.engine.default_sampling_params_list)
        sampling_params_list = coerce_param_message_types(
            sampling_params_list,
            is_streaming=True,
        )

        result_gen = None
        try:
            result_gen = self.engine.generate(
                prompt=streaming_input_gen,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
            )

            async for output in result_gen:
                stage_id = getattr(output, "stage_id", None)
                if stage_id == 0 and output.outputs:
                    first_output = output.outputs[0]
                    new_token_ids = list(first_output.token_ids)
                    if new_token_ids:
                        input_stream.put_nowait(new_token_ids)
                        assistant_token_ids.extend(new_token_ids)

                    if output.prompt_token_ids:
                        prompt_token_ids_len = max(
                            prompt_token_ids_len,
                            len(output.prompt_token_ids),
                        )
                        if not request_prompt_token_ids:
                            request_prompt_token_ids = list(output.prompt_token_ids)

                    delta_text = first_output.text or ""
                    full_text += delta_text
                    completion_tokens_len += len(new_token_ids)

                    if delta_text and tools_active:
                        content_delta, tool_deltas = extract_deltas(full_text, tool_state)
                        if content_delta:
                            await self.send(TranscriptionDelta(delta=content_delta))
                        if tool_deltas:
                            await self._emit_tool_call_deltas(tool_deltas)
                    elif delta_text:
                        # No tools this turn: stream the text as it comes, the way
                        # this endpoint did before tool calling existed. Nothing
                        # parses <tool_call>, so `tool_state` stays empty and every
                        # gate reading it below (audio suppression, the tool-result
                        # wait, the terminal event) takes the plain path.
                        await self.send(TranscriptionDelta(delta=delta_text))

                audio_chunks, sample_rate = self._extract_audio_chunks(output)
                if audio_chunks and not tool_state.has_tool_calls():
                    for chunk in audio_chunks:
                        sent_audio = True
                        await self.send_json(
                            {
                                "type": "response.audio.delta",
                                "audio": self._pcm16_b64(chunk),
                                "format": "pcm16",
                                "sample_rate_hz": sample_rate,
                            }
                        )
                # else: a tool-call turn - the pipeline still synthesizes audio for the
                # raw <tool_call> text (no cheap hook to skip talker/code2wav for just
                # this turn), we just don't forward it to the client.

                if not self._is_connected:
                    break

            if tool_state.has_tool_calls():
                for call in self._pending_tool_calls.values():
                    await self.send_json(
                        ResponseFunctionCallArgumentsDone(call_id=call.call_id, arguments=call.arguments).model_dump()
                    )
                if self._is_connected:
                    await self._await_tool_results_and_continue(request_prompt_token_ids, assistant_token_ids)
                return

            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )
            await self.send(TranscriptionDone(text=full_text, usage=usage))

            if sent_audio:
                await self.send_json({"type": "response.audio.done", "has_audio": True})
                audio_done_sent = True
        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")
        finally:
            # Close the generator explicitly so AsyncOmni.generate's cleanup
            # (input-pump cancellation and engine-side abort) runs now rather
            # than whenever the event loop garbage-collects the async
            # generator; the delay window is where a disconnected session
            # keeps cycling through the stages (issue #4271).
            if result_gen is not None:
                try:
                    await result_gen.aclose()
                except Exception:
                    logger.exception("Failed to close realtime result generator")
            # Always send terminal event so clients don't hang forever.
            if self._is_connected and not audio_done_sent and not tool_state.has_tool_calls():
                try:
                    await self.send_json({"type": "response.audio.done", "has_audio": sent_audio})
                except Exception:
                    logger.exception("Failed to send response.audio.done")
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

    @staticmethod
    def _close_assistant_turn(tokenizer, assistant_token_ids: list[int]) -> list[int]:
        """Terminate the model's tool-call turn with `<|im_end|>\\n` before a
        tool-result turn is appended.

        The tool-result suffix opens with `<|im_start|>user`, but the raw generated
        token ids stop at the tool call without the closing `<|im_end|>` that the
        chat template would emit. Splicing them directly yields
        `</tool_call><|im_start|>user`, leaving the assistant turn open - a
        malformed conversation the thinker responds to by re-emitting the same tool
        call instead of answering, looping until something bounds it. The reference
        HF path answers the identical prompt because `apply_chat_template` closes
        the turn. Idempotent: only the missing pieces are added.
        """
        ids = list(assistant_token_ids)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if im_end_id in (None, getattr(tokenizer, "unk_token_id", None)):
            return ids  # unexpected tokenizer; leave the splice untouched
        if im_end_id not in ids[-2:]:
            ids.append(im_end_id)
        if newline_ids and ids[-len(newline_ids) :] != newline_ids:
            ids.extend(newline_ids)
        return ids

    async def _await_tool_results_and_continue(
        self,
        prior_prompt_token_ids: list[int],
        assistant_token_ids: list[int],
    ) -> None:
        """Block until the client has submitted a `function_call_output` for
        every pending tool call from the turn that just finished, then splice
        the results onto the raw token sequence (prior prompt + the model's
        own generated tool-call tokens + a freshly-rendered tool-result turn)
        and continue generation - which streams the model's actual spoken
        reply as a normal `_run_generation` call, recursing again if the
        model chains into another tool call (bounded by MAX_TOOL_ROUNDS)."""
        self._tool_rounds += 1
        if self._tool_rounds > self.MAX_TOOL_ROUNDS:
            # A model that keeps re-emitting tool calls instead of answering
            # would otherwise recurse without bound. Observed during bring-up
            # when the spliced prompt was malformed (see _close_assistant_turn).
            logger.warning("tool-call chain exceeded %d rounds; abandoning turn", self.MAX_TOOL_ROUNDS)
            if self._is_connected:
                await self.send_error(
                    f"Tool-call chain exceeded {self.MAX_TOOL_ROUNDS} rounds without a final response",
                    "tool_call_loop",
                )
            return

        pending = dict(self._pending_tool_calls)
        call_id_to_index = {call.call_id: idx for idx, call in pending.items()}
        results_by_index: dict[int, str] = {}

        while len(results_by_index) < len(pending) and self._is_connected:
            # Bounded wait so a client that disappears mid-tool-call cannot park
            # this task forever; a slow-but-live client is unaffected.
            try:
                item = await asyncio.wait_for(self._tool_result_queue.get(), timeout=_TOOL_RESULT_POLL_S)
            except TimeoutError:
                continue
            call_id = item.get("call_id")
            idx = call_id_to_index.get(call_id)
            if idx is None:
                # Tell the client: silently dropping this would leave the turn
                # waiting for a result that is never going to match. Keep waiting
                # afterwards, since the correct result may still arrive.
                logger.warning("received function_call_output for unknown call_id=%s", call_id)
                await self.send_error(
                    f"No pending tool call with call_id={call_id!r}; expected one of {sorted(call_id_to_index)}",
                    "unknown_tool_call_id",
                )
                continue
            # `output` is validated as a string at ingress (_enqueue_tool_result).
            results_by_index[idx] = item["output"]

        if not self._is_connected:
            return

        model_config = self.serving.model_config
        tokenizer = cached_tokenizer_from_config(model_config)
        # Pass the processor's chat_template explicitly - see the matching
        # comment in Qwen3OmniMoeForConditionalGeneration.buffer_realtime_audio
        # for why relying on safe_apply_chat_template's own auto-resolution
        # is unsafe for this checkpoint.
        processor = cached_processor_from_config(model_config)
        # One `role="tool"` message PER result, in call order. The chat template
        # emits one <tool_response> block per tool message and groups consecutive
        # tool messages under a single <|im_start|>user turn, so passing separate
        # messages is what lets the model associate each result with its call.
        # Joining the results instead produces a single <tool_response> holding
        # both outputs, which breaks parallel calls.
        #
        # `sorted()` on the parser-assigned index is call order: extract_deltas
        # appends `tool_call_starts` in the order <tool_call> appears in the
        # generated text, so this is stable regardless of the order in which the
        # client returns the results.
        #
        # No `tools=` here: this continues a conversation whose token history
        # already carries the tools system preamble; it is not a fresh turn.
        tool_messages: list[dict[str, str]] = [
            {"role": "tool", "content": results_by_index[i]} for i in sorted(results_by_index)
        ]
        suffix_text = safe_apply_chat_template(
            model_config,
            tokenizer,
            tool_messages,
            chat_template=processor.chat_template,
            add_generation_prompt=True,
            tokenize=False,
        )
        # Splice onto the turn's PRE-expansion prompt, not the engine's
        # post-expansion `output.prompt_token_ids`. The latter carries the expanded
        # `<|audio_pad|>` run, and re-submitting it as a bare TokensPrompt drops the
        # audio itself: the thinker then sees placeholder tokens with no encoder
        # output, loses the user's question entirely, and answers by inventing
        # further tool calls (unrelated cities/items) instead of replying. Falling
        # back to `prior_prompt_token_ids` only matters for a continuation that never
        # went through buffer_realtime_audio (no audio to lose in that case).
        base_prompt = self._turn_prompt or {}
        base_token_ids = list(base_prompt.get("prompt_token_ids") or prior_prompt_token_ids)
        multi_modal_data = base_prompt.get("multi_modal_data")
        continuation_token_ids = (
            base_token_ids + self._close_assistant_turn(tokenizer, assistant_token_ids) + tokenizer.encode(suffix_text)
        )

        # Advance the base so a chained tool call next round splices onto this
        # turn's full history while the audio stays attached exactly once, at the
        # front, still un-expanded.
        if self._turn_prompt is not None:
            self._turn_prompt = {**base_prompt, "prompt_token_ids": continuation_token_ids}

        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        await self._run_generation(self._render_token_prompt(continuation_token_ids, multi_modal_data), input_stream)

    async def send_json(self, payload: dict):
        try:
            await self.websocket.send_text(json.dumps(payload))
        except Exception:
            # A failed send means the client is gone; flag it so the
            # generation loop stops instead of retrying into a dead socket.
            self._is_connected = False
            raise

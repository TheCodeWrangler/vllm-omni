"""Throwaway local LiveKit worker for testing Qwen3-Omni (via vllm-omni's
/v1/realtime) as a speech-to-speech "brain", with a fallback to OpenAI's real
Realtime API for comparison/demo purposes.

NOT part of services/voice-agent's production package — that service
deliberately dropped its LLM/brain in June 2026 (see its README/CLAUDE.md).
This is a standalone script for local GPU testing only.

Usage:
    source .venv/bin/activate
    BRAIN=openai python worker.py dev     # gpt-realtime via OPENAI_US_API_BASE_URL
    BRAIN=omni   python worker.py dev     # self-hosted Qwen3-Omni via vllm-omni

Then use voice-agent's existing test UI (unmodified) pointed at the same
local LiveKit server to talk to it in a browser.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import openai as lk_openai
from livekit.plugins.openai.realtime.realtime_model import InputAudioTranscription

from omni_realtime import OmniRealtimeModel

load_dotenv()

logger = logging.getLogger("omni_worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

BRAIN = os.environ.get("BRAIN", "openai").strip().lower()
OMNI_WS_URL = os.environ.get("OMNI_WS_URL", "ws://localhost:8091/v1/realtime")
OMNI_MODEL = os.environ.get("OMNI_MODEL", "qwen3-omni")
# Speaker/voice for the omni talker. The custom-voice checkpoint
# (~/omni_custom_build/qwen-omni-custom-voice) grafts a 4th speaker, "custom", alongside
# the stock chelsie/ethan/aiden; the checkpoint's own default is whichever key comes
# first (chelsie), so ask for the grafted one explicitly.
OMNI_VOICE = os.environ.get("OMNI_VOICE", "custom")
OPENAI_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")

# Everything the thinker writes is spoken by the talker, and the talker is conditioned on
# the thinker's hidden states - so no text normalisation downstream can reach it. Symbols the
# model writes ITSELF are therefore only fixable at generation time, here. Left unconstrained
# it also emits markdown (**bold**, "- " bullets), which gets read out as punctuation.
# Measured on this checkpoint: "Na+ 138" is spoken "Not thirty eight" and "2+ pitting edema"
# becomes "Atropine edema", so this is a correctness guard rather than polish.
_SPEAK_READY = (
    "Your replies are spoken aloud, so write every reply exactly as it should be SPOKEN. "
    "Never use markdown, bullet points or symbols - write them as words: '<' as 'less than', "
    "'<=' as 'less than or equal to', '/' as 'per' or 'over', 'Na+' as 'sodium', "
    "'2+' as 'two plus', ranges as 'three to five', times as 'eight thirty'. "
    "Never abbreviate units: say 'milligrams per deciliter'. "
    # "write numbers as words" alone produced digit-by-digit reading: 37.2 was spoken
    # "three seven point two". Give the grouping explicitly.
    "Say numbers the way a person would, grouped, not digit by digit: 37.2 is "
    "'thirty-seven point two', 120 is 'one hundred twenty', 138 is 'one hundred thirty-eight'. "
    # Units are safety-relevant: it reported 37.2 C as 'degrees Fahrenheit'.
    "State units exactly as given and never convert or substitute them. "
    # A broad query ("weather in Spain") gets fanned out over several cities, which is
    # sensible - it picked real Spanish cities. Telling it "one tool per turn" did not
    # hold and discourages useful behaviour, so ask for the landing instead: gather, then
    # summarise. The client also forces a wrap-up once the round budget is spent, so a
    # long gather ends in speech rather than silence (see _MAX_TOOL_ROUNDS).
    "For a broad request you may look up two or three examples, but then STOP and give one "
    "short spoken summary of what you found. Never end a turn having only called tools."
)

_INSTRUCTIONS = (
    "You are a friendly voice assistant for a local demo. Keep replies short. "
    "You have two tools available: get_weather and check_inventory. Use them when relevant. "
    + _SPEAK_READY
)

_MOCK_WEATHER = ["sunny and 72 degrees", "overcast with a light breeze", "raining lightly"]
_MOCK_INVENTORY = ["12 units in stock", "out of stock, restocking Friday", "3 units left, low stock"]


class DemoAgent(Agent):
    """Two mocked tools: sleep to simulate a slow backend call, then a cached/canned response.

    With BRAIN=omni these fire via OmniRealtimeSession's own detection of the
    server's Hermes-style `<tool_call>{...}</tool_call>` output (vllm-omni's
    /v1/realtime protocol still has no native function-call events, so the
    client parses the tag out of the transcript itself). They also work with
    BRAIN=openai using that provider's native function-calling events.
    """

    def __init__(self) -> None:
        # @function_tool-decorated methods below are auto-registered by Agent.__init__;
        # passing them again via tools=[...] here causes a "duplicate function name" error.
        super().__init__(instructions=_INSTRUCTIONS)

    @function_tool
    async def get_weather(self, city: str) -> str:
        """Look up the current weather for a city. Args: city: the city name."""
        logger.info("tool call: get_weather(%s)", city)
        await asyncio.sleep(1.5)
        return f"The weather in {city} is {random.choice(_MOCK_WEATHER)} (mocked)."

    @function_tool
    async def check_inventory(self, item: str) -> str:
        """Check warehouse inventory for an item. Args: item: the item name."""
        logger.info("tool call: check_inventory(%s)", item)
        await asyncio.sleep(1.5)
        return f"For {item}: {random.choice(_MOCK_INVENTORY)} (mocked)."


def _build_realtime_model():
    if BRAIN == "omni":
        logger.info("brain=omni -> vllm-omni %s at %s (voice=%s)", OMNI_MODEL, OMNI_WS_URL, OMNI_VOICE)
        return OmniRealtimeModel(ws_url=OMNI_WS_URL, model=OMNI_MODEL, voice=OMNI_VOICE)
    if BRAIN == "openai":
        base_url = os.environ.get("OPENAI_US_API_BASE_URL") or None
        api_key = os.environ.get("OPENAI_API_KEY")
        logger.info("brain=openai -> %s (base_url=%s)", OPENAI_MODEL, base_url or "default")
        # Explicit, not NOT_GIVEN: makes OpenAI transcribe the user's own speech too (not just
        # the agent's reply), so both sides show up as LiveKit native transcription segments.
        kwargs = {
            "model": OPENAI_MODEL,
            "api_key": api_key,
            "input_audio_transcription": InputAudioTranscription(model="gpt-4o-mini-transcribe"),
        }
        if base_url:
            kwargs["base_url"] = base_url
        return lk_openai.realtime.RealtimeModel(**kwargs)
    raise ValueError(f"unknown BRAIN={BRAIN!r}, expected 'openai' or 'omni'")


async def _log_message_text(msg) -> None:  # type: ignore[no-untyped-def]  # MessageGeneration
    chunks: list[str] = []
    async for delta in msg.text_stream:
        chunks.append(delta)
    text = "".join(chunks)
    if text:
        logger.info("agent said: %s", text)


async def _log_generation(ev) -> None:  # type: ignore[no-untyped-def]  # GenerationCreatedEvent
    async for msg in ev.message_stream:
        asyncio.create_task(_log_message_text(msg), name="log-agent-text")


async def _publish_tool_call(ctx: JobContext, call, output) -> None:  # type: ignore[no-untyped-def]
    # llm.FunctionCall / llm.FunctionCallOutput | None (None on e.g. unknown-tool errors).
    # Published on its own topic so the test-ui's wire log (which already renders every
    # RoomEvent.DataReceived message, see test_ui.py) surfaces tool calls with no frontend
    # changes needed - it just shows up as another "DATA [tool_call]" line per participant tab.
    payload = {
        "name": call.name,
        "arguments": call.arguments,
        "output": output.output if output is not None else None,
        "is_error": output.is_error if output is not None else True,
    }
    logger.info("tool_call data message: %s", payload)
    try:
        await ctx.room.local_participant.publish_data(json.dumps(payload).encode(), topic="tool_call")
    except Exception:
        logger.exception("failed to publish tool_call data message")
    else:
        logger.info(
            "tool_call data message published OK (room=%s, local_identity=%s, connected=%s)",
            ctx.room.name,
            ctx.room.local_participant.identity,
            ctx.room.isconnected() if hasattr(ctx.room, "isconnected") else "?",
        )


async def _accept_with_voice_agent_identity(req: JobRequest) -> None:
    # The real services/voice-agent worker joins each room as "voice-agent-<room>"
    # (see its control.agent_identity_for_room) rather than livekit-agents' default
    # random "agent-<job_id>" identity. The test-ui's sourceFor() routes wire-log/
    # bubble rendering by that prefix - anything identity.startswith("agent-") gets
    # bucketed into the unrelated "verbatim-streaming-service" tab instead of
    # rendering as agent bubbles. Match that convention so this throwaway worker
    # shows up the same way the real one does.
    identity = f"voice-agent-{req.room.name}"
    await req.accept(identity=identity)
    logger.info("accepted job with identity=%s (room=%s)", identity, req.room.name)


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    session = AgentSession(llm=_build_realtime_model())

    @session.on("generation_created")
    def _on_gen(ev) -> None:  # type: ignore[no-untyped-def]
        logger.info("generation_created (user_initiated=%s)", getattr(ev, "user_initiated", "?"))
        asyncio.create_task(_log_generation(ev), name="log-generation")

    @session.on("function_tools_executed")
    def _on_tools(ev) -> None:  # type: ignore[no-untyped-def]  # FunctionToolsExecutedEvent
        for call, output in zip(ev.function_calls, ev.function_call_outputs):
            asyncio.create_task(_publish_tool_call(ctx, call, output), name="publish-tool-call")

    await session.start(room=ctx.room, agent=DemoAgent())
    logger.info("omni_worker session started in room %s (brain=%s)", ctx.room.name, BRAIN)

    done = asyncio.Event()
    session.on("close", lambda _ev: done.set())
    await done.wait()


def main() -> None:
    agent_name = os.environ.get("AGENT_NAME", "voice-agent")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=_accept_with_voice_agent_identity,
            agent_name=agent_name,
        )
    )


if __name__ == "__main__":
    main()

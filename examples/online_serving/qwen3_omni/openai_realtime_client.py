"""Realtime client for vLLM-Omni /v1/realtime (audio + text events).

This client:
1) Reads a local WAV file (must be mono, 16-bit PCM, 16kHz),
2) Streams PCM16 chunks to /v1/realtime with OpenAI-style events,
3) Receives response.audio.* and transcription.* events,
4) Saves synthesized audio to an output WAV file and optional text file.

By default each ``response.audio.delta`` is treated as an **incremental PCM**
chunk and all chunks are concatenated into the final ``--output-wav``.

Optional debugging: pass ``--delta-dump-dir DIR`` to write every
``response.audio.delta`` payload as ``delta_000001.wav``, ``delta_000002.wav``, …

Tool (function) calling: pass ``--tools`` to declare tools on ``session.update``,
``--tool-choice`` to pick ``none``/``auto``/``required``, and ``--tool-output`` to
answer every call the model makes with one canned result. The model's calls are
printed as they stream, the result is returned with ``conversation.item.create``,
and generation then continues into the spoken reply within the same turn. The
server refuses tools in async-chunk mode, so serve with ``--no-async-chunk``.

Usage:
  python openai_realtime_client.py \
      --url ws://localhost:8091/v1/realtime \
      --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
      --input-wav input_16k_mono.wav \
      --output-wav realtime_output.wav \
      --delta-dump-dir ./rt_delta_wavs

  python openai_realtime_client.py \
      --input-wav whats_the_weather_in_berlin_16k_mono.wav \
      --tools tools.json \
      --tool-choice auto \
      --tool-output '{"temperature_c": 21, "conditions": "sunny"}'

  where tools.json holds a list of tool definitions, e.g.
  [{"type": "function",
    "function": {"name": "get_weather",
                 "description": "Get the weather in a city",
                 "parameters": {"type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"]}}}]

Dependencies:
  pip install websockets
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import wave
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    raise SystemExit(1)


def _read_wav_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        comptype = wf.getcomptype()
        nframes = wf.getnframes()

        if nchannels != 1:
            raise ValueError(f"Input WAV must be mono (got {nchannels} channels).")
        if sampwidth != 2:
            raise ValueError(f"Input WAV must be 16-bit PCM (got sample width={sampwidth}).")
        if framerate != 16000:
            raise ValueError(f"Input WAV must be 16kHz (got {framerate} Hz).")
        if comptype != "NONE":
            raise ValueError(f"Input WAV must be uncompressed PCM (got comptype={comptype}).")
        if nframes <= 0:
            raise ValueError("Input WAV has no audio frames.")

        return wf.readframes(nframes)


def _write_wav_pcm16(path: Path, pcm16_bytes: bytes, sample_rate_hz: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm16_bytes)


async def run_client(
    url: str,
    model: str,
    input_wav: Path,
    output_wav: Path,
    output_text: Path | None,
    chunk_ms: int,
    send_delay_ms: int,
    delta_dump_dir: Path | None,
    request_idx: int = 1,
    total_requests: int = 1,
    tools: list[dict] | None = None,
    tool_choice: str = "auto",
    tool_output: str | None = None,
) -> None:
    log_prefix = f"[req {request_idx:02d}/{total_requests:02d}] " if total_requests > 1 else ""
    pcm16 = _read_wav_pcm16(input_wav)
    bytes_per_ms = 16000 * 2 // 1000  # mono PCM16 at 16kHz
    chunk_bytes = max(bytes_per_ms * chunk_ms, 2)

    incremental_pcm_parts: list[bytes] = []
    output_sample_rate = 24000
    delta_index = 0
    text_chunks: list[str] = []
    final_text: str = ""
    # call_id -> function name, from response.output_item.added: the later
    # arguments events carry only the call_id.
    tool_call_names: dict[str, str] = {}
    unanswered_tool_call = False

    if delta_dump_dir is not None:
        delta_dump_dir.mkdir(parents=True, exist_ok=True)

    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        # 1) Validate model, and declare tools if the caller asked for them.
        #    `tools`/`tool_choice` go under `session`, the shape the OpenAI
        #    Realtime API uses (the server also accepts them flat on the event).
        #    `model` stays top-level: that field is vLLM's own addition.
        session_update: dict = {"type": "session.update", "model": model}
        if tools or tool_choice != "auto":
            session_update["session"] = {"tool_choice": tool_choice}
            if tools:
                session_update["session"]["tools"] = tools
        await ws.send(json.dumps(session_update))

        # 2) Start generation once (non-final commit).
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))

        # 3) Stream audio chunks.
        for i in range(0, len(pcm16), chunk_bytes):
            chunk = pcm16[i : i + chunk_bytes]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    }
                )
            )
            if send_delay_ms > 0:
                await asyncio.sleep(send_delay_ms / 1000.0)

        # 4) Final commit closes input stream.
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        # 5) Receive server events until audio done.
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                # We only expect JSON text frames.
                continue

            event = json.loads(message)
            event_type = event.get("type")

            if event_type == "session.created":
                continue

            if event_type == "response.audio.delta":
                sr = event.get("sample_rate_hz")
                if isinstance(sr, int) and sr > 0:
                    output_sample_rate = sr
                audio_b64 = event.get("audio", "")
                if audio_b64:
                    pcm_delta = base64.b64decode(audio_b64)
                    incremental_pcm_parts.append(pcm_delta)
                    if delta_dump_dir is not None and pcm_delta:
                        delta_index += 1
                        dump_path = delta_dump_dir / f"delta_{delta_index:06d}.wav"
                        _write_wav_pcm16(dump_path, pcm_delta, output_sample_rate)
                        print(
                            f"{log_prefix}delta dump #{delta_index}: {dump_path} "
                            f"(pcm bytes={len(pcm_delta)}, sr={output_sample_rate})"
                        )
                continue

            if event_type == "transcription.delta":
                delta = event.get("delta", "")
                if delta:
                    text_chunks.append(delta)
                    print(delta, end="", flush=True)
                continue

            if event_type == "transcription.done":
                final_text = event.get("text", "") or "".join(text_chunks)
                usage = event.get("usage")
                final_text_with_tag = f"Final transcription: {final_text}"
                if text_chunks:
                    print()
                print(f"{log_prefix}{final_text_with_tag}")
                if usage:
                    print(f"{log_prefix}text usage: {usage}")
                continue

            if event_type == "response.output_item.added":
                item = event.get("item") or {}
                call_id = item.get("call_id", "")
                tool_call_names[call_id] = item.get("name", "")
                print(f"{log_prefix}function call started: {item.get('name')} call_id={call_id}")
                continue

            if event_type == "response.function_call_arguments.delta":
                call_id = event.get("call_id", "")
                print(
                    f"{log_prefix}function call (delta): {tool_call_names.get(call_id, '')} "
                    f"call_id={call_id} arguments+={event.get('delta')}"
                )
                continue

            if event_type == "response.function_call_arguments.done":
                call_id = event.get("call_id", "")
                name = tool_call_names.get(call_id, "")
                print(f"{log_prefix}function call: {name}({event.get('arguments')}) call_id={call_id}")
                if tool_output is None:
                    # The server holds the turn open until every call it made has
                    # been answered, so nothing more is coming: stop instead of
                    # waiting forever.
                    print(f"{log_prefix}no --tool-output given, leaving the call unanswered")
                    unanswered_tool_call = True
                    break
                # Answer immediately, from inside the receive loop: with parallel
                # calls the next `.done` may still be on its way, and generation
                # only resumes once every call has a result.
                await ws.send(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": tool_output,
                            },
                        }
                    )
                )
                print(f"{log_prefix}sent tool result for call_id={call_id}: {tool_output}")
                continue

            if event_type == "response.audio.done":
                break

            if event_type == "error":
                raise RuntimeError(f"Server error: {event}")

        all_pcm16 = b"".join(incremental_pcm_parts)
        if not all_pcm16 and unanswered_tool_call:
            print(f"{log_prefix}No audio: the tool call was left unanswered (pass --tool-output to answer it).")
            return
        if not all_pcm16:
            raise RuntimeError("No audio received from server.")

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        _write_wav_pcm16(output_wav, all_pcm16, output_sample_rate)
        print(f"{log_prefix}Saved realtime audio to: {output_wav} (incremental chunks joined)")

        if output_text is not None:
            text_to_save = final_text if final_text else "".join(text_chunks)
            output_text.parent.mkdir(parents=True, exist_ok=True)
            output_text.write_text(text_to_save, encoding="utf-8")
            print(f"{log_prefix}Saved realtime text to: {output_text}")


def _indexed_output_path(path: Path | None, index: int, total: int) -> Path | None:
    if path is None or total <= 1:
        return path
    return path.with_name(f"{path.stem}_{index:02d}{path.suffix}")


async def run_clients_concurrent(
    *,
    url: str,
    model: str,
    input_wav: Path,
    output_wav: Path,
    output_text: Path | None,
    chunk_ms: int,
    send_delay_ms: int,
    delta_dump_dir: Path | None,
    num_requests: int,
    concurrency: int,
    tools: list[dict] | None = None,
    tool_choice: str = "auto",
    tool_output: str | None = None,
) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def _run_one(index: int) -> tuple[int, bool, str | None]:
        per_output_wav = _indexed_output_path(output_wav, index, num_requests)
        per_output_text = _indexed_output_path(output_text, index, num_requests)
        per_delta_dir = None
        if delta_dump_dir is not None:
            per_delta_dir = delta_dump_dir / f"req_{index:02d}"
        async with sem:
            try:
                await run_client(
                    url=url,
                    model=model,
                    input_wav=input_wav,
                    output_wav=per_output_wav,
                    output_text=per_output_text,
                    chunk_ms=chunk_ms,
                    send_delay_ms=send_delay_ms,
                    delta_dump_dir=per_delta_dir,
                    request_idx=index,
                    total_requests=num_requests,
                    tools=tools,
                    tool_choice=tool_choice,
                    tool_output=tool_output,
                )
                return index, True, None
            except Exception as exc:
                return index, False, str(exc)

    tasks = [asyncio.create_task(_run_one(i), name=f"rt-client-{i}") for i in range(1, num_requests + 1)]
    results = await asyncio.gather(*tasks)

    failed = [(idx, err) for idx, ok, err in results if not ok]
    succeeded = num_requests - len(failed)
    print(f"[summary] succeeded={succeeded}, failed={len(failed)}, total={num_requests}")
    if failed:
        for idx, err in failed:
            print(f"[summary] req {idx:02d} failed: {err}")
        raise RuntimeError(f"{len(failed)} concurrent request(s) failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime audio/text client for vLLM-Omni")
    parser.add_argument("--url", default="ws://localhost:8091/v1/realtime", help="WebSocket URL")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model name for session.update",
    )
    parser.add_argument("--input-wav", required=True, type=Path, help="Input WAV (mono, PCM16, 16kHz)")
    parser.add_argument("--output-wav", default=Path("realtime_output.wav"), type=Path, help="Output WAV path")
    parser.add_argument(
        "--output-text",
        default=None,
        type=Path,
        help="Optional output text path for final transcription",
    )
    parser.add_argument("--chunk-ms", type=int, default=200, help="Input chunk size in milliseconds")
    parser.add_argument(
        "--send-delay-ms",
        type=int,
        default=0,
        help="Delay between chunk sends; set >0 to simulate realtime upload",
    )
    parser.add_argument(
        "--delta-dump-dir",
        type=Path,
        default=None,
        help="If set, each response.audio.delta is saved as delta_NNNNNN.wav under this directory",
    )
    parser.add_argument("--num-requests", type=int, default=1, help="Total number of requests to send")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent websocket requests",
    )
    parser.add_argument(
        "--tools",
        type=Path,
        default=None,
        help=(
            "Optional JSON file holding a list of tool definitions (same shape as the chat "
            "completions `tools` field). Declared on session.update, so the model can answer "
            "with a function call instead of speaking. Requires a server started with "
            "--no-async-chunk"
        ),
    )
    parser.add_argument(
        "--tool-choice",
        default="auto",
        choices=["none", "auto", "required"],
        help=(
            "tool_choice for session.update (default: auto). `none` disables tool calling for "
            "the session; `required` is accepted but not enforced by this endpoint"
        ),
    )
    parser.add_argument(
        "--tool-output",
        default=None,
        help=(
            "Canned tool result (a JSON string) returned via conversation.item.create for every "
            "function call the model makes; without it the call is printed and left unanswered"
        ),
    )
    args = parser.parse_args()

    tools = json.loads(args.tools.read_text(encoding="utf-8")) if args.tools is not None else None

    if args.num_requests <= 0:
        raise ValueError("--num-requests must be >= 1")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be >= 1")
    concurrency = min(args.concurrency, args.num_requests)

    if args.num_requests == 1:
        asyncio.run(
            run_client(
                url=args.url,
                model=args.model,
                input_wav=args.input_wav,
                output_wav=args.output_wav,
                output_text=args.output_text,
                chunk_ms=args.chunk_ms,
                send_delay_ms=args.send_delay_ms,
                delta_dump_dir=args.delta_dump_dir,
                tools=tools,
                tool_choice=args.tool_choice,
                tool_output=args.tool_output,
            )
        )
    else:
        asyncio.run(
            run_clients_concurrent(
                url=args.url,
                model=args.model,
                input_wav=args.input_wav,
                output_wav=args.output_wav,
                output_text=args.output_text,
                chunk_ms=args.chunk_ms,
                send_delay_ms=args.send_delay_ms,
                delta_dump_dir=args.delta_dump_dir,
                num_requests=args.num_requests,
                concurrency=concurrency,
                tools=tools,
                tool_choice=args.tool_choice,
                tool_output=args.tool_output,
            )
        )


if __name__ == "__main__":
    main()

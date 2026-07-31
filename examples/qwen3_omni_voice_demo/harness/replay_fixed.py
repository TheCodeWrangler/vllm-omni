"""Replay a captured live turn verbatim: exact session.update payload + exact PCM.
This removes every synthesis difference between my probes and the real demo."""
import asyncio, base64, json, sys, wave
import websockets

WS = "ws://localhost:8091/v1/realtime"
STEM = sys.argv[1] if len(sys.argv) > 1 else "/tmp/omni_turn_dumps/turn-847ab17d"

session = json.load(open(f"{STEM}.session.json"))
w = wave.open(f"{STEM}.wav")
SR = w.getframerate()
pcm = w.readframes(w.getnframes())
print(f"replaying {len(pcm)/2/SR:.2f}s @ {SR}Hz, voice={session.get('voice')}, "
      f"tools={[t['function']['name'] for t in session.get('tools', [])]}")

_MOCK = ["sunny and 72 degrees", "overcast with a light breeze", "raining lightly"]

async def main():
    async with websockets.connect(WS, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps(session))               # verbatim
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))
        step = SR * 2 * 200 // 1000
        for i in range(0, len(pcm), step):
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(pcm[i:i+step]).decode()}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        rounds = 0
        while True:
            try:
                r = await asyncio.wait_for(ws.recv(), timeout=60)
            except asyncio.TimeoutError:
                print("TIMEOUT"); return
            if isinstance(r, (bytes, bytearray)):
                continue
            ev = json.loads(r); t = ev.get("type")
            if t == "response.function_call_arguments.done":
                rounds += 1
                name = ev.get("name") or "?"
                print(f"[round {rounds}] args={ev.get('arguments')!r}")
                if rounds >= 8:
                    print(">>> RUNAWAY REPRODUCED from the captured live turn")
                    return
                # mimic the demo's tools, incl. its per-call randomness
                import os as _os
                _fixed = _os.environ["FIXED_WEATHER"]
                args = json.loads(ev.get("arguments") or "{}")
                if "city" in args:
                    out = f"The weather in {args['city']} is {_fixed} (mocked)."
                else:
                    out = f"For {args.get('item')}: 3 units left, low stock (mocked)."
                print(f"    <- {out}")
                await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                    "type": "function_call_output", "call_id": ev.get("call_id"), "output": out}}))
            elif t == "transcription.done":
                print(f"[text] {ev.get('text')!r}")
            elif t == "response.audio.done":
                print(f">>> finished normally after {rounds} tool round(s)"); return
            elif t == "error":
                print("ERROR:", ev); return

asyncio.run(main())

"""Two-turn memory test over /v1/realtime.

Turn 1: "What is the weather in Madrid?"   (tool call -> answer)
Turn 2: "Which city did I just ask about?" (answerable ONLY from history)

Run with HISTORY=0 to prove the test is meaningful (should fail to recall).
"""
import asyncio, base64, json, os, wave
import websockets

WS = "ws://localhost:8091/v1/realtime"
SR = 16000
USE_HISTORY = os.environ.get("HISTORY", "1") == "1"
CFG = json.load(open("exact_tools.json"))

def pcm(path):
    import audioop
    w = wave.open(path)
    raw = w.readframes(w.getnframes())
    if w.getframerate() != SR:
        raw, _ = audioop.ratecv(raw, 2, w.getnchannels(), w.getframerate(), SR, None)
    return raw

TURNS = [("q_raw.wav", "turn1: what's the weather in Madrid?"),
         ("q_follow.wav", "turn2: which city did I just ask about?")]

async def run_turn(ws, audio, history):
    await ws.send(json.dumps({"type": "session.update", "model": "qwen3-omni",
                              "voice": "custom", "tools": CFG["tools"],
                              "instructions": CFG["instructions"]}))
    for past in history:
        await ws.send(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user",
            "content": [{"type": "input_audio", "audio": base64.b64encode(past["pcm"]).decode()}]}}))
        await ws.send(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": past["text"]}]}}))
    await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))
    step = SR * 2 * 200 // 1000
    for i in range(0, len(audio), step):
        await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                  "audio": base64.b64encode(audio[i:i+step]).decode()}))
    await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

    reply, rounds = [], 0
    while True:
        r = await asyncio.wait_for(ws.recv(), timeout=90)
        if isinstance(r, (bytes, bytearray)):
            continue
        ev = json.loads(r); t = ev.get("type")
        if t == "transcription.delta":
            reply.append(ev.get("delta", ""))
        elif t == "response.function_call_arguments.done":
            rounds += 1
            args = json.loads(ev.get("arguments") or "{}")
            out = (f"The weather in {args['city']} is sunny and 72 degrees (mocked)."
                   if "city" in args else "3 units left, low stock (mocked).")
            await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                "type": "function_call_output", "call_id": ev.get("call_id"), "output": out}}))
        elif t == "response.audio.done":
            return "".join(reply).strip(), rounds
        elif t == "error":
            return f"ERROR {ev}", rounds

async def main():
    history = []
    for path, label in TURNS:
        audio = pcm(path)
        async with websockets.connect(WS, max_size=64*1024*1024) as ws:   # fresh conn per turn, like the worker
            text, rounds = await run_turn(ws, audio, history if USE_HISTORY else [])
        print(f"{label}\n   -> {text!r}  (tool rounds: {rounds})")
        if text and not text.startswith("ERROR"):
            history.append({"pcm": audio, "text": text})
    ok = "madrid" in (history[-1]["text"].lower() if len(history) > 1 else "")
    print(f"\nHISTORY={'on' if USE_HISTORY else 'OFF'}  ->  recalled Madrid: {ok}")

asyncio.run(main())

"""Does the model emit several <tool_call> blocks in ONE generation (parallel), or one
per generation (serial)? Measures calls-per-phase and wall-clock for the whole turn.
Tools sleep 1.5s like the demo's, so serial vs parallel is visible in the timing."""
import asyncio, base64, json, os, time, wave, audioop
import websockets
WS="ws://localhost:8091/v1/realtime"; SR=16000
CFG=json.load(open("exact_tools.json"))
PARALLEL_HINT=(" If a request needs several lookups, emit ALL the tool calls together in a "
 "single reply rather than one at a time, then summarise once all results are back.")

def pcm(p):
    w=wave.open(p); raw=w.readframes(w.getnframes())
    if w.getframerate()!=SR: raw,_=audioop.ratecv(raw,2,w.getnchannels(),w.getframerate(),SR,None)
    return raw
AUDIO=pcm("q_two.wav")

async def run(instructions, label):
    t0=time.time(); phases=[]; text=[]
    async with websockets.connect(WS, max_size=64*1024*1024) as ws:
        await ws.send(json.dumps({"type":"session.update","model":"qwen3-omni","voice":"custom",
                                  "tools":CFG["tools"],"instructions":instructions}))
        await ws.send(json.dumps({"type":"input_audio_buffer.commit","final":False}))
        step=SR*2*200//1000
        for i in range(0,len(AUDIO),step):
            await ws.send(json.dumps({"type":"input_audio_buffer.append","audio":base64.b64encode(AUDIO[i:i+step]).decode()}))
        await ws.send(json.dumps({"type":"input_audio_buffer.commit","final":True}))
        cur=[]
        while True:
            try:
                ev=json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
            except asyncio.TimeoutError:
                break
            t=ev.get("type")
            if t=="transcription.delta": text.append(ev.get("delta",""))
            elif t=="response.function_call_arguments.done":
                args=json.loads(ev.get("arguments") or "{}")
                cur.append((ev.get("call_id"), args.get("city")))
            elif t=="response.audio.done":
                break
            elif t=="error":
                break
            # when the server stops emitting calls and blocks, answer them all at once
            if cur and len(cur) >= 1:
                await asyncio.sleep(0.35)          # let any sibling calls in the same reply arrive
                drained=[]
                while True:
                    try:
                        ev2=json.loads(await asyncio.wait_for(ws.recv(), timeout=0.4))
                    except asyncio.TimeoutError:
                        break
                    if ev2.get("type")=="response.function_call_arguments.done":
                        a=json.loads(ev2.get("arguments") or "{}")
                        cur.append((ev2.get("call_id"), a.get("city")))
                    elif ev2.get("type")=="transcription.delta":
                        text.append(ev2.get("delta",""))
                phases.append(len(cur))
                await asyncio.gather(*[
                    ws.send(json.dumps({"type":"conversation.item.create","item":{
                        "type":"function_call_output","call_id":cid,
                        "output":(f"The weather in {city} is sunny and 72 degrees (mocked)."
                                  if city else "12 spoons in stock (mocked).")}}))
                    for cid, city in cur])
                cur=[]
    return phases, time.time()-t0, "".join(text).strip()

async def main():
    base=CFG["instructions"]
    for label, instr in (("serial (current)", base), ("parallel hint", base + PARALLEL_HINT)):
        phases, dt, txt = await run(instr, label)
        print(f"{label:<18} phases={phases} (calls per generation)  total={dt:.1f}s")
        print(f"{'':<18} -> {txt[:110]!r}\n")
asyncio.run(main())

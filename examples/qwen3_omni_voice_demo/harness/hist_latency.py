"""How much does replaying N prior turns cost in wall-clock time per turn?
Context is cheap; the question is the per-turn audio re-encode (prefix caching off)."""
import asyncio, base64, json, time, wave, audioop
import websockets

WS="ws://localhost:8091/v1/realtime"; SR=16000
CFG=json.load(open("exact_tools.json"))
def pcm(p):
    w=wave.open(p); raw=w.readframes(w.getnframes())
    if w.getframerate()!=SR: raw,_=audioop.ratecv(raw,2,w.getnchannels(),w.getframerate(),SR,None)
    return raw
CUR=pcm("q_follow.wav"); PAST=pcm("q_raw.wav")
REPLY="It's currently sunny and about 72 degrees in Madrid right now."

async def timed(n):
    async with websockets.connect(WS, max_size=64*1024*1024) as ws:
        await ws.send(json.dumps({"type":"session.update","model":"qwen3-omni","voice":"custom",
                                  "tools":CFG["tools"],"instructions":CFG["instructions"]}))
        for _ in range(n):
            await ws.send(json.dumps({"type":"conversation.item.create","item":{"type":"message","role":"user",
                "content":[{"type":"input_audio","audio":base64.b64encode(PAST).decode()}]}}))
            await ws.send(json.dumps({"type":"conversation.item.create","item":{"type":"message","role":"assistant",
                "content":[{"type":"text","text":REPLY}]}}))
        t0=time.time()
        await ws.send(json.dumps({"type":"input_audio_buffer.commit","final":False}))
        step=SR*2*200//1000
        for i in range(0,len(CUR),step):
            await ws.send(json.dumps({"type":"input_audio_buffer.append","audio":base64.b64encode(CUR[i:i+step]).decode()}))
        await ws.send(json.dumps({"type":"input_audio_buffer.commit","final":True}))
        first=None
        while True:
            ev=json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
            t=ev.get("type")
            if t in ("transcription.delta","response.audio.delta") and first is None:
                first=time.time()-t0
            if t=="response.audio.done":
                return first, time.time()-t0

async def main():
    print(f"{'history turns':>13} {'first output':>13} {'total':>8}")
    for n in (0,5,10,20):
        f,tot=await timed(n)
        print(f"{n:>13} {f:>12.2f}s {tot:>7.2f}s")
asyncio.run(main())

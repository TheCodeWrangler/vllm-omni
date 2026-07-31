"""Prefix-cache correctness: identical prompt STRUCTURE, different history AUDIO.
If the cache keyed only on tokens (not audio), turn 2 would leak the wrong city."""
import asyncio, base64, json, wave, audioop
import websockets
WS="ws://localhost:8091/v1/realtime"; SR=16000
CFG=json.load(open("exact_tools.json"))
def pcm(p):
    w=wave.open(p); raw=w.readframes(w.getnframes())
    if w.getframerate()!=SR: raw,_=audioop.ratecv(raw,2,w.getnchannels(),w.getframerate(),SR,None)
    return raw
FOLLOW=pcm("q_follow.wav")          # "which city did I just ask about?"
CASES=[("Madrid","q_raw.wav"), ("Portland","q_pdx.wav")]

async def ask(hist_wav, reply_text):
    async with websockets.connect(WS, max_size=64*1024*1024) as ws:
        await ws.send(json.dumps({"type":"session.update","model":"qwen3-omni","voice":"custom",
                                  "tools":CFG["tools"],"instructions":CFG["instructions"]}))
        await ws.send(json.dumps({"type":"conversation.item.create","item":{"type":"message","role":"user",
            "content":[{"type":"input_audio","audio":base64.b64encode(pcm(hist_wav)).decode()}]}}))
        await ws.send(json.dumps({"type":"conversation.item.create","item":{"type":"message","role":"assistant",
            "content":[{"type":"text","text":reply_text}]}}))
        await ws.send(json.dumps({"type":"input_audio_buffer.commit","final":False}))
        step=SR*2*200//1000
        for i in range(0,len(FOLLOW),step):
            await ws.send(json.dumps({"type":"input_audio_buffer.append","audio":base64.b64encode(FOLLOW[i:i+step]).decode()}))
        await ws.send(json.dumps({"type":"input_audio_buffer.commit","final":True}))
        out=[]
        while True:
            ev=json.loads(await asyncio.wait_for(ws.recv(), timeout=120)); t=ev.get("type")
            if t=="transcription.delta": out.append(ev.get("delta",""))
            elif t=="response.audio.done": return "".join(out).strip()
            elif t=="error": return f"ERROR {ev}"

async def main():
    ok=True
    for city, wav in CASES:
        # identical assistant text in both cases, so ONLY the audio differs
        r = await ask(wav, "It's sunny and 72 degrees there.")
        hit = city.lower() in r.lower()
        other = [c for c,_ in CASES if c!=city][0]
        leak = other.lower() in r.lower()
        print(f"history audio = {city:8s} -> {r!r}")
        print(f"   correct city: {hit}   leaked '{other}': {leak}")
        ok = ok and hit and not leak
    print(f"\nCACHE CORRECTNESS: {'PASS' if ok else 'FAIL'}")
asyncio.run(main())

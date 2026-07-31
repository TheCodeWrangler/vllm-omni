"""Capture the audio the talker produces DURING a tool-call turn (normally
suppressed) and save it, so we can hear whether it's the spoken filler or the
JSON read aloud."""
import asyncio, base64, json, wave, audioop
import websockets
WS="ws://localhost:8091/v1/realtime"; SR=16000
CFG=json.load(open("exact_tools.json"))
def pcm(p):
    w=wave.open(p); raw=w.readframes(w.getnframes())
    if w.getframerate()!=SR: raw,_=audioop.ratecv(raw,2,w.getnchannels(),w.getframerate(),SR,None)
    return raw
AUDIO=pcm("q_pdx.wav")
INSTR=(CFG["instructions"] + " IMPORTANT: whenever you are about to use a tool, FIRST say a short "
       "spoken filler sentence out loud such as 'Let me check that for you.' and only "
       "AFTER saying it emit the tool call.")

async def main():
    async with websockets.connect(WS, max_size=64*1024*1024) as ws:
        await ws.send(json.dumps({"type":"session.update","model":"qwen3-omni","voice":"custom",
                                  "tools":CFG["tools"],"instructions":INSTR}))
        await ws.send(json.dumps({"type":"input_audio_buffer.commit","final":False}))
        step=SR*2*200//1000
        for i in range(0,len(AUDIO),step):
            await ws.send(json.dumps({"type":"input_audio_buffer.append","audio":base64.b64encode(AUDIO[i:i+step]).decode()}))
        await ws.send(json.dumps({"type":"input_audio_buffer.commit","final":True}))
        phase1, phase2, tool_seen, sr_out = [], [], False, 24000
        while True:
            ev=json.loads(await asyncio.wait_for(ws.recv(), timeout=120)); t=ev.get("type")
            if t=="response.audio.delta":
                sr_out=int(ev.get("sample_rate_hz") or 24000)
                (phase2 if tool_seen else phase1).append(base64.b64decode(ev["audio"]))
            elif t=="response.function_call_arguments.done":
                tool_seen=True
                await ws.send(json.dumps({"type":"conversation.item.create","item":{
                    "type":"function_call_output","call_id":ev.get("call_id"),
                    "output":"The weather in Portland is raining lightly (mocked)."}}))
            elif t=="response.audio.done": break
            elif t=="error": print("ERR",ev); break
        for name, chunks in (("toolphase", phase1), ("answerphase", phase2)):
            if not chunks: print(f"{name}: NO AUDIO"); continue
            data=b"".join(chunks)
            with wave.open(f"{name}.wav","wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr_out); w.writeframes(data)
            print(f"{name}.wav  {len(data)/2/sr_out:.2f}s @ {sr_out}Hz")
asyncio.run(main())

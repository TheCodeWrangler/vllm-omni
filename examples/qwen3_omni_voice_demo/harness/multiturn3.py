"""Multi-turn attribution test.

Each turn asks about a DIFFERENT city while history accumulates, so the tool-call
arguments reveal which audio the model actually attended to. If turn 4 calls
get_weather("Madrid") when we asked about Paris, it answered a REPLAYED turn.

env:
  MAXH=<n>   history turns to replay (0 = none)
"""
import asyncio, base64, json, os, wave, audioop
import websockets

WS="ws://localhost:8091/v1/realtime"; SR=16000
CFG=json.load(open("exact_tools.json"))
MAXH=int(os.environ.get("MAXH","20"))

TURNS=[("Madrid","q_raw.wav"),("Portland","q_pdx.wav"),("Tokyo","q_tokyo.wav"),
       ("Paris","q_paris.wav"),("Denver","q_denver.wav")]

def pcm(p):
    w=wave.open(p); raw=w.readframes(w.getnframes())
    if w.getframerate()!=SR: raw,_=audioop.ratecv(raw,2,w.getnchannels(),w.getframerate(),SR,None)
    return raw

async def one_turn(audio, history):
    async with websockets.connect(WS, max_size=64*1024*1024) as ws:
        await ws.send(json.dumps({"type":"session.update","model":"qwen3-omni","voice":"custom",
                                  "tools":CFG["tools"],"instructions":CFG["instructions"]}))
        # history is now a flat message list, like the real client sends
        if MAXH>0:
            uidx=[i for i,m in enumerate(history) if m["role"]=="user"]
            replay = history if len(uidx)<=MAXH else history[uidx[-MAXH]:]
        else:
            replay = []
        for past in replay:
            if past["role"]=="user":
                content=[{"type":"input_audio","audio":base64.b64encode(past["pcm"]).decode()}]
            else:
                content=[{"type":"text","text":past["text"]}]
            await ws.send(json.dumps({"type":"conversation.item.create",
                "item":{"type":"message","role":past["role"],"content":content}}))
        await ws.send(json.dumps({"type":"input_audio_buffer.commit","final":False}))
        step=SR*2*200//1000
        for i in range(0,len(audio),step):
            await ws.send(json.dumps({"type":"input_audio_buffer.append","audio":base64.b64encode(audio[i:i+step]).decode()}))
        await ws.send(json.dumps({"type":"input_audio_buffer.commit","final":True}))
        text=[]; calls=[]; MAXR=4
        while True:
            ev=json.loads(await asyncio.wait_for(ws.recv(), timeout=120)); t=ev.get("type")
            if t=="transcription.delta": text.append(ev.get("delta",""))
            elif t=="response.function_call_arguments.done":
                a=json.loads(ev.get("arguments") or "{}"); calls.append(a)
                if len(calls)>MAXR:
                    return "ERROR runaway: >%d tool rounds" % MAXR, calls, len(replay)
                city=a.get("city","?")
                await ws.send(json.dumps({"type":"conversation.item.create","item":{
                    "type":"function_call_output","call_id":ev.get("call_id"),
                    "output":f"The weather in {city} is sunny and 72 degrees (mocked)."}}))
            elif t=="response.audio.done": return "".join(text).strip(), calls, len(replay)
            elif t=="error": return f"ERROR {ev}", calls, len(replay)

async def main():
    print(f"MAXH={MAXH}\n{'turn':>4} {'asked':>9} {'replayed':>9} {'tool args':>26}  verdict / reply")
    hist=[]; fails=0
    for i,(city,wav) in enumerate(TURNS, start=1):
        audio=pcm(wav)
        text, calls, nrep = await one_turn(audio, hist)
        called=[c.get("city") for c in calls if "city" in c]
        ok = bool(called) and all(c and c.lower()==city.lower() for c in called)
        # also flag if the spoken answer names a DIFFERENT city we asked earlier
        stale=[c for c,_ in TURNS[:i-1] if c.lower() in text.lower() and c.lower()!=city.lower()]
        verdict = "OK" if ok and not stale else ("WRONG-CITY" if not ok else f"STALE:{stale}")
        if verdict!="OK": fails+=1
        print(f"{i:>4} {city:>9} {nrep:>9} {str(called):>26}  {verdict}  {text[:70]!r}")
        if text and not text.startswith("ERROR"):
            hist.append({"role":"user","pcm":audio})
            for c in called:
                hist.append({"role":"assistant",
                    "text":'<tool_call>\n{"name": "get_weather", "arguments": {"city": "%s"}}\n</tool_call>' % c})
                hist.append({"role":"tool","text":f"The weather in {c} is sunny and 72 degrees (mocked)."})
            hist.append({"role":"assistant","text":text})
    print(f"\nFAILURES: {fails}/{len(TURNS)}")
asyncio.run(main())

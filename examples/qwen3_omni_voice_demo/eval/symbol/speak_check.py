"""Speak a subset through the live stack and transcribe back, to see which untouched
categories actually need help vs which the model already handles."""
import base64, json, sys, urllib.request
URL="http://127.0.0.1:8091/v1/chat/completions"
def call(b,t=300):
    r=urllib.request.Request(URL,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=t))
def hear(text):
    r=call({"model":"qwen3-omni","modalities":["text","audio"],"speaker":"custom",
            "messages":[{"role":"user","content":f'Say exactly this and nothing else: "{text}"'}]})
    b64=next((c["message"]["audio"]["data"] for c in r["choices"] if (c["message"] or {}).get("audio")),None)
    if not b64: return "(no audio)"
    r2=call({"model":"qwen3-omni","modalities":["text"],"max_tokens":110,
             "messages":[{"role":"user","content":[
                 {"type":"input_audio","input_audio":{"data":b64,"format":"wav"}},
                 {"type":"text","text":"Transcribe verbatim. Output only the transcript."}]}]})
    return next((c["message"]["content"] for c in r2["choices"] if (c["message"] or {}).get("content")),"").strip()

SUBSET=[
 ("shorthand","Patient is s/p appendectomy."),
 ("shorthand","Admitted to r/o MI, c/o chest pain."),
 ("shorthand","Discharged w/o complications."),
 ("units","Dopamine 5 mcg/kg/min."),
 ("marker","Na+ 138, K+ 4.1."),
 ("marker","CD4+ count 350."),
 ("marker","HCO3- is 24."),
 ("grading","2+ pitting edema."),
 ("range","Goal potassium 3.5-5.0."),
 ("time","Next dose at 08:30."),
 ("time","Ceftriaxone q8h."),
 ("route","Metoprolol 25 mg PO BID."),
 ("roman","NYHA class III heart failure."),
 ("drug","Start hydrochlorothiazide 25 mg daily."),
]
print(f"{'cat':<10} {'raw':<40} spoken")
for cat, raw in SUBSET:
    print(f"{cat:<10} {raw[:40]:<40} {hear(raw)[:70]!r}", flush=True)

"""Speak clinical phrases containing symbols, transcribe the audio back, and check
whether the symbol survived as words. Same model does TTS and ASR, which is fine
here: we only ask whether the SPOKEN form contains the expected words."""
import base64, json, sys, urllib.request

URL="http://127.0.0.1:8091/v1/chat/completions"
VOICE="custom"
# (phrase, [acceptable spoken forms], label)
CASES=[
 ("Blood pressure is 120/80.",              ["over"],                                   "slash /"),
 ("Hold if creatinine is <= 1.2.",          ["less than or equal"],                     "<="),
 ("Saturation is < 88.",                    ["less than"],                              "<  (silent-drop risk)"),
 ("Target oxygen is >= 90.",                ["greater than or equal"],                  ">="),
 ("Weight change is +/- 2 kilograms.",      ["plus or minus", "plus minus"],            "+/-"),
 ("The result is != normal.",               ["not equal"],                              "!="),
 ("Dose is ~ 5 milligrams.",                ["approximately","about","around","tilde"], "~"),
 ("Temperature is 38.5 degrees C.",         ["degrees"],                                "degrees (control)"),
]

def call(body, timeout=300):
    req=urllib.request.Request(URL, data=json.dumps(body).encode(),
                               headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def speak(text):
    r=call({"model":"qwen3-omni","modalities":["text","audio"],"speaker":VOICE,
            "messages":[{"role":"user","content":
                f'Say exactly this and nothing else: "{text}"'}]})
    for c in r["choices"]:
        au=(c["message"] or {}).get("audio")
        if au and au.get("data"): return au["data"]
    return None

def transcribe(b64):
    r=call({"model":"qwen3-omni","modalities":["text"],"max_tokens":120,
            "messages":[{"role":"user","content":[
                {"type":"input_audio","input_audio":{"data":b64,"format":"wav"}},
                {"type":"text","text":"Transcribe verbatim. Output only the transcript."}]}]})
    for c in r["choices"]:
        t=(c["message"] or {}).get("content")
        if t: return t.strip()
    return ""

label=sys.argv[1] if len(sys.argv)>1 else "run"
print(f"=== {label} (voice={VOICE}) ===")
print(f"{'symbol':<26} {'ok':<4} spoken")
score=0
for text, expects, sym in CASES:
    b64=speak(text)
    if not b64:
        print(f"{sym:<26} {'ERR':<4} (no audio)"); continue
    heard=transcribe(b64).lower()
    ok=any(e in heard for e in expects)
    score+=ok
    print(f"{sym:<26} {'PASS' if ok else 'FAIL':<4} {heard[:78]!r}")
print(f"\nSCORE: {score}/{len(CASES)}")

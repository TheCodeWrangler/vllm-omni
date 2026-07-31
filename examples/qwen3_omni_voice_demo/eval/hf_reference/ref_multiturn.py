"""Validate the canonical multi-turn + tool-call history shape on the HF reference
path, independent of vLLM. Three turns, each about a DIFFERENT city, history built
the way Qwen's chat template expects (tool_call + tool_response preserved).

Pass = every turn calls the tool for the CURRENT city and never invents weather.
"""
import _env  # noqa: F401
import json, re, os
import numpy as np, soundfile as sf, torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

MODEL="/home/thecodewrangler/omni_custom_build/qwen-omni-custom-voice-v10b"
SCR="/tmp/claude-0/-home-thecodewrangler/e4ec7e8d-30a5-4321-97ec-902f755a4dd1/scratchpad"
CFG=json.load(open(f"{SCR}/exact_tools.json"))
DEVICE=os.environ.get("OMNI_DEVICE","cuda:4")
KEEP=os.environ.get("KEEP_TOOL_EXCHANGE","1")=="1"
TURNS=[("Madrid","q_raw.wav"),("Portland","q_pdx.wav"),("Tokyo","q_tokyo.wav")]
MOCK="sunny and 72 degrees"

print("loading reference model...", flush=True)
model=Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEVICE).eval()
proc=Qwen3OmniMoeProcessor.from_pretrained(MODEL)
TOOL_RE=re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
print(f"loaded. KEEP_TOOL_EXCHANGE={KEEP}", flush=True)

def gen(msgs, audios):
    text=proc.apply_chat_template(msgs, tools=CFG["tools"], add_generation_prompt=True, tokenize=False)
    kw=dict(text=text, return_tensors="pt", padding=True)
    if audios: kw.update(audio=audios, sampling_rate=16000)
    inp=proc(**kw)
    inp={k:(v.to(DEVICE).to(torch.bfloat16) if hasattr(v,"to") and v.is_floating_point() else
            (v.to(DEVICE) if hasattr(v,"to") else v)) for k,v in inp.items()}
    with torch.no_grad():
        out=model.generate(**inp, return_audio=False, do_sample=False, max_new_tokens=160)
    ids=out[0] if not isinstance(out,(tuple,list)) else out[0][0]
    return proc.batch_decode([ids[inp["input_ids"].shape[1]:]], skip_special_tokens=False)[0].replace("<|im_end|>","").strip()

hist=[]; audios=[]; fails=0
for i,(city,wav) in enumerate(TURNS, start=1):
    a,sr=sf.read(f"{SCR}/{wav}", dtype="float32")
    if a.ndim>1: a=a.mean(axis=1)
    if sr!=16000:
        n=int(len(a)/sr*16000); a=np.interp(np.linspace(0,len(a)/sr,n), np.linspace(0,len(a)/sr,len(a)), a).astype(np.float32)
    audios.append(a)
    msgs=[{"role":"system","content":CFG["instructions"]}]+hist+[{"role":"user","content":"<|audio_start|><|audio_pad|><|audio_end|>"}]
    gen1=gen(msgs, list(audios))
    m=TOOL_RE.search(gen1)
    if not m:
        print(f"turn {i} asked {city:9s}: NO TOOL CALL -> {gen1[:90]!r}"); fails+=1
        hist += [{"role":"user","content":"<|audio_start|><|audio_pad|><|audio_end|>"},
                 {"role":"assistant","content":gen1}]
        continue
    call=json.loads(m.group(1)); got=(call.get("arguments") or {}).get("city")
    result=f"The weather in {got} is {MOCK} (mocked)."
    msgs2=msgs+[{"role":"assistant","content":gen1},{"role":"tool","content":result}]
    final=gen(msgs2, list(audios))
    ok = got and got.lower()==city.lower()
    if not ok: fails+=1
    print(f"turn {i} asked {city:9s}: tool={got!r} {'OK' if ok else 'WRONG'} -> {final[:80]!r}", flush=True)
    if KEEP:
        hist += [{"role":"user","content":"<|audio_start|><|audio_pad|><|audio_end|>"},
                 {"role":"assistant","content":gen1},
                 {"role":"tool","content":result},
                 {"role":"assistant","content":final}]
    else:
        hist += [{"role":"user","content":"<|audio_start|><|audio_pad|><|audio_end|>"},
                 {"role":"assistant","content":final}]
print(f"\nFAILURES: {fails}/{len(TURNS)}   (KEEP_TOOL_EXCHANGE={KEEP})")

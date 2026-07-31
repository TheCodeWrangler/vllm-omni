"""Is the tool-call loop a vLLM serving bug or inherent to the weights?

Runs the SAME grafted checkpoint through the reference HF transformers path
(as QwenLM/Qwen3-Omni prescribes) with the SAME audio, SAME tool schema, SAME
instructions and SAME greedy decoding that vLLM used, then drives the identical
multi-round tool exchange. If it loops here too, it's the weights; if it answers,
it points at the vLLM realtime serving path.
"""
import _env  # noqa: F401  -- must precede transformers
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

MODEL = "/home/thecodewrangler/omni_custom_build/qwen-omni-custom-voice"
WAV = "/tmp/omni_turn_dumps/turn-847ab17d.wav"          # the real failing turn
CFG = json.load(open("/tmp/claude-0/-home-thecodewrangler/e4ec7e8d-30a5-4321-97ec-902f755a4dd1/scratchpad/exact_tools.json"))
TOOLS, INSTRUCTIONS = CFG["tools"], CFG["instructions"]
DEVICE = os.environ.get("OMNI_DEVICE", "cuda:4")
WEATHER = os.environ["FIXED_WEATHER"]
MAX_ROUNDS = 6

audio, sr = sf.read(WAV, dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
print(f"audio: {len(audio)/sr:.2f}s @ {sr}Hz   |   tool result: {WEATHER!r}", flush=True)

print("loading checkpoint (reference HF path)...", flush=True)
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map=DEVICE)
model.eval()
proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
print("loaded.", flush=True)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

conv = [
    {"role": "system", "content": [{"type": "text", "text": INSTRUCTIONS}]},
    {"role": "user", "content": [{"type": "audio", "audio": WAV}]},
]

for rnd in range(1, MAX_ROUNDS + 1):
    text = proc.apply_chat_template(conv, tools=TOOLS, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, audio=[audio], sampling_rate=sr, return_tensors="pt", padding=True)
    # Cast float tensors (audio features) to the model dtype; leave int tensors
    # (input_ids / attention_mask / grid indices) as-is.
    def _mv(v):
        if not hasattr(v, "to"):
            return v
        v = v.to(DEVICE)
        return v.to(torch.bfloat16) if v.is_floating_point() else v

    inputs = {k: _mv(v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, return_audio=False, do_sample=False,  # greedy, like vLLM temp=0.0
                             max_new_tokens=256)
    ids = out[0] if not isinstance(out, (tuple, list)) else out[0][0]
    gen = proc.batch_decode([ids[inputs["input_ids"].shape[1]:]], skip_special_tokens=False)[0]
    gen = gen.replace("<|im_end|>", "").strip()

    m = TOOL_CALL_RE.search(gen)
    if not m:
        print(f"[round {rnd}] ANSWERED: {gen[:160]!r}", flush=True)
        print(">>> reference impl TERMINATED normally")
        sys.exit(0)

    call = json.loads(m.group(1))
    print(f"[round {rnd}] tool_call: {call}", flush=True)
    city = (call.get("arguments") or {}).get("city", "?")
    result = (f"The weather in {city} is {WEATHER} (mocked)."
              if call.get("name") == "get_weather"
              else f"For {(call.get('arguments') or {}).get('item')}: 3 units left, low stock (mocked).")
    conv.append({"role": "assistant", "content": [{"type": "text", "text": gen}]})
    conv.append({"role": "tool", "content": [{"type": "text", "text": result}]})
    print(f"    <- {result}", flush=True)

print(f">>> LOOPED in the reference impl too: {MAX_ROUNDS} tool calls, never answered")

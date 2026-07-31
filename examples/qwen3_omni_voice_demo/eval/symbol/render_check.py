"""Render every case through the checkpoint's ACTUAL template (via AutoProcessor) and
show what text the model will receive. Catches mis-expansions with zero model calls."""
import _env  # noqa: F401
import sys
from transformers import AutoProcessor
sys.path.insert(0, ".")
from cases import CASES

M = "/home/thecodewrangler/omni_custom_build/qwen-omni-custom-voice-v10b"
proc = AutoProcessor.from_pretrained(M)

def render(text):
    out = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": text}]}],
        add_generation_prompt=True, tokenize=False)
    body = out.split("<|im_start|>user")[-1].split("<|im_end|>")[0]
    return " ".join(body.split())

print(f"{'cat':<10} {'raw':<42} rendered")
print("-" * 118)
bad = []
for cat, raw, _ in CASES:
    r = render(raw)
    changed = r != raw
    flag = " " if changed else "."          # '.' = template left it untouched
    print(f"{cat:<10} {raw[:42]:<42} {flag}{r[:62]}")
    # heuristic: clinical shorthand should NOT become "x over y"
    if cat == "shorthand" and " over " in r:
        bad.append((raw, r))
print("-" * 118)
if bad:
    print("\n*** MIS-EXPANDED clinical shorthand (the '/' rule firing where it must not) ***")
    for raw, r in bad:
        print(f"  {raw!r}\n    -> {r!r}")

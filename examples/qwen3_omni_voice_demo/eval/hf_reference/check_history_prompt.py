"""CPU-only: does the history prompt render as real interleaved turns, with the
audio placeholder count matching the audio list length?"""
import _env  # noqa: F401
import json
from transformers import AutoTokenizer, Qwen3OmniMoeProcessor

MODEL = "/home/thecodewrangler/omni_custom_build/qwen-omni-custom-voice-v10b"
CFG = json.load(open("/tmp/claude-0/-home-thecodewrangler/e4ec7e8d-30a5-4321-97ec-902f755a4dd1/scratchpad/exact_tools.json"))
proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
PH = "<|audio_start|><|audio_pad|><|audio_end|>"

def build(history, tools):
    msgs = []
    for past in history:
        msgs.append({"role": "user", "content": PH})
        msgs.append({"role": "assistant", "content": past["text"]})
    msgs.append({"role": "user", "content": PH})
    return proc.apply_chat_template(msgs, tools=tools or None,
                                    add_generation_prompt=True, tokenize=False)

for n, tools in ((0, None), (2, CFG["tools"])):
    hist = [{"text": f"reply number {i+1}"} for i in range(n)]
    p = build(hist, tools)
    n_ph = p.count("<|audio_pad|>")
    print(f"history={n} tools={'yes' if tools else 'no'}: "
          f"placeholders={n_ph} (expect {n+1})  ok={n_ph == n+1}  tokens={len(tok.encode(p))}")

print("\n=== 2-turn prompt, turn structure ===")
p = build([{"text": "It's sunny and 72 degrees in Madrid."}], None)
for line in p.split("<|im_start|>"):
    if line.strip():
        print("  <|im_start|>" + line.replace("\n", "\\n")[:96])

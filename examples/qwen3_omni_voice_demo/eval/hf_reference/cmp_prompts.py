"""Diff the continuation prompt vLLM builds (raw token splice) against what the
chat template produces for the same logical conversation (what HF sends)."""
import _env  # noqa: F401
import json, difflib
from transformers import Qwen3OmniMoeProcessor

MODEL = "/home/thecodewrangler/omni_custom_build/qwen-omni-custom-voice"
CFG = json.load(open("/tmp/claude-0/-home-thecodewrangler/e4ec7e8d-30a5-4321-97ec-902f755a4dd1/scratchpad/exact_tools.json"))
TOOLS, INSTR = CFG["tools"], CFG["instructions"]
proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)

TOOL_CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Portland"}}\n</tool_call>'
RESULT = "The weather in Portland is overcast with a light breeze (mocked)."

sys_user = [{"role": "system", "content": [{"type": "text", "text": INSTR}]},
            {"role": "user", "content": [{"type": "audio", "audio": "x.wav"}]}]

# what vLLM starts from: template(system+user_audio) ending in <|im_start|>assistant
base = proc.apply_chat_template(sys_user, tools=TOOLS, add_generation_prompt=True, tokenize=False)
# vLLM's tool-result suffix (safe_apply_chat_template on a lone tool message)
suffix = proc.apply_chat_template([{"role": "tool", "content": [{"type": "text", "text": RESULT}]}],
                                  add_generation_prompt=True, tokenize=False)
vllm_style = base + TOOL_CALL + suffix

# what HF/the template produces for the full conversation
full = sys_user + [{"role": "assistant", "content": [{"type": "text", "text": TOOL_CALL}]},
                   {"role": "tool", "content": [{"type": "text", "text": RESULT}]}]
hf_style = proc.apply_chat_template(full, tools=TOOLS, add_generation_prompt=True, tokenize=False)

print("=== identical? ", vllm_style == hf_style)
print("\n=== vLLM-style TAIL (last 320 chars) ===")
print(repr(vllm_style[-320:]))
print("\n=== HF/template TAIL (last 320 chars) ===")
print(repr(hf_style[-320:]))
print("\n=== unified diff ===")
for line in difflib.unified_diff(vllm_style.splitlines(), hf_style.splitlines(),
                                 "vllm_style", "hf_style", lineterm="", n=1):
    print(line)

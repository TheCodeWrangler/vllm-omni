# Qwen3-Omni speech-to-speech voice demo (local, with tool calling)

Snapshot of a **working** local voice agent: LiveKit browser client → this repo's
`/v1/realtime` endpoint → Qwen3-Omni (thinker → talker → code2wav), with tool
calling, multi-turn memory, a grafted custom voice, and speak-ready output.

This branch (`local-demo-combined`) is **not** intended to be merged. It exists to
record the exact combination of server changes the demo was verified against,
because that combination is spread across four separate upstream PRs that each
land cleanly alone but **conflict with each other**.

## The four upstream PRs behind this

All four are based on `main` and all four edit
`vllm_omni/entrypoints/openai/realtime_connection.py` and
`vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py`.

| PR | Branch | What it adds |
|----|--------|--------------|
| #5555 | `feat/realtime-tool-calling` | Hermes-style `<tool_call>` parsing + function-call events; bounded tool-call chains |
| #5565 | `pr2-voice-selection` | `voice`/`speaker` on `session.update`, threaded through `render_cmpl_async` |
| #5566 | `pr3-instructions` | `instructions` (system prompt) support |
| #5655 | `feat/realtime-audio-history` | Conversation history: replays prior turns' audio + the whole tool exchange |

These are now a **linear stack**, rebased onto current `main` and verified to
land in order:

```
1. #5555 tool-calling   CLEAN onto main
2. #5565 voice          CLEAN onto #5555
3. #5566 instructions   CLEAN onto #5565
4. #5655 history        CLEAN onto #5566
```

They were originally four independent siblings all based on `main`, which merged
cleanly *alone* but conflicted with each other (#5555 landed, then #5565
conflicted in all three files). Restacking fixed that; the merged tree is
byte-identical to this branch's tree modulo the two deltas noted below, and the
unit tests pass at every level (36 -> 44 -> 47 -> 54 tests).

Stacking also surfaced a bug neither PR had alone: with tool calling **and**
voice selection both present, the tool-call continuation builds its own
`TokensPrompt` and so dropped the `speaker`, making the voice audibly change
halfway through a tool-calling turn. Fixed in #5565.

This branch differs from the stack tip by exactly two things: it lacks the
"bound tool-call chains" commit, and it carries the `OMNI_DEBUG_DUMP_DIR` debug
capture, which is deliberately not proposed upstream.

### Bugs found while getting this working

Four real server-side bugs, all in the tool-call/history prompt-building paths:

1. **Continuations dropped the audio** — the post-expansion prompt (already
   containing `<|audio_pad|>` ids) was re-fed with no `multi_modal_data`. Fixed by
   splicing the *pre*-expansion prompt and re-attaching the audio.
2. **Continuations never closed the assistant turn** — produced
   `</tool_call><|im_start|>user` with no `<|im_end|>`, which drove endless
   tool-call loops. Fixed by `_close_assistant_turn`.
3. **History dropped the tool exchange** — replaying only the final assistant text
   made the model re-derive (and fabricate) tool results. Fixed by keeping history
   as a flat message list including `<tool_call>` markup and its result.
4. **`history[-0:]` returns the whole list** — "disable history" would have
   replayed everything. Needed an explicit guard.

Bugs 2 and 3 were both isolated by replaying the same prompt through the
**HF reference implementation** (`eval/hf_reference/`) to prove the behaviour was
prompt-shape, not serving. That technique is the single highest-leverage thing in
this directory.

## Layout

```
client/         LiveKit worker + the /v1/realtime adapter (the client-side half)
  worker.py         demo agent: 2 mocked tools, speak-ready instructions
  omni_realtime.py  livekit-agents RealtimeModel for vllm-omni's /v1/realtime
harness/        headless WS harnesses + audio fixtures (no browser needed)
eval/symbol/    51-case clinical symbol/number pronunciation test set
eval/hf_reference/  replay a prompt through HF reference to isolate serving bugs
```

`client/` is a standalone script, **not** part of any production package.

## Running it

```bash
# server (separate repo/config: qwen-omni-serve, stage 0 needs enable_prefix_caching)
# then:
cd client
BRAIN=omni OMNI_VOICE=custom OMNI_MAX_HISTORY_TURNS=20 \
  OMNI_DEBUG_DUMP_DIR=/tmp/omni_debug python worker.py dev
```

Set `BRAIN=openai` instead to A/B the same agent against `gpt-realtime`.

`OMNI_DEBUG_DUMP_DIR` writes per-turn `turn-<id>.json` / `.wav` /
`.generated.txt` — the fully-expanded prompt for **both** prompt-building paths.
Every bug above was diagnosed by inference across ~2.5-minute container restarts
before this existed; do not debug this pipeline without it.

## Known-good behaviour (verified in the browser)

- Tool calling, **serial and parallel**. Independent calls known upfront go out in
  one generation (three `check_inventory` calls dispatched within 1 ms, ~1.5 s
  total instead of 4.5 s); exploratory fan-out stays serial because each call
  depends on the previous result. Both end in a spoken summary.
- 20-turn memory with prefix caching on stage 0 (cache correctness checked by
  `harness/cache_collision.py`).
- Custom grafted voice (`custom`), plus `custom_slow` at the previous vector.

## Known limitations

- The within-turn tool-retry loop recurred **once** with the fix deployed and was
  never reproduced. It is bounded by `_MAX_TOOL_ROUNDS = 4` plus a forced wrap-up
  directive, so the worst case is a silent turn, not a runaway.
- **Unit and factual accuracy is not solved.** The model reported 37.2 °C as
  "98.6 degrees Fahrenheit". Prompting reduces this but cannot eliminate it —
  clinical values must come from a tool and be read back verbatim.
- Symbol pronunciation is generation-time only (the talker is conditioned on the
  thinker's hidden states, so no downstream text normalisation can reach it).
  Went 1/8 → 7/8 on the eval set via instructions; see `eval/symbol/`.

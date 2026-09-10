# Paying for audio before the recording ends (Qwen3-Omni, audio in / text out)

Status: design + measurements, 2026-09-10. Branch `feat/incremental-audio-prefill`.

## Question

A client records speech (say 20 s), then calls `/v1/chat/completions` with the
whole clip and waits for text. How much of that wait can be moved *into* the
recording window by processing early packets as they arrive, and what endpoint
shape does that need?

Everything below was measured against the deployed image (vLLM 0.24.0 +
vllm-omni 0.24.0, `qwen-omni-a`, 2x H100, stage-0 `enable_prefix_caching: true`,
`modalities: ["text"]`, client on the same host, unique audio per request).
Fixtures: 108 quiet-room clinical dictations with reference transcripts
(30 to 54 s), cropped to 20 s where noted. Script:
`benchmarks/incremental_audio/bench_prefix_priming.py`.

## Answer in one table

End-of-recording latency for a 20 s dictation (median, warm server, local client):

| Arm | What the client does | TTFT | Time to full text | Full-clip WER | Server change |
| --- | --- | --- | --- | --- | --- |
| whole (today) | one `input_audio` with the whole clip | 0.087 s | 0.36 s | 0.028 | none |
| chunked | 8 s `input_audio` parts, sent once at the end | 0.090 s | same | 0.034 | none |
| primed | 8 s parts + a `max_tokens=1` request after each part during recording | 0.058 s | 0.33 s | 0.034 | none |
| incremental | transcribe each ~5 s segment as its own turn during recording | 0.059 s | 0.105 s | 0.047 | none |

n = 16 clips for TTFT arms (two runs), 12 clips for chunked WER, 8 clips for
incremental. WER is on the full 30-54 s clips against the human reference;
the "whole" column is the same model on the same clips.

So:

- Amortising **prefill** (encoder + thinker prompt) is worth about **30 ms**
  on a 20 s clip. After priming, the final request sits at the per-request
  floor (~58 ms), which no amount of earlier audio work can remove.
- Amortising **decode** as well (incremental transcription) is worth about
  **250 ms** (0.36 s to 0.105 s), because the ~57-token transcript at
  ~4.7 ms/token is 75% of today's latency. It costs ~1.4 WER points on these
  clips, almost all at segment boundaries.
- Both are achievable today with **no server change**, from a client that
  sends the audio in pieces during recording.
- The server-side streaming session (`/v1/realtime`) is the right *endpoint
  shape* and removes the re-upload, but on this host its extra latency win
  over the client-side variants is in the tens of milliseconds.

Two things not in the table matter more than the 30 ms:

1. **Upload time.** 20 s of PCM16 at 16 kHz is 640 KB, 853 KB as base64 JSON.
   That is 0.14 s at 50 Mbit/s, 0.34 s at 20 Mbit/s, 1.4 s at 5 Mbit/s. On
   this host it is ~0 and so absent from every number above. Any design that
   streams audio during recording removes it entirely; the priming design
   only removes it if the final request can reference earlier chunks
   without re-sending them (see "UUID references" below).
2. **Sporadic multi-second stalls.** 6 of 200 timed requests (3%) had a TTFT
   between 0.76 s and 5.0 s. The server's own `[OmniTiming]` lines put all
   of them inside stage 0 (thinker engine), with `preprocess=0.00s`. Every
   one was a single-shot request carrying a >=20 s audio item; none of the
   ~130 priming, primed-final or 5 s-segment requests stalled. This is the
   dominant p95 term and is independent of the amortisation work.

## Where the time goes today

TTFT vs clip length, whole clip, unique audio (two rounds):

| clip | prompt tokens | TTFT round 1 | TTFT round 2 |
| --- | --- | --- | --- |
| 4 s | 106 | 0.056 | 0.054 |
| 8 s | 158 | 0.059 | 0.071 |
| 12 s | 210 | 0.113 | 0.083 |
| 16 s | 262 | 0.277 | 0.313 |
| 20 s | 314 | 0.158 | 0.096 |
| 24 s | 366 | 2.296 | 0.096 |
| 32 s | 470 | 2.402 | 0.104 |

Audio costs 13 thinker tokens per second (Whisper-style 100 Hz mel, 8x conv
downsample), plus ~54 tokens of template and prompt. Ignoring the stalls, TTFT
grows from ~55 ms at 4 s to ~100 ms at 32 s: roughly **1.5 ms per second of
audio** on top of a ~55 ms floor. The floor itself is bimodal: a minority of
requests in every arm completed in 20-33 ms (including 20 s whole-clip
requests), the rest in 55-60 ms, which points at a ~30 ms quantum somewhere
between the API server and the stage-0 engine rather than at model compute.
Worth a profile: it is as large as the whole priming gain.

16 s (262 tokens) was slow in both rounds and is the one shape that looked
systematic rather than sporadic. Not investigated.

After the first token, decode runs at ~4.7 ms/token (median 57 completion
tokens for a 20 s dictation = ~0.27 s), which is why the incremental arm wins
by so much more than the primed arm.

## Why early processing is exact for this model

The Qwen3-Omni audio encoder
(`vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py`,
`Qwen3OmniMoeAudioEncoder.forward`) is strictly window-local:

- Input mel features are split into chunks of `n_window * 2 = 100` frames
  (1 s). The conv stack and the positional embedding are applied **per chunk**;
  the positional embedding restarts at 0 in every chunk.
- Attention uses `cu_seqlens` built from `n_window_infer = 800` mel frames
  (8 s, 104 encoder tokens). Attention is block-diagonal over those 8 s
  windows: no token attends outside its window.
- The thinker is causal.

So the encoder output for audio in [0, 8k) seconds does not depend on any
audio after it, provided cuts fall on 8 s multiples. Two sources of
non-bit-identity remain, both negligible for speech: the Whisper feature
extractor clamps the log-mel spectrogram at `max - 8` where `max` is taken
over the whole item, and the STFT reflect-pads the item edges. That is why
`chunked` and `whole` transcripts differ slightly (WER 0.034 vs 0.028) rather
than being identical.

Consequence: chunk at 8 s boundaries and you get encoder results that are the
same as the whole clip would have produced, and thinker prefill that is a pure
prefix of the final prompt. Prefix caching does the rest.

## Option A: client-side priming over `/v1/chat/completions` (no server change)

During recording, every time an 8 s chunk completes, send:

```json
{"model": "qwen3-omni", "max_tokens": 1, "temperature": 0, "modalities": ["text"],
 "messages": [
   {"role": "system", "content": "<same system prompt as the final call>"},
   {"role": "user", "content": [
     {"type": "input_audio", "input_audio": {"data": "<chunk 1>", "format": "wav"}},
     {"type": "input_audio", "input_audio": {"data": "<chunk 2>", "format": "wav"}}
   ]}
 ]}
```

At the end, send the real request with all chunks followed by the text
instruction. Everything up to the last completed chunk hits the stage-0 prefix
cache (block hashes fold in each audio item's content hash), so the final
request only encodes and prefills the tail chunk. Measured: priming requests
take ~57 ms each, and the final request's TTFT drops from 87 ms to 58 ms.

Rules that make it work:

- The text instruction must come **after** the audio parts (or live in the
  system prompt). Anything that changes before the audio invalidates the prefix.
- Same system prompt, same chunking, same order in every priming request.
- Chunk length must be a multiple of 8 s to keep the encoder computation
  identical to the whole clip. 8 s is also small enough that the last chunk
  is at most 8 s of new work.
- Prefix-cache residency is best effort. Under load, blocks from a priming
  request can be evicted before the final request arrives, in which case the
  final request just pays full price. Nothing breaks.
- One decode step per priming request is wasted (the token is discarded).

Cost: N extra HTTP requests and the full audio re-uploaded in the final
request (853 KB for 20 s). Locally free; on a phone uplink it is the
largest term (see above).

### UUID references (would remove the re-upload) — not on the deployed image

vLLM's chat API accepts an optional `"uuid"` on each media part and documents
that a part whose UUID is already cached may omit its data
(`docs/features/multimodal_inputs.md`, "multi_modal_uuids"). That would let the
final request send only the last chunk's bytes and reference the earlier
chunks by ID. Tested on `qwen-omni-a` (vLLM 0.24.0): the request fails with
HTTP 500,

```text
vllm/renderers/hf.py:1138 _process_tokens_async -> assert_never(audio)
AssertionError: Expected code to be unreachable, but got: None
```

i.e. the renderer in 0.24 does not accept a data-less audio item. Retest on
vLLM 0.28 (`vllm/vllm-omni:latest` has 0.28.0) before relying on it.

## Option B: incremental transcription over `/v1/chat/completions` (no server change)

Instead of only warming the cache, transcribe as you go. Each segment becomes a
new user turn on a growing conversation:

```text
system:    "...The dictation arrives in consecutive audio segments, one per user
            message. Reply with the verbatim transcript of ONLY the newest segment,
            continuing seamlessly from your previous replies..."
user:      [audio segment 1]        -> assistant: "The patient is a pleasant adult"
user:      [audio segment 2]        -> assistant: "seen today for follow-up of ..."
...
user:      [audio segment k (last)] -> assistant: "... for routine evaluation."
```

Prefix caching means turn k only prefills segment k plus the previous reply
(~13 tokens) and decodes ~13 tokens. Measured on 20 s clips with ~5 s
segments cut at the quietest 200 ms within +-1.5 s of the nominal boundary:

| | TTFT | time to full text |
| --- | --- | --- |
| whole clip | 0.086 s | 0.355 s |
| last segment of incremental | 0.059 s | 0.105 s |

Across 98 segment requests (20 s and full clips, prompts up to 881 tokens):
TTFT median 0.058 s, p90 0.069 s, max 0.096 s; total median 0.117 s, max
0.180 s. Each segment is finished long before the next one is recorded, so the
per-segment work is fully hidden.

Quality: full-clip WER 0.033 (whole) vs 0.047 (incremental), n = 8; pairs
(whole, incremental): (0.069, 0.103) (0.034, 0.045) (0.034, 0.045)
(0.023, 0.034) (0.023, 0.045) (0.011, 0.022) (0.044, 0.033) (0.023, 0.045).
The errors are what you would expect from no look-ahead: a conjunction cut
in half at a boundary becomes "a" instead of "and", and sentence punctuation
lands at segment ends. Levers not yet tried: longer segments (8 s aligns with
the encoder window and halves the number of boundaries), a short overlap with
an instruction not to repeat, or a non-blocking consolidation pass after the
last segment for the stored transcript while the incremental text is shown
immediately.

## Option C: server-side streaming session on `WS /v1/realtime`

This is the endpoint shape that fits the use case, and most of it already
exists on upstream main.

What exists (`docs/serving/realtime_api.md`, `vllm_omni/entrypoints/openai/realtime_connection.py`,
`Qwen3OmniMoeForConditionalGeneration.buffer_realtime_audio` in
`vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py`):

```json
{"type": "session.update", "model": "qwen3-omni"}
{"type": "input_audio_buffer.commit", "final": false}     // starts the streaming request
{"type": "input_audio_buffer.append", "audio": "<b64 pcm16>"}   // repeat while recording
{"type": "input_audio_buffer.commit", "final": true}
```

Each appended chunk is buffered; in `async_chunk` mode every completed 5 s
segment is yielded as a `TokensPrompt` and becomes a vLLM `StreamingUpdate` on
the *same* engine request. The scheduler (`_update_request_as_session`) keeps
the request's KV blocks, keeps the computed output tokens as prompt, drops the
last sampled token, appends the new prompt tokens and mm features, and
re-queues the request for prefill of just the new tokens. That is precisely
the "prefill early, decode once" primitive, and it is why the incremental arm
in Option B can be reproduced server-side with one upload and one KV session.

What is missing for this use case:

1. **A prompt.** `buffer_realtime_audio` hard-codes
   `<|im_start|>user\n<audio><|im_end|>\n<|im_start|>assistant\n` with no system
   prompt and no text instruction, so the model *answers* the audio rather
   than transcribing it. Needs `session.update` to carry `instructions` (system
   prompt) and an optional trailing user text.
2. **Text-only output.** `_run_generation` uses the YAML default sampling
   params for all three stages, so the talker and code2wav run and audio is
   streamed back. Needs a `modalities: ["text"]` session option that sets
   `final_stage_id=0`, the same thing `/v1/chat/completions` does.
3. **An "accumulate" mode** alongside today's "segment" mode:
   - `segment` (exists): each segment is its own turn with its own reply.
     Equivalent to Option B; end latency ~0.1 s; boundary WER cost.
   - `accumulate` (new): each completed 8 s window is yielded as a prompt
     *fragment* (`<|audio_start|><|audio_pad|>...<|audio_end|>`) with
     `sampling_params.max_tokens = 1`; the scheduler prefills it and discards
     the one sampled token on the next update. The final commit yields the
     remaining audio plus the closing text and template with the real
     sampling params. The connection must suppress `transcription.delta`
     for non-final updates. Equivalent to Option A without the re-upload;
     end latency ~ floor + last-window prefill + full decode (~0.33 s).
   - Verify that the Qwen3-Omni multimodal processor accepts a
     placeholder-only fragment prompt (Qwen3-ASR ships
     `Qwen3ASRRealtimeMultiModalProcessor._maybe_apply_prompt_updates` for a
     related reason), that M-RoPE positions are recomputed over the whole
     accumulated token list (they are: `get_mrope_input_positions` runs on
     `_all_token_ids` with mm offsets rebased), and that the orchestrator does
     not forward the discarded per-update tokens to stage 1.
4. **Segment boundaries.** Today's buffer cuts at fixed 5 s. For `accumulate`
   the cut must be at 8 s multiples (encoder window); for `segment` a
   pause-aligned cut measurably helps and the client cannot do it
   server-side, so the buffer needs a minimum-energy search like
   `split_at_pauses` in the benchmark, or Server VAD.

Expected benefit over A/B on this host: tens of milliseconds (no re-render, no
re-hash, no HTTP per chunk). Real benefit: audio uploaded once while
recording, no 853 KB final POST, state pinned to one request instead of a
best-effort prefix cache, and an OpenAI-realtime-shaped wire protocol.

Cost: changes in `realtime_connection.py`, `buffer_realtime_audio`, the
realtime protocol models, and a compatibility question: upstream main
targets vLLM 0.28 while the deployed image is 0.24, so a prototype needs the
`vllm/vllm-omni:latest` image (0.28.0) with this worktree mounted, on the free
GPUs.

## Recommendation

1. **Now, client only:** send 8 s chunks during recording and prime
   (Option A). ~30 ms TTFT, quality-neutral, zero risk. If the client is on a
   real network, this alone will not remove the final upload; measure upload
   time there before deciding whether that matters.
2. **If ~250 ms matters:** Option B (incremental turns) is available today
   and cuts time-to-full-text from 0.36 s to ~0.1 s. Spend the effort on the
   boundary quality (8 s segments, overlap, or an off-critical-path
   consolidation pass) and re-run the WER comparison on more than 8 clips
   before shipping.
3. **Endpoint work on this branch:** extend `/v1/realtime` with
   `instructions`, `modalities`, and the `accumulate | segment` input mode
   (Option C). It is the durable shape and the only one that fixes upload.
   Prototype against `vllm/vllm-omni:latest` on GPUs 0,1.
4. **Independently, chase the 3% multi-second stage-0 stalls.** They only
   appear on single-shot requests with >=20 s audio items, they are inside the
   engine (not preprocessing), and they cost more p95 latency than everything
   above combined. Candidate causes to test on a second instance: FlashInfer
   MoE autotune on first-seen token counts (`enable_flashinfer_autotune` is on
   and stalls cluster on first-seen shapes but not exclusively), and GC in the
   engine core. A `py-spy record` on the stage-0 PID during a spike-hunt loop
   of unique 20 s clips will settle it.

## Reproduce

```bash
# fixtures: 16 kHz mono WAV + reference text -> meta.json (see script docstring)
python benchmarks/incremental_audio/bench_prefix_priming.py \
  --url http://localhost:8091/v1/chat/completions --fixtures meta.json \
  --mode all --n 8 --out results.jsonl
# incremental arm alone, 5 s pause-aligned segments
python benchmarks/incremental_audio/bench_prefix_priming.py \
  --fixtures meta.json --mode incremental --seg-s 5 --cut pause --n 8 --out results.jsonl
```

Traps that produced wrong numbers before they were caught: repeating audio
(prefix-cache hit, not a measurement); trusting a single run (3% of requests
stall for seconds); and reading `/metrics`, which is unpopulated on the
multi-stage server. Use the `[OmniTiming]` log lines for server-side
attribution instead.

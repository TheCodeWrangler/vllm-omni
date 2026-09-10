# Incremental audio: how much latency can be paid before the recording ends?

`bench_prefix_priming.py` measures, against a stock `/v1/chat/completions`
server with stage-0 prefix caching, four ways of sending a recording to
Qwen3-Omni for audio-in / text-out:

| arm | client behaviour | amortises |
| --- | --- | --- |
| `whole` | one `input_audio` with the entire clip (today) | nothing |
| `chunked` | 8 s `input_audio` parts sent once at the end | nothing (control) |
| `primed` | 8 s parts, plus a `max_tokens=1` request after each part during recording | prefill |
| `incremental` | each ~5 s segment transcribed as its own turn during recording | prefill + decode |

Findings, design options and the endpoint proposal are in
[`docs/design/incremental_audio_prefill.md`](../../docs/design/incremental_audio_prefill.md).

## Fixtures

`--fixtures meta.json` is a list of objects:

```json
[{"file": "clip-01", "full": "/abs/path/clip-01.wav", "duration": 38.2, "ref": "reference transcript ..."}]
```

WAVs must be 16 kHz mono. Clips should be at least 21 s (20 s crops are taken
from the start and the end of each clip). Every timed request uses audio the
server has not seen; a repeat is a prefix-cache hit, not a measurement, so
give the script enough distinct clips (`--skip` lets you continue through a
fixture list across invocations).

## Run

```bash
python bench_prefix_priming.py --fixtures meta.json --mode all --n 8 --out results.jsonl
python bench_prefix_priming.py --fixtures meta.json --mode incremental --seg-s 5 --cut pause --n 8 --skip 30 --out results.jsonl
```

Modes: `sweep` (TTFT vs clip length), `compare` (whole / chunked / primed on
20 s crops), `quality` (whole vs chunked WER on full clips), `incremental`
(latency on 20 s crops and WER on full clips). Results are appended to
`--out` as JSON lines, one per clip, with the transcripts included.

Requires `requests`, `soundfile`, `numpy`, `jiwer`.

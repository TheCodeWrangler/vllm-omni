#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""How much of Qwen3-Omni's audio-in / text-out latency can be paid *before*
the recording ends?

Four arms, all against a stock ``/v1/chat/completions`` server with stage-0
``enable_prefix_caching: true`` (no server changes):

* ``whole``       - one ``input_audio`` part with the entire clip.  Today's client.
* ``chunked``     - the clip split into N-second ``input_audio`` parts, sent once
                    at the end.  Isolates the cost of chunking itself.
* ``primed``      - same chunks, but every time a chunk completes during the
                    "recording" a priming request (``max_tokens=1``) is sent with
                    the chunks so far.  That runs the audio encoder + thinker
                    prefill for those chunks and leaves the result in the prefix
                    cache.  The final request then only pays for the last chunk.
                    Amortises *prefill* only.
* ``incremental`` - each segment is transcribed as it arrives, as a new user
                    turn on a growing conversation (prefix cache makes each turn
                    pay only for the new segment).  The end-of-recording latency
                    is the last segment's request alone.  Amortises prefill *and*
                    decode.  Quality is checked against the reference transcript
                    because segments are transcribed without look-ahead.

Default chunk length is 8 s because the Qwen3-Omni audio encoder's attention
is block-diagonal over 800 mel frames (``n_window_infer``) = 8 s, so chunking
at 8 s multiples does not change which audio each encoder token can see.

Every timing uses *unique* audio.  A repeat of identical audio is a prefix-cache
hit, not a measurement.

Usage:
    python bench_prefix_priming.py --fixtures meta.json --mode all --out results.jsonl
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time

import numpy as np
import requests
import soundfile as sf

SYSTEM = (
    "You are a verbatim medical dictation transcriber. Transcribe the audio exactly as spoken, "
    "with standard punctuation and capitalization. Output only the transcript."
)
SYSTEM_INCREMENTAL = (
    "You are a verbatim medical dictation transcriber. The dictation arrives in consecutive audio "
    "segments, one per user message. Reply with the verbatim transcript of ONLY the newest segment, "
    "continuing seamlessly from your previous replies. A segment may begin or end mid-sentence; "
    "transcribe exactly what is audible and nothing else. Use standard punctuation and "
    "capitalization. Output only the transcript."
)
USER_TEXT = "Transcribe the audio verbatim."
SR = 16000


def wav_b64(x: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def audio_part(x: np.ndarray) -> dict:
    return {"type": "input_audio", "input_audio": {"data": wav_b64(x), "format": "wav"}}


def messages(chunks: list[np.ndarray], with_text: bool) -> list[dict]:
    content: list[dict] = [audio_part(c) for c in chunks]
    if with_text:
        content.append({"type": "text", "text": USER_TEXT})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def call(url: str, model: str, msgs: list[dict], max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": 0,
        "modalities": ["text"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    text = ""
    usage = None
    ntok = 0
    with requests.post(url, json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                c = ch.get("delta", {}).get("content")
                if isinstance(c, str) and c:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text += c
                    ntok += 1
    total = time.perf_counter() - t0
    return {"ttft": ttft, "total": total, "text": text, "usage": usage, "stream_chunks": ntok}


def split(x: np.ndarray, chunk_s: float) -> list[np.ndarray]:
    n = int(chunk_s * SR)
    return [x[i : i + n] for i in range(0, len(x), n)]


def split_at_pauses(x: np.ndarray, seg_s: float, search_s: float = 1.5, win_s: float = 0.2) -> list[np.ndarray]:
    """Cut near multiples of ``seg_s`` at the quietest ``win_s`` window within +-``search_s``.

    A client can do this online: the decision for a cut at ~t needs audio up to t+search_s.
    """
    n = len(x)
    w = int(win_s * SR)
    cuts = [0]
    pos = 0
    while True:
        target = pos + int(seg_s * SR)
        if target >= n - int(0.5 * SR):
            break
        lo = max(pos + int(1.0 * SR), target - int(search_s * SR))
        hi = min(n, target + int(search_s * SR))
        seg = x[lo:hi]
        if len(seg) <= w:
            cut = target
        else:
            energy = np.convolve(seg.astype(np.float64) ** 2, np.ones(w), "valid")
            cut = lo + int(np.argmin(energy)) + w // 2
        cuts.append(cut)
        pos = cut
    cuts.append(n)
    return [x[a:b] for a, b in zip(cuts[:-1], cuts[1:])]


def load(path: str, start_s: float | None = None, dur_s: float | None = None) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, (path, sr)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if start_s is not None:
        x = x[int(start_s * SR) :]
    if dur_s is not None:
        x = x[: int(dur_s * SR)]
    return x


def wer(ref: str, hyp: str) -> float:
    import jiwer

    tr = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )
    return jiwer.wer(ref, hyp, reference_transform=tr, hypothesis_transform=tr)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def run_incremental(url: str, model: str, segs: list[np.ndarray], max_tokens: int) -> dict:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_INCREMENTAL}]
    per_seg = []
    texts = []
    for seg in segs:
        msgs.append({"role": "user", "content": [audio_part(seg)]})
        r = call(url, model, msgs, max_tokens)
        msgs.append({"role": "assistant", "content": r["text"]})
        per_seg.append(
            {
                "ttft": r["ttft"],
                "total": r["total"],
                "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"),
                "completion_tokens": (r["usage"] or {}).get("completion_tokens"),
                "dur": len(seg) / SR,
            }
        )
        texts.append(r["text"].strip())
    return {"per_seg": per_seg, "text": " ".join(texts), "last": per_seg[-1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8091/v1/chat/completions")
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--fixtures", required=True, help="meta.json: [{file, full, duration, ref}]")
    ap.add_argument("--mode", default="all", choices=["all", "sweep", "compare", "quality", "incremental"])
    ap.add_argument("--chunk-s", type=float, default=8.0, help="chunk length for chunked/primed arms")
    ap.add_argument("--seg-s", type=float, default=5.0, help="segment length for the incremental arm")
    ap.add_argument("--cut", default="pause", choices=["fixed", "pause"], help="incremental segment boundaries")
    ap.add_argument("--clip-s", type=float, default=20.0)
    ap.add_argument("--n", type=int, default=8, help="files per arm in compare/quality/incremental")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N fixtures (already-used audio)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = json.load(open(args.fixtures))[args.skip :]
    out = open(args.out, "a")

    def rec(**kw):
        kw["ts"] = time.time()
        out.write(json.dumps(kw) + "\n")
        out.flush()

    # Warm-up on throwaway noise so Triton JIT / cudagraph shapes are not in the numbers.
    rng = np.random.default_rng(0)
    for _ in range(2):
        call(args.url, args.model, messages([rng.normal(0, 0.05, SR * 3).astype(np.float32)], True), 4)

    files = iter(meta)

    if args.mode in ("all", "sweep"):
        # Whole-clip TTFT vs duration, unique audio each. Slope = amortisable per-second cost.
        durs = [4, 8, 12, 16, 20, 24, 32]
        table = {}
        for rnd in range(2):
            for d in durs:
                m = next(files)
                x = load(m["full"], start_s=0 if rnd == 0 else max(0, m["duration"] - d), dur_s=d)
                r = call(args.url, args.model, messages([x], True), args.max_tokens)
                rec(mode="sweep", dur=d, file=m["file"], **{k: r[k] for k in ("ttft", "total", "usage")})
                table.setdefault(d, []).append(r["ttft"])
                ptok = (r["usage"] or {}).get("prompt_tokens")
                print(f"sweep {d:>3}s ttft={r['ttft']:.3f} total={r['total']:.3f} prompt_tok={ptok}", flush=True)
        print("\nSWEEP median TTFT by duration:")
        for d in durs:
            print(f"  {d:>3}s  {med(table[d]):.3f}s")

    if args.mode in ("all", "compare"):
        # Paired: crop X (start) -> whole then chunked; crop Y (end) -> primed.
        rows = []
        for _ in range(args.n):
            m = next(files)
            x = load(m["full"], start_s=0, dur_s=args.clip_s)
            y = load(m["full"], start_s=max(0, m["duration"] - args.clip_s), dur_s=args.clip_s)

            a = call(args.url, args.model, messages([x], True), args.max_tokens)
            b = call(args.url, args.model, messages(split(x, args.chunk_s), True), args.max_tokens)

            ychunks = split(y, args.chunk_s)
            primes = []
            for k in range(1, len(ychunks)):
                p = call(args.url, args.model, messages(ychunks[:k], False), 1)
                primes.append(p["total"])
            c = call(args.url, args.model, messages(ychunks, True), args.max_tokens)

            row = dict(
                mode="compare",
                file=m["file"],
                whole_ttft=a["ttft"],
                chunked_ttft=b["ttft"],
                primed_ttft=c["ttft"],
                prime_latencies=primes,
                whole_total=a["total"],
                primed_total=c["total"],
                whole_prompt_tokens=(a["usage"] or {}).get("prompt_tokens"),
                chunked_prompt_tokens=(b["usage"] or {}).get("prompt_tokens"),
                primed_prompt_tokens=(c["usage"] or {}).get("prompt_tokens"),
                whole_completion_tokens=(a["usage"] or {}).get("completion_tokens"),
                whole_text=a["text"],
                chunked_text=b["text"],
                primed_text=c["text"],
            )
            rows.append(row)
            rec(**row)
            print(
                f"compare {m['file']}: whole={a['ttft']:.3f} chunked={b['ttft']:.3f} primed={c['ttft']:.3f} "
                f"primes={[round(p, 3) for p in primes]} "
                f"tok={row['whole_prompt_tokens']}/{row['chunked_prompt_tokens']}",
                flush=True,
            )
        print(f"\nCOMPARE ({args.clip_s:.0f}s clips, {args.chunk_s:.0f}s chunks, n={len(rows)}) median TTFT:")
        for k in ("whole_ttft", "chunked_ttft", "primed_ttft"):
            print(f"  {k:<13} {med([r[k] for r in rows]):.3f}s")
        print(f"  prime request median {med([p for r in rows for p in r['prime_latencies']]):.3f}s")

    if args.mode in ("all", "quality"):
        # Does chunking change the transcript? Full clips vs reference.
        rows = []
        for _ in range(args.n):
            m = next(files)
            x = load(m["full"])
            a = call(args.url, args.model, messages([x], True), 1024)
            b = call(args.url, args.model, messages(split(x, args.chunk_s), True), 1024)
            row = dict(
                mode="quality",
                file=m["file"],
                dur=m["duration"],
                wer_whole=wer(m["ref"], a["text"]),
                wer_chunked=wer(m["ref"], b["text"]),
                wer_chunked_vs_whole=wer(a["text"], b["text"]),
                whole_text=a["text"],
                chunked_text=b["text"],
                ref=m["ref"],
            )
            rows.append(row)
            rec(**row)
            print(
                f"quality {m['file']} ({m['duration']:.0f}s): WER whole={row['wer_whole']:.3f} "
                f"chunked={row['wer_chunked']:.3f} chunked-vs-whole={row['wer_chunked_vs_whole']:.3f}",
                flush=True,
            )
        print(
            f"\nQUALITY (n={len(rows)}): mean WER whole={statistics.mean(r['wer_whole'] for r in rows):.3f} "
            f"chunked={statistics.mean(r['wer_chunked'] for r in rows):.3f}"
        )

    if args.mode in ("all", "incremental"):
        # Segment-by-segment transcription on a growing conversation. Latency on 20 s crops
        # (paired with whole), quality on the full clip (paired with whole, vs reference).
        splitter = (
            (lambda x: split_at_pauses(x, args.seg_s)) if args.cut == "pause" else (lambda x: split(x, args.seg_s))
        )
        rows = []
        for _ in range(args.n):
            m = next(files)
            x = load(m["full"], start_s=0, dur_s=args.clip_s)
            a = call(args.url, args.model, messages([x], True), args.max_tokens)
            inc = run_incremental(args.url, args.model, splitter(x), args.max_tokens)

            xf = load(m["full"])
            af = call(args.url, args.model, messages([xf], True), 1024)
            incf = run_incremental(args.url, args.model, splitter(xf), 1024)

            row = dict(
                mode="incremental",
                file=m["file"],
                seg_s=args.seg_s,
                cut=args.cut,
                whole_ttft=a["ttft"],
                whole_total=a["total"],
                whole_completion_tokens=(a["usage"] or {}).get("completion_tokens"),
                inc_last_ttft=inc["last"]["ttft"],
                inc_last_total=inc["last"]["total"],
                inc_per_seg=inc["per_seg"],
                inc_text_20s=inc["text"],
                whole_text_20s=a["text"],
                wer_20s_inc_vs_whole=wer(a["text"], inc["text"]),
                dur=m["duration"],
                wer_whole=wer(m["ref"], af["text"]),
                wer_inc=wer(m["ref"], incf["text"]),
                wer_inc_vs_whole=wer(af["text"], incf["text"]),
                full_inc_per_seg=incf["per_seg"],
                whole_text=af["text"],
                inc_text=incf["text"],
                ref=m["ref"],
            )
            rows.append(row)
            rec(**row)
            segs = [round(s["total"], 3) for s in inc["per_seg"]]
            print(
                f"incremental {m['file']}: 20s whole total={a['total']:.3f} (ttft {a['ttft']:.3f}) | "
                f"inc last total={inc['last']['total']:.3f} (ttft {inc['last']['ttft']:.3f}) per-seg={segs} | "
                f"full-clip WER whole={row['wer_whole']:.3f} inc={row['wer_inc']:.3f}",
                flush=True,
            )
        print(
            f"\nINCREMENTAL ({args.clip_s:.0f}s clips, {args.seg_s:.0f}s segs, cut={args.cut}, n={len(rows)}) median:"
        )
        w_tot, w_ttft = med([r["whole_total"] for r in rows]), med([r["whole_ttft"] for r in rows])
        i_tot, i_ttft = med([r["inc_last_total"] for r in rows]), med([r["inc_last_ttft"] for r in rows])
        print(f"  whole total        {w_tot:.3f}s  (ttft {w_ttft:.3f}s)")
        print(f"  incremental last   {i_tot:.3f}s  (ttft {i_ttft:.3f}s)")
        print(
            f"  full-clip mean WER whole={statistics.mean(r['wer_whole'] for r in rows):.3f} "
            f"incremental={statistics.mean(r['wer_inc'] for r in rows):.3f}"
        )


if __name__ == "__main__":
    main()

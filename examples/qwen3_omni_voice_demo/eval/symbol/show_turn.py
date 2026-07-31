#!/usr/bin/env python3
"""Summarise a captured turn: what went in, what the model wrote.

    python show_turn.py                 # newest capture
    python show_turn.py turn-1a2b3c4d   # a specific one
"""
import glob, json, os, sys, wave

D = os.environ.get("OMNI_DEBUG_DUMP_DIR", "/tmp/omni_debug")
stem = sys.argv[1] if len(sys.argv) > 1 else None
if stem:
    base = os.path.join(D, stem.replace(".json", ""))
else:
    js = sorted(glob.glob(f"{D}/turn-*.json"), key=os.path.getmtime)
    if not js:
        sys.exit(f"no captures in {D} - talk to the agent with OMNI_DEBUG_DUMP_DIR set")
    base = js[-1][:-5]

d = json.load(open(f"{base}.json"))
su = d["session_update"]
print(f"=== {os.path.basename(base)} ===")
print(f"voice        : {su.get('voice')}")
print(f"tools        : {[t['function']['name'] for t in su.get('tools', [])]}")
print(f"instructions : {(su.get('instructions') or '')[:100]!r}")
if os.path.exists(f"{base}.wav"):
    w = wave.open(f"{base}.wav")
    print(f"user audio   : {w.getnframes()/w.getframerate():.2f}s @ {w.getframerate()}Hz")
print(f"history      : {len(d['history'])} message(s)")
for m in d["history"]:
    if m["role"] == "user":
        print(f"   user      <audio {m['pcm_bytes']}b>")
    else:
        print(f"   {m['role']:<9} {m['text'][:78]!r}")
gen = f"{base}.generated.txt"
if os.path.exists(gen):
    print(f"\nmodel wrote  : {open(gen).read()!r}")
else:
    print("\nmodel wrote  : (turn produced no reply - abandoned mid tool-loop?)")
print(f"\nreplay this turn:\n  python replay_fixed.py {base}")

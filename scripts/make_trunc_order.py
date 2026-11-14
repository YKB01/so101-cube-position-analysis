#!/usr/bin/env python
"""
Create protocol/trunc_trial_order.csv and protocol/trunc_model_key.json for the C_trunc vs C test (P1 and P2 only).

* trunc_trial_order.csv: P1 and P2 in random order; each position has 16 trials in 4 blocks of 4; each block is ABBA or
  BAAB (random) -> 8 A + 8 B per position, 32 trials (U01..U32). numpy default_rng(seed); the seed goes to
  protocol/trunc_order_seed.txt.
* trunc_model_key.json: random assignment of codes A/B to C_trunc / C checkpoints (drawn with `secrets`, never printed);
  SHA-256 of the file in protocol/trunc_model_key.sha256 (the key file itself is committed only after the test).
Refuses to overwrite existing files.
"""
import hashlib
import json
import secrets
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROTO = ROOT / "protocol"
CKPT = {"C_trunc": "/home/ykb/Checkpoints/act_red_cube_expert50_trunc20_ft5000/checkpoints/005000/pretrained_model",
        "C": "/home/ykb/Checkpoints/act_red_cube_expert50_expert20_ft5000/checkpoints/005000/pretrained_model"}
POSITIONS, N_BLOCKS, BLOCKS = ["P1", "P2"], 4, ("ABBA", "BAAB")
targets = [PROTO / f for f in ("trunc_trial_order.csv", "trunc_model_key.json", "trunc_model_key.sha256", "trunc_order_seed.txt")]
if any(t.exists() for t in targets):
    sys.exit("trunc protocol files already exist; refusing to overwrite")
seed = secrets.randbits(32)
rng = np.random.default_rng(seed)
rows, n = [], 0
for pid in rng.permutation(POSITIONS):
    for _ in range(N_BLOCKS):
        for code in BLOCKS[rng.integers(2)]:
            n += 1
            rows.append((f"U{n:02d}", pid, code))
df = pd.DataFrame(rows, columns=["trial_id", "position_id", "model_code"])
assert len(df) == 32 and (df.groupby(["position_id", "model_code"]).size() == 8).all()
df.to_csv(PROTO / "trunc_trial_order.csv", index=False)
(PROTO / "trunc_order_seed.txt").write_text(f"{seed}\n")
codes = ["A", "B"]; secrets.SystemRandom().shuffle(codes)
key = {"nonce": secrets.token_hex(16), "key": {codes[0]: {"model": "C_trunc", "checkpoint": CKPT["C_trunc"]},
                                               codes[1]: {"model": "C", "checkpoint": CKPT["C"]}}}
text = json.dumps(key, indent=2) + "\n"
(PROTO / "trunc_model_key.json").write_text(text)
(PROTO / "trunc_model_key.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
print("written: trunc_trial_order.csv (32 trials), trunc_order_seed.txt, trunc_model_key.json, trunc_model_key.sha256 (mapping not printed)")

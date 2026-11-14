#!/usr/bin/env python
"""
Create protocol/trial_order.csv and protocol/model_key.json for the pre-registered right-side retest.

* trial_order.csv: positions P1..P3 in random order; each position has 16 trials in 4 blocks of 4;
  each block is ABBA or BAAB (random) -> 8 A + 8 B per position, 48 trials in total.
  Drawn from numpy's default_rng(ORDER_SEED); the seed is written to protocol/order_seed.txt.
* model_key.json: random assignment of codes A/B to model R / model C checkpoints. Drawn with `secrets`
  (not reproducible on purpose). The mapping is NEVER printed. A SHA-256 of the file is written to
  protocol/model_key.sha256 so the key can be committed as a hash and revealed after the test.
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
CKPT = {"R": "/home/ykb/Checkpoints/act_red_cube_mixed_hil_ft_20261015_150904/checkpoints/005000/pretrained_model",
        "C": "/home/ykb/Checkpoints/act_red_cube_expert50_expert20_ft5000/checkpoints/005000/pretrained_model"}
N_BLOCKS, BLOCKS = 4, ("ABBA", "BAAB")

PROTO.mkdir(exist_ok=True)
targets = [PROTO / f for f in ("trial_order.csv", "model_key.json", "model_key.sha256", "order_seed.txt")]
if any(t.exists() for t in targets):
    sys.exit("protocol files already exist; refusing to overwrite")

positions = [p["position_id"] for p in json.loads((ROOT / "results/test_positions.json").read_text())["positions"]]
seed = secrets.randbits(32)
rng = np.random.default_rng(seed)
rows, n = [], 0
for pid in rng.permutation(positions):
    for _ in range(N_BLOCKS):
        for code in BLOCKS[rng.integers(2)]:
            n += 1
            rows.append((f"T{n:02d}", pid, code))
df = pd.DataFrame(rows, columns=["trial_id", "position_id", "model_code"])
assert len(df) == 48 and (df.groupby(["position_id", "model_code"]).size() == 8).all()
df.to_csv(PROTO / "trial_order.csv", index=False)
(PROTO / "order_seed.txt").write_text(f"{seed}\n")

codes = ["A", "B"]; secrets.SystemRandom().shuffle(codes)
key = {"nonce": secrets.token_hex(16),
       "key": {codes[0]: {"model": "R", "checkpoint": CKPT["R"]}, codes[1]: {"model": "C", "checkpoint": CKPT["C"]}}}
text = json.dumps(key, indent=2)
(PROTO / "model_key.json").write_text(text + "\n")
(PROTO / "model_key.sha256").write_text(hashlib.sha256((text + "\n").encode()).hexdigest() + "\n")
print("written: trial_order.csv (48 trials), order_seed.txt, model_key.json, model_key.sha256 (mapping not printed)")

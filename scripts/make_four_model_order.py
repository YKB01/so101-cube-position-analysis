#!/usr/bin/env python
"""
Create protocol/four_model_trial_order.csv and protocol/four_model_key.json for the four-model test (C, R, C_trunc, deep; P1 and P2, 8 trials per model and position, 64 trials V01-V64).

* four_model_trial_order.csv: 16 blocks of 4 trials that alternate between the two positions (P1 block, P2 block, ... - which position comes first is random); each position gets 8 blocks;
  every block is a random permutation of the four blind codes A-D (so each model is tested exactly 8 times per position). numpy default_rng(seed); the seed goes to protocol/four_model_order_seed.txt.
* four_model_key.json: random assignment of the codes A-D to the four checkpoints (drawn with `secrets`, never printed); SHA-256 of the file in protocol/four_model_key.sha256.
  The key file stays untracked until the test is finished. The deep checkpoint path is fixed now; the checkpoint is created by the training that precedes the test.
Refuses to overwrite existing files.
"""
import hashlib, json, secrets, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent; PROTO = ROOT / "protocol"; CK = "/home/ykb/Checkpoints"
CKPT = {"C": f"{CK}/act_red_cube_expert50_expert20_ft5000/checkpoints/005000/pretrained_model",
        "R": f"{CK}/act_red_cube_mixed_hil_ft_20261015_150904/checkpoints/005000/pretrained_model",
        "C_trunc": f"{CK}/act_red_cube_expert50_trunc20_ft5000/checkpoints/005000/pretrained_model",
        "deep": f"{CK}/act_red_cube_expert50_rewind_deep20_ft5000/checkpoints/005000/pretrained_model"}
targets = [PROTO / f for f in ("four_model_trial_order.csv", "four_model_key.json", "four_model_key.sha256", "four_model_order_seed.txt")]
if any(t.exists() for t in targets):
    sys.exit("four-model protocol files already exist; refusing to overwrite")
seed = secrets.randbits(32); rng = np.random.default_rng(seed)
rows, n = [], 0
first = "P1" if rng.integers(2) == 0 else "P2"; second = "P2" if first == "P1" else "P1"
for b in range(16):
    pid = first if b % 2 == 0 else second
    for code in rng.permutation(list("ABCD")):
        n += 1; rows.append((f"V{n:02d}", pid, code))
df = pd.DataFrame(rows, columns=["trial_id", "position_id", "model_code"])
assert len(df) == 64 and (df.groupby(["position_id", "model_code"]).size() == 8).all()
df.to_csv(PROTO / "four_model_trial_order.csv", index=False)
(PROTO / "four_model_order_seed.txt").write_text(f"{seed}\n")
codes = list("ABCD"); secrets.SystemRandom().shuffle(codes)
key = {"nonce": secrets.token_hex(16), "key": {codes[i]: {"model": m, "checkpoint": CKPT[m]} for i, m in enumerate(CKPT)}}
text = json.dumps(key, indent=2) + "\n"
(PROTO / "four_model_key.json").write_text(text); (PROTO / "four_model_key.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
print("written: four_model_trial_order.csv (64 trials), four_model_order_seed.txt, four_model_key.json, four_model_key.sha256 (mapping not printed)")

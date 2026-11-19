#!/usr/bin/env python
"""Verification of the merged deep-rewind training dataset (ykb_red_cube_expert50_rewind_deep20) against its sources (read only):
episodes 0-49 = expert 50, episodes 50-69 = rewind_deep_20 corrections 0-19 (state, action, timestamps, frame_index, task, features without `intervention`),
plus decoded images compared for the first / middle / last frame of every episode and 100 random frames. Writes rewind_offline/deep_dataset_verification.txt."""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]
from lerobot.datasets import LeRobotDataset
ROOT = Path.home() / ".cache/huggingface/lerobot/YKB0123"
M, E, D = "ykb_red_cube_expert50_rewind_deep20", "so101_red_cube_plastic_box_front_50_20260731_104214", "rewind_deep_20"
lines = []
def say(s):
    print(s); lines.append(s)
ld = lambda n: LeRobotDataset("YKB0123/" + n, root=ROOT / n)
m, e, d = ld(M), ld(E), ld(D)
ok = True
say(f"merged: {m.meta.total_episodes} episodes, {m.meta.total_frames} frames (expert {e.meta.total_episodes}/{e.meta.total_frames} + corrections {d.meta.total_episodes}/{d.meta.total_frames})")
ok &= m.meta.total_episodes == 70 and m.meta.total_frames == e.meta.total_frames + d.meta.total_frames
fm = {k: (v["dtype"], list(v["shape"]), v.get("names")) for k, v in m.meta.features.items()}; fe = {k: (v["dtype"], list(v["shape"]), v.get("names")) for k, v in e.meta.features.items()}
say(f"features identical to the expert dataset (no intervention column): {fm == fe}"); ok &= fm == fe
say(f"tasks: {list(m.meta.tasks.index)} (expert {list(e.meta.tasks.index)}, corrections {list(d.meta.tasks.index)})"); ok &= list(m.meta.tasks.index) == list(e.meta.tasks.index) == list(d.meta.tasks.index)
dm, de, dd = (x.hf_dataset.to_pandas() for x in (m, e, d))
bad = 0
for ep in range(70):
    a = dm[dm.episode_index == ep].sort_values("frame_index"); b = (de[de.episode_index == ep] if ep < 50 else dd[dd.episode_index == ep - 50]).sort_values("frame_index")
    same = len(a) == len(b) and np.array_equal(np.stack(a["observation.state"]), np.stack(b["observation.state"])) and np.array_equal(np.stack(a["action"]), np.stack(b["action"])) \
        and np.allclose(a.timestamp.to_numpy(), b.timestamp.to_numpy()) and (a.frame_index.to_numpy() == np.arange(len(a))).all()
    bad += int(not same)
say(f"state / action / timestamp / frame_index identical to the source in all 70 episodes: {bad == 0} ({bad} differ)"); ok &= bad == 0
say(f"index contiguous: {bool((dm['index'].to_numpy() == np.arange(len(dm))).all())}"); ok &= bool((dm["index"].to_numpy() == np.arange(len(dm))).all())
def img(ds, i):
    return (ds[int(i)]["observation.images.front"].numpy() * 255).round().astype(np.uint8)
me, ee, de_ = m.meta.episodes, e.meta.episodes, d.meta.episodes
pairs = []
for ep in range(70):
    src, sep = (e, ep) if ep < 50 else (d, ep - 50); n = int(me["length"][ep]); mf = int(me["dataset_from_index"][ep]); sf = int((ee if ep < 50 else de_)["dataset_from_index"][sep])
    for k in (0, n // 2, n - 1):
        pairs.append((m, mf + k, src, sf + k))
rng = np.random.default_rng(0)
for gi in rng.choice(len(dm), 100, replace=False):
    ep = int(dm.loc[gi, "episode_index"]); k = int(dm.loc[gi, "frame_index"]); src, sep = (e, ep) if ep < 50 else (d, ep - 50)
    pairs.append((m, int(gi), src, int((ee if ep < 50 else de_)["dataset_from_index"][sep]) + k))
nb = sum(int(np.abs(img(a, i).astype(int) - img(b, j).astype(int)).max() != 0) for a, i, b, j in pairs)
say(f"decoded images compared in {len(pairs)} frames (first / middle / last of every episode + 100 random): {'identical' if nb == 0 else str(nb) + ' differ'}"); ok &= nb == 0
say("RESULT: " + ("PASS" if ok else "FAIL"))
(HERE / "rewind_offline/deep_dataset_verification.txt").write_text("\n".join(lines) + "\n")

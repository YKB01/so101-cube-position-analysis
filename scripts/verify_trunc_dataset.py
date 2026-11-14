#!/usr/bin/env python
"""
Task B step 3 verification: frame-by-frame comparison of the merged dataset (50 expert + 20 path-truncated expert)
with the original raw datasets. Raw datasets are only read.

For merged episodes 0-49: identical to original expert episodes 0-49 (state, action, timestamp, every decoded image).
For merged episodes 50-69: frame k equals frame (cut + k) of the new expert episode e-50 (state, action, every decoded image),
with frame_index 0.., timestamp = frame_index / 30, contiguous global index. Also checks info/episode metadata and that
the normal LeRobotDataset loader returns the same images for random frames.
Writes trunc_control/verification.txt.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import analyze_cube_positions as A  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.datasets.video_utils import decode_video_frames  # noqa: E402

ROOT = A.LEROBOT_ROOT
EXP = ROOT / "so101_red_cube_plastic_box_front_50_20260731_104214"
NEW = ROOT / "so101_red_cube_plastic_box_expert_20_20261112_130439"
MER = ROOT / "ykb_red_cube_expert50_trunc20"
VK, FPS = "observation.images.front", 30
lines = []


def say(s):
    print(s); lines.append(s)


def load(root):
    ep = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))).sort_values("episode_index").set_index("episode_index")
    dat = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data/*/*.parquet")))).sort_values("index").reset_index(drop=True)
    return ep, dat


def images(root, ep_row, n):
    path = root / f"videos/{VK}/chunk-{int(ep_row[f'videos/{VK}/chunk_index']):03d}/file-{int(ep_row[f'videos/{VK}/file_index']):03d}.mp4"
    f0 = float(ep_row[f"videos/{VK}/from_timestamp"])
    return np.asarray(decode_video_frames(path, [f0 + k / FPS for k in range(n)], 1e-4, backend="pyav", return_uint8=True))


def main():
    plan = pd.read_csv(HERE / "trunc_control/truncation_plan.csv").set_index("new_episode").path_cut_frame.to_dict()
    me, md = load(MER); xe, xd = load(EXP); ne, nd = load(NEW)
    ok = True
    info = json.loads((MER / "meta/info.json").read_text())
    say(f"merged: episodes {len(me)}, frames {len(md)}, info total_frames {info['total_frames']}, total_episodes {info['total_episodes']}")
    ok &= len(me) == 70 and len(md) == info["total_frames"] == 29901 + int(sum(len(nd[nd.episode_index == e]) - c for e, c in plan.items()))
    # metadata continuity
    ok &= bool((me.dataset_from_index.to_numpy()[1:] == me.dataset_to_index.to_numpy()[:-1]).all()) and int(me.dataset_to_index.iloc[-1]) == len(md)
    ok &= bool((md["index"].to_numpy() == np.arange(len(md))).all())
    say(f"index contiguous: {bool((md['index'].to_numpy() == np.arange(len(md))).all())}; episode lengths match rows: "
        f"{bool((md.groupby('episode_index').size().to_numpy() == me.length.to_numpy()).all())}")
    ok &= bool((md.groupby("episode_index").size().to_numpy() == me.length.to_numpy()).all())
    nbad_total, nframes = 0, 0
    for e in range(70):
        m = md[md.episode_index == e].sort_values("frame_index")
        n = len(m)
        if e < 50:
            s = xd[xd.episode_index == e].sort_values("frame_index"); src_ep = xe.loc[e]; src_root = EXP; off = 0
        else:
            s = nd[nd.episode_index == e - 50].sort_values("frame_index"); c = plan[e - 50]
            s = s.iloc[c:]; src_ep = ne.loc[e - 50]; src_root = NEW; off = c
            ok &= (len(s) == n)
        st = np.abs(np.stack(m["observation.state"].to_numpy()) - np.stack(s["observation.state"].to_numpy())).max()
        ac = np.abs(np.stack(m["action"].to_numpy()) - np.stack(s["action"].to_numpy())).max()
        fi_ok = bool((m.frame_index.to_numpy() == np.arange(n)).all())
        ts_ok = bool(np.allclose(m.timestamp.to_numpy(), np.arange(n) / FPS, atol=1e-6))
        a = images(MER, me.loc[e], n)
        b = images(src_root, src_ep, off + n)[off:]
        img_diff = int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())
        good = st == 0 and ac == 0 and fi_ok and ts_ok and img_diff == 0
        ok &= good; nframes += n
        if not good:
            nbad_total += 1
            say(f"  episode {e}: MISMATCH state {st} action {ac} frame_index {fi_ok} timestamp {ts_ok} image max diff {img_diff}")
    say(f"compared {nframes} frames in 70 episodes (state, action, frame_index, timestamp, every decoded image): "
        f"{'all identical' if nbad_total == 0 else str(nbad_total) + ' episodes with differences'}")
    # standard loader on random frames
    ds = LeRobotDataset("YKB0123/ykb_red_cube_expert50_trunc20", root=MER)
    rng = np.random.default_rng(0)
    bad = 0
    for i in rng.choice(len(ds), 60, replace=False):
        it = ds[int(i)]
        e, k = int(it["episode_index"]), int(it["frame_index"])
        if e < 50:
            ref = images(EXP, xe.loc[e], k + 1)[k]
        else:
            ref = images(NEW, ne.loc[e - 50], plan[e - 50] + k + 1)[plan[e - 50] + k]
        got = (it[VK].numpy() * 255).round().astype(np.uint8)
        bad += int(np.abs(got.astype(int) - ref.astype(int)).max() != 0)
    say(f"LeRobotDataset loader, 60 random frames: {'identical' if bad == 0 else str(bad) + ' differ'}")
    ok &= bad == 0
    # stats sanity
    st = json.loads((MER / "meta/stats.json").read_text())
    sm = np.array(st["observation.state"]["mean"]); dm = np.stack(md["observation.state"].to_numpy()).mean(0)
    say(f"stats.json observation.state mean vs data mean: max abs diff {np.abs(sm - dm).max():.4f}")
    say(f"stats.json state count {st['observation.state']['count']} (frames {len(md)})")
    say("RESULT: " + ("PASS" if ok else "FAIL"))
    (HERE / "trunc_control/verification.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

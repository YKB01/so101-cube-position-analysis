#!/usr/bin/env python
"""
Task B step 3: build the truncated new-expert dataset and the merged "50 expert + 20 truncated expert" dataset.

Truncation is done at the metadata level, without re-encoding any video (LeRobot v3 episodes point into shared mp4 files by
timestamp): for every episode the frames before the cut frame are dropped from the data parquet (frame_index, timestamp and
index are renumbered), and the episode's video from_timestamp is advanced by cut/30 s. Per-episode stats are recomputed with
LeRobot's own compute_episode_stats (images: the sampled frames of the kept part), the dataset stats are re-aggregated, then
the merge uses LeRobot's merge_datasets (same code path as `lerobot-edit-dataset --operation.type merge`).
Raw datasets are only read. Cut frames come from trunc_control/truncation_plan.csv (scripts/pair_and_truncate.py).

Outputs (new directories under the LeRobot cache):
  YKB0123/ykb_new_expert20_pathtrunc            (20 truncated episodes)
  YKB0123/ykb_red_cube_expert50_trunc20         (episodes 0-49 = original expert, 50-69 = truncated expert)
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import analyze_cube_positions as A  # noqa: E402

from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats  # noqa: E402
from lerobot.datasets.dataset_tools import merge_datasets  # noqa: E402
from lerobot.datasets.video_utils import decode_video_frames  # noqa: E402

ROOT = A.LEROBOT_ROOT                      # ~/.cache/huggingface/lerobot/YKB0123
SRC_NEW = ROOT / "so101_red_cube_plastic_box_expert_20_20261112_130439"
SRC_EXPERT = ROOT / "so101_red_cube_plastic_box_front_50_20260731_104214"
TRUNC_NAME, MERGED_NAME = "ykb_new_expert20_pathtrunc", "ykb_red_cube_expert50_trunc20"
VKEY, FPS = "observation.images.front", 30


def main():
    plan = pd.read_csv(HERE / "trunc_control/truncation_plan.csv").sort_values("new_episode")
    cut = dict(zip(plan.new_episode, plan.path_cut_frame))
    dst = ROOT / TRUNC_NAME
    if dst.exists() or (ROOT / MERGED_NAME).exists():
        sys.exit("output directories already exist; refusing to overwrite")
    (dst / "meta/episodes/chunk-000").mkdir(parents=True)
    (dst / "data/chunk-000").mkdir(parents=True)
    (dst / f"videos/{VKEY}/chunk-000").mkdir(parents=True)

    # ---- data parquet ---------------------------------------------------------------------------------
    tbl = pq.read_table(SRC_NEW / "data/chunk-000/file-000.parquet")
    ep = tbl["episode_index"].to_numpy(); fi = tbl["frame_index"].to_numpy()
    keep = np.array([f >= cut[int(e)] for e, f in zip(ep, fi)])
    tbl = tbl.filter(pa.array(keep))
    ep = tbl["episode_index"].to_numpy(); fi_old = tbl["frame_index"].to_numpy()
    cut_arr = np.array([cut[int(e)] for e in ep])
    fi_new = fi_old - cut_arr
    idx_new = np.arange(len(ep))
    for name, arr, typ in (("frame_index", fi_new, pa.int64()), ("index", idx_new, pa.int64()),
                           ("timestamp", (fi_new / FPS).astype(np.float32), pa.float32())):
        tbl = tbl.set_column(tbl.schema.get_field_index(name), name, pa.array(arr, type=typ))
    pq.write_table(tbl, dst / "data/chunk-000/file-000.parquet")
    shutil.copy(SRC_NEW / f"videos/{VKEY}/chunk-000/file-000.mp4", dst / f"videos/{VKEY}/chunk-000/file-000.mp4")
    shutil.copy(SRC_NEW / "meta/tasks.parquet", dst / "meta/tasks.parquet")

    # ---- episodes metadata + recomputed stats --------------------------------------------------------------
    emeta = pq.read_table(SRC_NEW / "meta/episodes/chunk-000/file-000.parquet")
    df = emeta.to_pandas().sort_values("episode_index").reset_index(drop=True)
    info = json.loads((SRC_NEW / "meta/info.json").read_text())
    features = info["features"]
    data = tbl.to_pandas()
    vpath = SRC_NEW / f"videos/{VKEY}/chunk-000/file-000.mp4"
    ep_stats, start = {}, 0
    tmp = Path(tempfile.mkdtemp(prefix="trunc_stats_", dir="/tmp/claude-1000"))
    from lerobot.datasets.compute_stats import sample_indices
    for i, r in df.iterrows():
        e = int(r.episode_index)
        d = data[data.episode_index == e]
        n = len(d)
        c = cut[e]
        df.loc[i, "length"] = n
        df.loc[i, "dataset_from_index"] = start
        df.loc[i, "dataset_to_index"] = start + n
        f0 = float(r[f"videos/{VKEY}/from_timestamp"]) + c / FPS
        df.loc[i, f"videos/{VKEY}/from_timestamp"] = f0
        start += n
        # stats: numeric features from the kept rows, images from the sampled kept frames
        ep_data = {k: np.stack(d[k].to_numpy()) if k in ("action", "observation.state") else d[k].to_numpy()
                   for k in ("action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index")}
        sel = sample_indices(n)
        ts = [f0 + k / FPS for k in sel]
        frames = decode_video_frames(vpath, ts, 1e-4, backend="pyav", return_uint8=True)
        paths = [""] * n
        for k, fr in zip(sel, frames):
            import cv2
            p = tmp / f"e{e}_f{k}.png"
            cv2.imwrite(str(p), cv2.cvtColor(np.asarray(fr).transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
            paths[k] = str(p)
        ep_data[VKEY] = paths
        ep_stats[e] = compute_episode_stats(ep_data, features)
    shutil.rmtree(tmp, ignore_errors=True)

    schema = emeta.schema
    cols = {}
    for f in schema:
        name = f.name
        if name.startswith("stats/"):
            _, feat, stat = name.split("/", 2)
            vals = [np.asarray(ep_stats[int(e)][feat][stat]).tolist() for e in df.episode_index]
            cols[name] = pa.array(vals, type=f.type)
        elif name in ("length", "dataset_from_index", "dataset_to_index", "episode_index"):
            cols[name] = pa.array(df[name].astype("int64").tolist(), type=f.type)
        elif name == f"videos/{VKEY}/from_timestamp":
            cols[name] = pa.array(df[name].astype(float).tolist(), type=f.type)
        else:
            cols[name] = pa.array(df[name].tolist(), type=f.type)
    pq.write_table(pa.table(cols, schema=schema), dst / "meta/episodes/chunk-000/file-000.parquet")

    info["total_frames"] = int(len(tbl))
    (dst / "meta").mkdir(exist_ok=True)
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    agg = aggregate_stats([ep_stats[e] for e in sorted(ep_stats)])
    from lerobot.datasets.io_utils import write_stats
    write_stats(agg, dst)
    print(f"truncated dataset written: {dst}  episodes={len(df)} frames={len(tbl)}")

    # ---- merge: expert 50 first, then the truncated 20 ------------------------------------------------------------
    d_exp = LeRobotDataset("YKB0123/" + SRC_EXPERT.name, root=SRC_EXPERT)
    d_tr = LeRobotDataset("YKB0123/" + TRUNC_NAME, root=dst)
    merged = merge_datasets([d_exp, d_tr], output_repo_id="YKB0123/" + MERGED_NAME, output_dir=ROOT / MERGED_NAME)
    print(f"merged: {ROOT / MERGED_NAME}  episodes={merged.num_episodes} frames={merged.num_frames}")


if __name__ == "__main__":
    main()

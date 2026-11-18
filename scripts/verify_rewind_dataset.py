#!/usr/bin/env python
"""
Check a dataset written by collect_rewind.py against the existing intervention dataset (rollout_so101_red_cube_hil_front_20261015_134746):
features (dtype, shape, names), fps, task string, per-episode frame counts, intervention column (True for corrections, False for rewind segments),
timestamps, value ranges, and that the first and last video frame decode. Optionally cross-checks rewind_meta.csv.

  python scripts/verify_rewind_dataset.py YKB0123/rewind_deep_test [--segments] [--meta results/rewind_collect/run_xxx/rewind_meta.csv]
  --segments: the dataset is <repo_id>_rewind_seg (intervention must be all False, rewind_segment_meta.csv expected in its root)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

REF = Path.home() / ".cache/huggingface/lerobot/YKB0123/rollout_so101_red_cube_hil_front_20261015_134746"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id"); ap.add_argument("--root", default=None); ap.add_argument("--segments", action="store_true"); ap.add_argument("--meta", default=None)
    a = ap.parse_args()
    from lerobot.datasets import LeRobotDataset
    root = Path(a.root) if a.root else Path.home() / ".cache/huggingface/lerobot" / a.repo_id
    ds = LeRobotDataset(a.repo_id, root=root)
    ref = json.loads((REF / "meta/info.json").read_text())
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond); print(f"[{'OK' if cond else 'FAIL'}] {name} {detail}")
    print(f"dataset {a.repo_id}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames, fps {ds.meta.fps}")
    f1 = {k: (v["dtype"], list(v["shape"]), v.get("names")) for k, v in ds.meta.features.items()}
    f0 = {k: (v["dtype"], list(v["shape"]), v.get("names")) for k, v in ref["features"].items()}
    check("feature names / dtype / shape identical to the existing intervention dataset", f1 == f0, "" if f1 == f0 else f"\n   differences: { {k: (f1.get(k), f0.get(k)) for k in set(f1) | set(f0) if f1.get(k) != f0.get(k)} }")
    check("fps equal", ds.meta.fps == ref["fps"], f"({ds.meta.fps} vs {ref['fps']})")
    check("robot_type equal", ds.meta.info.get("robot_type") == ref.get("robot_type"))
    tasks_ref = list(pd.read_parquet(REF / "meta/tasks.parquet").index); tasks = list(ds.meta.tasks.index)
    check("task string equal", tasks == tasks_ref, f"({tasks})")
    df = ds.hf_dataset.to_pandas()
    iv = df["intervention"].apply(lambda x: bool(np.asarray(x).ravel()[0]))
    want = not a.segments
    check(f"intervention is {want} in every frame", (iv == want).all(), f"({int(iv.sum())}/{len(iv)} True)")
    lens = df.groupby("episode_index").size()
    print(f"   episode lengths (frames): {lens.tolist()}  (existing intervention dataset: median {int(pd.read_parquet(next((REF/'data').rglob('*.parquet'))).groupby('episode_index').size().median())})")
    ts = df.groupby("episode_index").timestamp.apply(lambda t: np.allclose(np.diff(t), 1 / ds.meta.fps, atol=1e-3))
    check("timestamps advance by 1/fps within every episode", ts.all())
    st, ac = np.stack(df["observation.state"]), np.stack(df["action"])
    ref_df = pd.concat([pd.read_parquet(f) for f in sorted((REF / "data").rglob("*.parquet"))])
    rs, ra = np.stack(ref_df["observation.state"]), np.stack(ref_df["action"])
    inr = lambda x, r: (x.min(0) >= r.min(0) - 15).all() and (x.max(0) <= r.max(0) + 15).all()
    check("state within the range of the existing data (+-15)", inr(st, rs), f"min {np.round(st.min(0), 0).tolist()} max {np.round(st.max(0), 0).tolist()}")
    check("action within the range of the existing data (+-15)", inr(ac, ra))
    check("no NaN", not (np.isnan(st).any() or np.isnan(ac).any()))
    # decode first and last frame of the first and last episode
    for ep in sorted({0, ds.meta.total_episodes - 1}):
        idx = df.index[df.episode_index == ep]
        for i in (idx[0], idx[-1]):
            im = ds[int(i)]["observation.images.front"]
            check(f"video frame decodes (episode {ep}, frame {i - idx[0]})", tuple(im.shape) == (3, 480, 640) and float(im.mean()) > 0.01, f"shape {tuple(im.shape)} mean {float(im.mean()):.2f}")
    if a.segments:
        p = root / "rewind_segment_meta.csv"
        check("rewind_segment_meta.csv present", p.exists())
        if p.exists():
            m = pd.read_csv(p); print(m.to_string(index=False)); check("rewind_segment=True for every row", m.rewind_segment.all())
    if a.meta:
        m = pd.read_csv(a.meta); print(f"rewind_meta.csv: {len(m)} rows; discarded {int(m.discarded.sum())}; levels {m.level.value_counts().to_dict()}; triggers {m.trigger_type.value_counts().to_dict()}")
        saved = m.saved_episode_index.dropna()
        if not a.segments:
            check("saved_episode_index values exist in the dataset", set(saved.astype(int)) <= set(range(ds.meta.total_episodes)))
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()

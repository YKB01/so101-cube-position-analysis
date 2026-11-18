#!/usr/bin/env python
"""
Step 1a of the lighting analysis: first frame of every episode of every dataset (training data and all evaluation sessions) and the recording time.
Read only. Writes lighting_analysis/episodes_index.csv and lighting_analysis/cache/first_frames.npy (full-resolution RGB, not committed).

Recording time: 11/14 retest and truncation trials: meta.json started_unix of the trial (exact). Every other dataset: the dataset directory name carries the
start time (stamped when the recording was created) and the last write of the data parquet is the end of the recording; the time of episode k is interpolated
linearly between the two ("interpolated" in the time_source column; the error is of the order of the episode spacing, about 40 s).
"""
import glob, json, os, re, sys, warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]
ROOT = Path.home() / ".cache/huggingface/lerobot/YKB0123"
OUT = HERE / "lighting_analysis"
JST = timezone(timedelta(hours=9))

GROUPS = [  # (group, label, directory name or glob, session)
    ("train_expert50", "expert 50 (7/31)", "so101_red_cube_plastic_box_front_50_20260731_104214", "train"),
    ("train_hil20", "recovery 20 (10/15)", "rollout_so101_red_cube_hil_front_20261015_134746", "train"),
    ("train_newexpert20", "new expert 20 (11/12)", "so101_red_cube_plastic_box_expert_20_20261112_130439", "train"),
    ("eval_mixed_smooth", "eval mixed smooth (11/4)", "rollout_eval_mixed_smooth_20261104_121950", "eval_1104"),
    ("eval_mixed_default", "eval mixed default (11/4)", "rollout_eval_mixed_default_20261104_124959", "eval_1104"),
    ("eval_expert_smooth", "eval expert smooth (11/4)", "rollout_eval_expert_smooth_20261104_131322", "eval_1104"),
    ("eval_expert_default", "eval expert default (11/4)", "rollout_eval_expert_default_20261104_133651", "eval_1104"),
    ("eval_recovery_default", "eval recovery default (11/4)", "rollout_eval_recovery_default_20261104_135927", "eval_1104"),
    ("eval_recovery_smooth", "eval recovery smooth (11/4)", "rollout_eval_recovery_smooth_20261104_141826", "eval_1104"),
    ("eval_expert70_smooth", "control standard eval (11/12)", "rollout_eval_expertmixed_smooth_20261112_134559", "eval_1112"),
    ("edge_mixed", "right-side extra, mixed (11/12)", "rollout_eval_edge_hilmixed_smooth_20261112_144609", "eval_1112"),
    ("edge_control", "right-side extra, control (11/12)", "rollout_eval_edge_expertmixed_smooth_20261112_145925", "eval_1112"),
]


def stamp(name):
    m = re.search(r"(\d{8})_(\d{6})$", name) or re.search(r"(\d{8})_(\d{6})", name)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=JST) if m else None


def first_frame(root, ep):
    from lerobot.datasets import LeRobotDataset
    ds = LeRobotDataset("YKB0123/" + Path(root).name, root=root)
    i = int(ds.meta.episodes["dataset_from_index"][ep])
    return ds, np.asarray(ds[i]["observation.images.front"].permute(1, 2, 0).clip(0, 1) * 255, dtype=np.uint8)


def main():
    rows, frames = [], []
    def add(group, label, session, root, ep, t, src, ds_name, extra=None):
        _, fr = first_frame(root, ep)
        frames.append(fr)
        rows.append(dict(group=group, label=label, session=session, dataset=ds_name, episode=ep, time=t.isoformat(timespec="seconds") if t else "", time_source=src, **(extra or {})))
    for group, label, dname, session in GROUPS:
        root = ROOT / dname
        info = json.loads((root / "meta/info.json").read_text()); n = info["total_episodes"]
        t0 = stamp(dname); files = sorted(glob.glob(str(root / "data/*/*.parquet")))
        t1 = datetime.fromtimestamp(max(os.path.getmtime(f) for f in files), JST)
        print(group, n, t0, t1, flush=True)
        ds = None
        from lerobot.datasets import LeRobotDataset
        ds = LeRobotDataset("YKB0123/" + dname, root=root)
        frm = ds.meta.episodes["dataset_from_index"]
        for ep in range(n):
            t = t0 + (t1 - t0) * (ep + 0.5) / n
            fr = np.asarray(ds[int(frm[ep])]["observation.images.front"].permute(1, 2, 0).clip(0, 1) * 255, dtype=np.uint8)
            frames.append(fr)
            rows.append(dict(group=group, label=label, session=session, dataset=dname, episode=ep, time=t.isoformat(timespec="seconds"), time_source="interpolated"))
    # 11/14 retest and truncation trials: one dataset per attempt, exact start time from meta.json
    for kind, folder, order_csv, prefix in (("retest", "rollout_retest", "protocol/trial_order.csv", "retest_raw"), ("trunc", "rollout_trunc", "protocol/trunc_trial_order.csv", "trunc_raw")):
        for mj in sorted(glob.glob(str(HERE / f"results/{prefix}/*/meta.json"))):
            meta = json.loads(Path(mj).read_text())
            name = meta["dataset_repo_id"].split("/", 1)[1]
            root = ROOT / folder / name
            t = datetime.fromtimestamp(meta["started_unix"], JST)
            _, fr = first_frame(root, 0)
            frames.append(fr)
            rows.append(dict(group=f"{kind}_1114", label=f"{kind} trials (11/14)", session=f"{kind}_1114", dataset=name, episode=0, time=t.isoformat(timespec="seconds"),
                             time_source="meta.json started_unix", trial_id=meta["trial_id"], attempt=meta["attempt"]))
        print(kind, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "episodes_index.csv", index=False)
    np.save(OUT / "cache/first_frames.npy", np.stack(frames))
    print(len(df), "frames", np.stack(frames).shape)

if __name__ == "__main__":
    main()

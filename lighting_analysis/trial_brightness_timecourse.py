#!/usr/bin/env python
"""Brightness inside each 11/14 trial (frames at 0, 1, ..., 10 s): is the first frame representative of what the policy sees? Writes trial_brightness_timecourse.csv."""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, cv2
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]
ROOT = Path.home() / ".cache/huggingface/lerobot/YKB0123"
R = {k: v for k, v in json.loads((HERE / "lighting_analysis/regions.json").read_text()).items() if not k.startswith("_")}
from lerobot.datasets import LeRobotDataset
rows = []
for kind, folder, prefix in (("retest", "rollout_retest", "retest_raw"), ("trunc", "rollout_trunc", "trunc_raw")):
    for mj in sorted((HERE / f"results/{prefix}").glob("*/meta.json")):
        meta = json.loads(mj.read_text()); name = meta["dataset_repo_id"].split("/", 1)[1]
        ds = LeRobotDataset("YKB0123/" + name, root=ROOT / folder / name)
        n = len(ds)
        for sec in range(0, 11):
            i = min(sec * 30, n - 1)
            im = np.asarray(ds[i]["observation.images.front"].permute(1, 2, 0).clip(0, 1) * 255, dtype=np.uint8)
            g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(float)
            x0, y0, x1, y1 = R["table"]; a0, b0, a1, b1 = R["bg_sheet_right"]
            rows.append(dict(kind=kind, trial_id=meta["trial_id"], attempt=meta["attempt"], sec=sec, gray_mean=g.mean(), table_gray=g[y0:y1, x0:x1].mean(), sheet_gray=g[b0:b1, a0:a1].mean()))
    print(kind, "done", flush=True)
pd.DataFrame(rows).to_csv(HERE / "lighting_analysis/trial_brightness_timecourse.csv", index=False, float_format="%.3f")

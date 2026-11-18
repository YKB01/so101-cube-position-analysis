#!/usr/bin/env python
"""Step 1c: per-region statistics of the first frames (regions in lighting_analysis/regions.json; red cube pixels excluded). Writes region_stats.csv and frame_stats_all.csv."""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, cv2
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import analyze_cube_positions as A
R = {k: v for k, v in json.loads((HERE / "lighting_analysis/regions.json").read_text()).items() if not k.startswith("_")}

def main():
    idx = pd.read_csv(HERE / "lighting_analysis/frame_stats.csv")
    F = np.load(HERE / "lighting_analysis/cache/first_frames.npy", mmap_mode="r")
    rows = []
    for i in range(len(idx)):
        rgb = np.ascontiguousarray(F[i]); red = A.red_mask(rgb) > 0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float); hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(float)
        row = {}
        for k, (x0, y0, x1, y1) in R.items():
            m = ~red[y0:y1, x0:x1]
            g = gray[y0:y1, x0:x1][m]; px = rgb[y0:y1, x0:x1][m].astype(float)
            row.update({f"{k}_gray": g.mean(), f"{k}_std": g.std(), f"{k}_r": px[:, 0].mean(), f"{k}_g": px[:, 1].mean(), f"{k}_b": px[:, 2].mean(),
                        f"{k}_rb": px[:, 0].mean() / px[:, 2].mean(), f"{k}_sat": hsv[y0:y1, x0:x1][..., 1][m].mean()})
        rows.append(row)
    S = pd.DataFrame(rows); S.to_csv(HERE / "lighting_analysis/region_stats.csv", index=False, float_format="%.4f")
    out = pd.concat([idx, S], axis=1); out.to_csv(HERE / "lighting_analysis/frame_stats_all.csv", index=False, float_format="%.4f")
    print(out.groupby("group")[[c for c in S if c.endswith("_gray")]].median().round(1).to_string())
    print(out.groupby("group")[[c for c in S if c.endswith("_rb")]].median().round(3).to_string())

if __name__ == "__main__":
    main()

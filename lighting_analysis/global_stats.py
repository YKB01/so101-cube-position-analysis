#!/usr/bin/env python
"""Step 1b: per-frame global statistics of the first frames (brightness, contrast, colour, cube detection). Writes lighting_analysis/frame_stats.csv."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, cv2
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import analyze_cube_positions as A

def stats(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(float)
    r, gg, b = (rgb[..., i].astype(float) for i in range(3))
    sat = hsv[..., 1] > 40
    hue = hsv[..., 0][sat] * 2 * np.pi / 180.0
    hue_mean = (np.degrees(np.arctan2(np.sin(hue).mean(), np.cos(hue).mean())) / 2) % 180 if sat.any() else np.nan     # circular mean over saturated pixels (OpenCV 0-179)
    m = A.red_mask(rgb)
    det = A.detect_cube(rgb)
    x0, y0, x1, y1 = A.ROI
    return dict(gray_mean=g.mean(), gray_std=g.std(), r_mean=r.mean(), g_mean=gg.mean(), b_mean=b.mean(), rb_ratio=r.mean() / b.mean(), blue_share=b.mean() / (r.mean() + gg.mean() + b.mean()),
                hue_mean_sat_pixels=hue_mean, sat_mean=hsv[..., 1].mean(), val_mean=hsv[..., 2].mean(),
                red_pixel_frac_roi=float((m[y0:y1, x0:x1] > 0).mean()), cube_detected=bool(det["detected"]), cube_area=float(det["area"]), cube_x=float(det["cx"]), cube_y=float(det["cy"]),
                clipped_dark_frac=float((g < 5).mean()), clipped_bright_frac=float((g > 250).mean()))

def main():
    idx = pd.read_csv(HERE / "lighting_analysis/episodes_index.csv")
    F = np.load(HERE / "lighting_analysis/cache/first_frames.npy", mmap_mode="r")
    S = pd.DataFrame([stats(np.ascontiguousarray(F[i])) for i in range(len(idx))])
    out = pd.concat([idx, S], axis=1); out.to_csv(HERE / "lighting_analysis/frame_stats.csv", index=False, float_format="%.4f")
    print(out.groupby("group")[["gray_mean", "gray_std", "rb_ratio", "sat_mean", "cube_area", "red_pixel_frac_roi"]].median().round(3).to_string())
    print("cube detected:", out.groupby("group").cube_detected.mean().round(2).to_dict())

if __name__ == "__main__":
    main()

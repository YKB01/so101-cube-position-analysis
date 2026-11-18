"""
Per-episode lighting covariates for the collection and test scripts (camera stays on automatic exposure; nothing is changed on the camera):
  frame_lighting(rgb)         first-frame statistics: table gray (box in lighting_analysis/regions.json, red cube pixels excluded), window-side gray (bg_sheet_right),
                              whole-frame gray, table R/B
  read_camera_controls(dev)   read-back of the UVC controls with `v4l2-ctl --list-ctrls` (read only; returns {} if v4l2-ctl is not available)
  inside_window(now, "13:30-15:30")
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_REGIONS = {k: v for k, v in json.loads((HERE / "lighting_analysis/regions.json").read_text()).items() if not k.startswith("_")}
CONTROLS = ["auto_exposure", "exposure_time_absolute", "exposure_dynamic_framerate", "white_balance_automatic", "white_balance_temperature", "gain",
            "backlight_compensation", "brightness", "contrast", "saturation", "gamma", "sharpness", "power_line_frequency"]


def frame_lighting(rgb) -> dict:
    import cv2
    import analyze_cube_positions as A
    rgb = np.asarray(rgb)
    red = A.red_mask(rgb) > 0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float)

    def region(name):
        x0, y0, x1, y1 = _REGIONS[name]
        m = ~red[y0:y1, x0:x1]
        return gray[y0:y1, x0:x1][m].mean(), rgb[y0:y1, x0:x1][m].astype(float)

    tg, px = region("table")
    sg, _ = region("bg_sheet_right")
    return dict(first_frame_table_gray=float(tg), first_frame_sheet_gray=float(sg), first_frame_gray_mean=float(gray.mean()),
                first_frame_table_rb=float(px[:, 0].mean() / px[:, 2].mean()))


def read_camera_controls(device: str = "/dev/video0", timeout_s: float = 3.0) -> dict:
    try:
        out = subprocess.run(["v4l2-ctl", "-d", device, "--list-ctrls"], capture_output=True, text=True, timeout=timeout_s).stdout
    except Exception:
        return {}
    vals = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:.*?value=(-?\d+)", line)
        if m and m.group(1) in CONTROLS:
            vals[f"cam_{m.group(1)}"] = int(m.group(3))
    return vals


def inside_window(now: datetime | None = None, window: str = "13:30-15:30") -> bool:
    now = now or datetime.now()
    a, b = window.split("-")
    h = lambda s: int(s[:2]) * 60 + int(s[3:])
    return h(a) <= now.hour * 60 + now.minute <= h(b)

#!/usr/bin/env python
"""
Live cube-placement helper for the pre-registered right-side retest.

Shows the front camera (same settings as data collection: index 0, 640x480, 30 fps, MJPG, 10 s warm-up),
draws the target cross and a 3 px tolerance circle for the current trial, detects the cube with the SAME
detector as the analysis (analyze_cube_positions.detect_cube, RGB, same HSV/ROI) and shows dx, dy.
The circle turns green when the cube centre (mean of the last N_STABLE frames) is inside the tolerance.

Only trial_id and position_id are shown. The model is never displayed (trial_order.csv has model_code, ignored here).

Keys:  n / p  next / previous trial        s  log placement (only while green)
       r  toggle live/reference blend       q or Esc  quit
At start-up the live view is blended with results/reference_frame.png (saved at the first run, or with
--reset-reference) and a phase-correlation shift estimate is printed to check that the camera did not move.

Inputs : results/test_positions.json, protocol/trial_order.csv (trial_id, position_id, model_code)
Outputs: results/placement_log.csv (appended), results/reference_frame.png (first run only)
Test without a camera: --source some_image_or_video.  Render one frame without GUI: --dump out.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import analyze_cube_positions as A  # noqa: E402  (detect_cube, ROI: same detector as the position analysis)

# ---- camera: copied from the data-collection / eval commands (--robot.cameras front: ...) ------------
CAM_INDEX, CAM_W, CAM_H, CAM_FPS, CAM_FOURCC, WARMUP_S = 0, 640, 480, 30, "MJPG", 10.0
TOL_PX = 3.0            # tolerance circle radius (px, image coordinates)
N_STABLE = 5            # green needs N consecutive detections whose mean is inside the tolerance
SHIFT_WARN_PX = 1.0     # camera-shift warning threshold (phase correlation)
LOG_COLUMNS = ["trial_id", "position_id", "target_x", "target_y", "measured_x", "measured_y",
               "dx", "dy", "dist", "timestamp_iso", "timestamp_unix"]
GREEN, RED, WHITE, YEL, CYAN = (60, 200, 60), (60, 60, 230), (255, 255, 255), (0, 220, 255), (255, 220, 0)


def load_plan(order_csv: Path, pos_json: Path):
    pos = {p["position_id"]: (p["cx"], p["cy"]) for p in json.loads(pos_json.read_text())["positions"]}
    order = pd.read_csv(order_csv)
    missing = set(order.position_id) - set(pos)
    if missing:
        sys.exit(f"position ids in {order_csv.name} missing from {pos_json.name}: {sorted(missing)}")
    return [(str(r.trial_id), str(r.position_id), *pos[r.position_id]) for r in order.itertuples()]


def logged_ids(path: Path) -> set[str]:
    return set(pd.read_csv(path).trial_id.astype(str)) if path.exists() and path.stat().st_size else set()


def append_log(path: Path, row: dict):
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, LOG_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


class Source:
    """Camera (V4L2, MJPG) or, for tests, an image / video file."""
    def __init__(self, src: str | None):
        self.static = None
        if src and Path(src).suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            self.static = cv2.imread(src)
            if self.static is None:
                sys.exit(f"cannot read image {src}")
            self.cap = None
            return
        self.cap = cv2.VideoCapture(src) if src else cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            sys.exit("cannot open camera (is another process using it? stop stale LeRobot processes first)")
        if not src:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAM_FOURCC))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
            w, h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (w, h) != (CAM_W, CAM_H):
                sys.exit(f"camera delivers {w}x{h}, expected {CAM_W}x{CAM_H}")

    def read(self):
        if self.static is not None:
            time.sleep(1 / 30)
            return self.static.copy()
        ok, f = self.cap.read()
        if not ok:
            return None
        return f

    def warmup(self, seconds: float):
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.read()


def camera_shift(ref_bgr, cur_bgr):
    g = lambda im: np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    (dx, dy), resp = cv2.phaseCorrelate(g(ref_bgr), g(cur_bgr), cv2.createHanningWindow(ref_bgr.shape[1::-1], cv2.CV_32F))
    return float(dx), float(dy), float(resp)


def put(img, text, org, scale=0.6, color=WHITE, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render(frame, det, recent, trial, idx, n, done, scale, blend_ref, ref, status_msg, shift_txt):
    base = cv2.addWeighted(frame, 0.5, ref, 0.5, 0) if blend_ref else frame
    img = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    S = lambda v: int(round(v * scale))
    tid, pid, tx, ty = trial
    x0, y0, x1, y1 = A.ROI
    cv2.rectangle(img, (S(x0), S(y0)), (S(x1), S(y1)), (110, 110, 110), 1)

    green, mean = False, None
    if det["detected"]:
        cv2.drawMarker(img, (S(det["cx"]), S(det["cy"])), CYAN, cv2.MARKER_CROSS, 12, 1)
    if len(recent) == N_STABLE:
        mean = np.mean(recent, axis=0)
        green = bool(np.hypot(mean[0] - tx, mean[1] - ty) <= TOL_PX)
    col = GREEN if green else RED
    # target: cross (long arms) + tolerance circle (radius TOL_PX in image px)
    cv2.line(img, (S(tx) - 14, S(ty)), (S(tx) + 14, S(ty)), col, 1, cv2.LINE_AA)
    cv2.line(img, (S(tx), S(ty) - 14), (S(tx), S(ty) + 14), col, 1, cv2.LINE_AA)
    cv2.circle(img, (S(tx), S(ty)), S(TOL_PX), col, 2, cv2.LINE_AA)

    put(img, f"{tid}   position {pid}   ({idx + 1}/{n})", (10, 26), 0.8, WHITE, 2)
    put(img, f"target   x={tx:.1f}  y={ty:.1f}", (10, 54), 0.6)
    if det["detected"]:
        dx, dy = det["cx"] - tx, det["cy"] - ty
        put(img, f"measured x={det['cx']:.1f}  y={det['cy']:.1f}", (10, 78), 0.6)
        put(img, f"dx={dx:+.1f}  dy={dy:+.1f}  dist={np.hypot(dx, dy):.1f} px   (tol {TOL_PX:g})", (10, 102), 0.6,
            GREEN if green else YEL)
    else:
        put(img, "cube not detected", (10, 78), 0.6, RED)
    h = img.shape[0]
    if tid in done:
        put(img, "LOGGED", (img.shape[1] - 110, 26), 0.7, GREEN, 2)
    put(img, "GREEN: press s to log" if green else "move cube until the circle is green", (10, h - 40), 0.6,
        GREEN if green else WHITE)
    put(img, "n/p trial   s log   r ref blend   q quit" + (f"   | {shift_txt}" if shift_txt else ""), (10, h - 14), 0.5)
    if status_msg:
        put(img, status_msg, (10, h - 66), 0.6, YEL)
    return img, green, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=Path, default=ROOT / "protocol/trial_order.csv")
    ap.add_argument("--positions", type=Path, default=ROOT / "results/test_positions.json")
    ap.add_argument("--log", type=Path, default=ROOT / "results/placement_log.csv")
    ap.add_argument("--reference", type=Path, default=ROOT / "results/reference_frame.png")
    ap.add_argument("--reset-reference", action="store_true", help="overwrite the reference frame with the current one")
    ap.add_argument("--source", default=None, help="test only: image/video file instead of the camera")
    ap.add_argument("--dump", type=Path, default=None, help="test only: render one frame to this PNG and exit")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--warmup-s", type=float, default=WARMUP_S)
    ap.add_argument("--start", default=None, help="trial_id to start at (default: first not yet logged)")
    args = ap.parse_args()

    plan = load_plan(args.order, args.positions)
    done = logged_ids(args.log)
    idx = next((i for i, t in enumerate(plan) if (t[0] == args.start if args.start else t[0] not in done)), 0)

    src = Source(args.source)
    if not args.source:
        print(f"warming up camera for {args.warmup_s:g} s ...")
        src.warmup(args.warmup_s)
    frame = src.read()
    if frame is None:
        sys.exit("no frame from source")
    args.reference.parent.mkdir(exist_ok=True)
    first_run = args.reset_reference or not args.reference.exists()
    if first_run:
        cv2.imwrite(str(args.reference), frame)
        print(f"saved reference frame: {args.reference}")
    ref = cv2.imread(str(args.reference))
    if ref is None or ref.shape != frame.shape:
        sys.exit("reference frame missing or different size; run with --reset-reference")
    dx_s, dy_s, resp = camera_shift(ref, frame)
    shift_txt = f"cam shift vs ref ({dx_s:+.1f},{dy_s:+.1f}) px"
    print(shift_txt + (f"  WARNING: > {SHIFT_WARN_PX} px, check the camera" if np.hypot(dx_s, dy_s) > SHIFT_WARN_PX else ""))

    recent: deque = deque(maxlen=N_STABLE)
    blend, msg, msg_t = True, "", 0.0      # start with the blended reference view; r toggles it
    if not args.dump:
        cv2.namedWindow("placement", cv2.WINDOW_AUTOSIZE)
    while True:
        frame = src.read()
        if frame is None:
            print("camera read failed"); break
        det = A.detect_cube(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if det["detected"]:
            recent.append((det["cx"], det["cy"]))
        else:
            recent.clear()
        img, green, mean = render(frame, det, recent, plan[idx], idx, len(plan), done, args.scale, blend, ref,
                                  msg if time.time() - msg_t < 3 else "", shift_txt)
        if args.dump and len(recent) == N_STABLE:
            cv2.imwrite(str(args.dump), img); print(f"rendered {args.dump} green={green}"); break
        cv2.imshow("placement", img)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("n"):
            idx = min(idx + 1, len(plan) - 1); recent.clear()
        elif k == ord("p"):
            idx = max(idx - 1, 0); recent.clear()
        elif k == ord("r"):
            blend = not blend
        elif k == ord("s"):
            tid, pid, tx, ty = plan[idx]
            if green:
                now = datetime.now().astimezone()
                append_log(args.log, dict(trial_id=tid, position_id=pid, target_x=f"{tx:.2f}", target_y=f"{ty:.2f}",
                                          measured_x=f"{mean[0]:.2f}", measured_y=f"{mean[1]:.2f}",
                                          dx=f"{mean[0] - tx:+.2f}", dy=f"{mean[1] - ty:+.2f}",
                                          dist=f"{np.hypot(mean[0] - tx, mean[1] - ty):.2f}",
                                          timestamp_iso=now.isoformat(timespec="seconds"), timestamp_unix=f"{now.timestamp():.3f}"))
                done.add(tid); msg, msg_t = f"logged {tid}", time.time()
            else:
                msg, msg_t = "not logged: circle is not green", time.time()
    if not args.dump:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

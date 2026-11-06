#!/usr/bin/env python
"""
Detect the red cube in the first frame of every episode of several LeRobot
(v3.0) datasets and relate eval test positions to training-data positions.

Dataset layout (verified on these datasets, codebase_version v3.0):
  meta/episodes/chunk-XXX/file-XXX.parquet  -> one row per episode with
      videos/observation.images.front/{chunk_index,file_index,from_timestamp}
  videos/observation.images.front/chunk-XXX/file-XXX.mp4  (AV1, many episodes
      concatenated per file). The first frame of an episode is the frame at
      `from_timestamp` inside that file.

Raw datasets are only read, never written. All outputs go to OUT_DIR.

Usage:
  python analyze_cube_positions.py                    # all datasets + analysis
  python analyze_cube_positions.py --datasets expert  # trial run, one dataset
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# =============================================================================
# CONFIG - adjust here
# =============================================================================
LEROBOT_ROOT = Path("~/.cache/huggingface/lerobot/YKB0123").expanduser()

DATASETS = {
    # key: directory name
    "expert": "so101_red_cube_plastic_box_front_50_20260731_104214",
    "recovery": "rollout_so101_red_cube_hil_front_20261015_134746",
    "eval_expert_smooth": "rollout_eval_expert_smooth_20261104_131322",
    "eval_expert_default": "rollout_eval_expert_default_20261104_133651",
    "eval_mixed_smooth": "rollout_eval_mixed_smooth_20261104_121950",
    "eval_mixed_default": "rollout_eval_mixed_default_20261104_124959",
    "eval_recovery_smooth": "rollout_eval_recovery_smooth_20261104_141826",
    "eval_recovery_default": "rollout_eval_recovery_default_20261104_135927",
}
EVAL_KEYS = [k for k in DATASETS if k.startswith("eval_")]
CAMERA_KEY = "observation.images.front"
TIMESTAMP_TOLERANCE_S = 1e-4  # same default as LeRobotDataset

OUT_DIR = Path("./cube_position_analysis")

# --- HSV red thresholds (OpenCV: H 0-179, S 0-255, V 0-255) -----------------
# Red wraps around 0/180, so two ranges are OR-ed together.
# Measured on expert ep0: cube H~1-2, S~175-190, V~190-255;
# wooden board H 0-8/175-179 but S only ~40-65 -> S_MIN separates them.
HSV_RED_LOW_1 = (0, 120, 70)
HSV_RED_HIGH_1 = (10, 255, 255)
HSV_RED_LOW_2 = (170, 120, 70)
HSV_RED_HIGH_2 = (179, 255, 255)
BLUR_KSIZE = 3            # Gaussian blur before HSV (0 = off)
MORPH_KSIZE = 3           # open+close kernel size (0 = off)
MIN_AREA = 80             # px^2; cube is ~340 px^2, red servo wires ~30 px^2
MAX_AREA = 5000           # px^2; anything bigger is not the cube
# Optional region of interest (x0, y0, x1, y1) in pixels; None = full frame.
ROI = (150, 120, 470, 380)  # wooden board area; excludes hands/phone at bottom-left (eval_mixed_default)

# --- anomaly flags (used in the report, esp. for recovery data) -------------
AREA_RATIO_RANGE = (0.5, 2.0)   # vs. median area of detected expert episodes
MAX_ASPECT = 1.8                # bbox long/short side; larger -> occluded/odd
SECOND_BLOB_RATIO = 0.3         # 2nd largest blob / largest -> ambiguous

# Recovery episodes flagged as anomalous are excluded from the
# "expert+recovery" nearest-neighbour distance (the "_all" column keeps them).
EXCLUDE_FLAGGED_RECOVERY = True

# --- test positions ----------------------------------------------------------
N_TEST_POSITIONS = 6
KMEANS_RESTARTS = 50
# Chinese label used in the success table -> English label used in figures
LABELS_ZH_EN = {
    "左上": "top-left", "左中": "mid-left", "左下": "bottom-left",
    "右上": "top-right", "右中": "mid-right", "右下": "bottom-right",
}
# Success counts (x / TRIALS_PER_POSITION), keyed by English label.
TRIALS_PER_POSITION = 4
SUCCESS = {
    #                mixed_s mixed_d expert_s expert_d recov_s recov_d
    "bottom-left":  (3, 4, 0, 0, 1, 1),
    "bottom-right": (2, 2, 0, 0, 1, 0),
    "mid-left":     (4, 3, 4, 4, 0, 0),
    "mid-right":    (4, 4, 0, 2, 2, 1),
    "top-left":     (2, 4, 2, 1, 4, 3),
    "top-right":    (1, 0, 0, 0, 2, 1),
}
SUCCESS_COLS = ["mixed_smooth", "mixed_default", "expert_smooth",
                "expert_default", "recovery_smooth", "recovery_default"]
BACKGROUND_FRAME = ("eval_mixed_smooth", 0)  # (dataset, episode) for plots
ZOOM_BOX = (300, 140, 460, 270)  # (x0, y0, x1, y1) zoom panel of position plots

# --- plot styling (reference palette, first 3 categorical slots) -------------
C_EXPERT = "#2a78d6"     # blue
C_RECOVERY = "#eb6834"   # orange
C_TEST = "#1baf7a"       # aqua
C_TEXT = "#0b0b0b"
C_TEXT_2 = "#52514e"
CONTACT_COLS = 4
CONTACT_SCALE = 0.5      # full-frame thumbnail scale
CONTACT_CROP = 60        # half-size of zoom crop around detection (px)
CONTACT_ZOOM_PX = 240    # zoom panel size

log = logging.getLogger("cube")


# =============================================================================
# Data access
# =============================================================================
def load_episode_meta(ds_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(ds_dir / "meta/episodes/*/*.parquet")))
    if not files:
        raise FileNotFoundError(f"no episode metadata in {ds_dir}")
    k = f"videos/{CAMERA_KEY}/"
    cols = ["episode_index", "length", k + "chunk_index", k + "file_index", k + "from_timestamp"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    df = df.rename(columns={k + "chunk_index": "v_chunk", k + "file_index": "v_file",
                            k + "from_timestamp": "v_from_ts"})
    return df.sort_values("episode_index").reset_index(drop=True)


def video_path(ds_dir: Path, chunk: int, file: int) -> Path:
    import json
    info = json.loads((ds_dir / "meta/info.json").read_text())
    rel = info["video_path"].format(video_key=CAMERA_KEY, chunk_index=chunk, file_index=file)
    return ds_dir / rel


def first_frames(ds_dir: Path):
    """Yield (episode_index, RGB uint8 HxWx3) for the first frame of each episode."""
    from lerobot.datasets.video_utils import decode_video_frames
    meta = load_episode_meta(ds_dir)
    for row in meta.itertuples():
        vp = video_path(ds_dir, int(row.v_chunk), int(row.v_file))
        fr = decode_video_frames(vp, [float(row.v_from_ts)], tolerance_s=TIMESTAMP_TOLERANCE_S,
                                 backend="pyav", return_uint8=True)
        yield int(row.episode_index), fr[0].permute(1, 2, 0).numpy().copy()


# =============================================================================
# Detection
# =============================================================================
def red_mask(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if BLUR_KSIZE:
        bgr = cv2.GaussianBlur(bgr, (BLUR_KSIZE, BLUR_KSIZE), 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, HSV_RED_LOW_1, HSV_RED_HIGH_1) | cv2.inRange(hsv, HSV_RED_LOW_2, HSV_RED_HIGH_2)
    if MORPH_KSIZE:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KSIZE, MORPH_KSIZE))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if ROI is not None:
        x0, y0, x1, y1 = ROI
        roi = np.zeros_like(m)
        roi[y0:y1, x0:x1] = 255
        m &= roi
    return m


def detect_cube(rgb: np.ndarray) -> dict:
    m = red_mask(rgb)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = [(cv2.contourArea(c), c) for c in contours]
    cands = sorted([ac for ac in cands if MIN_AREA <= ac[0] <= MAX_AREA], key=lambda ac: -ac[0])
    res = dict(cx=np.nan, cy=np.nan, area=np.nan, detected=False, bx=np.nan, by=np.nan,
               bw=np.nan, bh=np.nan, aspect=np.nan, n_candidates=len(cands),
               second_area=np.nan, contour=None, other_contours=[c for _, c in cands[1:]])
    if not cands:
        return res
    area, c = cands[0]
    M = cv2.moments(c)
    x, y, w, h = cv2.boundingRect(c)
    if M["m00"] > 0:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    else:
        cx, cy = x + w / 2, y + h / 2
    res.update(cx=cx, cy=cy, area=area, detected=True, bx=x, by=y, bw=w, bh=h,
               aspect=max(w, h) / max(1, min(w, h)), contour=c,
               second_area=cands[1][0] if len(cands) > 1 else np.nan)
    return res


# =============================================================================
# Contact sheet
# =============================================================================
def make_tile(rgb: np.ndarray, det: dict, title: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    full = bgr.copy()
    if ROI is not None:
        cv2.rectangle(full, ROI[:2], ROI[2:], (200, 200, 200), 1)
    for c in det["other_contours"]:
        cv2.drawContours(full, [c], -1, (0, 255, 255), 2)  # yellow = rejected candidates
    if det["detected"]:
        x, y, w, h = int(det["bx"]), int(det["by"]), int(det["bw"]), int(det["bh"])
        cv2.rectangle(full, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 255, 0), 2)
        cv2.drawMarker(full, (int(round(det["cx"])), int(round(det["cy"]))), (255, 0, 255),
                       cv2.MARKER_CROSS, 14, 2)
    th, tw = int(full.shape[0] * CONTACT_SCALE), int(full.shape[1] * CONTACT_SCALE)
    thumb = cv2.resize(full, (tw, th), interpolation=cv2.INTER_AREA)

    # zoom panel: raw crop with mask contour (1 px) and centre
    zoom = np.full((th, CONTACT_ZOOM_PX, 3), 40, np.uint8)
    if det["detected"]:
        H, W = bgr.shape[:2]
        cx, cy = int(round(det["cx"])), int(round(det["cy"]))
        x0, y0 = max(0, cx - CONTACT_CROP), max(0, cy - CONTACT_CROP)
        x1, y1 = min(W, cx + CONTACT_CROP), min(H, cy + CONTACT_CROP)
        s = CONTACT_ZOOM_PX / (2 * CONTACT_CROP)
        crop = cv2.resize(bgr[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        c = ((det["contour"].astype(np.float32) - [x0, y0]) * s).astype(np.int32)
        cv2.drawContours(crop, [c], -1, (0, 255, 0), 1)
        cv2.drawMarker(crop, (int((det["cx"] - x0) * s), int((det["cy"] - y0) * s)),
                       (255, 0, 255), cv2.MARKER_CROSS, 16, 1)
        zoom[: min(th, crop.shape[0]), : crop.shape[1]] = crop[:th]
    else:
        cv2.putText(zoom, "NOT DETECTED", (20, th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)

    tile = np.hstack([thumb, zoom])
    bar = np.full((26, tile.shape[1], 3), 255, np.uint8)
    color = (0, 0, 0) if det["detected"] else (0, 0, 255)
    cv2.putText(bar, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    tile = np.vstack([bar, tile])
    return cv2.copyMakeBorder(tile, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def save_contact_sheet(tiles: list[np.ndarray], header: str, path: Path):
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + CONTACT_COLS - 1) // CONTACT_COLS
    sheet = np.full((rows * th + 40, CONTACT_COLS * tw, 3), 255, np.uint8)
    cv2.putText(sheet, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    for i, t in enumerate(tiles):
        r, c = divmod(i, CONTACT_COLS)
        sheet[40 + r * th: 40 + (r + 1) * th, c * tw:(c + 1) * tw] = t
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


# =============================================================================
# Stage 1: detection for each dataset
# =============================================================================
def run_detection(keys: list[str]) -> tuple[pd.DataFrame, dict]:
    rows, frames = [], {}
    for key in keys:
        ds_dir = LEROBOT_ROOT / DATASETS[key]
        log.info("=== %s  (%s)", key, ds_dir)
        tiles = []
        for ep, rgb in first_frames(ds_dir):
            frames[(key, ep)] = rgb
            det = detect_cube(rgb)
            rows.append(dict(dataset=key, episode_index=ep,
                             **{k: v for k, v in det.items() if k not in ("contour", "other_contours")}))
            if det["detected"]:
                title = f"ep {ep}  ({det['cx']:.0f},{det['cy']:.0f})  area {det['area']:.0f}"
                if det["n_candidates"] > 1:
                    title += f"  +{det['n_candidates'] - 1} cand"
            else:
                title = f"ep {ep}  NOT DETECTED"
                log.warning("[%s] episode %d: cube NOT detected -> NaN", key, ep)
            tiles.append(make_tile(rgb, det, title))
        n_det = sum(r["detected"] for r in rows if r["dataset"] == key)
        log.info("[%s] detected %d / %d", key, n_det, len(tiles))
        hdr = (f"{key}: first frames, {n_det}/{len(tiles)} detected | green box = chosen blob, "
               f"magenta + = centre, yellow = other candidates | zoom: {2 * CONTACT_CROP}px crop")
        save_contact_sheet(tiles, hdr, OUT_DIR / "contact_sheets" / f"contact_{key}.png")
    return pd.DataFrame(rows), frames


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    exp = df[(df.dataset == "expert") & df.detected]
    ref = float(exp.area.median()) if len(exp) else float(df[df.detected].area.median())
    df["area_ratio"] = df.area / ref
    flags = []
    for r in df.itertuples():
        f = []
        if not r.detected:
            f.append("not_detected")
        else:
            if not (AREA_RATIO_RANGE[0] <= r.area_ratio <= AREA_RATIO_RANGE[1]):
                f.append(f"area_ratio={r.area_ratio:.2f}")
            if r.aspect > MAX_ASPECT:
                f.append(f"aspect={r.aspect:.2f}")
            if r.second_area / r.area >= SECOND_BLOB_RATIO:
                f.append(f"second_blob={r.second_area:.0f}")
        flags.append(";".join(f))
    df["flag_reasons"] = flags
    df["flagged"] = df["flag_reasons"] != ""
    log.info("reference cube area (median of detected expert) = %.1f px^2", ref)
    return df


# =============================================================================
# Stage 2: test positions
# =============================================================================
def kmeans(X: np.ndarray, k: int, restarts: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=restarts, random_state=0).fit(X)
    return km.cluster_centers_, km.labels_


def label_positions(C: np.ndarray) -> list[str]:
    """2 columns (left/right by x) x 3 rows (top/mid/bottom by y) in image coordinates."""
    if len(C) != 6:
        return [f"P{i}" for i in range(len(C))]
    order_x = np.argsort(C[:, 0])
    labels = [None] * 6
    for side, idx in (("left", order_x[:3]), ("right", order_x[3:])):
        for row, i in zip(("top", "mid", "bottom"), idx[np.argsort(C[idx, 1])]):
            labels[i] = f"{row}-{side}"
    return labels


def cluster_test_positions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = df[df.dataset.isin(EVAL_KEYS) & df.detected].copy()
    X = ev[["cx", "cy"]].to_numpy(float)
    C, lab = kmeans(X, N_TEST_POSITIONS, KMEANS_RESTARTS)
    names = label_positions(C)
    ev["position"] = [names[i] for i in lab]
    ev["dist_to_center"] = np.linalg.norm(X - C[lab], axis=1)
    centers = pd.DataFrame(dict(position=names, cx=C[:, 0], cy=C[:, 1]))
    g = ev.groupby("position")
    centers["n_episodes"] = centers.position.map(g.size())
    centers["mean_dist_to_center"] = centers.position.map(g.dist_to_center.mean())
    centers["max_dist_to_center"] = centers.position.map(g.dist_to_center.max())
    centers["label_zh"] = centers.position.map({v: k for k, v in LABELS_ZH_EN.items()})
    # consistency: each eval dataset should have TRIALS_PER_POSITION per position
    ct = pd.crosstab(ev.dataset, ev.position)
    log.info("episodes per (eval dataset, test position):\n%s", ct.to_string())
    bad = ct[(ct != TRIALS_PER_POSITION).any(axis=1)]
    if len(bad) or ct.shape[1] != N_TEST_POSITIONS:
        log.warning("cluster counts are NOT %d per position for: %s -> check clustering / detections",
                    TRIALS_PER_POSITION, list(bad.index))
    maj = ev.groupby("episode_index").position.agg(lambda x: x.value_counts().idxmax())
    ev["majority_position"] = ev.episode_index.map(maj)
    dev = ev[ev.position != ev.majority_position]
    for r in dev.itertuples():
        log.warning("%s episode %d is at %s, but episode %d is at %s in most eval datasets",
                    r.dataset, r.episode_index, r.position, r.episode_index, r.majority_position)
    return centers, ev


# =============================================================================
# Stage 3: distances
# =============================================================================
def nearest(points: pd.DataFrame, cx: float, cy: float) -> tuple[float, str]:
    if points.empty:
        return np.nan, ""
    d = np.hypot(points.cx.to_numpy() - cx, points.cy.to_numpy() - cy)
    i = int(np.argmin(d))
    r = points.iloc[i]
    return float(d[i]), f"{r.dataset}:{int(r.episode_index)}"


def compute_distances(df: pd.DataFrame, centers: pd.DataFrame) -> pd.DataFrame:
    exp = df[(df.dataset == "expert") & df.detected]
    rec_all = df[(df.dataset == "recovery") & df.detected]
    rec_clean = rec_all[~rec_all.flagged] if EXCLUDE_FLAGGED_RECOVERY else rec_all
    out = []
    for c in centers.itertuples():
        d_e, n_e = nearest(exp, c.cx, c.cy)
        d_er, n_er = nearest(pd.concat([exp, rec_clean]), c.cx, c.cy)
        d_era, n_era = nearest(pd.concat([exp, rec_all]), c.cx, c.cy)
        s = SUCCESS.get(c.position)
        row = dict(position=c.position, label_zh=c.label_zh, cx=round(c.cx, 1), cy=round(c.cy, 1),
                   d_expert=round(d_e, 1), nearest_expert=n_e,
                   d_expert_recovery=round(d_er, 1), nearest_expert_recovery=n_er,
                   d_expert_recovery_all=round(d_era, 1), nearest_expert_recovery_all=n_era)
        if s:
            row.update({f"succ_{k}": v for k, v in zip(SUCCESS_COLS, s)})
        out.append(row)
    order = ["top-left", "mid-left", "bottom-left", "top-right", "mid-right", "bottom-right"]
    t = pd.DataFrame(out)
    t["_o"] = t.position.map({p: i for i, p in enumerate(order)})
    return t.sort_values("_o").drop(columns="_o").reset_index(drop=True)


# =============================================================================
# Plots
# =============================================================================
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b5b4ae", "axes.labelcolor": C_TEXT,
                         "xtick.color": C_TEXT_2, "ytick.color": C_TEXT_2, "axes.titleweight": "bold",
                         "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})
    return plt


def _two_panel(plt, bg: np.ndarray, title: str, draw):
    """Left: full frame with zoom box. Right: zoom on ZOOM_BOX. `draw(ax, zoomed)` adds the marks."""
    from matplotlib.patches import Rectangle
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    x0, y0, x1, y1 = ZOOM_BOX
    for ax, zoomed in ((a0, False), (a1, True)):
        ax.imshow(bg, alpha=0.75)
        draw(ax, zoomed)
        if zoomed:
            ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        else:
            ax.set_xlim(0, bg.shape[1]); ax.set_ylim(bg.shape[0], 0)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=C_TEXT, lw=1.2, ls="--"))
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    a0.set_title("Full frame (dashed box = zoom area)", fontsize=10, fontweight="normal")
    a1.set_title("Zoom", fontsize=10, fontweight="normal")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig, (a0, a1)


def _stars(ax, centers: pd.DataFrame, zoomed: bool, label=True):
    import matplotlib.patheffects as pe
    ax.scatter(centers.cx, centers.cy, marker="*", s=380 if zoomed else 150, c=C_TEST,
               edgecolors="white", linewidths=1.5 if zoomed else 0.8, zorder=5,
               label="Test position (cluster centre)" if label else None)
    if not zoomed:
        return
    for c in centers.itertuples():
        left = "left" in c.position
        ax.annotate(c.position, (c.cx, c.cy), xytext=(-14 if left else 14, 0), textcoords="offset points",
                    ha="right" if left else "left", va="center", fontsize=10, fontweight="bold",
                    color=C_TEXT, path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=6)


def plot_label_check(plt, bg, centers, ev, path):
    cmap = dict(zip(centers.position, zip(centers.cx, centers.cy)))

    def draw(ax, zoomed):
        for r in ev.itertuples():
            cx0, cy0 = cmap[r.position]
            ax.plot([r.cx, cx0], [r.cy, cy0], color=C_TEXT_2, lw=0.6, alpha=0.7, zorder=3)
        ax.scatter(ev.cx, ev.cy, s=16 if zoomed else 6, c="white", edgecolors=C_TEXT, linewidths=0.7,
                   zorder=4, label=f"Eval episode first frame (n={len(ev)})")
        _stars(ax, centers, zoomed)
        if zoomed:
            ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    fig, _ = _two_panel(plt, bg, "Eval first-frame cube positions clustered into 6 test positions", draw)
    fig.savefig(path); plt.close(fig)


def plot_distribution(plt, bg, df, centers, path):
    exp = df[(df.dataset == "expert") & df.detected]
    rec = df[(df.dataset == "recovery") & df.detected]
    rec_ok, rec_fl = rec[~rec.flagged], rec[rec.flagged]
    n_rec_nd = int(((df.dataset == "recovery") & ~df.detected).sum())

    def draw(ax, zoomed):
        k = 1.0 if zoomed else 0.45
        ax.scatter(exp.cx, exp.cy, s=40 * k, c=C_EXPERT, edgecolors="white", linewidths=1.2 * k,
                   zorder=4, label=f"Expert demo, first frame (n={len(exp)})")
        ax.scatter(rec_ok.cx, rec_ok.cy, s=64 * k, marker="^", c=C_RECOVERY, edgecolors="white",
                   linewidths=1.2 * k, zorder=4,
                   label=f"Recovery, cube at intervention start (n={len(rec_ok)}; {n_rec_nd} not visible)")
        if len(rec_fl):
            ax.scatter(rec_fl.cx, rec_fl.cy, s=64 * k, marker="^", facecolors="none", edgecolors=C_RECOVERY,
                       linewidths=1.5, zorder=4, label=f"Recovery, flagged anomalous (n={len(rec_fl)})")
        _stars(ax, centers, zoomed)
        if zoomed:
            ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    fig, _ = _two_panel(plt, bg, "Initial cube positions: training data vs. test positions", draw)
    fig.savefig(path); plt.close(fig)


def plot_success_vs_distance(plt, table, dist_col, succ_col, xlabel, title, path):
    import matplotlib.patheffects as pe
    from scipy.stats import spearmanr
    x = table[dist_col].to_numpy(float)
    y = table[succ_col].to_numpy(float) / TRIALS_PER_POSITION
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(x, y, s=70, c=C_EXPERT, edgecolors="white", linewidths=2, zorder=3)
    rho, p = spearmanr(x, y)
    ax.set_xlabel(xlabel); ax.set_ylabel("Success rate")
    ax.set_ylim(-0.08, 1.32); ax.set_xlim(0, max(x) * 1.45)
    ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.grid(True, color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_title(title, pad=20)
    ax.text(0.5, 1.015, f"Spearman rho = {rho:.2f} (p = {p:.2f}, n = {len(x)}; descriptive only)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color=C_TEXT_2)
    # direct labels: a label moves up one level (14 pt, with a leader line) while its estimated
    # horizontal extent overlaps an already placed label on the same level at the same y.
    x0, x1 = ax.get_xlim()
    ax_w_pt = ax.get_position().width * fig.get_figwidth() * 72
    placed = []  # (y, level, xstart, xend) in data units
    for i in np.lexsort((x, y)):
        label = f"{table.position.iloc[i]} ({table[succ_col].iloc[i]}/{TRIALS_PER_POSITION})"
        xs = x[i] + 7 / ax_w_pt * (x1 - x0)
        xe = xs + (len(label) * 5.6 + 6) / ax_w_pt * (x1 - x0)
        level = 0
        while any(abs(py - y[i]) < 0.1 and pl == level and xs < pe_ and xe > ps
                  for py, pl, ps, pe_ in placed):
            level += 1
        placed.append((y[i], level, xs, xe))
        if level:  # leader line drawn below all text so the white text halo masks crossings
            ax.annotate("", (x[i], y[i]), xytext=(9, 5 + 14 * level), textcoords="offset points",
                        zorder=2, arrowprops=dict(arrowstyle="-", color=C_TEXT_2, lw=0.6, shrinkA=0, shrinkB=4))
        ax.annotate(label, (x[i], y[i]), xytext=(7, 7 + 14 * level), textcoords="offset points",
                    fontsize=9, color=C_TEXT, zorder=4,
                    path_effects=[pe.withStroke(linewidth=4, foreground="white")])
    fig.savefig(path); plt.close(fig)
    return rho, p


# =============================================================================
# Report
# =============================================================================
def write_report(df, centers, table, corr, path, ev=None):
    L = ["# Red cube position analysis", ""]
    L += ["## Detection summary", "", "| dataset | episodes | detected | flagged |", "|---|---|---|---|"]
    for k, g in df.groupby("dataset", sort=False):
        L.append(f"| {k} | {len(g)} | {int(g.detected.sum())} | {int(g.flagged.sum())} |")
    L += ["", "## Episodes where the cube was NOT detected (cx/cy = NaN)", ""]
    nd = df[~df.detected]
    L += [f"- {r.dataset} episode {r.episode_index}" for r in nd.itertuples()] or ["- none"]
    L += ["", "## Recovery (HIL) data: anomalous first-frame detections", "",
          "Recovery episodes start at the moment of human intervention, so the cube may already "
          "have been moved or be inside the gripper. Flags: not_detected, area_ratio outside "
          f"{AREA_RATIO_RANGE} (vs. expert median), bbox aspect > {MAX_ASPECT}, "
          f"second red blob >= {SECOND_BLOB_RATIO} x largest.", ""]
    rec = df[(df.dataset == "recovery") & df.flagged]
    L += [f"- episode {r.episode_index}: {r.flag_reasons}" + ("" if not r.detected else
          f" (cx={r.cx:.0f}, cy={r.cy:.0f}, area={r.area:.0f})") for r in rec.itertuples()] or ["- none"]
    L += [f"", f"Flagged recovery episodes are "
          f"{'EXCLUDED from' if EXCLUDE_FLAGGED_RECOVERY else 'INCLUDED in'} `d_expert_recovery`; "
          "`d_expert_recovery_all` includes every detected recovery episode. "
          "Note: an unflagged detection only means the blob looks like a cube; whether it is still "
          "at its initial position must be checked visually on the contact sheet."]
    oth = df[(df.dataset != "recovery") & df.flagged & df.detected]
    L += ["", "## Other flagged detections (non-recovery datasets)", ""]
    L += [f"- {r.dataset} episode {r.episode_index}: {r.flag_reasons}" for r in oth.itertuples()] or ["- none"]
    if centers is not None:
        L += ["", "## Test positions (k-means on eval first frames)", "",
              centers.round(1).to_markdown(index=False), "",
              "Labels are assigned in IMAGE coordinates (left/right by x, top/mid/bottom by y). "
              "Verify against `test_positions_labels.png` before trusting the success mapping.", "",
              "### Trials whose first-frame position differs from the usual position of that episode index", "",
              "All eval datasets follow the same episode order (4 consecutive episodes per position). "
              "Listed trials were run with the cube at a different test position than the other datasets "
              "used for that episode index; the success table (x/4 per position) may count them under the "
              "planned position.", "",
              *([f"- {r.dataset} episode {r.episode_index}: at **{r.position}** "
                 f"(cx={r.cx:.0f}, cy={r.cy:.0f}); expected {r.majority_position}"
                 for r in ev[ev.position != ev.majority_position].itertuples()] or ["- none"]), "",
              "Episodes per (eval dataset, test position):", "",
              pd.crosstab(ev.dataset, ev.position).to_markdown(), "",
              "", "## Distance from each test position to the nearest training sample (pixels)", "",
              table.to_markdown(index=False), "",
              "Pixel distances are measured in the image and are not metric: perspective makes "
              "the same pixel distance correspond to a larger table distance at the top (far) "
              "side of the frame. "
              f"For scale: the cube side is ~{np.sqrt(df[(df.dataset == 'expert') & df.detected].area.median()):.0f} px "
              "(sqrt of median expert blob area).", "", "## Success rate vs. distance (smooth condition)", ""]
        for name, (rho, p, col) in corr.items():
            L.append(f"- {name}: x = `{col}`, Spearman rho = {rho:.2f}, p = {p:.2f} (n = 6, descriptive only)")
    path.write_text("\n".join(L) + "\n")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                    help="subset of datasets (analysis stages need expert, recovery and all eval sets)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(OUT_DIR / "run.log", mode="w")])
    log.info("output dir: %s", OUT_DIR.resolve())

    df, frames = run_detection(args.datasets)
    df = add_flags(df)
    df[["dataset", "episode_index", "cx", "cy", "area", "detected"]].to_csv(
        OUT_DIR / "cube_positions.csv", index=False, float_format="%.2f")
    df.to_csv(OUT_DIR / "cube_positions_detail.csv", index=False, float_format="%.2f")
    log.info("wrote cube_positions.csv (%d rows)", len(df))

    nd = df[~df.detected]
    log.info("NOT detected (%d): %s", len(nd),
             ", ".join(f"{r.dataset}:{r.episode_index}" for r in nd.itertuples()) or "none")
    fl = df[df.flagged & df.detected]
    for r in fl.itertuples():
        log.warning("flagged %s episode %d: %s", r.dataset, r.episode_index, r.flag_reasons)

    needed = {"expert", "recovery", *EVAL_KEYS}
    if not needed.issubset(args.datasets):
        log.info("analysis stages skipped (need all of: %s)", sorted(needed))
        write_report(df, None, None, {}, OUT_DIR / "report.md")
        return

    centers, ev = cluster_test_positions(df)
    ev[["dataset", "episode_index", "cx", "cy", "position", "majority_position", "dist_to_center"]].to_csv(
        OUT_DIR / "eval_cluster_assignments.csv", index=False, float_format="%.2f")
    centers.to_csv(OUT_DIR / "test_positions.csv", index=False, float_format="%.2f")
    table = compute_distances(df, centers)
    table.to_csv(OUT_DIR / "test_position_distances.csv", index=False)
    log.info("test position distances:\n%s", table.to_string(index=False))

    plt = _setup_mpl()
    bg = frames[BACKGROUND_FRAME]
    plot_label_check(plt, bg, centers, ev, OUT_DIR / "test_positions_labels.png")
    plot_distribution(plt, bg, df, centers, OUT_DIR / "position_distribution.png")
    corr = {}
    rho, p = plot_success_vs_distance(
        plt, table, "d_expert", "succ_expert_smooth",
        "Distance to nearest expert demo (px)", "Expert policy (smooth): success vs. distance",
        OUT_DIR / "success_vs_distance_expert_smooth.png")
    corr["expert_smooth"] = (rho, p, "d_expert")
    rho, p = plot_success_vs_distance(
        plt, table, "d_expert_recovery", "succ_mixed_smooth",
        "Distance to nearest expert or recovery sample (px)",
        "Mixed policy (smooth): success vs. distance",
        OUT_DIR / "success_vs_distance_mixed_smooth.png")
    corr["mixed_smooth"] = (rho, p, "d_expert_recovery")
    write_report(df, centers, table, corr, OUT_DIR / "report.md", ev)
    log.info("done. report: %s", (OUT_DIR / "report.md").resolve())


if __name__ == "__main__":
    main()

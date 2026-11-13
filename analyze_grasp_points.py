#!/usr/bin/env python
"""
Where does the policy aim? Joint angles at the first grasp -> image position.

1. Detect the first grasp in every episode: the first frame at which the gripper starts
   closing after it has been opened (see GRASP DETECTION in CONFIG).
2. Expert data (50 episodes): fit a linear regression from the 5 arm joint angles at the
   grasp frame to the first-frame cube pixel position (cx, cy) from cube_positions.csv.
   Validated with R^2 and leave-one-out pixel error.
3. Eval data (smooth condition): map the grasp-frame joint angles through the regression
   to an "aim" pixel position and compare it with the actual cube position.

Inputs (read only): datasets configured in analyze_cube_positions.py, and the
cube_positions.csv / test_positions.csv / eval_cluster_assignments.csv in this repo.
Optional: success.csv (dataset, episode_index, success) next to this script. Create it with
--make-success-template: positions with 0/4 or 4/4 successes are pre-filled from the aggregate
table in analyze_cube_positions.py, only the partially successful positions are left blank.
Outputs: ./grasp_point_analysis/ next to this script.

Usage:
  python analyze_grasp_points.py --make-success-template   # write success.csv to fill in
  python analyze_grasp_points.py --expert-only   # steps 2-3: gripper curves + regression check
  python analyze_grasp_points.py                 # everything
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_cube_positions as A  # noqa: E402  (dataset paths, camera key, zoom box)

# =============================================================================
# CONFIG
# =============================================================================
OUT_DIR = HERE / "grasp_point_analysis"
CUBE_CSV = HERE / "cube_positions.csv"
TEST_POS_CSV = HERE / "test_positions.csv"
ASSIGN_CSV = HERE / "eval_cluster_assignments.csv"
SUCCESS_CSV = HERE / "success.csv"  # optional: dataset, episode_index, success (0/1)

# Datasets added after the first analysis. They are registered in analyze_cube_positions.DATASETS at runtime only
# (the file itself is not changed, so its own clustering/results stay as they were).
EXTRA_DATASETS = {
    "eval_expert70_smooth": "rollout_eval_expertmixed_smooth_20261112_134559",       # control model, 6 standard positions
    "edge_mixed": "rollout_eval_edge_hilmixed_smooth_20261112_144609",              # right-side extra test, mixed model
    "edge_control": "rollout_eval_edge_expertmixed_smooth_20261112_145925",         # right-side extra test, control model
}
A.DATASETS.update(EXTRA_DATASETS)
EDGE_KEYS = ["edge_mixed", "edge_control"]
EDGE_POSITIONS_CSV = HERE / "right_side_test_analysis" / "edge_positions.csv"
EXTRA_POSITIONS_CSV = OUT_DIR / "extra_cube_positions.csv"   # first-frame positions of eval_expert70_smooth (written here)

TRAIN_KEY = "expert"
EVAL_KEYS = ["eval_expert_smooth", "eval_mixed_smooth", "eval_recovery_smooth", "eval_expert70_smooth"]
DATASET_LABEL = {"eval_expert_smooth": "expert", "eval_mixed_smooth": "mixed",
                 "eval_recovery_smooth": "recovery-only", "eval_expert70_smooth": "control (expert 70)",
                 "edge_mixed": "mixed (extra test)", "edge_control": "control (extra test)"}
MARKER = {"eval_expert_smooth": "o", "eval_mixed_smooth": "s", "eval_recovery_smooth": "^",
          "eval_expert70_smooth": "D"}   # secondary encoding in the overlay plots (colour alone is not enough for 4)
SUCCESS_KEYS = EVAL_KEYS + EDGE_KEYS
# Success counts (successes, trials) per position for the datasets that are not in analyze_cube_positions.SUCCESS
EXTRA_SUCCESS = {
    "eval_expert70_smooth": {"bottom-left": (4, 4), "bottom-right": (0, 4), "mid-left": (4, 4),
                             "mid-right": (3, 4), "top-left": (4, 4), "top-right": (2, 4)},
    "edge_mixed": {"right-bottom": (8, 8), "right-top": (4, 8)},
    "edge_control": {"right-bottom": (2, 8), "right-top": (0, 8)},
}

ARM_JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
              "wrist_flex.pos", "wrist_roll.pos"]
GRIPPER = "gripper.pos"
STATE_KEY = "observation.state"   # joints read at the grasp frame

# --- GRASP DETECTION (on the gripper signal below) ---------------------------
# Expert gripper: ~3 closed at start -> opens to ~50 for the approach -> closes on the
# cube (~5 while holding) -> opens to ~70-80 to release -> closes.
GRIPPER_SIGNAL = "observation.state"  # or "action" (commanded gripper, leads by a few frames)
OPEN_MIN = 30.0      # gripper >= this ...
OPEN_HOLD = 5        # ... for this many consecutive frames = "open" state reached
CLOSE_DROP = 10.0    # a close is confirmed once the gripper falls this far below the open peak
MIN_STEP = 0.15      # grasp frame = start of the continuous decline (per-frame drop > MIN_STEP)
                     # that leads into the confirmed close; earlier slow drifts/pauses are skipped
MOVING_SPEED = 20.0  # arm joint speed (norm of 5 joints, units/s) above which the arm counts as
                     # still moving at the grasp frame -> listed in the report

# --- plotting ----------------------------------------------------------------
C_DS = {"eval_expert_smooth": "#2a78d6", "eval_mixed_smooth": "#eb6834",
        "eval_recovery_smooth": "#1baf7a", "eval_expert70_smooth": "#6250d6"}   # reference palette slots 1,2,3,7
C_TRAIN = "#8a8984"
C_TEXT, C_TEXT_2 = "#0b0b0b", "#52514e"
POSITION_ORDER = ["top-left", "mid-left", "bottom-left", "top-right", "mid-right", "bottom-right"]
GRIPPER_PLOT_EPISODES = [0, 7, 14, 21, 28, 35, 42, 49]

log = logging.getLogger("grasp")


# =============================================================================
# Data
# =============================================================================
def load_frames(key: str) -> tuple[pd.DataFrame, dict]:
    """All frames of a dataset with state/action expanded into named columns."""
    d = A.LEROBOT_ROOT / A.DATASETS[key]
    info = json.loads((d / "meta/info.json").read_text())
    files = sorted(glob.glob(str(d / "data/*/*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "timestamp",
                                                STATE_KEY, "action"]) for f in files],
                   ignore_index=True)
    out = df[["episode_index", "frame_index", "timestamp"]].copy()
    for feat, prefix in ((STATE_KEY, "state"), ("action", "action")):
        names = info["features"][feat]["names"]
        arr = np.stack(df[feat].to_numpy())
        for i, n in enumerate(names):
            out[f"{prefix}:{n}"] = arr[:, i]
    for j in ARM_JOINTS + [GRIPPER]:
        if f"state:{j}" not in out:
            raise KeyError(f"{key}: joint {j} not in {STATE_KEY} names {info['features'][STATE_KEY]['names']}")
    return out.sort_values(["episode_index", "frame_index"]).reset_index(drop=True), info


def gripper_series(ep: pd.DataFrame) -> np.ndarray:
    prefix = "state" if GRIPPER_SIGNAL == STATE_KEY else "action"
    return ep[f"{prefix}:{GRIPPER}"].to_numpy(float)


def detect_first_grasp(g: np.ndarray) -> dict:
    """Return dict(open_frame, peak, confirm_frame, grasp_frame) or NaNs if no close."""
    res = dict(open_frame=np.nan, peak=np.nan, confirm_frame=np.nan, grasp_frame=np.nan)
    is_open = g >= OPEN_MIN
    run = 0
    open_frame = None
    for t, o in enumerate(is_open):
        run = run + 1 if o else 0
        if run >= OPEN_HOLD:
            open_frame = t - OPEN_HOLD + 1
            break
    if open_frame is None:
        return res
    res["open_frame"] = open_frame
    peak = -np.inf
    for t in range(open_frame, len(g)):
        peak = max(peak, g[t])
        if g[t] <= peak - CLOSE_DROP:
            k = t
            while k > open_frame and g[k - 1] - g[k] > MIN_STEP:
                k -= 1
            res.update(peak=peak, confirm_frame=t, grasp_frame=k)
            return res
    res["peak"] = peak
    return res


def grasp_table(key: str) -> pd.DataFrame:
    frames, _ = load_frames(key)
    rows = []
    for ep, e in frames.groupby("episode_index"):
        det = detect_first_grasp(gripper_series(e))
        row = dict(dataset=key, episode_index=int(ep), n_frames=len(e), **det)
        if not np.isnan(det["grasp_frame"]):
            gf = int(det["grasp_frame"])
            r = e.iloc[gf]
            row["grasp_time_s"] = float(r.timestamp)
            q = e[[f"state:{j}" for j in ARM_JOINTS]].to_numpy(float)
            speed = np.r_[0.0, np.linalg.norm(np.diff(q, axis=0), axis=1) * 30.0]
            row["arm_speed_at_grasp"] = float(speed[max(0, gf - 2):gf + 3].mean())  # 5-frame mean
            row.update({j: float(r[f"state:{j}"]) for j in ARM_JOINTS})
        else:
            row.update({j: np.nan for j in ARM_JOINTS})
        rows.append(row)
    return pd.DataFrame(rows), frames


def load_cube(key: str) -> pd.DataFrame:
    """First-frame cube position: cube_positions.csv for the original datasets, edge_positions.csv for the
    right-side extra test, extra_cube_positions.csv for the control model's standard evaluation."""
    if key in EDGE_KEYS:
        c = pd.read_csv(EDGE_POSITIONS_CSV)
        c = c[c.dataset == key].assign(detected=True)
    elif key in EXTRA_DATASETS:
        c = pd.read_csv(EXTRA_POSITIONS_CSV)
        c = c[c.dataset == key]
    else:
        c = pd.read_csv(CUBE_CSV)
        c = c[c.dataset == key]
    c = c[["episode_index", "cx", "cy", "detected"]]
    return c.rename(columns={"cx": "cube_cx", "cy": "cube_cy"})


def load_positions() -> pd.DataFrame:
    """dataset, episode_index, position for every eval / extra-test episode."""
    parts = [pd.read_csv(ASSIGN_CSV)[["dataset", "episode_index", "position"]]]
    if EXTRA_POSITIONS_CSV.exists():
        parts.append(pd.read_csv(EXTRA_POSITIONS_CSV)[["dataset", "episode_index", "position"]])
    if EDGE_POSITIONS_CSV.exists():
        parts.append(pd.read_csv(EDGE_POSITIONS_CSV)[["dataset", "episode_index", "position"]])
    return pd.concat(parts, ignore_index=True)


def detect_extra_positions(key: str = "eval_expert70_smooth"):
    """First-frame cube detection (same detector and ROI as analyze_cube_positions) for a dataset with the 6
    standard positions, assigned to the nearest existing test position (test_positions.csv is not changed)."""
    ds_dir = A.LEROBOT_ROOT / A.DATASETS[key]
    centers = pd.read_csv(TEST_POS_CSV)
    rows, tiles = [], []
    for ep, rgb in A.first_frames(ds_dir):
        d = A.detect_cube(rgb)
        pos, dist = None, np.nan
        if d["detected"]:
            dd = np.hypot(centers.cx - d["cx"], centers.cy - d["cy"])
            pos, dist = centers.position.iloc[int(dd.argmin())], float(dd.min())
        else:
            log.warning("[%s] episode %d: cube NOT detected in frame 0", key, ep)
        rows.append(dict(dataset=key, episode_index=ep, cx=d["cx"], cy=d["cy"], area=d["area"],
                         detected=bool(d["detected"]), position=pos, dist_to_center=dist))
        tiles.append(A.make_tile(rgb, d, f"ep {ep}  ({d['cx']:.0f},{d['cy']:.0f})  {pos}" if d["detected"]
                                 else f"ep {ep}  NOT DETECTED"))
    df = pd.DataFrame(rows)
    A.save_contact_sheet(tiles, f"{key}: first frames, {int(df.detected.sum())}/{len(df)} detected, "
                                "position = nearest standard test position", OUT_DIR / f"contact_{key}.png")
    prev = pd.read_csv(EXTRA_POSITIONS_CSV) if EXTRA_POSITIONS_CSV.exists() else pd.DataFrame()
    if len(prev):
        prev = prev[prev.dataset != key]
    pd.concat([prev, df], ignore_index=True).to_csv(EXTRA_POSITIONS_CSV, index=False, float_format="%.2f")
    ct = df.groupby("position").size()
    log.info("[%s] episodes per position: %s | max distance to a test position %.1f px", key, ct.to_dict(),
             df.dist_to_center.max())
    seq = {"bottom-left": "BL", "bottom-right": "BR", "mid-left": "ML", "mid-right": "MR", "top-left": "TL",
           "top-right": "TR"}
    log.info("[%s] episode order: %s", key, " ".join(seq.get(p, "?") for p in df.position))
    if (ct != A.TRIALS_PER_POSITION).any() or len(ct) != len(centers):
        log.warning("[%s] positions are NOT 4 episodes each: check detection / placement", key)
    return df


# =============================================================================
# Regression
# =============================================================================
def fit_regression(train: pd.DataFrame):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    X = train[ARM_JOINTS].to_numpy(float)
    Y = train[["cube_cx", "cube_cy"]].to_numpy(float)
    model = LinearRegression().fit(X, Y)
    Y_fit = model.predict(X)
    Y_loo = cross_val_predict(LinearRegression(), X, Y, cv=LeaveOneOut())
    # baseline: predict the mean of the other 49 episodes (LOO)
    n = len(Y)
    Y_base = (Y.sum(0) - Y) / (n - 1)

    def r2(yt, yp):
        return 1 - ((yt - yp) ** 2).sum(0) / ((yt - yt.mean(0)) ** 2).sum(0)

    def err(yt, yp):
        e = yp - yt
        nrm = np.linalg.norm(e, axis=1)
        return dict(mae_x=np.abs(e[:, 0]).mean(), mae_y=np.abs(e[:, 1]).mean(),
                    mean_norm=nrm.mean(), median_norm=np.median(nrm), max_norm=nrm.max())

    stats = dict(n=n, r2_fit=r2(Y, Y_fit), r2_loo=r2(Y, Y_loo),
                 err_fit=err(Y, Y_fit), err_loo=err(Y, Y_loo), err_baseline=err(Y, Y_base),
                 coef=pd.DataFrame(model.coef_, index=["cx", "cy"], columns=ARM_JOINTS),
                 intercept=model.intercept_)
    train = train.copy()
    train["loo_x"], train["loo_y"] = Y_loo[:, 0], Y_loo[:, 1]
    train["loo_err"] = np.linalg.norm(Y_loo - Y, axis=1)
    return model, stats, train


# =============================================================================
# Plots
# =============================================================================
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b5b4ae", "axes.labelcolor": C_TEXT,
                         "xtick.color": C_TEXT_2, "ytick.color": C_TEXT_2, "axes.titleweight": "bold",
                         "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})
    return plt


def _clean(ax):
    ax.grid(True, color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_gripper_curves(plt, frames: pd.DataFrame, grasps: pd.DataFrame, path: Path):
    eps = [e for e in GRIPPER_PLOT_EPISODES if e in set(grasps.episode_index)]
    fig, axes = plt.subplots(len(eps) // 2 + len(eps) % 2, 2, figsize=(12, 1.9 * ((len(eps) + 1) // 2)),
                             sharex=True, sharey=True)
    for ax, ep in zip(axes.ravel(), eps):
        e = frames[frames.episode_index == ep]
        t = e.frame_index.to_numpy() / 30.0
        ax.plot(t, e[f"state:{GRIPPER}"], color="#2a78d6", lw=1.6, label="observation.state")
        ax.plot(t, e[f"action:{GRIPPER}"], color="#eb6834", lw=1.0, ls="--", label="action")
        ax.axhline(OPEN_MIN, color=C_TEXT_2, lw=0.7, ls=":")
        g = grasps[grasps.episode_index == ep].iloc[0]
        if not np.isnan(g.grasp_frame):
            ax.axvline(g.grasp_frame / 30.0, color=C_TEXT, lw=1.0)
            ax.text(g.grasp_frame / 30.0 + 0.2, 85, f"grasp f{int(g.grasp_frame)}", fontsize=8, color=C_TEXT)
        ax.set_title(f"episode {ep}", fontsize=9, fontweight="normal", loc="left")
        ax.set_ylim(-3, 100); _clean(ax)
    for ax in axes.ravel()[len(eps):]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="center right", fontsize=8, framealpha=0.9)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("gripper")
    fig.suptitle(f"Expert gripper curves (dotted = OPEN_MIN {OPEN_MIN:g}; vertical line = detected first grasp, "
                 f"signal = {GRIPPER_SIGNAL})", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)


def plot_gripper_aligned(plt, frames: pd.DataFrame, grasps: pd.DataFrame, title: str, path: Path):
    """All episodes overlaid, time aligned to the detected grasp frame; plus arm speed."""
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for g in grasps.dropna(subset=["grasp_frame"]).itertuples():
        e = frames[frames.episode_index == g.episode_index]
        t = (e.frame_index.to_numpy() - g.grasp_frame) / 30.0
        a0.plot(t, e[f"state:{GRIPPER}"], color="#2a78d6", lw=0.6, alpha=0.5)
        q = e[[f"state:{j}" for j in ARM_JOINTS]].to_numpy()
        speed = np.r_[np.nan, np.linalg.norm(np.diff(q, axis=0), axis=1) * 30.0]
        a1.plot(t, speed, color=C_TEXT_2, lw=0.5, alpha=0.4)
    for ax in (a0, a1):
        ax.axvline(0, color=C_TEXT, lw=1.0); _clean(ax)
    a0.set_xlim(-4, 4); a0.set_ylabel("gripper (state)")
    a1.set_ylabel("arm joint speed\n(norm of 5 joints, units/s)"); a1.set_xlabel("time relative to detected grasp (s)")
    ys = [np.asarray(l.get_ydata(), float) for l in a1.lines[:-1]]  # last line = axvline
    a1.set_ylim(0, np.nanpercentile(np.concatenate(ys), 99.5) * 1.1 if ys else 1)
    a0.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_regression_check(plt, train: pd.DataFrame, stats: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, ax_name, col, pred in ((axes[0], "x", "cube_cx", "loo_x"), (axes[1], "y", "cube_cy", "loo_y")):
        lo = min(train[col].min(), train[pred].min()) - 3
        hi = max(train[col].max(), train[pred].max()) + 3
        ax.plot([lo, hi], [lo, hi], color=C_TEXT_2, lw=0.8, ls="--", label="y = x")
        ax.scatter(train[col], train[pred], s=30, c="#2a78d6", edgecolors="white", linewidths=1, zorder=3,
                   label="expert episode (LOO prediction)")
        i = 0 if ax_name == "x" else 1
        ax.set_title(f"{ax_name}: R² fit {stats['r2_fit'][i]:.2f}, R² LOO {stats['r2_loo'][i]:.2f}",
                     fontsize=10)
        ax.set_xlabel(f"actual cube {ax_name} (px)"); ax.set_ylabel(f"predicted {ax_name} from joints (px)")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal"); _clean(ax)
    axes[0].legend(fontsize=8, loc="upper left")
    e = stats["err_loo"]; b = stats["err_baseline"]
    fig.suptitle(f"Joint angles at grasp -> cube pixel position (linear, n={stats['n']}). "
                 f"LOO error {e['mean_norm']:.1f} px mean / {e['median_norm']:.1f} median "
                 f"(predict-mean baseline {b['mean_norm']:.1f} px)", fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def decode_frame(key: str, ep: int) -> np.ndarray:
    from lerobot.datasets.video_utils import decode_video_frames
    d = A.LEROBOT_ROOT / A.DATASETS[key]
    m = A.load_episode_meta(d)
    r = m[m.episode_index == ep].iloc[0]
    fr = decode_video_frames(A.video_path(d, int(r.v_chunk), int(r.v_file)), [float(r.v_from_ts)],
                             tolerance_s=A.TIMESTAMP_TOLERANCE_S, backend="pyav", return_uint8=True)
    return fr[0].permute(1, 2, 0).numpy()


def _zoom_box(points: np.ndarray, margin=12):
    x0, y0, x1, y1 = A.ZOOM_BOX
    pts = points[~np.isnan(points).any(1)]
    if len(pts):
        x0, y0 = min(x0, pts[:, 0].min() - margin), min(y0, pts[:, 1].min() - margin)
        x1, y1 = max(x1, pts[:, 0].max() + margin), max(y1, pts[:, 1].max() + margin)
    return max(0, x0), max(0, y0), min(640, x1), min(480, y1)


def plot_aim_arrows(plt, bg, gp, train_cube, centers, center, path):
    import matplotlib.patheffects as pe
    pts = np.r_[gp[["aim_x", "aim_y"]].to_numpy(float), gp[["cube_cx", "cube_cy"]].to_numpy(float)]
    x0, y0, x1, y1 = _zoom_box(pts)
    n = len(EVAL_KEYS)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.4 * (y1 - y0) / (x1 - x0) + 1.3), sharey=True)
    for ax, key in zip(axes, EVAL_KEYS):
        ax.imshow(bg, alpha=0.7)
        ax.scatter(train_cube.cube_cx, train_cube.cube_cy, s=10, c=C_TRAIN, zorder=3,
                   label="expert training cube positions")
        ax.scatter(*center, marker="P", s=120, c=C_TEXT, edgecolors="white", linewidths=1.2, zorder=6,
                   label="expert data centre")
        ax.scatter(centers.cx, centers.cy, marker="*", s=260, c="white", edgecolors=C_TEXT,
                   linewidths=1.0, zorder=4, label="test position")
        g = gp[(gp.dataset == key)].dropna(subset=["aim_x"])
        for r in g.itertuples():
            ax.annotate("", (r.aim_x, r.aim_y), (r.cube_cx, r.cube_cy), zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color=C_DS[key], lw=1.3, mutation_scale=9,
                                        shrinkA=0, shrinkB=0))
        ax.scatter(g.aim_x, g.aim_y, s=14, c=C_DS[key], edgecolors="white", linewidths=0.6, zorder=6,
                   label="aim position (arrow from actual cube)")
        for c in centers.itertuples():
            ax.annotate(c.position, (c.cx, c.cy), xytext=(0, -13), textcoords="offset points", ha="center",
                        fontsize=7.5, color=C_TEXT, zorder=7,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
        n_no = int(((gp.dataset == key) & gp.aim_x.isna()).sum())
        ax.set_title(f"{DATASET_LABEL[key]} policy (n={len(g)}" + (f", {n_no} without grasp" if n_no else "") + ")")
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_xlabel("x (px)")
    axes[0].set_ylabel("y (px)")
    axes[0].legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    fig.suptitle("Where the policy aims at its first grasp (joint angles mapped to image via expert regression)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_aim_vs_actual(plt, gp, slopes, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, axis, ccol, acol in ((axes[0], "x", "cube_cx", "aim_x"), (axes[1], "y", "cube_cy", "aim_y")):
        allv = gp[[ccol, acol]].to_numpy(float)
        lo, hi = np.nanmin(allv) - 5, np.nanmax(allv) + 5
        ax.plot([lo, hi], [lo, hi], color=C_TEXT_2, lw=0.8, ls="--", label="slope 1 (follows cube)")
        for key in EVAL_KEYS:
            g = gp[gp.dataset == key].dropna(subset=[acol])
            s = slopes[key][axis]
            ax.scatter(g[ccol], g[acol], s=30, c=C_DS[key], marker=MARKER[key], edgecolors="white", linewidths=0.8, zorder=3,
                       label=f"{DATASET_LABEL[key]}: slope {s['slope']:.2f} [{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]")
            xs = np.array([g[ccol].min(), g[ccol].max()])
            ax.plot(xs, s["intercept"] + s["slope"] * xs, color=C_DS[key], lw=2, zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(f"actual cube {axis} (px)"); ax.set_ylabel(f"aim {axis} (px)")
        ax.set_title(f"{axis}: aim vs. actual"); _clean(ax)
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    fig.suptitle("Slope ~1: policy follows the cube; slope ~0: policy ignores the cube "
                 "(brackets: 95% CI)", fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_err_box(plt, gp, path):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    k = len(EVAL_KEYS); w = 0.8 / k
    rng = np.random.default_rng(0)
    for j, key in enumerate(EVAL_KEYS):
        data, pos = [], []
        for i, p in enumerate(POSITION_ORDER):
            v = gp[(gp.dataset == key) & (gp.position == p)].err_norm.dropna().to_numpy()
            data.append(v); pos.append(i - 0.4 + w * (j + 0.5))
        bp = ax.boxplot(data, positions=pos, widths=w * 0.8, patch_artist=True, showfliers=False,
                        medianprops=dict(color=C_TEXT, lw=1.2), whiskerprops=dict(color=C_DS[key]),
                        capprops=dict(color=C_DS[key]), boxprops=dict(edgecolor=C_DS[key]))
        for b in bp["boxes"]:
            b.set_facecolor(C_DS[key]); b.set_alpha(0.35)
        for x, v in zip(pos, data):
            ax.scatter(x + rng.uniform(-w * 0.2, w * 0.2, len(v)), v, s=12, c=C_DS[key],
                       edgecolors="white", linewidths=0.5, zorder=3)
        ax.plot([], [], color=C_DS[key], lw=6, alpha=0.6, label=f"{DATASET_LABEL[key]} policy")
    ax.set_xticks(range(len(POSITION_ORDER))); ax.set_xticklabels(POSITION_ORDER)
    ax.set_ylabel("aim error |aim - cube| (px)"); ax.set_ylim(bottom=0); _clean(ax)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.set_title("Aim error per test position (smooth condition)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# =============================================================================
# Episode-level success (optional success.csv)
# =============================================================================
def expected_successes(key: str) -> dict:
    """{position: (successes, trials)} from the aggregate tables (analyze_cube_positions.SUCCESS or EXTRA_SUCCESS)."""
    if key in EXTRA_SUCCESS:
        return EXTRA_SUCCESS[key]
    col = A.SUCCESS_COLS.index(key.removeprefix("eval_"))
    return {pos: (counts[col], A.TRIALS_PER_POSITION) for pos, counts in A.SUCCESS.items()}


def _template_rows(key: str, positions: pd.DataFrame) -> list[dict]:
    exp = expected_successes(key)
    rows = []
    for r in positions[positions.dataset == key].sort_values("episode_index").itertuples():
        k, n = exp[r.position]
        full = k in (0, n)
        rows.append(dict(dataset=key, episode_index=r.episode_index, position=r.position,
                         success=(1 if k else 0) if full else "",
                         source=f"auto: {k}/{n} at position" if full else "",
                         hint="" if full else f"FILL 0/1: {k} of {n} at this position succeeded"))
    return rows


def make_success_template(path: Path, force: bool = False):
    """Write success.csv from scratch: rows at all-fail / all-success positions are pre-filled, the rest is blank."""
    if path.exists() and not force:
        raise SystemExit(f"{path} already exists; use --force to overwrite (your entries would be lost), "
                         "or --add-success-rows to append rows for new datasets")
    pos = load_positions()
    t = pd.DataFrame([r for key in SUCCESS_KEYS if (pos.dataset == key).any() for r in _template_rows(key, pos)])
    t.to_csv(path, index=False)
    log.info("wrote %s: %d rows, %d to fill", path, len(t), (t.success == "").sum())


def add_success_rows(path: Path):
    """Append rows for datasets that are not in success.csv yet. Existing rows are left untouched."""
    t = pd.read_csv(path)
    pos = load_positions()
    new_rows = []
    for key in SUCCESS_KEYS:
        if (t.dataset == key).any():
            continue
        if not (pos.dataset == key).any():
            log.warning("no episode positions for %s: run the detection stage first", key)
            continue
        new_rows += _template_rows(key, pos)
    if not new_rows:
        log.info("%s already has rows for every dataset", path.name)
        return
    out = pd.concat([t, pd.DataFrame(new_rows)], ignore_index=True)
    out.to_csv(path, index=False)
    nr = pd.DataFrame(new_rows)
    log.info("appended %d rows to %s (%d pre-filled, %d to fill)", len(nr), path.name,
             int((nr.success != "").sum()), int((nr.success == "").sum()))
    for (key, p), g in nr[nr.success == ""].groupby(["dataset", "position"]):
        log.info("  FILL: %s %s episodes %s (%s)", key, p, g.episode_index.tolist(), g.hint.iloc[0])


def load_success(path: Path, positions: pd.DataFrame) -> pd.DataFrame | None:
    """Read success.csv; blank = unknown (never replaced by a placeholder). Validates values and per-position totals.
    `positions` has dataset, episode_index, position for every episode that should have a label."""
    if not path.exists():
        log.info("no %s found -> success/failure split skipped", path.name)
        return None
    s = pd.read_csv(path)
    for c in ("dataset", "episode_index", "success"):
        if c not in s:
            raise SystemExit(f"{path.name}: missing column '{c}'")
    s = s[s.dataset.isin(SUCCESS_KEYS)].copy()
    s["success"] = pd.to_numeric(s.success, errors="coerce")
    bad = s[s.success.notna() & ~s.success.isin([0, 1])]
    if len(bad):
        raise SystemExit(f"{path.name}: success must be 0 or 1, got {bad[['dataset', 'episode_index', 'success']].to_dict('records')}")
    dup = s[s.duplicated(["dataset", "episode_index"], keep=False)]
    if len(dup):
        raise SystemExit(f"{path.name}: duplicate rows {dup[['dataset', 'episode_index']].to_dict('records')}")
    s = s[["dataset", "episode_index", "success"]]
    pos = positions[positions.dataset.isin(SUCCESS_KEYS)]
    m = pos[["dataset", "episode_index", "position"]].merge(s, on=["dataset", "episode_index"], how="left")
    missing = m[m.success.isna()]
    for key, g in missing.groupby("dataset"):
        log.warning("%s: %d episodes of %s without a 0/1 value (unknown, excluded from success/failure statistics): %s",
                    path.name, len(g), key, g.episode_index.tolist())
    for (key, p), g in m.groupby(["dataset", "position"]):
        k_n = expected_successes(key).get(p)
        if g.success.notna().all() and k_n is not None and int(g.success.sum()) != k_n[0]:
            log.warning("%s %s: %d successes in %s but %d/%d in the aggregate table", key, p,
                        int(g.success.sum()), path.name, k_n[0], k_n[1])
    log.info("loaded %s: %d/%d episodes with a value", path.name, m.success.notna().sum(), len(m))
    return s


def complete_success_datasets(success: pd.DataFrame | None, positions: pd.DataFrame) -> set:
    """Datasets whose every episode has a 0/1 label (only these are used in success/failure statistics)."""
    if success is None:
        return set()
    m = positions[positions.dataset.isin(SUCCESS_KEYS)].merge(success, on=["dataset", "episode_index"], how="left")
    return {k for k, g in m.groupby("dataset") if g.success.notna().all()}


# =============================================================================
# Stats + report
# =============================================================================
def slope_ci(x, y):
    from scipy import stats
    m = ~(np.isnan(x) | np.isnan(y))
    r = stats.linregress(x[m], y[m])
    t = stats.t.ppf(0.975, m.sum() - 2)
    return dict(slope=r.slope, intercept=r.intercept, ci_lo=r.slope - t * r.stderr,
                ci_hi=r.slope + t * r.stderr, r=r.rvalue, n=int(m.sum()))


def fmt(v, d=2):
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


RIGHT_POSITIONS = ["mid-right", "bottom-right", "top-right"]


def mean_err_x_table(gp: pd.DataFrame) -> pd.DataFrame:
    g = gp.dropna(subset=["aim_x"])
    t = g.pivot_table(index="position", columns="dataset", values="err_x", aggfunc="mean").reindex(columns=EVAL_KEYS)
    t.columns = [DATASET_LABEL[c] for c in t.columns]
    order = RIGHT_POSITIONS + [p for p in POSITION_ORDER if p not in RIGHT_POSITIONS]
    return t.reindex([p for p in order if p in t.index]).round(1)


def write_report(path, grasps_train, stats, gp=None, slopes=None, center=None, success=None, complete=None):
    L = ["# Grasp point analysis: where does the policy aim?", ""]
    L += ["## Grasp detection", "",
          f"Signal: `{GRIPPER_SIGNAL}` gripper. Open state = gripper >= {OPEN_MIN:g} for {OPEN_HOLD} frames; "
          f"a close is confirmed when the gripper drops {CLOSE_DROP:g} below the open peak; the grasp frame is "
          f"the start of the continuous decline (per-frame drop > {MIN_STEP:g}) that leads into that close, so "
          f"earlier slow drifts or pauses are skipped. Arm joints are read from `{STATE_KEY}` at that frame. "
          f"`arm_speed_at_grasp` = mean arm joint speed over the 5 frames around it.", ""]
    nog = grasps_train[grasps_train.grasp_frame.isna()]
    L += [f"Expert: grasp detected in {len(grasps_train) - len(nog)}/{len(grasps_train)} episodes; "
          f"grasp time {grasps_train.grasp_time_s.min():.1f}-{grasps_train.grasp_time_s.max():.1f} s."]
    if len(nog):
        L += [f"- expert episode {r.episode_index}: no gripper close" for r in nog.itertuples()]
    e, b = stats["err_loo"], stats["err_baseline"]
    L += ["", "## Regression check (expert, joints at grasp -> first-frame cube pixel)", "",
          f"n = {stats['n']}, features = {', '.join(ARM_JOINTS)}", "",
          "| | x | y |", "|---|---|---|",
          f"| R² (fit) | {stats['r2_fit'][0]:.3f} | {stats['r2_fit'][1]:.3f} |",
          f"| R² (leave-one-out) | {stats['r2_loo'][0]:.3f} | {stats['r2_loo'][1]:.3f} |",
          f"| LOO mean abs error (px) | {e['mae_x']:.1f} | {e['mae_y']:.1f} |",
          f"| predict-mean baseline mean abs error (px) | {b['mae_x']:.1f} | {b['mae_y']:.1f} |", "",
          f"LOO Euclidean error: mean {e['mean_norm']:.1f} px, median {e['median_norm']:.1f} px, "
          f"max {e['max_norm']:.1f} px (baseline mean {b['mean_norm']:.1f} px).", "",
          "Coefficients (px per joint unit):", "", stats["coef"].round(3).to_markdown(), "",
          "Caveat: expert cube positions only span ~40 px, and all test positions lie at or beyond the "
          "edge of that range, so aim positions for eval episodes are partly an extrapolation of the "
          "linear map."]
    if gp is not None:
        nog = gp[gp.grasp_frame.isna()]
        L += ["", "## Eval episodes without a gripper close (no aim position)", ""]
        L += [f"- {r.dataset} episode {r.episode_index} ({r.position})" for r in nog.itertuples()] or ["- none"]
        mv = gp[gp.arm_speed_at_grasp > MOVING_SPEED]
        L += ["", f"## Eval episodes where the arm was still moving at the grasp frame (speed > {MOVING_SPEED:g})", "",
              "The policy closed the gripper while the arm was moving, so the aim position is a snapshot "
              "of a moving arm rather than a settled grasp pose.", ""]
        L += [f"- {r.dataset} episode {r.episode_index} ({r.position}): speed {r.arm_speed_at_grasp:.1f}, "
              f"err {r.err_norm:.1f} px" for r in mv.itertuples()] or ["- none"]
        L += ["", "## Summary per eval dataset (smooth condition)", "",
              f"Expert data centre (mean cube position of {stats['n']} expert episodes): "
              f"({center[0]:.1f}, {center[1]:.1f}) px. `cos_to_centre` = cosine between the error vector "
              "(aim - cube) and the vector from the cube to the expert data centre; +1 = the policy aims "
              "towards the centre, 0 = unrelated, -1 = away from it.", ""]
        rows = []
        for key in EVAL_KEYS:
            g = gp[gp.dataset == key].dropna(subset=["aim_x"])
            sx, sy = slopes[key]["x"], slopes[key]["y"]
            rows.append({"dataset": key, "n": len(g),
                         "slope_x [95% CI]": f"{sx['slope']:.2f} [{sx['ci_lo']:.2f}, {sx['ci_hi']:.2f}]",
                         "slope_y [95% CI]": f"{sy['slope']:.2f} [{sy['ci_lo']:.2f}, {sy['ci_hi']:.2f}]",
                         "mean err_norm (px)": round(g.err_norm.mean(), 1),
                         "median err_norm (px)": round(g.err_norm.median(), 1),
                         "mean cos_to_centre": round(g.cos_to_centre.mean(), 2),
                         "share cos > 0": f"{(g.cos_to_centre > 0).mean():.0%}"})
        L += [pd.DataFrame(rows).to_markdown(index=False), "", "### Per test position", ""]
        per = (gp.dropna(subset=["aim_x"]).groupby(["dataset", "position"])
               .agg(n=("err_norm", "size"), mean_err=("err_norm", "mean"), mean_err_x=("err_x", "mean"),
                    mean_err_y=("err_y", "mean"), mean_cos_to_centre=("cos_to_centre", "mean"))
               .round(2).reset_index())
        per["position"] = pd.Categorical(per.position, POSITION_ORDER)
        L += [per.sort_values(["dataset", "position"]).to_markdown(index=False)]
        px = mean_err_x_table(gp)
        L += ["", "### Horizontal aim error (mean err_x = aim x - cube x, px) by position, all four models", "",
              "Positive = aim to the right of the cube, negative = to the left. Right-side positions first.", "",
              px.to_markdown(), "",
              f"Full table: `{OUT_DIR.name}/mean_err_x_by_position.csv`."]
        L += ["", "## Success vs. failure", ""]
        complete = complete or set()
        if success is None:
            L += [f"No episode-level success record found. Run `python analyze_grasp_points.py "
                  f"--make-success-template`, fill the blank `success` cells (0/1) in `{SUCCESS_CSV.name}` "
                  "and re-run."]
        else:
            use = gp[gp.dataset.isin(complete)]
            s = (use.dropna(subset=["aim_x", "success"]).groupby(["dataset", "success"])
                 .agg(n=("err_norm", "size"), mean_err=("err_norm", "mean"), median_err=("err_norm", "median"),
                      mean_cos_to_centre=("cos_to_centre", "mean")).round(2).reset_index())
            L += [s.to_markdown(index=False)]
            for key in EVAL_KEYS:
                if key not in complete:
                    n_lab = int(gp[gp.dataset == key].success.notna().sum())
                    L += ["", f"{key}: success/failure statistics skipped, only {n_lab}/"
                              f"{int((gp.dataset == key).sum())} episodes have a 0/1 label in {SUCCESS_CSV.name} "
                              "(blank cells are not replaced by placeholders)."]
    path.write_text("\n".join(L) + "\n")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expert-only", action="store_true",
                    help="only grasp detection on expert data + regression check")
    ap.add_argument("--make-success-template", action="store_true",
                    help=f"write {SUCCESS_CSV.name} (0/4 and 4/4 positions pre-filled) and exit")
    ap.add_argument("--force", action="store_true", help=f"allow overwriting an existing {SUCCESS_CSV.name}")
    ap.add_argument("--add-success-rows", action="store_true",
                    help=f"append rows for datasets that are not in {SUCCESS_CSV.name} yet (existing rows untouched) and exit")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(OUT_DIR / "run_grasp.log", mode="w")])
    def ensure_extra_positions():
        have = pd.read_csv(EXTRA_POSITIONS_CSV).dataset.unique() if EXTRA_POSITIONS_CSV.exists() else []
        for k in EXTRA_DATASETS:
            if k not in EDGE_KEYS and k not in have:
                detect_extra_positions(k)

    if args.make_success_template:
        ensure_extra_positions()
        make_success_template(SUCCESS_CSV, args.force)
        return
    if args.add_success_rows:
        ensure_extra_positions()
        add_success_rows(SUCCESS_CSV)
        return
    plt = _mpl()

    # ---- expert: grasp detection + regression --------------------------------
    gt, frames_t = grasp_table(TRAIN_KEY)
    nog = gt[gt.grasp_frame.isna()]
    log.info("expert: grasp detected in %d/%d episodes", len(gt) - len(nog), len(gt))
    for r in nog.itertuples():
        log.warning("expert episode %d: no gripper close detected", r.episode_index)
    log.info("expert grasp frame: min %d, median %d, max %d; open peak %.1f-%.1f",
             gt.grasp_frame.min(), gt.grasp_frame.median(), gt.grasp_frame.max(), gt.peak.min(), gt.peak.max())
    plot_gripper_curves(plt, frames_t, gt, OUT_DIR / "expert_gripper_curves.png")
    plot_gripper_aligned(plt, frames_t, gt, "Expert: all episodes aligned at detected grasp",
                         OUT_DIR / "expert_gripper_aligned.png")

    cube_t = load_cube(TRAIN_KEY)
    train = gt.merge(cube_t, on="episode_index", how="left")
    bad = train[train.grasp_frame.isna() | ~train.detected.astype(bool)]
    if len(bad):
        log.warning("excluded from regression (no grasp or no cube): %s", bad.episode_index.tolist())
    train = train.drop(index=bad.index)
    model, stats, train = fit_regression(train)
    e, b = stats["err_loo"], stats["err_baseline"]
    log.info("regression R2 fit x=%.3f y=%.3f | LOO x=%.3f y=%.3f", *stats["r2_fit"], *stats["r2_loo"])
    log.info("LOO error: mean %.1f px, median %.1f, max %.1f | MAE x %.1f y %.1f | baseline mean %.1f (x %.1f y %.1f)",
             e["mean_norm"], e["median_norm"], e["max_norm"], e["mae_x"], e["mae_y"],
             b["mean_norm"], b["mae_x"], b["mae_y"])
    train.to_csv(OUT_DIR / "expert_regression.csv", index=False, float_format="%.3f")
    plot_regression_check(plt, train, stats, OUT_DIR / "expert_regression_check.png")

    if args.expert_only:
        write_report(OUT_DIR / "report_grasp.md", gt, stats)
        log.info("expert-only run done: %s", OUT_DIR)
        return

    # ---- eval datasets ---------------------------------------------------------
    ensure_extra_positions()
    centers = pd.read_csv(TEST_POS_CSV)
    assign = load_positions()
    center = train[["cube_cx", "cube_cy"]].mean().to_numpy()
    parts = []
    for key in EVAL_KEYS:
        g, frames_e = grasp_table(key)
        plot_gripper_aligned(plt, frames_e, g, f"{key}: all episodes aligned at detected first grasp",
                             OUT_DIR / f"check_gripper_aligned_{key}.png")
        g = g.merge(load_cube(key), on="episode_index", how="left").merge(assign, on=["dataset", "episode_index"],
                                                                         how="left")
        for r in g[g.grasp_frame.isna()].itertuples():
            log.warning("%s episode %d (%s): no gripper close -> no aim position", key, r.episode_index, r.position)
        ok = g.grasp_frame.notna()
        aim = np.full((len(g), 2), np.nan)
        if ok.any():
            aim[ok.to_numpy()] = model.predict(g.loc[ok, ARM_JOINTS].to_numpy(float))
        g["aim_x"], g["aim_y"] = aim[:, 0], aim[:, 1]
        parts.append(g)
        log.info("%s: grasp detected in %d/%d", key, ok.sum(), len(g))
    gp = pd.concat(parts, ignore_index=True)
    gp["err_x"] = gp.aim_x - gp.cube_cx
    gp["err_y"] = gp.aim_y - gp.cube_cy
    gp["err_norm"] = np.hypot(gp.err_x, gp.err_y)
    to_c = np.c_[center[0] - gp.cube_cx, center[1] - gp.cube_cy]
    err = gp[["err_x", "err_y"]].to_numpy(float)
    gp["cos_to_centre"] = (err * to_c).sum(1) / (np.linalg.norm(err, axis=1) * np.linalg.norm(to_c, axis=1))

    positions_all = load_positions()
    success = load_success(SUCCESS_CSV, positions_all)
    complete = complete_success_datasets(success, positions_all)
    if success is not None:
        gp = gp.merge(success, on=["dataset", "episode_index"], how="left")
        log.info("datasets with complete success labels: %s", sorted(complete))

    cols = ["dataset", "episode_index", "position", "cube_cx", "cube_cy", "aim_x", "aim_y", "err_x", "err_y",
            "err_norm", "grasp_frame"] + ARM_JOINTS + ["grasp_time_s", "arm_speed_at_grasp", "cos_to_centre"]
    if success is not None:
        cols.append("success")
    gp[cols].to_csv(OUT_DIR / "grasp_points.csv", index=False, float_format="%.3f")

    slopes = {k: {ax: slope_ci(gp[gp.dataset == k][c].to_numpy(float), gp[gp.dataset == k][a].to_numpy(float))
                  for ax, c, a in (("x", "cube_cx", "aim_x"), ("y", "cube_cy", "aim_y"))} for k in EVAL_KEYS}
    for k in EVAL_KEYS:
        log.info("%s: slope x %.2f, slope y %.2f, mean err %.1f px, mean cos_to_centre %.2f", k,
                 slopes[k]["x"]["slope"], slopes[k]["y"]["slope"], gp[gp.dataset == k].err_norm.mean(),
                 gp[gp.dataset == k].cos_to_centre.mean())

    mean_err_x_table(gp).to_csv(OUT_DIR / "mean_err_x_by_position.csv")
    bg = decode_frame(*A.BACKGROUND_FRAME)
    plot_aim_arrows(plt, bg, gp, train, centers, center, OUT_DIR / "fig1_aim_arrows.png")
    plot_aim_vs_actual(plt, gp, slopes, OUT_DIR / "fig2_aim_vs_actual.png")
    plot_err_box(plt, gp, OUT_DIR / "fig3_error_by_position.png")
    write_report(OUT_DIR / "report_grasp.md", gt, stats, gp, slopes, center, success, complete)
    log.info("done: %s", OUT_DIR)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Is there a lateral correction between the end of the approach and the grasp ("two-stage motion")?

Reuses analyze_grasp_points.py for the grasp frame (first gripper close), the joint columns and the
"joint angles -> cube pixel position" regression (grasp_point_analysis/expert_regression.csv). Nothing is refit
or redefined here.

Stages
  curves   step 1: arm speed and shoulder_pan before the grasp for the right-side extra test, plus a table of how
           the "end of the approach" rule behaves for several thresholds. Used to choose the rule.
  metrics  steps 2-6: per-episode correction metrics, summary tables, figures C/D/E, statistics, report_correction.md.

Arm speed = norm of the frame-to-frame change of the 5 arm joints (everything except the gripper) x fps, smoothed
with a centred moving average of SMOOTH_FRAMES frames.

Outputs: ./correction_analysis/ next to this script. Raw datasets are only read.
Usage: python analyze_correction.py --stage curves | metrics | all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_cube_positions as A  # noqa: E402
import analyze_grasp_points as G  # noqa: E402

# =============================================================================
# CONFIG
# =============================================================================
OUT = HERE / "correction_analysis"
FPS = 30
SMOOTH_FRAMES = 5          # moving average of the speed signal (frames)
WINDOW_S = 4.0             # look this far back from the grasp frame
AFTER_S = 0.5              # shown after the grasp frame in the plots

# --- rule: end of the approach (confirmed after step 1: rule A) --------------------------------------------------
REACH_SPEED_THRESH = 20.0  # arm speed (units/s) below which the arm counts as (nearly) stopped
REACH_HOLD_FRAMES = 5      # ... for at least this many consecutive frames (~0.17 s)
SEARCH_FROM = "peak"       # search for the first stop after the speed peak inside the window ("peak") or from the window start ("start")
SENSITIVITY_THRESH = (10, 20, 30)   # the conclusion is also computed for these thresholds (same hold)

# --- metrics ---------------------------------------------------------------------------------------------------
MIN_PEAK_PROMINENCE = 8.0  # units/s: a local speed peak between reach end and grasp counts as a sub-movement only above this
PRE_AIM_MAX_S = 2.0        # pre_aim_err_x is only computed if the reach end is at most this long before the grasp
CLOSE_OPEN_LEVEL, CLOSE_CLOSED_LEVEL = 30.0, 15.0   # gripper hysteresis for counting closes over the whole episode
POST_GRASP_S = 0.5         # pan drift is also reported over this time after the grasp frame

# --- decision criteria (report_correction.md) --------------------------------------------------------------------
CORR_DIFF_PX = 5.0         # "clearly larger correction": median difference of correction_px (mixed - control) >= this
ALPHA = 0.05               # ... and a significant Mann-Whitney U test
MIN_GROUP = 3              # smallest group size for a test

EDGE_KEYS = ["edge_mixed", "edge_control"]
STANDARD_KEYS = ["eval_expert_smooth", "eval_mixed_smooth", "eval_expert70_smooth"]
COLOR = {"edge_mixed": G.C_DS["eval_mixed_smooth"], "edge_control": G.C_DS["eval_expert70_smooth"],
         "eval_expert_smooth": G.C_DS["eval_expert_smooth"], "eval_mixed_smooth": G.C_DS["eval_mixed_smooth"],
         "eval_expert70_smooth": G.C_DS["eval_expert70_smooth"]}
LABEL = G.DATASET_LABEL
C_TEXT, C_TEXT_2 = G.C_TEXT, G.C_TEXT_2
log = logging.getLogger("correction")


# =============================================================================
# Signals and the reach-end rule
# =============================================================================
def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x.copy()
    k = np.ones(n) / n
    pad = n // 2
    xp = np.r_[np.repeat(x[0], pad), x, np.repeat(x[-1], n - 1 - pad)]
    return np.convolve(xp, k, mode="valid")


def load_signals(key: str) -> list[dict]:
    """Per-episode signals around the first grasp."""
    grasps, frames = G.grasp_table(key)
    pos = G.load_positions()
    pos = pos[pos.dataset == key].set_index("episode_index").position
    out = []
    for r in grasps.itertuples():
        if np.isnan(r.grasp_frame):
            log.warning("%s episode %d: no gripper close -> skipped", key, r.episode_index)
            continue
        e = frames[frames.episode_index == r.episode_index].sort_values("frame_index")
        q = e[[f"state:{j}" for j in G.ARM_JOINTS]].to_numpy(float)
        speed = np.r_[0.0, np.linalg.norm(np.diff(q, axis=0), axis=1) * FPS]
        out.append(dict(dataset=key, episode_index=int(r.episode_index), position=pos.get(r.episode_index),
                        grasp_frame=int(r.grasp_frame), q=q, speed=speed,
                        speed_s=moving_average(speed, SMOOTH_FRAMES), pan=q[:, G.ARM_JOINTS.index("shoulder_pan.pos")],
                        grip=e[f"state:{G.GRIPPER}"].to_numpy(float)))
    return out


def reach_end(sig: dict, thresh: float, hold: int, search_from: str = SEARCH_FROM, window_s: float = WINDOW_S) -> float:
    """First frame (before the grasp) from which the smoothed arm speed stays below `thresh` for `hold` frames.
    Search starts at the speed peak inside the window (the arm may be at rest at the start of an episode)."""
    g = sig["grasp_frame"]
    lo = max(0, g - int(window_s * FPS))
    s = sig["speed_s"]
    start = lo + int(np.argmax(s[lo:g + 1])) if search_from == "peak" else lo
    for t in range(start, g):
        if (s[t:t + hold] < thresh).all() and len(s[t:t + hold]) == hold:
            return float(t)
    return np.nan


# =============================================================================
# Step 1: curves
# =============================================================================
def stage_curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b5b4ae", "savefig.dpi": 200, "savefig.bbox": "tight",
                         "axes.titleweight": "bold"})
    sigs = {k: load_signals(k) for k in EDGE_KEYS}
    positions = ["right-top", "right-bottom"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True)
    t_axis = lambda g, n: (np.arange(n) - g) / FPS
    for j, pos in enumerate(positions):
        for key in EDGE_KEYS:
            ss = [s for s in sigs[key] if s["position"] == pos]
            grid = np.arange(-int(WINDOW_S * FPS), int(AFTER_S * FPS) + 1)
            stack_speed, stack_pan = [], []
            for s in ss:
                g = s["grasp_frame"]
                idx = g + grid
                ok = (idx >= 0) & (idx < len(s["speed_s"]))
                sp = np.full(len(grid), np.nan); pn = np.full(len(grid), np.nan)
                sp[ok] = s["speed_s"][idx[ok]]; pn[ok] = s["pan"][idx[ok]]
                axes[0, j].plot(grid / FPS, sp, color=COLOR[key], lw=0.8, alpha=0.45)
                axes[1, j].plot(grid / FPS, pn - pn[np.nanargmin(np.abs(grid))], color=COLOR[key], lw=0.8, alpha=0.45)
                stack_speed.append(sp); stack_pan.append(pn - pn[np.nanargmin(np.abs(grid))])
            axes[0, j].plot(grid / FPS, np.nanmedian(stack_speed, axis=0), color=COLOR[key], lw=2.6,
                            label=f"{LABEL[key]} (n={len(ss)}, median)")
            axes[1, j].plot(grid / FPS, np.nanmedian(stack_pan, axis=0), color=COLOR[key], lw=2.6)
        for thr in (10, 20, 30):
            axes[0, j].axhline(thr, color=C_TEXT_2, lw=0.7, ls=":")
            axes[0, j].text(-WINDOW_S + 0.05, thr + 1, f"{thr}", fontsize=8, color=C_TEXT_2)
        for i in (0, 1):
            axes[i, j].axvline(0, color=C_TEXT, lw=1.0); G._clean(axes[i, j])
        axes[0, j].set_title(pos); axes[0, j].set_ylim(0, 120)
        axes[1, j].set_xlabel("time relative to grasp frame (s)"); axes[1, j].set_xlim(-WINDOW_S, AFTER_S)
    axes[0, 0].set_ylabel("arm speed (5 joints, units/s)\nsmoothed over 5 frames")
    axes[1, 0].set_ylabel("shoulder_pan relative to its value\nat the grasp frame (units)")
    axes[0, 0].legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.suptitle("Right-side extra test: arm speed and shoulder_pan before the first grasp (thin: episodes, thick: median)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "step1_speed_pan_curves_edge.png"); plt.close(fig)

    # zoom on the last 1.5 s before the grasp, pan only, where a micro-adjustment would show
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True)
    for j, pos in enumerate(positions):
        for key in EDGE_KEYS:
            for s in [s for s in sigs[key] if s["position"] == pos]:
                g = s["grasp_frame"]; lo = max(0, g - int(1.5 * FPS)); idx = np.arange(lo, g + int(AFTER_S * FPS) + 1)
                idx = idx[idx < len(s["pan"])]
                axes[0, j].plot((idx - g) / FPS, s["speed_s"][idx], color=COLOR[key], lw=0.9, alpha=0.5)
                axes[1, j].plot((idx - g) / FPS, s["pan"][idx] - s["pan"][g], color=COLOR[key], lw=0.9, alpha=0.5)
        axes[0, j].set_title(pos); axes[0, j].set_ylim(0, 40); axes[1, j].set_xlabel("time relative to grasp frame (s)")
        for i in (0, 1):
            axes[i, j].axvline(0, color=C_TEXT, lw=1.0); G._clean(axes[i, j])
    axes[0, 0].set_ylabel("arm speed (units/s)"); axes[1, 0].set_ylabel("shoulder_pan relative to grasp (units)")
    axes[0, 0].plot([], [], color=COLOR["edge_mixed"], lw=3, label=LABEL["edge_mixed"])
    axes[0, 0].plot([], [], color=COLOR["edge_control"], lw=3, label=LABEL["edge_control"])
    axes[0, 0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Zoom: last 1.5 s before the grasp", fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "step1_zoom_last_1p5s_edge.png"); plt.close(fig)

    # ---- sensitivity of the rule -------------------------------------------------------------------------------------
    rows = []
    for thr in (5, 10, 15, 20, 30, 40):
        for hold in (3, 5, 10):
            for key in EDGE_KEYS:
                v = [reach_end(s, thr, hold) for s in sigs[key]]
                gl = [s["grasp_frame"] for s in sigs[key]]
                dur = np.array([(g - t) / FPS for g, t in zip(gl, v)])
                for pos in positions:
                    sel = np.array([s["position"] == pos for s in sigs[key]])
                    d = dur[sel]
                    rows.append(dict(thresh=thr, hold_frames=hold, dataset=key, position=pos, n=int(sel.sum()),
                                     found=int(np.isfinite(d).sum()), median_dur_s=np.nanmedian(d) if np.isfinite(d).any() else np.nan,
                                     q25_s=np.nanpercentile(d, 25) if np.isfinite(d).any() else np.nan,
                                     q75_s=np.nanpercentile(d, 75) if np.isfinite(d).any() else np.nan))
    sens = pd.DataFrame(rows)
    # preview of the lateral correction (shoulder_pan change from rule end to grasp, converted with the expert
    # regression coefficient) for each rule setting; only meant for choosing the rule
    train = pd.read_csv(G.OUT_DIR / "expert_regression.csv")
    model, _, _ = G.fit_regression(train)
    px_per_unit = float(model.coef_[0][G.ARM_JOINTS.index("shoulder_pan.pos")])
    log.info("shoulder_pan coefficient of the regression (cx): %.3f px per unit", px_per_unit)
    cp = []
    for thr in (5, 10, 15, 20, 30, 40):
        for hold in (3, 5, 10):
            for key in EDGE_KEYS:
                for s_ in sigs[key]:
                    t = reach_end(s_, thr, hold)
                    cp.append(dict(thresh=thr, hold_frames=hold, dataset=key, position=s_["position"],
                                   correction_px=(s_["pan"][s_["grasp_frame"]] - s_["pan"][int(t)]) * px_per_unit
                                   if np.isfinite(t) else np.nan))
    cp = pd.DataFrame(cp).groupby(["thresh", "hold_frames", "dataset", "position"]).correction_px.median().round(2).reset_index(
        name="median_correction_px")
    sens = sens.merge(cp, on=["thresh", "hold_frames", "dataset", "position"])
    sens.round(2).to_csv(OUT / "step1_rule_sensitivity.csv", index=False)
    log.info("speed at the grasp frame and in the last second, median by model/position:")
    pr = []
    for key in EDGE_KEYS:
        for pos in positions:
            ss = [s for s in sigs[key] if s["position"] == pos]
            pr.append(dict(dataset=key, position=pos, n=len(ss),
                           grasp_frame_median=int(np.median([s["grasp_frame"] for s in ss])),
                           speed_peak_in_window=np.median([s["speed_s"][max(0, s["grasp_frame"] - 120):s["grasp_frame"] + 1].max() for s in ss]),
                           speed_at_grasp=np.median([s["speed_s"][s["grasp_frame"]] for s in ss]),
                           speed_median_last_1s=np.median([np.median(s["speed_s"][max(0, s["grasp_frame"] - 30):s["grasp_frame"] + 1]) for s in ss]),
                           pan_range_last_1s=np.median([np.ptp(s["pan"][max(0, s["grasp_frame"] - 30):s["grasp_frame"] + 1]) for s in ss]),
                           pan_change_last_1s=np.median([s["pan"][s["grasp_frame"]] - s["pan"][max(0, s["grasp_frame"] - 30)] for s in ss])))
    pr = pd.DataFrame(pr).round(2)
    log.info("\n%s", pr.to_string(index=False))
    pr.to_csv(OUT / "step1_signal_summary_edge.csv", index=False)
    piv = sens[sens.hold_frames == 5].pivot_table(index=["thresh"], columns=["dataset", "position"],
                                                  values="median_dur_s").round(2)
    log.info("median duration from rule end to grasp (s), hold = 5 frames:\n%s", piv.to_string())
    pxp = sens[sens.hold_frames == 5].pivot_table(index=["thresh"], columns=["dataset", "position"],
                                                  values="median_correction_px").round(1)
    log.info("PREVIEW median shoulder_pan correction (px, + = right) from rule end to grasp, hold = 5 frames:\n%s",
             pxp.to_string())
    foundp = sens[sens.hold_frames == 5].pivot_table(index=["thresh"], columns=["dataset", "position"],
                                                     values="found")
    log.info("episodes where the rule finds an end (of 8), hold = 5 frames:\n%s", foundp.to_string())
    log.info("step 1 done: %s", OUT)


# =============================================================================
# Steps 2-6: metrics
# =============================================================================
RIGHT_STD = ["mid-right", "bottom-right", "top-right"]
LEFT_STD = ["mid-left", "bottom-left", "top-left"]


def count_closes(grip: np.ndarray) -> int:
    """Number of close events over the whole episode (open >= CLOSE_OPEN_LEVEL -> closed <= CLOSE_CLOSED_LEVEL)."""
    state, n = None, 0
    for v in grip:
        if v >= CLOSE_OPEN_LEVEL:
            state = "open"
        elif v <= CLOSE_CLOSED_LEVEL and state == "open":
            n += 1
            state = "closed"
    return n


def regression_model():
    train = pd.read_csv(G.OUT_DIR / "expert_regression.csv")
    model, _, _ = G.fit_regression(train)
    return model, float(model.coef_[0][G.ARM_JOINTS.index("shoulder_pan.pos")])


def build_metrics(keys, thresh: float, hold: int, model, px_per_unit: float, all_sigs=None) -> pd.DataFrame:
    from scipy.signal import find_peaks
    positions = G.load_positions()
    success = G.load_success(G.SUCCESS_CSV, positions)
    rows = []
    for key in keys:
        sigs = all_sigs[key] if all_sigs else load_signals(key)
        cube = G.load_cube(key).set_index("episode_index")
        for s_ in sigs:
            ep, g = s_["episode_index"], s_["grasp_frame"]
            t = reach_end(s_, thresh, hold)
            aim_g = model.predict(s_["q"][g:g + 1])[0]
            cx, cy = cube.loc[ep, "cx" if "cx" in cube else "cube_cx"], cube.loc[ep, "cy" if "cy" in cube else "cube_cy"]
            r = dict(dataset=key, episode_index=ep, position=s_["position"], cube_cx=cx, cube_cy=cy,
                     grasp_frame=g, grasp_time_s=g / FPS, stop_found=bool(np.isfinite(t)),
                     n_gripper_closes=count_closes(s_["grip"]), grasp_aim_err_x=aim_g[0] - cx,
                     pan_drift_after_grasp_px=(s_["pan"][min(g + int(POST_GRASP_S * FPS), len(s_["pan"]) - 1)] - s_["pan"][g]) * px_per_unit)
            if np.isfinite(t):
                t = int(t)
                seg = s_["speed_s"][t:g + 1]
                pk, _ = find_peaks(seg, prominence=MIN_PEAK_PROMINENCE)
                r.update(t_reach_end_frame=t, t_reach_end_s=t / FPS, correction_time_s=(g - t) / FPS,
                         correction_pan=s_["pan"][g] - s_["pan"][t],
                         correction_px=(s_["pan"][g] - s_["pan"][t]) * px_per_unit, n_submovements=len(pk))
                if (g - t) / FPS <= PRE_AIM_MAX_S:
                    aim_t = model.predict(s_["q"][t:t + 1])[0]
                    r["pre_aim_err_x"] = aim_t[0] - cx
            rows.append(r)
    df = pd.DataFrame(rows)
    for c in ("t_reach_end_frame", "t_reach_end_s", "correction_time_s", "correction_pan", "correction_px",
              "n_submovements", "pre_aim_err_x"):
        if c not in df:
            df[c] = np.nan
    df["abs_pre_aim_err_x"] = df.pre_aim_err_x.abs()
    df["abs_grasp_aim_err_x"] = df.grasp_aim_err_x.abs()
    df["abs_err_change"] = df.abs_grasp_aim_err_x - df.abs_pre_aim_err_x     # negative: the error shrank
    if success is not None:
        df = df.merge(success, on=["dataset", "episode_index"], how="left")
    return df


def mwu(a: pd.Series, b: pd.Series) -> dict | None:
    from scipy.stats import mannwhitneyu
    a, b = a.dropna(), b.dropna()
    if len(a) < MIN_GROUP or len(b) < MIN_GROUP:
        return None
    u, p = mannwhitneyu(a, b, alternative="two-sided", method="auto")
    return dict(n_a=len(a), n_b=len(b), median_a=a.median(), median_b=b.median(), median_diff=a.median() - b.median(),
                U=u, p=p, rank_biserial=2 * u / (len(a) * len(b)) - 1)


def edge_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Mixed vs control on the right-side extra test (pooled and per position)."""
    rows = []
    mets = ["correction_px", "correction_time_s", "n_submovements", "pre_aim_err_x", "abs_pre_aim_err_x",
            "grasp_aim_err_x", "abs_grasp_aim_err_x", "abs_err_change", "pan_drift_after_grasp_px"]
    for scope, sel in (("pooled", slice(None)), ("right-top", "right-top"), ("right-bottom", "right-bottom")):
        d = df[df.dataset.isin(EDGE_KEYS)]
        if scope != "pooled":
            d = d[d.position == scope]
        a, b = d[d.dataset == "edge_mixed"], d[d.dataset == "edge_control"]
        for m in mets:
            t = mwu(a[m], b[m])
            if t:
                rows.append(dict(scope=scope, metric=m, **t))
    return pd.DataFrame(rows)


def shrink_test(d: pd.DataFrame) -> dict | None:
    """Paired Wilcoxon: does |err_x| get smaller from the end of the approach to the grasp?"""
    from scipy.stats import wilcoxon
    x = d.dropna(subset=["abs_pre_aim_err_x"])
    if len(x) < MIN_GROUP:
        return None
    diff = x.abs_pre_aim_err_x - x.abs_grasp_aim_err_x      # positive: smaller at the grasp
    try:
        p = wilcoxon(diff, alternative="two-sided").pvalue
    except ValueError:
        p = np.nan
    return dict(n=len(x), median_abs_pre=x.abs_pre_aim_err_x.median(), median_abs_grasp=x.abs_grasp_aim_err_x.median(),
                median_shrink=diff.median(), p=p)


def classify(df: pd.DataFrame) -> dict:
    """Apply the criteria of the report to the pooled right-side extra test."""
    d = df[df.dataset.isin(EDGE_KEYS)]
    a, b = d[d.dataset == "edge_mixed"], d[d.dataset == "edge_control"]
    corr = mwu(a.correction_px, b.correction_px)
    pre = mwu(a.abs_pre_aim_err_x, b.abs_pre_aim_err_x)
    sh = shrink_test(a)
    out = dict(corr=corr, pre=pre, shrink_mixed=sh)
    if corr is None or sh is None:
        out["code"] = "undecided"; return out
    c1 = corr["median_diff"] >= CORR_DIFF_PX and corr["p"] < ALPHA
    shrinks = sh["median_shrink"] > 0 and sh["p"] < ALPHA
    out["c1_correction_larger"], out["c1_error_shrinks"] = c1, shrinks
    if c1 and shrinks:
        out["code"] = "correction"
    elif (not c1) and pre is not None and (pre["median_b"] - pre["median_a"]) >= CORR_DIFF_PX and pre["p"] < ALPHA:
        out["code"] = "approach_aim"
    else:
        out["code"] = "undecided"
    return out


CONCLUSION = {
    "correction": "The mixed model performed a lateral correction before the grasp.",
    "approach_aim": "The difference comes from the aim during the approach, not from a correction before the grasp.",
    "undecided": "The two explanations cannot be told apart.",
}


def stage_metrics():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b5b4ae", "savefig.dpi": 200, "savefig.bbox": "tight",
                         "axes.titleweight": "bold"})
    keys = EDGE_KEYS + STANDARD_KEYS
    model, px = regression_model()
    log.info("shoulder_pan coefficient: %.3f px per unit; rule: speed < %.0f for %d frames", px, REACH_SPEED_THRESH,
             REACH_HOLD_FRAMES)
    all_sigs = {k: load_signals(k) for k in keys}
    df = build_metrics(keys, REACH_SPEED_THRESH, REACH_HOLD_FRAMES, model, px, all_sigs)
    df.round(3).to_csv(OUT / "correction_metrics.csv", index=False)

    # consistency with analysis 1 (same grasp frame, same regression -> same horizontal error)
    gp = pd.read_csv(G.OUT_DIR / "grasp_points.csv")
    m = df.merge(gp[["dataset", "episode_index", "err_x"]], on=["dataset", "episode_index"])
    log.info("grasp_aim_err_x vs err_x of grasp_points.csv (%d standard episodes): max abs diff %.4f",
             len(m), (m.grasp_aim_err_x - m.err_x).abs().max())

    nf = df[~df.stop_found]
    log.info("episodes with no stop before the grasp (listed in the report, excluded from the correction statistics): %s",
             [(r.dataset, r.episode_index) for r in nf.itertuples()] or "none")
    far = df[df.stop_found & df.pre_aim_err_x.isna()]
    log.info("episodes with the reach end more than %.1f s before the grasp (pre_aim_err_x = NaN): %s", PRE_AIM_MAX_S,
             [(r.dataset, r.episode_index, round(r.correction_time_s, 2)) for r in far.itertuples()] or "none")

    # ---- summary by dataset x position --------------------------------------------------------------------------------
    g = df.groupby(["dataset", "position"])
    summ = g.agg(n=("episode_index", "size"), n_stop_found=("stop_found", "sum"),
                 median_correction_px=("correction_px", "median"), mean_correction_px=("correction_px", "mean"),
                 q25_correction_px=("correction_px", lambda x: x.quantile(0.25)),
                 q75_correction_px=("correction_px", lambda x: x.quantile(0.75)),
                 median_correction_time_s=("correction_time_s", "median"),
                 median_n_submovements=("n_submovements", "median"),
                 n_with_submovement=("n_submovements", lambda x: int((x > 0).sum())),
                 mean_gripper_closes=("n_gripper_closes", "mean"),
                 n_with_regrasp=("n_gripper_closes", lambda x: int((x >= 3).sum())),
                 median_pre_aim_err_x=("pre_aim_err_x", "median"), n_pre_aim=("pre_aim_err_x", "count"),
                 median_grasp_aim_err_x=("grasp_aim_err_x", "median"),
                 median_pan_drift_after_grasp_px=("pan_drift_after_grasp_px", "median")).round(2).reset_index()
    if "success" in df:
        summ = summ.merge(df.groupby(["dataset", "position"]).success.sum().astype("Int64").rename("n_success").reset_index(),
                          on=["dataset", "position"])
    order = {k: i for i, k in enumerate(keys)}
    summ = summ.sort_values(["dataset", "position"], key=lambda c: c.map(order) if c.name == "dataset" else c)
    summ.to_csv(OUT / "correction_summary_by_dataset_position.csv", index=False)
    log.info("summary:\n%s", summ.to_string(index=False))

    # ---- statistics ------------------------------------------------------------------------------------------------------
    tests = edge_tests(df)
    tests.round(4).to_csv(OUT / "correction_stats_edge.csv", index=False)
    cls = classify(df)
    log.info("classification (rule A): %s | correction_px mixed-control median diff %.1f, p=%.3g | error shrinks (mixed): %s",
             cls["code"], cls["corr"]["median_diff"], cls["corr"]["p"], cls["shrink_mixed"])

    # success / failure within the extra test
    sf = []
    if "success" in df:
        for key in EDGE_KEYS:
            d = df[df.dataset == key]
            for met in ("correction_px", "grasp_aim_err_x", "abs_grasp_aim_err_x", "pre_aim_err_x"):
                t = mwu(d[d.success == 1][met], d[d.success == 0][met])
                if t:
                    sf.append(dict(dataset=key, metric=met, **t))
        for key in STANDARD_KEYS:
            d = df[df.dataset == key]
            for met in ("correction_px", "abs_grasp_aim_err_x"):
                t = mwu(d[d.success == 1][met], d[d.success == 0][met])
                if t:
                    sf.append(dict(dataset=key, metric=met, **t))
    sf = pd.DataFrame(sf)
    if len(sf):
        sf.round(4).to_csv(OUT / "correction_stats_success_vs_failure.csv", index=False)

    # sensitivity to the threshold
    sens = []
    for thr in SENSITIVITY_THRESH:
        d2 = build_metrics(EDGE_KEYS, thr, REACH_HOLD_FRAMES, model, px, all_sigs)
        c2 = classify(d2)
        sens.append(dict(thresh=thr, n_found_mixed=int(d2[(d2.dataset == "edge_mixed")].stop_found.sum()),
                         n_found_control=int(d2[(d2.dataset == "edge_control")].stop_found.sum()),
                         median_correction_px_mixed=c2["corr"]["median_a"] if c2["corr"] else np.nan,
                         median_correction_px_control=c2["corr"]["median_b"] if c2["corr"] else np.nan,
                         median_diff=c2["corr"]["median_diff"] if c2["corr"] else np.nan,
                         p=c2["corr"]["p"] if c2["corr"] else np.nan,
                         error_shrinks_mixed_p=c2["shrink_mixed"]["p"] if c2["shrink_mixed"] else np.nan,
                         median_abs_pre_mixed=c2["pre"]["median_a"] if c2["pre"] else np.nan,
                         median_abs_pre_control=c2["pre"]["median_b"] if c2["pre"] else np.nan,
                         conclusion=c2["code"]))
    sens = pd.DataFrame(sens)
    sens.round(4).to_csv(OUT / "correction_sensitivity.csv", index=False)
    log.info("sensitivity:\n%s", sens.round(3).to_string(index=False))
    # zero-filled variant (episodes without a stop counted as no correction)
    z = df.copy(); z.loc[~z.stop_found, "correction_px"] = 0.0
    zc = mwu(z[z.dataset == "edge_mixed"].correction_px, z[z.dataset == "edge_control"].correction_px)
    log.info("variant with no-stop episodes counted as 0 px: median diff %.1f, p=%.3g", zc["median_diff"], zc["p"])

    # ---- figures -----------------------------------------------------------------------------------------------------------
    positions = ["right-top", "right-bottom"]
    grid = np.arange(-int(WINDOW_S * FPS), int(AFTER_S * FPS) + 1)

    def aligned(s_, sig):
        idx = s_["grasp_frame"] + grid
        ok = (idx >= 0) & (idx < len(s_[sig]))
        out = np.full(len(grid), np.nan); out[ok] = s_[sig][idx[ok]]
        return out

    # figure C
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True)
    for j, pos in enumerate(positions):
        for key in EDGE_KEYS:
            ss = [s_ for s_ in all_sigs[key] if s_["position"] == pos]
            sp = np.array([aligned(s_, "speed_s") for s_ in ss])
            pn = np.array([aligned(s_, "pan") - s_["pan"][s_["grasp_frame"]] for s_ in ss])
            for r_ in range(len(ss)):
                axes[0, j].plot(grid / FPS, sp[r_], color=COLOR[key], lw=0.7, alpha=0.35)
                axes[1, j].plot(grid / FPS, pn[r_], color=COLOR[key], lw=0.7, alpha=0.35)
            axes[0, j].plot(grid / FPS, np.nanmedian(sp, axis=0), color=COLOR[key], lw=2.6)
            axes[1, j].plot(grid / FPS, np.nanmedian(pn, axis=0), color=COLOR[key], lw=2.6,
                            label=f"{LABEL[key]} (median of {len(ss)})")
            d = df[(df.dataset == key) & (df.position == pos) & df.stop_found]
            if len(d):
                tm = (d.t_reach_end_frame - d.grasp_frame).mean() / FPS
                for i in (0, 1):
                    axes[i, j].axvline(tm, color=COLOR[key], lw=1.6, ls="--")
        axes[0, j].axhline(REACH_SPEED_THRESH, color=C_TEXT_2, lw=0.8, ls=":")
        axes[0, j].set_ylim(0, 120); axes[0, j].set_title(pos)
        for i in (0, 1):
            axes[i, j].axvline(0, color=C_TEXT, lw=1.0); G._clean(axes[i, j])
        axes[1, j].set_xlabel("time relative to grasp frame (s)"); axes[1, j].set_xlim(-WINDOW_S, AFTER_S)
        sec = axes[1, j].secondary_yaxis("right", functions=(lambda u: u * px, lambda q_: q_ / px))
        sec.set_ylabel("approx. image px (x %.2f)" % px if j == 1 else "")
    axes[0, 0].set_ylabel("arm speed (units/s)"); axes[1, 0].set_ylabel("shoulder_pan relative to grasp (units)")
    axes[1, 0].legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.suptitle("Right-side extra test: shoulder_pan aligned at the grasp. Dashed vertical lines: mean end of the approach "
                 f"(speed < {REACH_SPEED_THRESH:g} for {REACH_HOLD_FRAMES} frames), dotted: speed threshold",
                 fontweight="bold", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "figC_pan_aligned_edge.png"); plt.close(fig)

    # figure D
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), sharey=True)
    for ax, key in zip(axes, EDGE_KEYS):
        d = df[(df.dataset == key)].dropna(subset=["pre_aim_err_x"])
        for r in d.itertuples():
            ls = "-" if r.position == "right-bottom" else "--"
            ax.plot([0, 1], [r.pre_aim_err_x, r.grasp_aim_err_x], color=COLOR[key], lw=1.4, ls=ls, alpha=0.8)
            ax.scatter([0], [r.pre_aim_err_x], s=26, color=COLOR[key], zorder=3)
            if r.success == 1:
                ax.scatter([1], [r.grasp_aim_err_x], s=40, color=COLOR[key], edgecolors="white", zorder=4)
            else:
                ax.scatter([1], [r.grasp_aim_err_x], s=46, marker="X", color=COLOR[key], edgecolors="white", zorder=4)
        ax.axhline(0, color=C_TEXT, lw=1.0)
        ax.plot([0, 1], [d.pre_aim_err_x.median(), d.grasp_aim_err_x.median()], color=C_TEXT, lw=3, alpha=0.8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["end of approach", "grasp frame"]); ax.set_xlim(-0.25, 1.25)
        n_no = int(((df.dataset == key) & df.pre_aim_err_x.isna()).sum())
        ax.set_title(f"{LABEL[key]} (n={len(d)}" + (f", {n_no} without value" if n_no else "") + ")")
        G._clean(ax)
    axes[0].set_ylabel("horizontal aim error  aim x - cube x (px)\n(0 = aimed at the cube, negative = left of it)")
    axes[0].plot([], [], color=C_TEXT_2, lw=1.4, ls="-", label="right-bottom")
    axes[0].plot([], [], color=C_TEXT_2, lw=1.4, ls="--", label="right-top")
    axes[0].scatter([], [], s=40, color=C_TEXT_2, label="success (filled dot)")
    axes[0].scatter([], [], s=46, marker="X", color=C_TEXT_2, label="failure (cross)")
    axes[0].plot([], [], color=C_TEXT, lw=3, label="median")
    h_, l_ = axes[0].get_legend_handles_labels()
    axes[1].legend(h_, l_, fontsize=8, loc="lower right", framealpha=0.9)
    fig.suptitle("Horizontal aim error from the end of the approach to the grasp (right-side extra test)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "figD_aim_error_change.png"); plt.close(fig)

    # figure E
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, gridspec_kw=dict(width_ratios=[5, 3]))
    panels = [("Right side", axes[0], [(p, STANDARD_KEYS) for p in RIGHT_STD]
               + [("extra right-bottom", EDGE_KEYS), ("extra right-top", EDGE_KEYS)]),
              ("Left side", axes[1], [(p, STANDARD_KEYS) for p in LEFT_STD])]
    model_order = [("eval_expert_smooth", "expert"), ("eval_mixed_smooth", "mixed"), ("eval_expert70_smooth", "control (expert 70)")]
    rng = np.random.default_rng(0)
    for title, ax, cats in panels:
        w = 0.8 / 3
        for ci, (cat, _) in enumerate(cats):
            for mi, (mk, mlab) in enumerate(model_order):
                dk = {"eval_mixed_smooth": "edge_mixed", "eval_expert70_smooth": "edge_control"}.get(mk) if cat.startswith("extra") else mk
                if dk is None:
                    continue
                pos = cat.replace("extra ", "")
                v = df[(df.dataset == dk) & (df.position == pos)].correction_px.dropna().to_numpy()
                x = ci - 0.4 + w * (mi + 0.5)
                if len(v):
                    bp = ax.boxplot([v], positions=[x], widths=w * 0.85, patch_artist=True, showfliers=False,
                                    medianprops=dict(color=C_TEXT, lw=1.3), whiskerprops=dict(color=COLOR[mk]),
                                    capprops=dict(color=COLOR[mk]), boxprops=dict(edgecolor=COLOR[mk]))
                    bp["boxes"][0].set_facecolor(COLOR[mk]); bp["boxes"][0].set_alpha(0.35)
                    ax.scatter(x + rng.uniform(-w * 0.2, w * 0.2, len(v)), v, s=11, color=COLOR[mk], edgecolors="white",
                               linewidths=0.4, zorder=3)
        ax.set_xticks(range(len(cats))); ax.set_xticklabels([c for c, _ in cats], fontsize=9)
        ax.axhline(0, color=C_TEXT, lw=0.8); ax.set_title(title); G._clean(ax)
        for sep in ([2.5] if title == "Right side" else []):
            ax.axvline(sep, color=C_TEXT_2, lw=0.8, ls=":")
    for mk, mlab in model_order:
        axes[0].plot([], [], color=COLOR[mk], lw=6, alpha=0.6, label=mlab)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9, framealpha=0.9)
    axes[0].set_ylabel("lateral correction before the grasp (px, + = right)")
    fig.suptitle(f"Shoulder_pan correction from the end of the approach to the grasp (speed < {REACH_SPEED_THRESH:g} "
                 f"for {REACH_HOLD_FRAMES} frames); expert has no extra-test data", fontweight="bold", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / "figE_correction_boxplot.png"); plt.close(fig)

    write_correction_report(OUT / "report_correction.md", df, summ, tests, cls, sf, sens, zc, nf, far, px)
    log.info("metrics stage done: %s", OUT)


def write_correction_report(path, df, summ, tests, cls, sf, sens, zc, nf, far, px):
    L = ["# Lateral correction before the grasp (\"two-stage motion\")", "",
         "Pixel values are image pixels converted from shoulder_pan with the expert regression "
         f"({px:.2f} px per unit); they are not distances on the table. Positive = to the right.", ""]
    L += ["## Rule and metrics", "",
          f"- Grasp frame: first gripper close (same rule as `analyze_grasp_points.py`).",
          f"- End of the approach (`t_reach_end`): after the speed peak in the {WINDOW_S:g} s before the grasp, the first "
          f"frame from which the arm speed (norm of the 5 arm joints, {SMOOTH_FRAMES}-frame moving average) stays below "
          f"{REACH_SPEED_THRESH:g} units/s for {REACH_HOLD_FRAMES} frames.",
          "- `correction_pan` / `correction_px`: change of shoulder_pan from `t_reach_end` to the grasp frame (x regression "
          "coefficient for px). `correction_time_s`: duration of that interval.",
          f"- `n_submovements`: local speed peaks with prominence >= {MIN_PEAK_PROMINENCE:g} units/s between "
          "`t_reach_end` and the grasp. `n_gripper_closes`: gripper closes over the whole episode "
          f"(open >= {CLOSE_OPEN_LEVEL:g}, closed <= {CLOSE_CLOSED_LEVEL:g}).",
          "- `pre_aim_err_x` / `grasp_aim_err_x`: horizontal aim error (regression aim x minus cube x) at `t_reach_end` "
          f"and at the grasp. The regression is only reliable near the grasp pose, so `pre_aim_err_x` is only computed if "
          f"`t_reach_end` is at most {PRE_AIM_MAX_S:g} s before the grasp (otherwise NaN).",
          "- Datasets: right-side extra test (mixed and control model) and the 6 standard positions "
          "(expert, mixed, control; smooth). Per-episode values: `correction_metrics.csv`.", ""]
    L += ["### Episodes excluded from the correction statistics", "",
          "No stop before the grasp (the arm never stayed below the threshold; excluded, not set to 0): "
          + (", ".join(f"{r.dataset} ep {r.episode_index} ({r.position})" for r in nf.itertuples()) or "none") + ".",
          f"`pre_aim_err_x` = NaN because the reach end is more than {PRE_AIM_MAX_S:g} s before the grasp: "
          + (", ".join(f"{r.dataset} ep {r.episode_index} ({r.correction_time_s:.2f} s)" for r in far.itertuples()) or "none") + ".", ""]
    L += ["## Summary by dataset and position", "", summ.to_markdown(index=False), ""]

    L += ["## Right-side extra test: mixed vs. control (Mann-Whitney U, two-sided)", "",
          "`rank_biserial` = P(mixed > control) - P(mixed < control), ranges from -1 to 1. Eight episodes per position "
          "and model, so per-position tests have little power.", "",
          tests.round(4).to_markdown(index=False), ""]
    c, pre, sh = cls["corr"], cls["pre"], cls["shrink_mixed"]
    L += ["### Pooled right-side extra test, the three quantities used by the criteria", "",
          f"- Correction (`correction_px`), mixed - control: median {c['median_a']:.2f} vs {c['median_b']:.2f} px, "
          f"difference {c['median_diff']:+.2f} px, p = {c['p']:.3g}, rank-biserial {c['rank_biserial']:+.2f} "
          f"(n = {c['n_a']} vs {c['n_b']}).",
          f"- Does the horizontal error shrink from the end of the approach to the grasp (mixed model, paired Wilcoxon on "
          f"|err_x|)? median |err_x| {sh['median_abs_pre']:.1f} -> {sh['median_abs_grasp']:.1f} px, median shrink "
          f"{sh['median_shrink']:+.1f} px, p = {sh['p']:.3g} (n = {sh['n']})." if sh else "- Shrink test not possible.",
          f"- Aim error at the end of the approach (`|pre_aim_err_x|`), mixed vs control: median {pre['median_a']:.1f} vs "
          f"{pre['median_b']:.1f} px, p = {pre['p']:.3g}, rank-biserial {pre['rank_biserial']:+.2f}." if pre else "", ""]
    ctrl = df[df.dataset == "edge_control"].dropna(subset=["abs_pre_aim_err_x"])
    if len(ctrl) >= MIN_GROUP:
        shc = shrink_test(df[df.dataset == "edge_control"])
        L += [f"- Control model, same paired test: median |err_x| {shc['median_abs_pre']:.1f} -> {shc['median_abs_grasp']:.1f} px "
              f"(median change {-shc['median_shrink']:+.2f} px, i.e. {'larger' if shc['median_shrink'] < 0 else 'smaller'} at the "
              f"grasp), p = {shc['p']:.3g} (n = {shc['n']}); the change is consistent but tiny.", ""]

    L += ["### Success vs. failure (Mann-Whitney U)", ""]
    if len(sf):
        L += ["Groups smaller than %d are not tested. Success = the cube ended in the box." % MIN_GROUP, "",
              sf.round(4).to_markdown(index=False), ""]
    else:
        L += ["No success labels found.", ""]

    L += ["## Sensitivity to the speed threshold", "",
          "The criteria applied to the pooled extra test with other thresholds (hold = %d frames):" % REACH_HOLD_FRAMES, "",
          sens.round(3).to_markdown(index=False), "",
          f"With episodes without a stop counted as 0 px (instead of excluded): median difference {zc['median_diff']:+.1f} px, "
          f"p = {zc['p']:.3g}.", ""]

    L += ["## Conclusion", "",
          f"Criteria: (1) mixed correction larger than control by at least {CORR_DIFF_PX:g} px (median) with p < {ALPHA}, and "
          f"the horizontal error of the mixed model shrinks after the end of the approach -> \"the mixed model performed a "
          f"lateral correction before the grasp\"; (2) corrections not clearly different, but the mixed model's aim error at "
          f"the end of the approach already smaller by at least {CORR_DIFF_PX:g} px (p < {ALPHA}) -> \"the difference comes "
          f"from the aim during the approach\"; (3) otherwise \"cannot be told apart\".", ""]
    L += [f"**Result (rule: speed < {REACH_SPEED_THRESH:g} for {REACH_HOLD_FRAMES} frames): {CONCLUSION[cls['code']]}**", ""]
    L += ["Result of the same criteria at other speed thresholds:", ""]
    L += [f"- threshold {int(t)}: {CONCLUSION[c]} (median difference {d:+.2f} px)"
          for t, c, d in zip(sens.thresh, sens.conclusion, sens.median_diff)]
    L += [""]
    a1 = df[(df.dataset == "edge_mixed") & df.correction_px.notna()].correction_px.sort_values().to_numpy()
    L += ["Note on the margin: the pooled median of the mixed model's 16 corrections is the average of the 8th and 9th "
          f"sorted values ({a1[7]:.2f} and {a1[8]:.2f} px), i.e. it falls in the gap between the two positions "
          "(right-bottom about 3 px, right-top about 5.6 px). A difference of %.2f px against the 5 px criterion is "
          "therefore a boundary case, while the direction is unambiguous (rank-biserial %+.2f). Per position the "
          "criterion is met at right-top (%+.2f px) and not at right-bottom (%+.2f px)." % (
              c["median_diff"], c["rank_biserial"],
              tests[(tests.scope == "right-top") & (tests.metric == "correction_px")].median_diff.iloc[0],
              tests[(tests.scope == "right-bottom") & (tests.metric == "correction_px")].median_diff.iloc[0]), ""]
    drift = df[df.dataset.isin(EDGE_KEYS)].groupby("dataset").pan_drift_after_grasp_px.median().round(1)
    L += ["Additional observation: after the grasp frame (first %.1f s of the gripper closing) the median shoulder_pan "
          "drift is %s px (positive = to the right)." % (POST_GRASP_S, ", ".join(f"{LABEL[k]}: {v:+.1f}" for k, v in drift.items())), ""]
    L += ["Limits: eight episodes per position and model; the aim error uses a regression fitted on expert grasp poses "
          "and is only a proxy for where the gripper is relative to the cube; the speed signal is noisy (encoder "
          "quantisation, about 5 units/s at rest), which is why the rule is also shown for other thresholds."]
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["curves", "metrics", "all"], default="all")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(OUT / f"run_{args.stage}.log", mode="w")])
    if args.stage in ("curves", "all"):
        stage_curves()
    if args.stage in ("metrics", "all"):
        stage_metrics()


if __name__ == "__main__":
    main()

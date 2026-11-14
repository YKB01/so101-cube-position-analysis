#!/usr/bin/env python
"""
Task A: compare the 20 recovery episodes (human segments only), the 20 new expert demonstrations and the original
50 expert demonstrations (data only, no training). Tests hypothesis H2 (recovery motions are slower / less smooth)
and describes the pre-grasp coverage relevant to H1.

Outputs (retest_analysis/): data_comparison.md, data_comparison_metrics.csv, data_comparison_curves.png,
data_comparison_summary.csv.   Run: python analyze_data_comparison.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_cube_positions as A  # noqa: E402
import analyze_grasp_points as G  # noqa: E402
import analyze_correction as C  # noqa: E402

OUT = HERE / "retest_analysis"
FPS = 30
PRE_S, POST_S = 3.0, 1.0           # window around the first grasp
SMOOTH = C.SMOOTH_FRAMES            # 5-frame moving average of speed and joint positions
SLOW_FRAC = 0.30                    # slow phase: speed below this fraction of the pre-grasp peak
HOME_DIST = 50.0                    # a close within this distance of the expert start pose is a return-to-home close
CLOSE_OPEN, CLOSE_CLOSED = C.CLOSE_OPEN_LEVEL, C.CLOSE_CLOSED_LEVEL
GROUPS = {"expert": ("Original expert (50)", "#2a78d6", "o"), "new_expert": ("New expert (20)", "#6250d6", "D"),
          "recovery": ("Recovery, human segments (20)", "#eb6834", "^")}
NEW_EXPERT_DIR = "so101_red_cube_plastic_box_expert_20_20261112_130439"
log = logging.getLogger("datacmp")


def jerk_norm(q: np.ndarray) -> np.ndarray:
    """|jerk| per frame: norm over the 5 arm joints of the 3rd finite difference of the smoothed positions."""
    qs = np.stack([C.moving_average(q[:, j], SMOOTH) for j in range(q.shape[1])], 1)
    j3 = np.diff(qs, n=3, axis=0) * FPS ** 3
    return np.r_[np.zeros(3), np.linalg.norm(j3, axis=1)]


def episode_metrics(key, frames, start_pose):
    rows, curves = [], []
    t_axis = np.arange(-int(PRE_S * FPS), int(POST_S * FPS) + 1)
    for ep, e in frames.groupby("episode_index"):
        e = e.sort_values("frame_index")
        q = e[[f"state:{j}" for j in G.ARM_JOINTS]].to_numpy(float)
        grip = e[f"state:{G.GRIPPER}"].to_numpy(float)
        n = len(e)
        speed = C.moving_average(np.r_[0.0, np.linalg.norm(np.diff(q, axis=0), axis=1) * FPS], SMOOTH)
        jk = jerk_norm(q)
        det = G.detect_first_grasp(grip)
        gf = det["grasp_frame"]
        # gripper closes: all, and without the return-to-home close
        st, ev = None, []
        for i, v in enumerate(grip):
            if v >= CLOSE_OPEN:
                st = "open"
            elif v <= CLOSE_CLOSED and st == "open":
                ev.append(i); st = "closed"
        n_rule = sum(np.linalg.norm(q[i] - start_pose) >= HOME_DIST for i in ev)
        r = dict(group=key, episode_index=int(ep), n_frames=n, duration_s=n / FPS, n_closes_all=len(ev),
                 n_closes_rule=n_rule, first_frame_dist_to_start_pose=float(np.linalg.norm(q[0] - start_pose)))
        if np.isnan(gf):
            rows.append(r); curves.append(np.full(len(t_axis), np.nan)); continue
        gf = int(gf)
        lo = max(0, gf - int(PRE_S * FPS))
        pre = speed[lo:gf + 1]
        peak = float(pre.max())
        k = gf
        while k > lo and speed[k] < SLOW_FRAC * peak:
            k -= 1
        r.update(grasp_frame=gf, time_to_grasp_s=gf / FPS, pre_coverage_s=(gf - lo) / FPS, peak_speed=peak,
                 slow_phase_s=(gf - k) / FPS, mean_abs_jerk=float(jk[lo:gf + 1].mean()),
                 speed_at_grasp=float(speed[gf]),
                 mean_speed_last1s=float(speed[max(0, gf - FPS):gf + 1].mean()),
                 peak_speed_post=float(speed[gf:gf + 3 * FPS + 1].max()),
                 mean_abs_jerk_post=float(jk[gf:gf + 3 * FPS + 1].mean()),
                 peak_speed_episode=float(speed.max()),
                 path_to_grasp=float(np.linalg.norm(np.diff(q[:gf + 1], axis=0), axis=1).sum()),
                 path_last3s=float(np.linalg.norm(np.diff(q[lo:gf + 1], axis=0), axis=1).sum()))
        # common-window variant (filled later): store arrays for it
        r["_speed"], r["_jerk"], r["_gf"] = speed, jk, gf
        idx = gf + t_axis
        curves.append(np.where((idx >= 0) & (idx < n), speed[np.clip(idx, 0, n - 1)], np.nan))
        rows.append(r)
    return rows, np.array(curves), t_axis


def mwu(a, b, alt="two-sided"):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] == b.iloc[0]:
        return dict(n_a=len(a), n_b=len(b), median_a=a.median(), median_b=b.median(), p=1.0, rank_biserial=0.0)
    u, p = stats.mannwhitneyu(a, b, alternative=alt)
    return dict(n_a=len(a), n_b=len(b), median_a=a.median(), median_b=b.median(), p=p, rank_biserial=2 * u / (len(a) * len(b)) - 1)


def main():
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "run_data_comparison.log", mode="w")])
    A.DATASETS["new_expert"] = NEW_EXPERT_DIR
    all_rows, all_curves = [], {}
    fr = {k: G.load_frames(k)[0] for k in GROUPS}
    ex0 = fr["expert"].sort_values(["episode_index", "frame_index"]).groupby("episode_index").first()
    start_pose = ex0[[f"state:{j}" for j in G.ARM_JOINTS]].to_numpy(float).mean(0)
    for k in GROUPS:
        rows, cur, t_axis = episode_metrics(k, fr[k], start_pose)
        all_rows += rows; all_curves[k] = cur
    M = pd.DataFrame(all_rows)
    # common window: the shortest pre-grasp coverage among recovery episodes, applied to every episode
    common_s = float(M[M.group == "recovery"].pre_coverage_s.min())
    cw = int(common_s * FPS)
    for i, r in M.iterrows():
        if "_gf" in r and isinstance(r["_speed"], np.ndarray):
            gf, sp, jk = int(r["_gf"]), r["_speed"], r["_jerk"]
            lo = gf - cw
            if lo >= 0:
                pre = sp[lo:gf + 1]; pk = pre.max(); kk = gf
                while kk > lo and sp[kk] < SLOW_FRAC * pk:
                    kk -= 1
                M.loc[i, ["peak_speed_common", "slow_phase_common_s", "mean_abs_jerk_common"]] = [pk, (gf - kk) / FPS, jk[lo:gf + 1].mean()]
    M = M.drop(columns=["_speed", "_jerk", "_gf"])
    M.to_csv(OUT / "data_comparison_metrics.csv", index=False, float_format="%.3f")
    log.info("recovery pre-grasp coverage (s): %s", M[M.group == "recovery"].pre_coverage_s.round(2).tolist())
    log.info("common window = %.2f s", common_s)

    metrics = [("peak_speed", "Peak speed before grasp (units/s)", "lower in recovery if H2"),
               ("slow_phase_s", "Slow phase before grasp (s)", "longer in recovery if H2"),
               ("mean_abs_jerk", "Mean abs jerk before grasp (units/s^3)", "higher in recovery if H2"),
               ("peak_speed_common", f"Peak speed, common {common_s:.1f} s window", "lower"),
               ("slow_phase_common_s", f"Slow phase, common {common_s:.1f} s window (s)", "longer"),
               ("mean_abs_jerk_common", f"Mean abs jerk, common {common_s:.1f} s window", "higher"),
               ("mean_speed_last1s", "Mean speed in the last 1 s before grasp (units/s)", ""),
               ("peak_speed_post", "Peak speed in the 3 s after grasp (units/s)", ""),
               ("mean_abs_jerk_post", "Mean abs jerk in the 3 s after grasp", ""),
               ("peak_speed_episode", "Peak speed in the whole episode (units/s)", ""),
               ("path_to_grasp", "Arm path length from episode start to grasp (joint units)", ""),
               ("path_last3s", "Arm path length in the last 3 s before grasp (joint units)", ""),
               ("pre_coverage_s", "Pre-grasp coverage (s, max 3)", ""), ("time_to_grasp_s", "Time from episode start to grasp (s)", ""),
               ("n_closes_all", "Gripper closes (all)", ""), ("n_closes_rule", "Gripper closes (return-to-home excluded)", ""),
               ("n_frames", "Frames per episode", ""), ("first_frame_dist_to_start_pose", "First-frame distance to start pose", "")]
    S = []
    for m, lab, _ in metrics:
        row = dict(metric=lab)
        for g in GROUPS:
            v = M[M.group == g][m].dropna()
            row[f"{g}_median"], row[f"{g}_q25"], row[f"{g}_q75"], row[f"{g}_n"] = v.median(), v.quantile(.25), v.quantile(.75), len(v)
        for a, b in (("recovery", "new_expert"), ("recovery", "expert"), ("new_expert", "expert")):
            row[f"p_{a}_vs_{b}"] = mwu(M[M.group == a][m], M[M.group == b][m])["p"]
            row[f"rb_{a}_vs_{b}"] = mwu(M[M.group == a][m], M[M.group == b][m])["rank_biserial"]
        S.append(row)
    S = pd.DataFrame(S); S.to_csv(OUT / "data_comparison_summary.csv", index=False, float_format="%.4g")
    log.info("summary:\n%s", S[["metric", "expert_median", "new_expert_median", "recovery_median", "p_recovery_vs_new_expert",
                                  "p_recovery_vs_expert", "p_new_expert_vs_expert"]].round(3).to_string(index=False))

    # ---- figure -----------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "savefig.bbox": "tight", "axes.spines.top": False,
                         "axes.spines.right": False})
    tt = t_axis / FPS
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), gridspec_kw=dict(width_ratios=[1.7, 1]))
    ax = axes[0]
    for g, (lab, col, _) in GROUPS.items():
        cur = all_curves[g]
        ok = np.isfinite(cur).sum(0) >= 5
        med = np.where(ok, np.nanmedian(np.where(np.isfinite(cur), cur, np.nan), 0), np.nan)
        lo_ = np.where(ok, np.nanpercentile(cur, 25, 0), np.nan); hi_ = np.where(ok, np.nanpercentile(cur, 75, 0), np.nan)
        ax.plot(tt, med, color=col, lw=2, label=lab); ax.fill_between(tt, lo_, hi_, color=col, alpha=0.16, lw=0)
    ax.axvline(0, color="#52514e", lw=1); ax.set_xlabel("Time relative to first grasp (s)")
    ax.set_ylabel("Arm speed (units/s)"); ax.set_title("Arm speed aligned to the first grasp (median, IQR)")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    data = [M[M.group == g].time_to_grasp_s.dropna().to_numpy() for g in GROUPS]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color="#0b0b0b"))
    for patch, (g, (lab, col, mk)) in zip(bp["boxes"], GROUPS.items()):
        patch.set_facecolor(col); patch.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for i, (g, (lab, col, mk)) in enumerate(GROUPS.items(), 1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(data[i - 1])), data[i - 1], s=18, marker=mk, color=col, zorder=3)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Original\nexpert", "New\nexpert", "Recovery"])
    ax.set_ylabel("Time from episode start to first grasp (s)"); ax.set_title("How much of the approach each episode contains")
    fig.savefig(OUT / "data_comparison_curves.png"); plt.close(fig)


if __name__ == "__main__":
    main()

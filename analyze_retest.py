#!/usr/bin/env python
"""
Analysis of the pre-registered right-side retest (48 trials, model R vs model C), after unblinding.

Reads: protocol/trial_order.csv, protocol/model_key.json (hash is checked against protocol/model_key.sha256),
       results/retest_outcomes.csv (+ results/retest_corrections.csv), results/placement_log.csv,
       results/retest_raw/<trial>_a<k>/meta.json and the recorded LeRobot datasets (joint states).
Writes: retest_analysis/ (trial_table.csv, summary tables, figures, report_retest.md)
Primary (pre-registered): two-sided Fisher exact test of the pooled success rate, R vs C.
Secondary (exploratory, as in protocol/right_side_retest.md and protocol/addendum_01.md): per-position success,
failure types, gripper-close count (addendum 01 rule), pre-grasp speed curves / lateral correction
(same method as analyze_correction.py, rule A).
"""
from __future__ import annotations

import hashlib
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
RES = HERE / "results"
CLOSE_OPEN, CLOSE_CLOSED = C.CLOSE_OPEN_LEVEL, C.CLOSE_CLOSED_LEVEL     # 30 / 15 (hysteresis)
HOME_DIST = 50.0                  # addendum 01: close within this distance (joint units) of the first-frame pose
COL = {"R": "#eb6834", "C": "#6250d6"}
MARK = {"R": "^", "C": "o"}
log = logging.getLogger("retest")


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    """95% CI of p1 - p2 (Newcombe hybrid score)."""
    l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    return (p1 - p2 - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), p1 - p2 + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def fisher_row(label, d):
    t = [[int(((d.model == m) & (d.success == s)).sum()) for s in (1, 0)] for m in ("R", "C")]
    odds, p = stats.fisher_exact(t, alternative="two-sided")
    kr, kc = t[0][0], t[1][0]; nr, nc = sum(t[0]), sum(t[1])
    lo, hi = newcombe(kr, nr, kc, nc) if nr and nc else (np.nan, np.nan)
    return dict(analysis=label, R_success=kr, R_n=nr, R_rate=kr / nr if nr else np.nan, C_success=kc, C_n=nc,
                C_rate=kc / nc if nc else np.nan, diff_R_minus_C=(kr / nr - kc / nc) if nr and nc else np.nan,
                diff_ci_lo=lo, diff_ci_hi=hi, fisher_odds_ratio=odds, fisher_p_two_sided=p)


# ------------------------------------------------------------------------------------------------ data
def load_trials(sensitivity_t36_success=False) -> pd.DataFrame:
    order = pd.read_csv(RES.parent / "protocol/trial_order.csv")
    key_path = RES.parent / "protocol/model_key.json"
    assert hashlib.sha256(key_path.read_bytes()).hexdigest() == (RES.parent / "protocol/model_key.sha256").read_text().strip(), \
        "model_key.json does not match the pre-registered hash"
    key = {c: v["model"] for c, v in json.loads(key_path.read_text())["key"].items()}
    order["model"] = order.model_code.map(key)
    out = pd.read_csv(RES / "retest_outcomes.csv", dtype={"note": str}).fillna("")
    corr = pd.read_csv(RES / "retest_corrections.csv", dtype=str).fillna("")
    rows = []
    for r in order.itertuples():
        o = out[(out.trial_id == r.trial_id) & out.outcome.isin(["success", "fail"])]
        att = int(o.attempt.iloc[-1]); outc, ftype, note = o.outcome.iloc[-1], o.failure_type.iloc[-1], o.note.iloc[-1]
        c = corr[corr.trial_id == r.trial_id]
        if len(c) and not sensitivity_t36_success:
            att, outc, ftype = int(c.final_attempt.iloc[0]), c.outcome.iloc[0], c.failure_type.iloc[0]
            note = "corrected: " + c.reason.iloc[0][:60]
        rows.append(dict(trial_id=r.trial_id, position_id=r.position_id, model_code=r.model_code, model=r.model,
                         attempt=att, outcome=outc, success=int(outc == "success"), failure_type=ftype, note=note))
    return pd.DataFrame(rows)


def trial_metrics(trials: pd.DataFrame, model, px_per_unit) -> pd.DataFrame:
    pl = pd.read_csv(RES / "placement_log.csv")
    rows = []
    for t in trials.itertuples():
        d = RES / "retest_raw" / f"{t.trial_id}_a{t.attempt}"
        meta = json.loads((d / "meta.json").read_text())
        k = f"retest_{t.trial_id}_a{t.attempt}"
        A.DATASETS[k] = "rollout_retest/" + meta["dataset_repo_id"].split("/", 1)[1]   # datasets live in YKB0123/rollout_retest/
        g, frames = G.grasp_table(k)
        e = frames.sort_values("frame_index")
        q = e[[f"state:{j}" for j in G.ARM_JOINTS]].to_numpy(float)
        grip = e[f"state:{G.GRIPPER}"].to_numpy(float)
        # gripper closes (addendum 01)
        st, ev = None, []
        for i, v in enumerate(grip):
            if v >= CLOSE_OPEN:
                st = "open"
            elif v <= CLOSE_CLOSED and st == "open":
                ev.append(i); st = "closed"
        home = [i for i in ev if np.linalg.norm(q[i] - q[0]) < HOME_DIST]
        row = dict(trial_id=t.trial_id, n_frames=len(e), duration_s=len(e) / 30, n_close_events=len(ev),
                   n_home_closes=len(home), n_gripper_closes=len(ev) - len(home), returned_home=bool(home),
                   early_end_key=meta.get("early_end_key"))
        p = pl[pl.trial_id == t.trial_id]
        p = p[p.timestamp_unix <= meta["started_unix"]].iloc[-1]
        cx, cy = float(p.measured_x), float(p.measured_y)
        gf = g.grasp_frame.iloc[0]
        row.update(cube_x=cx, cube_y=cy, first_grasp_frame=gf)
        if not np.isnan(gf):
            gf = int(gf)
            speed = np.r_[0.0, np.linalg.norm(np.diff(q, axis=0), axis=1) * 30]
            sig = dict(grasp_frame=gf, speed_s=C.moving_average(speed, C.SMOOTH_FRAMES), q=q,
                       pan=q[:, G.ARM_JOINTS.index("shoulder_pan.pos")])
            te = C.reach_end(sig, C.REACH_SPEED_THRESH, C.REACH_HOLD_FRAMES)
            aim_g = model.predict(q[gf:gf + 1])[0]
            row.update(first_grasp_time_s=gf / 30, grasp_aim_err_x=aim_g[0] - cx, grasp_aim_err_y=aim_g[1] - cy)
            if np.isfinite(te):
                te = int(te)
                row.update(reach_end_s=te / 30, correction_time_s=(gf - te) / 30,
                           correction_px=(sig["pan"][gf] - sig["pan"][te]) * px_per_unit)
                if (gf - te) / 30 <= C.PRE_AIM_MAX_S:
                    row["pre_aim_err_x"] = model.predict(q[te:te + 1])[0][0] - cx
            # speed curve aligned to the grasp (4 s before .. 0.5 s after)
            lo, hi = gf - int(C.WINDOW_S * 30), gf + int(C.AFTER_S * 30)
            idx = np.arange(lo, hi + 1)
            row["_curve"] = np.where((idx >= 0) & (idx < len(speed)), sig["speed_s"][np.clip(idx, 0, len(speed) - 1)], np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- main
def mwu(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 3 or len(b) < 3:
        return dict(n_R=len(a), n_C=len(b), median_R=a.median(), median_C=b.median(), p=np.nan, rank_biserial=np.nan)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return dict(n_R=len(a), n_C=len(b), median_R=a.median(), median_C=b.median(), p=p, rank_biserial=2 * u / (len(a) * len(b)) - 1)


def main():
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "run.log", mode="w")])
    trials = load_trials()
    reg, px = C.regression_model()
    met = trial_metrics(trials, reg, px)
    curves = np.stack(met.pop("_curve").to_numpy()) if "_curve" in met else None
    T = trials.merge(met, on="trial_id")
    T.drop(columns=[c for c in T if c.startswith("_")]).to_csv(OUT / "trial_table.csv", index=False, float_format="%.3f")

    # ---- primary + sensitivity ------------------------------------------------------------------
    prim = [fisher_row("PRIMARY: pooled 3 positions (T36 = failure, attempt 1)", T)]
    T_alt = load_trials(sensitivity_t36_success=True)
    prim.append(fisher_row("sensitivity: T36 as recorded originally (attempt 2, success)", T_alt))
    for pid, d in T.groupby("position_id"):
        prim.append(fisher_row(f"exploratory: position {pid}", d))
    P = pd.DataFrame(prim); P.to_csv(OUT / "success_tests.csv", index=False, float_format="%.4f")
    log.info("success tests:\n%s", P.round(3).to_string(index=False))

    ft = pd.crosstab(T[T.success == 0].failure_type, T[T.success == 0].model).reindex(columns=["R", "C"], fill_value=0)
    ft.to_csv(OUT / "failure_types.csv"); log.info("failure types:\n%s", ft)

    # ---- secondary: closes, returned home, correction ----------------------------------------------
    rows = []
    for name, f in (("n_gripper_closes", "closes (addendum 01)"), ("first_grasp_time_s", "first grasp time (s)"),
                    ("correction_px", "pan correction before grasp (px)"), ("correction_time_s", "correction time (s)"),
                    ("grasp_aim_err_x", "aim error x at grasp (px)"), ("pre_aim_err_x", "aim error x at reach end (px)")):
        r = mwu(T[T.model == "R"][name], T[T.model == "C"][name]); r["metric"] = f; rows.append(r)
    S = pd.DataFrame(rows)[["metric", "n_R", "n_C", "median_R", "median_C", "p", "rank_biserial"]]
    S.to_csv(OUT / "secondary_mwu.csv", index=False, float_format="%.4f"); log.info("secondary:\n%s", S.round(3).to_string(index=False))
    rt = pd.crosstab(T.model, T.n_gripper_closes); rt.to_csv(OUT / "close_count_distribution.csv"); log.info("closes:\n%s", rt)
    rh = T.groupby("model").returned_home.agg(["sum", "count"]); log.info("returned home:\n%s", rh)
    byc = T.groupby(["model", "success"]).n_gripper_closes.agg(["mean", "median", "count"]).round(2); log.info("closes by outcome:\n%s", byc)
    bypos = T.groupby(["position_id", "model"]).agg(n=("success", "size"), success=("success", "sum"),
                                                     closes_mean=("n_gripper_closes", "mean"),
                                                     corr_px_med=("correction_px", "median")).round(2)
    bypos.to_csv(OUT / "by_position_model.csv"); log.info("by position:\n%s", bypos)
    # ---- figures ---------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "savefig.bbox": "tight", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    groups = [("All (pooled)", T)] + [(f"{p}", T[T.position_id == p]) for p in sorted(T.position_id.unique())]
    w = 0.36
    for i, (lab, d) in enumerate(groups):
        for j, m in enumerate(("R", "C")):
            s = d[d.model == m].success; k, n = int(s.sum()), len(s)
            lo, hi = wilson(k, n)
            x = i + (j - 0.5) * w
            ax.bar(x, k / n, w * 0.92, color=COL[m], label=f"Model {m}" if i == 0 else None, zorder=2)
            ax.plot([x, x], [lo, hi], color="#0b0b0b", lw=1.2, zorder=3)
            ax.text(x, 0.03, f"{k}/{n}", ha="center", va="bottom", fontsize=9, color="white", fontweight="bold", zorder=4)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success rate (Wilson 95% CI)"); ax.axvline(0.5, color="#b5b4ae", lw=0.8, ls=":")
    pm = P.iloc[0]
    ax.set_title(f"Right-side retest: success rate. Pooled Fisher exact p = {pm.fisher_p_two_sided:.3f} (two-sided)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.savefig(OUT / "fig1_success.png"); plt.close(fig)

    if curves is not None:
        tt = np.arange(-int(C.WINDOW_S * 30), int(C.AFTER_S * 30) + 1) / 30
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        has = T.first_grasp_frame.notna().to_numpy()
        for m in ("R", "C"):
            sel = (T.model == m).to_numpy() & has
            cm = curves[sel[has]] if curves.shape[0] == has.sum() else curves[sel]
            ax.plot(tt, np.nanmedian(cm, 0), color=COL[m], lw=2, label=f"Model {m} (median, n={sel.sum()})")
            ax.fill_between(tt, np.nanpercentile(cm, 25, 0), np.nanpercentile(cm, 75, 0), color=COL[m], alpha=0.18, lw=0)
        ax.axvline(0, color="#52514e", lw=1); ax.set_xlabel("Time relative to first grasp (s)")
        ax.set_ylabel("Arm speed (units/s)"); ax.set_title("Pre-grasp arm speed, aligned to the first grasp (IQR band)")
        ax.legend(frameon=False); fig.savefig(OUT / "fig2_speed_aligned.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for j, m in enumerate(("R", "C")):
        v = T[T.model == m].n_gripper_closes.value_counts().sort_index()
        ax.bar(v.index + (j - 0.5) * 0.36, v.values, 0.34, color=COL[m], label=f"Model {m}")
    ax.set_xlabel("Gripper closes before the cube is released (addendum 01 rule)"); ax.set_ylabel("Trials")
    ax.set_xticks(sorted(T.n_gripper_closes.unique())); ax.legend(frameon=False)
    fig.savefig(OUT / "fig3_closes.png"); plt.close(fig)
    log.info("done")


if __name__ == "__main__":
    main()

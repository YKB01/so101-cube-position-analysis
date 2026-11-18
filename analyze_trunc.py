#!/usr/bin/env python
"""
Analysis of the pre-registered C_trunc vs C test (protocol/trunc_test.md), after unblinding.
Primary: two-sided Fisher exact, C_trunc vs same-session C, P1+P2 pooled (16 vs 16).
Secondary: C_trunc vs retest R / C (not interleaved), drift check (this C vs retest C), per-position, failure types,
gripper closes (addendum 01), pre-grasp speed, aim error, pan correction (method of analyze_retest.py).
Reads protocol/trunc_*.csv/.json (hash checked), results/trunc_*.csv, results/trunc_raw/*/meta.json, retest_analysis/trial_table.csv.
Writes trunc_analysis/.
"""
from __future__ import annotations
import hashlib, json, logging, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_cube_positions as A  # noqa
import analyze_grasp_points as G  # noqa
import analyze_correction as C  # noqa
import analyze_retest as AR  # noqa

OUT, RES = HERE / "trunc_analysis", HERE / "results"
COL = {"C_trunc": "#d03a3a", "C": "#6250d6", "R": "#eb6834"}
log = logging.getLogger("trunc")


def fisher(k1, n1, k2, n2):
    odds, p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")
    lo, hi = AR.newcombe(k1, n1, k2, n2)
    return dict(k1=k1, n1=n1, k2=k2, n2=n2, diff=k1 / n1 - k2 / n2, ci_lo=lo, ci_hi=hi, p=p)


def load_trials():
    order = pd.read_csv(HERE / "protocol/trunc_trial_order.csv")
    kp = HERE / "protocol/trunc_model_key.json"
    assert hashlib.sha256(kp.read_bytes()).hexdigest() == (HERE / "protocol/trunc_model_key.sha256").read_text().strip()
    key = {c: v["model"] for c, v in json.loads(kp.read_text())["key"].items()}
    order["model"] = order.model_code.map(key)
    out = pd.read_csv(RES / "trunc_outcomes.csv", dtype={"note": str}).fillna("")
    rows = []
    for r in order.itertuples():
        o = out[(out.trial_id == r.trial_id) & out.outcome.isin(["success", "fail"])]
        assert len(o) == 1, r.trial_id
        rows.append(dict(trial_id=r.trial_id, position_id=r.position_id, model_code=r.model_code, model=r.model,
                         attempt=int(o.attempt.iloc[0]), outcome=o.outcome.iloc[0], success=int(o.outcome.iloc[0] == "success"),
                         failure_type=o.failure_type.iloc[0], note=o.note.iloc[0]))
    return pd.DataFrame(rows)


def trial_metrics(trials, model, px):
    pl = pd.read_csv(RES / "trunc_placement_log.csv")
    rows = []
    for t in trials.itertuples():
        meta = json.loads((RES / "trunc_raw" / f"{t.trial_id}_a{t.attempt}" / "meta.json").read_text())
        k = f"trunc_{t.trial_id}_a{t.attempt}"
        A.DATASETS[k] = "rollout_trunc/" + meta["dataset_repo_id"].split("/", 1)[1]   # datasets live in YKB0123/rollout_trunc/
        g, frames = G.grasp_table(k)
        e = frames.sort_values("frame_index")
        q = e[[f"state:{j}" for j in G.ARM_JOINTS]].to_numpy(float)
        grip = e[f"state:{G.GRIPPER}"].to_numpy(float)
        st, ev = None, []
        for i, v in enumerate(grip):
            if v >= AR.CLOSE_OPEN: st = "open"
            elif v <= AR.CLOSE_CLOSED and st == "open": ev.append(i); st = "closed"
        home = [i for i in ev if np.linalg.norm(q[i] - q[0]) < AR.HOME_DIST]
        row = dict(trial_id=t.trial_id, n_frames=len(e), duration_s=len(e) / 30, n_close_events=len(ev),
                   n_gripper_closes=len(ev) - len(home), returned_home=bool(home), early_end_key=meta.get("early_end_key"))
        p = pl[pl.trial_id == t.trial_id]; p = p[p.timestamp_unix <= meta["started_unix"]].iloc[-1]
        cx, cy = float(p.measured_x), float(p.measured_y)
        gf = g.grasp_frame.iloc[0]
        row.update(cube_x=cx, cube_y=cy, place_dist=float(p.dist), first_grasp_frame=gf)
        speed = np.r_[0.0, np.linalg.norm(np.diff(q, axis=0), axis=1) * 30]
        sm = C.moving_average(speed, C.SMOOTH_FRAMES)
        row.update(mean_speed_first10s=float(sm[:300].mean()), max_speed=float(sm.max()),
                   path_len_arm=float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()))
        if not np.isnan(gf):
            gf = int(gf)
            sig = dict(grasp_frame=gf, speed_s=sm, q=q, pan=q[:, G.ARM_JOINTS.index("shoulder_pan.pos")])
            te = C.reach_end(sig, C.REACH_SPEED_THRESH, C.REACH_HOLD_FRAMES)
            aim_g = model.predict(q[gf:gf + 1])[0]
            row.update(first_grasp_time_s=gf / 30, grasp_aim_err_x=aim_g[0] - cx, grasp_aim_err_y=aim_g[1] - cy)
            if np.isfinite(te):
                te = int(te)
                row.update(reach_end_s=te / 30, correction_time_s=(gf - te) / 30, correction_px=(sig["pan"][gf] - sig["pan"][te]) * px)
                if (gf - te) / 30 <= C.PRE_AIM_MAX_S:
                    row["pre_aim_err_x"] = model.predict(q[te:te + 1])[0][0] - cx
            idx = np.arange(gf - int(C.WINDOW_S * 30), gf + int(C.AFTER_S * 30) + 1)
            row["_curve"] = np.where((idx >= 0) & (idx < len(speed)), sm[np.clip(idx, 0, len(speed) - 1)], np.nan)
        else:
            # no grasp detected: speed curve over the first 10 s for reference
            row["_curve"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def mwu(a, b, la, lb):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    r = dict(**{f"n_{la}": len(a), f"n_{lb}": len(b), f"median_{la}": a.median(), f"median_{lb}": b.median()})
    if len(a) >= 3 and len(b) >= 3:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided"); r.update(p=p, rank_biserial=2 * u / (len(a) * len(b)) - 1)
    return r


def main():
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "run.log", mode="w")])
    T = load_trials()
    reg, px = C.regression_model()
    met = trial_metrics(T, reg, px)
    curves = met.pop("_curve")
    T = T.merge(met, on="trial_id")
    T.to_csv(OUT / "trial_table.csv", index=False, float_format="%.3f")

    # retest reference (R and C at P1, P2 only; 13:45-15:11, interleaved R/C, not with C_trunc)
    rt = pd.read_csv(HERE / "retest_analysis/trial_table.csv")
    rt = rt[rt.position_id.isin(["P1", "P2"])]
    kR, kRc = int(rt[rt.model == "R"].success.sum()), int(rt[rt.model == "C"].success.sum())
    assert (kR, kRc, (rt.model == "R").sum(), (rt.model == "C").sum()) == (7, 15, 16, 16)

    kt = lambda d, m: (int(d[d.model == m].success.sum()), int((d.model == m).sum()))
    (kT, nT), (kC, nC) = kt(T, "C_trunc"), kt(T, "C")
    rows = [
        dict(analysis="PRIMARY: C_trunc vs C (this session), P1+P2", **fisher(kT, nT, kC, nC)),
        dict(analysis="secondary 1a: C_trunc vs retest R (7/16)", **fisher(kT, nT, kR, 16)),
        dict(analysis="secondary 1b: C_trunc vs retest C (15/16)", **fisher(kT, nT, kRc, 16)),
        dict(analysis="drift check: C (this session) vs retest C (15/16)", **fisher(kC, nC, kRc, 16)),
        dict(analysis="reference: retest R vs retest C (P1+P2)", **fisher(kR, 16, kRc, 16)),
        dict(analysis="reference: C (this session) vs retest R (7/16)", **fisher(kC, nC, kR, 16)),
    ]
    for pid in ("P1", "P2"):
        d = T[T.position_id == pid]; a, b = kt(d, "C_trunc"), kt(d, "C")
        rows.append(dict(analysis=f"exploratory {pid}: C_trunc vs C", **fisher(*a, *b)))
    P = pd.DataFrame(rows); P.to_csv(OUT / "success_tests.csv", index=False, float_format="%.4f")
    log.info("success tests:\n%s", P.round(3).to_string(index=False))

    # chronological drift inside the session
    T["order"] = T.trial_id.str[1:].astype(int)
    for m in ("C", "C_trunc"):
        d = T[T.model == m].sort_values("order")
        log.info("%s successes by trial: %s", m, "".join(str(x) for x in d.success))
    d = T[T.model == "C"]
    log.info("C by half of session (U01-16 vs U17-32): %d/8 vs %d/8", d[d.order <= 16].success.sum(), d[d.order > 16].success.sum())

    ft = pd.crosstab(T[T.success == 0].failure_type, T[T.success == 0].model).reindex(columns=["C_trunc", "C"], fill_value=0)
    ft.to_csv(OUT / "failure_types.csv"); log.info("failure types:\n%s", ft)

    rows = []
    for name, lab in (("n_gripper_closes", "closes (addendum 01)"), ("first_grasp_time_s", "first grasp time (s)"),
                      ("mean_speed_first10s", "mean arm speed, first 10 s"), ("correction_px", "pan correction before grasp (px)"),
                      ("correction_time_s", "correction time (s)"), ("grasp_aim_err_x", "aim error x at grasp (px)"),
                      ("grasp_aim_err_y", "aim error y at grasp (px)"), ("pre_aim_err_x", "aim error x at reach end (px)")):
        r = mwu(T[T.model == "C_trunc"][name], T[T.model == "C"][name], "Ctrunc", "C"); r["metric"] = lab; rows.append(r)
    S = pd.DataFrame(rows)[["metric", "n_Ctrunc", "n_C", "median_Ctrunc", "median_C", "p", "rank_biserial"]]
    S.to_csv(OUT / "secondary_mwu.csv", index=False, float_format="%.4f"); log.info("secondary (C_trunc vs C):\n%s", S.round(3).to_string(index=False))
    log.info("n trials with a detected grasp: %s", T.groupby("model").first_grasp_frame.apply(lambda s: int(s.notna().sum())).to_dict())
    log.info("closes distribution:\n%s", pd.crosstab(T.model, T.n_gripper_closes))
    log.info("returned home (success-only expected): %s", T.groupby("model").returned_home.sum().to_dict())
    byp = T.groupby(["position_id", "model"]).agg(n=("success", "size"), success=("success", "sum"), closes_mean=("n_gripper_closes", "mean"),
                                                   grasp_t_med=("first_grasp_time_s", "median"), corr_px_med=("correction_px", "median"),
                                                   aim_x_med=("grasp_aim_err_x", "median"), aim_y_med=("grasp_aim_err_y", "median")).round(2)
    byp.to_csv(OUT / "by_position_model.csv"); log.info("by position:\n%s", byp)

    # same metrics for retest R / C at P1, P2 (different time slot)
    byr = rt.groupby("model").agg(n=("success", "size"), success=("success", "sum"), closes_mean=("n_gripper_closes", "mean"),
                                  grasp_t_med=("first_grasp_time_s", "median"), corr_px_med=("correction_px", "median"),
                                  aim_x_med=("grasp_aim_err_x", "median"), aim_y_med=("grasp_aim_err_y", "median")).round(2)
    log.info("retest reference (P1+P2):\n%s", byr)
    byt = T.groupby("model").agg(n=("success", "size"), success=("success", "sum"), closes_mean=("n_gripper_closes", "mean"),
                                 grasp_t_med=("first_grasp_time_s", "median"), corr_px_med=("correction_px", "median"),
                                 aim_x_med=("grasp_aim_err_x", "median"), aim_y_med=("grasp_aim_err_y", "median")).round(2)
    log.info("this session (P1+P2):\n%s", byt)
    # cube placement quality (balance check)
    log.info("placement distance (px) mean by model: %s", T.groupby("model").place_dist.mean().round(2).to_dict())
    log.info("failures with note 'timeout'-like / early end: %s", T.groupby("model").early_end_key.sum().to_dict())

    # ---- figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    bars = [("P1+P2", "C_trunc", kT, nT, "this session"), ("P1+P2", "C", kC, nC, "this session")]
    cols = [("All (P1+P2)", T), ("P1", T[T.position_id == "P1"]), ("P2", T[T.position_id == "P2"])]
    w = 0.26
    for i, (lab, d) in enumerate(cols):
        for j, m in enumerate(("C_trunc", "C")):
            k, n = kt(d, m); lo, hi = AR.wilson(k, n); x = i + (j - 0.5) * w * 1.1
            ax.bar(x, k / n, w, color=COL[m], label=m if i == 0 else None, zorder=2)
            ax.plot([x, x], [lo, hi], color="#0b0b0b", lw=1.2, zorder=3)
            ax.text(x, min(k / n, 0.9) + 0.02 if k else 0.03, f"{k}/{n}", ha="center", va="bottom", fontsize=9, zorder=4)
    for lab, m, k in (("R", "R", kR), ("C", "C", kRc)):
        ax.axhline(k / 16, color=COL[m], lw=1, ls="--", alpha=0.8)
        ax.text(2.55, k / 16, f"retest {lab} {k}/16", color=COL[m], va="bottom", ha="right", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in cols]); ax.set_ylim(0, 1.08)
    ax.set_ylabel("Success rate (Wilson 95% CI)")
    ax.set_title(f"C_trunc vs C: pooled Fisher p = {P.p.iloc[0]:.4f}; dashed = retest R / C (not interleaved)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.85)); fig.savefig(OUT / "fig1_success.png"); plt.close(fig)

    # speed curves aligned to first grasp
    tt = np.arange(-int(C.WINDOW_S * 30), int(C.AFTER_S * 30) + 1) / 30
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for m in ("C_trunc", "C"):
        sel = [c for c, mm, gf in zip(curves, T.model, T.first_grasp_frame) if mm == m and c is not None]
        if len(sel) >= 2:
            cm = np.stack(sel)
            ax.plot(tt, np.nanmedian(cm, 0), color=COL[m], lw=2, label=f"{m} (median, n={len(sel)})")
            ax.fill_between(tt, np.nanpercentile(cm, 25, 0), np.nanpercentile(cm, 75, 0), color=COL[m], alpha=0.18, lw=0)
    ax.axvline(0, color="#52514e", lw=1); ax.set_xlabel("Time relative to first grasp (s)"); ax.set_ylabel("Arm speed (units/s)")
    ax.set_title("Pre-grasp arm speed (IQR band); trials with a detected grasp"); ax.legend(frameon=False)
    fig.savefig(OUT / "fig2_speed_aligned.png"); plt.close(fig)

    # grasp-aim error scatter
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for m in ("C_trunc", "C"):
        d = T[T.model == m]
        ax.scatter(d.grasp_aim_err_x, d.grasp_aim_err_y, s=36, color=COL[m], label=m, alpha=0.85, edgecolor="white", lw=0.6)
    ax.axhline(0, color="#b5b4ae", lw=0.8); ax.axvline(0, color="#b5b4ae", lw=0.8)
    ax.set_xlabel("aim error x at first grasp (px; + = right of cube)"); ax.set_ylabel("aim error y (px)")
    ax.set_title("Where the gripper closed relative to the cube"); ax.legend(frameon=False)
    fig.savefig(OUT / "fig3_aim_error.png"); plt.close(fig)
    log.info("done")


if __name__ == "__main__":
    main()

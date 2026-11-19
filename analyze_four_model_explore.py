#!/usr/bin/env python
"""Exploratory analysis of the four-model test (protocol/four_model_test.md section 7), after unblinding.
Metrics by the method of analyze_trunc.py. Reads results/four_model_*, writes four_model_analysis/."""
from __future__ import annotations
import hashlib, json, logging, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_cube_positions as A  # noqa
import analyze_grasp_points as G    # noqa
import analyze_correction as C      # noqa
import analyze_retest as AR         # noqa
OUT, RES = HERE / "four_model_analysis", HERE / "results"
COL = {"deep": "#1f9d6b", "R": "#eb6834", "C": "#6250d6", "C_trunc": "#d03a3a"}
MODELS = ["deep", "R", "C", "C_trunc"]
log = logging.getLogger("four")
BRIGHT = 117.5

def load():
    kp = HERE / "protocol/four_model_key.json"
    assert hashlib.sha256(kp.read_bytes()).hexdigest() == (HERE / "protocol/four_model_key.sha256").read_text().strip()
    key = {c: v["model"] for c, v in json.loads(kp.read_text())["key"].items()}
    order = pd.read_csv(HERE / "protocol/four_model_trial_order.csv"); order["model"] = order.model_code.map(key)
    out = pd.read_csv(RES / "four_model_outcomes.csv", dtype={"note": str}).fillna("")
    rows = []
    for r in order.itertuples():
        o = out[(out.trial_id == r.trial_id) & out.outcome.isin(["success", "fail"])]; assert len(o) == 1
        rows.append(dict(trial_id=r.trial_id, position_id=r.position_id, model=r.model, attempt=int(o.attempt.iloc[0]),
                         success=int(o.outcome.iloc[0] == "success"), failure_type=o.failure_type.iloc[0], note=o.note.iloc[0],
                         order=int(r.trial_id[1:]), block=(int(r.trial_id[1:]) - 1) // 4 + 1))
    return pd.DataFrame(rows)

def metrics(T, model, px):
    pl = pd.read_csv(RES / "four_model_placement_log.csv"); rows = []
    for t in T.itertuples():
        meta = json.loads((RES / "four_model_raw" / f"{t.trial_id}_a{t.attempt}" / "meta.json").read_text())
        k = f"four_{t.trial_id}"; A.DATASETS[k] = meta["dataset_repo_id"].split("/", 1)[1]
        g, frames = G.grasp_table(k); e = frames.sort_values("frame_index")
        q = e[[f"state:{j}" for j in G.ARM_JOINTS]].to_numpy(float); grip = e[f"state:{G.GRIPPER}"].to_numpy(float)
        st, ev = None, []
        for i, v in enumerate(grip):
            if v >= AR.CLOSE_OPEN: st = "open"
            elif v <= AR.CLOSE_CLOSED and st == "open": ev.append(i); st = "closed"
        home = [i for i in ev if np.linalg.norm(q[i] - q[0]) < AR.HOME_DIST]
        row = dict(trial_id=t.trial_id, n_frames=len(e), duration_s=len(e) / 30, n_gripper_closes=len(ev) - len(home),
                   returned_home=bool(home), early_end_key=meta.get("early_end_key"),
                   table_gray=meta.get("first_frame_table_gray"), loop_dt_mean=meta.get("loop_dt_mean_ms"),
                   exposure=(meta.get("camera_controls") or {}).get("exposure_time_absolute"),
                   started=pd.to_datetime(meta["started_unix"], unit="s", utc=True).tz_convert("Asia/Tokyo").strftime("%H:%M"))
        p = pl[pl.trial_id == t.trial_id]; p = p[p.timestamp_unix <= meta["started_unix"]].iloc[-1]
        cx, cy = float(p.measured_x), float(p.measured_y); gf = g.grasp_frame.iloc[0]
        row.update(cube_x=cx, cube_y=cy, place_dist=float(p.dist), first_grasp_frame=gf)
        speed = np.r_[0.0, np.linalg.norm(np.diff(q, axis=0), axis=1) * 30]; sm = C.moving_average(speed, C.SMOOTH_FRAMES)
        row.update(mean_speed_first10s=float(sm[:300].mean()), max_speed=float(sm.max()))
        if not np.isnan(gf):
            gf = int(gf); sig = dict(grasp_frame=gf, speed_s=sm, q=q, pan=q[:, G.ARM_JOINTS.index("shoulder_pan.pos")])
            te = C.reach_end(sig, C.REACH_SPEED_THRESH, C.REACH_HOLD_FRAMES); aim = model.predict(q[gf:gf + 1])[0]
            row.update(first_grasp_time_s=gf / 30, grasp_aim_err_x=aim[0] - cx, grasp_aim_err_y=aim[1] - cy)
            if np.isfinite(te):
                te = int(te); row.update(reach_end_s=te / 30, correction_time_s=(gf - te) / 30, correction_px=(sig["pan"][gf] - sig["pan"][te]) * px)
            idx = np.arange(gf - int(C.WINDOW_S * 30), gf + int(C.AFTER_S * 30) + 1)
            row["_curve"] = np.where((idx >= 0) & (idx < len(speed)), sm[np.clip(idx, 0, len(speed) - 1)], np.nan)
        else: row["_curve"] = None
        rows.append(row)
    return pd.DataFrame(rows)

def mwu(a, b, la, lb):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    r = {f"n_{la}": len(a), f"n_{lb}": len(b), f"median_{la}": a.median(), f"median_{lb}": b.median()}
    if len(a) >= 3 and len(b) >= 3:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided"); r.update(p=p, rank_biserial=2 * u / (len(a) * len(b)) - 1)
    return r

def main():
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "explore.log", mode="w")])
    T = load(); reg, px = C.regression_model(); met = metrics(T, reg, px); curves = met.pop("_curve")
    T = T.merge(met, on="trial_id"); T["bright"] = (T.table_gray >= BRIGHT).astype(int)
    T.to_csv(OUT / "trial_table.csv", index=False, float_format="%.3f")
    # ---- metrics by model
    rows = []
    for name, lab in (("n_gripper_closes", "closes (addendum 01)"), ("first_grasp_time_s", "first grasp time (s)"), ("mean_speed_first10s", "mean speed first 10 s"),
                      ("correction_px", "pan correction before grasp (px)"), ("correction_time_s", "correction time (s)"),
                      ("grasp_aim_err_x", "aim x at grasp (px)"), ("grasp_aim_err_y", "aim y at grasp (px)")):
        for a, b in (("deep", "R"), ("deep", "C"), ("R", "C")):
            r = mwu(T[T.model == a][name], T[T.model == b][name], a, b); rows.append(dict(metric=lab, pair=f"{a} vs {b}", **{k.replace(a, "A").replace(b, "B"): v for k, v in r.items()}))
    S = pd.DataFrame(rows); S.to_csv(OUT / "mwu.csv", index=False, float_format="%.4f")
    log.info("MWU (deep/R/C):\n%s", S[["metric", "pair", "median_A", "median_B", "p"]].round(3).to_string(index=False))
    agg = T.groupby("model").agg(n=("success", "size"), success=("success", "sum"), closes_mean=("n_gripper_closes", "mean"), grasp_t_med=("first_grasp_time_s", "median"),
                                 grasp_detected=("first_grasp_frame", lambda s: int(s.notna().sum())), speed10=("mean_speed_first10s", "median"),
                                 corr_px_med=("correction_px", "median"), aim_x_med=("grasp_aim_err_x", "median"), aim_y_med=("grasp_aim_err_y", "median"),
                                 early=("early_end_key", "sum"), dur_med=("duration_s", "median")).reindex(MODELS).round(2)
    log.info("by model:\n%s", agg); agg.to_csv(OUT / "by_model.csv")
    byp = T.groupby(["position_id", "model"]).agg(n=("success", "size"), success=("success", "sum"), closes=("n_gripper_closes", "mean"), grasp_t=("first_grasp_time_s", "median"),
                                                   aim_x=("grasp_aim_err_x", "median"), aim_y=("grasp_aim_err_y", "median")).round(2)
    log.info("by position:\n%s", byp); byp.to_csv(OUT / "by_position_model.csv")
    log.info("closes distribution:\n%s", pd.crosstab(T.model, T.n_gripper_closes).reindex(MODELS))
    # ---- time / lighting / covariates
    log.info("table gray by model: %s", T.groupby("model").table_gray.mean().round(1).reindex(MODELS).to_dict())
    log.info("bright state (>=%.1f) share overall: %.2f", BRIGHT, T.bright.mean())
    ct = T.groupby(["model", "bright"]).success.agg(["sum", "size"]); log.info("success by model x exposure state (sum,size):\n%s", ct)
    for m in MODELS:
        d = T[T.model == m]; r, p = stats.pointbiserialr(d.success, d.table_gray) if d.success.nunique() > 1 else (np.nan, np.nan)
        log.info("%s: table_gray vs success point-biserial r=%.2f p=%.3f", m, r, p)
    log.info("exposure state vs success pooled (Fisher):"); 
    b, d_ = T[T.bright == 1], T[T.bright == 0]
    log.info("  bright %d/%d, dark %d/%d, p=%.3f", b.success.sum(), len(b), d_.success.sum(), len(d_), stats.fisher_exact([[b.success.sum(), len(b)-b.success.sum()], [d_.success.sum(), len(d_)-d_.success.sum()]])[1])
    log.info("trial order / start time / gray / exposure / success:")
    for m in MODELS:
        d = T[T.model == m].sort_values("order"); log.info("%s: %s", m, " ".join(f"{r.trial_id[1:]}{'o' if r.success else 'x'}@{r.started}g{r.table_gray:.0f}" for r in d.itertuples()))
    log.info("loop_dt_mean by model: %s", T.groupby("model").loop_dt_mean.mean().round(3).reindex(MODELS).to_dict())
    log.info("placement dist mean by model: %s", T.groupby("model").place_dist.mean().round(2).reindex(MODELS).to_dict())
    # ---- R second half
    for m in MODELS:
        d = T[T.model == m]; a, b = d[d.order <= 32], d[d.order > 32]
        log.info("%s first half V01-32: %d/%d  second half V33-64: %d/%d  Fisher p=%.3f", m, a.success.sum(), len(a), b.success.sum(), len(b),
                 stats.fisher_exact([[a.success.sum(), len(a)-a.success.sum()], [b.success.sum(), len(b)-b.success.sum()]])[1])
    # ---- reference sessions
    rt = pd.read_csv(HERE / "retest_analysis/trial_table.csv"); rt = rt[rt.position_id.isin(["P1", "P2"])]
    tr = pd.read_csv(HERE / "trunc_analysis/trial_table.csv")
    ref = {"retest R": rt[rt.model == "R"].success, "retest C": rt[rt.model == "C"].success,
           "trunc-session C": tr[tr.model == "C"].success, "trunc-session C_trunc": tr[tr.model == "C_trunc"].success}
    log.info("reference sessions: %s", {k: f"{int(v.sum())}/{len(v)}" for k, v in ref.items()})
    log.info("this session: %s", {m: f"{int(T[T.model==m].success.sum())}/16" for m in MODELS})
    for m, k in (("R", "retest R"), ("C", "retest C"), ("C", "trunc-session C"), ("C_trunc", "trunc-session C_trunc")):
        a = T[T.model == m].success; b = ref[k]
        log.info("  %s (this) %d/16 vs %s %d/%d: Fisher p=%.3f", m, a.sum(), k, b.sum(), len(b), stats.fisher_exact([[a.sum(), len(a)-a.sum()], [b.sum(), len(b)-b.sum()]])[1])
    # ---- figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.8, 4.4)); w = 0.2
    cols = [("All", T), ("P1", T[T.position_id == "P1"]), ("P2", T[T.position_id == "P2"])]
    for i, (lab, d) in enumerate(cols):
        for j, m in enumerate(MODELS):
            dd = d[d.model == m]; k, n = int(dd.success.sum()), len(dd); lo, hi = AR.wilson(k, n); x = i + (j - 1.5) * w * 1.1
            ax.bar(x, k / n, w, color=COL[m], label=m if i == 0 else None, zorder=2); ax.plot([x, x], [lo, hi], color="#0b0b0b", lw=1.2, zorder=3)
            ax.text(x, min(k / n, 0.93) + 0.02 if k else 0.03, f"{k}/{n}", ha="center", va="bottom", fontsize=8, zorder=4)
    ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in cols]); ax.set_ylim(0, 1.12); ax.set_ylabel("Success rate (Wilson 95% CI)")
    ax.set_title("Four-model test: success by model (64 trials)"); ax.legend(frameon=False, ncol=4, loc="upper right"); fig.savefig(OUT / "fig1_success.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for j, m in enumerate(MODELS):
        d = T[T.model == m].sort_values("order")
        ax.scatter(d.order, [j] * len(d), s=70, c=[COL[m] if s else "white" for s in d.success], edgecolor=COL[m], lw=1.6, zorder=3)
    ax.axvline(32.5, color="#b5b4ae", ls="--", lw=1); ax.text(32.8, -0.45, "10 min rest", fontsize=8, color="#52514e")
    ax.set_yticks(range(4)); ax.set_yticklabels(MODELS); ax.set_xlabel("trial number (filled = success)"); ax.set_title("Outcomes in test order")
    fig.savefig(OUT / "fig2_order.png"); plt.close(fig)
    tt = np.arange(-int(C.WINDOW_S * 30), int(C.AFTER_S * 30) + 1) / 30
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for m in MODELS:
        sel = [c for c, mm in zip(curves, T.model) if mm == m and c is not None]
        if len(sel) >= 2:
            cm = np.stack(sel); ax.plot(tt, np.nanmedian(cm, 0), color=COL[m], lw=2, label=f"{m} (n={len(sel)})")
            ax.fill_between(tt, np.nanpercentile(cm, 25, 0), np.nanpercentile(cm, 75, 0), color=COL[m], alpha=0.12, lw=0)
    ax.axvline(0, color="#52514e", lw=1); ax.set_xlabel("Time relative to first grasp (s)"); ax.set_ylabel("Arm speed (units/s)"); ax.legend(frameon=False)
    ax.set_title("Pre-grasp arm speed (median, IQR)"); fig.savefig(OUT / "fig3_speed_aligned.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    for m in MODELS:
        d = T[T.model == m]; ax.scatter(d.grasp_aim_err_x, d.grasp_aim_err_y, s=36, color=COL[m], label=m, alpha=0.85, edgecolor="white", lw=0.6)
    ax.axhline(0, color="#b5b4ae", lw=0.8); ax.axvline(0, color="#b5b4ae", lw=0.8); ax.set_xlabel("aim error x at first grasp (px)"); ax.set_ylabel("aim error y (px)"); ax.legend(frameon=False)
    ax.set_title("Where the gripper closed relative to the cube"); fig.savefig(OUT / "fig4_aim_error.png"); plt.close(fig)
    log.info("done")

if __name__ == "__main__":
    main()

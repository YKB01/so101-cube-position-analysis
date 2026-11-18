#!/usr/bin/env python
"""
Step 2: offline rewind-target selection on recorded failure trajectories.
Data: eval_expert_smooth (baseline policy, 24 episodes, success.csv) + the right-side retest (48 trials; final attempt per trial
from retest_analysis/trial_table.csv, i.e. with the T36 correction). Read only.
Uses rewind_target.py (the module meant for the real-robot loop) and rewind_offline/rewind_index.npz / rewind_params.json.
Writes rewind_offline/: rewind_targets.csv, rewind_false_triggers.csv, rewind_targets_sensitivity.csv, rewind_joint_contrib.csv,
rewind_summary.csv, step2_*.csv/png, step2_log.txt
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import analyze_cube_positions as A   # noqa
import analyze_grasp_points as G     # noqa
import rewind_target as RT           # noqa

OUT = HERE / "rewind_offline"
FPS = 30
BUF_S = 15.0
CLOSE_OPEN, CLOSE_CLOSED, HOME_DIST = 30.0, 15.0, 50.0     # addendum 01 close rule (hysteresis 30/15; home closes excluded)
SETTLE_FRAMES = 30            # the gripper reading is taken 1.0 s after the close event (mean of frames +26..+30)
CLOSE_COUNT_TRIGGER = 3
LEAD_S = 1.0
P_MAIN = {"shallow": 0.9, "deep": 0.4}
P_SENS = [0.3, 0.5, 0.7]
ARMC = [f"state:{j}" for j in G.ARM_JOINTS]
GRIP = f"state:{G.GRIPPER}"
JN = ["pan", "lift", "elbow", "wflex", "wroll"]


def P(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n")


# ------------------------------------------------------------------------------------------------- data
def load_trials():
    trials = []
    f, _ = G.load_frames("eval_expert_smooth")
    sc = pd.read_csv(HERE / "success.csv"); sc = sc[sc.dataset == "eval_expert_smooth"].set_index("episode_index")
    for ep, e in f.groupby("episode_index"):
        trials.append(dict(dataset="eval_expert_smooth", trial=f"E{int(ep):02d}", ep=int(ep), success=int(sc.loc[ep, "success"]),
                           position=sc.loc[ep, "position"], failure_type="", repo="YKB0123/" + A.DATASETS["eval_expert_smooth"], root=None,
                           e=e.sort_values("frame_index").reset_index(drop=True)))
    rt = pd.read_csv(HERE / "retest_analysis/trial_table.csv")
    for r in rt.itertuples():
        meta = json.loads((HERE / f"results/retest_raw/{r.trial_id}_a{r.attempt}/meta.json").read_text())
        A.DATASETS["_rt"] = "rollout_retest/" + meta["dataset_repo_id"].split("/", 1)[1]   # datasets live in YKB0123/rollout_retest/
        ff, _ = G.load_frames("_rt")
        trials.append(dict(dataset="retest", trial=r.trial_id, ep=0, success=int(r.success), position=r.position_id,
                           failure_type="" if pd.isna(r.failure_type) else r.failure_type, repo=meta["dataset_repo_id"], root=str(A.LEROBOT_ROOT / "rollout_retest" / meta["dataset_repo_id"].split("/", 1)[1]),
                           e=ff.sort_values("frame_index").reset_index(drop=True)))
    for t in trials:
        t["q"] = t["e"][ARMC].to_numpy(float); t["g"] = t["e"][GRIP].to_numpy(float)
        t["states"] = np.c_[t["q"], t["g"]]
    return trials


def close_events(t):
    st, ev = None, []
    for i, v in enumerate(t["g"]):
        if v >= CLOSE_OPEN: st = "open"
        elif v <= CLOSE_CLOSED and st == "open": ev.append(i); st = "closed"
    return [i for i in ev if np.linalg.norm(t["q"][i] - t["q"][0]) >= HOME_DIST]       # home closes excluded


def settle_reading(g, i):
    w = g[i + SETTLE_FRAMES - 4: i + SETTLE_FRAMES + 1]
    return float(w.mean()) if len(w) == 5 else np.nan


# ------------------------------------------------------------------------------------------------- gripper reading after a close
def analyse_closes(trials):
    rows = []
    for t in trials:
        ev = close_events(t); t["closes"] = ev
        for k, i in enumerate(ev):
            last = k == len(ev) - 1
            if t["success"] and last: grp = "held cube (last close of a successful trial)"
            elif t["success"]: grp = "earlier close of a successful trial"
            elif t["failure_type"] == "dropped_after_grasp" and k == 0: grp = "failure: grasped then dropped"
            else: grp = "failure: other closes"
            rows.append(dict(dataset=t["dataset"], trial=t["trial"], success=t["success"], k=k, frame=i, group=grp, reading=settle_reading(t["g"], i)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- triggers
def find_trigger(t, theta):
    n = len(t["g"]); cands = []
    for k, i in enumerate(t["closes"]):
        r = settle_reading(t["g"], i)
        if not np.isnan(r) and r < theta:
            cands.append((i + SETTLE_FRAMES, "empty_grasp", k))
    if len(t["closes"]) >= CLOSE_COUNT_TRIGGER:
        cands.append((t["closes"][CLOSE_COUNT_TRIGGER - 1], "closes>=3", CLOSE_COUNT_TRIGGER - 1))
    if cands:
        f, ty, _ = min(cands)
        return f, ty
    return n - 1, "timeout"


def release_frame(t):
    """Success trials: first frame after the last (non-home) close at which the gripper opens again (cube released)."""
    if not t["closes"]:
        return len(t["g"])
    after = np.flatnonzero(t["g"][t["closes"][-1]:] >= CLOSE_OPEN)
    return t["closes"][-1] + int(after[0]) if len(after) else len(t["g"])


# ------------------------------------------------------------------------------------------------- selection
def run_selection(t, trig, index, params, buf_s=BUF_S, p=None, lead=False):
    lo = 0 if buf_s is None else max(0, trig - int(buf_s * FPS))
    buf = t["states"][lo: trig + 1]; ti = len(buf) - 1
    res = RT.select_rewind_target_lead(buf, ti, int(LEAD_S * FPS), index, params) if lead else RT.select_rewind_target(buf, ti, p, index, params)
    return res, lo, buf


def describe(res, lo, trig, t):
    if res.idx is None:
        return dict(frame=np.nan, rewind_s=np.nan, progress=np.nan, nn_dist=np.nan, joint_dist_deg=np.nan, joint_dist_norm=np.nan, reason=res.reason)
    f = lo + res.idx
    q0, q1 = t["q"][trig], t["q"][f]
    return dict(frame=f, rewind_s=(trig - f) / FPS, progress=res.progress, nn_dist=res.distance,
                joint_dist_deg=float(np.linalg.norm(q0 - q1)), joint_dist_norm=float(np.linalg.norm((q0 - q1) / np.array(INDEX.sigma))), reason="")


def main():
    global LOG, INDEX
    LOG = open(OUT / "step2_log.txt", "w")
    index, params = RT.load_index(OUT / "rewind_index.npz", OUT / "rewind_params.json")
    INDEX = index
    trials = load_trials()
    P(f"trials: {len(trials)} ({sum(t['success'] for t in trials)} successful); datasets: "
      f"{pd.Series([t['dataset'] for t in trials]).value_counts().to_dict()}; failures: "
      f"{pd.Series([t['dataset'] for t in trials if not t['success']]).value_counts().to_dict()}")
    P(f"params: tau={params['tau']:.4f} (sensitivity {params['tau_sensitivity']:.4f}); open threshold {params['gripper_open_threshold']}")

    # ---- 1a. gripper reading after a close: empty vs holding the cube --------------------------------------
    C = analyse_closes(trials); C.to_csv(OUT / "step2_close_readings.csv", index=False, float_format="%.3f")
    P("\nsettled gripper reading 1 s after each (non-home) close:\n" + C.groupby("group").reading.describe().round(2).to_string())
    held = C[C.group.str.startswith("held")].reading.dropna()
    emp = C[C.group == "failure: other closes"].reading.dropna()
    sweep = []
    for th in np.arange(3.0, 5.01, 0.1):
        sweep.append(dict(theta=round(th, 2), held_flagged_empty=int((held < th).sum()), n_held=len(held),
                          fail_closes_flagged_empty=int((emp < th).sum()), n_fail_closes=len(emp)))
    SW = pd.DataFrame(sweep); SW.to_csv(OUT / "step2_empty_threshold_sweep.csv", index=False)
    THETA = 4.2
    P(f"empty-grasp threshold theta = {THETA} (lowest held-cube reading = {held.min():.2f}; empty readings cluster at 2.6-3.7): "
      f"held flagged empty {int((held < THETA).sum())}/{len(held)} ({(held < THETA).mean():.1%}); "
      f"failure closes flagged {int((emp < THETA).sum())}/{len(emp)} ({(emp < THETA).mean():.1%}), missed {int((emp >= THETA).sum())}")
    for ds, d in C.groupby("dataset"):
        h = d[d.group.str.startswith("held")].reading; e_ = d[d.group == "failure: other closes"].reading
        P(f"   {ds}: held min {h.min():.2f} max {h.max():.2f} (n={len(h)}); failure closes below theta {(e_ < THETA).sum()}/{len(e_)}")
    P("failure closes NOT flagged (reading >= theta): " + str(C[(C.group == 'failure: other closes') & (C.reading >= THETA)][["trial", "reading"]].round(2).values.tolist()))

    # ---- 1b. triggers ---------------------------------------------------------------------------------------
    fail_rows, succ_rows = [], []
    for t in trials:
        f, ty = find_trigger(t, THETA); t["trig"], t["trig_type"] = f, ty
        if t["success"]:
            rel = release_frame(t)
            # every trigger candidate before the release counts as a false trigger
            cand = [(i + SETTLE_FRAMES, "empty_grasp") for i in t["closes"] if not np.isnan(settle_reading(t["g"], i)) and settle_reading(t["g"], i) < THETA]
            if len(t["closes"]) >= CLOSE_COUNT_TRIGGER: cand.append((t["closes"][CLOSE_COUNT_TRIGGER - 1], "closes>=3"))
            cand = [c for c in cand if c[0] < rel]
            first = min(cand) if cand else (None, "")
            succ_rows.append(dict(dataset=t["dataset"], trial=t["trial"], position=t["position"], n_closes=len(t["closes"]), release_frame=rel,
                                  false_trigger=bool(cand), trigger_type=first[1], trigger_frame=first[0],
                                  trigger_time_s=None if first[0] is None else first[0] / FPS))
    SF = pd.DataFrame(succ_rows); SF.to_csv(OUT / "rewind_false_triggers.csv", index=False, float_format="%.3f")
    fails = [t for t in trials if not t["success"]]
    TY = pd.DataFrame([dict(dataset=t["dataset"], trial=t["trial"], trigger_type=t["trig_type"], n_closes=len(t["closes"])) for t in fails])
    P("\nfailures by earliest trigger:\n" + pd.crosstab(TY.dataset, TY.trigger_type, margins=True).to_string())
    P(f"false triggers in successful trials: {int(SF.false_trigger.sum())}/{len(SF)} " + str(SF[SF.false_trigger][["dataset", "trial", "trigger_type"]].values.tolist()))
    # what would each rule alone cover
    cov = {}
    for t in fails:
        e = sum(1 for i in t["closes"] if settle_reading(t["g"], i) < THETA); c3 = len(t["closes"]) >= CLOSE_COUNT_TRIGGER
        cov[t["trial"]] = (e > 0, c3)
    P(f"failures with an empty-grasp event: {sum(v[0] for v in cov.values())}/{len(cov)}; with >=3 closes: {sum(v[1] for v in cov.values())}/{len(cov)}; "
      f"either: {sum(v[0] or v[1] for v in cov.values())}/{len(cov)}")
    # the real cases that only the timeout catches: what is their failure like?
    to = [t for t in fails if t["trig_type"] == "timeout"]
    P("timeout-only failures: " + str([(t["trial"], len(t["closes"]), t["failure_type"]) for t in to]))

    # ---- 2. selection --------------------------------------------------------------------------------------
    rows, contrib_rows, sens_rows = [], [], []
    taus = {"default": params["tau"], "sens": params["tau_sensitivity"]}
    settings = [("shallow", dict(p=0.9)), ("deep", dict(p=0.4)), ("shallow_lead1s", dict(lead=True))]
    sens_settings = [(f"p={p}", dict(p=p)) for p in P_SENS] + settings
    for t in fails:
        trig = t["trig"]
        row = dict(dataset=t["dataset"], trial=t["trial"], position=t["position"], failure_type=t["failure_type"],
                   trigger_type=t["trig_type"], trigger_frame=trig, trigger_time_s=trig / FPS, n_frames=len(t["g"]))
        for name, kw in settings:
            res, lo, buf = run_selection(t, trig, index, params, **kw)
            d = describe(res, lo, trig, t)
            row.update({f"{name}_{k}": v for k, v in d.items()})
            row[f"{name}_n_candidates"] = res.n_candidates
            # joint contributions: no target -> nearest miss; chosen with distance >= 0.8 tau -> that state
            if res.idx is None or res.distance >= 0.8 * params["tau"]:
                s = buf[: (len(buf) if res.idx is None else res.idx + 1)]
                if res.idx is not None:
                    pick = res.idx
                else:
                    op = np.flatnonzero(s[:, 5] >= params["gripper_open_threshold"])
                    cand_i = op if len(op) else np.arange(len(s))
                    dd, _, _ = RT.joint_contributions(s[cand_i, :5], index)
                    pick = int(cand_i[int(np.argmin(dd))])
                dd, share, ad = RT.joint_contributions(s[pick:pick + 1, :5], index)
                contrib_rows.append(dict(dataset=t["dataset"], trial=t["trial"], setting=name,
                                         case="no_target" if res.idx is None else "near_tau",
                                         reason=res.reason or "", frame=lo + pick, nn_dist=float(dd[0]),
                                         **{f"share_{n}": float(share[0, j]) for j, n in enumerate(JN)},
                                         **{f"absdiff_{n}": float(ad[0, j]) for j, n in enumerate(JN)},
                                         dominant=JN[int(np.argmax(share[0]))]))
        # frame difference between the two shallow definitions
        a, b = row["shallow_frame"], row["shallow_lead1s_frame"]
        row["shallow_p_vs_lead_diff_frames"] = (a - b) if not (np.isnan(a) or np.isnan(b)) else np.nan
        rows.append(row)
        # sensitivity: tau default vs 0.404, buffer 15 s vs unlimited, p = 0.3/0.5/0.7 + main
        for tau_name, tau in taus.items():
            pr = dict(params, tau=tau)
            for buf_s, bname in ((BUF_S, "15s"), (None, "unlimited")):
                for name, kw in sens_settings:
                    res, lo, _ = run_selection(t, trig, index, pr, buf_s=buf_s, **kw)
                    d = describe(res, lo, trig, t)
                    sens_rows.append(dict(dataset=t["dataset"], trial=t["trial"], tau=tau_name, buffer=bname, setting=name, **d))
    R = pd.DataFrame(rows); R.to_csv(OUT / "rewind_targets.csv", index=False, float_format="%.4f")
    S = pd.DataFrame(sens_rows); S.to_csv(OUT / "rewind_targets_sensitivity.csv", index=False, float_format="%.4f")
    CB = pd.DataFrame(contrib_rows); CB.to_csv(OUT / "rewind_joint_contrib.csv", index=False, float_format="%.4f")

    # ---- summaries --------------------------------------------------------------------------------------------
    def summ(df, setting_col):
        out = []
        for ds, d in list(df.groupby("dataset")) + [("all", df)]:
            for name in ("shallow", "deep", "shallow_lead1s"):
                x = d[f"{name}_rewind_s"]; ok = x.notna()
                out.append(dict(dataset=ds, setting=name, n_failures=len(d), selected=int(ok.sum()), selected_rate=ok.mean(),
                                rewind_s_median=x.median(), rewind_s_min=x.min(), rewind_s_max=x.max(),
                                nn_dist_median=d[f"{name}_nn_dist"].median(), joint_dist_deg_median=d[f"{name}_joint_dist_deg"].median()))
        return pd.DataFrame(out)
    SM = summ(R, None); SM.to_csv(OUT / "rewind_summary.csv", index=False, float_format="%.3f")
    P("\nSUMMARY (default tau, 15 s cache):\n" + SM.round(2).to_string(index=False))
    P("\nreasons for no target (default settings):")
    for name in ("shallow", "deep", "shallow_lead1s"):
        r = R[R[f"{name}_frame"].isna()]
        P(f"  {name}: {len(r)} -> " + str(r[f"{name}_reason"].value_counts().to_dict()))
    d = R.dropna(subset=["shallow_p_vs_lead_diff_frames"]).shallow_p_vs_lead_diff_frames
    P(f"\nshallow p=0.9 vs shallow lead>=1.0 s: both selected in {len(d)} trials; frame difference (p-rule minus lead-rule) median {d.median():.0f} "
      f"(IQR {d.quantile(.25):.0f}..{d.quantile(.75):.0f}), same frame in {(d == 0).sum()}; p-rule later than lead-rule in {(d > 0).sum()}, earlier in {(d < 0).sum()}")
    # sensitivity summary
    SS = S.assign(found=S.frame.notna()).groupby(["tau", "buffer", "setting"]).agg(
        n=("found", "size"), found=("found", "sum"), rewind_s_median=("rewind_s", "median"), rewind_s_max=("rewind_s", "max")).reset_index()
    SS.to_csv(OUT / "rewind_sensitivity_summary.csv", index=False, float_format="%.3f")
    P("\nSENSITIVITY (all 29 failures):\n" + SS.round(2).to_string(index=False))
    # 15 s buffer sufficiency
    U = S[(S.tau == "default") & (S.buffer == "unlimited")]
    for name in ("shallow", "deep", "shallow_lead1s"):
        u = U[U.setting == name].set_index("trial"); b = S[(S.tau == "default") & (S.buffer == "15s") & (S.setting == name)].set_index("trial")
        P(f"{name}: found with unlimited buffer {u.frame.notna().sum()}, with 15 s {b.frame.notna().sum()}; unlimited-buffer targets farther than 15 s: {(u.rewind_s > BUF_S).sum()}; "
          f"max lead {u.rewind_s.max():.1f} s")
    # contributions
    if len(CB):
        shcols = [f"share_{n}" for n in JN]
        CBu = CB.drop_duplicates(["trial", "frame"])          # the same state can be listed under several settings
        P(f"\njoint contributions: {len(CB)} rows, {len(CBu)} distinct states in {CBu.trial.nunique()} trials ({CBu.case.value_counts().to_dict()}; no 'no_target' case occurred); "
          "mean share of squared distance: " + str(CBu[shcols].mean().round(2).to_dict()) + "; dominant joint: " + str(CBu.dominant.value_counts().to_dict()))
        P("   (a uniform split would be 0.20 each; wroll sigma is only 5.6 deg)")
        P("   mean abs deviation per joint in joint units (deg): " + str((CBu[[f"absdiff_{n}" for n in JN]].mean() * np.array(index.sigma)).round(1).to_dict()))
        P("   per state:\n" + CBu[["trial", "setting", "nn_dist", "dominant"] + shcols].round(2).to_string(index=False))
    # reference: which joints dominate the distance of expert frames above tau in the leave-one-episode-out check?
    z = np.load(OUT / "rewind_index.npz"); from scipy.spatial import cKDTree
    eps, Zs = z["episode"], z["states_norm"]; ref = []
    for ep in np.unique(eps):
        m = eps == ep; _, i = cKDTree(Zs[~m]).query(Zs[m]); diff = Zs[m] - Zs[~m][i]; sq = diff ** 2
        d = np.sqrt(sq.sum(1)); sel = d > params["tau"]
        ref.append(sq[sel] / sq[sel].sum(1, keepdims=True))
    ref = np.concatenate(ref)
    P(f"reference, expert frames above tau in the leave-one-episode-out check (n={len(ref)}): mean share " + str(dict(zip(JN, ref.mean(0).round(2)))) +
      "; dominant: " + str(pd.Series([JN[k] for k in ref.argmax(1)]).value_counts().to_dict()))
    return trials, R, S, C, SW, SF, CB, params, index


def get_frame(repo, global_idx):
    from lerobot.datasets import LeRobotDataset
    ds = LeRobotDataset(repo)
    return ds, np.asarray(ds[global_idx]["observation.images.front"].permute(1, 2, 0).clip(0, 1) * 255, dtype=np.uint8)


def make_figures(trials, R, C, SW, params, index):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9.5, "savefig.dpi": 170, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
    by = {(t["dataset"], t["trial"]): t for t in trials}
    # --- choose 3 representative failures
    r1 = R[(R.dataset == "eval_expert_smooth")]; r1 = r1.iloc[(r1.deep_rewind_s - r1.deep_rewind_s.median()).abs().argsort()[:1]]
    r2 = R[(R.dataset == "retest") & (R.trigger_type == "empty_grasp")]; r2 = r2.iloc[(r2.deep_rewind_s - r2.deep_rewind_s.median()).abs().argsort()[:1]]
    r3 = R[R.trigger_type == "timeout"].sort_values("deep_rewind_s").iloc[-1:]
    pick = pd.concat([r1, r2, r3]); pick.to_csv(OUT / "step2_figure_trials.csv", index=False)
    P("figure trials: " + str(pick[["dataset", "trial", "trigger_type", "shallow_rewind_s", "deep_rewind_s"]].values.tolist()))
    # --- Fig A
    fig, axs = plt.subplots(3, 3, figsize=(15, 8.6), sharex="col")
    for c, r in enumerate(pick.itertuples()):
        t = by[(r.dataset, r.trial)]; tt = np.arange(len(t["g"])) / FPS
        d, nn = index.query(t["q"] / 1.0); prog = index.progress[nn]
        ax = axs[0, c]; ax.plot(tt, d, color="#2a78d6", lw=1.2)
        ax.axhline(params["tau"], color="#0b0b0b", ls="-", lw=1); ax.axhline(params["tau_sensitivity"], color="#0b0b0b", ls=":", lw=1)
        ax.set_ylabel("NN distance (normalised)"); ax.set_title(f"{r.dataset} / {r.trial}  (trigger: {r.trigger_type})", fontsize=10)
        ax.text(tt[-1], params["tau"], " tau", va="bottom", ha="right", fontsize=8)
        ax = axs[1, c]; ax.plot(tt, prog, color="#6250d6", lw=1.2); ax.axhline(0.9, color="#eb6834", ls="--", lw=1); ax.axhline(0.4, color="#1baf7a", ls="--", lw=1)
        ax.set_ylabel("Estimated progress"); ax.set_ylim(-0.05, 1.05)
        ax = axs[2, c]; ax.plot(tt, t["g"], color="#8a8984", lw=1.2); ax.axhline(params["gripper_open_threshold"], color="#0b0b0b", ls=":", lw=1)
        ax.set_ylabel("Gripper reading"); ax.set_xlabel("Time (s)")
        for ax in axs[:, c]:
            ax.axvline(r.trigger_time_s, color="#d03a3a", lw=1.5)
            ax.axvline(r.shallow_frame / FPS, color="#eb6834", lw=1.5); ax.axvline(r.deep_frame / FPS, color="#1baf7a", lw=1.5)
            ax.axvspan(max(0, r.trigger_frame / FPS - BUF_S), r.trigger_time_s, color="#f0efe9", alpha=0.5, zorder=0)
        axs[0, c].text(r.trigger_time_s, axs[0, c].get_ylim()[1], " trigger", color="#d03a3a", va="top", ha="right", fontsize=8)
    from matplotlib.lines import Line2D
    axs[0, 0].legend(handles=[Line2D([0], [0], color="#d03a3a", lw=1.5, label="trigger"), Line2D([0], [0], color="#eb6834", lw=1.5, label="shallow target (p=0.9)"),
                              Line2D([0], [0], color="#1baf7a", lw=1.5, label="deep target (p=0.4)")], frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Figure A: nearest-neighbour distance and estimated progress along three failed trials (shaded = 15 s cache; solid line = tau, dotted = sensitivity tau)", y=0.995)
    fig.savefig(OUT / "step2_figA_distance_progress.png"); plt.close(fig)
    # --- Fig B
    fig, axs = plt.subplots(3, 3, figsize=(13.5, 9.6))
    for rr, r in enumerate(pick.itertuples()):
        t = by[(r.dataset, r.trial)]
        e0 = 0
        for cc, (nm, fr, col) in enumerate((("Trigger", int(r.trigger_frame), "#d03a3a"), ("Shallow target (p = 0.9)", int(r.shallow_frame), "#eb6834"), ("Deep target (p = 0.4)", int(r.deep_frame), "#1baf7a"))):
            ds, img = get_frame_ep(t, fr)
            ax = axs[rr, cc]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(col); sp.set_linewidth(3)
            if nm == "Trigger": sub = f"t = {fr / FPS:.1f} s\n({r.trigger_type})"
            else:
                key = "shallow" if "Shallow" in nm else "deep"
                sub = f"t = {fr / FPS:.1f} s ({getattr(r, key + '_rewind_s'):.1f} s before trigger)\nprogress {getattr(r, key + '_progress'):.2f}, joint distance {getattr(r, key + '_joint_dist_deg'):.0f} deg"
            ax.set_title(f"{r.dataset}/{r.trial}: {nm}\n{sub}", fontsize=8.5)
    fig.suptitle("Figure B: camera view at the trigger, the shallow target and the deep target", y=0.995)
    fig.savefig(OUT / "step2_figB_camera_views.png"); plt.close(fig)
    # --- Fig C
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    names = [("shallow", "Shallow\n(p = 0.9)", "#eb6834"), ("shallow_lead1s", "Shallow\n(>= 1.0 s lead)", "#d6a52a"), ("deep", "Deep\n(p = 0.4)", "#1baf7a")]
    rng = np.random.default_rng(0)
    for i, (nm, lab, col) in enumerate(names):
        x = R[f"{nm}_rewind_s"].dropna().to_numpy()
        ax.boxplot(x, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color="#0b0b0b"), boxprops=dict(color=col), whiskerprops=dict(color=col), capprops=dict(color=col))
        for ds, mk in (("eval_expert_smooth", "o"), ("retest", "^")):
            y = R[R.dataset == ds][f"{nm}_rewind_s"].dropna().to_numpy(); ax.scatter(i + rng.uniform(-0.18, 0.18, len(y)), y, s=22, marker=mk, color=col, alpha=0.75, edgecolor="white", lw=0.5,
                                                                              label=ds if i == 0 else None)
    ax.axhline(BUF_S, color="#d03a3a", ls="--", lw=1); ax.text(2.45, BUF_S, "15 s cache", color="#d03a3a", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels([n[1] for n in names]); ax.set_ylabel("Rewind duration: trigger minus target (s)"); ax.set_ylim(0, 16.5)
    ax.legend(frameon=False, loc="upper left"); ax.set_title("Figure C: rewind duration per setting (29 failed trials)")
    fig.savefig(OUT / "step2_figC_rewind_duration.png"); plt.close(fig)
    # extra: gripper reading after a close
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    order = ["held cube (last close of a successful trial)", "earlier close of a successful trial", "failure: grasped then dropped", "failure: other closes"]
    cols = ["#2a78d6", "#8a8984", "#d6a52a", "#d03a3a"]
    ax = axs[0]
    for k, (g_, c_) in enumerate(zip(order, cols)):
        y = C[C.group == g_].reading.dropna().clip(upper=13); ax.scatter(y, np.full(len(y), k) + rng.uniform(-0.15, 0.15, len(y)), s=18, color=c_, alpha=0.8)
    ax.axvline(4.2, color="#0b0b0b", lw=1.2); ax.set_yticks(range(4)); ax.set_yticklabels([o.replace(" (", "\n(") for o in order], fontsize=7.5)
    ax.set_xlabel("Gripper reading 1 s after the close (clipped at 13)"); ax.set_title("Reading after a close; line = empty-grasp threshold 4.2")
    ax = axs[1]
    ax.plot(SW.theta, SW.held_flagged_empty / SW.n_held, color="#2a78d6", label="held-cube closes flagged empty (false alarm)")
    ax.plot(SW.theta, SW.fail_closes_flagged_empty / SW.n_fail_closes, color="#d03a3a", label="failure closes flagged empty (detected)")
    ax.axvline(4.2, color="#0b0b0b", lw=1.2); ax.set_xlabel("Threshold"); ax.set_ylabel("Fraction"); ax.legend(frameon=False, fontsize=8, loc="center left"); ax.set_title("Threshold sweep")
    fig.savefig(OUT / "step2_fig_empty_grasp_reading.png"); plt.close(fig)


_DS_CACHE = {}
def get_frame_ep(t, fr):
    from lerobot.datasets import LeRobotDataset
    ds = _DS_CACHE.setdefault(t["repo"], LeRobotDataset(t["repo"], root=t["root"]))
    base = int(ds.meta.episodes["dataset_from_index"][t["ep"]])
    return ds, np.asarray(ds[base + fr]["observation.images.front"].permute(1, 2, 0).clip(0, 1) * 255, dtype=np.uint8)


if __name__ == "__main__":
    trials, R, S, C, SW, SF, CB, params, index = main()
    make_figures(trials, R, C, SW, params, index)

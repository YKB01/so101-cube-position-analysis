#!/usr/bin/env python
"""
Phase 0: corrected deep-rewind rule (rewind_target.select_rewind_target_deep) re-run on the 29 failed trajectories of step 2.
Writes rewind_offline/: phase0_deep_fix.csv, phase0_summary.txt, phase0_figB_four_trials.png
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "rewind_offline"))
import step2_offline_targets as S2   # noqa
import rewind_target as RT           # noqa

OUT = HERE / "rewind_offline"
FPS = 30


def main():
    index, params = RT.load_index(OUT / "rewind_index.npz", OUT / "rewind_params.json")
    S2.INDEX = index
    log = open(OUT / "phase0_summary.txt", "w")
    def P(*a):
        s = " ".join(str(x) for x in a); print(s); log.write(s + "\n")
    trials = S2.load_trials(); S2.analyse_closes(trials)
    rows = []
    for t in trials:
        if t["success"]: continue
        trig, ty = S2.find_trigger(t, 4.2); t["trig"], t["trig_type"] = trig, ty
        lo = max(0, trig - int(S2.BUF_S * FPS)); buf = t["states"][lo: trig + 1]; ti = len(buf) - 1
        sh = RT.select_rewind_target_lead(buf, ti, int(params["shallow_lead_s"] * FPS), index, params)
        old = RT.select_rewind_target(buf, ti, params["p_deep"], index, params)
        new = RT.select_rewind_target_deep(buf, ti, index, params)
        # literal reading (level 2 without the gap requirement), for comparison only
        lit = RT.select_rewind_target(buf, ti, params["deep_fallback_progress"], index, params)
        q_trig = t["q"][trig]
        def jd(pose): return float(np.linalg.norm(q_trig - pose))
        row = dict(dataset=t["dataset"], trial=t["trial"], trigger_type=ty, trigger_frame=trig, buffer_start=lo,
                   shallow_frame=lo + sh.idx, shallow_s=(ti - sh.idx) / FPS, shallow_joint_deg=jd(buf[sh.idx, :5]),
                   old_deep_frame=lo + old.idx, old_deep_s=(ti - old.idx) / FPS, old_gap_s=(old.idx and (sh.idx - old.idx) / FPS),
                   level=new.level, new_deep_frame=np.nan if new.idx is None else lo + new.idx,
                   new_deep_progress=new.progress, new_deep_nn=new.distance, deep_joint_deg=jd(new.target_pose),
                   literal_level2_frame=np.nan if lit.idx is None else lo + lit.idx)
        if new.level == "start_pose":
            row["new_deep_s"] = ti / FPS             # playback of the whole cache (then interpolation to the start pose)
            row["new_gap_s"] = np.nan
        else:
            row["new_deep_s"] = (ti - new.idx) / FPS; row["new_gap_s"] = (sh.idx - new.idx) / FPS
        row["old_gap_s"] = (sh.idx - old.idx) / FPS
        row["start_pose_joint_deg_from_trigger"] = jd(np.asarray(params["start_pose"]))
        row["buffer_s"] = ti / FPS
        rows.append(row)
    R = pd.DataFrame(rows); R.to_csv(OUT / "phase0_deep_fix.csv", index=False, float_format="%.3f")
    P(f"params: deep_min_gap_s={params['deep_min_gap_s']}, deep_fallback_progress={params['deep_fallback_progress']}, start_pose={np.round(params['start_pose'], 1).tolist()}")
    P(f"29 failures; level used: {R.level.value_counts().to_dict()}")
    P("by dataset:\n" + pd.crosstab(R.dataset, R.level).to_string())
    P("\nold deep gap to shallow (s): median %.2f, min %.2f; trials with gap < 2.0 s: %d %s" % (R.old_gap_s.median(), R.old_gap_s.min(), (R.old_gap_s < 2).sum(), R[R.old_gap_s < 2].trial.tolist()))
    P("\ntrials that changed level:\n" + R[R.level != "original"][["trial", "trigger_type", "level", "shallow_s", "old_deep_s", "old_gap_s", "new_deep_s", "new_gap_s", "new_deep_progress", "deep_joint_deg", "start_pose_joint_deg_from_trigger"]].round(2).to_string(index=False))
    P("\nafter the correction: shallow->deep time difference (s), non-start-pose levels: median %.2f, min %.2f, max %.2f (n=%d)" %
      (R.new_gap_s.median(), R.new_gap_s.min(), R.new_gap_s.max(), R.new_gap_s.notna().sum()))
    P("rewind duration (s) new deep: median %.2f, min %.2f, max %.2f (start_pose rows = whole cache length)" % (R.new_deep_s.median(), R.new_deep_s.min(), R.new_deep_s.max()))
    P("joint distance trigger->deep target (deg): median %.1f, min %.1f, max %.1f; shallow: median %.1f" % (R.deep_joint_deg.median(), R.deep_joint_deg.min(), R.deep_joint_deg.max(), R.shallow_joint_deg.median()))
    P("literal reading of level 2 (without the gap condition) would pick a state for: %d of the %d trials that left level 1" % (R[R.level != 'original'].literal_level2_frame.notna().sum(), (R.level != 'original').sum()))
    P("levels where the level-2 candidate also fails the gap: " + str(R[(R.level == 'start_pose')].trial.tolist()))
    # ---- replay points and estimated duration, before / after resampling (max 1.5 deg per joint per step)
    MAXSTEP = params["rewind_max_step_deg_default"]; rows2 = []
    for t in trials:
        if t["success"]: continue
        trig = t["trig"]; lo = max(0, trig - int(S2.BUF_S * FPS)); buf = t["states"][lo: trig + 1]; ti = len(buf) - 1
        sh = RT.select_rewind_target_lead(buf, ti, int(params["shallow_lead_s"] * FPS), index, params)
        dp = RT.select_rewind_target_deep(buf, ti, index, params)
        for name, idx in (("shallow", sh.idx), ("deep", dp.idx)):
            if name == "deep" and dp.level == "start_pose": idx = None
            w0 = RT.plan_rewind_path(buf, ti, idx, params, resample=False); w1 = RT.plan_rewind_path(buf, ti, idx, params, resample=True)
            s0, s1 = RT.estimate_rewind_steps(w0, MAXSTEP), RT.estimate_rewind_steps(w1, MAXSTEP)
            L = lambda w: float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum())
            rows2.append(dict(dataset=t["dataset"], trial=t["trial"], setting=name, level=("shallow" if name == "shallow" else dp.level),
                              points_before=len(w0), points_after=len(w1), steps_before=s0, steps_after=s1, est_s_before=s0 / FPS, est_s_after=s1 / FPS,
                              path_deg_before=L(w0), path_deg_after=L(w1), frames_in_segment=len(w0)))
    R2 = pd.DataFrame(rows2); R2.to_csv(OUT / "phase0_replay_points.csv", index=False, float_format="%.3f")
    P("\nREPLAY: waypoints and estimated duration at %.1f deg/joint/step, min step %.1f deg (before = every frame of the segment, after = resampled)" % (MAXSTEP, params["rewind_resample_min_step_deg"]))
    for name in ("shallow", "deep"):
        d = R2[R2.setting == name]
        P(f"  {name}: waypoints median {d.points_before.median():.0f} -> {d.points_after.median():.0f} (total {d.points_before.sum()} -> {d.points_after.sum()}); "
          f"estimated duration median {d.est_s_before.median():.2f} s -> {d.est_s_after.median():.2f} s, max {d.est_s_before.max():.1f} s -> {d.est_s_after.max():.1f} s")
    P("  per level (deep): " + str(R2[R2.setting == "deep"].groupby("level")[["points_before", "points_after", "est_s_before", "est_s_after"]].median().round(2).to_dict("index")))
    P("  T21 / T46 / E01 / T08 (deep):\n" + R2[(R2.setting == "deep") & R2.trial.isin(["T21", "T46", "E01", "T08"])][["trial", "level", "frames_in_segment", "points_before", "points_after", "est_s_before", "est_s_after", "path_deg_before", "path_deg_after"]].round(2).to_string(index=False))
    # duration of the start-pose replays at other speed limits, and direct interpolation for comparison
    for t in trials:
        if t["success"] or t["trial"] not in ("T21", "T46"): continue
        trig = t["trig"]; lo = max(0, trig - int(S2.BUF_S * FPS)); buf = t["states"][lo: trig + 1]; ti = len(buf) - 1
        w = RT.plan_rewind_path(buf, ti, None, params)
        direct = np.vstack([buf[ti, :5], np.asarray(params["start_pose"])])
        P(f"  {t['trial']} start-pose replay (resampled, {len(w)} waypoints): " + ", ".join(f"{m:g} deg/step -> {RT.estimate_rewind_steps(w, m) / FPS:.1f} s" for m in (1.5, 2.0, 3.0))
          + f"; direct interpolation trigger->start pose: " + ", ".join(f"{m:g} -> {RT.estimate_rewind_steps(direct, m) / FPS:.1f} s" for m in (1.5, 2.0, 3.0)))
    P("  all trials, segments where the 1.5 deg limit (not the recorded speed) sets the duration: before %d of %d deep+shallow" % ((R2.steps_before > R2.frames_in_segment).sum(), len(R2)))
    # ---- figure: four trials
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    by = {(t["dataset"], t["trial"]): t for t in trials}
    four = ["T46", "E01", "E16", "T21", "T08"]      # the four named in the request + T08 (also < 2 s apart with the time-rule shallow target)
    sel = R.set_index("trial").loc[four].reset_index()
    fig, axs = plt.subplots(5, 4, figsize=(17, 15.4))
    for rr, r in enumerate(sel.itertuples()):
        t = by[(r.dataset, r.trial)]
        items = [("Trigger", int(r.trigger_frame), "#d03a3a", f"t = {r.trigger_frame / FPS:.1f} s ({r.trigger_type})"),
                 ("Shallow (>= 1.0 s lead)", int(r.shallow_frame), "#eb6834", f"{r.shallow_s:.1f} s before trigger, {r.shallow_joint_deg:.0f} deg away"),
                 ("Old deep (p = 0.4)", int(r.old_deep_frame), "#8a8984", f"{r.old_deep_s:.1f} s before trigger ({r.old_gap_s:.1f} s before shallow)")]
        if r.level == "start_pose":
            items.append(("New deep: start pose", int(t["e"].frame_index.iloc[0]), "#1baf7a", f"start pose (first frame shown), {r.deep_joint_deg:.0f} deg away"))
        else:
            items.append((f"New deep ({'p <= 0.2' if r.level == 'progress_le_0.2' else 'original rule'})", int(r.new_deep_frame), "#1baf7a",
                          f"{r.new_deep_s:.1f} s before trigger, {r.new_deep_joint_deg if False else r.deep_joint_deg:.0f} deg away"))
        for cc, (nm, fr, col, sub) in enumerate(items):
            _, img = S2.get_frame_ep(t, fr)
            ax = axs[rr, cc]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(col); sp.set_linewidth(3)
            ax.set_title(f"{r.dataset}/{r.trial}: {nm}\n{sub}", fontsize=8.5)
    fig.suptitle("Phase 0: T46, E01, E16, T21 (named in the request) and T08, with the corrected deep rule (rightmost column); E16 keeps the original rule", y=0.995)
    fig.text(0.5, 0.085, "Note: in the rightmost column, 'start pose' rows show the FIRST FRAME of the trial, not the actual camera view after a rewind. "
             "Their targets are the stored start pose (median of the idle expert frames).", ha="center", fontsize=9)
    fig.savefig(OUT / "phase0_figB_four_trials.png", dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()

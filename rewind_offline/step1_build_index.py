#!/usr/bin/env python
"""
Step 1 of the rewind-target prototype: expert index, in-distribution threshold tau, progress, gripper threshold.
Reads the 50 expert demonstrations (config of analyze_grasp_points.py; first-grasp rule = analyze_grasp_points.detect_first_grasp).
Writes (all in rewind_offline/): rewind_index.npz, rewind_params.json, step1_*.csv, step1_*.png, step1_log.txt.
The original dataset is only read.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import analyze_grasp_points as G   # noqa: E402
import analyze_correction as C     # noqa: E402

OUT = HERE / "rewind_offline"
ARM = [f"state:{j}" for j in G.ARM_JOINTS]
GRIP = f"state:{G.GRIPPER}"
ONSET_SPEED, ONSET_HOLD, ONSET_BACK_SPEED = 15.0, 10, 3.0   # arm-motion onset (see report)
GRASP_DETECTION_OPEN = 30.0     # = analyze_grasp_points.OPEN_MIN (grasp detection keeps using this)
GRIPPER_OPEN_THRESHOLD = 40.0   # "gripper is open" for the rewind target (user decision after step 1: stricter, avoids transition frames)
P_SHALLOW, P_DEEP = 0.9, 0.4
DEEP_MIN_GAP_S, DEEP_FALLBACK_PROGRESS = 2.0, 0.2   # phase-0 correction of the deep rule (see rewind_target.select_rewind_target_deep)


def motion_onset(q: np.ndarray) -> int:
    """First frame of the first run of >= ONSET_HOLD frames with smoothed arm speed > ONSET_SPEED units/s,
    moved back to where the speed last was <= ONSET_BACK_SPEED. (A single-frame start-up glitch in episode 0 is ignored.)"""
    sp = np.r_[0.0, np.linalg.norm(np.diff(q, axis=0), axis=1) * 30.0]
    sm = C.moving_average(sp, C.SMOOTH_FRAMES)
    run = 0
    for t, a in enumerate(sm > ONSET_SPEED):
        run = run + 1 if a else 0
        if run >= ONSET_HOLD:
            k = t - ONSET_HOLD + 1
            while k > 0 and sm[k - 1] > ONSET_BACK_SPEED:
                k -= 1
            return k
    return 0


def main():
    OUT.mkdir(exist_ok=True)
    log = open(OUT / "step1_log.txt", "w")
    def P(*a):
        s = " ".join(str(x) for x in a); print(s); log.write(s + "\n")

    f, info = G.load_frames("expert")
    names = info["features"]["observation.state"]["names"]
    P("state names:", names, "| fps:", info["fps"], "| episodes:", f.episode_index.nunique(), "| frames:", len(f))
    P(f[ARM + [GRIP]].describe().loc[["min", "max", "mean", "std"]].round(2).to_string())
    P("units: meta/info.json lists only names (no unit field). In the installed LeRobot, SOFollowerConfig.use_degrees defaults to True -> "
      "arm joints are in degrees (MotorNormMode.DEGREES), the gripper in 0..100 (RANGE_0_100). The data agree: shoulder_lift spans "
      "-104..74, elbow 96..-94, which exceeds the -100..100 of the other normalisation mode only slightly and is a plausible degree range.")

    # ---- per-frame table ---------------------------------------------------------------------
    rows = []
    for ep, e in f.groupby("episode_index"):
        q = e[ARM].to_numpy(float); g = e[GRIP].to_numpy(float)
        gf = int(G.detect_first_grasp(G.gripper_series(e))["grasp_frame"])
        on = motion_onset(q)
        t = np.arange(len(e))
        prog = np.where(t < on, 0.0, np.where(t <= gf, (t - on) / max(gf - on, 1), np.nan))
        phase = np.where(t < on, "pre_motion", np.where(t <= gf, "approach", "post_grasp"))
        rows.append(pd.DataFrame(dict(episode=ep, frame=t, progress=prog, phase=phase, gripper=g,
                                      **{f"q{i}": q[:, i] for i in range(5)}, onset=on, grasp_frame=gf)))
    D = pd.concat(rows, ignore_index=True)
    ep_tab = D.groupby("episode")[["onset", "grasp_frame"]].first()
    P(f"\nmotion onset (frames): median {ep_tab.onset.median():.0f}, min {ep_tab.onset.min()}, max {ep_tab.onset.max()}; "
      f"first grasp: median {ep_tab.grasp_frame.median():.0f}, min {ep_tab.grasp_frame.min()}, max {ep_tab.grasp_frame.max()}")
    P("frames by phase:", D.phase.value_counts().to_dict(),
      f"| idle share of the pre-grasp part: {(D.phase == 'pre_motion').sum() / (D.phase != 'post_grasp').sum():.1%}")
    ep_tab.to_csv(OUT / "step1_episode_onsets.csv")

    q_all = D[[f"q{i}" for i in range(5)]].to_numpy(float)
    sigma = q_all.std(axis=0, ddof=0)
    P("per-joint std (normalisation):", dict(zip(G.ARM_JOINTS, sigma.round(2))))
    D["match"] = D.phase != "post_grasp"          # frames that may be matched to
    M = D[D.match].reset_index(drop=True)
    Z = M[[f"q{i}" for i in range(5)]].to_numpy(float) / sigma

    # ---- leave-one-episode-out distances and progress ------------------------------------------------
    eps = M.episode.to_numpy()
    dist = np.full(len(M), np.nan); prog_est = np.full(len(M), np.nan)
    for ep in np.unique(eps):
        tr = eps != ep
        tree = cKDTree(Z[tr]); d, i = tree.query(Z[~tr])
        dist[~tr] = d; prog_est[~tr] = M.progress.to_numpy()[tr][i]
    M["loo_dist"], M["progress_est"] = dist, prog_est
    # sensitivity: every frame (including post-grasp) against every frame of the other demos
    Zall = D[[f"q{i}" for i in range(5)]].to_numpy(float) / sigma
    dall = np.full(len(D), np.nan)
    for ep in D.episode.unique():
        m = (D.episode == ep).to_numpy(); d, _ = cKDTree(Zall[~m]).query(Zall[m]); dall[m] = d
    D["loo_dist_all"] = dall

    moving = (M.phase == "approach").to_numpy()
    variants = {
        "V1 all pre-grasp frames (incl. idle start) vs pre-grasp frames  [all pre-grasp, sensitivity]": dist,
        "V2 moving pre-grasp frames only vs pre-grasp frames  [DEFAULT]": dist[moving],
        "V3 all frames incl. post-grasp vs all frames": dall,
    }
    rows = []
    for k, v in variants.items():
        rows.append(dict(variant=k, n=len(v), p50=np.percentile(v, 50), p90=np.percentile(v, 90),
                         p95=np.percentile(v, 95), p99=np.percentile(v, 99), max=v.max()))
    TV = pd.DataFrame(rows); TV.to_csv(OUT / "step1_tau_variants.csv", index=False, float_format="%.4f")
    P("\nleave-one-episode-out nearest-neighbour distance (normalised units):\n" + TV.round(3).to_string(index=False))
    dmov = dist[moving]
    tau90, tau, tau99 = (float(np.percentile(dmov, q)) for q in (90, 95, 99))      # DEFAULT: moving frames only (user decision)
    tau_v1 = float(np.percentile(dist, 95))                                            # sensitivity: all pre-grasp frames
    P(f"\ntau (default: 95th percentile, moving pre-grasp frames) = {tau:.4f}; 90th = {tau90:.4f}; 99th = {tau99:.4f}; "
      f"sensitivity tau_V1 (all pre-grasp frames, 95th) = {tau_v1:.4f}")
    # distance in joint units for orientation
    P("tau expressed per joint (tau * sigma, one joint alone): ", dict(zip(G.ARM_JOINTS, (tau * sigma).round(1))))
    # fraction of frames beyond tau, by phase
    P("share of frames above tau by phase:", (M.assign(a=M.loo_dist > tau).groupby("phase").a.mean().round(3)).to_dict())
    P("per-episode share of frames above tau: max", round(float(M.assign(a=M.loo_dist > tau).groupby("episode").a.mean().max()), 3))

    # ---- progress reliability ----------------------------------------------------------------------
    err = (M.progress_est - M.progress).abs()
    mae_all, mae_mov = float(err.mean()), float(err[moving].mean())
    P(f"\nprogress estimate (leave-one-episode-out): MAE all pre-grasp frames = {mae_all:.3f}; moving frames only = {mae_mov:.3f}")
    P(f"  median abs error moving = {float(err[moving].median()):.3f}; 90th pct = {float(np.percentile(err[moving], 90)):.3f}")
    bins = pd.cut(M.progress[moving], [0, .2, .4, .6, .8, 1.0], include_lowest=True)
    byb = err[moving].groupby(bins, observed=True).agg(["mean", "size"]).round(3)
    P("MAE by true-progress bin (moving frames):\n" + byb.to_string())
    # in terms of time: error in seconds of the approach phase (median approach duration)
    dur = (ep_tab.grasp_frame - ep_tab.onset).median() / 30
    P(f"median approach duration = {dur:.2f} s -> MAE {mae_mov * dur:.2f} s equivalent")
    # signed bias and decision accuracy for the two p values
    for p in (P_SHALLOW, P_DEEP):
        tp = M.progress[moving] <= p; ep_ = M.progress_est[moving] <= p
        P(f"classification 'progress <= {p}' on moving frames: agreement {(tp == ep_).mean():.3f}; "
          f"true<=p but est>p {((tp) & (~ep_)).mean():.3f}; true>p but est<=p {((~tp) & (ep_)).mean():.3f}")
    M.drop(columns=["match"]).to_csv(OUT / "step1_expert_loo.csv", index=False, float_format="%.4f")

    # ---- gripper threshold ---------------------------------------------------------------------------
    g_pre = M.gripper.to_numpy(); g_all = D.gripper.to_numpy()
    mid = ((g_all > 8) & (g_all < 44)).mean()
    P(f"\ngripper reading: closed mode median {np.median(g_all[g_all < 8]):.1f}; open plateau (>=44) median "
      f"{np.median(g_all[g_all >= 44]):.1f}; frames between 8 and 44 (transitions) = {mid:.1%} of all, "
      f"{((g_pre > 8) & (g_pre < 44)).mean():.1%} of pre-grasp frames")
    P(f"rewind open threshold = {GRIPPER_OPEN_THRESHOLD:g} (open plateau starts ~44); grasp detection keeps {GRASP_DETECTION_OPEN:g}; "
      f"frames in 30-40 band (pre-grasp): {((g_pre >= 30) & (g_pre < 40)).mean():.1%}")

    # ---- save ------------------------------------------------------------------------------------------
    np.savez(OUT / "rewind_index.npz", sigma=sigma, states_norm=Z, progress=M.progress.to_numpy(),
             episode=M.episode.to_numpy(), frame=M.frame.to_numpy(), states_raw=M[[f"q{i}" for i in range(5)]].to_numpy(float),
             gripper=M.gripper.to_numpy(), phase=M.phase.to_numpy().astype(str),
             # post-grasp frames are kept for reference only; they are never matched
             post_episode=D[D.phase == "post_grasp"].episode.to_numpy(), post_frame=D[D.phase == "post_grasp"].frame.to_numpy(),
             post_states_raw=D[D.phase == "post_grasp"][[f"q{i}" for i in range(5)]].to_numpy(float),
             post_gripper=D[D.phase == "post_grasp"].gripper.to_numpy())
    params = dict(tau=tau, tau_p90=tau90, tau_p99=tau99, tau_sensitivity=tau_v1,
                  tau_definition="95th percentile of leave-one-episode-out nearest-neighbour distance over the MOVING pre-grasp frames "
                  "(arm motion onset .. first close) of the 50 expert demos, matched against pre-grasp frames of the other 49 demos; "
                  "5 arm joints divided by their expert std. tau_sensitivity = same percentile over all pre-grasp frames incl. idle start.",
                  gripper_open_threshold=GRIPPER_OPEN_THRESHOLD, gripper_open_threshold_grasp_detection=GRASP_DETECTION_OPEN,
                  p_shallow=P_SHALLOW, p_deep=P_DEEP, shallow_lead_s=1.0,
                  deep_min_gap_s=DEEP_MIN_GAP_S, deep_fallback_progress=DEEP_FALLBACK_PROGRESS,
                  start_pose=[float(x) for x in np.median(M[M.phase == "pre_motion"][[f"q{i}" for i in range(5)]].to_numpy(float), axis=0)],
                  start_pose_definition="per-joint median of the 5 arm joints over the idle frames before the arm-motion onset, all 50 expert demos",
                  units_note="not read from the recording configuration; inferred: LeRobot SOFollowerConfig.use_degrees default True "
                             "(arm joints in degrees, gripper 0-100)",
                  fps=30, arm_joints=G.ARM_JOINTS, sigma=sigma.tolist(),
                  onset=dict(speed=ONSET_SPEED, hold_frames=ONSET_HOLD, back_speed=ONSET_BACK_SPEED),
                  progress_mae_moving=mae_mov, n_expert_frames_indexed=int(len(M)))
    (OUT / "rewind_params.json").write_text(json.dumps(params, indent=2))

    # ---- figures (English) -----------------------------------------------------------------------------
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 180, "savefig.bbox": "tight",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ax = axs[0]
    bins_ = np.linspace(0, np.percentile(dist, 99.9), 80)
    ax.hist(dist, bins=bins_, color="#2a78d6", alpha=0.85, label="all pre-grasp frames (V1)")
    ax.hist(dist[moving], bins=bins_, histtype="step", color="#eb6834", lw=1.5, label="moving frames only (V2)")
    ytop = ax.get_ylim()[1]
    for v, lab, st, dy in ((tau90, "90th", ":", 0.9), (tau, "95th = tau", "-", 0.35), (tau99, "99th", "--", 0.9)):
        ax.axvline(v, color="#0b0b0b", ls=st, lw=1.2); ax.text(v + 0.01, ytop * dy, f"{lab}: {v:.3f}", fontsize=8, va="top")
    ax.set_yscale("log"); ax.set_xlabel("Leave-one-episode-out nearest-neighbour distance (normalised units)")
    ax.set_ylabel("Frames"); ax.set_title("Distance of expert frames to the other 49 demos"); ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    ax = axs[1]
    for ph, c in (("pre_motion", "#8a8984"), ("approach", "#2a78d6")):
        m = (M.phase == ph).to_numpy(); ax.scatter(M.progress[m] + 0 * dist[m], dist[m], s=2, color=c, alpha=0.35, label=ph.replace("_", " "))
    ax.axhline(tau, color="#0b0b0b", lw=1.2); ax.set_yscale("log"); ax.set_xlabel("True progress"); ax.set_ylabel("NN distance")
    ax.set_title("Distance by progress (line = tau)"); ax.legend(frameon=False, markerscale=4)
    fig.savefig(OUT / "step1_fig1_distance_distribution.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axs[0]
    mm = M[moving]
    h = ax.hist2d(mm.progress, mm.progress_est, bins=40, cmap="Blues", norm=matplotlib.colors.LogNorm())
    ax.plot([0, 1], [0, 1], color="#eb6834", lw=1.2); plt.colorbar(h[3], ax=ax, label="frames")
    ax.set_xlabel("True progress"); ax.set_ylabel("Estimated progress")
    ax.set_title(f"Leave-one-episode-out estimate (moving frames), MAE = {mae_mov:.3f}", fontsize=10)
    ax = axs[1]
    for ep in (0, 12, 24, 36, 48):
        e = M[(M.episode == ep) & moving]; ax.plot(e.progress, e.progress_est, lw=1, label=f"episode {ep}")
    ax.plot([0, 1], [0, 1], color="#0b0b0b", lw=1, ls="--"); ax.set_xlabel("True progress"); ax.set_ylabel("Estimated progress")
    ax.set_title("Five example episodes"); ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "step1_fig2_progress_scatter.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, g, ttl in ((axs[0], g_all, "all frames of the 50 demos"), (axs[1], g_pre, "pre-grasp frames only")):
        ax.hist(g, bins=np.arange(0, 92, 2), color="#6250d6")
        for v, c, lab in ((GRASP_DETECTION_OPEN, "#8a8984", "grasp detection"), (GRIPPER_OPEN_THRESHOLD, "#eb6834", "rewind open")):
            ax.axvline(v, color=c, lw=1.5); ax.text(v + 1, ax.get_ylim()[1] * (0.9 if v == GRIPPER_OPEN_THRESHOLD else 0.5), f"{lab} = {v:g}", color=c)
        ax.set_yscale("log"); ax.set_xlabel("Gripper reading (state, 0-100)"); ax.set_ylabel("Frames"); ax.set_title(ttl)
    fig.savefig(OUT / "step1_fig3_gripper_distribution.png"); plt.close(fig)
    P("\nsaved to", OUT)

if __name__ == "__main__":
    main()

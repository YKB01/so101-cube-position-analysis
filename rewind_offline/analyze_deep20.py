#!/usr/bin/env python
"""
Descriptive summary of the deep-rewind collection (protocol/rewind_collect_deep.md, section 9) and a check of the idea "go straight to the rewind target instead of replaying the path backwards".
Reads results/rewind_collect/run_20261119_120625 (rewind_meta.csv, episodes/*/autonomous_states.npz), the datasets YKB0123/rewind_deep_20 and *_rewind_seg, and the 10/15 / 11/12 / 7/31 reference datasets.
Writes rewind_offline/deep20_attempts.csv, deep20_vs_reference.csv, deep20_summary.txt, fig_deep20_path.png, fig_deep20_cube.png
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analyze_cube_positions as A, analyze_grasp_points as G
import rewind_target as RT
from lerobot.datasets import LeRobotDataset

OUT = HERE / "rewind_offline"; RUN = HERE / "results/rewind_collect/run_20261119_120625"; ROOT = Path.home() / ".cache/huggingface/lerobot/YKB0123"
P = json.loads((OUT / "rewind_params.json").read_text())
LOG = open(OUT / "deep20_summary.txt", "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n")
plt.rcParams.update({"font.size": 9.5, "savefig.dpi": 160, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})

m = pd.read_csv(RUN / "rewind_meta.csv")
saved = m[~m.discarded]
# ------------------------------------------------------------------ path: replay versus direct
rows = []
for r in m.itertuples():
    S = np.load(RUN / f"episodes/ep_{r.episode_id:04d}/autonomous_states.npz")["state"]
    ti = int(r.trigger_frame); buf = S[max(0, ti - 450): ti + 1]; bi = len(buf) - 1
    idx = None if pd.isna(r.target_buffer_index) else int(r.target_buffer_index)
    wp = RT.plan_rewind_path(buf, bi, idx, P, resample=True)
    D = float(np.linalg.norm(wp[-1] - wp[0])); L = float(np.linalg.norm(np.diff(wp, axis=0), axis=1).sum())
    a, b = wp[0], wp[-1]; t = np.clip(((wp - a) @ (b - a)) / max((b - a) @ (b - a), 1e-9), 0, 1)
    rows.append(dict(episode_id=r.episode_id, direct_dist_deg=D, replay_len_deg=L, len_ratio=L / D, dev_from_line_max_deg=float(np.linalg.norm(wp - (a + t[:, None] * (b - a)), axis=1).max()),
                     direct_s=int(np.ceil(np.abs(b - a).max() / 1.5)) / 30, replay_planned_s=RT.estimate_rewind_steps(wp, 1.5) / 30, n_waypoints=len(wp)))
Pa = pd.DataFrame(rows); att = m.merge(Pa, on="episode_id")
# ------------------------------------------------------------------ cube movement during the rewind (segment videos, fully visible detections only)
nominal = float(pd.read_csv(HERE / "cube_positions.csv").query("dataset == 'expert' and detected").area.median())
seg = LeRobotDataset("YKB0123/rewind_deep_20_rewind_seg", root=ROOT / "rewind_deep_20_rewind_seg"); frm, to = seg.meta.episodes["dataset_from_index"], seg.meta.episodes["dataset_to_index"]
cube = []
for k, r in enumerate(m.itertuples()):
    pts = []
    for i in range(int(frm[k]), int(to[k])):
        im = np.asarray(seg[i]["observation.images.front"].permute(1, 2, 0).clip(0, 1) * 255, dtype=np.uint8); d = A.detect_cube(im)
        pts.append((d["cx"], d["cy"]) if (d["detected"] and d["area"] >= 0.8 * nominal and d["n_candidates"] == 1) else (np.nan, np.nan))
    Q = np.array(pts); ok = ~np.isnan(Q[:, 0]); idx = np.flatnonzero(ok)
    acts = np.stack(seg.hf_dataset.select(range(int(frm[k]), int(to[k])))["action"]) if False else None
    row = dict(episode_id=r.episode_id, n_seg_frames=len(Q), n_full_visible=int(ok.sum()))
    if len(idx) >= 2:
        ref = Q[idx[0]]; dev = np.hypot(Q[idx, 0] - ref[0], Q[idx, 1] - ref[1])
        row.update(start_to_first_px=float(np.hypot(*(ref - np.array([r.cube_start_x, r.cube_start_y])))), first_to_last_px=float(dev[-1]), max_shift_in_segment_px=float(dev.max()),
                   first_move_frame=int(idx[np.flatnonzero(dev > 3)[0]]) if (dev > 3).any() else -1)
    cube.append(row)
att = att.merge(pd.DataFrame(cube), on="episode_id"); att.to_csv(OUT / "deep20_attempts.csv", index=False, float_format="%.3f")

pr(f"collection run {RUN.name}: {len(m)} attempts, {len(saved)} saved, {int(m.discarded.sum())} discarded; {m.timestamp.min()} .. {m.timestamp.max()}")
pr(f"triggers: {m.trigger_type.value_counts().to_dict()}; trigger time median {m.trigger_time_s.median():.1f} s (range {m.trigger_time_s.min():.1f}-{m.trigger_time_s.max():.1f})")
pr(f"rewind level (saved): {saved.level.value_counts().to_dict()}; (all attempts): {m.level.value_counts().to_dict()}")
pr(f"queue remaining at the trigger: median {m.queue_remaining_at_trigger.median():.0f} (range {m.queue_remaining_at_trigger.min():.0f}-{m.queue_remaining_at_trigger.max():.0f}); after clear {m.queue_len_after_clear.unique().tolist()}; at start {m.queue_len_at_start.unique().tolist()}")
pr(f"target {(m.trigger_time_s - m.target_time_s).median():.1f} s before the trigger (range {(m.trigger_time_s - m.target_time_s).min():.1f}-{(m.trigger_time_s - m.target_time_s).max():.1f}), progress median {m.target_progress.median():.2f}, nn distance max {m.target_nn_distance.max():.2f} (tau {P['tau']:.3f})")
pr(f"rewind: planned {m.rewind_planned_s.median():.1f} s, actual {m.rewind_actual_s.median():.1f} s (max {m.rewind_actual_s.max():.1f}); joint distance {m.joint_distance_deg.median():.0f} deg ({m.joint_distance_deg.min():.0f}-{m.joint_distance_deg.max():.0f}); final error {m.final_err_max_deg.median():.1f} deg (max {m.final_err_max_deg.max():.1f}), {m.final_err_norm_over_tau.max():.2f} of tau at most")
pr(f"align: error {m.align_err_max_deg.median():.2f} deg (max {m.align_err_max_deg.max():.2f}), retries {m.align_retries.value_counts().to_dict()}; human correction {saved.human_s.median():.1f} s (range {saved.human_s.min():.1f}-{saved.human_s.max():.1f})")
pr(f"loop period: autonomous mean {m.autonomous_dt_mean_ms.mean():.2f} ms, max over episodes {m.autonomous_dt_max_ms.max():.0f} ms; human mean {m.human_dt_mean_ms.mean():.2f} ms, max {m.human_dt_max_ms.max():.0f} ms")
pr(f"lighting: first-frame table gray median {m.first_frame_table_gray.median():.1f} (range {m.first_frame_table_gray.min():.1f}-{m.first_frame_table_gray.max():.1f}); bright state (>= 117.5) in {int((m.first_frame_table_gray >= 117.5).sum())} of {len(m)} attempts; camera auto exposure {m.cam_auto_exposure.unique().tolist()}, white balance auto {m.cam_white_balance_automatic.unique().tolist()}")
pr(f"placement distance {saved.placement_dist_px.median():.1f} px (max {m.placement_dist_px.max():.1f}); inside the window: {int((~m.outside_time_window).sum())} of {len(m)}")
pr("\n--- replay versus direct move (all 30 attempts) ---")
pr(f"direct joint-space distance trigger -> target: median {att.direct_dist_deg.median():.0f} deg; replayed path length (points < 2 deg merged): median {att.replay_len_deg.median():.0f} deg = {att.len_ratio.median():.2f}x (max {att.len_ratio.max():.1f}x, episode {int(att.loc[att.len_ratio.idxmax(), 'episode_id'])}); "
   f"largest departure of the replayed path from the straight joint-space line: median {att.dev_from_line_max_deg.median():.0f} deg (max {att.dev_from_line_max_deg.max():.0f})")
pr(f"time at 1.5 deg per joint per tick: direct {att.direct_s.median():.1f} s, replay planned {att.replay_planned_s.median():.1f} s, replay actual {att.rewind_actual_s.median():.1f} s")
v = att[att.n_full_visible >= 2]
pr(f"\n--- cube movement (segment videos; fully visible detections only; repeat detections of an unmoved cube agree within ~1 px) ---")
pr(f"evaluable segments: {len(v)} of 30; cube moved > 3 px between the start of the episode and the first visible frame of the segment (policy phase): {int((v.start_to_first_px > 3).sum())}; "
   f"moved > 3 px inside the rewind segment: {int((v.max_shift_in_segment_px > 3).sum())} (episodes {v[v.max_shift_in_segment_px > 3].episode_id.tolist()}), of which first movement after the rewind motion started (frame >= 32): "
   f"{v[(v.max_shift_in_segment_px > 3) & (v.first_move_frame >= 32)].episode_id.tolist()}, during the gripper opening (frame < 32): {v[(v.max_shift_in_segment_px > 3) & (v.first_move_frame < 32) & (v.first_move_frame >= 0)].episode_id.tolist()}")
# ------------------------------------------------------------------ comparison with the earlier data
def stats(repo, root, label):
    df = LeRobotDataset("YKB0123/" + Path(root).name, root=root).hf_dataset.to_pandas(); rows = []; sp0 = np.array(P["start_pose"])
    for ep, e in df.groupby("episode_index"):
        e = e.sort_values("frame_index"); S = np.stack(e["observation.state"]); q = S[:, :5]; gf = G.detect_first_grasp(S[:, 5])["grasp_frame"]
        qs = np.stack([np.convolve(q[:, j], np.ones(5) / 5, mode="same") for j in range(5)], 1); sp = np.r_[0, np.linalg.norm(np.diff(q, axis=0), axis=1) * 30]
        rows.append(dict(frames=len(e), first_frame_dist_from_start=np.linalg.norm(q[0] - sp0), time_to_first_grasp_s=gf / 30, mean_speed=sp.mean(), pre_grasp_speed=sp[:int(gf)].mean(), jerk=np.abs(np.diff(qs, 3, axis=0)).mean() * 30 ** 3))
    r = pd.DataFrame(rows).median(); r["label"] = label; r["n_episodes"] = len(rows); return r
ref = pd.DataFrame([stats(None, ROOT / "rewind_deep_20", "new corrections after deep rewind (20)"), stats(None, ROOT / "rollout_so101_red_cube_hil_front_20261015_134746", "10/15 DAgger corrections (20)"),
                    stats(None, ROOT / "so101_red_cube_plastic_box_expert_20_20261112_130439", "11/12 new expert (20)"), stats(None, ROOT / "so101_red_cube_plastic_box_front_50_20260731_104214", "7/31 expert (50)")]).set_index("label")
ref.to_csv(OUT / "deep20_vs_reference.csv", float_format="%.2f"); pr("\n" + ref.round(1).to_string())
# ------------------------------------------------------------------ figures
fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8)); x = np.arange(len(att))
ax = axs[0]; ax.scatter(att.direct_dist_deg, att.replay_len_deg, s=22, color="#2a78d6", lw=0); lim = [0, max(att.direct_dist_deg.max(), att.replay_len_deg.max()) * 1.05]; ax.plot(lim, lim, color="#8a8984", lw=1, ls="--")
ax.set_xlabel("Direct joint-space distance trigger -> target (deg)"); ax.set_ylabel("Replayed path length (deg)"); ax.set_title("Path length: replay vs direct", fontsize=10)
ax = axs[1]; ax.scatter(att.direct_s, att.rewind_actual_s, s=22, color="#eb6834", lw=0, label="replay, actual"); ax.scatter(att.direct_s, att.replay_planned_s, s=22, color="#1baf7a", lw=0, label="replay, planned")
l2 = [0, max(att.rewind_actual_s.max(), att.direct_s.max()) * 1.05]; ax.plot(l2, l2, color="#8a8984", lw=1, ls="--"); ax.set_xlabel("Direct move at 1.5 deg/tick (s)"); ax.set_ylabel("Replay (s)"); ax.legend(frameon=False, fontsize=8); ax.set_title("Time", fontsize=10)
ax = axs[2]; ax.hist(att.dev_from_line_max_deg, bins=12, color="#6250d6"); ax.set_xlabel("Largest departure of the replay from the straight joint line (deg)"); ax.set_ylabel("Attempts"); ax.set_title("How far the replay strays", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_deep20_path.png"); plt.close(fig)

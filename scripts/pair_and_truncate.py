#!/usr/bin/env python
"""
Task B steps 1-2 (planning only, nothing is written to any dataset). Two truncation rules are evaluated:
  ORIGINAL (infeasible, kept for the record): distance-from-start-pose rule described below.
  PATH-ALIGNED (used): keep the part of each new expert episode whose arm path (5 arm joints, observation.state) up to the
  first grasp matches the paired recovery episode's path from its first frame to its first grasp (cut frame = the frame
  whose remaining path is closest to the recovery path), plus everything after the grasp. -> trunc_control/truncation_plan.csv

Original rule:
 1. pair the 20 new expert episodes with the 20 recovery episodes by cube position (optimal assignment, Euclidean px)
 2. for each pair, find the first frame of the new expert episode whose arm distance (5 arm joints, observation.state)
    from the expert start pose is >= the first-frame distance of the paired recovery episode; keep that frame to the end.
Outputs: trunc_control/pairing_and_truncation.csv, trunc_control/truncation_report.md (numbers), prints a summary.
Start pose = mean of the first-frame states of the 50 original expert episodes (same as data_comparison).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import analyze_cube_positions as A  # noqa: E402
import analyze_grasp_points as G  # noqa: E402

OUT = HERE / "trunc_control"
NEW_DIR = "so101_red_cube_plastic_box_expert_20_20261112_130439"
FPS = 30


def main():
    OUT.mkdir(exist_ok=True)
    A.DATASETS["new_expert"] = NEW_DIR
    rec_pos = pd.read_csv(HERE / "recovery_position_check/recovery_positions_resolved.csv")[["episode_index", "cx", "cy"]]
    new_pos = pd.read_csv(HERE / "new_expert_position_check/new_expert_positions.csv")[["episode_index", "cx", "cy"]]
    D = np.hypot(new_pos.cx.to_numpy()[:, None] - rec_pos.cx.to_numpy()[None], new_pos.cy.to_numpy()[:, None] - rec_pos.cy.to_numpy()[None])
    ri, ci = linear_sum_assignment(D)
    pair = pd.DataFrame(dict(new_episode=new_pos.episode_index.to_numpy()[ri], recovery_episode=rec_pos.episode_index.to_numpy()[ci],
                             pair_dist_px=D[ri, ci]))
    idx_dist = np.hypot(new_pos.cx - rec_pos.cx, new_pos.cy - rec_pos.cy)       # naive pairing by episode index
    print(f"pairing distance (optimal): mean {pair.pair_dist_px.mean():.1f}, median {pair.pair_dist_px.median():.1f}, "
          f"max {pair.pair_dist_px.max():.1f} px | by episode index: mean {idx_dist.mean():.1f}, max {idx_dist.max():.1f} px")

    fr = {k: G.load_frames(k)[0] for k in ("expert", "recovery", "new_expert")}
    J = [f"state:{j}" for j in G.ARM_JOINTS]
    start = fr["expert"].sort_values(["episode_index", "frame_index"]).groupby("episode_index").first()[J].to_numpy(float).mean(0)
    rec = {ep: e.sort_values("frame_index") for ep, e in fr["recovery"].groupby("episode_index")}
    new = {ep: e.sort_values("frame_index") for ep, e in fr["new_expert"].groupby("episode_index")}
    rows = []
    for r in pair.itertuples():
        rq = rec[r.recovery_episode][J].to_numpy(float)
        thr = float(np.linalg.norm(rq[0] - start))
        e = new[r.new_episode]
        q = e[J].to_numpy(float); d = np.linalg.norm(q - start, axis=1)
        hit = np.flatnonzero(d >= thr)
        t = int(hit[0]) if len(hit) else -1
        row = dict(new_episode=r.new_episode, recovery_episode=r.recovery_episode, pair_dist_px=r.pair_dist_px,
                   recovery_first_dist=thr, new_len=len(e), recovery_len=len(rq), max_dist_new=d.max(), cut_frame=t)
        if t >= 0:
            kept = q[t:]
            gf = G.detect_first_grasp(e[f"state:{G.GRIPPER}"].to_numpy(float))["grasp_frame"]
            row.update(kept_len=len(kept), first_dist_after_cut=float(d[t]), orig_grasp_frame=gf,
                       grasp_in_kept=(gf >= t) if not np.isnan(gf) else False,
                       path_to_grasp_after_cut=float(np.linalg.norm(np.diff(q[t:int(gf) + 1], axis=0), axis=1).sum()) if (not np.isnan(gf) and gf >= t) else np.nan,
                       time_to_grasp_after_cut_s=(gf - t) / FPS if (not np.isnan(gf) and gf >= t) else np.nan)
        # PATH-ALIGNED rule
        gfn = G.detect_first_grasp(e[f"state:{G.GRIPPER}"].to_numpy(float))["grasp_frame"]
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        rg_ = G.detect_first_grasp(rec[r.recovery_episode][f"state:{G.GRIPPER}"].to_numpy(float))["grasp_frame"]
        rec_path = float(np.linalg.norm(np.diff(rq[:int(rg_) + 1], axis=0), axis=1).sum())
        gfn = int(gfn)
        rem = cum[gfn] - cum[: gfn + 1]                        # remaining path to the grasp if the episode starts at frame t
        tp = int(np.argmin(np.abs(rem - rec_path)))
        row.update(path_cut_frame=tp, path_kept_len=len(q) - tp, path_new_grasp_frame=gfn, path_new_path_to_grasp=float(rem[tp]),
                   path_target=rec_path, path_time_to_grasp_s=(gfn - tp) / FPS, path_first_frame_dist=float(d[tp]),
                   path_total_new_path_to_grasp=float(cum[gfn]))
        # recovery reference
        rg = G.detect_first_grasp(rec[r.recovery_episode][f"state:{G.GRIPPER}"].to_numpy(float))["grasp_frame"]
        row.update(recovery_grasp_frame=rg, recovery_path_to_grasp=float(np.linalg.norm(np.diff(rq[:int(rg) + 1], axis=0), axis=1).sum()),
                   recovery_time_to_grasp_s=rg / FPS)
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "pairing_and_truncation.csv", index=False, float_format="%.3f")
    ok = T[T.cut_frame >= 0]
    tot, rtot = int(ok.kept_len.sum()), int(T.recovery_len.sum())
    print(f"episodes where the distance threshold is reached: {len(ok)}/20")
    print(f"total frames: truncated {tot} vs recovery {rtot}  ({100 * (tot - rtot) / rtot:+.1f}%)")
    print(f"kept length per episode: median {ok.kept_len.median():.0f} [{ok.kept_len.quantile(.25):.0f}-{ok.kept_len.quantile(.75):.0f}] vs recovery median {T.recovery_len.median():.0f}")
    print(f"first-frame distance after cut (threshold = paired recovery): median {ok.first_dist_after_cut.median():.1f}, recovery median {T.recovery_first_dist.median():.1f}")
    print(f"cut frame: median {ok.cut_frame.median():.0f} ({ok.cut_frame.median() / FPS:.1f} s) range {ok.cut_frame.min()}-{ok.cut_frame.max()}")
    print(f"grasp inside kept part: {int(ok.grasp_in_kept.sum())}/{len(ok)}")
    g = ok[ok.grasp_in_kept]
    print(f"path to grasp after cut: median {g.path_to_grasp_after_cut.median():.1f} (recovery paired median {g.recovery_path_to_grasp.median():.1f}, all recovery {T.recovery_path_to_grasp.median():.1f})")
    print(f"time to grasp after cut: median {g.time_to_grasp_after_cut_s.median():.2f} s (recovery {T.recovery_time_to_grasp_s.median():.2f} s)")
    T[["new_episode", "recovery_episode", "pair_dist_px", "path_cut_frame", "path_kept_len", "path_new_grasp_frame", "path_target",
       "path_new_path_to_grasp", "path_time_to_grasp_s", "path_first_frame_dist", "recovery_len", "recovery_time_to_grasp_s"]].to_csv(
        OUT / "truncation_plan.csv", index=False, float_format="%.3f")
    print(f"PATH-ALIGNED: total kept frames {int(T.path_kept_len.sum())} vs recovery {rtot} ({100 * (T.path_kept_len.sum() - rtot) / rtot:+.1f}%)")
    print(T[["new_episode", "recovery_episode", "pair_dist_px", "recovery_first_dist", "max_dist_new", "cut_frame", "kept_len", "recovery_len", "path_to_grasp_after_cut", "recovery_path_to_grasp"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()

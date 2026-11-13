#!/usr/bin/env python
"""
Resolve the cube position of recovery (HIL) episodes whose first frame does not show the cube
(the arm hides it at the moment of intervention).

Rule: before the first grasp of the episode (first gripper close after the gripper is open, same
detector as analyze_grasp_points.py), take the first frame from which the cube is detected for
STILL_FRAMES consecutive frames with a normal area/aspect and a position that stays within STILL_PX
of the window median; the position is the median over that window. Episodes without such a window
are reported as unresolved. The rule is applied to all recovery episodes, so the ones whose first
frame already shows the cube serve as a check of the rule.

Note: the result is the cube position when it first lies visible on the table during the recovery
episode. The original placement happened during the policy rollout before the intervention, which
is not part of this dataset.

Inputs (read only): recovery dataset, cube_positions.csv, test_positions.csv, and optionally
new_expert_position_check/new_expert_positions.csv. Outputs: ./recovery_position_check/.
"""
from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_cube_positions as A  # noqa: E402
import analyze_grasp_points as G  # noqa: E402

KEY = "recovery"
OUT = HERE / "recovery_position_check"
STILL_FRAMES = 10   # consecutive frames the cube must be visible and still
STILL_PX = 2.0      # max distance of each window frame from the window median (px)
NEW_EXPERT_CSV = HERE / "new_expert_position_check" / "new_expert_positions.csv"
C_NEW = "#6250d6"
log = logging.getLogger("recovery_pos")


def episode_frames(ds_dir: Path, row):
    """Sequentially decode all frames of one episode from the concatenated v3.0 video file."""
    vp = A.video_path(ds_dir, int(row.v_chunk), int(row.v_file))
    with av.open(str(vp)) as c:
        s = c.streams.video[0]
        c.seek(int(row.v_from_ts / s.time_base), stream=s, backward=True)
        n = 0
        for fr in c.decode(s):
            if float(fr.pts * s.time_base) < row.v_from_ts - 1e-3:
                continue
            if n >= row.length:
                break
            yield n, fr.to_ndarray(format="rgb24")
            n += 1


def first_still_window(seq: pd.DataFrame, end: int) -> int | None:
    """First frame t < end such that frames t..t+STILL_FRAMES-1 are valid and still."""
    v = seq.valid.to_numpy()
    xy = seq[["cx", "cy"]].to_numpy(float)
    for t in range(0, max(0, min(end, len(seq)) - STILL_FRAMES + 1)):
        if not v[t:t + STILL_FRAMES].all():
            continue
        w = xy[t:t + STILL_FRAMES]
        if np.hypot(*(w - np.median(w, axis=0)).T).max() <= STILL_PX:
            return t
    return None


def main():
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "run.log", mode="w")])
    D = A.LEROBOT_ROOT / A.DATASETS[KEY]
    meta = A.load_episode_meta(D)
    frames_df, _ = G.load_frames(KEY)
    ref = pd.read_csv(HERE / "cube_positions.csv")
    ref_area = ref[(ref.dataset == "expert") & ref.detected].area.median()
    first = ref[ref.dataset == KEY].set_index("episode_index")

    rows, tiles = [], []
    for row in meta.itertuples():
        ep = int(row.episode_index)
        e = frames_df[frames_df.episode_index == ep]
        g = e[f"state:{G.GRIPPER}"].to_numpy(float)
        grasp = G.detect_first_grasp(g)
        end = int(grasp["grasp_frame"]) if not np.isnan(grasp["grasp_frame"]) else len(g)
        seq, imgs = [], {}
        for i, rgb in episode_frames(D, row):
            if i >= end:
                break
            d = A.detect_cube(rgb)
            ok = bool(d["detected"]) and A.AREA_RATIO_RANGE[0] <= d["area"] / ref_area <= A.AREA_RATIO_RANGE[1] \
                and d["aspect"] <= A.MAX_ASPECT
            seq.append(dict(f=i, valid=ok, cx=d["cx"], cy=d["cy"], area=d["area"]))
            if i == 0 or ok:
                imgs[i] = (rgb, d)
        seq = pd.DataFrame(seq)
        t = first_still_window(seq, end)
        f0_det = bool(first.loc[ep, "detected"])
        res = dict(episode_index=ep, first_frame_detected=f0_det,
                   first_frame_cx=first.loc[ep, "cx"], first_frame_cy=first.loc[ep, "cy"],
                   grasp_frame=grasp["grasp_frame"], search_end=end)
        if t is None:
            res.update(cx=np.nan, cy=np.nan, area=np.nan, frame_index=np.nan, time_s=np.nan, gripper=np.nan,
                       method="unresolved")
            log.warning("episode %d: no still, visible cube before the first grasp (frame %s)", ep, end)
        else:
            w = seq.iloc[t:t + STILL_FRAMES]
            res.update(cx=w.cx.median(), cy=w.cy.median(), area=w.area.median(), frame_index=t,
                       time_s=t / 30.0, gripper=float(g[t]),
                       method="first_frame" if f0_det else "first_still_before_grasp")
        rows.append(res)
        # contact sheet: frame 0 (context) + chosen frame
        rgb0, d0 = imgs[0]
        tiles.append(A.make_tile(rgb0, d0, f"ep {ep} frame 0" + ("" if d0["detected"] else " (cube hidden)")))
        if t is not None:
            rgb, d = imgs[t]
            tiles.append(A.make_tile(rgb, d, f"ep {ep} f{t} ({t / 30:.1f}s) -> ({res['cx']:.0f},{res['cy']:.0f}) "
                                             f"grip {g[t]:.0f}"))
        else:
            tiles.append(np.full_like(tiles[-1], 255))

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "recovery_positions_resolved.csv", index=False, float_format="%.2f")
    # contact sheet: hidden episodes first (they need checking), then the rule check on the others
    order = list(r[~r.first_frame_detected].index) + list(r[r.first_frame_detected].index)
    tiles2 = [tiles[2 * i + k] for i in order for k in (0, 1)]
    A.save_contact_sheet(tiles2[: 2 * (~r.first_frame_detected).sum()],
                         "Recovery episodes with the cube hidden in frame 0: left = frame 0, right = chosen frame "
                         f"(first {STILL_FRAMES} still frames before the first grasp)",
                         OUT / "contact_recovery_hidden_resolved.png")
    A.save_contact_sheet(tiles2[2 * (~r.first_frame_detected).sum():],
                         "Rule check on recovery episodes whose frame 0 already shows the cube: left = frame 0, "
                         "right = frame chosen by the rule", OUT / "contact_recovery_rule_check.png")

    # ---- rule check on episodes visible in frame 0 ----------------------------------------------
    chk = r[r.first_frame_detected & (r.method != "unresolved")].copy()
    chk["diff_px"] = np.hypot(chk.cx - chk.first_frame_cx, chk.cy - chk.first_frame_cy)
    log.info("rule check (%d episodes visible in frame 0): chosen frame %s, position diff max %.1f px, median %.1f",
             len(chk), chk.frame_index.astype(int).tolist(), chk.diff_px.max(), chk.diff_px.median())
    hid = r[~r.first_frame_detected]
    log.info("hidden episodes resolved: %d/%d", (hid.method != "unresolved").sum(), len(hid))
    log.info("\n%s", hid[["episode_index", "frame_index", "time_s", "cx", "cy", "area", "gripper", "grasp_frame",
                          "method"]].round(1).to_string(index=False))

    # ---- distribution comparisons with the full recovery set ------------------------------------
    rec_all = r[r.method != "unresolved"].rename(columns={})
    rec_f0 = rec_all[rec_all.method == "first_frame"]
    rec_hid = rec_all[rec_all.method == "first_still_before_grasp"]
    exp = ref[(ref.dataset == "expert") & ref.detected]
    centers = pd.read_csv(HERE / "test_positions.csv")
    new = pd.read_csv(NEW_EXPERT_CSV) if NEW_EXPERT_CSV.exists() else None
    if new is not None:
        new = new[new.detected]
        from scipy.stats import ks_2samp
        for lab, rr in (("recovery, 11 first-frame only", rec_f0), (f"recovery, all {len(rec_all)} resolved", rec_all)):
            inside = new[new.cx.between(rr.cx.min(), rr.cx.max()) & new.cy.between(rr.cy.min(), rr.cy.max())]
            log.info("new expert vs %s: KS cx p=%.3f, cy p=%.3f; inside bbox %d/%d; recovery cx %.1f±%.1f cy %.1f±%.1f",
                     lab, ks_2samp(new.cx, rr.cx).pvalue, ks_2samp(new.cy, rr.cy).pvalue, len(inside), len(new),
                     rr.cx.mean(), rr.cx.std(), rr.cy.mean(), rr.cy.std())

    def dmin(df, p):
        return np.hypot(df.cx - p.cx, df.cy - p.cy).min() if len(df) else np.nan
    dist = []
    for p in centers.itertuples():
        row = dict(position=p.position, expert50=dmin(exp, p),
                   mixed_rec11=dmin(pd.concat([exp, rec_f0]), p),
                   mixed_rec_all=dmin(pd.concat([exp, rec_all]), p))
        if new is not None:
            row["expert50_new20"] = dmin(pd.concat([exp, new]), p)
        dist.append(row)
    dist = pd.DataFrame(dist).round(1)
    dist.to_csv(OUT / "nearest_train_distance_by_test_position.csv", index=False)
    log.info("nearest training sample per test position (px):\n%s\nmean: %s", dist.to_string(index=False),
             dist.drop(columns="position").mean().round(1).to_dict())

    # ---- plot ------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = pd.concat([exp[["cx", "cy"]], rec_all[["cx", "cy"]], centers[["cx", "cy"]]]
                    + ([new[["cx", "cy"]]] if new is not None else []))
    m = 12
    A.ZOOM_BOX = (max(0, pts.cx.min() - m), max(0, pts.cy.min() - m), min(640, pts.cx.max() + m),
                  min(480, pts.cy.max() + m))
    bg = G.decode_frame(*A.BACKGROUND_FRAME)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b5b4ae", "savefig.dpi": 200, "savefig.bbox": "tight"})

    def draw(ax, zoomed):
        k = 1.0 if zoomed else 0.45
        ax.scatter(exp.cx, exp.cy, s=36 * k, c=A.C_EXPERT, edgecolors="white", linewidths=1.0 * k, zorder=4,
                   label=f"Old expert demo (n={len(exp)})")
        ax.scatter(rec_f0.cx, rec_f0.cy, s=64 * k, marker="^", c=A.C_RECOVERY, edgecolors="white",
                   linewidths=1.0 * k, zorder=5, label=f"Recovery, visible in frame 0 (n={len(rec_f0)})")
        ax.scatter(rec_hid.cx, rec_hid.cy, s=64 * k, marker="^", facecolors="white", edgecolors=A.C_RECOVERY,
                   linewidths=1.8 * k, zorder=5, label=f"Recovery, first still frame before grasp (n={len(rec_hid)})")
        if new is not None:
            ax.scatter(new.cx, new.cy, s=56 * k, marker="D", c=C_NEW, edgecolors="white", linewidths=1.0 * k,
                       zorder=5, label=f"New expert (n={len(new)})")
        A._stars(ax, centers, zoomed)
        if zoomed:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, framealpha=0.9, fontsize=8.5)

    fig, _ = A._two_panel(plt, bg, "Initial cube positions incl. recovery episodes resolved from later frames", draw)
    fig.savefig(OUT / "position_distribution_all_recovery.png")
    plt.close(fig)
    log.info("done: %s", OUT)


if __name__ == "__main__":
    main()

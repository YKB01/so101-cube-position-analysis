#!/usr/bin/env python
"""
Pre-registered selection of 3 right-side test positions (deterministic, no manual choice).

Inputs (all read from this repo, nothing re-detected):
  cube_positions.csv                                   expert 50 (first-frame detection)
  recovery_position_check/recovery_positions_resolved.csv   recovery 20 (9 resolved by first-still-before-grasp)
  new_expert_position_check/new_expert_positions.csv   new expert 20
  test_positions.csv                                   6 standard test positions
  right_side_test_analysis/edge_positions.csv          old extra positions (cluster means of the two edge sets)
Pixel coordinates: front camera 640x480, cube centre in the first frame.

Outputs: results/test_positions.json, results/test_positions.png
Usage: python scripts/select_test_positions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"

# ------------------------------- thresholds (all constants) ---------------------------------------
GRID_PX = 2.0                 # candidate grid spacing
IMG_W, IMG_H = 640, 480
D_MIN_PX, D_MAX_PX = 3.0, 10.0   # d_R and d_C must both lie in [D_MIN, D_MAX]
D_DIFF_MAX_PX = 1.5           # |d_R - d_C| <= this
MIN_SEP_PX = 15.0             # distance to the 6 standard + 2 old extra positions
TIE_ROUND_PX = 0.01           # |d_R-d_C| values equal after rounding to this are tied
# "Right side" = same definition as analyze_cube_positions.label_positions: the 3 standard positions
# with the larger x are the right column. The boundary is the midpoint between the right-most left
# position and the left-most right position (x of cluster centres). Rows (top/mid/bottom) = the
# right-column standard positions ordered by y; row boundaries are midpoints between neighbouring
# rows, and the "segment centre" used for tie-breaking is the standard position of that row.
SEGMENTS = ("top", "mid", "bottom")


def load():
    exp = pd.read_csv(ROOT / "cube_positions.csv").query("dataset=='expert' and detected")[["cx", "cy"]]
    rec = pd.read_csv(ROOT / "recovery_position_check/recovery_positions_resolved.csv")[["cx", "cy"]]
    new = pd.read_csv(ROOT / "new_expert_position_check/new_expert_positions.csv").query("detected")[["cx", "cy"]]
    std = pd.read_csv(ROOT / "test_positions.csv")[["position", "cx", "cy"]]
    edge = pd.read_csv(ROOT / "right_side_test_analysis/edge_positions.csv")
    old = edge.groupby("position")[["cx", "cy"]].mean().reset_index()
    assert (len(exp), len(rec), len(new), len(std), len(old)) == (50, 20, 20, 6, 2), (len(exp), len(rec), len(new))
    return exp, rec, new, std, old


def nearest(P, S):
    return np.sqrt(((P[:, None, :] - S[None, :, :]) ** 2).sum(-1)).min(1)


def main():
    RES.mkdir(exist_ok=True)
    exp, rec, new, std, old = load()
    R = pd.concat([exp, rec]).to_numpy(float)       # model R training positions (70)
    C = pd.concat([exp, new]).to_numpy(float)       # model C training positions (70)
    S = std.set_index("position")
    right_names = S.cx.sort_values().index[3:]
    left_names = S.cx.sort_values().index[:3]
    x_split = float((S.loc[left_names, "cx"].max() + S.loc[right_names, "cx"].min()) / 2)
    rows = S.loc[right_names].sort_values("cy")
    row_cy = rows.cy.to_numpy(); row_names = dict(zip(SEGMENTS, rows.index))
    y_edges = [(row_cy[0] + row_cy[1]) / 2, (row_cy[1] + row_cy[2]) / 2]
    seg_centre = {s: (float(rows.cx.iloc[i]), float(rows.cy.iloc[i])) for i, s in enumerate(SEGMENTS)}

    # ---- candidates -------------------------------------------------------------------------------
    xs = np.arange(np.ceil(x_split / GRID_PX) * GRID_PX, IMG_W + 1e-9, GRID_PX)
    ys = np.arange(0, IMG_H + 1e-9, GRID_PX)
    gx, gy = np.meshgrid(xs, ys)
    P = np.c_[gx.ravel(), gy.ravel()]
    dR, dC = nearest(P, R), nearest(P, C)
    inR = Delaunay(R).find_simplex(P) >= 0
    inC = Delaunay(C).find_simplex(P) >= 0
    ref = np.vstack([S[["cx", "cy"]].to_numpy(float), old[["cx", "cy"]].to_numpy(float)])
    dref = nearest(P, ref)

    ok_hull = inR & inC
    ok_dist = (dR >= D_MIN_PX) & (dR <= D_MAX_PX) & (dC >= D_MIN_PX) & (dC <= D_MAX_PX)
    ok_diff = np.abs(dR - dC) <= D_DIFF_MAX_PX
    ok_new = dref >= MIN_SEP_PX
    conds = {"2_hull_R_and_C": ok_hull, "3a_distance_3_to_10px": ok_dist, "3b_|dR-dC|<=1.5": ok_diff,
             "4_>=15px_from_existing": ok_new}
    allok = ok_hull & ok_dist & ok_diff & ok_new

    seg_id = np.where(P[:, 1] < y_edges[0], 0, np.where(P[:, 1] < y_edges[1], 1, 2))
    report, chosen = {}, {}
    for i, s in enumerate(SEGMENTS):
        m = seg_id == i
        # sequential exclusion in rule order, and "each condition alone" counts
        seq, cur = {}, m.copy()
        seq["candidates_in_segment"] = int(cur.sum())
        for k, c in conds.items():
            seq[f"after_{k}"] = int((cur := cur & c).sum())
        alone = {f"fails_{k}": int((m & ~c).sum()) for k, c in conds.items()}
        report[s] = dict(sequential=seq, excluded_by_each_condition_alone=alone)
        idx = np.flatnonzero(m & allok)
        if len(idx) == 0:
            chosen[s] = None
            continue
        diff = np.round(np.abs(dR - dC)[idx] / TIE_ROUND_PX) * TIE_ROUND_PX
        best = idx[diff == diff.min()]
        cx0, cy0 = seg_centre[s]
        best = best[np.lexsort((P[best, 0], P[best, 1], np.hypot(P[best, 0] - cx0, P[best, 1] - cy0)))][0]
        chosen[s] = dict(position_id=f"P{i + 1}", segment=s, cx=float(P[best, 0]), cy=float(P[best, 1]),
                         d_R=round(float(dR[best]), 3), d_C=round(float(dC[best]), 3),
                         abs_dR_minus_dC=round(float(abs(dR[best] - dC[best])), 3),
                         n_feasible_in_segment=int(len(idx)), n_tied_at_min=int((diff == diff.min()).sum()),
                         nearest_existing_px=round(float(dref[best]), 2))

    out = dict(
        coordinate_system="front camera 640x480 px, cube centre in first frame",
        thresholds=dict(GRID_PX=GRID_PX, D_MIN_PX=D_MIN_PX, D_MAX_PX=D_MAX_PX, D_DIFF_MAX_PX=D_DIFF_MAX_PX,
                        MIN_SEP_PX=MIN_SEP_PX, TIE_ROUND_PX=TIE_ROUND_PX),
        right_side_definition=dict(x_split=round(x_split, 2), right_standard_positions=list(right_names),
                                   segment_y_edges=[round(v, 2) for v in y_edges],
                                   segment_centres={k: [round(a, 2) for a in v] for k, v in seg_centre.items()},
                                   segment_to_standard_row=row_names),
        training_sets=dict(R="expert 50 + recovery 20 (9 recovery positions = first still frame before grasp)",
                           C="expert 50 + new expert 20"),
        n_candidates_total=int(len(P)), n_feasible_total=int(allok.sum()),
        exclusion_counts=report,
        positions=[v for v in chosen.values() if v is not None],
        status="complete" if all(v is not None for v in chosen.values()) else
               "INCOMPLETE: " + ", ".join(k for k, v in chosen.items() if v is None) + " segment(s) empty",
    )
    (RES / "test_positions.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("status", "n_candidates_total", "n_feasible_total", "right_side_definition",
                                          "exclusion_counts", "positions")}, indent=2, ensure_ascii=False))

    # ---- figure -------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "savefig.bbox": "tight"})
    C_EXP, C_REC, C_NEW, C_STD, C_OLD = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#52514e"
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.scatter(P[allok, 0], P[allok, 1], s=9, c="#f2c94c", marker="s", linewidths=0, zorder=1,
               label=f"Feasible candidates (n={int(allok.sum())})")
    ax.scatter(exp.cx, exp.cy, s=34, c=C_EXP, edgecolors="white", linewidths=0.8, zorder=3, label="Expert 50 (R and C)")
    ax.scatter(rec.cx, rec.cy, s=55, marker="^", c=C_REC, edgecolors="white", linewidths=0.8, zorder=4,
               label="Recovery 20 (model R only)")
    ax.scatter(new.cx, new.cy, s=48, marker="D", c=C_NEW, edgecolors="white", linewidths=0.8, zorder=4,
               label="New expert 20 (model C only)")
    ax.scatter(S.cx, S.cy, marker="*", s=230, c=C_STD, edgecolors="white", linewidths=0.8, zorder=5,
               label="Standard test positions (6)")
    ax.scatter(old.cx, old.cy, marker="X", s=110, c=C_OLD, edgecolors="white", linewidths=0.8, zorder=5,
               label="Old extra positions (2)")
    for _, r in old.iterrows():
        ax.annotate(r.position, (r.cx, r.cy), xytext=(6, -10), textcoords="offset points", fontsize=8, color=C_OLD)
    for nm, r in S.iterrows():
        ax.annotate(nm, (r.cx, r.cy), xytext=(-10 if "left" in nm else 10, 8), textcoords="offset points",
                    fontsize=8, color="#0f7a55", ha="right" if "left" in nm else "left")
    for v in out["positions"]:
        ax.scatter(v["cx"], v["cy"], marker="o", s=210, facecolors="none", edgecolors="#d6249f", linewidths=2.4, zorder=6)
        ax.annotate(f"{v['position_id']} ({v['cx']:.0f}, {v['cy']:.0f})\nd_R={v['d_R']:.1f}, d_C={v['d_C']:.1f}",
                    (v["cx"], v["cy"]), xytext=(432, v["cy"]), textcoords="data", fontsize=9, color="#d6249f",
                    fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="-", color="#d6249f", lw=0.8, shrinkA=0, shrinkB=9))
    ax.axvline(x_split, color="#b5b4ae", lw=0.8, ls="--")
    for e in y_edges:
        ax.axhline(e, color="#b5b4ae", lw=0.8, ls=":")
    ax.plot([], [], color="#b5b4ae", ls="--", label="Left/right split; dotted = segment bounds")
    ax.scatter([], [], marker="o", s=100, facecolors="none", edgecolors="#d6249f", linewidths=2, label="Selected (3)")
    pts = np.vstack([R, C, ref]); m = 14
    ax.set_xlim(max(0, pts[:, 0].min() - m), min(IMG_W, pts[:, 0].max() + 62))
    ax.set_ylim(min(IMG_H, pts[:, 1].max() + m), max(0, pts[:, 1].min() - m))
    ax.set_aspect("equal"); ax.set_xlabel("x (px)"); ax.set_ylabel("y (px, down)")
    ax.set_title("Pre-registered right-side test positions (front camera pixel coordinates)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(RES / "test_positions.png"); plt.close(fig)
    return 0 if out["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())

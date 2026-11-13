#!/usr/bin/env python
"""
Check the cube positions of newly collected expert episodes before training:
detect the cube in each first frame (same detector/thresholds as analyze_cube_positions.py),
compare with the recovery data, and measure the distance to the 6 test positions.

Outputs go to ./new_expert_position_check/ next to this script. Raw datasets are only read.
Usage: python check_new_expert_positions.py [--dataset-dir DIR] [--name NAME]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_cube_positions as A  # noqa: E402
import analyze_grasp_points as G  # noqa: E402  (decode_frame)

NEW_DIR = A.LEROBOT_ROOT / "so101_red_cube_plastic_box_expert_20_20261112_130439"
NAME = "new_expert"
OUT = HERE / "new_expert_position_check"
MIN_DIST_PX = 5.0           # samples closer than this to a test position are listed
C_NEW = "#6250d6"           # reference palette slot 7 (violet); expert blue / recovery orange / test aqua
log = logging.getLogger("newcheck")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=NEW_DIR)
    ap.add_argument("--name", default=NAME)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "run.log", mode="w")])

    ref = pd.read_csv(HERE / "cube_positions.csv")
    ref_area = ref[(ref.dataset == "expert") & ref.detected].area.median()
    rows, tiles = [], []
    for ep, rgb in A.first_frames(args.dataset_dir):
        det = A.detect_cube(rgb)
        flags = []
        if not det["detected"]:
            flags.append("not_detected"); log.warning("episode %d: cube NOT detected", ep)
        else:
            if not (A.AREA_RATIO_RANGE[0] <= det["area"] / ref_area <= A.AREA_RATIO_RANGE[1]):
                flags.append(f"area_ratio={det['area'] / ref_area:.2f}")
            if det["aspect"] > A.MAX_ASPECT:
                flags.append(f"aspect={det['aspect']:.2f}")
            if det["second_area"] / det["area"] >= A.SECOND_BLOB_RATIO:
                flags.append(f"second_blob={det['second_area']:.0f}")
        rows.append(dict(dataset=args.name, episode_index=ep, cx=det["cx"], cy=det["cy"], area=det["area"],
                         detected=det["detected"], flags=";".join(flags)))
        title = (f"ep {ep}  ({det['cx']:.0f},{det['cy']:.0f})  area {det['area']:.0f}" if det["detected"]
                 else f"ep {ep}  NOT DETECTED")
        tiles.append(A.make_tile(rgb, det, title))
    new = pd.DataFrame(rows)
    A.save_contact_sheet(tiles, f"{args.name}: first frames, {int(new.detected.sum())}/{len(new)} detected",
                         OUT / f"contact_{args.name}.png")
    log.info("detected %d/%d; flagged: %s", new.detected.sum(), len(new),
             new[new["flags"] != ""][["episode_index", "flags"]].values.tolist() or "none")
    new.to_csv(OUT / f"{args.name}_positions.csv", index=False, float_format="%.2f")

    # ---- compare with recovery / old expert --------------------------------------------------
    centers = pd.read_csv(HERE / "test_positions.csv")
    rec = ref[(ref.dataset == "recovery") & ref.detected]
    exp = ref[(ref.dataset == "expert") & ref.detected]
    ok = new[new.detected]

    def summ(name, d):
        return dict(group=name, n=len(d), cx_mean=d.cx.mean(), cx_std=d.cx.std(), cx_min=d.cx.min(), cx_max=d.cx.max(),
                    cy_mean=d.cy.mean(), cy_std=d.cy.std(), cy_min=d.cy.min(), cy_max=d.cy.max())
    summary = pd.DataFrame([summ("old expert (50)", exp), summ(f"{args.name} (20)", ok), summ("recovery (11 detected)", rec)])
    summary.round(1).to_csv(OUT / "distribution_summary.csv", index=False)
    log.info("distribution summary:\n%s", summary.round(1).to_string(index=False))

    from scipy.stats import ks_2samp
    for lab, other in (("recovery", rec), ("old expert", exp)):
        log.info("KS test new vs %s: cx p=%.3f, cy p=%.3f", lab, ks_2samp(ok.cx, other.cx).pvalue,
                 ks_2samp(ok.cy, other.cy).pvalue)
    inside = ok[(ok.cx.between(rec.cx.min(), rec.cx.max())) & (ok.cy.between(rec.cy.min(), rec.cy.max()))]
    log.info("new samples inside the recovery bounding box: %d/%d", len(inside), len(ok))

    # ---- distance to test positions -----------------------------------------------------------
    P = centers[["cx", "cy"]].to_numpy(float)
    d = np.hypot(ok.cx.to_numpy()[:, None] - P[:, 0], ok.cy.to_numpy()[:, None] - P[:, 1])
    ok = ok.assign(nearest_test=centers.position.to_numpy()[d.argmin(1)], dist_to_nearest_test=d.min(1))
    ok.to_csv(OUT / f"{args.name}_distance_to_test.csv", index=False, float_format="%.2f")
    log.info("distance to nearest test position: min %.1f, median %.1f, max %.1f px",
             ok.dist_to_nearest_test.min(), ok.dist_to_nearest_test.median(), ok.dist_to_nearest_test.max())
    close = ok[ok.dist_to_nearest_test < MIN_DIST_PX]
    log.info("samples closer than %.0f px to a test position: %s", MIN_DIST_PX,
             close[["episode_index", "nearest_test", "dist_to_nearest_test"]].round(1).values.tolist() or "none")
    per = ok.groupby("nearest_test").dist_to_nearest_test.agg(["count", "min", "median"]).round(1)
    log.info("per test position (new samples whose nearest test position it is):\n%s", per.to_string())
    # nearest test position for the older training data, for comparison
    for lab, df in (("old expert", exp), ("recovery", rec)):
        dd = np.hypot(df.cx.to_numpy()[:, None] - P[:, 0], df.cy.to_numpy()[:, None] - P[:, 1]).min(1)
        log.info("%s: min distance to a test position %.1f px (median %.1f)", lab, dd.min(), np.median(dd))

    # ---- plot -----------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    allpts = pd.concat([exp[["cx", "cy"]], rec[["cx", "cy"]], ok[["cx", "cy"]], centers[["cx", "cy"]]])
    m = 12
    A.ZOOM_BOX = (max(0, allpts.cx.min() - m), max(0, allpts.cy.min() - m),
                  min(640, allpts.cx.max() + m), min(480, allpts.cy.max() + m))
    bg = G.decode_frame(*A.BACKGROUND_FRAME)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b5b4ae", "savefig.dpi": 200, "savefig.bbox": "tight"})

    def draw(ax, zoomed):
        k = 1.0 if zoomed else 0.45
        ax.scatter(exp.cx, exp.cy, s=36 * k, c=A.C_EXPERT, edgecolors="white", linewidths=1.0 * k, zorder=4,
                   label=f"Old expert demo (n={len(exp)})")
        ax.scatter(rec.cx, rec.cy, s=60 * k, marker="^", c=A.C_RECOVERY, edgecolors="white", linewidths=1.0 * k,
                   zorder=4, label=f"Recovery (n={len(rec)}; 9 not visible)")
        ax.scatter(ok.cx, ok.cy, s=60 * k, marker="D", c=C_NEW, edgecolors="white", linewidths=1.0 * k, zorder=5,
                   label=f"New expert (n={len(ok)})")
        A._stars(ax, centers, zoomed)
        if zoomed:
            ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    fig, _ = A._two_panel(plt, bg, "Initial cube positions: new expert data vs. existing training data and test positions", draw)
    fig.savefig(OUT / "position_distribution_new_expert.png"); plt.close(fig)
    log.info("done: %s", OUT)


if __name__ == "__main__":
    main()

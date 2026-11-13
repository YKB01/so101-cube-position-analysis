#!/usr/bin/env python
"""
Right-side extra test: can the difference between the two models be explained by how close the
training data is to the test positions?

Models (both fine-tuned 5,000 steps from the same baseline checkpoint):
  mixed   = expert 50 + recovery 20        training set "mixed"
  control = expert 50 + new expert 20      training set "control"

Stages
  detect   steps 1-3: cube position of every extra-test episode (first frame; if the cube is hidden,
           the first 10-frame still, visible window before the first grasp), contact sheets, and
           k-means (k=2) into the two extra test positions, with a check that both models were
           tested at the same positions.
  analyze  steps 4-7: distances from the two extra test positions to both training sets, comparison
           with the 6 standard test positions, figures A and B, report_right_side.md.

Pixel distances are IMAGE distances, not distances on the table (perspective: the same pixel
distance is longer on the table at the top/far side of the frame).

Inputs are only read. Outputs: ./right_side_test_analysis/ next to this script.
Usage: python analyze_right_side_test.py --stage detect | analyze | all
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
import analyze_grasp_points as G  # noqa: E402
import resolve_recovery_cube_positions as R  # noqa: E402  (still-window rule for hidden cubes)

# =============================================================================
# CONFIG
# =============================================================================
OUT = HERE / "right_side_test_analysis"
EDGE_DATASETS = {   # key -> directory under A.LEROBOT_ROOT
    "edge_mixed": "rollout_eval_edge_hilmixed_smooth_20261112_144609",
    "edge_control": "rollout_eval_edge_expertmixed_smooth_20261112_145925",
}
MODEL_LABEL = {"edge_mixed": "mixed model", "edge_control": "control model"}
for k, v in EDGE_DATASETS.items():
    A.DATASETS[k] = v   # lets analyze_cube_positions / analyze_grasp_points helpers find the dataset

CUBE_CSV = HERE / "cube_positions.csv"                                         # expert 50
RECOVERY_CSV = HERE / "recovery_position_check" / "recovery_positions_resolved.csv"
NEW_EXPERT_CSV = HERE / "new_expert_position_check" / "new_expert_positions.csv"
TEST_POS_CSV = HERE / "test_positions.csv"

N_EDGE_POSITIONS = 2
EDGE_NAMES = ("right-top", "right-bottom")          # by y: smaller y (higher in the image) first
SUCCESS = {   # extra test successes out of TRIALS (per model, per position)
    "edge_mixed": {"right-bottom": 8, "right-top": 4},
    "edge_control": {"right-bottom": 2, "right-top": 0},
}
TRIALS = 8
CUBE_PX = 17          # cube width in the image (sqrt of the median detected expert blob area)
FLAG_PX = 10.0        # a trial farther than this from its cluster centre is listed as a possible misplacement

# conclusion thresholds (step 7)
DIST_SAME_PX = 5.0    # nearest distances within this: same
COUNT_SAME = 1        # counts within one cube width differing by at most this: similar
COUNT_CLEARLY_MORE = 3  # mixed has at least this many more samples within one cube width: clearly more

C_EXPERT, C_RECOVERY, C_NEW, C_STAR = "#8a8984", "#eb6834", "#2a78d6", "#1baf7a"
C_TEXT, C_TEXT_2 = "#0b0b0b", "#52514e"
log = logging.getLogger("rightside")


# =============================================================================
# Stage 1-3: detection and clustering
# =============================================================================
def detect_edge(key: str, ref_area: float):
    ds_dir = A.LEROBOT_ROOT / A.DATASETS[key]
    meta = A.load_episode_meta(ds_dir)
    firsts = {ep: (rgb, A.detect_cube(rgb)) for ep, rgb in A.first_frames(ds_dir)}
    frames_df = None
    rows, tiles = [], []
    for row in meta.itertuples():
        ep = int(row.episode_index)
        rgb, d = firsts[ep]
        res = dict(dataset=key, episode_index=ep, first_frame_detected=bool(d["detected"]),
                   cx=d["cx"], cy=d["cy"], area=d["area"], frame_index=0 if d["detected"] else np.nan,
                   method="first_frame" if d["detected"] else "unresolved")
        tile_rgb, tile_det, title = rgb, d, None
        if d["detected"]:
            title = f"ep {ep}  ({d['cx']:.0f},{d['cy']:.0f})  area {d['area']:.0f}"
        else:
            log.warning("[%s] episode %d: cube not visible in frame 0 -> looking for a still window before the first grasp",
                        key, ep)
            if frames_df is None:
                frames_df, _ = G.load_frames(key)
            g = frames_df[frames_df.episode_index == ep][f"state:{G.GRIPPER}"].to_numpy(float)
            grasp = G.detect_first_grasp(g)
            end = int(grasp["grasp_frame"]) if not np.isnan(grasp["grasp_frame"]) else len(g)
            seq, imgs = [], {}
            for i, frgb in R.episode_frames(ds_dir, row):
                if i >= end:
                    break
                dd = A.detect_cube(frgb)
                ok = bool(dd["detected"]) and A.AREA_RATIO_RANGE[0] <= dd["area"] / ref_area <= A.AREA_RATIO_RANGE[1] \
                    and dd["aspect"] <= A.MAX_ASPECT
                seq.append(dict(f=i, valid=ok, cx=dd["cx"], cy=dd["cy"], area=dd["area"]))
                if ok:
                    imgs[i] = (frgb, dd)
            seq = pd.DataFrame(seq)
            t = R.first_still_window(seq, end) if len(seq) else None
            if t is None:
                log.warning("[%s] episode %d: UNRESOLVED (no still visible cube before the first grasp)", key, ep)
                title = f"ep {ep}  NOT DETECTED"
            else:
                w = seq.iloc[t:t + R.STILL_FRAMES]
                res.update(cx=w.cx.median(), cy=w.cy.median(), area=w.area.median(), frame_index=t,
                           method="first_still_before_grasp")
                tile_rgb, tile_det = imgs[t]
                title = f"ep {ep} f{t} (hidden in f0) ({res['cx']:.0f},{res['cy']:.0f})"
                log.info("[%s] episode %d resolved at frame %d: (%.1f, %.1f)", key, ep, t, res["cx"], res["cy"])
        rows.append(res)
        tiles.append(A.make_tile(tile_rgb, tile_det, title))
    df = pd.DataFrame(rows)
    A.save_contact_sheet(tiles, f"{key}: {MODEL_LABEL[key]} extra test, first frames, "
                                f"{int(df.cx.notna().sum())}/{len(df)} with a position",
                         OUT / f"contact_{key}.png")
    return df


def cluster_edge(edge: pd.DataFrame):
    from sklearn.cluster import KMeans
    ok = edge[edge.cx.notna()].copy()
    X = ok[["cx", "cy"]].to_numpy(float)
    km = KMeans(n_clusters=N_EDGE_POSITIONS, n_init=50, random_state=0).fit(X)
    order = np.argsort(km.cluster_centers_[:, 1])          # smaller y = higher in the image = top
    names = {int(c): EDGE_NAMES[r] for r, c in enumerate(order)}
    ok["position"] = [names[int(l)] for l in km.labels_]
    centers = pd.DataFrame([dict(position=names[c], cx=km.cluster_centers_[c, 0], cy=km.cluster_centers_[c, 1])
                            for c in range(N_EDGE_POSITIONS)]).sort_values("position", ascending=False)
    c = centers.set_index("position")
    ok["dist_to_center"] = np.hypot(ok.cx - ok.position.map(c.cx), ok.cy - ok.position.map(c.cy))
    return centers.reset_index(drop=True), ok


def stage_detect():
    ref = pd.read_csv(CUBE_CSV)
    ref_area = ref[(ref.dataset == "expert") & ref.detected].area.median()
    edge = pd.concat([detect_edge(k, ref_area) for k in EDGE_DATASETS], ignore_index=True)
    edge.to_csv(OUT / "edge_positions_raw.csv", index=False, float_format="%.2f")
    miss = edge[edge.cx.isna()]
    log.info("episodes without a position: %s", [(r.dataset, r.episode_index) for r in miss.itertuples()] or "none")
    centers, ok = cluster_edge(edge)
    ok.to_csv(OUT / "edge_positions.csv", index=False, float_format="%.2f")
    centers.to_csv(OUT / "edge_test_positions_pooled.csv", index=False, float_format="%.2f")

    ct = pd.crosstab(ok.dataset, ok.position)
    log.info("episodes per (dataset, extra test position):\n%s", ct.to_string())
    per = ok.groupby(["dataset", "position"]).agg(n=("cx", "size"), cx=("cx", "mean"), cy=("cy", "mean"),
                                                  cx_std=("cx", "std"), cy_std=("cy", "std"),
                                                  max_dist=("dist_to_center", "max")).round(1).reset_index()
    per.to_csv(OUT / "edge_test_positions_per_model.csv", index=False)
    log.info("per model:\n%s", per.to_string(index=False))
    for pos in EDGE_NAMES[::-1]:
        a = per[(per.dataset == "edge_mixed") & (per.position == pos)].iloc[0]
        b = per[(per.dataset == "edge_control") & (per.position == pos)].iloc[0]
        log.info("%s: centre mixed (%.1f, %.1f) vs control (%.1f, %.1f): difference dx %.1f, dy %.1f, |d| %.1f px",
                 pos, a.cx, a.cy, b.cx, b.cy, b.cx - a.cx, b.cy - a.cy, np.hypot(b.cx - a.cx, b.cy - a.cy))
    log.info("pooled centres:\n%s", centers.round(1).to_string(index=False))
    for r in ok[ok.dist_to_center > FLAG_PX].itertuples():
        log.warning("possible misplaced trial: %s episode %d at (%.0f, %.0f) is %.1f px from the %s centre",
                    r.dataset, r.episode_index, r.cx, r.cy, r.dist_to_center, r.position)
    ab = {"right-top": "RT", "right-bottom": "RB"}
    for k in EDGE_DATASETS:
        g = ok[ok.dataset == k].sort_values("episode_index")
        log.info("%-13s episode order: %s", k, " ".join(ab[p] for p in g.position))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    std = pd.read_csv(TEST_POS_CSV)
    pts = pd.concat([ok[["cx", "cy"]], centers[["cx", "cy"]], std[["cx", "cy"]]])
    m = 14
    box = (pts.cx.min() - m, pts.cy.min() - m, pts.cx.max() + m, pts.cy.max() + m)
    bg = G.decode_frame(*A.BACKGROUND_FRAME)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b5b4ae", "savefig.dpi": 200, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(8.5, 8.5 * (box[3] - box[1]) / (box[2] - box[0]) + 0.4))
    ax.imshow(bg, alpha=0.7)
    for k, mk, col in (("edge_mixed", "o", C_RECOVERY), ("edge_control", "s", C_NEW)):
        g = ok[ok.dataset == k]
        ax.scatter(g.cx, g.cy, s=34, marker=mk, c=col, edgecolors="white", linewidths=0.8, zorder=4,
                   label=f"{MODEL_LABEL[k]} extra-test trial (n={len(g)})")
    ax.scatter(std.cx, std.cy, marker="*", s=200, facecolors="none", edgecolors=C_STAR, linewidths=1.6, zorder=5,
               label="standard test position")
    ax.scatter(centers.cx, centers.cy, marker="*", s=360, c=C_STAR, edgecolors="white", linewidths=1.5, zorder=6,
               label="extra test position (cluster centre)")
    import matplotlib.patheffects as pe
    for c in centers.itertuples():
        ax.annotate(c.position, (c.cx, c.cy), xytext=(14, 0), textcoords="offset points", va="center",
                    fontweight="bold", path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=7)
    ax.set_xlim(box[0], box[2]); ax.set_ylim(box[3], box[1])
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.set_title("Extra test: first-frame cube positions and the two cluster centres", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8.5, framealpha=0.9)
    fig.savefig(OUT / "check_edge_clusters.png"); plt.close(fig)
    log.info("detect stage done: %s", OUT)


# =============================================================================
# Stage 4-7: distances to the training sets
# =============================================================================
def load_training_sets() -> dict[str, pd.DataFrame]:
    exp = pd.read_csv(CUBE_CSV)
    exp = exp[(exp.dataset == "expert") & exp.detected][["episode_index", "cx", "cy"]].assign(
        src="expert50", method="first_frame")
    rec = pd.read_csv(RECOVERY_CSV)
    rec = rec[rec.method != "unresolved"][["episode_index", "cx", "cy", "method"]].assign(src="recovery20")
    new = pd.read_csv(NEW_EXPERT_CSV)
    new = new[new.detected][["episode_index", "cx", "cy"]].assign(src="new_expert20", method="first_frame")
    for name, d, n in (("expert", exp, 50), ("recovery", rec, 20), ("new expert", new, 20)):
        if len(d) != n:
            log.warning("%s: %d positions instead of %d", name, len(d), n)
    return {"mixed": pd.concat([exp, rec], ignore_index=True), "control": pd.concat([exp, new], ignore_index=True),
            "_exp": exp, "_rec": rec, "_new": new}


def metrics(cx, cy, train: pd.DataFrame) -> dict:
    d = np.hypot(train.cx.to_numpy() - cx, train.cy.to_numpy() - cy)
    o = np.argsort(d)
    n = train.iloc[o[0]]
    r = train.iloc[int(train.cx.to_numpy().argmax())]
    return dict(nearest_px=d[o[0]], nearest_src=f"{n.src}:{int(n.episode_index)}", nearest_method=n.method,
                mean3_px=d[o[:3]].mean(), n_within_17=int((d <= CUBE_PX).sum()), n_within_34=int((d <= 2 * CUBE_PX).sum()),
                x_margin_px=cx - train.cx.max(), rightmost_src=f"{r.src}:{int(r.episode_index)}")


def verdict(m: dict, c: dict) -> tuple[str, str]:
    """Return (code, text) comparing mixed (m) and control (c) metrics at one position."""
    dd = c["nearest_px"] - m["nearest_px"]          # > 0: mixed is closer
    dn = m["n_within_17"] - c["n_within_17"]        # > 0: mixed has more samples within one cube width
    if abs(dd) <= DIST_SAME_PX and abs(dn) <= COUNT_SAME:
        return "not_explain", "same distance and similar counts"
    if dd > DIST_SAME_PX or dn >= COUNT_CLEARLY_MORE:
        return "may_explain", "mixed set clearly closer" if dd > DIST_SAME_PX else "mixed set has clearly more samples within 17 px"
    if dd < -DIST_SAME_PX:
        return "not_explain", "control set is closer, so coverage does not favour the mixed model"
    return "undecided", "between the criteria"


def stage_analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import Circle
    ok = pd.read_csv(OUT / "edge_positions.csv")
    centers = pd.read_csv(OUT / "edge_test_positions_pooled.csv").set_index("position")
    std = pd.read_csv(TEST_POS_CSV)
    T = load_training_sets()
    sets = {"mixed": T["mixed"], "control": T["control"]}

    # ---- step 4: per position metrics ---------------------------------------------------------
    rows, vrows = [], []
    for pos in EDGE_NAMES[::-1]:
        c = centers.loc[pos]
        ms = {k: metrics(c.cx, c.cy, s) for k, s in sets.items()}
        for k, m in ms.items():
            rows.append(dict(position=pos, cx=round(c.cx, 1), cy=round(c.cy, 1), training_set=k, **m))
        code, why = verdict(ms["mixed"], ms["control"])
        vrows.append(dict(position=pos, code=code, why=why, d_mixed=ms["mixed"]["nearest_px"],
                          d_control=ms["control"]["nearest_px"],
                          n17_mixed=ms["mixed"]["n_within_17"], n17_control=ms["control"]["n_within_17"]))
    pm = pd.DataFrame(rows)
    pm.round(1).to_csv(OUT / "edge_position_distance_metrics.csv", index=False)
    log.info("per position metrics:\n%s", pm.round(1).to_string(index=False))
    vd = pd.DataFrame(vrows)

    # which recovery / new-expert samples lie within 34 px of each extra position
    near = []
    for pos in EDGE_NAMES[::-1]:
        c = centers.loc[pos]
        for lab, s in (("recovery20", T["_rec"]), ("new_expert20", T["_new"]), ("expert50", T["_exp"])):
            d = np.hypot(s.cx - c.cx, s.cy - c.cy)
            for i in np.nonzero(d <= 2 * CUBE_PX)[0]:
                r = s.iloc[i]
                near.append(dict(position=pos, src=lab, episode_index=int(r.episode_index), cx=round(r.cx, 1),
                                 cy=round(r.cy, 1), dist_px=round(d.iloc[i], 1), method=r.method))
    near = pd.DataFrame(near).sort_values(["position", "dist_px"])
    near.to_csv(OUT / "training_samples_within_34px.csv", index=False)

    # ---- per-episode distances -----------------------------------------------------------------
    pe_rows = []
    for r in ok.itertuples():
        row = dict(dataset=r.dataset, episode_index=r.episode_index, position=r.position,
                   cx=round(r.cx, 1), cy=round(r.cy, 1))
        for k, s in sets.items():
            m = metrics(r.cx, r.cy, s)
            row.update({f"{k}_nearest_px": round(m["nearest_px"], 1), f"{k}_nearest_src": m["nearest_src"],
                        f"{k}_n_within_17": m["n_within_17"]})
        pe_rows.append(row)
    pe_df = pd.DataFrame(pe_rows)
    pe_df.to_csv(OUT / "edge_episode_distances.csv", index=False)
    agg = pe_df.groupby(["dataset", "position"])[["mixed_nearest_px", "control_nearest_px"]].agg(["mean", "max"]).round(1)
    log.info("per-episode nearest distance (mean/max):\n%s", agg.to_string())

    # ---- step 5: comparison with the 6 standard positions -------------------------------------
    comp = []
    for pos in EDGE_NAMES[::-1]:
        c = centers.loc[pos]
        for s in std.itertuples():
            comp.append(dict(position=pos, standard=s.position, dx_right_px=round(c.cx - s.cx, 1),
                             dy_down_px=round(c.cy - s.cy, 1), dist_px=round(np.hypot(c.cx - s.cx, c.cy - s.cy), 1)))
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "edge_vs_standard_positions.csv", index=False)
    nearest_std = comp.loc[comp.groupby("position").dist_px.idxmin()]
    log.info("edge vs standard positions:\n%s", comp[comp.standard.isin(["mid-right", "bottom-right", "top-right"])]
             .to_string(index=False))

    # ---- figure A ---------------------------------------------------------------------------------
    cdf = centers.reset_index()
    allp = pd.concat([T["mixed"][["cx", "cy"]], T["control"][["cx", "cy"]], std[["cx", "cy"]], centers[["cx", "cy"]]])
    m_ = 14
    box = (allp.cx.min() - m_, centers.cy.min() - 2 * CUBE_PX - m_ - 14, centers.cx.max() + 2 * CUBE_PX + m_,
           allp.cy.max() + m_)
    bg = G.decode_frame(*A.BACKGROUND_FRAME)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b5b4ae", "savefig.dpi": 200, "savefig.bbox": "tight"})
    h = 7.2 * (box[3] - box[1]) / (box[2] - box[0]) * 1.0
    fig, axes = plt.subplots(1, 2, figsize=(14, h + 1.2), sharey=True)
    for ax, k, title in zip(axes, ("mixed", "control"),
                            ("Mixed training set (expert 50 + recovery 20)", "Control training set (expert 50 + new expert 20)")):
        ax.imshow(bg, alpha=0.7)
        e = T["_exp"]
        ax.scatter(e.cx, e.cy, s=18, c=C_EXPERT, zorder=3, label=f"expert 50")
        if k == "mixed":
            r = T["_rec"]; vis = r[r.method == "first_frame"]; hid = r[r.method != "first_frame"]
            ax.scatter(vis.cx, vis.cy, s=52, marker="^", c=C_RECOVERY, edgecolors="white", linewidths=0.8, zorder=4,
                       label=f"recovery 20: visible in frame 0 (n={len(vis)})")
            ax.scatter(hid.cx, hid.cy, s=52, marker="^", facecolors="white", edgecolors=C_RECOVERY, linewidths=1.6,
                       zorder=4, label=f"recovery 20: resolved later (n={len(hid)})")
        else:
            n_ = T["_new"]
            ax.scatter(n_.cx, n_.cy, s=46, marker="s", c=C_NEW, edgecolors="white", linewidths=0.8, zorder=4,
                       label="new expert 20")
        ax.scatter(std.cx, std.cy, marker="*", s=200, facecolors="none", edgecolors=C_STAR, linewidths=1.6, zorder=5,
                   label="standard test position")
        for c in cdf.itertuples():
            for rad, ls in ((CUBE_PX, "-"), (2 * CUBE_PX, "--")):
                ax.add_patch(Circle((c.cx, c.cy), rad, fill=False, ec=C_STAR, lw=1.2, ls=ls, zorder=5))
        ax.scatter(centers.cx, centers.cy, marker="*", s=380, c=C_STAR, edgecolors="white", linewidths=1.5, zorder=6,
                   label="extra test position (circles: 17 / 34 px)")
        for c in cdf.itertuples():
            mm = metrics(c.cx, c.cy, sets[k])
            ax.annotate(f"{c.position}\nnearest {mm['nearest_px']:.1f} px\n{mm['n_within_17']} within 17 px, {mm['n_within_34']} within 34 px",
                        (c.cx, c.cy), xytext=(0, 40 if "top" in c.position else -52), textcoords="offset points",
                        ha="center", fontsize=8.5, fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=7)
        ax.set_xlim(box[0], box[2]); ax.set_ylim(box[3], box[1]); ax.set_xlabel("x (px)")
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, framealpha=0.9)
    axes[0].set_ylabel("y (px)")
    fig.suptitle("Training-data cube positions around the two extra test positions", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "figA_training_coverage.png"); plt.close(fig)

    # ---- figure B ---------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    w = 0.36
    for j, (k, col, lab) in enumerate((("mixed", C_RECOVERY, "mixed training set"), ("control", C_NEW, "control training set"))):
        d = [pm[(pm.position == p) & (pm.training_set == k)].nearest_px.iloc[0] for p in EDGE_NAMES[::-1]]
        xs = np.arange(2) + (j - 0.5) * w
        ax.bar(xs, d, w * 0.9, color=col, label=lab, zorder=3)
        mk = "edge_mixed" if k == "mixed" else "edge_control"
        for x, v, p in zip(xs, d, EDGE_NAMES[::-1]):
            ax.text(x, v + 0.5, f"{v:.1f} px", ha="center", va="bottom", fontsize=9, color=C_TEXT)
            ax.text(x, v / 2, f"success\n{SUCCESS[mk][p]}/{TRIALS}", ha="center", va="center", fontsize=10,
                    color="white", fontweight="bold")
    ax.axhline(CUBE_PX, color=C_TEXT_2, lw=1, ls="--", zorder=2)
    ax.text(1.62, CUBE_PX + 0.4, "one cube width (17 px)", ha="right", fontsize=8.5, color=C_TEXT_2)
    ax.set_xticks(range(2)); ax.set_xticklabels(EDGE_NAMES[::-1])
    ax.set_ylabel("distance to nearest training sample (px)")
    ax.set_ylim(0, max(pm.nearest_px.max() * 1.25, 22)); ax.grid(axis="y", color="#e6e5e0"); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Nearest training sample at the two extra test positions", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "figB_nearest_distance.png"); plt.close(fig)

    write_report(OUT / "report_right_side.md", ok, centers, pm, vd, near, pe_df, comp, nearest_std, T)
    log.info("analyze stage done: %s", OUT)


def write_report(path, ok, centers, pm, vd, near, pe_df, comp, nearest_std, T):
    per = pd.read_csv(OUT / "edge_test_positions_per_model.csv")
    L = ["# Right-side extra test: can training-data position coverage explain the model difference?", "",
         "Pixel distances are image distances, not distances on the table. Because of perspective, the same pixel "
         "distance corresponds to a longer table distance at the top (far side) of the frame. "
         f"Cube width in the image: {CUBE_PX} px.", ""]
    L += ["## Extra test positions", "",
          "Extra test: 2 positions x 8 trials per model (smooth inference), first-frame cube positions clustered "
          f"with k-means (k={N_EDGE_POSITIONS}), labelled by y.", ""]
    ct = pd.crosstab(ok.dataset, ok.position)
    L += [ct.to_markdown(), ""]
    L += [per.to_markdown(index=False), ""]
    for pos in EDGE_NAMES[::-1]:
        a = per[(per.dataset == "edge_mixed") & (per.position == pos)].iloc[0]
        b = per[(per.dataset == "edge_control") & (per.position == pos)].iloc[0]
        L.append(f"- {pos}: centres differ by dx {b.cx - a.cx:+.1f}, dy {b.cy - a.cy:+.1f} px "
                 f"(|d| = {np.hypot(b.cx - a.cx, b.cy - a.cy):.1f} px) between the two models.")
    fl = ok[ok.dist_to_center > FLAG_PX]
    L += ["", f"Trials more than {FLAG_PX:g} px from their cluster centre (possible misplacement): "
          + (", ".join(f"{r.dataset} ep {r.episode_index} ({r.dist_to_center:.1f} px)" for r in fl.itertuples())
             if len(fl) else "none") + "."]
    unres = ok.method.eq("first_still_before_grasp").sum()
    L += [f"Episodes whose cube was hidden in frame 0 and resolved from a later still window: {int(unres)}.", ""]

    L += ["## Distance from the extra test positions to the training sets", "",
          "mixed = expert 50 + recovery 20, control = expert 50 + new expert 20. `x_margin_px` = test x minus the "
          "x of the rightmost training sample (positive: the test position lies to the right of all training data).", "",
          pm.round(1).to_markdown(index=False), ""]
    n_exp = near[near.src == "expert50"].groupby("position").size()
    L += ["### Recovery / new expert samples within 34 px (two cube widths) of an extra test position", "",
          "Expert 50 samples within 34 px: " + ", ".join(f"{p} {int(n_exp.get(p, 0))}" for p in EDGE_NAMES[::-1])
          + " (same in both training sets; full list in `training_samples_within_34px.csv`).", "",
          near[near.src != "expert50"].to_markdown(index=False), ""]
    L += ["### Per-episode nearest distance (mean / max over the 8 trials)", ""]
    agg = pe_df.groupby(["dataset", "position"]).agg(
        mixed_mean=("mixed_nearest_px", "mean"), mixed_max=("mixed_nearest_px", "max"),
        control_mean=("control_nearest_px", "mean"), control_max=("control_nearest_px", "max")).round(1).reset_index()
    L += [agg.to_markdown(index=False), "", "Full table: `edge_episode_distances.csv`.", ""]

    L += ["## Comparison with the 6 standard test positions", "",
          "Offsets of each extra position from the standard positions (dx: positive = further right, "
          "dy: positive = further down).", "",
          comp[comp.standard.isin(["mid-right", "bottom-right", "top-right"])].to_markdown(index=False), "",
          "Nearest standard position: " + "; ".join(
              f"{r.position} -> {r.standard} ({r.dx_right_px:+.1f}, {r.dy_down_px:+.1f}) px" for r in nearest_std.itertuples()), ""]

    L += ["## Success rates", "", "| position | mixed model | control model |", "|---|---|---|"]
    for p in EDGE_NAMES[::-1]:
        L.append(f"| {p} | {SUCCESS['edge_mixed'][p]}/{TRIALS} | {SUCCESS['edge_control'][p]}/{TRIALS} |")
    L += ["", "## Conclusion", "",
          f"Criteria: nearest distances within {DIST_SAME_PX:g} px and counts within {CUBE_PX} px differing by at most "
          f"{COUNT_SAME} -> position coverage does not explain the difference; mixed set closer by more than "
          f"{DIST_SAME_PX:g} px or with at least {COUNT_CLEARLY_MORE} more samples within {CUBE_PX} px -> it may "
          "explain it; otherwise undecided.", "",
          vd.round(1).to_markdown(index=False), ""]
    codes = set(vd.code)
    if "may_explain" in codes:
        c = "Position coverage may still explain the difference."
        rec_near = near[(near.src == "recovery20") & (near.dist_px <= 2 * CUBE_PX)]
        extra = ("Recovery samples within 34 px: " + ", ".join(
            f"ep {r.episode_index} at {r.position} ({r.dist_px:.1f} px, {r.method})" for r in rec_near.itertuples())
                 if len(rec_near) else "no recovery samples within 34 px.")
    elif codes == {"not_explain"}:
        c, extra = "Position coverage does not explain the difference.", ""
    else:
        c, extra = "Cannot tell whether position coverage explains the difference.", ""
    L += [f"**{c}** {extra}", ""]
    dd = pe_df.assign(diff=pe_df.mixed_nearest_px - pe_df.control_nearest_px).groupby(["dataset", "position"])["diff"].mean().round(1)
    L += ["Per-trial check (trials as actually run; positive = the control training set is closer): "
          + "; ".join(f"{MODEL_LABEL[d]} at {p}: {v:+.1f} px" for (d, p), v in dd.items()) + ". "
          + ("The control set is closer or equal for the trials that were run, so coverage, if anything, favours the "
             "control model, the opposite of the observed success rates." if (dd >= -DIST_SAME_PX).all() and (dd > 0).any() else ""), ""]
    ns = nearest_std.set_index("position")
    L += ["Notes:", "",
          f"- right-top is only {ns.loc['right-top'].dist_px:.1f} px from the standard top-right position "
          f"(dx {ns.loc['right-top'].dx_right_px:+.1f}, dy {ns.loc['right-top'].dy_down_px:+.1f}), so it is not a new "
          f"position; right-bottom lies {ns.loc['right-bottom'].dx_right_px:+.1f} px right and "
          f"{ns.loc['right-bottom'].dy_down_px:+.1f} px down of mid-right.",
          "- Both extra positions are within one cube width of at least 6 training samples in both training sets; "
          "neither lies far outside the training data.",
          "- The recovery positions that were hidden in frame 0 (9 of 20) come from the first still, visible window "
          "before the first grasp, not from the moment of intervention; their positions may differ from where the "
          "cube was originally placed.",
          "- Eight trials per position and model: the success difference itself (8/8 vs 2/8) is large, but this "
          "analysis can only exclude position coverage, not identify the cause."]
    L += [""]
    L += ["Success counts are 8 trials per position per model; this analysis shows where the training data is "
          "closer, not why the models behave differently."]
    path.write_text("\n".join(L) + "\n")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["detect", "analyze", "all"], default="all")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(OUT / f"run_{args.stage}.log", mode="w")])
    if args.stage in ("detect", "all"):
        stage_detect()
    if args.stage in ("analyze", "all"):
        stage_analyze()


if __name__ == "__main__":
    main()

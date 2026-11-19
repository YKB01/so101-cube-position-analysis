#!/usr/bin/env python
"""
Position coverage of the training sets of the four models for the test positions P1 and P2 (same method as scripts/select_test_positions.py / analyze_right_side_test.py):
distance from the test position to the nearest training sample (px) and the number of training samples within one cube width (17 px).
Training sets: R = expert 50 + recovery 20 (resolved positions), C and C_trunc = expert 50 + new expert 20, deep = expert 50 + the 20 deep-rewind corrections.
Position of a deep correction = the cube in the FIRST FRAME of the correction episode; if the arm hides it there (not fully visible), the last fully visible position
during the rewind segment, else its position at the start of the episode (column `source`).
Writes coverage_four_models.csv, coverage_deep20_positions.csv, fig_coverage_four_models.png
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import analyze_cube_positions as A
from lerobot.datasets import LeRobotDataset
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = HERE / "rewind_offline"; RUN = HERE / "results/rewind_collect/run_20261119_120625"; ROOT = Path.home() / ".cache/huggingface/lerobot/YKB0123"
CUBE_PX = 17.0
FULL = float(pd.read_csv(HERE / "cube_positions.csv").query("dataset == 'expert' and detected").area.median()) * 0.8

def full(rgb):
    d = A.detect_cube(np.asarray(rgb)); return (float(d["cx"]), float(d["cy"])) if (d["detected"] and d["area"] >= FULL and d["n_candidates"] == 1) else None

exp = pd.read_csv(HERE / "cube_positions.csv").query("dataset == 'expert' and detected")[["cx", "cy"]].to_numpy()
rec = pd.read_csv(HERE / "recovery_position_check/recovery_positions_resolved.csv")[["cx", "cy"]].to_numpy()
new = pd.read_csv(HERE / "new_expert_position_check/new_expert_positions.csv").query("detected")[["cx", "cy"]].to_numpy()
assert (len(exp), len(rec), len(new)) == (50, 20, 20)
# deep corrections
m = pd.read_csv(RUN / "rewind_meta.csv"); sv = m[~m.discarded].sort_values("saved_episode_index")
cor = LeRobotDataset("YKB0123/rewind_deep_20", root=ROOT / "rewind_deep_20"); seg = LeRobotDataset("YKB0123/rewind_deep_20_rewind_seg", root=ROOT / "rewind_deep_20_rewind_seg")
cf, sf, st = cor.meta.episodes["dataset_from_index"], seg.meta.episodes["dataset_from_index"], seg.meta.episodes["dataset_to_index"]
rows = []
for r in sv.itertuples():
    k = int(r.saved_episode_index); im = np.asarray(cor[int(cf[k])]["observation.images.front"].permute(1, 2, 0).clip(0, 1) * 255, dtype=np.uint8); p = full(im); src = "correction first frame"
    if p is None:                                                     # hidden by the arm: last fully visible position during the rewind segment
        sk = int(r.episode_id) - 1; last = None
        for i in range(int(sf[sk]), int(st[sk])):
            q = full(np.asarray(seg[i]["observation.images.front"].permute(1, 2, 0).clip(0, 1) * 255, dtype=np.uint8))
            if q is not None: last = q
        p, src = (last, "last visible during the rewind") if last is not None else ((r.cube_start_x, r.cube_start_y), "episode start")
    rows.append(dict(saved_episode_index=k, episode_id=int(r.episode_id), position_id=r.position_id, cx=p[0], cy=p[1], source=src, cube_start_x=r.cube_start_x, cube_start_y=r.cube_start_y))
D = pd.DataFrame(rows); D["shift_from_start_px"] = np.hypot(D.cx - D.cube_start_x, D.cy - D.cube_start_y); D.to_csv(OUT / "coverage_deep20_positions.csv", index=False, float_format="%.2f")
print("deep correction positions by source:", D.source.value_counts().to_dict(), "| shift from the episode-start position: median %.1f px, max %.1f" % (D.shift_from_start_px.median(), D.shift_from_start_px.max()))
deep = D[["cx", "cy"]].to_numpy()
tests = {"P1": (386.0, 182.0), "P2": (394.0, 212.0), "P3": (390.0, 230.0)}
sets = {"R": ("recovery 20", rec), "C": ("new expert 20", new), "C_trunc": ("new expert 20 (truncated, same positions)", new), "deep": ("deep-rewind corrections 20", deep)}
res = []
for mname, (lab, X) in sets.items():
    S = np.vstack([exp, X])
    for pn, (x, y) in tests.items():
        d = np.hypot(S[:, 0] - x, S[:, 1] - y); dx = np.hypot(X[:, 0] - x, X[:, 1] - y); de = np.hypot(exp[:, 0] - x, exp[:, 1] - y)
        res.append(dict(model=mname, extra_set=lab, position=pn, nearest_px=d.min(), nearest_extra_px=dx.min(), n_within_17_all=int((d <= CUBE_PX).sum()), n_within_17_extra=int((dx <= CUBE_PX).sum()),
                        n_within_34_all=int((d <= 2 * CUBE_PX).sum()), mean3_px=np.sort(d)[:3].mean(), nearest_expert_px=de.min()))
Rr = pd.DataFrame(res); Rr.to_csv(OUT / "coverage_four_models.csv", index=False, float_format="%.2f")
pd.set_option("display.width", 200)
print(Rr[Rr.position.isin(["P1", "P2"])].round(1).to_string(index=False))
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
col = {"R": "#eb6834", "C": "#6250d6", "deep": "#1baf7a"}
for ax, pn in zip(axs, ("P1", "P2")):
    x, y = tests[pn]
    ax.scatter(exp[:, 0], exp[:, 1], s=14, color="#8a8984", label="expert 50", lw=0, alpha=0.8)
    for mname, lab in (("R", "recovery 20"), ("C", "new expert 20 (C, C_trunc)"), ("deep", "deep-rewind corrections 20")):
        X = sets[mname][1]; ax.scatter(X[:, 0], X[:, 1], s=30, color=col[mname], label=lab, lw=0.6, edgecolor="white", alpha=0.9, marker={"R": "^", "C": "s", "deep": "o"}[mname])
    ax.scatter([x], [y], marker="*", s=260, facecolors="none", edgecolors="#0b0b0b", lw=1.6, label=f"{pn} test position")
    ax.add_patch(plt.Circle((x, y), CUBE_PX, fill=False, color="#0b0b0b", lw=0.8, ls=":")); ax.set_xlim(x - 60, x + 60); ax.set_ylim(y + 60, y - 60); ax.set_aspect("equal")
    ax.set_title(f"{pn} ({x:.0f}, {y:.0f}); dotted circle = one cube width", fontsize=10); ax.set_xlabel("x (px)")
axs[0].set_ylabel("y (px)"); axs[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_coverage_four_models.png"); plt.close(fig)

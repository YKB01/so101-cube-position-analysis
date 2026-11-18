#!/usr/bin/env python
"""fig6: policy sensitivity to brightness / colour temperature (from policy_sensitivity_summary.csv)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 9.5, "savefig.dpi": 160, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
S = pd.read_csv(OUT / "policy_sensitivity_summary.csv")
order = ["brightness x0.70", "brightness x0.85", "brightness x0.95", "brightness x1.05", "brightness x1.15", "brightness x1.30", "cool (R-10%, B+10%)", "warm (R+10%, B-10%)"]
models = ["C", "R", "C_trunc", "baseline"]; col = {"C": "#6250d6", "R": "#eb6834", "C_trunc": "#d03a3a", "baseline": "#8a8984"}
fig, axs = plt.subplots(1, 2, figsize=(13, 3.9))
for ax, (m, lab) in zip(axs, (("joint_diff_deg", "Mean |joint difference|, first 30 chunk steps (deg)"), ("aim_shift_px", "Shift of the predicted aim point (px)"))):
    w = 0.2
    for j, mod in enumerate(models):
        y = [S[(S["transform"] == t) & (S["model"] == mod)][m].iloc[0] for t in order]
        ax.bar(np.arange(len(order)) + (j - 1.5) * w, y, w * 0.9, color=col[mod], label=mod, zorder=2)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([o.replace("brightness ", "bright. ").replace(" (", "\n(") for o in order], fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel(lab, fontsize=8.5); ax.axvspan(1.5, 3.5, color="#b5b4ae", alpha=0.2, lw=0, zorder=0)
axs[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.72, 1.0))
axs[0].text(2.5, axs[0].get_ylim()[1] * 0.72, "about the range seen\nbetween sessions (+-5%)", ha="center", va="top", fontsize=7.5, color="#52514e")
fig.suptitle("Policy sensitivity to image brightness / colour temperature (18 frames from 6 successful retest trials)", y=1.02)
fig.savefig(OUT / "fig6_policy_sensitivity.png"); plt.close(fig)

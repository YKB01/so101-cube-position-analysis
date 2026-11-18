#!/usr/bin/env python
"""Lighting analysis step 2.3: behaviour of the 11/14 trials against the brightness of their first frame. Reads trial_lighting_behavior.csv."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = HERE / "lighting_analysis"
plt.rcParams.update({"font.size": 9.5, "savefig.dpi": 160, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
LOG = open(OUT / "step2b_log.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n")

B = pd.read_csv(OUT / "trial_lighting_behavior.csv")
SERIES = ["retest C", "retest R", "trunc session C", "trunc session C_trunc"]
COL = {"retest C": "#6250d6", "retest R": "#eb6834", "trunc session C": "#2a78d6", "trunc session C_trunc": "#d03a3a"}
LIGHT = {"gray_mean": "whole-frame gray", "table_gray": "table gray", "bg_sheet_right_gray": "window-side gray", "tod": "time of day (h)"}
BEH = {"success": "success (0/1)", "speed10": "mean arm speed, first 10 s", "onset_s": "motion onset (s)", "first_grasp_time_s": "first close (s)", "aim_err_x": "aim error x at grasp (px)"}

rows = []
for s in SERIES:
    x = B[B.series == s]
    for lm in LIGHT:
        for bm in BEH:
            y = x[[lm, bm]].dropna()
            if len(y) >= 6 and y[bm].nunique() > 1:
                r, p = stats.spearmanr(y[lm], y[bm]); rows.append(dict(series=s, light=lm, behaviour=bm, n=len(y), spearman=r, p=p))
Cr = pd.DataFrame(rows); Cr.to_csv(OUT / "correlations.csv", index=False, float_format="%.4f")
P("Spearman correlations of behaviour with first-frame lighting (n = 16 or 24 per series; |rho| > 0.5 needed for p < 0.05 at n = 16):")
P(Cr[Cr.light != "tod"].pivot_table(index=["series", "behaviour"], columns="light", values="spearman").round(2).to_string())
sig = Cr[(Cr.p < 0.05) & (Cr.light != "tod")]
P(f"\n{len(sig)} of {len(Cr[Cr.light != 'tod'])} light-behaviour correlations have p < 0.05 (no multiplicity correction; about {0.05 * len(Cr[Cr.light != 'tod']):.0f} expected by chance):")
P(sig.round(3).to_string(index=False))
P("\ntime of day against behaviour (same series):")
P(Cr[Cr.light == "tod"].pivot_table(index="series", columns="behaviour", values="spearman").round(2).to_string())

# pooled over series with model-session as a fixed effect: partial effect of brightness after removing series means
Bp = B.copy()
for lm in ("gray_mean", "table_gray", "bg_sheet_right_gray"):
    Bp[lm + "_c"] = Bp[lm] - Bp.groupby("series")[lm].transform("mean")
for bm in ("speed10", "first_grasp_time_s", "aim_err_x"):
    Bp[bm + "_c"] = Bp[bm] - Bp.groupby("series")[bm].transform("mean")
P("\nwithin-series (series mean removed) correlation, all 80 trials pooled:")
for lm in ("gray_mean", "table_gray", "bg_sheet_right_gray"):
    for bm in ("speed10", "first_grasp_time_s", "aim_err_x"):
        y = Bp[[lm + "_c", bm + "_c"]].dropna(); r, p = stats.spearmanr(y.iloc[:, 0], y.iloc[:, 1]); P(f"  {lm:22s} vs {bm:20s}: rho {r:+.2f} (p {p:.2f}, n {len(y)})")
# success against brightness (Mann-Whitney within series)
P("\nbrightness of successful vs failed trials (median whole-frame gray / table gray):")
for s in SERIES:
    x = B[B.series == s]
    if x.success.nunique() > 1:
        a, b = x[x.success == 1], x[x.success == 0]
        p = stats.mannwhitneyu(a.table_gray, b.table_gray).pvalue
        P(f"  {s:24s} success n={len(a)} table {a.table_gray.median():.1f} gray {a.gray_mean.median():.1f} | fail n={len(b)} table {b.table_gray.median():.1f} gray {b.gray_mean.median():.1f} | MWU p(table)={p:.2f}")
    else:
        P(f"  {s:24s} all {'success' if x.success.iloc[0] else 'fail'} (n={len(x)})")

# stalled at the start (C_trunc): onset later than every C trial's onset (max 4.77 s over C and R)
thr = B[B.series.isin(["retest C", "retest R", "trunc session C"])].onset_s.max()
ct = B[B.series == "trunc session C_trunc"].copy(); ct["stalled"] = ct.onset_s > thr
P(f"\nC_trunc trials that stand still at the start (motion onset > {thr:.2f} s, the latest onset of any C or R trial): {int(ct.stalled.sum())}/{len(ct)}: {ct[ct.stalled].trial_id.tolist()}")
for lm in ("gray_mean", "table_gray", "bg_sheet_right_gray", "gray_std", "table_rb"):
    a, b = ct[ct.stalled][lm], ct[~ct.stalled][lm]
    P(f"   {lm:22s} stalled median {a.median():.2f} (n={len(a)}) | not stalled {b.median():.2f} (n={len(b)}) | MWU p={stats.mannwhitneyu(a, b).pvalue:.2f}")
P("   onset vs trial order (Spearman), C_trunc: rho %.2f p %.3f" % stats.spearmanr(ct.trial_id.str[1:].astype(int), ct.onset_s))
P("   stalled-ness by position (P1 trials come first): " + str(ct.groupby("position").stalled.sum().to_dict()))
ct.to_csv(OUT / "ctrunc_stall.csv", index=False, float_format="%.3f")

# ------------------------------------------------------------------ figures
beh_plot = ["speed10", "onset_s", "first_grasp_time_s", "aim_err_x"]
fig, axs = plt.subplots(len(beh_plot), 2, figsize=(10, 3.0 * len(beh_plot)), sharex="col")
for j, lm in enumerate(("table_gray", "bg_sheet_right_gray")):
    for i, bm in enumerate(beh_plot):
        ax = axs[i, j]
        for s in SERIES:
            x = B[B.series == s]
            ok = x[x.success == 1]; ko = x[x.success == 0]
            ax.scatter(ok[lm], ok[bm], s=24, color=COL[s], label=s if (i == 0 and j == 0) else None, lw=0, alpha=0.85)
            ax.scatter(ko[lm], ko[bm], s=26, color=COL[s], marker="x", lw=1.2, alpha=0.85)
        ax.set_ylabel(BEH[bm]); 
        if i == len(beh_plot) - 1: ax.set_xlabel(LIGHT[lm] + " (first frame)")
axs[0, 0].legend(frameon=False, fontsize=8, title="dot = success, x = failure", title_fontsize=8)
fig.suptitle("11/14 trials: behaviour against first-frame brightness", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig4_behavior_vs_brightness.png"); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
ct = ct.sort_values("trial_id")
ax = axs[0]; order = ct.trial_id.str[1:].astype(int)
ax.scatter(order[~ct.stalled], ct.onset_s[~ct.stalled], color="#d03a3a", s=28, alpha=0.5, label="moves normally"); ax.scatter(order[ct.stalled], ct.onset_s[ct.stalled], color="#d03a3a", s=34, label="stands still at the start")
ax.axhline(thr, color="#52514e", ls=":", lw=1); ax.set_xlabel("Trial number (time order)"); ax.set_ylabel("Motion onset (s)"); ax.set_title("C_trunc: onset over the session", fontsize=10); ax.legend(frameon=False, fontsize=8)
ax = axs[1]
ax.scatter(ct.table_gray[~ct.stalled], ct.onset_s[~ct.stalled], color="#d03a3a", s=28, alpha=0.5); ax.scatter(ct.table_gray[ct.stalled], ct.onset_s[ct.stalled], color="#d03a3a", s=34)
ax.set_xlabel("Table gray (first frame)"); ax.set_ylabel("Motion onset (s)"); ax.set_title("C_trunc: onset against brightness", fontsize=10)
ax = axs[2]
for s in SERIES:
    x = B[B.series == s]; ax.scatter(x.tod, x.table_gray, s=14, color=COL[s], label=s, lw=0, alpha=0.85)
ax.set_xlabel("Time of day (h)"); ax.set_ylabel("Table gray (first frame)"); ax.set_title("Brightness of the trials over the afternoon", fontsize=10); ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT / "fig5_ctrunc_stall.png"); plt.close(fig)

#!/usr/bin/env python
"""
Lighting analysis step 2: brightness / colour against time of day, training vs test distributions, and the 11/14 trials' behaviour against the brightness
of their first frame. Reads lighting_analysis/frame_stats_all.csv, retest_analysis/trial_table.csv, trunc_analysis/trial_table.csv, results/*_raw/meta.json.
Writes lighting_analysis/ fig1..fig5, lighting_vs_training.csv, session_trends.csv, trial_lighting_behavior.csv, correlations.csv, step2_log.txt.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analyze_cube_positions as A, analyze_grasp_points as G, analyze_correction as C

OUT = HERE / "lighting_analysis"
plt.rcParams.update({"font.size": 9.5, "savefig.dpi": 160, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
LOG = open(OUT / "step2_log.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n")

SER = {"gray_mean": ("whole frame", "#2a78d6"), "table_gray": ("table (board)", "#eb6834"), "bg_sheet_right_gray": ("window-side background (right)", "#1baf7a")}
TEST_1114 = {"retest 13:45-15:11": ("13:44", "15:11"), "truncation test 16:29-17:22": ("16:29", "17:22")}

def tod(t):
    return t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600

def load():
    d = pd.read_csv(OUT / "frame_stats_all.csv"); d["time"] = pd.to_datetime(d.time, utc=True).dt.tz_convert("Asia/Tokyo")
    d["date"] = d.time.dt.strftime("%m/%d"); d["tod"] = tod(d.time)
    return d

def fig_time(d, cols, title, fname, ylabel):
    dates = sorted(d.date.unique(), key=lambda s: (d[d.date == s].time.min()))
    fig, axs = plt.subplots(1, len(dates), figsize=(3.3 * len(dates), 3.3), sharey=True)
    for ax, dt in zip(axs, dates):
        x = d[d.date == dt]
        for c, (lab, col) in cols.items():
            ax.scatter(x.tod, x[c], s=9, color=col, alpha=0.75, lw=0, label=lab)
        if dt == "11/14":
            for (lab, (a, b)), shade in zip(TEST_1114.items(), ("#b5b4ae", "#8a8984")):
                h = lambda s: int(s[:2]) + int(s[3:]) / 60
                ax.axvspan(h(a), h(b), color=shade, alpha=0.2, lw=0)
                ax.text((h(a) + h(b)) / 2, ax.get_ylim()[1] if False else 0.97, lab.split(" ")[0], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5, color="#52514e")
        ax.set_title(dt, fontsize=10); ax.set_xlabel("Time of day (h)")
        lo, hi = x.tod.min(), x.tod.max(); pad = max(0.15, (hi - lo) * 0.08); ax.set_xlim(lo - pad, hi + pad)
    axs[0].set_ylabel(ylabel); h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), markerscale=2); fig.suptitle(title, y=1.02)
    fig.savefig(OUT / fname); plt.close(fig)

def main():
    d = load()
    # ---------------- 2.1 time plots
    fig_time(d, SER, "Brightness of the first frame against time of day (gray level, 0-255)", "fig1_brightness_vs_time.png", "Gray level")
    fig_time(d, {"table_rb": ("table R/B", "#eb6834"), "bg_sheet_right_rb": ("window-side R/B", "#1baf7a"), "bg_desk_rb": ("white desk R/B", "#2a78d6")}, "Colour balance (R/B ratio of the mean colour) against time of day", "fig2_color_vs_time.png", "R / B")
    # ---------------- 2.2 training vs test
    train = d[d.group.isin(["train_expert50", "train_hil20", "train_newexpert20"])]
    sess = {"expert 50\n7/31": d.group == "train_expert50", "recovery 20\n10/15": d.group == "train_hil20", "new expert 20\n11/12": d.group == "train_newexpert20",
            "eval 6 sets\n11/4": d.session == "eval_1104", "eval control+edge\n11/12": d.session == "eval_1112", "retest\n11/14": d.group == "retest_1114", "truncation\n11/14": d.group == "trunc_1114"}
    mets = {"gray_mean": "Whole-frame gray", "table_gray": "Table gray", "bg_sheet_right_gray": "Window-side gray", "table_rb": "Table R/B", "gray_std": "Contrast (gray sd)", "sat_mean": "Mean saturation"}
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    cols = ["#8a8984"] * 3 + ["#2a78d6"] * 4
    for ax, (m, lab) in zip(axs.ravel(), mets.items()):
        data = [d.loc[mask, m].dropna().to_numpy() for mask in sess.values()]
        bp = ax.boxplot(data, widths=0.55, showfliers=True, flierprops=dict(markersize=2.5, alpha=0.5), patch_artist=True, medianprops=dict(color="#0b0b0b"))
        for p, c in zip(bp["boxes"], cols):
            p.set_facecolor(c); p.set_alpha(0.35); p.set_edgecolor(c)
        ax.set_xticklabels([k.replace("\n", " ") for k in sess], fontsize=7.5, rotation=30, ha="right"); ax.set_title(lab, fontsize=10)
        lo, hi = np.percentile(train[m], [5, 95]); ax.axhspan(lo, hi, color="#b5b4ae", alpha=0.18, lw=0, zorder=0)
    fig.suptitle("Training data (grey) and test sessions (blue): first-frame statistics. Grey band = 5th-95th percentile of all training episodes", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "fig3_distribution_by_session.png"); plt.close(fig)
    rows = []
    for name, mask in sess.items():
        for m in mets:
            lo, hi = np.percentile(train[m], [5, 95]); x = d.loc[mask, m].dropna()
            rows.append(dict(session=name.replace("\n", " "), metric=m, n=len(x), median=x.median(), p5=np.percentile(x, 5), p95=np.percentile(x, 95), train_p5=lo, train_p95=hi,
                             pct_inside_train_5_95=100 * float(((x >= lo) & (x <= hi)).mean()), pct_below=100 * float((x < lo).mean()), pct_above=100 * float((x > hi).mean())))
    R = pd.DataFrame(rows); R.to_csv(OUT / "lighting_vs_training.csv", index=False, float_format="%.3f")
    P("test sessions against the training range (5th-95th percentile of all 90 training frames); % of frames inside / below / above:")
    P(R[R.metric.isin(["gray_mean", "table_gray", "bg_sheet_right_gray", "table_rb"])].pivot(index="session", columns="metric", values="pct_inside_train_5_95").round(0).to_string())
    P(R[R.metric.isin(["gray_mean", "table_gray", "bg_sheet_right_gray", "table_rb"])][["session", "metric", "median", "train_p5", "train_p95", "pct_below", "pct_above"]].round(2).to_string(index=False))
    # ---------------- trends within sessions
    tr = []
    for g in ["train_expert50", "train_hil20", "train_newexpert20", "eval_mixed_smooth", "eval_mixed_default", "eval_expert_smooth", "eval_expert_default", "eval_recovery_default", "eval_recovery_smooth",
              "eval_expert70_smooth", "edge_mixed", "edge_control", "retest_1114", "trunc_1114"]:
        x = d[d.group == g].sort_values("time")
        for m in ["gray_mean", "table_gray", "bg_sheet_right_gray", "table_rb", "gray_std"]:
            h = (x.time - x.time.min()).dt.total_seconds() / 3600
            if len(x) < 5 or h.max() == 0: continue
            sl = stats.linregress(h, x[m]); half = len(x) // 2
            tr.append(dict(group=g, metric=m, n=len(x), span_h=h.max(), slope_per_h=sl.slope, p=sl.pvalue, first_half=x[m].iloc[:half].mean(), second_half=x[m].iloc[half:].mean(), sd=x[m].std()))
    T = pd.DataFrame(tr); T.to_csv(OUT / "session_trends.csv", index=False, float_format="%.4f")
    P("\nwithin-session trends (linear fit against time; sd = spread across the session):")
    P(T[T.metric.isin(["gray_mean", "table_gray", "bg_sheet_right_gray", "table_rb"])].pivot(index="group", columns="metric", values="slope_per_h").round(2).to_string())
    P(T[T.group.isin(["retest_1114", "trunc_1114"])][["group", "metric", "slope_per_h", "p", "first_half", "second_half", "sd"]].round(3).to_string(index=False))
    # ---------------- 2.3 behaviour on 11/14
    def trial_states(meta_json, folder):
        meta = json.loads(Path(meta_json).read_text()); name = meta["dataset_repo_id"].split("/", 1)[1]
        A.DATASETS["_t"] = f"{folder}/{name}"; g, f = G.grasp_table("_t"); e = f.sort_values("frame_index")
        return e[[f"state:{j}" for j in G.ARM_JOINTS]].to_numpy(float)
    def onset_s(q):
        sp = np.r_[0.0, np.linalg.norm(np.diff(q, axis=0), axis=1) * 30]; sm = C.moving_average(sp, 5); run = 0
        for t, a in enumerate(sm > 15):
            run = run + 1 if a else 0
            if run >= 10: return (t - 9) / 30
        return np.nan
    rows = []
    for sessn, tab, folder in (("retest", "retest_analysis/trial_table.csv", "rollout_retest"), ("trunc", "trunc_analysis/trial_table.csv", "rollout_trunc")):
        t = pd.read_csv(HERE / tab)
        for r in t.itertuples():
            q = trial_states(HERE / f"results/{'retest_raw' if sessn == 'retest' else 'trunc_raw'}/{r.trial_id}_a{r.attempt}/meta.json", folder)
            sp = np.r_[0.0, np.linalg.norm(np.diff(q, axis=0), axis=1) * 30]
            rows.append(dict(session=sessn, trial_id=r.trial_id, attempt=r.attempt, position=r.position_id, model=r.model, success=r.success, first_grasp_time_s=r.first_grasp_time_s,
                             aim_err_x=r.grasp_aim_err_x, aim_err_y=r.grasp_aim_err_y, speed10=float(C.moving_average(sp, 5)[:300].mean()), onset_s=onset_s(q), n_closes=r.n_gripper_closes))
    B = pd.DataFrame(rows)
    fs = d[d.group.isin(["retest_1114", "trunc_1114"])][["trial_id", "attempt", "time", "tod", "gray_mean", "table_gray", "bg_sheet_right_gray", "table_rb", "gray_std", "clipped_bright_frac"]]
    B = B.merge(fs, on=["trial_id", "attempt"], how="left")
    B["series"] = np.where(B.session == "retest", "retest " + B.model, "trunc session " + B.model)
    B.to_csv(OUT / "trial_lighting_behavior.csv", index=False, float_format="%.4f")
    P("\n11/14 trials with lighting:", len(B), "| missing lighting rows:", int(B.gray_mean.isna().sum()))
    # stalled at the start: onset of arm motion later than 4 s
    P("motion onset (s) by series: " + str(B.groupby("series").onset_s.describe()[["count", "min", "50%", "max"]].round(2).to_dict("index")))
    return d, B, T

if __name__ == "__main__":
    d, B, T = main()

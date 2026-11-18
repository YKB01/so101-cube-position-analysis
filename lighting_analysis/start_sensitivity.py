#!/usr/bin/env python
"""
Lighting analysis step 3b: does a small brightness change flip the policy's behaviour at the START of an episode (the C_trunc "stands still" trials)?
For every 11/14 truncation-session trial (the model that was actually running: C_trunc or C) the first frame and the frames at 1 s and 2 s are fed to that model; we
record the predicted arm displacement 1 s ahead (norm of chunk[30,:5] - current state) and compare it with the trial's brightness and its actual motion onset.
Then the first frame is scaled by x0.95 ... x1.05 to see how much brightness alone moves the predicted displacement.
Writes start_sensitivity.csv, start_sensitivity_summary.txt, fig7_start_sensitivity.png.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lerobot.datasets import LeRobotDataset
from lerobot.policies.act import ACTPolicy
from lerobot.policies import make_pre_post_processors
OUT = HERE / "lighting_analysis"; CK = Path.home() / "Checkpoints"; ROOT = Path.home() / ".cache/huggingface/lerobot/YKB0123"
MODELS = {"C": CK / "act_red_cube_expert50_expert20_ft5000/checkpoints/005000/pretrained_model", "C_trunc": CK / "act_red_cube_expert50_trunc20_ft5000/checkpoints/005000/pretrained_model"}
FACT = (0.95, 0.97, 1.0, 1.03, 1.05)
LOG = open(OUT / "start_sensitivity_summary.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n")

B = pd.read_csv(OUT / "trial_lighting_behavior.csv"); B = B[B.session == "trunc"].copy()
rows = []
pol_cache = {}
for mname, path in MODELS.items():
    pol = ACTPolicy.from_pretrained(str(path)).to("cuda").eval()
    pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=str(path), preprocessor_overrides={"device_processor": {"device": "cuda"}})
    def chunk(smp, im):
        obs = {k: smp[k] for k in pol.config.input_features if k in smp}; obs["observation.images.front"] = im
        with torch.inference_mode():
            ch = post(pol.predict_action_chunk(pre(obs)))
        return (ch["action"] if isinstance(ch, dict) else ch).squeeze(0).cpu().numpy()
    for r in B[B.model == mname].itertuples():
        meta = json.loads((HERE / f"results/trunc_raw/{r.trial_id}_a{r.attempt}/meta.json").read_text()); name = meta["dataset_repo_id"].split("/", 1)[1]
        ds = LeRobotDataset("YKB0123/" + name, root=ROOT / "rollout_trunc" / name)
        for fr in (0, 30, 60):
            smp = ds[fr]; st = np.asarray(smp["observation.state"])[:5]; im0 = smp["observation.images.front"]
            for f in FACT:
                ch = chunk(smp, (im0 * f).clamp(0, 1))
                rows.append(dict(model=mname, trial=r.trial_id, frame=fr, factor=f, disp30=float(np.linalg.norm(ch[30, :5] - st)), disp60=float(np.linalg.norm(ch[60, :5] - st)),
                                 gripper30=float(ch[30, 5]), onset_s=r.onset_s, table_gray=r.table_gray, gray_mean=r.gray_mean, sheet=r.bg_sheet_right_gray, position=r.position, series=r.series))
    print(mname, "done", flush=True); del pol; torch.cuda.empty_cache()
D = pd.DataFrame(rows); D.to_csv(OUT / "start_sensitivity.csv", index=False, float_format="%.4f")
base = D[(D.factor == 1.0)]
P("predicted arm displacement 1 s ahead from the untouched frame (joint units, norm over 5 joints), by model and frame:")
P(base.groupby(["model", "frame"]).disp30.describe()[["count", "mean", "min", "50%", "max"]].round(1).to_string())
for mname in MODELS:
    x = base[(base.model == mname) & (base.frame == 0)]
    for c in ("table_gray", "gray_mean", "sheet"):
        r, p = stats.spearmanr(x[c], x.disp30); r2, p2 = stats.spearmanr(x[c], x.onset_s)
        P(f"{mname}: frame 0 predicted displacement vs {c}: rho {r:+.2f} (p {p:.3f}); actual onset vs {c}: rho {r2:+.2f} (p {p2:.3f}); n={len(x)}")
    r, p = stats.spearmanr(x.disp30, x.onset_s); P(f"{mname}: predicted displacement at frame 0 vs actual onset: rho {r:+.2f} (p {p:.3f})")
# within-trial effect of the brightness factor on the predicted displacement
w = D[D.frame == 0].pivot_table(index=["model", "trial"], columns="factor", values="disp30")
P("\nchange of the predicted displacement (frame 0) when the same image is scaled (mean over trials, joint units):")
for mname in MODELS:
    ww = w.loc[mname]; P(f"  {mname}: " + ", ".join(f"x{f}: {(ww[f] - ww[1.0]).mean():+.2f} (sd {(ww[f] - ww[1.0]).std():.2f})" for f in FACT if f != 1.0) + f" | between-trial sd of the baseline: {ww[1.0].std():.2f}")
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
col = {"C_trunc": "#d03a3a", "C": "#2a78d6"}
ax = axs[0]
for mname in MODELS:
    x = base[(base.model == mname) & (base.frame == 0)]; ax.scatter(x.table_gray, x.disp30, color=col[mname], s=28, label=mname, lw=0, alpha=0.85)
ax.set_xlabel("Table gray (first frame)"); ax.set_ylabel("Predicted displacement 1 s ahead (frame 0)"); ax.legend(frameon=False); ax.set_title("Prediction at the start against brightness", fontsize=10)
ax = axs[1]
for mname in MODELS:
    x = base[(base.model == mname) & (base.frame == 0)]; ax.scatter(x.disp30, x.onset_s, color=col[mname], s=28, lw=0, alpha=0.85)
ax.set_xlabel("Predicted displacement 1 s ahead (frame 0)"); ax.set_ylabel("Actual motion onset (s)"); ax.set_title("Prediction against what the robot did", fontsize=10)
ax = axs[2]
for mname in MODELS:
    ww = w.loc[mname]
    for f_ in FACT:
        pass
    m = [(ww[f] - ww[1.0]).mean() for f in FACT]; s_ = [(ww[f] - ww[1.0]).std() for f in FACT]
    ax.errorbar(FACT, m, yerr=s_, color=col[mname], marker="o", ms=4, lw=1.2, capsize=2, label=mname)
ax.axhline(0, color="#b5b4ae", lw=0.8); ax.set_xlabel("Brightness factor applied to the first frame"); ax.set_ylabel("Change of predicted displacement"); ax.set_title("Same image, scaled (mean +- sd over trials)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig7_start_sensitivity.png"); plt.close(fig)

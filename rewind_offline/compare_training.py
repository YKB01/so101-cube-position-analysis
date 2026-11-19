#!/usr/bin/env python
"""Training comparison of the four models (deep, C, R, C_trunc): train_config.json diff and the loss curves from the training logs.
Writes rewind_offline/training_comparison.txt, training_loss_curves.csv, fig_training_loss.png"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parents[1]; OUT = HERE / "rewind_offline"; CK = Path.home() / "Checkpoints"
M = {"deep": "act_red_cube_expert50_rewind_deep20_ft5000", "C": "act_red_cube_expert50_expert20_ft5000", "R": "act_red_cube_mixed_hil_ft_20261015_150904", "C_trunc": "act_red_cube_expert50_trunc20_ft5000"}
lines = []
def say(s):
    print(s); lines.append(s)
def flat(d, pre=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict): o.update(flat(v, pre + k + "."))
        else: o[pre + k] = v
    return o
cfg = {k: flat(json.loads((CK / v / "checkpoints/005000/pretrained_model/train_config.json").read_text())) for k, v in M.items()}
keys = sorted(set().union(*[set(c) for c in cfg.values()]))
diff = [k for k in keys if len({str(cfg[m].get(k)) for m in cfg}) > 1]
say(f"train_config.json: {len(keys)} keys compared across deep, C, R, C_trunc; differing keys: {diff}")
say("the only differences are the dataset, the job name and the output directory; steps {}, batch size {}, seed {}, lr {}, backbone lr {}, weight decay {}, grad clip {}, num_workers {}, log_freq {}".format(
    *[cfg['deep'][k] for k in ('steps', 'batch_size', 'seed', 'policy.optimizer_lr', 'policy.optimizer_lr_backbone', 'policy.optimizer_weight_decay', 'optimizer.grad_clip_norm', 'num_workers', 'log_freq')]))
pc = {k: json.loads((CK / v / "checkpoints/005000/pretrained_model/config.json").read_text()) for k, v in M.items()}
say(f"policy config.json identical in all four models: {all(pc[k] == pc['deep'] for k in pc)}; pretrained_path {pc['deep']['pretrained_path']}")
curves = {}
for k, v in M.items():
    txt = (CK / "training_logs" / f"{v}.log").read_bytes().decode(errors="ignore")
    rows = re.findall(r"step:([0-9.]+[KM]?) smpl:\S+ ep:\S+ epch:\S+ loss:([0-9.]+) grdn:([0-9.]+) lr:", txt)
    # the log prints the step rounded ("1K"), so the step is taken from the order: one entry every log_freq = 200 steps
    curves[k] = pd.DataFrame([(200 * (i + 1), float(b), float(c)) for i, (a, b, c) in enumerate(rows)], columns=["step", "loss", "grad_norm"])
C = pd.concat([d.assign(model=k) for k, d in curves.items()]); C.to_csv(OUT / "training_loss_curves.csv", index=False)
P = C.pivot(index="step", columns="model", values="loss")[list(M)]
say("\nloss at logged steps (every 200 steps), steps 200..5000:\n" + P.round(3).to_string())
say("\nsummary: final loss (step 5000) " + ", ".join(f"{k} {P[k].iloc[-1]:.3f}" for k in M) + "; mean of the last 5 logged values " + ", ".join(f"{k} {P[k].iloc[-5:].mean():.3f}" for k in M)
    + "; mean over steps 200-1000 " + ", ".join(f"{k} {P[k].loc[200:1000].mean():.3f}" for k in M))
say("number of logged points: " + str({k: len(d) for k, d in curves.items()}))
(OUT / "training_comparison.txt").write_text("\n".join(lines) + "\n")
plt.rcParams.update({"font.size": 9.5, "savefig.dpi": 160, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
col = {"deep": "#1baf7a", "C": "#6250d6", "R": "#eb6834", "C_trunc": "#d03a3a"}
fig, axs = plt.subplots(1, 2, figsize=(12, 3.9))
for k in M:
    axs[0].plot(P.index, P[k], color=col[k], lw=1.6, label=k)
    g = curves[k]; axs[1].plot(g.step, g.grad_norm, color=col[k], lw=1.4)
axs[0].set_xlabel("Step"); axs[0].set_ylabel("Training loss (logged every 200 steps)"); axs[0].legend(frameon=False); axs[0].set_title("Loss", fontsize=10)
axs[1].set_xlabel("Step"); axs[1].set_ylabel("Gradient norm"); axs[1].set_title("Gradient norm", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_training_loss.png"); plt.close(fig)

#!/usr/bin/env python
"""
Offline prediction check (supports the interpretation of the C_trunc test; not part of the pre-registered tests).

For the 20 new expert demonstrations, each model predicts an action chunk from the recorded observation at several
frames; the chunk is compared with the recorded actions at chunk steps 0/10/20/30/60 (mean abs error over the 5 arm
joints, joint units) and the predicted displacement over the first 30 chunk steps is compared with the recorded one.
Phases are relative to the path-aligned truncation point (trunc_control/truncation_plan.csv):
  before cut = frames the C_trunc training data does NOT contain (the removed early approach),
  after cut  = frames C_trunc was trained on.
Caveat: model C was trained on all of these frames (in-sample); R and C_trunc never saw the "before cut" frames of
these episodes (R never saw these episodes at all).
Writes retest_analysis/offline_prediction_check.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from pathlib import Path
from lerobot.datasets import LeRobotDataset
from lerobot.policies.act import ACTPolicy
from lerobot.policies import make_pre_post_processors

HERE = Path(__file__).resolve().parents[1]
CK = Path.home() / "Checkpoints"
MODELS = {
    "R": CK / "act_red_cube_mixed_hil_ft_20261015_150904/checkpoints/005000/pretrained_model",
    "C": CK / "act_red_cube_expert50_expert20_ft5000/checkpoints/005000/pretrained_model",
    "C_trunc": CK / "act_red_cube_expert50_trunc20_ft5000/checkpoints/005000/pretrained_model",
}
IDX = [0, 10, 20, 30, 60]


def run(name, path, ds, plan):
    pol = ACTPolicy.from_pretrained(str(path)).to("cuda").eval()
    pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=str(path),
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    ef = ds.meta.episodes["dataset_from_index"]
    rows = []
    for ep in range(20):
        base, cut = int(ef[ep]), int(plan[ep])
        for t in (0, 60, cut - 30, cut + 30):
            if t < 0:
                continue
            s = ds[base + t]
            obs = {k: s[k] for k in pol.config.input_features if k in s}
            with torch.inference_mode():
                ch = post(pol.predict_action_chunk(pre(obs)))
            ch = (ch["action"] if isinstance(ch, dict) else ch).squeeze(0).cpu().numpy()
            gt = np.stack([np.asarray(ds[base + min(t + k, len(ds.hf_dataset) - 1)]["action"]) for k in IDX])
            rows.append(dict(model=name, episode=ep, t=t,
                             phase="before cut" if t < cut - 10 else "after cut" if t >= cut else "near cut",
                             err=np.abs(ch[IDX] - gt)[:, :5].mean(),
                             disp_ratio=np.linalg.norm(ch[30, :5] - ch[0, :5]) / (np.linalg.norm(gt[3, :5] - gt[0, :5]) + 1e-6)))
    del pol; torch.cuda.empty_cache()
    return rows


def main():
    ds = LeRobotDataset("YKB0123/so101_red_cube_plastic_box_expert_20_20261112_130439")
    plan = pd.read_csv(HERE / "trunc_control/truncation_plan.csv").set_index("new_episode").path_cut_frame
    rows = []
    for n, p in MODELS.items():
        rows += run(n, p, ds, plan)
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "retest_analysis/offline_prediction_check.csv", index=False, float_format="%.4f")
    print(d.groupby(["phase", "model"]).agg(err=("err", "mean"), n=("err", "size"),
                                              disp_ratio_median=("disp_ratio", "median")).round(2).to_string())


if __name__ == "__main__":
    main()

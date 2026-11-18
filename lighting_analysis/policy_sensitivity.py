#!/usr/bin/env python
"""
Lighting analysis step 3: sensitivity of the policies to image brightness / colour temperature (offline, no robot).
From the 11/14 retest, for each position the first successful trial of model C and of model R; frames 1, 2, 3 s after the arm starts to move. Each image is
transformed (brightness x0.7, 0.85, 0.95, 1.05, 1.15, 1.3; colour: R +10% / B -10% ("warm") and R -10% / B +10% ("cool")) and fed to the four policies;
the 100-step action chunk is compared with the chunk of the untouched image: mean absolute joint difference over the first 30 steps (5 arm joints),
and the shift of the aim point (expert_regression of the arm pose at the first predicted close, else at the chunk's last step) in pixels.
Writes policy_sensitivity.csv (every case), policy_sensitivity_summary.csv, fig6_policy_sensitivity.png.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import analyze_cube_positions as A, analyze_grasp_points as G, analyze_correction as C
from lerobot.datasets import LeRobotDataset
from lerobot.policies.act import ACTPolicy
from lerobot.policies import make_pre_post_processors

OUT = HERE / "lighting_analysis"; CK = Path.home() / "Checkpoints"; ROOT = Path.home() / ".cache/huggingface/lerobot/YKB0123"
MODELS = {"C": CK / "act_red_cube_expert50_expert20_ft5000/checkpoints/005000/pretrained_model",
          "R": CK / "act_red_cube_mixed_hil_ft_20261015_150904/checkpoints/005000/pretrained_model",
          "C_trunc": CK / "act_red_cube_expert50_trunc20_ft5000/checkpoints/005000/pretrained_model",
          "baseline": CK / "act_red_cube_50_100000"}
TRANSFORMS = {"brightness x0.70": ("b", 0.70), "brightness x0.85": ("b", 0.85), "brightness x0.95": ("b", 0.95), "brightness x1.05": ("b", 1.05),
              "brightness x1.15": ("b", 1.15), "brightness x1.30": ("b", 1.30), "warm (R+10%, B-10%)": ("c", 0.10), "cool (R-10%, B+10%)": ("c", -0.10)}
AFTER_ONSET_S = (1, 2, 3)

def apply(im, kind, v):
    im = im.clone()
    if kind == "b":
        im = im * v
    else:
        im[0] = im[0] * (1 + v); im[2] = im[2] * (1 - v)
    return im.clamp(0, 1)

def onset_frame(q):
    sp = np.r_[0.0, np.linalg.norm(np.diff(q, axis=0), axis=1) * 30]; sm = C.moving_average(sp, 5); run = 0
    for t, a in enumerate(sm > 15):
        run = run + 1 if a else 0
        if run >= 10: return t - 9
    return None

def main():
    tt = pd.read_csv(HERE / "retest_analysis/trial_table.csv")
    sel = []
    for pid in ("P1", "P2", "P3"):
        for mdl in ("C", "R"):
            r = tt[(tt.position_id == pid) & (tt.model == mdl) & (tt.success == 1)].sort_values("trial_id").iloc[0]
            sel.append(r)
    cases = []
    for r in sel:
        meta = json.loads((HERE / f"results/retest_raw/{r.trial_id}_a{r.attempt}/meta.json").read_text()); name = meta["dataset_repo_id"].split("/", 1)[1]
        ds = LeRobotDataset("YKB0123/" + name, root=ROOT / "rollout_retest" / name)
        A.DATASETS["_t"] = f"rollout_retest/{name}"; _, f = G.grasp_table("_t"); q = f.sort_values("frame_index")[[f"state:{j}" for j in G.ARM_JOINTS]].to_numpy(float)
        o = onset_frame(q)
        for s in AFTER_ONSET_S:
            i = o + 30 * s; smp = ds[i]
            cases.append(dict(trial=r.trial_id, position=r.position_id, src_model=r.model, sec=s, sample=smp))
    print(f"{len(cases)} frames from {len(sel)} trials: {[(r.trial_id, r.model) for r in sel]}", flush=True)
    reg, px = C.regression_model()
    rows = []
    for mname, path in MODELS.items():
        pol = ACTPolicy.from_pretrained(str(path)).to("cuda").eval()
        pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=str(path), preprocessor_overrides={"device_processor": {"device": "cuda"}})
        def chunk(smp, im):
            obs = {k: smp[k] for k in pol.config.input_features if k in smp}; obs["observation.images.front"] = im
            with torch.inference_mode():
                ch = post(pol.predict_action_chunk(pre(obs)))
            return (ch["action"] if isinstance(ch, dict) else ch).squeeze(0).cpu().numpy()
        def aim(ch):
            close = np.flatnonzero(ch[:, 5] < 15)
            k = int(close[0]) if len(close) else len(ch) - 1
            return reg.predict(ch[k:k + 1, :5])[0], bool(len(close))
        for c in cases:
            im0 = c["sample"]["observation.images.front"]; base = chunk(c["sample"], im0); a0, found0 = aim(base)
            for tname, (kind, v) in TRANSFORMS.items():
                ch = chunk(c["sample"], apply(im0, kind, v)); a1, found1 = aim(ch)
                rows.append(dict(model=mname, trial=c["trial"], position=c["position"], src_model=c["src_model"], sec=c["sec"], transform=tname,
                                 joint_diff_30=float(np.abs(ch[:30, :5] - base[:30, :5]).mean()), gripper_diff_30=float(np.abs(ch[:30, 5] - base[:30, 5]).mean()),
                                 aim_shift_px=float(np.hypot(*(a1 - a0))), aim_dx=float(a1[0] - a0[0]), aim_dy=float(a1[1] - a0[1]), close_in_chunk=found0 and found1))
        print(mname, "done", flush=True); del pol; torch.cuda.empty_cache()
    D = pd.DataFrame(rows); D.to_csv(OUT / "policy_sensitivity.csv", index=False, float_format="%.4f")
    S = D.groupby(["transform", "model"]).agg(joint_diff_deg=("joint_diff_30", "mean"), gripper_diff=("gripper_diff_30", "mean"), aim_shift_px=("aim_shift_px", "mean"),
                                              aim_shift_px_max=("aim_shift_px", "max"), n=("joint_diff_30", "size")).reset_index()
    S.to_csv(OUT / "policy_sensitivity_summary.csv", index=False, float_format="%.3f")
    print(S.pivot(index="transform", columns="model", values="joint_diff_deg").round(2).to_string())
    print(S.pivot(index="transform", columns="model", values="aim_shift_px").round(1).to_string())
    print("overall mean joint diff by model:", D.groupby("model").joint_diff_30.mean().round(2).to_dict(), "| aim shift:", D.groupby("model").aim_shift_px.mean().round(1).to_dict())

if __name__ == "__main__":
    main()

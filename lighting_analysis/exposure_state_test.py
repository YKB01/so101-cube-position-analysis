#!/usr/bin/env python
"""
Lighting analysis step 3c: the first-frame brightness of the 11/14 trials falls into two clusters (table gray ~115 and ~120+): the camera's automatic exposure
settles in different states from one start to the next. Do the real differences between the two states (not just a global scale) change what the policy does at the start?
1. median first frame of the bright state and of the dark state (truncation session, one image per trial); per-channel cubic mapping dark -> bright and bright -> dark
   fitted on all pixels of the two median images;
2. each first frame of the truncation session is mapped to the OTHER state and fed to the model that was running (C_trunc / C): change of the predicted arm
   displacement 1 s ahead (norm of chunk[30,:5] - state), in joint units.
Writes exposure_state_test.csv and exposure_state_summary.txt.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
from lerobot.datasets import LeRobotDataset
from lerobot.policies.act import ACTPolicy
from lerobot.policies import make_pre_post_processors
OUT = HERE / "lighting_analysis"; CK = Path.home() / "Checkpoints"; ROOT = Path.home() / ".cache/huggingface/lerobot/YKB0123"
MODELS = {"C": CK / "act_red_cube_expert50_expert20_ft5000/checkpoints/005000/pretrained_model", "C_trunc": CK / "act_red_cube_expert50_trunc20_ft5000/checkpoints/005000/pretrained_model"}
LOG = open(OUT / "exposure_state_summary.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n")

B = pd.read_csv(OUT / "trial_lighting_behavior.csv"); B = B[B.session == "trunc"].copy()
idx = pd.read_csv(OUT / "episodes_index.csv"); F = np.load(OUT / "cache/first_frames.npy", mmap_mode="r")
row = {(r.trial_id, r.attempt): i for i, r in enumerate(idx.itertuples()) if r.group == "trunc_1114"}
B["fi"] = [row[(t, a)] for t, a in zip(B.trial_id, B.attempt)]
THR = 117.5
B["state"] = np.where(B.table_gray >= THR, "bright", "dark")
P(f"table gray split at {THR}: {B.state.value_counts().to_dict()}; stalled-at-start (onset > 4.77 s) by state: {B.assign(st=B.onset_s > 4.77).groupby(['model', 'state']).st.agg(['sum', 'count']).to_dict('index')}")
med = {s: np.median(np.stack([np.asarray(F[i], float) for i in B[B.state == s].fi]), axis=0) for s in ("dark", "bright")}
def fit(src, dst):
    coefs = []
    for c in range(3):
        x, y = src[..., c].ravel() / 255.0, dst[..., c].ravel() / 255.0
        coefs.append(np.polyfit(x, y, 3))
    return coefs
def apply_map(im, coefs):
    out = np.empty_like(im)
    for c in range(3):
        out[c] = np.polyval(coefs[c], im[c].numpy())
    return torch.from_numpy(out).clamp(0, 1)
m_db, m_bd = fit(med["dark"], med["bright"]), fit(med["bright"], med["dark"])
x = np.linspace(0, 1, 5)
P("fitted mapping dark -> bright at input levels 0, .25, .5, .75, 1 (per channel R/G/B): " + "; ".join(str(np.round(np.polyval(m_db[c], x), 3).tolist()) for c in range(3)))
rows = []
for mname, path in MODELS.items():
    pol = ACTPolicy.from_pretrained(str(path)).to("cuda").eval()
    pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=str(path), preprocessor_overrides={"device_processor": {"device": "cuda"}})
    def disp(smp, im):
        obs = {k: smp[k] for k in pol.config.input_features if k in smp}; obs["observation.images.front"] = im
        with torch.inference_mode():
            ch = post(pol.predict_action_chunk(pre(obs)))
        ch = (ch["action"] if isinstance(ch, dict) else ch).squeeze(0).cpu().numpy()
        return float(np.linalg.norm(ch[30, :5] - np.asarray(smp["observation.state"])[:5]))
    for r in B[B.model == mname].itertuples():
        meta = json.loads((HERE / f"results/trunc_raw/{r.trial_id}_a{r.attempt}/meta.json").read_text()); name = meta["dataset_repo_id"].split("/", 1)[1]
        ds = LeRobotDataset("YKB0123/" + name, root=ROOT / "rollout_trunc" / name)
        smp = ds[0]; im = smp["observation.images.front"]
        d0 = disp(smp, im)
        d1 = disp(smp, apply_map(im, m_bd if r.state == "bright" else m_db))
        rows.append(dict(model=mname, trial=r.trial_id, state=r.state, onset_s=r.onset_s, table_gray=r.table_gray, disp_orig=d0, disp_other_state=d1, change=d1 - d0))
    print(mname, "done", flush=True); del pol; torch.cuda.empty_cache()
D = pd.DataFrame(rows); D.to_csv(OUT / "exposure_state_test.csv", index=False, float_format="%.4f")
P("\npredicted arm displacement 1 s ahead (frame 0): original image vs the same image mapped to the other exposure state")
P(D.groupby(["model", "state"]).agg(n=("change", "size"), orig=("disp_orig", "mean"), other=("disp_other_state", "mean"), change=("change", "mean"), change_sd=("change", "std")).round(2).to_string())
for mname in MODELS:
    x_ = D[D.model == mname]; r, p = stats.spearmanr(x_.disp_orig, x_.onset_s); P(f"{mname}: predicted displacement (original) vs actual onset: rho {r:+.2f} (p {p:.3f})")

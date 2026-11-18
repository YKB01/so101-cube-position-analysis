#!/usr/bin/env python
"""Software test of --policy_only with a simulated follower/camera and a fake policy engine that replays a recorded trajectory."""
import sys, json, tempfile, types
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "rewind_offline"))
import collect_rewind as CR, test_collect_dry_fake as T
from test_collect_full_fake import CamRobot

class FakeEngine:
    def __init__(self, Q): self.Q, self.i = Q, 0
    def reset(self): self.i = 0
    def start(self): pass
    def get_action(self, obs_frame):
        a = self.Q[min(self.i, len(self.Q) - 1)]; self.i += 1
        return torch.tensor(a, dtype=torch.float32)

def main():
    HIL = Path.home() / ".cache/huggingface/lerobot/YKB0123/rollout_so101_red_cube_hil_front_20261015_134746"
    info = json.loads((HIL / "meta/info.json").read_text())
    feats = {k: v for k, v in info["features"].items() if k in ("action", "observation.state", "observation.images.front")}
    cfg = CR.load_config(["--policy_only", "--policy_path", "x", "--fps", "60", "--policy_only_episode_s", "6", "--compare_model", "C",
                          "--out_dir", tempfile.mkdtemp(prefix="fake_policy_")])
    col = CR.Collector(cfg)
    Q = CR.load_trial_states("E12")
    col.robot = CamRobot(Q[0]); col.start_pose6 = Q[0].copy(); col.last_cmd6 = Q[0].copy()
    from lerobot.utils.action_interpolator import ActionInterpolator
    col.interp = ActionInterpolator(multiplier=1)
    col.engine = FakeEngine(Q)
    pol = types.SimpleNamespace(config=types.SimpleNamespace(type="act", n_action_steps=1, temporal_ensemble_coeff=0.01, chunk_size=100, n_obs_steps=1, use_amp=False), temporal_ensembler=object())
    col.rcfg = types.SimpleNamespace(device="cuda", inference=types.SimpleNamespace(type="sync"), interpolation_multiplier=1, fps=30.0)
    col.processors = types.SimpleNamespace(robot_action_processor=lambda x: x[0], robot_observation_processor=lambda o: o, teleop_action_processor=lambda x: x[0])
    col.ctx = types.SimpleNamespace(policy=types.SimpleNamespace(policy=pol, inference=col.engine), processors=col.processors,
                                    data=types.SimpleNamespace(dataset_features=feats, ordered_action_keys=CR.KEYS6),
                                    hardware=types.SimpleNamespace(robot_wrapper=col.robot))
    col.features = feats; col.ds = None
    col.keys = T.FakeKeys({3: ["s"], 3000: ["q"]})
    col.run_policy_only()
    print(open(col.out / "policy_only_summary.txt").read())
    print("files:", sorted(p.name for p in col.out.iterdir()))

if __name__ == "__main__":
    main()

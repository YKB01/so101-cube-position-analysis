#!/usr/bin/env python
"""Software test of the full flow of collect_rewind.py after the trigger (select -> rewind -> align -> wait for start -> human recording ->
save) with simulated follower, leader and camera. Writes to a temporary directory."""
import json, sys, shutil, tempfile, types, csv
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "rewind_offline"))
import collect_rewind as CR
import test_collect_dry_fake as T
import rewind_offline.step2_offline_targets as S2

HIL = Path.home() / ".cache/huggingface/lerobot/YKB0123/rollout_so101_red_cube_hil_front_20261015_134746"

class CamRobot(T.FakeRobot):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.cameras = {"front": 1}; self.name = "so_follower"
        img = np.full((480, 640, 3), 90, np.uint8); img[250:268, 300:318] = (230, 20, 20); self.img = img
    def get_observation(self):
        o = super().get_observation(); o["front"] = self.img; return o

class FakeTeleop:
    def __init__(self, pos6): self.pos = np.array(pos6, float); self.goal = self.pos.copy(); self.torque = False; self.fb = []
    def get_action(self):
        if self.torque: self.pos += 0.5 * (self.goal - self.pos)
        else: self.pos += 0.0
        return {k: float(v) for k, v in zip(CR.KEYS6, self.pos)}
    def send_feedback(self, d): self.goal = np.array([d[k] for k in CR.KEYS6], float)
    def enable_torque(self): self.torque = True
    def disable_torque(self): self.torque = False

def main(mode="deep"):
    info = json.loads((HIL / "meta/info.json").read_text())
    feats = {k: v for k, v in info["features"].items() if k in ("action", "observation.state", "observation.images.front")}
    tmp = Path(tempfile.mkdtemp(prefix="rewind_fake_"))
    cfg = CR.load_config(["--rewind_mode", mode, "--repo_id", "YKB0123/test_rewind_fake2", "--policy_path", "x", "--out_dir", str(tmp / "out"), "--fps", "30",
                          "--root", str(tmp / "main"), "--root_rewind_seg", str(tmp / "seg")])
    col = CR.Collector(cfg)
    col.ctx = types.SimpleNamespace(data=types.SimpleNamespace(dataset_features=feats))
    trial = [t for t in S2.load_trials() if t["trial"] == "E12"][0]
    Q = trial["states"]; trig = 306
    col.robot = CamRobot(Q[trig]); col.start_pose6 = Q[0].copy(); col.last_cmd6 = Q[trig].copy()
    col.teleop = FakeTeleop(Q[trig] + 20)
    col.processors = types.SimpleNamespace(robot_action_processor=lambda x: x[0], robot_observation_processor=lambda o: o, teleop_action_processor=lambda x: x[0])
    col._make_datasets()
    col.fps, col.dt = 600.0, 1 / 600.0          # run the loops 20x faster than real time (the datasets were created at 30 fps)
    col.buffer = CR.RC.StateBuffer(15, 30)
    for s in Q[max(0, trig - 450): trig + 1]: col.buffer.append(s)
    col.keys = T.FakeKeys({4000: ["s"], 4600: ["e"]})
    meta = dict(episode_id=1, rewind_mode=mode, discarded=False, trigger_frame=trig, trigger_time_s=trig / 30, trigger_type="manual")
    ed = tmp / "out/ep1"
    col.triggered(meta, ed)
    plan = col.select(meta)
    col.rewind(meta, plan)
    col.align(meta, ed)
    col.wait_start(meta)
    col.human(meta)
    col.write_meta(meta)
    print("\nmeta:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta.items() if k in ("level", "segment_episode_index", "saved_episode_index", "settled", "final_err_norm", "align_err_max_deg", "align_retries", "cube_before_x", "cube_after_x", "cube_shift_px", "human_s")})
    col.stack.close()
    from lerobot.datasets import LeRobotDataset
    for repo, root, interv in (("YKB0123/test_rewind_fake2", tmp / "main", True), ("YKB0123/test_rewind_fake2_rewind_seg", tmp / "seg", False)):
        d = LeRobotDataset(repo, root=root); s = d[0]
        print(repo, "episodes", d.meta.total_episodes, "frames", d.meta.total_frames, "intervention as expected", bool(s["intervention"].item()) == interv)
    print("sidecar:", (tmp / "seg/rewind_segment_meta.csv").read_text().strip().replace("\n", " | "))
    print("leader torque after human:", col.teleop.torque)
    shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "deep")

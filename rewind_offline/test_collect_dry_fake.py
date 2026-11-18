#!/usr/bin/env python
"""Software test of the dry-run state machine of collect_rewind.py with a simulated follower (no hardware).
Usage: python rewind_offline/test_collect_dry_fake.py [trial] [shallow|deep] -> prints the dry-run report and the saved meta row."""
import sys, json, time, csv
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import collect_rewind as CR

class FakeRobot:
    def __init__(self, start6, alpha=0.4):
        self.pos = np.array(start6, float); self.goal = self.pos.copy(); self.alpha = alpha; self.cameras = {}; self.log = []
    def get_observation(self):
        self.pos += self.alpha * (self.goal - self.pos)
        return {k: float(v) for k, v in zip(CR.KEYS6, self.pos)}
    def send_action(self, a):
        self.goal = np.array([a[k] for k in CR.KEYS6], float); self.log.append(self.goal.copy()); return a
    def disconnect(self): pass

class FakeKeys:
    def __init__(self, schedule): self.n, self.schedule = 0, dict(schedule)
    def start(self): pass
    def stop(self): pass
    def poll(self):
        self.n += 1
        return self.schedule.get(self.n, [])

def main():
    trial = sys.argv[1] if len(sys.argv) > 1 else "E12"; mode = sys.argv[2] if len(sys.argv) > 2 else "deep"
    cfg = CR.load_config(["--dry_run", "--rewind_mode", mode, "--dry_run_source", trial, "--fps", "600", "--out_dir", str(__import__("tempfile").mkdtemp(prefix=f"fake_dry_{trial}_{mode}_"))])
    col = CR.Collector(cfg)
    col.robot = FakeRobot(col.params["start_pose"] + [3.0])
    col.start_pose6 = np.array(col.params["start_pose"] + [3.0]); col.last_cmd6 = col.start_pose6.copy()
    col.keys = FakeKeys({3000: ["t"], 6000: ["r"], **({int(sys.argv[3]): ["space"]} if len(sys.argv) > 3 else {})})   # manual trigger if the online trigger has not fired by then; optional e-stop at poll N
    meta = col.run_episode()
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta.items()}, indent=1, default=str))
    rows = list(csv.DictReader(open(col.meta_path))); print("meta rows written:", len(rows))

if __name__ == "__main__":
    main()

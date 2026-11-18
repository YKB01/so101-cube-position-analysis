#!/usr/bin/env python
"""Software test of --align_only with a simulated follower and leader (no hardware)."""
import sys, json, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "rewind_offline"))
import collect_rewind as CR, test_collect_dry_fake as T
from test_collect_full_fake import FakeTeleop

class SagTeleop(FakeTeleop):
    """After the torque is released the leader slowly sags (shoulder_lift falls) like an unloaded arm."""
    def get_action(self):
        if not self.torque: self.pos[1] -= 0.02
        return super().get_action()

def main():
    cfg = CR.load_config(["--align_only", "--fps", "600", "--align_follow_s", "3", "--out_dir", tempfile.mkdtemp(prefix="fake_align_")])
    col = CR.Collector(cfg)
    start = np.array(col.params["start_pose"] + [3.0])
    col.robot = T.FakeRobot(start); col.start_pose6 = start.copy(); col.last_cmd6 = start.copy()
    col.teleop = SagTeleop(start + 15)
    col.keys = T.FakeKeys({5: ["s"], 9000: ["s"]})
    col.run_align_only()
    rep = json.loads((col.out / "align_only_report.json").read_text())
    print("keys of the report:", list(rep))
    print("align retries:", rep["align_retries"], "| leader drift first 2 s:", round(rep["leader_drift_max_first_2s_deg"], 2))

if __name__ == "__main__":
    main()

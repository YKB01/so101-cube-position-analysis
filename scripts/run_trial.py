#!/usr/bin/env python
"""
Blind runner for one trial of the pre-registered right-side retest.

    python scripts/run_trial.py T01            # run the trial (creates a new attempt: T01_a1, T01_a2, ...)
    python scripts/run_trial.py T01 --dry-run  # show what would run (model never shown)

What it does
  1. looks up the A/B code of the trial in protocol/trial_order.csv and the checkpoint in protocol/model_key.json
     (read inside the script; the mapping and checkpoint path are never printed or written to disk)
  2. checks that the cube was placed for this trial (results/placement_log.csv, newer than any earlier attempt)
  3. runs `lerobot-rollout` (episodic, 1 episode, 30 s) with the fixed inference settings, in a child process
     whose output is filtered (checkpoint path/name replaced by <policy>) into results/retest_raw/<trial>_a<k>/run.log
  4. saves, per attempt: the LeRobot dataset (video, joint states, actions) as YKB0123/rollout_retest_<trial>_a<k>_<timestamp> (LeRobot adds the timestamp)
     (not pushed), chunks.npz (full action chunk of every step + ensembled action), meta.json
  5. asks for the outcome and appends it to results/retest_outcomes.csv

Keys during the run (press them in THIS terminal): Right arrow / e / n / Enter = end the episode early (saved
normally); Ctrl-C = interrupt (partial data kept, redo as a new attempt). Press only once; a second Ctrl-C forces exit. Do NOT use Left arrow (it discards the episode without a trace).
Stop scripts/place_cube_overlay.py before running (one process at a time can use the camera).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import select
import signal
import subprocess
import termios
import threading
import tty
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOLS = {   # --protocol: which pre-registered test is being run (files are kept separate)
    "retest": dict(order="protocol/trial_order.csv", key="protocol/model_key.json", raw="results/retest_raw",
                   outcomes="results/retest_outcomes.csv", placement="results/placement_log.csv", prefix="retest"),
    "trunc": dict(order="protocol/trunc_trial_order.csv", key="protocol/trunc_model_key.json", raw="results/trunc_raw",
                  outcomes="results/trunc_outcomes.csv", placement="results/trunc_placement_log.csv", prefix="trunc"),
}
ORDER_CSV, KEY_JSON, PREFIX = ROOT / "protocol/trial_order.csv", ROOT / "protocol/model_key.json", "retest"
RAW = ROOT / "results/retest_raw"
PRACTICE_CKPT = str(Path.home() / "Checkpoints/act_red_cube_50_100000")   # baseline, NOT one of the two test models
OUTCOMES = ROOT / "results/retest_outcomes.csv"
PLACEMENT = ROOT / "results/placement_log.csv"
HF_USER = "YKB0123"
EPISODE_TIME_S = 30
INFERENCE = dict(n_action_steps=1, temporal_ensemble_coeff=0.01)
ROBOT_ARGS = ["--robot.type=so101_follower", "--robot.port=/dev/ttyACM0", "--robot.id=YKBFollowerR",
              "--robot.cameras={front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, "
              "warmup_s: 10, fourcc: 'MJPG'}}"]
TASK = "Pick up the red cube and put it into the box"
FAIL_TYPES = {"1": "edge_not_grasped", "2": "arm_cannot_reach", "3": "dropped_after_grasp", "4": "cube_pushed_away"}


class EarlyEnd:
    """Reads the terminal of this (parent) process; Right arrow / e / n / Enter sends ONE SIGINT to the child,
    which LeRobot treats as a clean shutdown (episode saved). Independent of LeRobot's own key listener."""
    def __init__(self, proc):
        self.proc, self.pressed, self._run, self._old, self._t = proc, False, False, None, None

    def start(self):
        if not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(fd)
        termios.tcflush(fd, termios.TCIFLUSH)
        tty.setcbreak(fd)
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()

    def _read(self, timeout):
        r, _, _ = select.select([sys.stdin.fileno()], [], [], timeout)
        return os.read(sys.stdin.fileno(), 1).decode(errors="ignore") if r else ""

    def _loop(self):
        while self._run:
            ch = self._read(0.1)
            if not ch:
                continue
            hit = ch in ("e", "n", "\r", "\n")
            if ch == "\x1b":                                  # arrow keys arrive as ESC [ C (right)
                hit = self._read(0.05) in ("[", "O") and self._read(0.05) == "C"
            if hit and not self.pressed and self.proc.poll() is None:
                self.pressed = True
                print("  early end requested (saving the episode)...", flush=True)
                self.proc.send_signal(signal.SIGINT)

    def stop(self):
        self._run = False
        if self._t:
            self._t.join(timeout=0.5)
        if self._old is not None:
            fd = sys.stdin.fileno()
            termios.tcsetattr(fd, termios.TCSADRAIN, self._old)
            termios.tcflush(fd, termios.TCIFLUSH)
            self._old = None


def lookup(trial_id: str) -> tuple[dict, str]:
    order = pd.read_csv(ORDER_CSV)
    row = order[order.trial_id == trial_id]
    if row.empty:
        sys.exit(f"unknown trial id {trial_id}")
    row = row.iloc[0].to_dict()
    key = json.loads(KEY_JSON.read_text())["key"]
    return row, key[row["model_code"]]["checkpoint"]


def attempts(trial_id: str) -> list[Path]:
    return sorted(RAW.glob(f"{trial_id}_a*"), key=lambda p: int(p.name.rsplit("_a", 1)[1]))


def placement_row(trial_id: str) -> dict | None:
    if not PLACEMENT.exists():
        return None
    df = pd.read_csv(PLACEMENT)
    df = df[df.trial_id.astype(str) == trial_id]
    return None if df.empty else df.iloc[-1].to_dict()


def child_cmd(ckpt: str, repo_id: str) -> list[str]:
    return ["--strategy.type=episodic", "--strategy.reset_to_initial_position=true",
            f"--policy.path={ckpt}", f"--policy.n_action_steps={INFERENCE['n_action_steps']}",
            f"--policy.temporal_ensemble_coeff={INFERENCE['temporal_ensemble_coeff']}",
            "--device=cuda", "--inference.type=sync", *ROBOT_ARGS, f"--task={TASK}",
            f"--dataset.repo_id={repo_id}", f"--dataset.single_task={TASK}", "--dataset.num_episodes=1",
            f"--dataset.episode_time_s={EPISODE_TIME_S}", "--dataset.reset_time_s=0",
            "--dataset.push_to_hub=false", "--display_data=false"]


def child_main(outdir: Path):
    """Runs inside the child process: install the chunk hook, run lerobot-rollout, dump the chunks."""
    sys.path.insert(0, str(HERE))
    import chunk_logger as CL
    CL.install()
    from lerobot.scripts import lerobot_rollout as R
    sys.argv = ["lerobot-rollout", *json.loads(os.environ["RT_ARGS"])]
    try:
        R.main()
    finally:
        (outdir / "chunk_summary.json").write_text(json.dumps(CL.dump(outdir / "chunks.npz")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial_id", nargs="?", help="T01..T48 (retest), U01..U32 (trunc), or PRACTICE (baseline checkpoint, no guards/outcome)")
    ap.add_argument("--protocol", choices=list(PROTOCOLS), default="retest")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args()
    global ORDER_CSV, KEY_JSON, PREFIX, RAW, OUTCOMES, PLACEMENT
    cfg = PROTOCOLS[a.protocol]
    ORDER_CSV, KEY_JSON, PREFIX = ROOT / cfg["order"], ROOT / cfg["key"], cfg["prefix"]
    RAW, OUTCOMES, PLACEMENT = ROOT / cfg["raw"], ROOT / cfg["outcomes"], ROOT / cfg["placement"]
    if a.child:
        return child_main(a.child)
    tid = a.trial_id or sys.exit("trial_id required")
    practice = tid == "PRACTICE"
    pl = None
    if practice:
        row, ckpt, prev = dict(position_id="-"), PRACTICE_CKPT, []
        k = int(time.time())
        repo_id, outdir = f"{HF_USER}/rollout_retest_practice_{k}", ROOT / f"results/retest_practice/{k}"
    else:
        row, ckpt = lookup(tid)
        prev = attempts(tid)
        k = len(prev) + 1
        repo_id = f"{HF_USER}/rollout_{PREFIX}_{tid}_a{k}"
        outdir = RAW / f"{tid}_a{k}"

    # ---- guards (not for PRACTICE) ---------------------------------------------------------------
    if not practice and not a.dry_run:
        done = pd.read_csv(OUTCOMES) if OUTCOMES.exists() else pd.DataFrame(columns=["trial_id", "outcome"])
        if (done[done.trial_id == tid].outcome.isin(["success", "fail"])).any():
            sys.exit(f"{tid} already has a completed outcome; a redo is only for interrupted attempts")
        pl = placement_row(tid)
        last_end = max([json.loads((p / "meta.json").read_text()).get("ended_unix", 0) for p in prev
                        if (p / "meta.json").exists()] or [0])
        if pl is None or float(pl["timestamp_unix"]) <= last_end:
            sys.exit(f"{tid}: no cube placement logged since the last attempt; run place_cube_overlay.py and press s")
        if subprocess.run(["pgrep", "-f", "place_cube_overlay.py"], capture_output=True).stdout.strip():
            sys.exit("place_cube_overlay.py is still running; close it first (camera is in use)")

    args = child_cmd(ckpt, repo_id)
    if a.dry_run:
        shown = [x.replace(ckpt, "<policy>") for x in args]
        print(f"{tid} position {row['position_id']} attempt {k}\n  dataset {repo_id}\n  out {outdir}")
        print("  args:", " ".join(shown)); return
    outdir.mkdir(parents=True, exist_ok=True)
    scrub = [ckpt, Path(ckpt).parents[2].name, str(Path(ckpt).parents[2])]
    scrub = sorted(set(scrub), key=len, reverse=True)
    started = time.time()
    print(f"{tid} (position {row['position_id']}), attempt {k}: starting; robot moves after the policy is loaded.\n"
          f"Right arrow / e = end early (saved), Ctrl-C = interrupt. Full log: {outdir / 'run.log'}")
    env = dict(os.environ, RT_ARGS=json.dumps(args), PYTHONUNBUFFERED="1")
    sys.path.insert(0, str(ROOT))
    import lighting_probe as LP
    if not LP.inside_window(datetime.now(), "13:30-15:30"):
        print(f"  WARNING: outside the protocol window 13:30-15:30 (now {datetime.now():%H:%M}); the trial is flagged.")
    cam = {}
    def _read_cam():                                   # camera controls read back while the policy runs (read only)
        time.sleep(18)
        cam.update(LP.read_camera_controls("/dev/video0")); cam["cam_read_while_running"] = proc.poll() is None
    threading.Thread(target=_read_cam, daemon=True).start()
    proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--child", str(outdir)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=ROOT)
    pat = re.compile("|".join(re.escape(s) for s in scrub))
    early = EarlyEnd(proc); early.start()
    with open(outdir / "run.log", "w") as log:
        while True:
            try:
                line = proc.stdout.readline()
            except KeyboardInterrupt:
                continue                      # the child receives the same SIGINT and shuts down cleanly
            if not line:
                break
            line = pat.sub("<policy>", line)
            log.write(line); log.flush()
            if re.search(r"ERROR|Traceback|Exception|Record loop is running slower", line):
                print("  ", line.rstrip())
    rc = proc.wait()
    early.stop()
    ended = time.time()
    m = re.search(r"Dataset ready: (\S+)", (outdir / "run.log").read_text())
    actual_repo = m.group(1) if m else repo_id          # LeRobot appends a timestamp to rollout dataset names
    summ = json.loads((outdir / "chunk_summary.json").read_text()) if (outdir / "chunk_summary.json").exists() else {}
    meta = dict(trial_id=tid, position_id=row["position_id"], attempt=k, dataset_repo_id=actual_repo,
                started_unix=started, ended_unix=ended, early_end_key=early.pressed, return_code=rc, placement=pl, inference=INFERENCE,
                episode_time_s=EPISODE_TIME_S, chunks=summ)
    meta["outside_time_window"] = not LP.inside_window(datetime.fromtimestamp(started), "13:30-15:30")
    meta["camera_controls"] = dict(cam)
    try:                                               # control-loop period from the logged policy-step times
        t = np.load(outdir / "chunks.npz")["t"].astype(float); dt = np.diff(t) * 1e3
        meta["loop_dt_mean_ms"], meta["loop_dt_max_ms"] = float(dt.mean()), float(dt.max())
    except Exception as e:
        meta["loop_dt_error"] = str(e)
    try:                                               # first-frame lighting covariates from the recorded dataset
        from lerobot.datasets import LeRobotDataset
        name = actual_repo.split("/", 1)[1]
        root = Path.home() / ".cache/huggingface/lerobot" / actual_repo
        im = np.asarray(LeRobotDataset(actual_repo, root=root)[0]["observation.images.front"].permute(1, 2, 0).clip(0, 1) * 255, dtype=np.uint8)
        meta.update(LP.frame_lighting(im))
    except Exception as e:
        meta["lighting_error"] = str(e)
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"finished (return code {rc}); recorded {summ.get('n_steps', 0)} policy steps with full action chunks.")

    # ---- outcome ----------------------------------------------------------------------------------
    if practice:
        print("practice run: no outcome recorded"); return
    try:
        o = ""
        while o not in ("s", "f", "i"):
            o = input("Outcome  s = success (cube in box AND gripper released within 30 s) | f = failure | "
                      "i = interrupted (redo) : ").strip().lower()
        ftype, note = "", ""
        if o == "f":
            while ftype not in FAIL_TYPES:
                ftype = input("Failure type  1 edge/not grasped | 2 arm cannot reach | 3 dropped after grasp | "
                              "4 cube pushed away : ").strip()
        note = input("Note (optional): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\noutcome not recorded; add it by hand to", OUTCOMES); return
    new = not OUTCOMES.exists()
    with open(OUTCOMES, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["trial_id", "attempt", "outcome", "failure_type", "note", "timestamp_unix"])
        w.writerow([tid, k, {"s": "success", "f": "fail", "i": "interrupted"}[o], FAIL_TYPES.get(ftype, ""), note,
                    f"{ended:.3f}"])
    print("recorded." + ("  Redo: place the cube again, then run this command again." if o == "i" else ""))


if __name__ == "__main__":
    main()

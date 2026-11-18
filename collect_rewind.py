#!/usr/bin/env python
"""
Collect human corrections after an automatic rewind (real robot), with two test modes.

  python collect_rewind.py --rewind_mode deep --policy_path <checkpoint> --repo_id YKB0123/rewind_deep_20          # full flow
  python collect_rewind.py --dry_run --rewind_mode deep --dry_run_source E12                                       # follower only
  python collect_rewind.py --replay --rewind_mode deep                                                             # no hardware
  python collect_rewind.py --align_only                                                                            # follower + leader, no policy
  python collect_rewind.py --policy_only --policy_path <checkpoint> [--compare_model C] [--repo_id YKB0123/rollout_x]  # policy + camera + follower

Per episode: RESET (teleop, not recorded) -> AUTONOMOUS (policy, 15 s state cache) -> TRIGGERED (hold, open gripper, wait)
-> select target -> REWIND (measured trajectory replayed backwards, per-joint step limit) -> ALIGN (leader driven to the follower)
-> WAIT_START (leader torque on until the operator presses the start key) -> HUMAN (leader released, recording with intervention=True)
-> saved -> RESET. The rewind itself is saved to a separate dataset (<repo_id>_rewind_seg by default, intervention=False).
Keys (terminal, single key): s start (episode / begin human recording)  t manual trigger  e end correction  x discard this episode
                              c continue after a prompt  r rewind to the start pose (when no target)  q quit (in RESET)  SPACE emergency stop.
Emergency stop: sends the measured position once, then no further commands; torque is NOT switched off. Quitting afterwards needs confirmation
and first moves the arm slowly back to its start-up position.
All parameters: collect_rewind_config.json (defaults) and command-line overrides; rule thresholds: rewind_offline/rewind_params.json.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import queue
import select
import signal
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rewind_control as RC  # noqa: E402
import rewind_target as RT   # noqa: E402

ARM_KEYS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos"]
GRIP_KEY = "gripper.pos"
KEYS6 = ARM_KEYS + [GRIP_KEY]
META_COLUMNS = [
    "episode_id", "timestamp", "saved_episode_index", "segment_episode_index", "rewind_mode", "level", "rewind_segment_saved",
    "cube_start_x", "cube_start_y", "trigger_type", "trigger_frame", "trigger_time_s",
    "target_buffer_index", "target_episode_frame", "target_time_s", "target_progress", "target_nn_distance",
    "rewind_planned_s", "rewind_actual_s", "n_waypoints", "joint_distance_deg", "settled",
    "final_err_max_deg", "final_err_norm", "final_err_norm_over_tau", "final_err_shoulder_pan", "final_err_shoulder_lift", "final_err_elbow_flex", "final_err_wrist_flex", "final_err_wrist_roll",
    "cube_before_x", "cube_before_y", "cube_after_x", "cube_after_y", "cube_shift_px",
    "align_err_max_deg", "align_retries", "human_s", "discarded", "discard_reason", "autonomous_dir"]


# ====================================================================================================== config
def load_config(argv=None):
    cfg = json.loads((HERE / "collect_rewind_config.json").read_text())
    cfg.pop("_comment", None)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="JSON file overriding collect_rewind_config.json")
    ap.add_argument("--replay", action="store_true", help="no robot: drive the state machine with recorded failed trials")
    ap.add_argument("--dry_run", action="store_true", help="follower only: manual trigger -> open gripper -> select -> rewind -> settle")
    ap.add_argument("--align_only", action="store_true", help="follower + leader, no policy: move to a pose, align the leader, release, follow 10 s")
    ap.add_argument("--policy_only", action="store_true", help="policy + camera + follower, no leader, no rewind: full 30 s episodes at evaluation settings")
    ap.add_argument("--rewind_mode", choices=["shallow", "deep"], default="deep")
    ap.add_argument("--policy_path", default=None, help="ACT checkpoint (pretrained_model directory); needed for the full flow")
    for k, v in cfg.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=None)
    a = ap.parse_args(argv)
    if a.config:
        cfg.update({k: v for k, v in json.loads(Path(a.config).read_text()).items() if k in cfg})
    for k in cfg:
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    cfg.update(replay=a.replay, dry_run=a.dry_run, align_only=a.align_only, policy_only=a.policy_only, rewind_mode=a.rewind_mode, policy_path=a.policy_path)
    if sum([a.replay, a.dry_run, a.align_only, a.policy_only]) > 1:
        ap.error("choose at most one of --replay --dry_run --align_only --policy_only")
    for k in ("rewind_params", "rewind_index"):
        cfg[k] = str((HERE / cfg[k]) if not os.path.isabs(cfg[k]) else cfg[k])
    if not cfg["out_dir"]:
        cfg["out_dir"] = str(HERE / "results" / "rewind_collect" / f"{'replay' if a.replay else 'dry_run' if a.dry_run else 'align_only' if a.align_only else 'policy_only' if a.policy_only else 'run'}_{datetime.now():%Y%m%d_%H%M%S}")
    if a.policy_only:
        if not cfg["policy_path"]:
            ap.error("--policy_only needs --policy_path")
    elif not (a.replay or a.dry_run or a.align_only):
        if not cfg["repo_id"] or not cfg["policy_path"]:
            ap.error("the full flow needs --repo_id and --policy_path")
        if not cfg["repo_id_rewind_seg"]:
            cfg["repo_id_rewind_seg"] = cfg["repo_id"] + "_rewind_seg"
    return cfg


# ====================================================================================================== keys
class KeyReader:
    """Single-key reader on this terminal (cbreak). poll() returns the keys pressed since the last call."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self._run, self._old, self._t = False, None, None

    def start(self):
        if not sys.stdin.isatty():
            print("WARNING: stdin is not a terminal; keys are unavailable", flush=True)
            return
        fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(fd)
        termios.tcflush(fd, termios.TCIFLUSH)
        tty.setcbreak(fd)
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        fd = sys.stdin.fileno()
        while self._run:
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                ch = os.read(fd, 1).decode(errors="ignore")
                if ch == "\x1b":                       # swallow arrow-key escape sequences
                    while select.select([fd], [], [], 0.01)[0]:
                        os.read(fd, 1)
                    continue
                self.q.put("space" if ch == " " else ch.lower())

    def poll(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def stop(self):
        self._run = False
        if self._t:
            self._t.join(timeout=0.5)
        if self._old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)
            self._old = None


class EStop(Exception):
    pass


class Discard(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


# ====================================================================================================== helpers
def vec6(obs) -> np.ndarray:
    return np.array([obs[k] for k in KEYS6], float)


def cube_position(rgb):
    import analyze_cube_positions as A
    d = A.detect_cube(np.asarray(rgb))
    return (float(d["cx"]), float(d["cy"])) if d["detected"] else (float("nan"), float("nan"))


def save_png(path, rgb):
    import cv2
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))


class AutoLog:
    """Everything the policy did in one episode (not part of the training datasets): video + states/actions."""

    def __init__(self, d: Path, fps, size):
        import cv2
        d.mkdir(parents=True, exist_ok=True)
        self.d, self.t, self.state, self.action = d, [], [], []
        self.vw = cv2.VideoWriter(str(d / "autonomous.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size) if size else None

    def add(self, t, state, action, rgb):
        import cv2
        self.t.append(t); self.state.append(state); self.action.append(action)
        if self.vw is not None and rgb is not None:
            self.vw.write(cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))

    def close(self):
        if self.vw is not None:
            self.vw.release()
        np.savez(self.d / "autonomous_states.npz", t=np.array(self.t), state=np.array(self.state), action=np.array(self.action))


# ====================================================================================================== collector
class Collector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fps = cfg["fps"]
        self.dt = 1.0 / self.fps
        self.params = json.loads(Path(cfg["rewind_params"]).read_text())
        self.index, _ = RT.load_index(cfg["rewind_index"])
        self.sigma = np.array(self.params["sigma"])
        self.keys = KeyReader()
        self.estop_flag = threading.Event()
        self.out = Path(cfg["out_dir"]); self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config_used.json").write_text(json.dumps(cfg, indent=2, default=str))
        self.meta_path = self.out / "rewind_meta.csv"
        self.dry = cfg["dry_run"]
        self.align_only, self.policy_only = cfg["align_only"], cfg["policy_only"]
        self.ctx = self.robot = self.teleop = self.processors = None
        self.ds = self.seg_ds = None
        self.last_cmd6 = None
        self.phase = "init"
        self.episode_counter = 0
        self.seg_rows: dict[int, dict] = {}
        self.leader_torque_on = False
        self.stack = contextlib.ExitStack()
        self.dry_rows: list[dict] = []
        self.report_hook = None

    # ------------------------------------------------------------------------------------------- hardware
    def connect(self):
        cfg = self.cfg
        if self.dry or self.align_only:
            from lerobot.robots import make_robot_from_config
            from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
            self.robot = make_robot_from_config(SO101FollowerConfig(port=cfg["robot_port"], id=cfg["robot_id"], cameras={}))
            self.robot.connect()
            self.processors = None
            if self.align_only:
                from lerobot.teleoperators import make_teleoperator_from_config
                from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
                self.teleop = make_teleoperator_from_config(SO101LeaderConfig(port=cfg["leader_port"], id=cfg["leader_id"]))
                self.teleop.connect()
        else:
            # config registries: the parser resolves --robot.type / --teleop.type / camera types through these imports (as in lerobot-rollout)
            from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
            from lerobot.robots import so_follower  # noqa: F401
            from lerobot.teleoperators import so_leader  # noqa: F401
            from lerobot.configs import parser
            from lerobot.rollout import RolloutConfig, build_rollout_context
            from lerobot.utils.action_interpolator import ActionInterpolator
            cams = (f"{{front: {{type: opencv, index_or_path: {cfg['camera_index']}, width: {cfg['camera_width']}, height: {cfg['camera_height']}, "
                    f"fps: {int(self.fps)}, warmup_s: {cfg['camera_warmup_s']}, fourcc: '{cfg['camera_fourcc']}'}}}}")
            argv = ["--robot.type=so101_follower", f"--robot.port={cfg['robot_port']}", f"--robot.id={cfg['robot_id']}", f"--robot.cameras={cams}",
                    *([] if self.policy_only else ["--teleop.type=so101_leader", f"--teleop.port={cfg['leader_port']}", f"--teleop.id={cfg['leader_id']}"]),
                    f"--policy.path={cfg['policy_path']}", f"--policy.n_action_steps={cfg['n_action_steps']}",
                    f"--policy.temporal_ensemble_coeff={cfg['temporal_ensemble_coeff']}", f"--device={cfg['device']}",
                    "--inference.type=sync", "--strategy.type=base", f"--task={cfg['task']}", f"--fps={int(self.fps)}", "--play_sounds=false"]

            def _parse(c):
                return c
            _parse.__annotations__ = {"c": RolloutConfig}       # this file uses postponed annotations; lerobot's parser needs the real class
            _parse = parser.wrap()(_parse)
            old = sys.argv
            sys.argv = [old[0]] + argv
            try:
                rcfg = _parse()
            finally:
                sys.argv = old
            self.shutdown_event = threading.Event()
            self.ctx = build_rollout_context(rcfg, self.shutdown_event)
            self.robot, self.teleop, self.processors = self.ctx.hardware.robot_wrapper, self.ctx.hardware.teleop, self.ctx.processors
            assert list(self.ctx.data.ordered_action_keys) == KEYS6, f"action keys differ: {self.ctx.data.ordered_action_keys}"
            self.engine = self.ctx.policy.inference
            self.interp = ActionInterpolator(multiplier=rcfg.interpolation_multiplier)
            self.engine.reset(); self.engine.start()
            self.rcfg = rcfg
            if self.policy_only:
                if cfg["repo_id"]:
                    self._make_eval_dataset()
            else:
                self._make_datasets()
        obs = self.robot.get_observation()
        self.start_pose6 = vec6(obs)                 # slow return target at exit
        self.last_cmd6 = self.start_pose6.copy()
        self.keys.start()

    def _open_dataset(self, repo_id, root, feats):
        from lerobot.configs.dataset import DatasetRecordConfig
        from lerobot.datasets import LeRobotDataset, VideoEncodingManager
        cfg = self.cfg
        dc = DatasetRecordConfig(repo_id=repo_id, single_task=cfg["task"], root=root or None, fps=int(self.fps),
                                 streaming_encoding=cfg["streaming_encoding"], encoder_threads=cfg["encoder_threads"])
        nimg = 4 * len(self.robot.cameras)
        if cfg["resume"]:
            ds = LeRobotDataset.resume(repo_id, root=dc.root, batch_encoding_size=1, camera_encoder=dc.camera_encoder,
                                       streaming_encoding=dc.streaming_encoding, encoder_queue_maxsize=dc.encoder_queue_maxsize,
                                       encoder_threads=dc.encoder_threads, image_writer_processes=0, image_writer_threads=nimg)
        else:
            ds = LeRobotDataset.create(repo_id, int(self.fps), root=dc.root, robot_type=self.robot.name, features=feats, use_videos=True,
                                       image_writer_processes=0, image_writer_threads=nimg, batch_encoding_size=1,
                                       camera_encoder=dc.camera_encoder, streaming_encoding=dc.streaming_encoding,
                                       encoder_queue_maxsize=dc.encoder_queue_maxsize, encoder_threads=dc.encoder_threads)
        self.stack.enter_context(VideoEncodingManager(ds))
        return ds

    def _make_eval_dataset(self):
        """policy_only: same feature set as lerobot-rollout's episodic evaluation datasets (no intervention column)."""
        self.features = dict(self.ctx.data.dataset_features)
        self.ds = self._open_dataset(self.cfg["repo_id"], self.cfg["root"], self.features)
        print(f"evaluation dataset: {self.ds.repo_id} ({self.ds.num_episodes} episodes)", flush=True)

    def _make_datasets(self):
        cfg = self.cfg
        feats = dict(self.ctx.data.dataset_features)
        feats["intervention"] = {"dtype": "bool", "shape": (1,), "names": None}
        self.features = feats

        make = lambda repo_id, root: self._open_dataset(repo_id, root, feats)
        self.ds = make(cfg["repo_id"], cfg["root"])
        self.seg_ds = make(cfg["repo_id_rewind_seg"], cfg["root_rewind_seg"])
        self.episode_counter = self.ds.num_episodes
        print(f"datasets: corrections {self.ds.repo_id} ({self.ds.num_episodes} episodes) | rewind segments {self.seg_ds.repo_id} ({self.seg_ds.num_episodes})", flush=True)

    # ------------------------------------------------------------------------------------------- I/O primitives
    def get_obs(self):
        return self.robot.get_observation()

    def send6(self, cmd6, obs=None):
        cmd6 = np.asarray(cmd6, float).copy()
        cmd6[:5] = RC.clamp_arm(cmd6[:5], self.params, self.cfg["joint_margin_deg"])
        d = {k: float(v) for k, v in zip(KEYS6, cmd6)}
        if self.processors is not None:
            d_sent = self.processors.robot_action_processor((d, obs if obs is not None else self.get_obs()))
        else:
            d_sent = d
        self.robot.send_action(d_sent)
        self.last_cmd6 = cmd6
        return d

    def sleep_to(self, t_loop_start):
        from lerobot.utils.robot_utils import precise_sleep
        r = self.dt - (time.perf_counter() - t_loop_start)
        if r > 0:
            precise_sleep(r)

    def poll_keys(self, allow_discard=True, extra=()):
        """Raises EStop / Discard; returns other keys of interest."""
        out = []
        if self.estop_flag.is_set():
            raise EStop()
        for k in self.keys.poll():
            if k == "space":
                raise EStop()
            if k == "x" and allow_discard:
                raise Discard("operator_x")
            if k in extra:
                out.append(k)
        return out

    def seg_frame(self, obs, cmd_dict):
        if self.seg_ds is None:
            return
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame
        op = self.processors.robot_observation_processor(obs)
        self.seg_ds.add_frame({**build_dataset_frame(self.features, op, prefix=OBS_STR), **build_dataset_frame(self.features, cmd_dict, prefix=ACTION),
                               "task": self.cfg["task"], "intervention": np.array([False], dtype=bool)})

    # ------------------------------------------------------------------------------------------- meta
    def write_meta(self, meta):
        new = not self.meta_path.exists()
        with open(self.meta_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=META_COLUMNS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: meta.get(k, "") for k in META_COLUMNS})

    def write_seg_sidecar(self):
        if self.seg_ds is None:
            return
        p = Path(self.seg_ds.root) / "rewind_segment_meta.csv"
        cols = ["segment_episode_index", "rewind_segment", "episode_id", "rewind_mode", "level", "discarded", "discard_reason"]
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for k in sorted(self.seg_rows):
                w.writerow(self.seg_rows[k])

    # ------------------------------------------------------------------------------------------- waiting with keys
    def wait_key(self, text, wanted, hold=True):
        """Block (holding the follower) until one of `wanted` is pressed; x discards, SPACE e-stops."""
        print(f"\n>>> {text}", flush=True)
        while True:
            t0 = time.perf_counter()
            ks = self.poll_keys(extra=wanted)
            if ks:
                return ks[0]
            if hold and self.last_cmd6 is not None:
                self.send6(self.last_cmd6)
            self.sleep_to(t0)

    # ------------------------------------------------------------------------------------------- phases
    def slide_follower_to_leader(self):
        """Bring the follower to the leader's pose with a step limit (leader torque off, operator holds the leader)."""
        c = self.cfg
        while True:
            t0 = time.perf_counter()
            self.poll_keys(allow_discard=False)
            obs = self.get_obs(); cur = vec6(obs)
            tgt = np.array([self.teleop.get_action()[k] for k in KEYS6], float)
            err = np.abs(tgt - cur).max()
            if err <= c["reset_slide_tol_deg"]:
                return
            step = np.clip(tgt - self.last_cmd6, -c["reset_slide_step_deg"], c["reset_slide_step_deg"])
            self.send6(self.last_cmd6 + step, obs)
            self.sleep_to(t0)

    def teleop_tick(self, obs, record=False):
        if self.processors is None:                         # align_only: no processors, same key space
            ta = self.teleop.get_action()
            self.robot.send_action(ta)
            self.last_cmd6 = np.array([ta[k] for k in KEYS6], float)
            return ta
        ta = self.teleop.get_action()
        pt = self.processors.teleop_action_processor((ta, obs))
        ra = self.processors.robot_action_processor((pt, obs))
        self.robot.send_action(ra)
        self.last_cmd6 = np.array([ra[k] for k in KEYS6], float)
        return pt

    def reset_phase(self):
        self.phase = "reset"
        if self.leader_torque_on:
            self.teleop.disable_torque(); self.leader_torque_on = False
        print("\n=== RESET: teleoperate the arm to its start pose (not recorded), place the cube. [s] start episode   [q] quit ===", flush=True)
        self.slide_follower_to_leader()
        while True:
            t0 = time.perf_counter()
            ks = self.poll_keys(allow_discard=False, extra=("s", "q"))
            obs = self.get_obs()
            self.teleop_tick(obs)
            if "q" in ks:
                return "quit"
            if "s" in ks:
                return "start"
            self.sleep_to(t0)

    def run_episode(self):
        c = self.cfg
        self.episode_counter_attempt = getattr(self, "episode_counter_attempt", 0) + 1
        eid = self.episode_counter_attempt
        meta = dict(episode_id=eid, timestamp=datetime.now().isoformat(timespec="seconds"), rewind_mode=c["rewind_mode"], discarded=False,
                    rewind_segment_saved=False)
        ed = self.out / "episodes" / f"ep_{eid:04d}"
        meta["autonomous_dir"] = str(ed)
        self.autolog = None
        try:
            self.autonomous(meta, ed)
            self.triggered(meta, ed)
            plan = self.select(meta)
            self.rewind(meta, plan)
            if self.dry:
                return meta
            self.align(meta, ed)
            self.wait_start(meta)
            self.human(meta)
        except Discard as d:
            self.discard(meta, d.reason)
        except EStop:
            meta["discarded"], meta["discard_reason"] = True, "estop"
            raise
        finally:
            if self.autolog is not None:
                with contextlib.suppress(Exception):
                    self.autolog.close()
            self.write_meta(meta)
            if meta.get("segment_episode_index", "") != "":
                self.seg_rows[int(meta["segment_episode_index"])].update(discarded=meta["discarded"], discard_reason=meta.get("discard_reason", ""))
                self.write_seg_sidecar()
        return meta

    def discard(self, meta, reason):
        print(f"\n*** episode discarded: {reason}", flush=True)
        meta["discarded"], meta["discard_reason"] = True, reason
        for ds in (self.ds, self.seg_ds):
            if ds is not None and ds.has_pending_frames():
                ds.clear_episode_buffer()
        if self.last_cmd6 is not None and self.robot is not None:
            with contextlib.suppress(Exception):
                self.send6(self.last_cmd6)

    # --- AUTONOMOUS
    def autonomous(self, meta, ed):
        c = self.cfg
        self.phase = "autonomous"
        self.buffer = RC.StateBuffer(self.params["buffer_s"], self.fps)
        self.trigger = RC.OnlineTrigger(self.params, self.fps)
        obs = self.get_obs()
        rgb = obs.get("front")
        meta["cube_start_x"], meta["cube_start_y"] = cube_position(rgb) if rgb is not None else (float("nan"),) * 2
        if rgb is not None:
            save_png(ed / "cube_start.png", rgb)
        size = (rgb.shape[1], rgb.shape[0]) if rgb is not None else None
        self.autolog = AutoLog(ed, self.fps, size)
        if self.dry:
            return self.autonomous_dry(meta)
        self.engine.reset(); self.interp.reset()
        from lerobot.rollout.strategies.core import send_next_action
        print(f"\n=== episode {meta['episode_id']}: AUTONOMOUS (cube at {meta['cube_start_x']:.0f},{meta['cube_start_y']:.0f}); "
              f"[t] manual trigger  [x] discard  [SPACE] e-stop ===", flush=True)
        frame, t_start, last_action = 0, time.perf_counter(), None
        while True:
            t0 = time.perf_counter()
            manual = "t" in self.poll_keys(extra=("t",))
            obs = self.get_obs(); state = vec6(obs)
            op = self.processors.robot_observation_processor(obs)
            ad = send_next_action(op, obs, self.ctx, self.interp)
            if ad is not None:
                last_action = np.array([ad[k] for k in KEYS6], float)
                self.last_cmd6 = last_action
            self.buffer.append(state, last_action)
            self.autolog.add(time.perf_counter() - t_start, state, last_action if last_action is not None else state, obs.get("front"))
            fired = self.trigger.update(frame, state)
            if manual:
                fired = "manual"
            if fired:
                meta["trigger_type"], meta["trigger_frame"], meta["trigger_time_s"] = fired, frame, frame / self.fps
                print(f"*** TRIGGER: {fired} at {frame / self.fps:.1f} s", flush=True)
                self.trigger_state6 = state
                return
            frame += 1
            self.sleep_to(t0)

    def autonomous_dry(self, meta):
        """Dry run: the follower replays a recorded failed trial (step-limited) so that a realistic state cache exists."""
        c = self.cfg
        import analyze_grasp_points as G
        Q = load_trial_states(c["dry_run_source"])
        gfr = G.detect_first_grasp(Q[:, 5])["grasp_frame"]
        stop_frame = None if (np.isnan(gfr) or c["dry_run_stop_before_grasp_s"] <= 0) else int(gfr - c["dry_run_stop_before_grasp_s"] * 30)
        print(f"\n=== DRY RUN episode {meta['episode_id']}: first moving to the start of {c['dry_run_source']} then replaying it (step limit "
              f"{c['dry_run_play_step_deg']} deg/tick); the replay stops and triggers {c['dry_run_stop_before_grasp_s']} s of recording before the first grasp "
              f"(frame {stop_frame}) so that the arm does not press on the table. [t] trigger now  [SPACE] e-stop ===", flush=True)
        step = c["dry_run_play_step_deg"]
        # 1) go to the first state
        obs = self.get_obs(); self.last_cmd6 = vec6(obs)
        ctrl0 = RC.RewindController([Q[0, :5]], self.last_cmd6[:5], step)
        gr = RC.GripperRamp(self.last_cmd6[5], Q[0, 5], c["gripper_step"])
        while not (ctrl0.done and gr.done):
            t0 = time.perf_counter(); self.poll_keys(allow_discard=False)
            obs = self.get_obs(); cmd5, _ = ctrl0.step(); self.send6(np.r_[cmd5, gr.next()], obs); self.sleep_to(t0)
        for _ in range(int(0.7 * self.fps)):
            t0 = time.perf_counter(); self.poll_keys(allow_discard=False); self.send6(self.last_cmd6); self.sleep_to(t0)
        # 2) replay, building the cache from the MEASURED states
        ctrl = RC.RewindController(Q[:, :5], self.last_cmd6[:5], step)
        gr = RC.GripperRamp(self.last_cmd6[5], Q[0, 5], c["gripper_step"])
        frame, t_start = 0, time.perf_counter()
        while True:
            t0 = time.perf_counter()
            manual = "t" in self.poll_keys(allow_discard=False, extra=("t",))
            obs = self.get_obs(); state = vec6(obs)
            self.buffer.append(state, None)
            self.autolog.add(time.perf_counter() - t_start, state, self.last_cmd6, None)
            fired = self.trigger.update(frame, state)
            if manual:
                fired = "manual"
            if fired is None and stop_frame is not None and ctrl.k >= stop_frame:
                fired = "auto_stop_before_grasp"            # keep the replay above the table: stop before the recorded grasp
            if fired:
                meta["trigger_type"], meta["trigger_frame"], meta["trigger_time_s"] = fired, frame, frame / self.fps
                self.trigger_state6 = state
                print(f"*** TRIGGER: {fired} at {frame / self.fps:.1f} s", flush=True)
                return
            if not ctrl.done:
                cmd5, _ = ctrl.step()
                gr.target = Q[min(ctrl.k, len(Q) - 1), 5]
                self.send6(np.r_[cmd5, gr.next()], obs)
            else:
                self.send6(self.last_cmd6, obs)
                if frame % (5 * self.fps) == 0:
                    print("   (recorded trial finished; press [t] to trigger)", flush=True)
            frame += 1
            self.sleep_to(t0)

    # --- TRIGGERED
    def triggered(self, meta, ed):
        c = self.cfg
        self.phase = "triggered"
        obs = self.get_obs(); state = vec6(obs)
        rgb = obs.get("front")
        if rgb is not None:
            save_png(ed / "at_trigger.png", rgb)
            meta["cube_before_x"], meta["cube_before_y"] = cube_position(rgb)
        hold5 = state[:5].copy()
        gr = RC.GripperRamp(state[5], self.params["gripper_open_value"], c["gripper_step"])
        wait_ticks = int(round(c["trigger_wait_s"] * self.fps))
        self.seg_t0 = time.perf_counter()
        print("TRIGGERED: holding position, opening the gripper", flush=True)
        extra = 0
        while True:
            t0 = time.perf_counter()
            self.poll_keys()
            obs = self.get_obs()
            d = self.send6(np.r_[hold5, gr.next()], obs)
            self.seg_frame(obs, d)
            if gr.done:
                extra += 1
                if extra >= wait_ticks:
                    return
            self.sleep_to(t0)

    # --- selection
    def select(self, meta):
        c, p = self.cfg, self.params
        buf = self.buffer.array(); ti = len(buf) - 1
        mode = c["rewind_mode"]
        level, idx, progress, dist = mode, None, None, None
        if mode == "shallow":
            r = RT.select_rewind_target_lead(buf, ti, int(round(p["shallow_lead_s"] * self.fps)), self.index, p)
            idx, progress, dist = r.idx, r.progress, r.distance
            if idx is None:
                k = self.wait_key(f"No rewind target ({r.reason}). [r] rewind to the start pose   [x] discard", ("r",))
                level = "start_pose_fallback"
        else:
            r = RT.select_rewind_target_deep(buf, ti, self.index, p)
            level, idx, progress, dist = r.level, r.idx, r.progress, r.distance
        pose = buf[ti, :5] if idx is None and level == "shallow" else None
        wp = RT.plan_rewind_path(buf, ti, idx, p) if (idx is not None or level in ("start_pose", "start_pose_fallback")) else None
        target5 = buf[idx, :5] if idx is not None else np.asarray(p["start_pose"], float)
        meta.update(level=level, target_buffer_index=idx if idx is not None else "", target_progress=progress if progress is not None else "",
                    target_nn_distance=dist if dist is not None else "", n_waypoints=len(wp),
                    joint_distance_deg=float(np.linalg.norm(buf[ti, :5] - target5)),
                    rewind_planned_s=RT.estimate_rewind_steps(wp, c["max_step_deg"]) / self.fps)
        if idx is not None:
            meta["target_episode_frame"] = meta["trigger_frame"] - (ti - idx)
            meta["target_time_s"] = meta["target_episode_frame"] / self.fps
        print(f"REWIND PLAN: mode={mode} level={level} target buffer idx={idx} ({'' if idx is None else f'{(ti - idx) / self.fps:.1f} s before the trigger'}) "
              f"waypoints={len(wp)} planned {meta['rewind_planned_s']:.1f} s, joint distance {meta['joint_distance_deg']:.0f} deg", flush=True)
        return dict(wp=wp, target5=target5, level=level)

    # --- REWIND
    def rewind(self, meta, plan):
        c, p = self.cfg, self.params
        self.phase = "rewind"
        wp = np.asarray(plan["wp"], float)
        wp = np.array([RC.clamp_arm(w, p, c["joint_margin_deg"]) for w in wp])
        obs = self.get_obs(); cur = vec6(obs)
        ctrl = RC.RewindController(wp, cur[:5], c["max_step_deg"])
        gopen = p["gripper_open_value"]
        limit = int(c["max_rewind_s"] * self.fps); ticks = 0; t_begin = time.perf_counter()
        self.dry_rows = []
        print("REWIND: replaying the measured trajectory backwards", flush=True)
        while not ctrl.done:
            t0 = time.perf_counter()
            self.poll_keys()
            obs = self.get_obs(); meas = vec6(obs)
            cmd5, hit = ctrl.step()
            d = self.send6(np.r_[cmd5, gopen], obs)
            self.seg_frame(obs, d)
            if hit is not None:
                self.dry_rows.append(dict(waypoint=hit, **{f"sent_{k}": wp[hit][i] for i, k in enumerate(ARM_KEYS)},
                                          **{f"meas_{k}": meas[i] for i, k in enumerate(ARM_KEYS)}, tick=ticks))
            ticks += 1
            if ticks > limit and not ctrl.done:
                k = self.wait_key(f"Rewind took longer than max_rewind_s={c['max_rewind_s']} s (waypoint {ctrl.k}/{len(wp)}). [c] continue   [x] discard", ("c",))
                limit = ticks + int(c["max_rewind_s"] * self.fps)
            self.sleep_to(t0)
        final5 = wp[-1]
        det = RC.SettleDetector(c["settle_tol_deg"], c["settle_consec"], int(c["settle_timeout_s"] * self.fps), c["settle_still_deg"], c["settle_max_err_deg"])
        res = None
        while res is None:
            t0 = time.perf_counter()
            self.poll_keys()
            obs = self.get_obs(); meas = vec6(obs)
            d = self.send6(np.r_[final5, gopen], obs)
            self.seg_frame(obs, d)
            res = det.update(meas[:5], final5)
            self.sleep_to(t0)
        err = np.abs(meas[:5] - final5)
        nerr = RC.normalised_error(meas[:5] - final5, self.sigma)
        meta.update({f"final_err_{k.split('.')[0]}": float(meas[i] - final5[i]) for i, k in enumerate(ARM_KEYS)})
        meta.update(rewind_actual_s=time.perf_counter() - t_begin, settled=(res == "settled"), final_err_max_deg=float(err.max()),
                    final_err_norm=nerr, final_err_norm_over_tau=nerr / p["tau"])
        self.final_report = dict(err_per_joint=dict(zip(ARM_KEYS, err.round(3))), norm=nerr, tau=p["tau"], settled=res)
        print(f"REWIND DONE ({res}): final error per joint {np.round(err, 2).tolist()} deg, normalised {nerr:.3f} (tau {p['tau']:.3f}, ratio {nerr / p['tau']:.2f}), "
              f"{meta['rewind_actual_s']:.1f} s (planned {meta['rewind_planned_s']:.1f} s)", flush=True)
        self.hold_target6 = np.r_[meas[:5], gopen]
        if self.seg_ds is not None and self.seg_ds.has_pending_frames():
            idx = self.seg_ds.num_episodes
            self.seg_ds.save_episode()
            meta["segment_episode_index"], meta["rewind_segment_saved"] = idx, True
            self.seg_rows[idx] = dict(segment_episode_index=idx, rewind_segment=True, episode_id=meta["episode_id"], rewind_mode=c["rewind_mode"],
                                      level=meta["level"], discarded=False, discard_reason="")
            self.write_seg_sidecar()
        if self.dry:
            self.save_dry(meta, wp)

    def save_dry(self, meta, wp):
        eid = meta["episode_id"]
        import pandas as pd
        df = pd.DataFrame(self.dry_rows)
        for k in ARM_KEYS:
            df[f"lag_{k}"] = df[f"meas_{k}"] - df[f"sent_{k}"]
        df.to_csv(self.out / f"dry_run_{eid:02d}_waypoints.csv", index=False, float_format="%.3f")
        lag = df[[f"lag_{k}" for k in ARM_KEYS]].abs().to_numpy()
        print(f"   tracking lag at waypoint arrival (|measured - sent|, deg): median {np.median(lag.max(1)):.2f}, max {lag.max():.2f}  (file dry_run_{eid:02d}_waypoints.csv)", flush=True)

    # --- ALIGN
    def align_leader(self, tgt6, log=None):
        """Drive the leader (torque on) to tgt6 in an interruptible loop (same primitives as lerobot's teleop_smooth_move_to: enable_torque,
        linear interpolation, send_feedback) and verify the error. Returns (err_vector, retries). Raises EStop / Discard."""
        c = self.cfg
        self.phase = "align"
        tgt6 = np.asarray(tgt6, float)
        self.teleop.enable_torque(); self.leader_torque_on = True
        retries, prev_emax, carry = 0, None, None
        while True:
            cur = np.array([self.teleop.get_action()[k] for k in KEYS6], float)
            dur = max(c["align_min_s"], np.abs(tgt6 - cur).max() / c["align_speed_deg_s"])
            steps = max(int(dur * self.fps), 1)
            print(f"ALIGN: driving the leader to the follower ({dur:.1f} s, attempt {retries + 1}); keep your hands off the leader", flush=True)
            try:
                for s_ in range(1, steps + 1):
                    t0 = time.perf_counter()
                    self.poll_keys()
                    a = s_ / steps
                    self.teleop.send_feedback({k: float(cur[i] * (1 - a) + tgt6[i] * a) for i, k in enumerate(KEYS6)})
                    self.send6(self.last_cmd6)
                    self.sleep_to(t0)
                # trim phase: the leader's position controller has a steady-state error under gravity (a few degrees), so the goal is moved
                # beyond the target by an integrated error (bounded by align_trim_max_deg) until the measured leader pose meets the target
                goal, max_off, ki = (tgt6.copy() if carry is None else carry.copy()), c["align_trim_max_deg"], c["align_trim_gain_per_s"]
                for _ in range(max(int(c["align_hold_s"] * self.fps), 1)):
                    t0 = time.perf_counter(); self.poll_keys()
                    now = np.array([self.teleop.get_action()[k] for k in KEYS6], float)
                    goal = np.clip(goal + ki * self.dt * (tgt6 - now), tgt6 - max_off, tgt6 + max_off)
                    self.teleop.send_feedback({k: float(goal[i]) for i, k in enumerate(KEYS6)})
                    self.send6(self.last_cmd6); self.sleep_to(t0)
            except EStop:
                with contextlib.suppress(Exception):                       # keep the leader where it is
                    self.teleop.send_feedback({k: float(v) for k, v in self.teleop.get_action().items()})
                raise
            lp = np.array([self.teleop.get_action()[k] for k in KEYS6], float)
            err = lp - tgt6
            carry = goal.copy()
            emax = float(np.abs(err).max())
            if log is not None:
                log.append(dict(attempt=retries + 1, duration_s=dur, **{f"err_{k}": float(err[i]) for i, k in enumerate(KEYS6)}, err_max=emax,
                                leader=lp.tolist(), target=tgt6.tolist(), trim_offset=(goal - tgt6).tolist()))
                if self.report_hook is not None:
                    self.report_hook()
            if emax <= c["align_tol_deg"]:
                return err, retries
            worst = int(np.argmax(np.abs(err)))
            print(f"   attempt {retries + 1}: leader - target per joint {np.round(err, 1).tolist()} deg (worst: {KEYS6[worst]} {err[worst]:+.1f}); tolerance {c['align_tol_deg']}; "
                  f"goal offset used {np.round(goal - tgt6, 1).tolist()}", flush=True)
            # stall: the error did not improve by >= 0.5 deg compared with the previous attempt -> the leader is blocked (table, joint limit, ...): stop pushing
            if retries >= 1 and prev_emax is not None and emax > prev_emax - 0.5:
                self.teleop.disable_torque(); self.leader_torque_on = False
                self.wait_key(f"The leader is stalled at {KEYS6[worst]} (error {err[worst]:+.1f} deg, not improving): it is probably blocked (table / joint limit). Torque released. "
                              f"[c] try again from scratch   [x] discard", ("c",), hold=True)
                self.teleop.enable_torque(); self.leader_torque_on = True
                retries, prev_emax = 0, None
                continue
            prev_emax = emax
            retries += 1
            if retries > c["align_max_retries"]:
                self.wait_key(f"Leader did not reach the follower (max error {np.abs(err).max():.1f} deg > {c['align_tol_deg']}). [c] try again   [x] discard", ("c",), hold=True)
                retries = 0

    def align(self, meta, ed):
        obs = self.get_obs(); tgt = vec6(obs)
        err, retries = self.align_leader(tgt)
        meta.update(align_err_max_deg=float(np.abs(err).max()), align_retries=retries)
        obs = self.get_obs(); rgb = obs.get("front")
        if rgb is not None:
            save_png(ed / "after_align.png", rgb)
            ax, ay = cube_position(rgb)
            meta["cube_after_x"], meta["cube_after_y"] = ax, ay
            bx, by = meta.get("cube_before_x", np.nan), meta.get("cube_before_y", np.nan)
            meta["cube_shift_px"] = float(np.hypot(ax - bx, ay - by)) if not np.isnan(ax) and not np.isnan(bx) else ""
            if np.isnan(ax):
                self.wait_key("Cube not detected after the rewind (pushed out of the workspace, or hidden by the arm?). "
                              "[c] keep the episode   [x] discard (cube out of workspace)", ("c",))
        print(f"ALIGN DONE: max leader-follower error {np.abs(err).max():.2f} deg", flush=True)

    # --- WAIT_START
    def wait_start(self, meta):
        self.phase = "wait_start"
        try:
            self.wait_key("Aligned. Hands on the leader, then press [s] to release it and start recording   ([x] discard)", ("s",))
        except Discard as d:
            self.teleop.disable_torque(); self.leader_torque_on = False
            raise

    # --- HUMAN
    def human(self, meta):
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame
        c = self.cfg
        self.teleop.disable_torque(); self.leader_torque_on = False
        self.phase = "human"
        print("\n=== HUMAN: recording the correction. [e] end and save   [x] discard   [SPACE] e-stop ===", flush=True)
        t_begin = time.perf_counter()
        while True:
            t0 = time.perf_counter()
            ks = self.poll_keys(extra=("e",))
            obs = self.get_obs()
            op = self.processors.robot_observation_processor(obs)
            pt = self.teleop_tick(obs)
            self.ds.add_frame({**build_dataset_frame(self.features, op, prefix=OBS_STR), **build_dataset_frame(self.features, pt, prefix=ACTION),
                               "task": c["task"], "intervention": np.array([True], dtype=bool)})
            if "e" in ks:
                break
            self.sleep_to(t0)
        meta["human_s"] = time.perf_counter() - t_begin
        idx = self.ds.num_episodes
        self.ds.save_episode()
        meta["saved_episode_index"] = idx
        print(f"saved correction episode {idx} ({meta['human_s']:.1f} s)", flush=True)

    # ------------------------------------------------------------------------------------------- top level
    def run(self):
        try:
            try:
                if self.dry:
                    self.run_dry()
                elif self.align_only:
                    self.run_align_only()
                elif self.policy_only:
                    self.run_policy_only()
                else:
                    self.run_full()
            except EStop:
                for ds in (self.ds, self.seg_ds):
                    if ds is not None and ds.has_pending_frames():
                        ds.clear_episode_buffer()
                self.estop()
        finally:
            self.shutdown()

    def run_full(self):
        print(__doc__.split("Keys")[1].split("All parameters")[0], flush=True)
        while True:
            if self.reset_phase() == "quit":
                return
            self.run_episode()

    def run_dry(self):
        print("DRY RUN: follower only. Keep a hand on the power switch.", flush=True)
        while True:
            m = self.run_episode()
            print("\n[s] run again   [q] quit", flush=True)
            while True:
                t0 = time.perf_counter(); ks = self.poll_keys(allow_discard=False, extra=("s", "q"))
                self.send6(self.last_cmd6)
                if ks:
                    break
                self.sleep_to(t0)
            if ks[0] == "q":
                return

    def estop(self):
        print("\n*** EMERGENCY STOP: holding the measured position; no further commands. Torque stays on. [q] quit (asks for confirmation) ***", flush=True)
        self.phase = "estop"
        with contextlib.suppress(Exception):
            obs = self.get_obs()
            m = vec6(obs)
            self.robot.send_action({k: float(v) for k, v in zip(KEYS6, m)})
        while True:
            for k in self.keys.poll():
                if k == "q":
                    print("Quit after the emergency stop? The arm will first move slowly back to its start-up position. Press [y] to confirm, any other key to stay.", flush=True)
                    while True:
                        ks = self.keys.poll()
                        if ks:
                            break
                        time.sleep(0.05)
                    if ks[0] == "y":
                        return
            time.sleep(0.05)

    def shutdown(self):
        print("\nshutting down: returning to the start-up position...", flush=True)
        with contextlib.suppress(Exception):
            if self.robot is not None:
                obs = self.get_obs(); cur = vec6(obs)
                steps = max(int(self.cfg["exit_return_s"] * self.fps), 1)
                for s in range(1, steps + 1):
                    t0 = time.perf_counter()
                    a = s / steps
                    self.robot.send_action({k: float(cur[i] * (1 - a) + self.start_pose6[i] * a) for i, k in enumerate(KEYS6)})
                    self.sleep_to(t0)
        self.keys.stop()
        if self.teleop is not None and self.leader_torque_on:
            with contextlib.suppress(Exception):
                self.teleop.disable_torque()
        with contextlib.suppress(Exception):
            self.stack.close()                    # finalises both datasets
        for dev in (getattr(self, "robot", None), getattr(self, "teleop", None)):
            with contextlib.suppress(Exception):
                if dev is not None:
                    (dev.inner if hasattr(dev, "inner") else dev).disconnect()


def load_trial_states(trial_id: str) -> np.ndarray:
    """Measured joint states (N, 6) of a recorded failed trial: E## = eval_expert_smooth episode, T## = right-side retest trial (final attempt)."""
    import pandas as pd
    import analyze_cube_positions as A
    import analyze_grasp_points as G
    if trial_id.startswith("E"):
        f, _ = G.load_frames("eval_expert_smooth")
        e = f[f.episode_index == int(trial_id[1:])]
        if e.empty:
            raise SystemExit(f"unknown trial {trial_id}")
    else:
        rt = pd.read_csv(HERE / "retest_analysis/trial_table.csv")
        r = rt[rt.trial_id == trial_id]
        if r.empty:
            raise SystemExit(f"unknown trial {trial_id}")
        meta = json.loads((HERE / f"results/retest_raw/{trial_id}_a{int(r.attempt.iloc[0])}/meta.json").read_text())
        A.DATASETS["_rt"] = "rollout_retest/" + meta["dataset_repo_id"].split("/", 1)[1]   # datasets live in YKB0123/rollout_retest/
        e, _ = G.load_frames("_rt")
    e = e.sort_values("frame_index")
    return e[[f"state:{j}" for j in G.ARM_JOINTS] + [f"state:{G.GRIPPER}"]].to_numpy(float)


def reference_curves(model: str):
    """Pre-grasp arm-speed curves (aligned to the first grasp, -4 s .. +0.5 s) and first-grasp times of the retest trials of `model` (R or C)."""
    import pandas as pd
    import analyze_grasp_points as G
    import analyze_correction as C
    cache = HERE / f"retest_analysis/reference_curves_{model}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["curves"], z["grasp_t"]
    rt = pd.read_csv(HERE / "retest_analysis/trial_table.csv")
    curves, gts = [], []
    for tid in rt[rt.model == model].trial_id:
        S = load_trial_states(tid)
        gf = G.detect_first_grasp(S[:, 5])["grasp_frame"]
        if np.isnan(gf):
            continue
        gf = int(gf)
        curves.append(speed_curve(S[:, :5], gf)); gts.append(gf / 30)
    curves, gts = np.array(curves), np.array(gts)
    np.savez(cache, curves=curves, grasp_t=gts)
    return curves, gts


def speed_curve(q5, gf, window_s=4.0, after_s=0.5):
    import analyze_correction as C
    sp = np.r_[0.0, np.linalg.norm(np.diff(q5, axis=0), axis=1) * 30.0]
    sm = C.moving_average(sp, C.SMOOTH_FRAMES)
    idx = np.arange(gf - int(window_s * 30), gf + int(after_s * 30) + 1)
    return np.where((idx >= 0) & (idx < len(sm)), sm[np.clip(idx, 0, len(sm) - 1)], np.nan)


def _collector_methods():
    """Extra Collector methods for --align_only and --policy_only (attached below)."""

    def align_pose6(self):
        c, p = self.cfg, self.params
        if c["align_pose_joints"]:
            arm = np.array([float(x) for x in c["align_pose_joints"].split(",")])
            assert len(arm) == 5, "--align_pose_joints needs 5 comma-separated values"
            src = "given joints"
        else:
            import pandas as pd
            trial, _, which = c["align_pose"].partition(":")
            which = which or "deep"
            Q = load_trial_states(trial)
            if which == "shallow":
                f = int(pd.read_csv(HERE / "rewind_offline/rewind_targets.csv").set_index("trial").loc[trial, "shallow_lead1s_frame"])
            else:
                f = int(pd.read_csv(HERE / "rewind_offline/phase0_deep_fix.csv").set_index("trial").loc[trial, "new_deep_frame"])
            arm = Q[f, :5]
            src = f"{which} rewind target of {trial} (frame {f})"
        arm = RC.clamp_arm(arm, p, c["joint_margin_deg"])
        return np.r_[arm, p["gripper_open_value"]], src

    def run_align_only(self):
        c, p = self.cfg, self.params
        tgt6, src = self.align_pose6()
        obs = self.get_obs(); cur = vec6(obs)
        print(f"\nALIGN-ONLY test. Follower pose to hold: {np.round(tgt6, 1).tolist()} ({src}); "
              f"{np.abs(tgt6 - cur).max():.0f} deg away from the current pose (max over joints).", flush=True)
        report = dict(target_pose=tgt6.tolist(), source=src, start_pose=cur.tolist(), align_attempts=[])
        save_report = lambda: (self.out / "align_only_report.json").write_text(json.dumps(report, indent=2))
        save_report()
        if c["align_move_confirm"]:
            print("Check that the path is clear. [s] move the follower there   [q] quit", flush=True)
            while True:
                t0 = time.perf_counter(); ks = self.poll_keys(allow_discard=False, extra=("s", "q"))
                if "q" in ks:
                    return
                if "s" in ks:
                    break
                self.sleep_to(t0)
        # 1) move the follower (step limited) and let it settle
        self.phase = "move"
        ctrl = RC.RewindController([tgt6[:5]], cur[:5], c["max_step_deg"]); gr = RC.GripperRamp(cur[5], tgt6[5], c["gripper_step"])
        det = RC.SettleDetector(c["settle_tol_deg"], c["settle_consec"], int(c["settle_timeout_s"] * self.fps), c["settle_still_deg"], c["settle_max_err_deg"])
        res = None
        while res is None:
            t0 = time.perf_counter(); self.poll_keys(allow_discard=False)
            obs = self.get_obs(); meas = vec6(obs)
            cmd5, _ = ctrl.step(); self.send6(np.r_[cmd5, gr.next()], obs)
            if ctrl.done and gr.done:
                res = det.update(meas[:5], tgt6[:5])
            self.sleep_to(t0)
        obs = self.get_obs(); fol = vec6(obs)
        report["follower_reached"] = dict(result=res, err_per_joint=(fol - tgt6).tolist())
        save_report()
        print(f"follower at the pose ({res}); error per joint {np.round(fol - tgt6, 2).tolist()} deg", flush=True)
        if res != "settled":
            print("WARNING: the follower did not reach the pose (blocked, e.g. by the table?). It keeps pushing against the commanded pose. [c] continue with its measured pose   [q] quit", flush=True)
            while True:
                t0 = time.perf_counter(); ks = self.poll_keys(allow_discard=False, extra=("c", "q"))
                if "q" in ks:
                    return
                if "c" in ks:
                    break
                self.send6(np.r_[fol[:5], tgt6[5]]); self.sleep_to(t0)     # stop pushing: command the measured pose
            obs = self.get_obs(); fol = vec6(obs)
        # 2) align the leader to the follower's measured pose
        log = report["align_attempts"]
        self.report_hook = save_report
        try:
            err, retries = self.align_leader(fol, log)
        except Discard:
            print("aborted", flush=True)
            return
        report.update(align_attempts=log, align_retries=retries, align_err_per_joint=err.tolist(), align_err_max=float(np.abs(err).max()))
        print(f"ALIGN DONE: retries {retries}, error per joint {np.round(err, 2).tolist()} deg (max {np.abs(err).max():.2f}, tolerance {c['align_tol_deg']})", flush=True)
        # 3) wait for the start key, then release the torque and let the follower follow for align_follow_s
        try:
            self.wait_key(f"Aligned. Hands on the leader, then press [s] to release it; the follower then follows the leader for {c['align_follow_s']:.0f} s (not recorded). [SPACE] e-stop", ("s",))
        except Discard:
            print("aborted", flush=True)
            return
        obs = self.get_obs(); lead_before = np.array([self.teleop.get_action()[k] for k in KEYS6], float); fol_before = vec6(obs)
        self.teleop.disable_torque(); self.leader_torque_on = False
        self.phase = "follow"
        rows, t_begin = [], time.perf_counter()
        print("released: following...", flush=True)
        while time.perf_counter() - t_begin < c["align_follow_s"]:
            t0 = time.perf_counter(); self.poll_keys(allow_discard=False)
            obs = self.get_obs(); fol = vec6(obs)
            ta = self.teleop.get_action(); lead = np.array([ta[k] for k in KEYS6], float)
            self.robot.send_action(ta); self.last_cmd6 = lead
            rows.append(np.r_[time.perf_counter() - t_begin, lead, fol])
            self.sleep_to(t0)
        R = np.array(rows)
        np.savez(self.out / "align_only_follow.npz", t=R[:, 0], leader=R[:, 1:7], follower=R[:, 7:13], leader_before=lead_before, follower_before=fol_before)
        lead_t, fol_t = R[:, 1:7], R[:, 7:13]
        def at(sec):
            i = int(np.argmin(np.abs(R[:, 0] - sec)))
            return dict(t=float(R[i, 0]), leader=lead_t[i].tolist(), follower=fol_t[i].tolist())
        drift = np.abs(lead_t - lead_before).max(axis=1)
        report.update(before_release=dict(leader=lead_before.tolist(), follower=fol_before.tolist(), leader_minus_follower=(lead_before - fol_before).tolist()),
                      after_release={f"{s}s": at(s) for s in (0.5, 1.0, 2.0)} | {"end": at(R[-1, 0])},
                      leader_drift_max_first_2s_deg=float(drift[R[:, 0] <= 2.0].max()) if (R[:, 0] <= 2.0).any() else None,
                      leader_drift_max_deg=float(drift.max()),
                      follower_tracking_error_deg=dict(median=float(np.median(np.abs(lead_t - fol_t).max(axis=1))), max=float(np.abs(lead_t - fol_t).max())))
        (self.out / "align_only_report.json").write_text(json.dumps(report, indent=2))
        print(f"\nBEFORE release: leader - follower per joint {np.round(lead_before - fol_before, 2).tolist()} deg")
        print(f"AFTER release: max leader drift from the aligned pose in the first 2 s: {report['leader_drift_max_first_2s_deg']:.2f} deg; over {c['align_follow_s']:.0f} s: {report['leader_drift_max_deg']:.1f} deg")
        print(f"follower tracking error (|leader - follower|, max over joints): median {report['follower_tracking_error_deg']['median']:.2f}, max {report['follower_tracking_error_deg']['max']:.2f} deg")
        print(f"report: {self.out / 'align_only_report.json'}", flush=True)

    def print_policy_config(self):
        pol = self.ctx.policy.policy; pc = pol.config
        eff = dict(policy_type=getattr(pc, "type", type(pol).__name__), n_action_steps=pc.n_action_steps, temporal_ensemble_coeff=pc.temporal_ensemble_coeff,
                   chunk_size=pc.chunk_size, n_obs_steps=getattr(pc, "n_obs_steps", None), device=self.rcfg.device, use_amp=getattr(pc, "use_amp", None),
                   inference=getattr(self.rcfg.inference, "type", "sync"), interpolation_multiplier=self.rcfg.interpolation_multiplier, fps=self.rcfg.fps,
                   has_temporal_ensembler=hasattr(pol, "temporal_ensembler"), checkpoint=self.cfg["policy_path"])
        want = dict(n_action_steps=1, temporal_ensemble_coeff=0.01, inference="sync", interpolation_multiplier=1, fps=30.0)
        print("\nEFFECTIVE POLICY CONFIG:", flush=True)
        bad = []
        for k, v in eff.items():
            flag = ""
            if k in want and (v != want[k]):
                flag = f"   <-- DIFFERS from the evaluation setting ({want[k]})"; bad.append(k)
            print(f"   {k:26s} {v}{flag}", flush=True)
        if not eff["has_temporal_ensembler"]:
            bad.append("temporal_ensembler"); print("   <-- no temporal ensembler is active", flush=True)
        print("   => " + ("MATCHES the evaluation settings (n_action_steps=1, temporal_ensemble_coeff=0.01, sync, 30 fps)" if not bad else f"MISMATCH: {bad}"), flush=True)
        (self.out / "policy_config.json").write_text(json.dumps(eff | dict(mismatch=bad), indent=2, default=str))
        return bad

    def run_policy_only(self):
        import pandas as pd
        c = self.cfg
        bad = self.print_policy_config()
        if bad:
            print("Press [c] to continue anyway or [q] to quit.", flush=True)
            while True:
                ks = self.poll_keys(allow_discard=False, extra=("c", "q"))
                if "q" in ks:
                    return
                if "c" in ks:
                    break
                time.sleep(0.05)
        print("\nPOLICY-ONLY: full episodes at evaluation settings, no leader, no rewind. [s] start   [e] end the episode early   [x] discard   [q] quit   [SPACE] e-stop", flush=True)
        self.summary_rows, self.curves = [], []
        eid = 0
        while True:
            self.phase = "wait"
            print("\nPlace the cube, then [s] to start (or [q] to quit).", flush=True)
            while True:
                t0 = time.perf_counter(); ks = self.poll_keys(allow_discard=False, extra=("s", "q"))
                if ks:
                    break
                self.send6(self.last_cmd6); self.sleep_to(t0)
            if ks[0] == "q":
                break
            eid += 1
            try:
                self.policy_episode(eid)
            except Discard as d:
                print(f"*** episode {eid} discarded ({d.reason})", flush=True)
                if self.ds is not None and self.ds.has_pending_frames():
                    self.ds.clear_episode_buffer()
            self.return_to_start()
        self.policy_only_report()

    def return_to_start(self):
        self.phase = "return"
        print("returning to the start pose...", flush=True)
        while True:
            t0 = time.perf_counter(); self.poll_keys(allow_discard=False)
            obs = self.get_obs(); cur = vec6(obs)
            step = np.clip(self.start_pose6 - self.last_cmd6, -self.cfg["return_step_deg"], self.cfg["return_step_deg"])
            self.send6(self.last_cmd6 + step, obs)
            if np.abs(self.start_pose6 - cur).max() < 2.0 and np.abs(step).max() < 0.5:
                return
            self.sleep_to(t0)

    def policy_episode(self, eid):
        import pandas as pd
        import analyze_grasp_points as G
        from lerobot.rollout.strategies.core import send_next_action
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame
        c = self.cfg
        ed = self.out / "episodes" / f"ep_{eid:04d}"
        obs = self.get_obs(); rgb = obs.get("front")
        cx, cy = cube_position(rgb) if rgb is not None else (float("nan"),) * 2
        if rgb is not None:
            save_png(ed / "cube_start.png", rgb)
        self.autolog = AutoLog(ed, self.fps, (rgb.shape[1], rgb.shape[0]) if rgb is not None else None)
        self.engine.reset(); self.interp.reset()
        T, dts, t_obs, t_inf, states, actions = [], [], [], [], [], []
        last_action, early = None, False
        print(f"=== episode {eid} (cube at {cx:.0f},{cy:.0f}) running {c['policy_only_episode_s']:.0f} s ===", flush=True)
        t_start = time.perf_counter(); t_prev = None
        try:
            while time.perf_counter() - t_start < c["policy_only_episode_s"]:
                t0 = time.perf_counter()
                if "e" in self.poll_keys(extra=("e",)):
                    early = True
                    break
                to = time.perf_counter(); obs = self.get_obs(); t_obs.append((time.perf_counter() - to) * 1e3)
                state = vec6(obs)
                op = self.processors.robot_observation_processor(obs)
                ti = time.perf_counter(); ad = send_next_action(op, obs, self.ctx, self.interp); t_inf.append((time.perf_counter() - ti) * 1e3)
                if ad is not None:
                    last_action = np.array([ad[k] for k in KEYS6], float); self.last_cmd6 = last_action
                    if self.ds is not None:
                        self.ds.add_frame({**build_dataset_frame(self.features, op, prefix=OBS_STR), **build_dataset_frame(self.features, ad, prefix=ACTION), "task": c["task"]})
                T.append(t0 - t_start); dts.append(0.0 if t_prev is None else (t0 - t_prev) * 1e3); t_prev = t0
                states.append(state); actions.append(last_action if last_action is not None else state)
                self.autolog.add(T[-1], state, actions[-1], obs.get("front"))
                self.sleep_to(t0)
        except (Discard, EStop):
            self.autolog.close()
            raise
        self.autolog.close()
        if self.ds is not None and self.ds.has_pending_frames():
            self.ds.save_episode()
        S = np.array(states); dts = np.array(dts[1:]); t_inf, t_obs = np.array(t_inf), np.array(t_obs)
        np.savez(ed / "ticks.npz", t=np.array(T), dt_ms=np.array([0.0] + list(dts)), obs_ms=t_obs, infer_ms=t_inf)
        gd = G.detect_first_grasp(S[:, 5]); gf = gd["grasp_frame"]
        sp = np.r_[0.0, np.linalg.norm(np.diff(S[:, :5], axis=0), axis=1) * 30.0]
        row = dict(episode_id=eid, timestamp=datetime.now().isoformat(timespec="seconds"), cube_start_x=cx, cube_start_y=cy, duration_s=T[-1] if T else 0.0,
                   n_ticks=len(S), early_end=early, hz_mean=1e3 / dts.mean() if len(dts) else np.nan, hz_p5=1e3 / np.percentile(dts, 95) if len(dts) else np.nan,
                   dt_ms_mean=dts.mean(), dt_ms_p95=np.percentile(dts, 95), dt_ms_max=dts.max(),
                   slow_ticks_pct=100 * float((dts > c["slow_tick_factor"] * 1e3 / self.fps).mean()),
                   infer_ms_mean=t_inf.mean(), infer_ms_p50=np.median(t_inf), infer_ms_p95=np.percentile(t_inf, 95), infer_ms_max=t_inf.max(), obs_ms_mean=t_obs.mean(),
                   first_grasp_frame=gf, first_grasp_time_s=gf / 30 if not np.isnan(gf) else np.nan,
                   mean_arm_speed_first10s=float(np.mean(sp[:300])), max_arm_speed=float(sp.max()))
        self.summary_rows.append(row)
        pd.DataFrame(self.summary_rows).to_csv(self.out / "policy_only_summary.csv", index=False, float_format="%.3f")
        if not np.isnan(gf):
            self.curves.append(speed_curve(S[:, :5], int(gf)))
        print(f"   control {row['hz_mean']:.1f} Hz (p5 {row['hz_p5']:.1f}), slow ticks {row['slow_ticks_pct']:.1f}%, inference {row['infer_ms_mean']:.1f} ms mean / {row['infer_ms_p95']:.1f} p95 / {row['infer_ms_max']:.1f} max, "
              f"first close at {row['first_grasp_time_s']:.2f} s, mean arm speed first 10 s {row['mean_arm_speed_first10s']:.1f}", flush=True)

    def policy_only_report(self):
        if not self.summary_rows:
            return
        import pandas as pd
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        df = pd.DataFrame(self.summary_rows)
        lines = [f"policy-only run: {len(df)} episodes",
                 f"control rate: mean {df.hz_mean.mean():.1f} Hz (target {self.fps:.0f}); worst episode p5 {df.hz_p5.min():.1f} Hz; slow ticks (> {self.cfg['slow_tick_factor']}x the period): {df.slow_ticks_pct.mean():.1f}%",
                 f"inference+send per tick: mean {df.infer_ms_mean.mean():.1f} ms, p95 {df.infer_ms_p95.mean():.1f} ms, max {df.infer_ms_max.max():.1f} ms (period {1e3 / self.fps:.1f} ms)",
                 f"first close time: median {df.first_grasp_time_s.median():.2f} s (range {df.first_grasp_time_s.min():.2f}-{df.first_grasp_time_s.max():.2f})",
                 f"mean arm speed in the first 10 s: median {df.mean_arm_speed_first10s.median():.1f} units/s"]
        m = self.cfg["compare_model"]
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
        tt = np.arange(-120, 16) / 30
        if m:
            rc, rg = reference_curves(m)
            axs[0].plot(tt, np.nanmedian(rc, 0), color="#6250d6", lw=2, label=f"retest model {m} (median, n={len(rc)})")
            axs[0].fill_between(tt, np.nanpercentile(rc, 25, 0), np.nanpercentile(rc, 75, 0), color="#6250d6", alpha=0.18, lw=0)
            lines.append(f"reference (retest model {m}, all positions, n={len(rg)}): first close median {np.median(rg):.2f} s (IQR {np.percentile(rg, 25):.2f}-{np.percentile(rg, 75):.2f}); "
                         f"new run median {df.first_grasp_time_s.median():.2f} s")
            axs[1].boxplot([rg], positions=[0], widths=0.5, showfliers=False)
        for cv in self.curves:
            axs[0].plot(tt, cv, color="#eb6834", lw=0.9, alpha=0.8)
        if self.curves:
            axs[0].plot([], [], color="#eb6834", label="this run (each episode)")
        axs[0].axvline(0, color="#52514e", lw=1); axs[0].set_xlabel("Time relative to first close (s)"); axs[0].set_ylabel("Arm speed (units/s)")
        axs[0].legend(frameon=False, fontsize=8); axs[0].set_title("Pre-grasp arm speed")
        axs[1].scatter(np.full(len(df), 1) + np.random.default_rng(0).uniform(-0.1, 0.1, len(df)), df.first_grasp_time_s, color="#eb6834")
        axs[1].set_xticks([0, 1] if m else [1]); axs[1].set_xticklabels([f"retest {m}", "this run"] if m else ["this run"]); axs[1].set_ylabel("First close time (s)")
        fig.savefig(self.out / "policy_only_speed_curves.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        (self.out / "policy_only_summary.txt").write_text("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines), flush=True)

    return dict(align_pose6=align_pose6, run_align_only=run_align_only, print_policy_config=print_policy_config, run_policy_only=run_policy_only,
                return_to_start=return_to_start, policy_episode=policy_episode, policy_only_report=policy_only_report)


for _n, _f in _collector_methods().items():
    setattr(Collector, _n, _f)


# ====================================================================================================== replay
def run_replay(cfg):
    """No hardware: drive trigger detection, target selection, path planning and the step-limited controller with the recorded failed trials."""
    import pandas as pd
    import rewind_offline.step2_offline_targets as S2
    params = json.loads(Path(cfg["rewind_params"]).read_text())
    index, _ = RT.load_index(cfg["rewind_index"])
    sigma = np.array(params["sigma"]); fps = cfg["fps"]
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    S2.INDEX = index
    trials = S2.load_trials()
    off = pd.read_csv(HERE / "rewind_offline/rewind_targets.csv").set_index("trial")
    p0 = pd.read_csv(HERE / "rewind_offline/phase0_deep_fix.csv").set_index("trial")
    mode, step = cfg["rewind_mode"], cfg["max_step_deg"]
    rows, curves = [], {}
    for t in trials:
        if t["success"]:
            continue
        S = t["states"]
        trig = RC.OnlineTrigger(params, fps)
        fired, f_fire = None, None
        for f in range(len(S)):
            fired = trig.update(f, S[f])
            if fired:
                f_fire = f
                break
        timeout_only = fired is None            # the recording ended before the 30 s timeout: offline rule = end of the data
        if timeout_only:
            f_fire, fired = len(S) - 1, "timeout(end of data)"
        buf = S[max(0, f_fire - int(params["buffer_s"] * fps)): f_fire + 1]; ti = len(buf) - 1; lo = f_fire - ti
        if mode == "shallow":
            r = RT.select_rewind_target_lead(buf, ti, int(round(params["shallow_lead_s"] * fps)), index, params)
            idx, level = r.idx, "shallow"
            ref_col = "shallow_lead1s_frame"; ref = off.loc[t["trial"], ref_col]
        else:
            r = RT.select_rewind_target_deep(buf, ti, index, params)
            idx, level = r.idx, r.level
            ref = p0.loc[t["trial"], "new_deep_frame"]
        wp = RT.plan_rewind_path(buf, ti, idx, params)
        wp = np.array([RC.clamp_arm(w, params, cfg["joint_margin_deg"]) for w in wp])
        ctrl = RC.RewindController(wp, buf[ti, :5], step)
        meas = buf[ti, :5].copy(); cmds, meass = [ctrl.cmd.copy()], [meas.copy()]
        n = 0
        while not ctrl.done:
            c5, _ = ctrl.step(); meas = meas + cfg["replay_lag_alpha"] * (c5 - meas); cmds.append(c5); meass.append(meas.copy()); n += 1
        for _ in range(int(cfg["settle_timeout_s"] * fps)):
            meas = meas + cfg["replay_lag_alpha"] * (wp[-1] - meas); meass.append(meas.copy())
            if np.max(np.abs(meas - wp[-1])) <= cfg["settle_tol_deg"]:
                break
        cm = np.array(cmds); jump = np.abs(np.diff(cm, axis=0)).max() if len(cm) > 1 else 0.0
        off_trig = off.loc[t["trial"], "trigger_frame"]
        rows.append(dict(trial=t["trial"], dataset=t["dataset"], trigger_type=fired, trigger_frame=f_fire, offline_trigger_frame=off_trig,
                         trigger_match=bool(f_fire == off_trig), level=level, target_frame=np.nan if idx is None else lo + idx,
                         offline_target_frame=ref, target_match=bool((idx is None and (pd.isna(ref))) or (idx is not None and lo + idx == ref)),
                         n_waypoints=len(wp), planned_ticks=RT.estimate_rewind_steps(np.vstack([buf[ti, :5], wp]), step), actual_ticks=n,
                         max_cmd_jump_deg=float(jump), final_err_max_deg=float(np.abs(meas - wp[-1]).max()),
                         final_err_norm=RC.normalised_error(meas - wp[-1], sigma),
                         first_wp_dist_to_trigger_state=float(np.abs(wp[0] - buf[ti, :5]).max())))
        curves[t["trial"]] = (np.array(cmds), np.array(meass), wp)
    R = pd.DataFrame(rows); R.to_csv(out / f"replay_check_{mode}.csv", index=False, float_format="%.3f")
    lines = [f"replay check, mode={mode}, max step {step} deg/tick, {len(R)} failed trials",
             f"trigger frame equals the offline analysis: {int(R.trigger_match.sum())}/{len(R)}  ({R[~R.trigger_match].trial.tolist()})",
             f"target frame equals the offline analysis: {int(R.target_match.sum())}/{len(R)}  ({R[~R.target_match].trial.tolist()})",
             f"levels: {R.level.value_counts().to_dict()}",
             f"max command change per tick over all trials: {R.max_cmd_jump_deg.max():.4f} deg (limit {step})",
             f"actual ticks == planned ticks: {int((R.actual_ticks == R.planned_ticks).sum())}/{len(R)}",
             f"final error (simulated plant, lag alpha {cfg['replay_lag_alpha']}): max {R.final_err_max_deg.max():.2f} deg, normalised max {R.final_err_norm.max():.3f} (tau {params['tau']:.3f})",
             f"waypoints: median {R.n_waypoints.median():.0f}; duration median {R.actual_ticks.median() / fps:.1f} s, max {R.actual_ticks.max() / fps:.1f} s",
             f"first waypoint = trigger state: max deviation {R.first_wp_dist_to_trigger_state.max():.2f} deg"]
    (out / f"replay_check_{mode}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = list(curves)
    nc = 6; nr = int(np.ceil(len(names) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.3 * nr), sharey=True)
    for ax, nm in zip(axs.ravel(), names):
        cm, ms, wp = curves[nm]
        for j, lab in enumerate(["pan", "lift", "elbow", "wflex", "wroll"]):
            ax.plot(np.arange(len(cm)) / fps, cm[:, j], lw=1, label=lab if nm == names[0] else None)
        ax.set_title(f"{nm} ({R.set_index('trial').loc[nm, 'level']})", fontsize=8); ax.tick_params(labelsize=7)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    axs[0, 0].legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle(f"Commanded joint angles (deg) during the replayed rewind, mode={mode}, limit {step} deg/tick; x = time (s)", y=1.0)
    fig.tight_layout(); fig.savefig(out / f"replay_rewind_trajectories_{mode}.png", dpi=130); plt.close(fig)


# ====================================================================================================== main
def main():
    cfg = load_config()
    if cfg["replay"]:
        run_replay(cfg)
        return
    col = Collector(cfg)
    signal.signal(signal.SIGINT, lambda *a: col.estop_flag.set())
    col.connect()
    col.run()


if __name__ == "__main__":
    main()

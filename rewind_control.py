"""
Pure (hardware-free) building blocks of the rewind collection loop, shared by collect_rewind.py (real robot, --dry_run)
and its --replay test mode. numpy only.

- StateBuffer:      ring buffer of the last buffer_s seconds of measured joint states (+ commanded actions)
- OnlineTrigger:    rule-based failure detection (empty grasp, timeout) frame by frame; same rules as rewind_offline/step2
- RewindController: per-tick command generator that replays waypoints with a per-joint step limit
- SettleDetector:   "arrived" test (every joint error below a tolerance for N consecutive ticks, or timeout)
- GripperRamp:      opens the gripper with a bounded step per tick
"""
from __future__ import annotations

from collections import deque

import numpy as np

N_ARM = 5


class StateBuffer:
    def __init__(self, seconds: float, fps: float):
        self.maxlen = int(round(seconds * fps)) + 1       # the offline analysis used trigger - 15 s .. trigger inclusive (451 states)
        self.states: deque = deque(maxlen=self.maxlen)
        self.actions: deque = deque(maxlen=self.maxlen)

    def append(self, state6, action6=None):
        self.states.append(np.asarray(state6, float).copy())
        self.actions.append(None if action6 is None else np.asarray(action6, float).copy())

    def array(self) -> np.ndarray:
        return np.stack(list(self.states)) if self.states else np.zeros((0, 6))

    def clear(self):
        self.states.clear(); self.actions.clear()

    def __len__(self):
        return len(self.states)


class OnlineTrigger:
    """
    Empty grasp: a close (gripper falls to <= close_closed_level after having been >= close_open_level, and the arm is not within
    home_dist_deg of the pose at the episode start) whose reading, averaged over frames +26..+30 after the close, is below
    empty_grasp_threshold. Timeout: episode_timeout_s after the start. Returns the trigger type at the frame it fires, else None.
    """

    def __init__(self, params: dict, fps: float = 30.0):
        self.p, self.fps = params, fps
        self.reset(None)

    def reset(self, q0):
        self.q0 = None if q0 is None else np.asarray(q0, float)[:N_ARM].copy()
        self.g: list[float] = []
        self.state = None
        self.pending: list[int] = []
        self.fired = None

    def update(self, frame_idx: int, state6) -> str | None:
        s = np.asarray(state6, float)
        g = float(s[N_ARM])
        if self.q0 is None:
            self.q0 = s[:N_ARM].copy()
        self.g.append(g)
        assert len(self.g) == frame_idx + 1, "update() must be called once per frame, starting at frame 0"
        if g >= self.p["close_open_level"]:
            self.state = "open"
        elif g <= self.p["close_closed_level"] and self.state == "open":
            self.state = "closed"
            if np.linalg.norm(s[:N_ARM] - self.q0) >= self.p["home_dist_deg"]:    # closes at the home pose are not grasps
                self.pending.append(frame_idx)
        settle, w = self.p["empty_grasp_settle_frames"], self.p["empty_grasp_window_frames"]
        for i in list(self.pending):
            if frame_idx >= i + settle:
                self.pending.remove(i)
                if np.mean(self.g[i + settle - w + 1: i + settle + 1]) < self.p["empty_grasp_threshold"]:
                    self.fired = "empty_grasp"
                    return self.fired
        if frame_idx >= int(round(self.p["episode_timeout_s"] * self.fps)):
            self.fired = "timeout"
            return self.fired
        return None


class RewindController:
    """Replays waypoints (K, 5) one per tick at most `max_step_deg` per joint per tick. step() returns the 5-joint command."""

    def __init__(self, waypoints, start_cmd5, max_step_deg: float):
        self.w = np.asarray(waypoints, float)
        self.cmd = np.asarray(start_cmd5, float).copy()
        self.step_deg = float(max_step_deg)
        self.k = 0
        self.arrived: list[int] = []

    @property
    def done(self) -> bool:
        return self.k >= len(self.w)

    def step(self):
        """-> (cmd5, index of the waypoint reached on this tick or None)."""
        if self.done:
            return self.cmd.copy(), None
        d = np.clip(self.w[self.k] - self.cmd, -self.step_deg, self.step_deg)
        self.cmd = self.cmd + d
        hit = None
        if np.max(np.abs(self.w[self.k] - self.cmd)) < 1e-6:
            self.cmd = self.w[self.k].copy()
            hit = self.k
            self.arrived.append(self.k)
            self.k += 1
        return self.cmd.copy(), hit


class SettleDetector:
    """
    'settled' when either every joint error stays below tol_deg for n_consec ticks, or - for poses with a gravity-induced steady-state error -
    the arm stands still (no joint moves more than still_deg per tick) for n_consec ticks while the error is below max_err_deg.
    'timeout' after timeout_ticks. The error actually reached is always reported by the caller, not hidden by this test.
    """

    def __init__(self, tol_deg: float, n_consec: int, timeout_ticks: int, still_deg: float | None = None, max_err_deg: float | None = None):
        self.tol, self.n, self.timeout = tol_deg, n_consec, timeout_ticks
        self.still_deg, self.max_err = still_deg, max_err_deg
        self.ok, self.still, self.ticks, self.prev = 0, 0, 0, None

    def update(self, measured5, target5) -> str | None:
        """-> 'settled' | 'timeout' | None"""
        self.ticks += 1
        m = np.asarray(measured5, float)
        err = np.max(np.abs(m - np.asarray(target5, float)))
        self.ok = self.ok + 1 if err <= self.tol else 0
        moving = np.max(np.abs(m - self.prev)) if self.prev is not None else np.inf
        self.prev = m.copy()
        self.still = self.still + 1 if (self.still_deg is not None and moving <= self.still_deg) else 0
        if self.ok >= self.n:
            return "settled"
        if self.max_err is not None and self.still >= self.n and err <= self.max_err:
            return "settled"
        return "timeout" if self.ticks >= self.timeout else None


class GripperRamp:
    def __init__(self, start, target, step):
        self.v, self.target, self.step = float(start), float(target), float(step)

    @property
    def done(self) -> bool:
        return abs(self.v - self.target) < 1e-6

    def next(self) -> float:
        self.v += float(np.clip(self.target - self.v, -self.step, self.step))
        return self.v


def clamp_arm(q5, params: dict, margin_deg: float):
    lim = params["joint_limits_expert"]
    return np.clip(np.asarray(q5, float), np.asarray(lim["min"]) - margin_deg, np.asarray(lim["max"]) + margin_deg)


def normalised_error(err5, sigma) -> float:
    return float(np.linalg.norm(np.asarray(err5, float) / np.asarray(sigma, float)))

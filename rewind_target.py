"""
Rewind-target selection for "failure -> automatic rewind -> human correction".

Pure numpy/scipy: no LeRobot dependency, so it can be called from a real-robot control loop.

Definition (see rewind_offline/report_rewind_offline.md): the rewind target is the LAST state in the cache (at or before the
trigger) that satisfies
  (a) in distribution: nearest-neighbour distance to the expert states (5 arm joints, each divided by its expert
      standard deviation) <= tau;
  (b) the gripper is open (gripper reading >= gripper_open_threshold);
  (c) estimated task progress <= p, where progress is the progress of the nearest expert frame
      (0 = arm starts to move, 1 = first gripper close).
"In distribution" looks at joint states only, not at images.

Typical use:
    index, params = load_index("rewind_offline/rewind_index.npz", "rewind_offline/rewind_params.json")
    res = select_rewind_target(buffer_states, trigger_idx, p=0.9, index=index, params=params)
    if res.idx is None: ...res.reason...
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

N_ARM = 5            # first 5 entries of a state are the arm joints, the 6th is the gripper


@dataclass
class RewindIndex:
    sigma: np.ndarray        # (5,) per-joint std of the expert data (normalisation)
    states_norm: np.ndarray  # (M, 5) normalised arm states of the matched expert frames (pre-grasp frames)
    progress: np.ndarray     # (M,) progress in [0, 1]
    episode: np.ndarray      # (M,) expert episode index
    frame: np.ndarray        # (M,) frame index within the episode
    tree: cKDTree = field(repr=False, default=None)

    def __post_init__(self):
        if self.tree is None:
            self.tree = cKDTree(self.states_norm)

    def query(self, arm_states: np.ndarray):
        """arm_states (N, 5) raw joint units -> (distance (N,), index into the expert arrays (N,))."""
        d, i = self.tree.query(np.asarray(arm_states, float) / self.sigma, k=1)
        return d, i


@dataclass
class RewindResult:
    idx: int | None                # index into buffer_states (None = no target)
    distance: float | None         # nearest-neighbour distance (normalised units) of the chosen state
    progress: float | None         # estimated progress of the chosen state
    reason: str | None = None      # why no target was found
    n_candidates: int = 0          # states in the cache that satisfy all three conditions
    counts: dict = field(default_factory=dict)   # how many states survived each condition


def load_index(npz_path, params_path=None):
    z = np.load(npz_path)
    idx = RewindIndex(sigma=z["sigma"], states_norm=z["states_norm"], progress=z["progress"],
                      episode=z["episode"], frame=z["frame"])
    params = json.loads(Path(params_path).read_text()) if params_path else None
    return idx, params


def select_rewind_target(buffer_states, trigger_idx, p, index: RewindIndex, params: dict) -> RewindResult:
    """
    buffer_states: (N, 6) joint states (5 arm joints + gripper), oldest first, raw joint units.
    trigger_idx:   index in buffer_states of the trigger moment; only states 0..trigger_idx are considered.
    p:             maximum task progress of the target (0.9 = shallow, 0.4 = deep).
    params:        dict with "tau" (distance threshold) and "gripper_open_threshold".
    """
    b = np.asarray(buffer_states, float)
    if b.ndim != 2 or b.shape[1] < N_ARM + 1:
        raise ValueError("buffer_states must have shape (N, 6): 5 arm joints + gripper")
    if not 0 <= trigger_idx < len(b):
        return RewindResult(None, None, None, "trigger_idx outside the buffer")
    s = b[: trigger_idx + 1]
    dist, nn = index.query(s[:, :N_ARM])
    prog = index.progress[nn]
    in_dist = dist <= params["tau"]
    opened = s[:, N_ARM] >= params["gripper_open_threshold"]
    shallow_enough = prog <= p
    counts = dict(buffer=len(s), in_distribution=int(in_dist.sum()),
                  in_dist_and_open=int((in_dist & opened).sum()),
                  all_three=int((in_dist & opened & shallow_enough).sum()))
    ok = in_dist & opened & shallow_enough
    if not ok.any():
        if not in_dist.any():
            why = "no state in the cache is in distribution (nearest-neighbour distance > tau)"
        elif not (in_dist & opened).any():
            why = "no in-distribution state has an open gripper"
        else:
            why = f"all in-distribution open-gripper states have estimated progress > {p}"
        return RewindResult(None, None, None, why, 0, counts)
    i = int(np.flatnonzero(ok)[-1])
    return RewindResult(i, float(dist[i]), float(prog[i]), None, int(ok.sum()), counts)


def select_rewind_target_lead(buffer_states, trigger_idx, min_lead_frames, index: RewindIndex, params: dict) -> RewindResult:
    """
    Time-based shallow rewind: the last state at least `min_lead_frames` before the trigger that is in distribution
    (distance <= tau) and has an open gripper (>= gripper_open_threshold). No progress condition; the estimated
    progress of the chosen state is still returned for information.
    """
    b = np.asarray(buffer_states, float)
    if b.ndim != 2 or b.shape[1] < N_ARM + 1:
        raise ValueError("buffer_states must have shape (N, 6): 5 arm joints + gripper")
    if not 0 <= trigger_idx < len(b):
        return RewindResult(None, None, None, "trigger_idx outside the buffer")
    last = trigger_idx - int(min_lead_frames)
    if last < 0:
        return RewindResult(None, None, None, f"buffer shorter than the minimum lead ({min_lead_frames} frames)")
    s = b[: last + 1]
    dist, nn = index.query(s[:, :N_ARM])
    in_dist = dist <= params["tau"]
    opened = s[:, N_ARM] >= params["gripper_open_threshold"]
    counts = dict(buffer=len(s), in_distribution=int(in_dist.sum()), in_dist_and_open=int((in_dist & opened).sum()),
                  all_three=int((in_dist & opened).sum()))
    ok = in_dist & opened
    if not ok.any():
        why = ("no state (at least the minimum lead before the trigger) is in distribution" if not in_dist.any()
               else "no in-distribution state (at least the minimum lead before the trigger) has an open gripper")
        return RewindResult(None, None, None, why, 0, counts)
    i = int(np.flatnonzero(ok)[-1])
    return RewindResult(i, float(dist[i]), float(index.progress[nn[i]]), None, int(ok.sum()), counts)


def joint_contributions(arm_states, index: RewindIndex):
    """
    For each state: nearest-neighbour distance and the share of the squared distance contributed by each of the 5
    joints (rows sum to 1), plus the normalised absolute difference per joint. Shapes: (N,), (N, 5), (N, 5).
    """
    z = np.atleast_2d(np.asarray(arm_states, float)[..., :N_ARM]) / index.sigma
    d, i = index.tree.query(z, k=1)
    diff = z - index.states_norm[i]
    sq = diff ** 2
    return d, sq / np.maximum(sq.sum(1, keepdims=True), 1e-12), np.abs(diff)

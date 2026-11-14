"""
Records, at every control step, the policy's full predicted action chunk (chunk_size x action_dim, in robot
joint units, i.e. after the checkpoint's own un-normalisation) and the ensembled action that was sent.
No LeRobot file is modified: hooks are installed at run time by `install()`.

Hooks
  ACTPolicy.predict_action_chunk       -> keeps the raw chunk of the latest call
  SyncInferenceEngine.get_action       -> post-processes that chunk and appends (t, chunk, action)
  SyncInferenceEngine.reset            -> starts a new buffer (episode start / re-record)
Alignment with the recorded dataset: the k-th call is the k-th control tick (interpolation multiplier 1, no
torch.compile warm-up skipping), t is seconds since the first call of the episode.
"""
from __future__ import annotations

import time

import numpy as np
import torch

_state = {"t0": None, "t": [], "chunk": [], "action": [], "postproc_failed": 0, "installed": False}


def reset_buffer():
    _state.update(t0=None, t=[], chunk=[], action=[], postproc_failed=0)


def install():
    if _state["installed"]:
        return
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.rollout.inference.sync import SyncInferenceEngine

    orig_pred = ACTPolicy.predict_action_chunk
    orig_get = SyncInferenceEngine.get_action
    orig_reset = SyncInferenceEngine.reset

    def predict_action_chunk(self, batch, *a, **k):
        out = orig_pred(self, batch, *a, **k)
        self._last_chunk = out.detach()
        return out

    def get_action(self, obs_frame):
        action = orig_get(self, obs_frame)
        chunk = getattr(self._policy, "_last_chunk", None)
        if action is None or chunk is None:
            return action
        now = time.perf_counter()
        if _state["t0"] is None:
            _state["t0"] = now
        try:
            with torch.inference_mode():
                c = self._postprocessor(chunk)
            c = (c["action"] if isinstance(c, dict) else c).squeeze(0).cpu().numpy().astype(np.float32)
        except Exception:                                   # keep the run alive; mark as failed
            _state["postproc_failed"] += 1
            c = np.full(tuple(chunk.squeeze(0).shape), np.nan, np.float32)
        _state["t"].append(now - _state["t0"])
        _state["chunk"].append(c)
        _state["action"].append(action.cpu().numpy().astype(np.float32))
        return action

    def reset(self):
        reset_buffer()
        return orig_reset(self)

    ACTPolicy.predict_action_chunk = predict_action_chunk
    SyncInferenceEngine.get_action = get_action
    SyncInferenceEngine.reset = reset
    _state["installed"] = True


def dump(path) -> dict:
    """Write the buffer to an .npz and return a small summary (no model information)."""
    n = len(_state["t"])
    if n:
        np.savez_compressed(path, t=np.array(_state["t"], np.float32), chunk=np.stack(_state["chunk"]),
                            action=np.stack(_state["action"]))
    return dict(n_steps=n, chunk_shape=list(_state["chunk"][0].shape) if n else None,
                postproc_failed=_state["postproc_failed"])

#!/usr/bin/env python
"""Aggregate the dry runs in results/rewind_collect/dry_run_*: one row per run with the rewind plan, planned vs actual duration, final error
(max, normalised, ratio to tau) and the tracking lag at the waypoints. Writes rewind_offline/dry_run_summary.csv."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parents[1]
rows = []
for d in sorted(glob.glob(str(HERE / "results/rewind_collect/dry_run_*"))):
    cfg = json.loads(Path(d, "config_used.json").read_text())
    m = pd.read_csv(Path(d, "rewind_meta.csv"))
    for _, r in m.iterrows():
        w = pd.read_csv(Path(d, f"dry_run_{int(r.episode_id):02d}_waypoints.csv"))
        lag = w[[c for c in w.columns if c.startswith("lag_")]].abs().max(axis=1)
        rows.append(dict(run=Path(d).name.replace("dry_run_", ""), episode=int(r.episode_id), source=cfg["dry_run_source"], mode=cfg["rewind_mode"], trigger=r.trigger_type,
                         level=r.level, n_waypoints=int(r.n_waypoints), joint_distance_deg=r.joint_distance_deg, planned_s=r.rewind_planned_s, actual_s=r.rewind_actual_s,
                         settled=r.settled, final_err_max_deg=r.final_err_max_deg, final_err_norm=r.final_err_norm, final_err_over_tau=r.final_err_norm_over_tau,
                         lag_median_deg=float(lag.median()), lag_max_deg=float(lag.max())))
S = pd.DataFrame(rows); S.to_csv(HERE / "rewind_offline/dry_run_summary.csv", index=False, float_format="%.3f")
print(S.round(2).to_string(index=False))

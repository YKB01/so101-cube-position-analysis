#!/usr/bin/env python
"""
Check the covariates that scripts/run_trial.py now writes into meta.json, and that the new camera-reading thread did not lengthen the control loop.
  python scripts/run_trial.py PRACTICE            (1-2 runs; practice runs write results/retest_practice/<k>/meta.json)
  python scripts/check_trial_covariates.py [meta.json ...]     (default: the practice runs, newest first)
Compares the control-loop period of the new run(s) with the 80 earlier 11/14 trials (mean, max, share of periods above 50 ms) and prints the lighting covariates.
"""
import glob, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]

def dts(chunks):
    t = np.load(chunks)["t"].astype(float); return np.diff(t) * 1e3

def main():
    files = [Path(a) for a in sys.argv[1:]] or sorted((HERE / "results/retest_practice").glob("*/meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]
    ref = []
    for c in sorted(glob.glob(str(HERE / "results/retest_raw/*/chunks.npz")) + glob.glob(str(HERE / "results/trunc_raw/*/chunks.npz"))):
        d = dts(c); ref.append((d.mean(), d.max(), float((d > 50).mean() * 100)))
    ref = np.array(ref)
    print(f"reference ({len(ref)} earlier trials): control period mean {ref[:, 0].mean():.2f} ms (range {ref[:, 0].min():.2f}-{ref[:, 0].max():.2f}); max {np.median(ref[:, 1]):.0f} ms median, 95th percentile {np.percentile(ref[:, 1], 95):.0f} ms; "
          f"periods > 50 ms: {ref[:, 2].mean():.2f}% on average")
    ok = True
    for mj in files:
        m = json.loads(Path(mj).read_text()); d = dts(Path(mj).parent / "chunks.npz")
        print(f"\n{mj}")
        need = ["outside_time_window", "camera_controls", "loop_dt_mean_ms", "loop_dt_max_ms", "first_frame_table_gray", "first_frame_sheet_gray", "first_frame_gray_mean", "first_frame_table_rb"]
        miss = [k for k in need if k not in m]
        print(f"   fields present: {'all' if not miss else 'MISSING ' + str(miss)}; errors: {[m[k] for k in ('lighting_error', 'loop_dt_error') if k in m]}")
        cam = m.get("camera_controls", {})
        print(f"   camera controls: {len(cam)} values read ({'while the policy was running' if cam.get('cam_read_while_running') else 'AFTER the run: not representative'}); "
              f"auto_exposure={cam.get('cam_auto_exposure')}, exposure_time_absolute={cam.get('cam_exposure_time_absolute')}, white_balance_automatic={cam.get('cam_white_balance_automatic')}")
        print(f"   first-frame table gray {m.get('first_frame_table_gray', float('nan')):.1f}, window-side {m.get('first_frame_sheet_gray', float('nan')):.1f}, whole {m.get('first_frame_gray_mean', float('nan')):.1f}, R/B {m.get('first_frame_table_rb', float('nan')):.3f}; outside window: {m.get('outside_time_window')}")
        print(f"   control period: mean {d.mean():.2f} ms, max {d.max():.1f} ms, periods > 50 ms: {(d > 50).mean() * 100:.2f}% ({int((d > 50).sum())} of {len(d)})")
        slow = d.mean() > ref[:, 0].max() + 0.5 or d.max() > np.percentile(ref[:, 1], 95) * 1.5
        print(f"   => {'LONGER than the earlier trials: look into it' if slow else 'within the range of the earlier trials'}")
        # a reading of the controls at ~18 s after the start must not leave a mark: periods around the middle of the run
        mid = d[int(0.4 * len(d)): int(0.7 * len(d))]; print(f"   periods around the camera read (40-70% of the run): max {mid.max():.1f} ms")
        ok &= (not miss) and not slow
    print("\nOK" if ok else "\nCHECK THE POINTS ABOVE")

if __name__ == "__main__":
    main()

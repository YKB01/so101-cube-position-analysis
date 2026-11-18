#!/usr/bin/env python
"""
Check, on today's hardware, that the empty-grasp threshold (4.2, rewind_offline/rewind_params.json) still separates a gripper that holds the
cube from one that closed on nothing. Follower + leader only (no camera, no policy); you teleoperate and the follower follows.

  python scripts/gripper_reading_check.py [--robot_port /dev/ttyACM0] [--leader_port /dev/ttyACM2] [--n 10]

Block 1: n grasps WITH the cube (close the gripper on the cube and keep it closed for ~2 s, then open).
Block 2: n grasps on NOTHING (close the gripper completely and keep it closed for ~2 s, then open).
A close is detected automatically (gripper >= 30 then <= 15); the reading is the mean of frames +26..+30 after it (same rule as the online trigger).
Keys: x discard the last sample, n finish the current block early, q quit. Output: results/gripper_check/<timestamp>/ (samples.csv, traces.npz, summary.txt, plot).
"""
import argparse, json, os, select, sys, termios, time, tty
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
KEYS6 = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_port", default="/dev/ttyACM0"); ap.add_argument("--robot_id", default="YKBFollowerR")
    ap.add_argument("--leader_port", default="/dev/ttyACM2"); ap.add_argument("--leader_id", default="YKBLeaderR")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--params", default=str(HERE / "rewind_offline/rewind_params.json"))
    a = ap.parse_args()
    P = json.loads(Path(a.params).read_text()); theta = P["empty_grasp_threshold"]
    settle, w = P["empty_grasp_settle_frames"], P["empty_grasp_window_frames"]
    out = HERE / "results/gripper_check" / datetime.now().strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)

    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
    from lerobot.utils.robot_utils import precise_sleep
    robot = make_robot_from_config(SO101FollowerConfig(port=a.robot_port, id=a.robot_id, cameras={}))
    teleop = make_teleoperator_from_config(SO101LeaderConfig(port=a.leader_port, id=a.leader_id))
    robot.connect(); teleop.connect()
    obs = robot.get_observation(); start = np.array([obs[k] for k in KEYS6])

    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd); tty.setcbreak(fd)
    def key():
        return os.read(fd, 1).decode(errors="ignore").lower() if select.select([fd], [], [], 0)[0] else ""

    rows, traces = [], []
    blocks = [("cube", "WITH the cube: grasp the cube and keep the gripper closed for ~2 s, then open"),
              ("empty", "on NOTHING: close the gripper completely and keep it closed for ~2 s, then open")]
    try:
        # first make the follower meet the leader smoothly
        cur = start.copy()
        for _ in range(300):
            tgt = np.array([teleop.get_action()[k] for k in KEYS6]); step = np.clip(tgt - cur, -2, 2); cur = cur + step
            robot.send_action({k: float(v) for k, v in zip(KEYS6, cur)}); precise_sleep(1 / a.fps)
            if np.abs(tgt - cur).max() < 3: break
        for label, text in blocks:
            print(f"\n=== block '{label}': {a.n} grasps {text}. [x] discard last   [n] next block   [q] quit ===", flush=True)
            n_done, g, state, pending = 0, [], None, []
            while n_done < a.n:
                t0 = time.perf_counter()
                k = key()
                if k == "q": raise KeyboardInterrupt
                if k == "n": break
                if k == "x" and rows and rows[-1]["label"] == label:
                    rows.pop(); traces.pop(); n_done -= 1; print("   last sample discarded", flush=True)
                ta = {kk: v for kk, v in teleop.get_action().items()}
                robot.send_action(ta)
                o = robot.get_observation(); gv = float(o["gripper.pos"]); g.append(gv); i = len(g) - 1
                if gv >= 30: state = "open"
                elif gv <= 15 and state == "open": state = "closed"; pending.append(i)
                for j in list(pending):
                    if i >= j + settle:
                        pending.remove(j)
                        win = g[j + settle - w + 1: j + settle + 1]
                        r = float(np.mean(win)); released_early = r > 15
                        if released_early:
                            print(f"   close released too early (reading {r:.1f}); not counted - keep it closed ~2 s", flush=True)
                            continue
                        n_done += 1
                        rows.append(dict(label=label, k=n_done, reading=r, below_threshold=r < theta, min_after=min(g[j:j + settle + 1]),
                                         trace_start=j)); traces.append(np.array(g[max(0, j - 15): j + 61]))
                        print(f"   {label} {n_done}/{a.n}: reading {r:.2f}  ({'EMPTY' if r < theta else 'HELD'} by the threshold {theta})", flush=True)
                precise_sleep(max(0, 1 / a.fps - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        try:
            cur = np.array([robot.get_observation()[k] for k in KEYS6])
            for s in range(1, 121):
                robot.send_action({k: float(cur[i] * (1 - s / 120) + start[i] * s / 120) for i, k in enumerate(KEYS6)}); time.sleep(1 / 30)
            robot.disconnect(); teleop.disconnect()
        except Exception as e:
            print("disconnect:", e)

    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(out / "samples.csv", index=False, float_format="%.3f")
    np.savez(out / "traces.npz", *traces)
    lines = [f"threshold {theta}; earlier data (rewind_offline, 43 held / 53 empty-type closes): held 4.54-6.52, empty 2.56-3.7 (+ a few partial)"]
    for label in ("cube", "empty"):
        d = df[df.label == label].reading if len(df) else []
        if len(d): lines.append(f"{label}: n={len(d)} min {d.min():.2f} median {np.median(d):.2f} max {d.max():.2f}; flagged empty: {int((d < theta).sum())}/{len(d)}")
    if len(df) and set(df.label) == {"cube", "empty"}:
        hmin, emax = df[df.label == "cube"].reading.min(), df[df.label == "empty"].reading.max()
        lines.append(f"gap: lowest held reading {hmin:.2f} - highest empty reading {emax:.2f} = {hmin - emax:.2f}; threshold {theta} inside the gap: {emax < theta <= hmin}")
        lines.append("false alarms (held flagged empty): %d/%d; missed empty: %d/%d" % ((df[df.label == 'cube'].reading < theta).sum(), (df.label == 'cube').sum(),
                                                                                       (df[df.label == 'empty'].reading >= theta).sum(), (df.label == 'empty').sum()))
    (out / "summary.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines)); print("saved", out)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for i, (label, col) in enumerate((("cube", "#2a78d6"), ("empty", "#d03a3a"))):
            y = df[df.label == label].reading.to_numpy() if len(df) else []
            ax.scatter(y, np.full(len(y), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(y)), color=col, s=26)
        ax.axvline(theta, color="k"); ax.set_yticks([0, 1]); ax.set_yticklabels(["with cube", "on nothing"]); ax.set_xlabel("gripper reading 1 s after the close")
        ax.set_title(f"Gripper check (threshold {theta})"); fig.savefig(out / "gripper_check.png", dpi=150, bbox_inches="tight")
    except Exception as e:
        print("plot:", e)

if __name__ == "__main__":
    main()

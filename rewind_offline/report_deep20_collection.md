# Deep-rewind collection (20 corrections): summary, and the idea of moving straight to the target

Data: `YKB0123/rewind_deep_20` (corrections, `intervention=True`) and `YKB0123/rewind_deep_20_rewind_seg` (rewind segments, `intervention=False`); run `results/rewind_collect/run_20261119_120625`, protocol `protocol/rewind_collect_deep.md`.
Numbers are in `deep20_summary.txt`, `deep20_attempts.csv`, `deep20_vs_reference.csv`, `deep20_direct_vs_replay_distribution.csv` (script `analyze_deep20.py`). Descriptive only; no training or test result is part of this report.

## 1. What was collected
| | |
|---|---|
| Attempts / saved / discarded | 30 / **20** / 10 (12:07-12:30, setting A, inside the 10:00-16:00 window) |
| Positions | R01-R20 once each in the pre-registered random order; placement distance median 1.7 px, max 2.9 px |
| Triggers | 30 of 30 automatic empty grasp (median 7.9 s into the episode, 6.8-17.5 s); no manual trigger, no timeout, no emergency stop |
| Deep-rewind level (saved 20) | original rule 19, progress <= 0.2 1, start pose 0 (all 30 attempts: 29 / 1 / 0) |
| Target | median 4.1 s before the trigger (3.5-13.5), estimated progress 0.39, nearest-neighbour distance <= 0.35 (tau 0.449), joint distance trigger -> target median 68 deg (45-149) |
| ACT action queue | 8-94 actions (median 62) were still queued at the trigger and cleared; 0 at every start and after every clear |
| Rewind | planned 2.6 s, actual 3.0 s (max 8.8 s, episode 8); settled 30/30; final error median 3.0 deg (max 3.9), at most 0.25 of tau |
| Leader alignment | error median 1.2 deg (max 1.7), one retry in 30 |
| Human correction | 16.1 s (13.2-23.2 s); 9,810 frames in 20 episodes (median 482 per episode) |
| Control loop | 33.5 ms mean in both phases; maximum 53 ms (autonomous) / 36 ms (human); none above 100 ms |
| Lighting | first-frame table gray median 117.2 (105-123); bright state (>= 117.5) in 12 of 30 attempts; automatic exposure and white balance in every attempt (`exposure_time_absolute` reads 157 in automatic mode, not informative) |
| Discards | arm hid the cube after the rewind 6; cube pushed 2 (R12, 17-24 px); cube slightly moved 2 (episodes 2, 8) - see the deviation log of the protocol |

## 2. Against the earlier data (medians; `deep20_vs_reference.csv`)
| | frames | first frame: distance from the start pose | time to the first grasp | mean speed | speed before the grasp | jerk |
|---|---|---|---|---|---|---|
| **new corrections after deep rewind (20)** | 482 | **138** | 3.4 s | 25.1 | 19.9 | 2332 |
| 10/15 DAgger corrections (20) | 456 | 217 | 3.7 s | 29.4 | 14.6 | 2982 |
| 11/12 new expert (20) | 598 | 8 | 7.2 s | 30.1 | 29.7 | 2496 |
| 7/31 expert (50) | 598 | 6 | 7.2 s | 29.8 | 33.4 | 2384 |
The new corrections start closer to the start pose than the 10/15 corrections (138 against 217 joint units) but still far from the start pose of an expert demonstration (6-8), and contain about as much pre-grasp time (3.4 against 3.7 s); the deep target is about 4 s before the failed grasp, not the beginning of the approach.
Their jerk (2332) is as low as the expert data (2384-2496), so the operator's unsteadiness during the collection is not visible in this measure.

## 3. Idea: move straight to the rewind target instead of replaying the path backwards
**What the recorded rewinds show.**
- The replayed path is longer than the direct route: path length median 1.36x the joint-space distance trigger -> target (105 against 68 deg; up to 7x in episode 8, which wandered for 17.5 s before the trigger), and the replayed path departs from the straight joint-space line by up to 15 deg (median) / 42 deg (max)
  (`fig_deep20_path.png`). At 1.5 deg per joint per tick a direct move takes **1.2 s** against 2.6 s planned / 3.0 s actual for the replay, i.e. about 1.8 s less per episode.
- **The cube was displaced while the rewind ran** (segment videos; only frames in which the whole cube is visible are used, repeat detections of an unmoved cube agree within ~1 px; 28 of 30 segments evaluable): more than 3 px in **7 of 28 segments (25%)**:
  5 after the rewind motion had started (episodes 2, 11, 15, 23, 27; shifts 3.6-12.4 px) and 2 while the gripper was opening at the trigger pose, before any motion (episodes 12, 26; about 3 px).
  Episode 23 (saved, `fig_deep20_cube_example.png`, top row) is the clear case: the cube moves about 12 px while the arm lifts and swings along the replayed path. Episode 2 (bottom row) was discarded for a slightly moved cube.
  For comparison, the cube moved by more than 3 px already during the policy phase (before the trigger) in **12 of 28** attempts; the rewind cannot influence that part.
- The six "arm hid the cube" discards come from the **target pose**, which is the same for a direct move; a different path does not remove them (section 4).

**What the data cannot show.** Whether a straight move would not touch the cube. The data contain no forward kinematics or clearance model for the gripper, and the straight line passes through poses the arm never visited in this episode; the replay is, apart from the cube, a path the arm has already travelled.
One proxy: the poses on the straight joint-space line are as typical of the expert demonstrations as the replayed path (largest nearest-neighbour distance along the path: median 0.28 for the line, 0.29 for the replay; at least one point beyond tau in 6 of 30 lines and 7 of 30 replays, `deep20_direct_vs_replay_distribution.csv`). This is evidence that the line does not leave the region of usual arm poses, not a collision check.

**Assessment.** The idea is plausible and cheap: it removes the detours and the retracing, saves time, and the rewinds that moved the cube did so after the motion started, i.e. along the replayed path. The target pose, and therefore the human correction data, would not change.
I would implement it as an option and test it before using it for collection:
1. `--rewind_path replay|direct`: `direct` = two waypoints (trigger pose, target pose) with the same per-joint step limit, joint-limit clamp and settle test. (The replay stays the default for the data already collected.)
2. Open the gripper while the arm lifts only if the grasp is empty (empty-grasp triggers: reading < 4.2); the two cube displacements at the trigger pose came from opening the gripper while the arm held still on the table. Keep opening first if the reading shows the cube is held.
3. Test: `--dry_run` for the pose check (no cube); then 10 episodes per path type on the same positions with the cube-shift measurement used here (segment video, fully visible detections, > 3 px) as the outcome; interleave the two types. The sample is small, so only a clear difference would count.
4. Independent of the path: take the "cube moved / hidden" decision from the segment video (last fully visible position of the cube during the rewind) instead of the one frame after the alignment; that frame is where the arm hides the cube (6 of the 10 discards).

## 4. Other observations
- The failed attempt itself moved the cube by more than 3 px in 12 of 28 attempts; 9 of these were kept as saved episodes, so some corrections start from a cube displaced by the policy (`start_to_first_px` in `deep20_attempts.csv`).
- Rewind final error (median 3.0 deg) is larger than in the dry runs (0.6-1.2 deg) because of the steady-state error at these poses; it is below 0.25 of tau and the leader is aligned to the measured pose, so the human part is not affected.
- 12 of 30 attempts were in the bright automatic-exposure state, within the range of the 11/14 sessions.

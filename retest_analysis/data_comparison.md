# Task A: recovery vs new expert vs original expert (data only, no training)

Purpose: explain why model R (50 expert + 20 human recovery) did worse than model C (50 expert + 20 new expert) in the
pre-registered retest. H1: the recovery data lack the full approach from the start pose to the grasp. H2: human
recovery motions are slower / less smooth than expert demonstrations.

## Definitions (constants at the top of `analyze_data_comparison.py`)
- Signals: `observation.state` of the 5 arm joints; speed = norm of the frame difference x 30, 5-frame moving average.
- **First grasp frame**: the same rule as the earlier grasp analysis (`analyze_grasp_points.detect_first_grasp`): the
  gripper has been >= 30 for 5 frames (open), the close is confirmed once the gripper falls 10 below its open peak, and the
  grasp frame is the start of that continuous decline (per-frame drop > 0.15).
- Window: 3 s before to 1 s after the grasp. Coverage per episode = min(3 s, grasp frame / 30); it is stored in the CSV.
- Peak speed = maximum speed in the 3 s before the grasp. **Slow phase** = length of the final contiguous stretch before
  the grasp with speed < 30% of that peak. **Mean abs jerk** = norm over the 5 joints of the third finite difference of the
  5-frame-smoothed positions (x 30^3), averaged over the 3 s before the grasp (also over the 3 s after the grasp).
- Gripper closes: hysteresis 30 / 15; the "rule" count excludes a close made within 50 joint units of the mean expert
  start pose (return to home, addendum 01).
- "Common window" = the shortest recovery coverage (2.73 s) applied to every episode.
- Tests: two-sided Mann-Whitney U. Medians [IQR]. n: expert 50, new expert 20, recovery 20.

## Results
| metric | original expert | new expert | recovery | p rec vs new | p rec vs exp | p new vs exp |
|---|---|---|---|---|---|---|
| Peak speed before grasp (units/s) | 110 [83.0–128] | 129 [119–140] | 47.5 [36.7–54.1] | <0.001 | <0.001 | 0.025 |
| Slow phase before grasp (s) | 1.37 [1.00–1.57] | 0.95 [0.68–1.32] | 0.50 [0.36–0.65] | 0.001 | <0.001 | 0.034 |
| Mean abs jerk before grasp (units/s^3) | 1561 [1366–1751] | 2344 [2023–2518] | 2029 [1625–2498] | 0.208 | <0.001 | <0.001 |
| Peak speed, common 2.7 s window | 93.9 [66.3–120] | 123 [106–138] | 47.5 [36.2–54.1] | <0.001 | <0.001 | 0.017 |
| Slow phase, common 2.7 s window (s) | 1.25 [0.91–1.52] | 0.80 [0.49–1.11] | 0.50 [0.36–0.65] | 0.038 | <0.001 | 0.002 |
| Mean abs jerk, common 2.7 s window | 1490 [1286–1673] | 2370 [1924–2464] | 2079 [1643–2511] | 0.379 | <0.001 | <0.001 |
| Mean speed in the last 1 s before grasp (units/s) | 5.94 [3.09–9.96] | 15.3 [13.0–23.1] | 9.91 [6.92–12.1] | 0.002 | 0.026 | <0.001 |
| Peak speed in the 3 s after grasp (units/s) | 64.0 [58.1–72.1] | 64.5 [59.4–73.3] | 68.2 [62.9–77.9] | 0.298 | 0.097 | 0.721 |
| Mean abs jerk in the 3 s after grasp | 1525 [1245–1713] | 1773 [1520–2039] | 1887 [1690–2166] | 0.310 | <0.001 | 0.006 |
| Peak speed in the whole episode (units/s) | 156 [146–161] | 148 [141–155] | 156 [141–163] | 0.091 | 0.871 | 0.008 |
| Arm path length from episode start to grasp (joint units) | 240 [231–251] | 226 [206–248] | 57.8 [47.9–69.7] | <0.001 | <0.001 | 0.083 |
| Arm path length in the last 3 s before grasp (joint units) | 124 [83.9–148] | 184 [142–206] | 45.8 [40.4–54.0] | <0.001 | <0.001 | 0.003 |
| Pre-grasp coverage (s, max 3) | 3.00 [3.00–3.00] | 3.00 [3.00–3.00] | 3.00 [3.00–3.00] | 0.163 | 0.026 | 1.000 |
| Time from episode start to grasp (s) | 7.25 [6.52–7.68] | 7.23 [6.89–7.57] | 3.72 [3.24–4.50] | <0.001 | <0.001 | 0.785 |
| Gripper closes (all) | 2.00 [2.00–2.00] | 2.00 [2.00–2.00] | 2.00 [2.00–2.00] | 0.342 | 1.000 | 0.121 |
| Gripper closes (return-to-home excluded) | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 0.342 | 0.121 | 1.000 |
| Frames per episode | 598 [598–598] | 598 [598–598] | 456 [421–496] | <0.001 | <0.001 | 0.548 |
| First-frame distance to start pose | 6.31 [4.03–7.84] | 8.23 [6.60–10.4] | 217 [212–222] | <0.001 | <0.001 | 0.014 |

Files: `data_comparison_metrics.csv` (one row per episode), `data_comparison_summary.csv`, `data_comparison_curves.png`.

### What the data say
1. **Coverage (H1, descriptive).** Every recovery episode has about 3 s before the grasp (coverage 2.73–3.0 s), so the
   3 s window is complete for all of them. But a recovery episode contains only ~3.7 s from its first frame to the grasp
   (median), against ~7.2 s for both expert groups, and an arm path of 58 joint units from the first frame to the grasp
   (median) against 240 (original expert) and 226 (new expert). The first frame is ~217 joint units from the start pose
   (experts: 6–8). So the recovery data have no part of the approach from the start pose, roughly the first half of it.
2. **Speed before the grasp is much lower in the recovery data** (peak 48 vs 129 for new expert and 110 for original
   expert; same result in the common window), and the slow phase is shorter (0.5 s vs 0.95 and 1.37 s). **This is mostly the
   phase confound of H1**: the recovery window shows the final fine positioning (arm path in the last 3 s: 46 vs 184 / 124
   units), whereas the expert window still contains the end of the fast reach. It does not by itself show that the human is
   slower.
3. **Speed and smoothness after the grasp are comparable.** Peak speed in the 3 s after the grasp: 68 (recovery) vs 64 / 64
   (p = 0.30 vs new expert; 0.10 vs original); peak speed over the whole episode: 156 vs 148 / 155 (p = 0.09 / 0.87).
4. **Jerk**: before the grasp, recovery is not different from the new expert (2029 vs 2344, p = 0.21) and higher than the
   original expert (1561, p < 0.001); after the grasp 1887 vs 1773 (p = 0.31) and 1525 (p < 0.001). The **new expert
   demonstrations themselves are jerkier and faster than the original expert** (jerk 2344 vs 1561, peak speed 129 vs 110,
   both p <= 0.025), so both added datasets differ from the original 50 in the same direction.
5. Gripper closes: 1 per episode in all three groups (rule count; 2 including the final return-to-home close); no
   retries in the recovery grasps.

### H2: not supported (as a general "slower / less smooth" effect)
Pre-set reading: H2 would be supported if, relative to the new expert, recovery had a lower peak speed AND a longer slow phase
or higher jerk (p < 0.05). Observed: lower peak speed (p < 0.001), but a **shorter** slow phase (p = 0.001) and no jerk
difference (p = 0.21). The lower pre-grasp speed is explained by what the window contains (H1), and speed after the grasp
and over the whole episode is similar. The data do show H1 in the descriptive sense above (no approach from the start pose).
Whether that missing part actually explains the retest result is what Task B (truncating the new expert data) is meant to test.

## Training settings (items 4)
Read from `train_config.json` of both checkpoints and the training logs. The two configs are **identical except for the
dataset** (same baseline checkpoint `act_red_cube_50_100000`, 5000 steps, batch 8, lr 5e-6, seed 42, no image transforms, ACT
chunk 100, `sample_weighting = None`).
- No upsampling and no sampling weights: LeRobot's `EpisodeAwareSampler` draws every frame uniformly with shuffling
  (`drop_n_last_frames` = 0), so each group's share per epoch equals its share of frames.
- Frames: R's dataset 39,314 = 29,901 expert (76.1%) + 9,413 recovery (23.9%). C's dataset 41,861 = 29,901 expert (71.4%)
  + 11,960 new expert (28.6%).
- 5000 steps x 8 = 40,000 samples = 1.02 epochs for R and 0.96 epochs for C, so each frame is seen about once. Expected
  samples from the added data: R about 9,580, C about 11,430 (**C sees about 19% more samples from its added data**, and its added
  data have 27% more frames, because the new expert episodes are complete and longer: 598 vs 455 frames median).
- The "20 episodes" are matched in number and position, not in frames. This is an additional difference between R and C
  that the planned C_trunc control (Task B) would remove, because truncated expert episodes are about as long as the recovery ones.

## Caveats
- n = 20 per added dataset; the first-grasp rule was tuned for full episodes, but all 60 episodes have a detected grasp.
- Speed and jerk are computed from the measured follower state (not the leader command), at 30 fps.
- The 20 new expert demonstrations are faster and jerkier than the original 50 (mean abs jerk before the grasp 2344 vs 1561, peak speed 129 vs 110 units/s, both p <= 0.025). Model C's result may therefore also contain this difference, not only the difference in data content between recovery and new expert.
- The comparison is of the training data only; it does not show how the policies use these data.

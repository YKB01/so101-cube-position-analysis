# Rewind-target selection: offline prototype

Goal: after a policy failure, choose a *rewind target* from a cache of recent joint states, so that the arm can move there
automatically and a human can start the correction from it. Target = the last cached state that is (a) in distribution
(nearest-neighbour distance to the 50 expert demos <= tau), (b) gripper open, (c) estimated task progress <= p.
The selection function is `rewind_target.py` (numpy/scipy only, no LeRobot dependency); this folder holds the offline evaluation.

**Limitations that apply to everything below**
- "In distribution" uses **joint states only** (5 arm joints, each divided by its expert standard deviation). Images are not used,
  so a state can be in distribution while the scene (e.g. the cube position) is not what the expert saw.
- **Units were not read from the recording configuration.** `meta/info.json` has no unit field. I assumed the LeRobot default
  (`SOFollowerConfig.use_degrees = True`): arm joints in degrees, gripper 0-100. The value ranges agree with this; if the data were
  recorded with another setting, the "deg" numbers in this report change, the normalised distances do not.
- The rewind itself is not executed here; only the choice of target is evaluated, on 29 recorded failures (18 baseline-policy
  episodes from `eval_expert_smooth`, 11 retest trials: R 9 + C 2 failures, with the T36 correction).
- Thresholds (tau, the gripper thresholds, the empty-grasp threshold) were chosen on the same recordings that are evaluated here.
  There is no held-out test.

## Step 1 - expert index (50 demos, 29,901 frames)
| item | value |
|---|---|
| progress 0 | arm-motion onset: smoothed arm speed > 15 units/s for >= 10 frames, moved back to where speed was <= 3 (median frame 81, range 42-129; 37.5% of the pre-grasp frames are idle) |
| progress 1 | first gripper close (rule of `analyze_grasp_points.detect_first_grasp`; median frame 218); linear in between; later frames are `post_grasp` and not matched |
| **tau (default)** | **0.449**: 95th percentile of the leave-one-episode-out nearest-neighbour distance over the moving pre-grasp frames (90th: 0.368, 99th: 0.724) |
| tau (sensitivity) | 0.404: same percentile over all pre-grasp frames incl. the idle start |
| gripper "open" (rewind target) | >= 40 (readings are bimodal: closed ~3, open plateau 44-68); grasp detection keeps 30 |
| progress estimate | leave-one-episode-out MAE 0.091 (about 0.41 s) on moving frames; 0.04 for true progress < 0.2, 0.15 for > 0.8 |

Figures: `step1_fig1_distance_distribution.png`, `step1_fig2_progress_scatter.png`, `step1_fig3_gripper_distribution.png`.
Details: `step1_log.txt`. With the default tau, 5% of the moving expert frames are beyond tau; for the worst single episode 39% of its frames are.

## Step 2 - offline run on the failed trajectories

### Triggers (rule-based, simulated offline)
- **Empty grasp.** 1.0 s after each gripper close (hysteresis 30/15; returns to the home pose excluded, as in addendum 01) the gripper
  reading is averaged over frames +26..+30. A held cube reads 4.5-6.5 (43 successful trials, last close), an empty close 2.6-3.7.
  **One threshold separates them: 4.2** - 0/43 held-cube closes flagged (0%); 45/53 closes of failed trials flagged (85%).
  The 8 not flagged: E00 4.28 and E01 4.21 (just above the threshold), E18 4.80, T36 4.99, T47 4.88 and T46 5.22 (inside the held-cube range:
  a cube held loosely or at the edge looks like a held cube), T29 11.9 and T46 7.25 (gripper stopped half-closed).
  The held-cube class is a proxy label (last close of a successful trial); the failure class is also a proxy (it contains
  grasped-and-dropped closes). Figure: `step2_fig_empty_grasp_reading.png`; sweep: `step2_empty_threshold_sweep.csv`.
  Eval-only held-cube readings (n = 6) are 5.3-6.1, retest (n = 37) 4.5-6.5.
- **>= 3 gripper closes.**
- **Timeout** (end of the episode) for failures that trigger neither.
- Earliest trigger per failure: **27/29 empty grasp, 0 by close count, 2 timeout** (T21 grasped-then-dropped, T46 cube pushed away;
  both have two closes with readings above 4.2). Close count >= 3 occurs in 1/29 failures and never first.
- **False triggers in successful trials:** 4/43 (9%), all retest (T03, T04, T39, T42; 4/37 of the retest successes, 0/6 baseline):
  an earlier empty grasp before a self-recovered success. A deployed system would have interrupted these.
  (A false trigger = trigger before the cube release of a successful trial.)

### Targets (`rewind_targets.csv`, `rewind_summary.csv`; default tau 0.449, 15 s cache)
Cache = the 15 s of states before the trigger. A target was found in **29/29 failures for every setting** (shallow p = 0.9, deep p = 0.4,
shallow by time, and p = 0.3 / 0.5 / 0.7).

| dataset | setting | found | rewind duration, median (range) | NN distance, median | joint distance trigger->target, median |
|---|---|---|---|---|---|
| eval_expert_smooth (18) | shallow p=0.9 | 18/18 | 1.9 s (1.5-4.3) | 0.17 | 12 deg |
| | deep p=0.4 | 18/18 | 4.8 s (3.7-5.4) | 0.20 | 64 deg |
| | shallow, >= 1.0 s lead | 18/18 | 1.75 s (1.5-3.7) | 0.17 | 9 deg |
| retest (11) | shallow p=0.9 | 11/11 | 2.4 s (0.8-7.3) | 0.30 | 21 deg |
| | deep p=0.4 | 11/11 | 6.0 s (0.8-7.8) | 0.19 | 104 deg |
| | shallow, >= 1.0 s lead | 11/11 | 2.4 s (1.0-7.3) | 0.28 | 21 deg |
| all (29) | shallow / deep / lead | 29/29 each | 2.0 s / 4.8 s / 1.9 s | 0.20 / 0.20 / 0.20 | 12 / 91 / 11 deg |

Sensitivity (`rewind_targets_sensitivity.csv`, `rewind_sensitivity_summary.csv`): median rewind duration by p: 0.3 -> 5.3 s, 0.4 -> 4.8 s,
0.5 -> 4.3 s, 0.7 -> 2.8 s, 0.9 -> 2.0 s. With tau = 0.404 the target frame changes in 4/29 trials for the shallow setting (up to 24 frames)
and 1/29 (1 frame) for the deep setting; 29/29 are still found.

**Shallow p = 0.9 versus shallow by time (>= 1.0 s before the trigger, open, in distribution):** the same frame in 18/29 trials. In 10/29 the
p-rule target is earlier (by 2 to 68 frames, i.e. 0.07-2.3 s; E16 and E18 by 2.3 s because the states just before the close have estimated
progress > 0.9 although the time rule accepts them), in 1/29 (T21) later by 7 frames. The two shallow definitions are therefore
interchangeable in most trials; the time rule is simpler and does not depend on the progress estimate that is least reliable near the grasp
(MAE 0.15 for true progress > 0.8). The estimated progress at the p = 0.9 targets has a median of only 0.73 (max 0.87).

### Joint contributions near tau
No trial ended without a target, so the "no target" case did not occur. Near tau (chosen state with distance >= 0.8 tau) there are 8 distinct
states in 6 trials. Share of the squared distance per joint (mean): pan 0.18, lift 0.10, elbow 0.14, wrist_flex 0.25, **wrist_roll 0.32**
(uniform would be 0.20). Dominant joint per state: wrist_roll in 3 states of 2 trials (E03 70%, T47 77% / 44%); the other 5 states: lift (2), pan, wrist_flex, elbow.
In degrees the deviations are small for these joints (mean wrist_roll 1.2 deg, wrist_flex 3.0 deg, against 6.7 deg lift and 8.1 deg elbow).
For reference, in the expert leave-one-out check the frames above tau are dominated by wrist_roll in 124/338 and wrist_flex in 101/338 (mean share
0.38 and 0.20). So the borderline decisions are not *mainly* wrist_roll in these failures, but the two joints with the smallest spread
(wrist_roll sd 5.6 deg, wrist_flex 16.7 deg) carry more of the distance than their physical deviation suggests, and they also shape tau itself.
A floor on the per-joint standard deviation (or a cap on one joint's contribution) is an option if this matters on the robot; not applied here.
Table: `rewind_joint_contrib.csv`.

## Answers to the four questions
**1. Are tau and the progress estimate reliable?** Mostly, with limits. Leave-one-episode-out, tau = 0.449 (95th percentile, moving frames) flags
5% of expert frames by construction; the recorded failures stay mostly in distribution before the trigger (median target distance 0.20), so tau does
not reject the useful targets. Changing tau to 0.404 moves 4 shallow targets and never loses one. The progress estimate has MAE 0.09 (about 0.4 s of
the 4.5 s approach) and is accurate early (0.04) but poor close to the grasp (0.15 at true progress > 0.8; the estimate often stalls on a plateau
while the arm hovers). The deep setting (p = 0.4) is therefore well determined (94% agreement of "progress <= 0.4" with the truth), the shallow
setting by progress (87% agreement for p = 0.9) less so; the time-based shallow rule avoids that weakness. Both depend on joint states only.

**2. How many failures does the rule-based detection cover, and how many false triggers?** Empty-grasp detection fires first in 27/29 failures (93%);
the close-count rule adds nothing here; the remaining 2 (a dropped cube, a pushed-away cube) are only caught by the timeout, i.e. not early. 4/43
successful trials (9%, all retest) would be interrupted by a false empty-grasp trigger. The empty-grasp threshold separates held-cube closes from
empty closes cleanly in this data (0/43 false alarms), but it was set on the same data and 4 of the 8 non-flagged closes of failed trials read like a held cube.

**3. Are shallow and deep targets visibly different?** Yes for the empty-grasp failures. `step2_figB_camera_views.png` (E12, T19, T46): in E12 and T19 the shallow
target shows the gripper open at the cube just before the close (joint distance 6 and 22 deg from the trigger); the deep target shows the arm still high in
the approach (94-106 deg away). For the timeout trigger T46 the two targets are almost the same view (7.3 vs 7.8 s back), because the arm left the
expert distribution for ~5 s after the failure and the last in-distribution open state is the same approach pass; this also happens in E01, E16 and T21
(shallow and deep within 1 s of each other). Also note that the rewind restores the arm only: a failed attempt can have moved the cube (T46 is a "cube pushed away" failure),
which joint states do not show.

**4. Is the 15 s cache enough?** Here yes. Every target was found with 15 s and the unlimited-buffer result is identical; the largest rewind duration is 7.8 s
(deep) and 7.3 s (shallow). Caveat: the early empty-grasp trigger fires 8-15 s into the trial, so the cache usually contains the whole approach. A
failure that is only detected late (the 2 timeout cases are the only examples; both still within 7.8 s) could need more than 15 s of history,
especially if the policy keeps retrying off-distribution for a long time.

## Files
`rewind_target.py` (module; `select_rewind_target`, `select_rewind_target_lead`, `joint_contributions`, `load_index`),
`rewind_index.npz`, `rewind_params.json`, `step1_build_index.py`, `step2_offline_targets.py`, `rewind_targets.csv`,
`rewind_false_triggers.csv`, `rewind_targets_sensitivity.csv`, `rewind_sensitivity_summary.csv`, `rewind_summary.csv`,
`rewind_joint_contrib.csv`, `step2_close_readings.csv`, `step2_empty_threshold_sweep.csv`, `step2_figure_trials.csv`, step1/step2 logs,
figures A (`step2_figA_distance_progress.png`), B (`step2_figB_camera_views.png`), C (`step2_figC_rewind_duration.png`).

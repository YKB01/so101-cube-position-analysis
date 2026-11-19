# Collection protocol: human corrections after an automatic DEEP rewind

Written before any collection; not changed during the collection (deviations go into the log at the end of the file).
Status: inference settings confirmed (option A, section 3); collection starts after the pre-collection checks (section 3b) and with `git_dirty = false`.

## 1. Purpose
Collect 20 corrected episodes in which the policy first fails on its own, the arm is then rewound automatically to a **deep** target (earlier part of the approach), and a human completes the task from there.
These data are compared later with the 20 DAgger corrections of 10/15 (`rollout_so101_red_cube_hil_front_20261015_134746`), which all start in the middle of the task (first frame about 217 joint units from the start pose,
about 3.7 s of motion before the grasp). Whether and how to train on them is a separate decision with its own pre-registration; this protocol only covers collection.
Data layout is the same as the existing intervention dataset (features, `intervention=True`, task string), checked by `scripts/verify_rewind_dataset.py`.

## 2. Datasets
| | repo id | content |
|---|---|---|
| corrections | `YKB0123/rewind_deep_20` | the human part, `intervention=True` (training-compatible) |
| rewind segments | `YKB0123/rewind_deep_20_rewind_seg` | the automatic rewind (gripper opening to the settled target), `intervention=False`, marked in `rewind_segment_meta.csv` |
| per-episode metadata | `results/rewind_collect/run_<time>/rewind_meta.csv` | trigger, rewind level, errors, covariates (section 7), discards |
| policy-phase log | `results/rewind_collect/run_<time>/episodes/ep_NNNN/` | video + states + actions of the whole autonomous phase, trigger / aligned frames |
Collection ends when **20 corrections are saved** (discarded episodes do not count). Test datasets (`rewind_deep_test*`) are not part of the collection.

## 3. Policy and inference settings (confirmed: A, the settings of the existing intervention data)
- Checkpoint (autonomous phase): baseline `~/Checkpoints/act_red_cube_50_100000` (expert 50 only; 6/24 success in the 11/4 smooth evaluation, 7/24 default; it fails often enough to yield corrections).
- Settings: `chunk_size=100, n_action_steps=100, temporal_ensemble_coeff=None` (open-loop execution of whole 100-step chunks), `--inference.type=sync`, `--device=cuda`, 30 fps, interpolation multiplier 1 - the settings of the 10/15 DAgger collection
  (`lerobot-rollout --strategy.type=dagger --policy.path=.../act_red_cube_50_100000 --inference.type=sync`, no `--policy.n_action_steps` / `--policy.temporal_ensemble_coeff`, i.e. the checkpoint's saved config).
  The retest, the C_trunc test and the earlier `--policy_only` runs used `n_action_steps=1, temporal_ensemble_coeff=0.01`; this collection deliberately does not.
- Run with `--config protocol/rewind_collect_deep_config.json` (it holds these values; `policy_config.json` of the run records what was effective, and the run prints and checks them against the request).
- Offline check of the trigger rule on the 11/4 baseline evaluations: the empty-grasp trigger fires in 16/24 episodes with the default settings (7 successes recorded, 17 failures) and in 18/24 with the evaluation settings (6 successes, 18 failures).
- **ACT action queue.** With `n_action_steps=100` the policy keeps up to 99 queued actions. At every trigger the queue is cleared (`engine.reset()` -> `policy.reset()`, plus the action interpolator) so that no action of the old chunk is executed during the rewind or the human correction;
  the number of actions still in the queue at the trigger (`queue_remaining_at_trigger`) and after the clear (`queue_len_after_clear`, must be 0) are recorded. Before the first action of every episode the queue is cleared and verified empty (`queue_len_at_start`, must be 0; otherwise the episode is aborted).

## 3b. Checks before the collection
1. **Two `--policy_only` runs with setting A** (about 3 episodes each is enough):
   `python collect_rewind.py --policy_only --policy_path ~/Checkpoints/act_red_cube_50_100000 --n_action_steps 100 --temporal_ensemble_coeff none --compare_model baseline_default`
   Pass when: the printed configuration says "MATCHES the requested settings" with `n_action_steps 100`, `temporal_ensemble_coeff None`, `has_temporal_ensembler False`; the network runs once per 100 ticks (the action queue is refilled every 100 ticks, queue length up to 99: open-loop chunks; counted from the queue, not from timing, because the observation pre-processing costs ~17 ms on every tick);
   control rate about 30 Hz; the first close time is within the 11/4 baseline evaluation with the default settings (median 6.33 s, IQR 6.23-6.42, range 5.9-7.8 s) and the speed curve looks like its reference.
2. `scripts/run_trial.py PRACTICE` once or twice, then `python scripts/check_trial_covariates.py`: the covariates are written and the control-loop period is not longer than in the earlier trials (new camera-reading thread).
3. `git status` clean for tracked files (`git_dirty` false in `config_used.json`).

## 4. Environment
- Robot: follower `/dev/ttyACM0` (`YKBFollowerR`), leader port as verified before the session; camera `front`, opencv index 0, 640x480, 30 fps, MJPG, warmup 10 s.
- **Camera: automatic exposure and automatic white balance, not locked** (like all training and test data). The room lights stay on, the curtains stay open. Nothing is changed on the camera.
- **Daytime, 10:00-16:00; avoid after 16:00.** The script only warns outside the window and marks the episodes `outside_time_window` (never blocks); such episodes are deviations (log below) and are reported separately from the main summary.
- Covariates are recorded for every episode (section 7). The lighting analysis (`lighting_analysis/report_lighting.md`) found two automatic-exposure states (table gray ~115 and >= 120), the bright one more common late in the afternoon; they are recorded, not controlled or filtered.
- Code: commit recorded in `config_used.json` (`git_commit`, `git_dirty` must be false at the start of a session). Rule thresholds: `rewind_offline/rewind_params.json` (tau 0.449, gripper open >= 40, empty-grasp threshold 4.2, timeout 30 s, 15 s cache).

## 5. Cube positions
The 20 cube positions of the 10/15 recovery episodes (`protocol/rewind_collect_positions.json`, R01-R20 = recovery episode 1-20; x 337-407, y 177-253 px), one episode each, in the random order of `protocol/rewind_collect_order.csv` (seed in `rewind_collect_order_seed.txt`).
Placement with `python scripts/place_cube_overlay.py --order protocol/rewind_collect_order.csv --positions protocol/rewind_collect_positions.json --log results/rewind_collect_placement_log.csv` (3 px tolerance, same method as the retest).
A discarded episode is repeated at the same position; the position pointer of the script advances only after a saved episode.

## 6. Procedure per episode
1. RESET (teleoperation, not recorded): arm to the start pose, cube placed at the announced position (`NEXT POSITION: Rxx` is printed), `s`.
2. AUTONOMOUS: the policy acts (max 30 s). Triggers, whichever comes first: **empty grasp** (gripper reading < 4.2 one second after a close), **timeout** 30 s, or **manual `t`** only when the policy fails in a way the rules do not catch (cube pushed out of reach, arm stuck); every manual trigger is noted in the session log with the reason.
3. TRIGGERED: the policy action queue is cleared (remaining count recorded), the arm holds, the gripper opens to 57 and waits 0.5 s; the deep target is selected (original rule progress <= 0.4; if that is less than 2 s before the shallow target, progress <= 0.2 at least 2 s earlier; otherwise the start pose). All three levels are accepted; the level is stored (`level`).
4. REWIND: the measured trajectory is replayed backwards at <= 1.5 deg per joint per tick (points closer than 2 deg merged), gripper open. If it exceeds 40 s the operator chooses `c` (continue) or `x`.
5. ALIGN: the leader is driven to the follower (hands off the leader); if it does not reach the pose (stall) the torque is released and the operator chooses `c` or `x`. Then `s` (hands on the leader): torque released, recording starts.
6. HUMAN: the operator completes the task from the rewind pose - approach, grasp, place the cube in the box, release - at the normal demonstration speed and style of the expert demonstrations, then `e`.
7. The correction is saved; RESET for the next episode. The operator does not look at the policy's behaviour to decide whether to continue; every episode with a valid trigger is collected.

## 7. Recorded for every episode (`rewind_meta.csv`)
`position_id`, trigger type and time, rewind mode and **level** (`original` / `progress_le_0.2` / `start_pose`), target index, estimated progress, nearest-neighbour distance, planned and actual rewind time, joint distance trigger -> target,
settle result and final error (per joint, normalised, ratio to tau), cube position at the start / at the trigger / after the rewind and the shift from the start, align error and retries, human correction duration, discard flag and reason, and the covariates:
`queue_len_at_start`, `queue_remaining_at_trigger`, `queue_len_after_clear`, the **first-frame table gray** (lighting-analysis table box), window-side gray, whole-frame gray, table R/B, the **camera controls read back while the camera streams** (`cam_auto_exposure`, `cam_exposure_time_absolute`, `cam_white_balance_automatic`, `cam_white_balance_temperature`, `cam_gain`,
`cam_backlight_compensation`, `cam_exposure_dynamic_framerate`, brightness, contrast, saturation, gamma, sharpness, power-line frequency), `outside_time_window`, and the **control-loop period** (mean and max, ms) of the autonomous and the human phase.

## 8. Discard and stop rules
- Discard (`x`, reason recorded): cube pushed out of the workspace; hardware fault (comms error, arm stalled by a collision); accidental key; the operator cannot complete the correction. The reason is written to the log; the episode is repeated at the same position.
- Start-pose episodes are **not** discarded; they are reported with and without them.
- Stop and investigate: 3 discards in a row for hardware reasons; an emergency stop; control-loop max period above 100 ms in two consecutive episodes; leader alignment failing twice in a row.
- No episode is repeated because the result looks unsatisfactory.

## 9. Planned summary (descriptive only)
Per-level counts, trigger types, rewind durations and joint distances, final rewind error against tau, human correction duration, frames per episode, distance of the first correction frame from the start pose (compared with 217 for the 10/15 corrections),
and the lighting covariates (table gray, exposure state) of the collected episodes. No training or test comparison is part of this protocol.

## Deviation log
(none yet)

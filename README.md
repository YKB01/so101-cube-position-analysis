# SO-101 red cube position analysis

Detects the red cube in the first frame of every episode of several LeRobot (v3.0) datasets
(SO-101 single arm, pick the red cube and place it in a box) and relates the 6 evaluation
test positions to the cube positions seen in the training data.

- `analyze_cube_positions.py`: the analysis script. HSV thresholds, ROI, dataset paths and
  success counts are in the CONFIG block at the top.
- `report.md`: detection summary, anomalies (recovery data, mis-placed trials), test position
  table, distance table and correlations.
- `cube_positions.csv`: `dataset, episode_index, cx, cy, area, detected` (pixels, 640x480).
- `cube_positions_detail.csv`: same plus bbox, aspect ratio and anomaly flags.
- `test_positions.csv`, `eval_cluster_assignments.csv`, `test_position_distances.csv`.
- `test_positions_labels.png`, `position_distribution.png`,
  `success_vs_distance_{expert,mixed}_smooth.png`.
- `contact_sheets/`: first frame of every episode with the detection drawn, one sheet per dataset.

## Run

Requires the datasets locally under `~/.cache/huggingface/lerobot/YKB0123/` and an environment
with lerobot (0.5.x), opencv, matplotlib, pandas, scipy, scikit-learn and tabulate.

```bash
python analyze_cube_positions.py                    # all datasets + analysis
python analyze_cube_positions.py --datasets expert  # detection + contact sheet for one dataset
```

Outputs are written to `./cube_position_analysis/` relative to the working directory.
Note: pixel distances are image distances, not metric table distances.

## Grasp point analysis: where does the policy aim?

`analyze_grasp_points.py` detects the first grasp (start of the first gripper close after opening)
in every episode, fits a linear map from the 5 arm joint angles at that frame to the cube pixel
position on the 50 expert episodes (validated with R² and leave-one-out error), and applies it to
the smooth-condition eval datasets to get the position each policy aimed at.

- `success.csv`: episode-level outcome (cube ended in the box = 1). Created with
  `--make-success-template`; 0/4 and 4/4 positions are pre-filled from the aggregate table.
- `grasp_point_analysis/report_grasp.md`: slopes (aim vs. actual), aim errors, direction of the
  error relative to the expert data centre, and success vs. failure.
- `grasp_point_analysis/grasp_points.csv`, `fig1_aim_arrows.png`, `fig2_aim_vs_actual.png`,
  `fig3_error_by_position.png`, plus gripper/regression check plots.

```bash
python analyze_grasp_points.py --make-success-template   # then fill the blank success cells
python analyze_grasp_points.py --expert-only             # grasp detection + regression check
python analyze_grasp_points.py                           # full analysis
```

## Data-volume control experiment and extra right-side test

Control model: expert 50 + 20 newly collected expert episodes, fine-tuned with exactly the settings of the
mixed model (expert 50 + recovery 20): same baseline checkpoint, 5,000 steps, batch size 8, lr 5e-6, seed 42.

- `check_new_expert_positions.py` -> `new_expert_position_check/`: first-frame cube positions of the 20 new
  expert episodes, distance to the 6 test positions, contact sheet, distribution plot.
- `resolve_recovery_cube_positions.py` -> `recovery_position_check/`: cube position of the 9 recovery
  episodes whose cube is hidden in frame 0 (first 10-frame still, visible window before the first grasp),
  validated on the 11 episodes where frame 0 already shows the cube.
- `control_training_check/`: training loss curves of the two fine-tuning runs.
- `analyze_right_side_test.py` -> `right_side_test_analysis/` (`--stage detect | analyze | all`): extra test at
  two positions on the right side (8 trials each). Clusters the first-frame cube positions into the two test
  positions and compares the distance to the nearest training sample for both training sets.
  Result and criteria: `right_side_test_analysis/report_right_side.md`.

Pixel distances are image distances, not distances on the table.

## Control-model aim analysis and lateral correction before the grasp

Both analyses reuse the grasp-frame rule and the "joint angles -> cube pixel position" regression of
`analyze_grasp_points.py` (`grasp_point_analysis/expert_regression.csv`).

- **Four models side by side** (`analyze_grasp_points.py`): the control model's standard evaluation
  (`eval_expert70_smooth`, expert 50 + new expert 20) was added next to expert / mixed / recovery-only. Its
  first-frame cube positions are in `grasp_point_analysis/extra_cube_positions.csv` (assigned to the nearest
  standard test position; `test_positions.csv` is unchanged). Results of the other three models are unchanged.
  New: `grasp_point_analysis/mean_err_x_by_position.csv`, four-panel `fig1_aim_arrows.png`.
- **`success.csv`** now also has rows for the control model's standard evaluation and for the two right-side
  extra-test datasets (`edge_mixed`, `edge_control`). `python analyze_grasp_points.py --add-success-rows` appends
  rows for datasets that are not in the file yet (existing rows untouched); blank cells are treated as unknown and
  never replaced by placeholders.
- **Lateral correction before the grasp** (`analyze_correction.py` -> `correction_analysis/`):
  `--stage curves` plots arm speed and shoulder_pan before the grasp (used to choose the rule: the end of the
  approach is the first time after the speed peak that the arm speed stays below 20 units/s for 5 frames);
  `--stage metrics` computes per-episode metrics (`correction_metrics.csv`), summary tables, figures C/D/E, tests
  (Mann-Whitney U with rank-biserial effect size, threshold sensitivity) and `report_correction.md`.

## Rewind-target selection (offline prototype)
First step towards "failure -> automatic rewind -> human correction" on the real robot. `rewind_target.py` (repo root) is a
LeRobot-free module: `select_rewind_target(buffer_states, trigger_idx, p, index, params)` returns the last cached state that is in
distribution (nearest-neighbour distance to the 50 expert demos <= tau, joint states only), has an open gripper (>= 40) and an estimated task
progress <= p; `select_rewind_target_lead(...)` is the time-based shallow variant (>= 1.0 s before the trigger). Everything else is in `rewind_offline/`:
- `python rewind_offline/step1_build_index.py` builds `rewind_index.npz` / `rewind_params.json` (tau, progress, gripper thresholds; figures and log).
- `python rewind_offline/step2_offline_targets.py` simulates rule-based triggers on the baseline-policy evaluation and the right-side retest
  (29 failures, 43 successes), selects shallow / deep targets, and writes `rewind_targets.csv`, summaries, sensitivity tables and figures A-C.
- `rewind_offline/report_rewind_offline.md` answers whether tau/progress are reliable, how many failures the triggers cover, whether the two
  depths look different, and whether a 15 s cache is enough. Joint units are inferred (not read from the recording configuration).

### Corrected deep rule (phase 0)
`select_rewind_target_deep` in `rewind_target.py`: original deep target (progress <= 0.4); if it is less than 2.0 s earlier than the time-rule shallow target,
the last state with progress <= 0.2 that is at least 2.0 s earlier; otherwise the start pose (median of the idle expert frames before the arm moves). The
result records which level was used. `python rewind_offline/phase0_deep_fix.py` re-runs the 29 failed trajectories (`phase0_deep_fix.csv`, `phase0_summary.txt`, `phase0_figB_four_trials.png`).

Replay path (phase 0 addendum): `plan_rewind_path` replays the recorded measured trajectory backwards and merges waypoints closer than
`rewind_resample_min_step_deg` (2.0 deg, joint-space distance) so that standing still and small back-and-forth adjustments are skipped;
`estimate_rewind_steps` gives the number of control steps at a per-joint limit (default 1.5 deg/step). Start-pose targets (`level = "start_pose"`)
are not discarded: the level is stored in the per-episode metadata so that results can be reported with and without them.

## Online collection with automatic rewind (`collect_rewind.py`)
Real-robot script for "policy fails -> automatic rewind -> human corrects". Parameters: `collect_rewind_config.json` (defaults, overridable as `--key value`)
and `rewind_offline/rewind_params.json` (rule thresholds). Hardware-free parts are in `rewind_control.py` (state buffer, online trigger, step-limited rewind controller).
- Full flow: `python collect_rewind.py --rewind_mode deep --policy_path <checkpoint>/pretrained_model --repo_id YKB0123/rewind_deep_20`
  (rewind segments go to `<repo_id>_rewind_seg`, `intervention=False`, marked in `rewind_segment_meta.csv` in that dataset's root).
  State machine: RESET (teleop, not recorded) -> AUTONOMOUS (ACT, 15 s state cache; triggers: empty grasp, 30 s timeout, manual) -> TRIGGERED (hold, open gripper,
  0.5 s) -> target selection (shallow = time rule, deep = corrected rule) -> REWIND (measured trajectory backwards, <= 1.5 deg/joint/tick, gripper open) ->
  ALIGN (leader driven to the follower's measured pose, tolerance and retries in the config) -> WAIT_START (leader torque stays on until `s`) ->
  HUMAN (leader released, recorded with `intervention=True`, `e` to end) -> RESET. Per-episode metadata: `<out_dir>/rewind_meta.csv` (includes the `level` field).
- Keys: `s` start, `t` manual trigger, `e` end correction, `x` discard, `c` continue after a prompt, `r` rewind to the start pose when no target, `q` quit (RESET), SPACE emergency stop
  (sends the measured pose once, then no commands; torque stays on; quitting needs confirmation and first moves the arm back to its start-up pose).
- `--dry_run`: follower only (no leader, no camera, no cube); the follower replays a recorded failed trial (`--dry_run_source E12|T21|...`, step-limited) to build a realistic cache,
  then manual trigger / online trigger -> open gripper -> select -> rewind -> settle; writes sent vs measured per waypoint and the final error (per joint, normalised, compared with tau).
- `--replay`: no hardware; the 29 recorded failed trials drive trigger detection, selection, path planning and the controller (`rewind_offline/replay_*`).
- Software tests with simulated hardware: `rewind_offline/test_collect_dry_fake.py`, `test_collect_full_fake.py`, `test_dataset_roundtrip.py`.

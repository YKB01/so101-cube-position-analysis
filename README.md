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

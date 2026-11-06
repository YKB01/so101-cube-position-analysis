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

# Grasp point analysis: where does the policy aim?

## Grasp detection

Signal: `observation.state` gripper. Open state = gripper >= 30 for 5 frames; a close is confirmed when the gripper drops 10 below the open peak; the grasp frame is the start of the continuous decline (per-frame drop > 0.15) that leads into that close, so earlier slow drifts or pauses are skipped. Arm joints are read from `observation.state` at that frame. `arm_speed_at_grasp` = mean arm joint speed over the 5 frames around it.

Expert: grasp detected in 50/50 episodes; grasp time 5.3-9.2 s.

## Regression check (expert, joints at grasp -> first-frame cube pixel)

n = 50, features = shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos

| | x | y |
|---|---|---|
| R² (fit) | 0.898 | 0.871 |
| R² (leave-one-out) | 0.867 | 0.837 |
| LOO mean abs error (px) | 3.0 | 2.8 |
| predict-mean baseline mean abs error (px) | 7.8 | 7.3 |

LOO Euclidean error: mean 4.6 px, median 4.2 px, max 10.4 px (baseline mean 11.6 px).

Coefficients (px per joint unit):

|    |   shoulder_pan.pos |   shoulder_lift.pos |   elbow_flex.pos |   wrist_flex.pos |   wrist_roll.pos |
|:---|-------------------:|--------------------:|-----------------:|-----------------:|-----------------:|
| cx |              2.64  |               0.017 |            0.186 |            0.232 |            0.001 |
| cy |             -0.209 |               2.216 |            1.887 |            0.675 |           -0.005 |

Caveat: expert cube positions only span ~40 px, and all test positions lie at or beyond the edge of that range, so aim positions for eval episodes are partly an extrapolation of the linear map.

## Eval episodes without a gripper close (no aim position)

- none

## Eval episodes where the arm was still moving at the grasp frame (speed > 20)

The policy closed the gripper while the arm was moving, so the aim position is a snapshot of a moving arm rather than a settled grasp pose.

- eval_recovery_smooth episode 0 (bottom-left): speed 30.0, err 16.9 px
- eval_recovery_smooth episode 1 (bottom-left): speed 20.1, err 15.9 px
- eval_recovery_smooth episode 3 (bottom-left): speed 23.1, err 11.3 px
- eval_recovery_smooth episode 6 (bottom-right): speed 22.7, err 35.0 px

## Summary per eval dataset (smooth condition)

Expert data centre (mean cube position of 50 expert episodes): (381.3, 211.9) px. `cos_to_centre` = cosine between the error vector (aim - cube) and the vector from the cube to the expert data centre; +1 = the policy aims towards the centre, 0 = unrelated, -1 = away from it.

| dataset              |   n | slope_x [95% CI]   | slope_y [95% CI]   |   mean err_norm (px) |   median err_norm (px) |   mean cos_to_centre | share cos > 0   |
|:---------------------|----:|:-------------------|:-------------------|---------------------:|-----------------------:|---------------------:|:----------------|
| eval_expert_smooth   |  24 | 0.39 [0.35, 0.44]  | 0.31 [0.27, 0.34]  |                 20.7 |                   23.1 |                 0.96 | 100%            |
| eval_mixed_smooth    |  24 | 0.70 [0.61, 0.79]  | 0.50 [0.44, 0.56]  |                 16.4 |                   16   |                 0.78 | 96%             |
| eval_recovery_smooth |  24 | 0.72 [0.57, 0.87]  | 0.56 [0.37, 0.75]  |                 17.7 |                   14.4 |                 0.41 | 75%             |
| eval_expert70_smooth |  24 | 0.61 [0.53, 0.69]  | 0.60 [0.49, 0.71]  |                 18   |                   18.1 |                 0.66 | 83%             |

### Per test position

| dataset              | position     |   n |   mean_err |   mean_err_x |   mean_err_y |   mean_cos_to_centre |
|:---------------------|:-------------|----:|-----------:|-------------:|-------------:|---------------------:|
| eval_expert70_smooth | top-left     |   4 |      29.6  |        10.78 |        27.46 |                 0.99 |
| eval_expert70_smooth | mid-left     |   4 |      18.71 |         6.07 |        17.62 |                 0.81 |
| eval_expert70_smooth | bottom-left  |   4 |       9.09 |        -1.26 |         8.33 |                -0.71 |
| eval_expert70_smooth | top-right    |   4 |      22.38 |        -6.43 |        21.14 |                 0.98 |
| eval_expert70_smooth | mid-right    |   4 |      13.75 |       -12.31 |         5.36 |                 0.97 |
| eval_expert70_smooth | bottom-right |   4 |      14.53 |       -13.71 |        -3.19 |                 0.91 |
| eval_expert_smooth   | top-left     |   4 |      27.11 |         7.54 |        25.69 |                 0.97 |
| eval_expert_smooth   | mid-left     |   4 |      13.82 |         2.8  |        13.5  |                 0.78 |
| eval_expert_smooth   | bottom-left  |   4 |       6.3  |         4.86 |        -3.99 |                 1    |
| eval_expert_smooth   | top-right    |   4 |      29.87 |       -14.97 |        25.83 |                 1    |
| eval_expert_smooth   | mid-right    |   4 |      22.86 |       -18.95 |        12.78 |                 1    |
| eval_expert_smooth   | bottom-right |   4 |      24.53 |       -22.7  |        -9.21 |                 0.99 |
| eval_mixed_smooth    | top-left     |   4 |      26.88 |        12.16 |        23.73 |                 0.99 |
| eval_mixed_smooth    | mid-left     |   4 |      18.19 |         0.31 |        18.17 |                 0.59 |
| eval_mixed_smooth    | bottom-left  |   4 |       3.48 |         2.55 |         1.62 |                 0.32 |
| eval_mixed_smooth    | top-right    |   4 |      26.03 |        -4.13 |        25.6  |                 0.96 |
| eval_mixed_smooth    | mid-right    |   4 |      13.02 |        -6.68 |        11.11 |                 0.93 |
| eval_mixed_smooth    | bottom-right |   4 |      10.59 |       -10.27 |        -2.44 |                 0.91 |
| eval_recovery_smooth | top-left     |   4 |      12.58 |        -4.4  |        11.27 |                 0.71 |
| eval_recovery_smooth | mid-left     |   4 |      16.41 |        -8.27 |        13.3  |                 0.05 |
| eval_recovery_smooth | bottom-left  |   4 |      12.09 |       -11.95 |        -1.16 |                -0.7  |
| eval_recovery_smooth | top-right    |   4 |      15.8  |       -14.54 |         4.71 |                 0.81 |
| eval_recovery_smooth | mid-right    |   4 |      12.16 |        -9.94 |        -1.42 |                 0.59 |
| eval_recovery_smooth | bottom-right |   4 |      37.36 |       -30.58 |       -21.25 |                 1    |

### Horizontal aim error (mean err_x = aim x - cube x, px) by position, all four models

Positive = aim to the right of the cube, negative = to the left. Right-side positions first.

| position     |   expert |   mixed |   recovery-only |   control (expert 70) |
|:-------------|---------:|--------:|----------------:|----------------------:|
| mid-right    |    -18.9 |    -6.7 |            -9.9 |                 -12.3 |
| bottom-right |    -22.7 |   -10.3 |           -30.6 |                 -13.7 |
| top-right    |    -15   |    -4.1 |           -14.5 |                  -6.4 |
| top-left     |      7.5 |    12.2 |            -4.4 |                  10.8 |
| mid-left     |      2.8 |     0.3 |            -8.3 |                   6.1 |
| bottom-left  |      4.9 |     2.5 |           -12   |                  -1.3 |

Full table: `grasp_point_analysis/mean_err_x_by_position.csv`.

## Success vs. failure

| dataset              |   success |   n |   mean_err |   median_err |   mean_cos_to_centre |
|:---------------------|----------:|----:|-----------:|-------------:|---------------------:|
| eval_expert70_smooth |         0 |   7 |      17.75 |        18.8  |                 0.95 |
| eval_expert70_smooth |         1 |  17 |      18.12 |        17.34 |                 0.54 |
| eval_expert_smooth   |         0 |  18 |      21.79 |        23.49 |                 1    |
| eval_expert_smooth   |         1 |   6 |      17.61 |        15    |                 0.84 |
| eval_mixed_smooth    |         0 |   8 |      21.71 |        25.97 |                 0.95 |
| eval_mixed_smooth    |         1 |  16 |      13.69 |        14.74 |                 0.7  |
| eval_recovery_smooth |         0 |  14 |      20.63 |        17.22 |                 0.27 |
| eval_recovery_smooth |         1 |  10 |      13.67 |        11.83 |                 0.61 |

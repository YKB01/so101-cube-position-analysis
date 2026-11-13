# Lateral correction before the grasp ("two-stage motion")

Pixel values are image pixels converted from shoulder_pan with the expert regression (2.64 px per unit); they are not distances on the table. Positive = to the right.

## Rule and metrics

- Grasp frame: first gripper close (same rule as `analyze_grasp_points.py`).
- End of the approach (`t_reach_end`): after the speed peak in the 4 s before the grasp, the first frame from which the arm speed (norm of the 5 arm joints, 5-frame moving average) stays below 20 units/s for 5 frames.
- `correction_pan` / `correction_px`: change of shoulder_pan from `t_reach_end` to the grasp frame (x regression coefficient for px). `correction_time_s`: duration of that interval.
- `n_submovements`: local speed peaks with prominence >= 8 units/s between `t_reach_end` and the grasp. `n_gripper_closes`: gripper closes over the whole episode (open >= 30, closed <= 15).
- `pre_aim_err_x` / `grasp_aim_err_x`: horizontal aim error (regression aim x minus cube x) at `t_reach_end` and at the grasp. The regression is only reliable near the grasp pose, so `pre_aim_err_x` is only computed if `t_reach_end` is at most 2 s before the grasp (otherwise NaN).
- Datasets: right-side extra test (mixed and control model) and the 6 standard positions (expert, mixed, control; smooth). Per-episode values: `correction_metrics.csv`.

### Episodes excluded from the correction statistics

No stop before the grasp (the arm never stayed below the threshold; excluded, not set to 0): edge_control ep 4 (right-bottom).
`pre_aim_err_x` = NaN because the reach end is more than 2 s before the grasp: none.

## Summary by dataset and position

| dataset              | position     |   n |   n_stop_found |   median_correction_px |   mean_correction_px |   q25_correction_px |   q75_correction_px |   median_correction_time_s |   median_n_submovements |   n_with_submovement |   mean_gripper_closes |   n_with_regrasp |   median_pre_aim_err_x |   n_pre_aim |   median_grasp_aim_err_x |   median_pan_drift_after_grasp_px |   n_success |
|:---------------------|:-------------|----:|---------------:|-----------------------:|---------------------:|--------------------:|--------------------:|---------------------------:|------------------------:|---------------------:|----------------------:|-----------------:|-----------------------:|------------:|-------------------------:|----------------------------------:|------------:|
| edge_mixed           | right-bottom |   8 |              8 |                   3.02 |                 3.71 |                2.67 |                4.35 |                       1.1  |                       0 |                    0 |                  2    |                0 |                 -20.33 |           8 |                   -17.36 |                              1.74 |           8 |
| edge_mixed           | right-top    |   8 |              8 |                   5.57 |                 6.32 |                5.45 |                6.5  |                       1.23 |                       0 |                    0 |                  2    |                0 |                 -15.84 |           8 |                   -10.45 |                              2.9  |           4 |
| edge_control         | right-bottom |   8 |              7 |                   0    |                 0    |               -0.46 |                0.35 |                       0.27 |                       0 |                    0 |                  1.75 |                2 |                  -9.7  |           7 |                   -10.99 |                             -0.12 |           2 |
| edge_control         | right-top    |   8 |              8 |                   0    |                 0.03 |               -0.06 |                0.06 |                       0.25 |                       0 |                    0 |                  2.62 |                5 |                 -10.37 |           8 |                   -10.6  |                              0    |           0 |
| eval_expert_smooth   | bottom-left  |   4 |              4 |                   1.74 |                 2.03 |                1.51 |                2.26 |                       0.93 |                       0 |                    0 |                  2    |                0 |                   3.44 |           4 |                     4.45 |                              0.58 |           0 |
| eval_expert_smooth   | bottom-right |   4 |              4 |                   2.2  |                 1.97 |                1.86 |                2.32 |                       1.1  |                       0 |                    0 |                  2    |                0 |                 -23.33 |           4 |                   -22.09 |                              0.35 |           0 |
| eval_expert_smooth   | mid-left     |   4 |              4 |                   1.86 |                 1.97 |                1.57 |                2.26 |                       1.28 |                       0 |                    0 |                  2    |                0 |                   1.53 |           4 |                     2.87 |                              1.39 |           4 |
| eval_expert_smooth   | mid-right    |   4 |              4 |                   2.67 |                 2.55 |                2.38 |                2.84 |                       1.42 |                       0 |                    0 |                  1.75 |                0 |                 -20.81 |           4 |                   -19.01 |                              1.28 |           0 |
| eval_expert_smooth   | top-left     |   4 |              4 |                   1.97 |                 1.97 |                1.57 |                2.38 |                       1.48 |                       0 |                    0 |                  2    |                0 |                   6.23 |           4 |                     7.47 |                              1.39 |           2 |
| eval_expert_smooth   | top-right    |   4 |              4 |                   2.67 |                 2.61 |                2.26 |                3.02 |                       1.27 |                       0 |                    0 |                  1.75 |                0 |                 -16.91 |           4 |                   -14.97 |                              0.46 |           0 |
| eval_mixed_smooth    | bottom-left  |   4 |              4 |                  -0.12 |                -0.06 |               -0.29 |                0.12 |                       0.43 |                       0 |                    0 |                  1.25 |                0 |                   1.6  |           4 |                     1.4  |                             -0.12 |           3 |
| eval_mixed_smooth    | bottom-right |   4 |              4 |                   1.16 |                 1.22 |                1.1  |                1.28 |                       0.65 |                       0 |                    0 |                  2    |                0 |                 -11.08 |           4 |                   -10.38 |                              0.46 |           2 |
| eval_mixed_smooth    | mid-left     |   4 |              4 |                   0.46 |                 0.41 |                0.12 |                0.75 |                       0.57 |                       0 |                    0 |                  2    |                0 |                   0.41 |           4 |                     0.29 |                              0    |           4 |
| eval_mixed_smooth    | mid-right    |   4 |              4 |                   9.86 |                 9.92 |                8.47 |               11.31 |                       1.48 |                       0 |                    0 |                  2    |                0 |                 -16.69 |           4 |                    -7.31 |                              3.02 |           4 |
| eval_mixed_smooth    | top-left     |   4 |              4 |                   2.44 |                 2.09 |                1.8  |                2.73 |                       1.37 |                       0 |                    0 |                  2    |                0 |                  10.82 |           4 |                    11.78 |                              0    |           2 |
| eval_mixed_smooth    | top-right    |   4 |              4 |                   6.27 |                 6.9  |                6.03 |                7.14 |                       1.33 |                       0 |                    0 |                  2    |                0 |                 -11.17 |           4 |                    -4.62 |                              1.74 |           1 |
| eval_expert70_smooth | bottom-left  |   4 |              4 |                  -1.86 |                -2.09 |               -2.67 |               -1.28 |                       0.58 |                       0 |                    0 |                  2    |                0 |                   1.04 |           4 |                    -1.98 |                             -0.35 |           4 |
| eval_expert70_smooth | bottom-right |   4 |              4 |                   0    |                 0.06 |               -0.06 |                0.12 |                       0.53 |                       0 |                    0 |                  3    |                4 |                 -12.59 |           4 |                   -13.25 |                              0.35 |           0 |
| eval_expert70_smooth | mid-left     |   4 |              4 |                  -2.2  |                -3.02 |               -3.6  |               -1.62 |                       0.57 |                       0 |                    0 |                  2    |                0 |                   9.73 |           4 |                     5.29 |                             -0.93 |           4 |
| eval_expert70_smooth | mid-right    |   4 |              4 |                   0.35 |                 0.41 |                0    |                0.75 |                       0.23 |                       0 |                    0 |                  2.25 |                1 |                 -11.98 |           4 |                   -11.87 |                              0    |           3 |
| eval_expert70_smooth | top-left     |   4 |              4 |                  -0.35 |                -0.46 |               -0.87 |                0.06 |                       1.03 |                       0 |                    0 |                  2.5  |                2 |                  11.51 |           4 |                    11.06 |                              0    |           4 |
| eval_expert70_smooth | top-right    |   4 |              4 |                  -0.12 |                 0    |               -0.23 |                0.12 |                       0.25 |                       0 |                    0 |                  2.25 |                1 |                  -4.82 |           4 |                    -5.24 |                             -0.46 |           2 |

## Right-side extra test: mixed vs. control (Mann-Whitney U, two-sided)

`rank_biserial` = P(mixed > control) - P(mixed < control), ranges from -1 to 1. Eight episodes per position and model, so per-position tests have little power.

| scope        | metric                   |   n_a |   n_b |   median_a |   median_b |   median_diff |     U |        p |   rank_biserial |
|:-------------|:-------------------------|------:|------:|-----------:|-----------:|--------------:|------:|---------:|----------------:|
| pooled       | correction_px            |    16 |    15 |     4.9896 |     0      |        4.9896 | 239.5 |   0      |          0.9958 |
| pooled       | correction_time_s        |    16 |    15 |     1.2    |     0.2667 |        0.9333 | 240   |   0      |          1      |
| pooled       | n_submovements           |    16 |    15 |     0      |     0      |        0      | 120   | nan      |          0      |
| pooled       | pre_aim_err_x            |    16 |    15 |   -17.6891 |   -10.3244 |       -7.3647 |   4   |   0      |         -0.9667 |
| pooled       | abs_pre_aim_err_x        |    16 |    15 |    17.6891 |    10.3244 |        7.3647 | 236   |   0      |          0.9667 |
| pooled       | grasp_aim_err_x          |    16 |    16 |   -15.3898 |   -10.598  |       -4.7918 |  81   |   0.0797 |         -0.3672 |
| pooled       | abs_grasp_aim_err_x      |    16 |    16 |    15.3898 |    10.598  |        4.7918 | 175   |   0.0797 |          0.3672 |
| pooled       | abs_err_change           |    16 |    15 |    -4.3541 |     0.3038 |       -4.6579 |   1   |   0      |         -0.9917 |
| pooled       | pan_drift_after_grasp_px |    16 |    16 |     2.4368 |     0      |        2.4368 | 242   |   0      |          0.8906 |
| right-top    | correction_px            |     8 |     8 |     5.5698 |     0      |        5.5698 |  64   |   0.0008 |          1      |
| right-top    | correction_time_s        |     8 |     8 |     1.2333 |     0.25   |        0.9833 |  64   |   0.0009 |          1      |
| right-top    | n_submovements           |     8 |     8 |     0      |     0      |        0      |  32   | nan      |          0      |
| right-top    | pre_aim_err_x            |     8 |     8 |   -15.8388 |   -10.3715 |       -5.4672 |   4   |   0.0019 |         -0.875  |
| right-top    | abs_pre_aim_err_x        |     8 |     8 |    15.8388 |    10.3715 |        5.4672 |  60   |   0.0019 |          0.875  |
| right-top    | grasp_aim_err_x          |     8 |     8 |   -10.4484 |   -10.598  |        0.1496 |  41   |   0.3823 |          0.2812 |
| right-top    | abs_grasp_aim_err_x      |     8 |     8 |    10.4484 |    10.598  |       -0.1496 |  23   |   0.3823 |         -0.2812 |
| right-top    | abs_err_change           |     8 |     8 |    -4.8359 |     0.248  |       -5.0839 |   0   |   0.0002 |         -1      |
| right-top    | pan_drift_after_grasp_px |     8 |     8 |     2.901  |     0      |        2.901  |  64   |   0.0007 |          1      |
| right-bottom | correction_px            |     8 |     7 |     3.017  |     0      |        3.017  |  55.5 |   0.0017 |          0.9821 |
| right-bottom | correction_time_s        |     8 |     7 |     1.1    |     0.2667 |        0.8333 |  56   |   0.0014 |          1      |
| right-bottom | n_submovements           |     8 |     7 |     0      |     0      |        0      |  28   | nan      |          0      |
| right-bottom | pre_aim_err_x            |     8 |     7 |   -20.3299 |    -9.7015 |      -10.6283 |   0   |   0.0003 |         -1      |
| right-bottom | abs_pre_aim_err_x        |     8 |     7 |    20.3299 |     9.7015 |       10.6283 |  56   |   0.0003 |          1      |
| right-bottom | grasp_aim_err_x          |     8 |     8 |   -17.3628 |   -10.9876 |       -6.3751 |   0   |   0.0002 |         -1      |
| right-bottom | abs_grasp_aim_err_x      |     8 |     8 |    17.3628 |    10.9876 |        6.3751 |  64   |   0.0002 |          1      |
| right-bottom | abs_err_change           |     8 |     7 |    -2.4423 |     0.4114 |       -2.8537 |   1   |   0.0006 |         -0.9643 |
| right-bottom | pan_drift_after_grasp_px |     8 |     8 |     1.7406 |    -0.116  |        1.8566 |  57   |   0.0099 |          0.7812 |

### Pooled right-side extra test, the three quantities used by the criteria

- Correction (`correction_px`), mixed - control: median 4.99 vs 0.00 px, difference +4.99 px, p = 2.34e-06, rank-biserial +1.00 (n = 16 vs 15).
- Does the horizontal error shrink from the end of the approach to the grasp (mixed model, paired Wilcoxon on |err_x|)? median |err_x| 17.7 -> 15.4 px, median shrink +4.4 px, p = 3.05e-05 (n = 16).
- Aim error at the end of the approach (`|pre_aim_err_x|`), mixed vs control: median 17.7 vs 10.3 px, p = 4.98e-06, rank-biserial +0.97.

- Control model, same paired test: median |err_x| 10.3 -> 10.5 px (median change +0.30 px, i.e. larger at the grasp), p = 0.0125 (n = 15); the change is consistent but tiny.

### Success vs. failure (Mann-Whitney U)

Groups smaller than 3 are not tested. Success = the cube ended in the box.

| dataset              | metric              |   n_a |   n_b |   median_a |   median_b |   median_diff |    U |      p |   rank_biserial |
|:---------------------|:--------------------|------:|------:|-----------:|-----------:|--------------:|-----:|-------:|----------------:|
| edge_mixed           | correction_px       |    12 |     4 |     4.5255 |     5.3378 |       -0.8123 | 23.5 | 1      |         -0.0208 |
| edge_mixed           | grasp_aim_err_x     |    12 |     4 |   -15.744  |   -11.171  |       -4.573  | 16   | 0.3791 |         -0.3333 |
| edge_mixed           | abs_grasp_aim_err_x |    12 |     4 |    15.744  |    11.171  |        4.573  | 32   | 0.3791 |          0.3333 |
| edge_mixed           | pre_aim_err_x       |    12 |     4 |   -18.4042 |   -15.9207 |       -2.4834 | 10   | 0.1033 |         -0.5833 |
| eval_expert_smooth   | correction_px       |     6 |    18 |     2.2047 |     2.2047 |        0      | 52.5 | 0.9466 |         -0.0278 |
| eval_expert_smooth   | abs_grasp_aim_err_x |     6 |    18 |     2.9285 |    15.348  |      -12.4195 |  1   | 0      |         -0.9815 |
| eval_mixed_smooth    | correction_px       |    16 |     8 |     1.2764 |     1.8566 |       -0.5802 | 61.5 | 0.9025 |         -0.0391 |
| eval_mixed_smooth    | abs_grasp_aim_err_x |    16 |     8 |     3.9918 |     9.6794 |       -5.6876 | 30   | 0.0382 |         -0.5312 |
| eval_expert70_smooth | correction_px       |    17 |     7 |    -0.9283 |     0      |       -0.9283 | 25.5 | 0.0319 |         -0.5714 |
| eval_expert70_smooth | abs_grasp_aim_err_x |    17 |     7 |     5.5716 |    13.2335 |       -7.662  | 12   | 0.0014 |         -0.7983 |

## Sensitivity to the speed threshold

The criteria applied to the pooled extra test with other thresholds (hold = 5 frames):

|   thresh |   n_found_mixed |   n_found_control |   median_correction_px_mixed |   median_correction_px_control |   median_diff |   p |   error_shrinks_mixed_p |   median_abs_pre_mixed |   median_abs_pre_control | conclusion   |
|---------:|----------------:|------------------:|-----------------------------:|-------------------------------:|--------------:|----:|------------------------:|-----------------------:|-------------------------:|:-------------|
|       10 |              16 |                10 |                        3.133 |                          0     |         3.133 |   0 |                       0 |                 17.142 |                   11.216 | undecided    |
|       20 |              16 |                15 |                        4.99  |                          0     |         4.99  |   0 |                       0 |                 17.689 |                   10.324 | undecided    |
|       30 |              16 |                15 |                        7.542 |                          0.464 |         7.078 |   0 |                       0 |                 19.997 |                   10.484 | correction   |

With episodes without a stop counted as 0 px (instead of excluded): median difference +5.0 px, p = 1.49e-06.

## Conclusion

Criteria: (1) mixed correction larger than control by at least 5 px (median) with p < 0.05, and the horizontal error of the mixed model shrinks after the end of the approach -> "the mixed model performed a lateral correction before the grasp"; (2) corrections not clearly different, but the mixed model's aim error at the end of the approach already smaller by at least 5 px (p < 0.05) -> "the difference comes from the aim during the approach"; (3) otherwise "cannot be told apart".

**Result (rule: speed < 20 for 5 frames): The two explanations cannot be told apart.**

Result of the same criteria at other speed thresholds:

- threshold 10: The two explanations cannot be told apart. (median difference +3.13 px)
- threshold 20: The two explanations cannot be told apart. (median difference +4.99 px)
- threshold 30: The mixed model performed a lateral correction before the grasp. (median difference +7.08 px)

Note on the margin: the pooled median of the mixed model's 16 corrections is the average of the 8th and 9th sorted values (4.87 and 5.11 px), i.e. it falls in the gap between the two positions (right-bottom about 3 px, right-top about 5.6 px). A difference of 4.99 px against the 5 px criterion is therefore a boundary case, while the direction is unambiguous (rank-biserial +1.00). Per position the criterion is met at right-top (+5.57 px) and not at right-bottom (+3.02 px).

Additional observation: after the grasp frame (first 0.5 s of the gripper closing) the median shoulder_pan drift is control (extra test): +0.0, mixed (extra test): +2.4 px (positive = to the right).

Limits: eight episodes per position and model; the aim error uses a regression fitted on expert grasp poses and is only a proxy for where the gripper is relative to the cube; the speed signal is noisy (encoder quantisation, about 5 units/s at rest), which is why the rule is also shown for other thresholds.

# Red cube position analysis

## Detection summary

| dataset | episodes | detected | flagged |
|---|---|---|---|
| expert | 50 | 50 | 0 |
| recovery | 20 | 11 | 9 |
| eval_expert_smooth | 24 | 24 | 0 |
| eval_expert_default | 24 | 24 | 0 |
| eval_mixed_smooth | 24 | 24 | 0 |
| eval_mixed_default | 24 | 24 | 0 |
| eval_recovery_smooth | 24 | 24 | 0 |
| eval_recovery_default | 24 | 24 | 0 |

## Episodes where the cube was NOT detected (cx/cy = NaN)

- recovery episode 5
- recovery episode 8
- recovery episode 9
- recovery episode 10
- recovery episode 12
- recovery episode 13
- recovery episode 15
- recovery episode 16
- recovery episode 17

## Recovery (HIL) data: anomalous first-frame detections

Recovery episodes start at the moment of human intervention, so the cube may already have been moved or be inside the gripper. Flags: not_detected, area_ratio outside (0.5, 2.0) (vs. expert median), bbox aspect > 1.8, second red blob >= 0.3 x largest.

- episode 5: not_detected
- episode 8: not_detected
- episode 9: not_detected
- episode 10: not_detected
- episode 12: not_detected
- episode 13: not_detected
- episode 15: not_detected
- episode 16: not_detected
- episode 17: not_detected

Flagged recovery episodes are EXCLUDED from `d_expert_recovery`; `d_expert_recovery_all` includes every detected recovery episode. Note: an unflagged detection only means the blob looks like a cube; whether it is still at its initial position must be checked visually on the contact sheet.

## Other flagged detections (non-recovery datasets)

- none

## Test positions (k-means on eval first frames)

| position     |    cx |    cy |   n_episodes |   mean_dist_to_center |   max_dist_to_center | label_zh   |
|:-------------|------:|------:|-------------:|----------------------:|---------------------:|:-----------|
| bottom-right | 405.6 | 227.1 |           23 |                   3.2 |                  9   | 右下       |
| mid-left     | 361.3 | 196.5 |           25 |                   4   |                  9.8 | 左中       |
| top-right    | 401   | 179.2 |           24 |                   3.5 |                  9   | 右上       |
| bottom-left  | 361.3 | 226.7 |           25 |                   3.3 |                 10.1 | 左下       |
| mid-right    | 401.8 | 198.3 |           24 |                   3.3 |                  6.4 | 右中       |
| top-left     | 362.4 | 178   |           23 |                   4.5 |                  9.7 | 左上       |

Labels are assigned in IMAGE coordinates (left/right by x, top/mid/bottom by y). Verify against `test_positions_labels.png` before trusting the success mapping.

### Trials whose first-frame position differs from the usual position of that episode index

All eval datasets follow the same episode order (4 consecutive episodes per position). Listed trials were run with the cube at a different test position than the other datasets used for that episode index; the success table (x/4 per position) may count them under the planned position.

- eval_expert_default episode 19: at **mid-left** (cx=358, cy=200); expected top-left
- eval_recovery_default episode 4: at **bottom-left** (cx=357, cy=226); expected bottom-right

Episodes per (eval dataset, test position):

| dataset               |   bottom-left |   bottom-right |   mid-left |   mid-right |   top-left |   top-right |
|:----------------------|--------------:|---------------:|-----------:|------------:|-----------:|------------:|
| eval_expert_default   |             4 |              4 |          5 |           4 |          3 |           4 |
| eval_expert_smooth    |             4 |              4 |          4 |           4 |          4 |           4 |
| eval_mixed_default    |             4 |              4 |          4 |           4 |          4 |           4 |
| eval_mixed_smooth     |             4 |              4 |          4 |           4 |          4 |           4 |
| eval_recovery_default |             5 |              3 |          4 |           4 |          4 |           4 |
| eval_recovery_smooth  |             4 |              4 |          4 |           4 |          4 |           4 |


## Distance from each test position to the nearest training sample (pixels)

| position     | label_zh   |    cx |    cy |   d_expert | nearest_expert   |   d_expert_recovery | nearest_expert_recovery   |   d_expert_recovery_all | nearest_expert_recovery_all   |   succ_mixed_smooth |   succ_mixed_default |   succ_expert_smooth |   succ_expert_default |   succ_recovery_smooth |   succ_recovery_default |
|:-------------|:-----------|------:|------:|-----------:|:-----------------|--------------------:|:--------------------------|------------------------:|:------------------------------|--------------------:|---------------------:|---------------------:|----------------------:|-----------------------:|------------------------:|
| top-left     | 左上       | 362.4 | 178   |       22.4 | expert:39        |                11.2 | recovery:18               |                    11.2 | recovery:18                   |                   2 |                    4 |                    2 |                     1 |                      4 |                       3 |
| mid-left     | 左中       | 361.3 | 196.5 |        3.9 | expert:39        |                 3.9 | expert:39                 |                     3.9 | expert:39                     |                   4 |                    3 |                    4 |                     4 |                      0 |                       0 |
| bottom-left  | 左下       | 361.3 | 226.7 |        5   | expert:42        |                 4.1 | recovery:1                |                     4.1 | recovery:1                    |                   3 |                    4 |                    0 |                     0 |                      1 |                       1 |
| top-right    | 右上       | 401   | 179.2 |       17.1 | expert:48        |                17.1 | expert:48                 |                    17.1 | expert:48                     |                   1 |                    0 |                    0 |                     0 |                      2 |                       1 |
| mid-right    | 右中       | 401.8 | 198.3 |        6   | expert:47        |                 6   | expert:47                 |                     6   | expert:47                     |                   4 |                    4 |                    0 |                     2 |                      2 |                       1 |
| bottom-right | 右下       | 405.6 | 227.1 |       11.6 | expert:2         |                11.6 | expert:2                  |                    11.6 | expert:2                      |                   2 |                    2 |                    0 |                     0 |                      1 |                       0 |

Pixel distances are measured in the image and are not metric: perspective makes the same pixel distance correspond to a larger table distance at the top (far) side of the frame. For scale: the cube side is ~17 px (sqrt of median expert blob area).

## Success rate vs. distance (smooth condition)

- expert_smooth: x = `d_expert`, Spearman rho = -0.17, p = 0.75 (n = 6, descriptive only)
- mixed_smooth: x = `d_expert_recovery`, Spearman rho = -0.88, p = 0.02 (n = 6, descriptive only)

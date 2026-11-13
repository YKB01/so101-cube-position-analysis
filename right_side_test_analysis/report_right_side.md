# Right-side extra test: can training-data position coverage explain the model difference?

Pixel distances are image distances, not distances on the table. Because of perspective, the same pixel distance corresponds to a longer table distance at the top (far side) of the frame. Cube width in the image: 17 px.

## Extra test positions

Extra test: 2 positions x 8 trials per model (smooth inference), first-frame cube positions clustered with k-means (k=2), labelled by y.

| dataset      |   right-bottom |   right-top |
|:-------------|---------------:|------------:|
| edge_control |              8 |           8 |
| edge_mixed   |              8 |           8 |

| dataset      | position     |   n |    cx |    cy |   cx_std |   cy_std |   max_dist |
|:-------------|:-------------|----:|------:|------:|---------:|---------:|-----------:|
| edge_control | right-bottom |   8 | 407.6 | 207.1 |      2.1 |      0.5 |        5.3 |
| edge_control | right-top    |   8 | 402.7 | 182.5 |      0.9 |      2.6 |        6   |
| edge_mixed   | right-bottom |   8 | 414   | 208.1 |      2.5 |      1.2 |        6.5 |
| edge_mixed   | right-top    |   8 | 404.6 | 181.7 |      3.5 |      0.6 |        8.1 |

- right-bottom: centres differ by dx -6.4, dy -1.0 px (|d| = 6.5 px) between the two models.
- right-top: centres differ by dx -1.9, dy +0.8 px (|d| = 2.1 px) between the two models.

Trials more than 10 px from their cluster centre (possible misplacement): none.
Episodes whose cube was hidden in frame 0 and resolved from a later still window: 0.

## Distance from the extra test positions to the training sets

mixed = expert 50 + recovery 20, control = expert 50 + new expert 20. `x_margin_px` = test x minus the x of the rightmost training sample (positive: the test position lies to the right of all training data).

| position     |    cx |    cy | training_set   |   nearest_px | nearest_src     | nearest_method           |   mean3_px |   n_within_17 |   n_within_34 |   x_margin_px | rightmost_src   |
|:-------------|------:|------:|:---------------|-------------:|:----------------|:-------------------------|-----------:|--------------:|--------------:|--------------:|:----------------|
| right-bottom | 410.8 | 207.6 | mixed          |          7.3 | recovery20:10   | first_still_before_grasp |        9.3 |             8 |            43 |           3.5 | recovery20:17   |
| right-bottom | 410.8 | 207.6 | control        |          5   | new_expert20:5  | first_frame              |        8.3 |             7 |            42 |          -0   | new_expert20:5  |
| right-top    | 403.6 | 182.1 | mixed          |          8.4 | recovery20:16   | first_still_before_grasp |       10.4 |             6 |            25 |          -3.6 | recovery20:17   |
| right-top    | 403.6 | 182.1 | control        |          8.2 | new_expert20:15 | first_frame              |       10   |             6 |            25 |          -7.2 | new_expert20:5  |

### Recovery / new expert samples within 34 px (two cube widths) of an extra test position

Expert 50 samples within 34 px: right-bottom 35, right-top 18 (same in both training sets; full list in `training_samples_within_34px.csv`).

| position     | src          |   episode_index |    cx |    cy |   dist_px | method                   |
|:-------------|:-------------|----------------:|------:|------:|----------:|:-------------------------|
| right-bottom | new_expert20 |               5 | 410.8 | 212.6 |       5   | first_frame              |
| right-bottom | recovery20   |              10 | 405.8 | 213   |       7.3 | first_still_before_grasp |
| right-bottom | new_expert20 |              10 | 403.1 | 209.1 |       7.8 | first_frame              |
| right-bottom | recovery20   |               5 | 403.3 | 212.1 |       8.7 | first_still_before_grasp |
| right-bottom | recovery20   |              12 | 405.3 | 218   |      11.8 | first_still_before_grasp |
| right-bottom | new_expert20 |              13 | 408.4 | 219.5 |      12.1 | first_frame              |
| right-bottom | new_expert20 |              12 | 404.4 | 218.9 |      13   | first_frame              |
| right-bottom | recovery20   |              13 | 405.6 | 220.3 |      13.7 | first_still_before_grasp |
| right-bottom | recovery20   |               8 | 402.1 | 196.6 |      14   | first_still_before_grasp |
| right-bottom | recovery20   |              17 | 407.2 | 192.7 |      15.3 | first_still_before_grasp |
| right-bottom | recovery20   |               9 | 402.4 | 193.6 |      16.4 | first_still_before_grasp |
| right-bottom | new_expert20 |              17 | 408.4 | 191.2 |      16.6 | first_frame              |
| right-bottom | new_expert20 |               9 | 400.8 | 194   |      16.8 | first_frame              |
| right-bottom | new_expert20 |               8 | 396.7 | 196.1 |      18.2 | first_frame              |
| right-bottom | recovery20   |              16 | 396.5 | 177.7 |      33.1 | first_still_before_grasp |
| right-top    | new_expert20 |              15 | 402   | 174.1 |       8.2 | first_frame              |
| right-top    | recovery20   |              16 | 396.5 | 177.7 |       8.4 | first_still_before_grasp |
| right-top    | new_expert20 |              17 | 408.4 | 191.2 |      10.3 | first_frame              |
| right-top    | recovery20   |              17 | 407.2 | 192.7 |      11.2 | first_still_before_grasp |
| right-top    | new_expert20 |              16 | 401.5 | 170.9 |      11.4 | first_frame              |
| right-top    | recovery20   |               9 | 402.4 | 193.6 |      11.6 | first_still_before_grasp |
| right-top    | recovery20   |              15 | 392.8 | 176.8 |      12.1 | first_still_before_grasp |
| right-top    | new_expert20 |               9 | 400.8 | 194   |      12.3 | first_frame              |
| right-top    | recovery20   |               8 | 402.1 | 196.6 |      14.6 | first_still_before_grasp |
| right-top    | new_expert20 |               8 | 396.7 | 196.1 |      15.7 | first_frame              |
| right-top    | new_expert20 |              10 | 403.1 | 209.1 |      27.1 | first_frame              |
| right-top    | recovery20   |               5 | 403.3 | 212.1 |      30   | first_still_before_grasp |
| right-top    | recovery20   |              10 | 405.8 | 213   |      31   | first_still_before_grasp |
| right-top    | new_expert20 |               5 | 410.8 | 212.6 |      31.3 | first_frame              |

### Per-episode nearest distance (mean / max over the 8 trials)

| dataset      | position     |   mixed_mean |   mixed_max |   control_mean |   control_max |
|:-------------|:-------------|-------------:|------------:|---------------:|--------------:|
| edge_control | right-bottom |          6.2 |         8.3 |            4.6 |           6.1 |
| edge_control | right-top    |          7   |         8.7 |            7.1 |           8   |
| edge_mixed   | right-bottom |          9.7 |        11.5 |            6   |           6.8 |
| edge_mixed   | right-top    |          8.8 |        12.6 |            8.5 |          10.8 |

Full table: `edge_episode_distances.csv`.

## Comparison with the 6 standard test positions

Offsets of each extra position from the standard positions (dx: positive = further right, dy: positive = further down).

| position     | standard     |   dx_right_px |   dy_down_px |   dist_px |
|:-------------|:-------------|--------------:|-------------:|----------:|
| right-bottom | bottom-right |           5.2 |        -19.5 |      20.2 |
| right-bottom | top-right    |           9.8 |         28.4 |      30   |
| right-bottom | mid-right    |           8.9 |          9.3 |      12.9 |
| right-top    | bottom-right |          -1.9 |        -45.1 |      45.1 |
| right-top    | top-right    |           2.6 |          2.8 |       3.9 |
| right-top    | mid-right    |           1.8 |        -16.2 |      16.3 |

Nearest standard position: right-bottom -> mid-right (+8.9, +9.3) px; right-top -> top-right (+2.6, +2.8) px

## Success rates

| position | mixed model | control model |
|---|---|---|
| right-bottom | 8/8 | 2/8 |
| right-top | 4/8 | 0/8 |

## Conclusion

Criteria: nearest distances within 5 px and counts within 17 px differing by at most 1 -> position coverage does not explain the difference; mixed set closer by more than 5 px or with at least 3 more samples within 17 px -> it may explain it; otherwise undecided.

| position     | code        | why                              |   d_mixed |   d_control |   n17_mixed |   n17_control |
|:-------------|:------------|:---------------------------------|----------:|------------:|------------:|--------------:|
| right-bottom | not_explain | same distance and similar counts |       7.3 |         5   |           8 |             7 |
| right-top    | not_explain | same distance and similar counts |       8.4 |         8.2 |           6 |             6 |

**Position coverage does not explain the difference.** 

Per-trial check (trials as actually run; positive = the control training set is closer): control model at right-bottom: +1.7 px; control model at right-top: -0.1 px; mixed model at right-bottom: +3.7 px; mixed model at right-top: +0.3 px. The control set is closer or equal for the trials that were run, so coverage, if anything, favours the control model, the opposite of the observed success rates.

Notes:

- right-top is only 3.9 px from the standard top-right position (dx +2.6, dy +2.8), so it is not a new position; right-bottom lies +8.9 px right and +9.3 px down of mid-right.
- Both extra positions are within one cube width of at least 6 training samples in both training sets; neither lies far outside the training data.
- The recovery positions that were hidden in frame 0 (9 of 20) come from the first still, visible window before the first grasp, not from the moment of intervention; their positions may differ from where the cube was originally placed.
- Eight trials per position and model: the success difference itself (8/8 vs 2/8) is large, but this analysis can only exclude position coverage, not identify the cause.

Success counts are 8 trials per position per model; this analysis shows where the training data is closer, not why the models behave differently.

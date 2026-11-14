# Right-side retest (pre-registered): model R vs model C

Model R = expert 50 + recovery 20; model C = expert 50 + 20 new expert demos at the recovery positions.
Positions P1–P3 and the protocol were fixed before testing (`protocol/right_side_retest.md`, addenda 01 and 02).
48 trials, 8 per model and position, blind (the A/B key was opened only after all 48 results were recorded; its hash
matches `protocol/model_key.sha256`). Inputs: `results/retest_outcomes.csv`, `results/retest_corrections.csv`.

## Primary result (pre-registered): the prediction (R > C) is **not** supported; the difference goes the other way
| | success | rate (Wilson 95% CI) |
|---|---|---|
| Model R | 15 / 24 | 62.5% (42.7–78.8%) |
| Model C | 22 / 24 | 91.7% (74.2–97.7%) |

R − C = −29.2 points (Newcombe 95% CI −49.9 to −5.2); odds ratio 0.15; **two-sided Fisher exact p = 0.036**.

### The result depends on one trial (T36)
T36 attempt 1 timed out (failure) but no outcome was recorded; the trial was redone and attempt 2 was recorded as success.
A redo is only allowed for interruptions unrelated to the model, so the final result is attempt 1 (failure; operator
confirmed the mistake). The primary analysis uses this corrected result. With T36 as originally recorded (R 16/24):
p = 0.072. The conclusion at alpha = 0.05 flips on this single trial, so the result should be read as suggestive.

## Per position (exploratory, 8 per cell, no multiplicity correction)
| position | R | C | p (two-sided Fisher) |
|---|---|---|---|
| P1 (386, 182) | 3/8 | 7/8 | 0.119 |
| P2 (394, 212) | 4/8 | 8/8 | 0.077 |
| P3 (390, 230) | 8/8 | 7/8 | 1.000 |

R is worse at P1 and P2 and not worse at P3 (the position nearest to the old bottom-right test position). No
position is significant by itself.

## Failure types (9 R + 2 C failures)
| type | R | C |
|---|---|---|
| edge not grasped | 6 | 2 |
| dropped after grasp | 2 | 0 |
| cube pushed away | 1 | 0 |
| arm cannot reach | 0 | 0 |

## Secondary analyses (exploratory; Mann–Whitney, two-sided; medians)
| metric | R | C | p |
|---|---|---|---|
| gripper closes (addendum 01 rule) | 2 | 1 | 0.002 |
| first grasp time (s) | 7.7 | 6.4 | < 0.001 |
| pan correction before the grasp (px) | 3.9 | 0.0 | < 0.001 |
| correction time (s) | 1.15 | 0.40 | < 0.001 |
| aim error x at the first grasp (px) | −9.6 | −3.4 | < 0.001 |
| aim error x at the end of the approach (px) | −10.5 | −3.0 | < 0.001 |

- **Closes**: model C closes once in 21/24 trials; model R closes twice or more in 14/24 trials (C in 3/24). R retries more.
- **Returned home on its own** (policy finished before 30 s): C 22/24, R 10/24. Five R successes ended at about 29.6 s and had
  not yet returned home: they finished just inside the limit after a retry (T02, T03, T39, T42, T43).
- **Speed before the grasp** (`fig2_speed_aligned.png`): R starts moving earlier and decelerates for longer, with a lower
  speed peak (~83 vs ~110 units/s) and a longer final slow phase before the grasp; C approaches faster and stops sharply.
- **Aim error** (joint-to-pixel regression trained on expert data, a proxy): R aims further left of the cube than C
  (about 10 px vs 3 px) at the end of the approach and at the grasp, and corrects more to the right before closing. This
  is the same pattern as in the earlier standard evaluation, but here R is the less successful model.

## Caveats
- Small sample: 8 per cell; confidence intervals are wide; per-position tests are exploratory.
- The T36 correction decides the significance of the primary test (see above).
- Not controlled: the 3 positions were run in blocks (P3, then P1, then P2); within each position A/B order was ABBA/BAAB.
- Placement was within the 3 px tolerance for all 48 trials; mean cube positions of R and C trials differ by < 1 px.
- 10 of the 11 failures are at P1 and P2; success is judged by the operator (cube ends in the box).
- Success needs completion within 30 s; slower approaches (R) are penalised by the time limit as well as by failures.

Files: `success_tests.csv`, `failure_types.csv`, `secondary_mwu.csv`, `close_count_distribution.csv`,
`by_position_model.csv`, `trial_table.csv`, figures `fig1_success.png`, `fig2_speed_aligned.png`, `fig3_closes.png`.

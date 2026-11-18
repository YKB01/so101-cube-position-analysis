# C_trunc vs C (pre-registered): result

C = expert 50 + 20 full new expert demos; C_trunc = expert 50 + the same 20 demos with the early approach removed
(path-aligned truncation). Protocol: `protocol/trunc_test.md`. 32 trials (P1, P2; 8 per model per position),
ABBA/BAAB, blind codes. The sealed key (`protocol/trunc_model_key.json`, A = C, B = C_trunc) was opened only after
all 32 results were recorded; its SHA-256 matches `protocol/trunc_model_key.sha256`.
Session: 2026-11-14 16:29–17:22 (the retest R/C ran 13:45–15:11 the same day).

## Primary (pre-registered): C_trunc is clearly worse than C
| | success | rate (Wilson 95% CI) |
|---|---|---|
| C_trunc | 0 / 16 | 0% (0–19.4%) |
| C (same session, interleaved) | 8 / 16 | 50% (28.0–72.0%) |

Difference −50.0 points (Newcombe 95% CI −72.0 to −20.7); **two-sided Fisher exact p = 0.0024**.
Per position (exploratory): P1 0/8 vs 2/8 (p = 0.47); P2 0/8 vs 6/8 (p = 0.007).

## Secondary 1: against the retest R and C (not interleaved with C_trunc)
| comparison | result | Fisher p |
|---|---|---|
| C_trunc 0/16 vs retest R 7/16 | −43.8 points | 0.007 |
| C_trunc 0/16 vs retest C 15/16 | −93.8 points | 5.7e-8 |

## Secondary 2: drift check — **the session's C was not the same as the retest's C**
This session's C: 8/16; retest C (P1+P2): 15/16; Fisher p = 0.015. So conditions on the day (or the placement/grasp
reliability of the setup) were not equivalent between the retest and this session, and comparisons across the two
time slots must be read with caution. Details:
- C this session by position: P1 2/8 (retest 7/8), P2 6/8 (retest 8/8). Within the session, C was 2/8 in U01–U16 (all P1)
  and 6/8 in U17–U32 (all P2), so position and time are confounded.
- C's time to first grasp is unchanged (median 6.75 s vs 6.65 s, p = 0.76), but its grasp aim is further from the cube
  (x −6.7 vs −4.3 px, p = 0.017; y +13.3 vs +9.6 px, p = 0.048) and the close-count distribution is worse (p = 0.016);
  exploratory, 16 trials each. Cause unknown (camera/robot drift, cube-detector offset, or chance); not tested.
- This session's C (8/16) is numerically the same as retest R (7/16), so the "C_trunc is close to R" comparison with
  R measured earlier cannot be separated from drift. The interleaved comparison (primary) is not affected by drift.

## Interpretation under the pre-registered rule
Prediction (H1): C_trunc lower than C and close to R (7/16).
- Direction vs C: confirmed (0/16 vs 8/16, p = 0.0024).
- Close to R: **not met** — C_trunc is significantly different from R (p = 0.007, not interleaved), and in the opposite
  direction from "close": it is below R.
- The pre-registered rule classifies this as "significantly different from both" → reported as "between / inconclusive,
  not forced", although here C_trunc is below both, not between them.

What can be said: removing the early approach from the new expert demos did **not** produce a model like R; it produced
a model that failed all 16 trials (all at the grasp, "edge not grasped"). Removing the early approach is therefore
not shown to be the cause of R's lower success; whatever C_trunc lacks or distorts is more damaging than R's data
(which has the same missing approach plus human recoveries). Possible reasons (not tested): the truncated episodes start
mid-trajectory at high speed, which may change the learned start behaviour; the truncation removed 1.6 s of typical
approach from the expert-only content; one seed. H1 as formulated ("missing approach explains R's deficit") is not supported;
H3 is not tested by this experiment.

## Failure types and behaviour (exploratory; C_trunc vs C, medians)
| | C_trunc | C |
|---|---|---|
| failures | 16 (all "edge not grasped") | 8 (7 edge not grasped, 1 cube pushed away) |
| time to first grasp | 9.07 s | 6.75 s (p < 0.001) |
| mean arm speed, first 10 s | 23.4 | 27.8 (p < 0.001; no overlap) |
| grasp closes (addendum 01) | 2 in 16/16 | 1 in 9, 2 in 6, 3 in 1 (p = 0.003) |
| aim error x / y at first grasp (px) | −9.7 / +15.4 | −6.7 / +13.3 (p = 0.09 / 0.49) |
| pan correction before grasp (px) | 0.8 | 0.0 (p = 0.10) |

C_trunc starts moving earlier but peaks lower (~75 vs ~117 units/s before the grasp, `fig2_speed_aligned.png`),
i.e. a slower, more hesitant approach, then reaches the cube and misses the grasp. Against retest R (P1+P2): C_trunc first
grasp later (9.07 vs 7.87 s, p < 0.001), less lateral correction (0.8 vs 5.3 px, p < 0.001), same x aim error (−9.7 vs
−10.4 px).

## Offline prediction check (`scripts/offline_prediction_check.py`)
Mean abs error (joint units) of predicted action chunks vs the recorded actions of the 20 new expert demos:
| frames | C | C_trunc | R |
|---|---|---|---|
| removed early approach (60 frames) | 1.75 | 4.26 | 4.13 |
| kept part, after the cut (20 frames) | 1.40 | 1.46 | 4.30 |

C_trunc and R both fail to predict the removed early approach (as expected: they never saw it), C does not (but C
trained on those frames: in-sample). Displacement ratio over the first 30 chunk steps (median): C 1.07, C_trunc 0.83, R 0.99.
This check was run before the test and predicted that C_trunc would be unlike C in the early approach; it does not predict
the 0/16 grasp failure.

## Deviations, limitations
- **Blinding was partly compromised**: after 3 trials the operator saw that one model behaved clearly worse (slow, not
  reaching the cube). Outcome judgments here ("cube in box within 30 s", failure type by addendum 02) are largely
  objective, and the order and placement were fixed in advance, but a bias in notes or placement cannot be excluded.
  The placement error is similar between models (1.5 px vs 1.3 px mean).
- **U19**: attempt 1 (model C) was ended by a Right-arrow press after 3.2 s and redone (attempt 2: success). No reason is
  written in the outcomes file; the protocol allows redoing for reasons unrelated to the model. Excluding U19 would give
  C 7/15 and does not change the primary conclusion (Fisher p = 0.0024).
- 16 trials per model, one training seed per model; no multiplicity correction; the secondary comparisons are exploratory.
- C this session differs from the retest C (drift check, p = 0.015); P1/P2 are confounded with time within the session.
- C_trunc and R differ in more than the missing approach (the new expert data is faster and jerkier; no human recoveries).
- Gripper-close counts for C_trunc are 2 for every trial; the count is a detector-based event count (hysteresis 30/15)
  and was not checked manually against video.

Files: `success_tests.csv`, `failure_types.csv`, `secondary_mwu.csv`, `by_position_model.csv`, `trial_table.csv`,
`fig1_success.png`, `fig2_speed_aligned.png`, `fig3_aim_error.png`, `../retest_analysis/offline_prediction_check.csv`.

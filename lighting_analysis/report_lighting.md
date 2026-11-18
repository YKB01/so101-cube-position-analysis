# Lighting analysis: does the outdoor light explain the 11/14 drift?

Scope: first frame of 373 episodes (training data, every evaluation session, the 11/14 retest and truncation trials), their recording times, the behaviour of the 11/14 trials, an offline
test of how strongly four policies react to brightness / colour changes, and the camera's current controls. Nothing was changed on the camera or on any dataset.
Regions (confirmed): `regions.json` / `regions.png` (table = lower board; near_cube; bg_desk = white desk; bg_top_strip; bg_sheet_right = plastic sheet and wall on the right,
used as the window-side background because the window is at the right-rear of the camera). Gray levels are 0-255; "table gray" is the mean gray of the table box (red cube pixels excluded).

**Limits that apply to everything below.** (1) The camera runs automatic exposure and automatic white balance (see Q4), so the frames show brightness *after* the camera compensated,
not the outdoor light itself; the white desk is already saturated (gray ~250) in every session. (2) Recording times: exact for the 11/14 trials (`meta.json`); for every other
dataset interpolated between the directory time stamp and the last file write (about +-40 s per episode). No dataset had to be left out for lack of a time. (3) Brightness is measured on the first frame; the
policy sees all frames, but inside a trial the table region stays within 1.4 gray levels of the first frame (the whole-frame mean drops by ~8 once the arm enters the view). (4) 16-24 trials per series: correlations are
exploratory, 57 light-behaviour correlations were computed without multiplicity correction (about 3 would reach p < 0.05 by chance).

## Q1. How much does the light change during the day, and when is it most stable?
- **Small.** Whole-frame gray of the first frame: 120-138 over all 373 frames; standard deviation inside a session 1.1-2.8 levels (coefficient of variation 1.1-2.8%), table gray 2.2-4.7 levels (1.9-4.3%).
  Figures: `fig1_brightness_vs_time.png` (brightness against time of day, one panel per date), `fig2_color_vs_time.png`.
- **No time-of-day trend inside a session.** 11/14 retest (13:44-15:10): -0.34 gray/hour (p = 0.50); 11/14 truncation test (16:29-17:21): -0.17 gray/hour (p = 0.90). The truncation test was *not* darker as the sun set; the other
  sessions show no consistent trend either (`session_trends.csv`).
- **Most stable:** the 11/14 retest (whole-frame CV 1.1%, table 2.2%, 13:44-15:10) and the 7/31 expert recording (1.2%, 1.9%, 10:42-11:10); the 11/4 and 11/12 evaluation sessions (12:20-15:10) are noisier (CV 2.3-2.8% whole frame, 2.5-4.3% table).
- **But the brightness comes in two discrete states.** In the 11/14 data the table gray clusters near 115 ("dark state") and near 120-124 ("bright state") (`fig5_ctrunc_stall.png`, right), i.e. the automatic exposure settles in
  different states from one camera start to the next. The bright state is 22% of the retest trials (13:44-15:10), 58% of the truncation trials (16:29-17:21) and 0% of the 11/12 control evaluation (13:46-14:01); in the 90 training frames it occurs once.
  The window-side background is also brighter on 11/14 (172 retest, 176 truncation) than in the training data (165).
- Colour balance (table R/B) was steady inside the sessions (slope 0.006 and -0.018 per hour); between days it differs (7/31 expert: 1.19, later sessions 1.29-1.38), see Q2.

## Q2. Are the training conditions and the test conditions the same?
Training range = 5th-95th percentile of the 90 training frames (`lighting_vs_training.csv`, `fig3_distribution_by_session.png`): whole-frame gray 125.8-132.9, table gray 106.1-116.0, window-side gray 148.7-167.7, table R/B 1.18-1.39.

| session (frames) | whole-frame gray inside | table gray inside | window-side gray inside | table R/B inside |
|---|---|---|---|---|
| expert 50 (7/31) | 98% | 94% | 100% | 92% |
| recovery 20 (10/15) | 80% | 95% | 75% | 100% |
| new expert 20 (11/12) | 75% | 70% | 75% | 75% |
| eval, 6 sets (11/4) | 74% | 69% | 26% | 99% |
| control + right-side evaluation (11/12) | 75% | 86% | 57% | 98% |
| **retest (11/14)** | 62% | 68% | **0%** | 100% |
| **truncation test (11/14)** | **6%** | 36% | **0%** | 100% |

- The 11/14 sessions are brighter than anything in training: truncation test whole-frame median 136.1 (training P95 132.9; 94% of its frames above), table 118.6 (64% above), window side 176.2 (100% above). The retest is in between (132.6; 38% above; window side 100% above).
- Contrast and mean saturation are inside the training range in every session. Colour temperature (R/B) is inside the range, but the training sets differ from each other: the 7/31 expert data are bluer and less saturated (table R/B 1.19, saturation 62) than the 10/15 and 11/12 data (1.29-1.38, 71-72). The training data therefore already mixes colour conditions; the test sessions look like the later ones.
- The recovery data (10/15) are first frames from the middle of a task (arm in view; cube detected in only 55%); region values there are less clean.

## Q3. Can the 11/14 truncation-session drift be explained by the light?
The drift = C 15/16 (retest) -> 8/16 (same day, 16:29-17:22; p = 0.015) and the C_trunc trials that stand still at the start.

**For the drop of model C: no, not by brightness or colour balance as measured here.**
- Inside the truncation session C's success does not depend on the exposure state (bright 5/10, dark 3/6); in the retest also not (bright 4/5, dark 18/19 for C). Success vs table gray: Mann-Whitney p = 0.96 (truncation session C), 0.18 (retest C).
- Offline sensitivity (`policy_sensitivity_summary.csv`, `fig6_policy_sensitivity.png`; 18 frames from 6 successful retest trials, 1-3 s after motion onset): changing the image brightness by +-5% moves the first 30 steps of the action chunk by 0.2-0.3 deg and the predicted aim point by 0.3-0.8 px; +-15% by 0.4-1.2 deg and 1.1-2.6 px; x1.30 by 1.5-2.5 deg and 4.1-6.7 px. The
  brightness difference between the sessions (about +2.6% on the table) is therefore worth well under 0.5 px of aim, while C's aim at the grasp moved by 2.4-3.7 px between the sessions. Colour shifts of +-10% (R/B in opposite directions) move the aim by 1.1-1.9 px on average.
- The camera did not move: phase correlation of the median first frames of the retest and the truncation session gives 0.6 / 0.7 px, below the method's own bias.
- Most sensitive model (all variants, mean joint difference / aim shift): C 0.87 deg / 2.0 px, C_trunc 0.83 / 2.1, R 0.70 / 2.0, baseline 0.60 / 1.7. C is the most sensitive to brightness (x1.30: 2.5 deg), R has the largest worst case for the cool shift (aim 6.2 px at worst, against 2-3 px for the others), the baseline the least; the differences between models are small (18 frames).

**For the C_trunc "stands still at the start" trials: associated with the bright state, but not through what the model sees in the first frame.**
- 9 of the 16 C_trunc trials moved late (motion onset > 4.77 s, the latest of any C or R trial; 6 clearly, 5.5-11.8 s, and 3 mildly, 4.9-5.0 s): 8 of 9 trials in the bright state, 1 of 7 in the dark state (Fisher p = 0.009). Brightness vs onset: Spearman 0.56-0.66 (p = 0.005-0.025);
  after removing trial order the partial correlation is 0.77-0.81 (p <= 0.001). C trials of the same session never stood still, in either state (`ctrunc_stall.csv`, `fig4_behavior_vs_brightness.png`, `fig5_ctrunc_stall.png`).
- Counter-evidence: scaling the first frame by +-5% changes C_trunc's predicted 1 s displacement by 0.02 +- 0.07 units (between-trial spread 1.3), and mapping each first frame to the *other* exposure state with a fitted per-channel mapping changes it by 0.08-0.11 units (displacement itself ~7-8) (`start_sensitivity.csv`, `exposure_state_summary.txt`, `fig7_start_sensitivity.png`). The first
  chunk's reaction to the image difference is far too small to decide between standing still and moving.
- So the association is real in this session (n = 16, one afternoon) but its mechanism is not the brightness in the image. What the bright state may stand for (shadows, exposure time and frame timing, latency, something else that changed with the light) was not measured here. Time order is also confounded with position (P1 trials first); within P2 the correlation stays (rho 0.84, n = 8).
- It does not explain the earlier observation that C_trunc's success is 0/16 in both states: even the dark-state trials fail at the grasp (`edge_not_grasped`).

## Q4. When to collect and test, and should the camera be locked?
**Current camera controls** (`camera_controls.txt`, read with `v4l2-ctl -d /dev/video0 --list-ctrls`, today):
`auto_exposure = 3 (Aperture Priority Mode, automatic)`, `exposure_time_absolute = 157 (inactive while automatic)`, `gain = 0`, `white_balance_automatic = 1`, `white_balance_temperature = 4600 (inactive)`, `backlight_compensation = 1`,
`brightness 0, contrast 42, saturation 64, gamma 100, sharpness 3`, `exposure_dynamic_framerate = 1`, `power_line_frequency = 50 Hz`. Automatic exposure and white balance are on, which fits the two brightness states and the stable global mean.
`exposure_dynamic_framerate = 1` lets the camera lengthen the exposure and lower its frame rate in dim light (not measured here).

**Recommendation**
1. **Lock exposure and white balance** for collection and for every test (yes, needed): the bright state is outside the training range, it was 58% of the late-afternoon trials against 22% in the early afternoon, and the only strong behavioural association in these data (C_trunc) is with it.
2. **Time window**: until the camera is locked, use 13:45-15:15 (the retest window; most stable, 22% bright state) and avoid after 16:30 (58% bright). With a locked camera the time of day should matter little: the first-frame brightness showed no trend with the sun, and the policy reacts weakly to +-5%.
   Whatever is chosen, run the C and R/C_trunc comparisons interleaved in the same window, and keep the table gray of each first frame (`global_stats.py` / `region_stats.py`) as a covariate.
3. **How to lock** (do not apply before you confirm; `v4l2-ctl` settings are lost when the camera is unplugged and may be reset when a program opens it, so read them back after starting the script):
   ```
   v4l2-ctl -d /dev/video0 -c auto_exposure=1                      # manual exposure
   v4l2-ctl -d /dev/video0 -c exposure_time_absolute=<E>           # 1-5000 (units of 100 us); start near 157
   v4l2-ctl -d /dev/video0 -c gain=0
   v4l2-ctl -d /dev/video0 -c white_balance_automatic=0
   v4l2-ctl -d /dev/video0 -c white_balance_temperature=<T>        # 2800-6500; start at 4600
   v4l2-ctl -d /dev/video0 -c backlight_compensation=0
   v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0
   ```
   Choose `<E>` and `<T>` so that a first frame matches the training frames: table gray about 113 (training median 113.6, P5-P95 106-116), window-side gray about 165 (148.7-167.7), table R/B about 1.3 (the 10/15-11/14 data;
   the 7/31 expert set is bluer at 1.19), whole-frame gray about 130. Procedure: set the controls, record 5 first frames at the usual time of day, run `python lighting_analysis/region_stats.py`-style statistics on them, adjust `<E>`, repeat; then check at a second time of day that the table gray stays within +-3.
   Because the policies were trained with automatic exposure, a locked image should be matched to the training statistics rather than to a visually "good" picture.
4. Before the next large collection, run `--policy_only` once with the locked camera and compare the first-close time and speed with the C reference as in the earlier check.

## Files
`episodes_index.csv` (373 first frames with recording time and its source), `frame_stats_all.csv` (global, region and cube-detection statistics), `fig1`-`fig3`, `lighting_vs_training.csv`, `session_trends.csv`, `trial_lighting_behavior.csv`,
`correlations.csv`, `fig4`, `fig5`, `ctrunc_stall.csv`, `trial_brightness_timecourse.csv`, `policy_sensitivity*.csv`, `fig6`, `start_sensitivity*`, `fig7`, `exposure_state_*`, `camera_controls.txt`; scripts `extract_first_frames.py`, `global_stats.py`, `region_stats.py`,
`analyze_time_behavior.py`, `analyze_behavior_lighting.py`, `trial_brightness_timecourse.py`, `policy_sensitivity.py`, `plot_sensitivity.py`, `start_sensitivity.py`, `exposure_state_test.py`.

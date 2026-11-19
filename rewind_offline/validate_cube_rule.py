#!/usr/bin/env python
"""
Offline validation of the new cube-position rule of collect_rewind.py: the cube position after the rewind is the last FULLY visible detection (area >= 0.8 of the nominal
cube area, a single candidate) between the trigger and the end of the alignment; the operator is only asked when the cube moved more than a threshold from its start position,
or when it was never fully visible. Replayed on the 30 attempts of the deep-rewind collection with the segment videos and compared with the operator's keep / discard choices.
Writes cube_rule_validation.csv, cube_rule_validation.txt.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, cv2
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import analyze_cube_positions as A
from lerobot.datasets import LeRobotDataset
OUT = HERE / "rewind_offline"; RUN = HERE / "results/rewind_collect/run_20261119_120625"; ROOT = Path.home() / ".cache/huggingface/lerobot/YKB0123"
FULL_AREA = float(pd.read_csv(HERE / "cube_positions.csv").query("dataset == 'expert' and detected").area.median()) * 0.8
LOG = open(OUT / "cube_rule_validation.txt", "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n")

def full(rgb):
    d = A.detect_cube(np.asarray(rgb))
    return (float(d["cx"]), float(d["cy"])) if (d["detected"] and d["area"] >= FULL_AREA and d["n_candidates"] == 1) else None

m = pd.read_csv(RUN / "rewind_meta.csv"); reasons = pd.read_csv(RUN / "discard_reasons.csv").set_index("episode_id")
seg = LeRobotDataset("YKB0123/rewind_deep_20_rewind_seg", root=ROOT / "rewind_deep_20_rewind_seg"); frm, to = seg.meta.episodes["dataset_from_index"], seg.meta.episodes["dataset_to_index"]
rows = []
for k, r in enumerate(m.itertuples()):
    ed = RUN / f"episodes/ep_{r.episode_id:04d}"
    last, src = None, None
    for name in ("at_trigger", "after_align"):                         # at the trigger and after the alignment: single images saved by the script
        d = full(cv2.cvtColor(cv2.imread(str(ed / f"{name}.png")), cv2.COLOR_BGR2RGB))
        if d is not None and name == "at_trigger":
            last, src = d, "at_trigger"
    seg_last = None
    for i in range(int(frm[k]), int(to[k])):
        im = np.asarray(seg[i]["observation.images.front"].permute(1, 2, 0).clip(0, 1) * 255, dtype=np.uint8); d = full(im)
        if d is not None:
            seg_last = d
    if seg_last is not None:
        last, src = seg_last, "last_visible_in_rewind"
    after = full(cv2.cvtColor(cv2.imread(str(ed / "after_align.png")), cv2.COLOR_BGR2RGB))
    if after is not None:
        last, src = after, "after_align"
    start = np.array([r.cube_start_x, r.cube_start_y])
    shift = float(np.hypot(*(np.array(last) - start))) if last is not None else float("nan")
    rows.append(dict(episode_id=r.episode_id, position=r.position_id, operator_kept=not r.discarded, operator_reason=("" if not r.discarded else reasons.loc[r.episode_id, "category"]),
                     cube_ref_source=src or "never fully visible", shift_from_start_px=shift, old_cube_detected_after_align=not np.isnan(r.cube_after_x)))
V = pd.DataFrame(rows); V.to_csv(OUT / "cube_rule_validation.csv", index=False, float_format="%.2f")
pr(f"nominal full-cube area threshold {FULL_AREA:.0f} px^2; reference position sources: {V.cube_ref_source.value_counts().to_dict()}")
pr("old rule (position from the single frame after the alignment): cube not detected in " + str(int((~V.old_cube_detected_after_align).sum())) + " of 30 attempts -> operator prompt in all of them")
pd.set_option("display.width", 220)
for T in (3, 5, 10, 15):
    ask = V.shift_from_start_px.isna() | (V.shift_from_start_px > T)
    agree_keep = (~ask) & V.operator_kept; agree_disc = ask & ~V.operator_kept
    dis_keep = ask & V.operator_kept; dis_disc = (~ask) & ~V.operator_kept
    pr(f"\nthreshold {T} px: the new rule asks the operator in {int(ask.sum())} of 30 attempts; agrees with the operator (keep/keep + would ask/discarded): {int(agree_keep.sum() + agree_disc.sum())}; "
       f"asks but the operator kept: {int(dis_keep.sum())} {V[dis_keep].episode_id.tolist()}; does not ask but the operator discarded: {int(dis_disc.sum())} {V[dis_disc].episode_id.tolist()}")
pr("\nper attempt (threshold 10 px):")
V["new_rule_asks_at_10px"] = V.shift_from_start_px.isna() | (V.shift_from_start_px > 10)
pr(V[(V.new_rule_asks_at_10px == V.operator_kept) | ~V.operator_kept | V.new_rule_asks_at_10px][["episode_id", "position", "operator_kept", "operator_reason", "cube_ref_source", "shift_from_start_px", "new_rule_asks_at_10px"]].round(1).to_string(index=False))

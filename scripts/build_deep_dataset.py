#!/usr/bin/env python
"""
Merged training dataset for the deep-rewind model: expert 50 (first) + the 20 deep-rewind corrections (rewind_deep_20), WITHOUT the rewind segments.
The corrections carry an extra `intervention` column; the R and C training sets (merged datasets) have no such column, so a copy without it is made first
(the original datasets are not modified), then merged with lerobot's merge_datasets as for the C_trunc set.
Output: ~/.cache/huggingface/lerobot/YKB0123/ykb_rewind_deep20_nointerv (intermediate) and .../ykb_red_cube_expert50_rewind_deep20 (merged, 70 episodes).
"""
import shutil, sys
from pathlib import Path
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets, remove_feature
ROOT = Path.home() / ".cache/huggingface/lerobot/YKB0123"
EXPERT, DEEP = "so101_red_cube_plastic_box_front_50_20260731_104214", "rewind_deep_20"
NOINT, MERGED = "ykb_rewind_deep20_nointerv", "ykb_red_cube_expert50_rewind_deep20"
for n in (NOINT, MERGED):
    if (ROOT / n).exists():
        sys.exit(f"{ROOT / n} already exists; remove it first (it is a derived dataset)")
d_deep = LeRobotDataset("YKB0123/" + DEEP, root=ROOT / DEEP)
d_noint = remove_feature(d_deep, "intervention", output_dir=ROOT / NOINT, repo_id="YKB0123/" + NOINT)
print("copy without intervention:", d_noint.num_episodes, "episodes", d_noint.num_frames, "frames; features", list(d_noint.meta.features))
d_exp = LeRobotDataset("YKB0123/" + EXPERT, root=ROOT / EXPERT)
merged = merge_datasets([d_exp, d_noint], output_repo_id="YKB0123/" + MERGED, output_dir=ROOT / MERGED)
print("merged:", ROOT / MERGED, merged.num_episodes, "episodes", merged.num_frames, "frames; features", list(merged.meta.features))

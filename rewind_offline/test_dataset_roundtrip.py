#!/usr/bin/env python
"""Offline test of the dataset code path of collect_rewind.py (no robot): create the two datasets exactly as Collector._make_datasets does,
write a correction episode (intervention=True) and a rewind segment (intervention=False) with random frames, save, reload and compare the
features with the existing intervention dataset. Writes to a temporary directory only."""
import json, sys, shutil, tempfile, types
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import collect_rewind as CR

HIL = Path.home() / ".cache/huggingface/lerobot/YKB0123/rollout_so101_red_cube_hil_front_20261015_134746"
def main():
    info = json.loads((HIL / "meta/info.json").read_text())
    feats = {k: v for k, v in info["features"].items() if k in ("action", "observation.state", "observation.images.front", "intervention")}
    tmp = Path(tempfile.mkdtemp(prefix="rewind_fake_"))
    cfg = CR.load_config(["--dry_run", "--rewind_mode", "deep", "--out_dir", str(tmp / "out"), "--repo_id", "YKB0123/test_rewind_fake",
                          "--repo_id_rewind_seg", "YKB0123/test_rewind_fake_rewind_seg", "--root", str(tmp / "main"), "--root_rewind_seg", str(tmp / "seg")])
    cfg["dry_run"] = False
    col = CR.Collector(cfg)
    ctx_feats = {k: v for k, v in feats.items() if k != "intervention"}
    col.ctx = types.SimpleNamespace(data=types.SimpleNamespace(dataset_features=ctx_feats))
    col.robot = types.SimpleNamespace(cameras={"front": 1}, name="so_follower")
    col._make_datasets()
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame
    rng = np.random.default_rng(0)
    for ds, interv, n in ((col.ds, True, 45), (col.seg_ds, False, 30)):
        for i in range(n):
            op = {**{k: float(rng.normal()) for k in CR.KEYS6}, "front": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)}
            ad = {k: float(rng.normal()) for k in CR.KEYS6}
            ds.add_frame({**build_dataset_frame(col.features, op, prefix=OBS_STR), **build_dataset_frame(col.features, ad, prefix=ACTION),
                          "task": cfg["task"], "intervention": np.array([interv], dtype=bool)})
        ds.save_episode()
    col.stack.close()
    from lerobot.datasets import LeRobotDataset
    for repo, root, interv in (("YKB0123/test_rewind_fake", tmp / "main", True), ("YKB0123/test_rewind_fake_rewind_seg", tmp / "seg", False)):
        d = LeRobotDataset(repo, root=root)
        s = d[0]
        print(repo, "episodes", d.meta.total_episodes, "frames", d.meta.total_frames, "intervention as expected", bool(s["intervention"].item()) == interv,
              "image", tuple(s["observation.images.front"].shape), "task", s["task"][:30])
        ok = {k: (v["dtype"], list(v["shape"])) for k, v in d.meta.features.items()} == {k: (v["dtype"], list(v["shape"])) for k, v in info["features"].items()}
        print("   features identical to the existing intervention dataset (dtype, shape):", ok)
        names_ok = all(d.meta.features[k].get("names") == info["features"][k].get("names") for k in ("action", "observation.state", "intervention"))
        print("   feature names identical:", names_ok)
    shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()

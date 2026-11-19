# 四模型对照测试（预注册）：深回退 vs R、C、C_trunc

本文件在测试开始前写定，测试期间不修改（偏差记入文末的偏差日志）。
状态：深回退模型已训练完成，第 3 节和第 8 节的赛前检查已完成（2026-11-19 记录）；下一步是周六的 64 次测试。
修订（测试前）：第一版（提交 888e111）把 P1、P2 分成前后两段；已按要求改为 P1、P2 区块逐块交替，重新生成了试验顺序、种子和 key（新哈希见第 2 节）；旧的 key 文件从未提交。预测的写法同时具体化（第 1 节）。

## 1. 目的与假设
- 已有结果：重测（11/14 13:45–15:11，P1+P2）R 7/16、C 15/16；截断对照那一场（16:29–17:22）C 8/16、C_trunc 0/16。
- 新增模型 **deep**：专家 50 条 + 本次采集的 20 条"深回退后人工纠正"（`YKB0123/rewind_deep_20`，只用纠正段，不用回退段）。
  这 20 条的第一帧离起始姿态 138 个关节单位（10/15 的纠正数据是 217），第一次抓取前约 3.4 秒的运动（10/15 是 3.7 秒），见 `rewind_offline/report_deep20_collection.md`。
- **主要预测**：**deep 的成功率高于 R**（P1+P2 合并，16 对 16）。理由：deep 的纠正数据包含一部分接近过程（第一帧离起始姿态 138 个关节单位，R 的纠正数据是 217），
  并且平滑程度接近专家演示（抖动指标 2332，专家数据 2384–2496，10/15 的纠正数据 2982）。
- **次要预测**：(a) deep 与 C 没有明显差别；(b) R 低于 C（重测 R 7/16、C 15/16）。
- **判定**：每个比较用双侧 Fisher 精确检验，p < 0.05 才称有差异。
- **功效的说明**：每个模型只有 16 次，只有观察到的差距达到约 40 个百分点以上才能显著（例如 6/16 对 13/16，p = 0.029；差 44 个百分点是 n = 16 对 16 时 p < 0.05 的最小差距）；
  即使真实差距有 40 个百分点，检验力也只有约 50%–60%。**不显著不等于没有差别**，次要预测 (a)（deep 与 C 无明显差别）尤其不能由不显著来证实。
- **机制的说明（保留）**：C_trunc 测试（0/16，比 R 还差）没有支持"纠正数据缺少接近过程"是 R 较差的原因，所以即使主要预测成立，也不能据此断定机制；
  这次测试只检验 deep 与 R 的成功率有无差别，不检验机制。

## 2. 被比较的模型（四个，盲测）
| 模型 | 训练数据 | checkpoint |
|---|---|---|
| C | 专家 50 + 新增专家 20 | `act_red_cube_expert50_expert20_ft5000` |
| R | 专家 50 + 10/15 介入纠正 20 | `act_red_cube_mixed_hil_ft_20261015_150904` |
| C_trunc | 专家 50 + 路程对齐截断的新增专家 20 | `act_red_cube_expert50_trunc20_ft5000` |
| deep | 专家 50 + 深回退纠正 20 | `act_red_cube_expert50_rewind_deep20_ft5000`（待训练） |
- 四个模型都从 `act_red_cube_50_100000` 微调，5,000 步，批大小 8，lr 5e-6，种子 42，无上采样、无采样权重；deep 的 `train_config.json` 与 R、C、C_trunc 的差别（数据集、job 名、输出目录）在训练后核对并记入第 3 节。
- 测试时模型以 A–D 编码，对应关系封存在 `protocol/four_model_key.json`，其 SHA-256 在 `protocol/four_model_key.sha256`（`f117e9a7124fa5219e6384bd980da8252c31bd602150b7dcb489342a3143d0d2`）；
  key 文件在测试全部结束后才提交，测试期间操作者不查看。key 里 deep 的 checkpoint 路径已固定（训练必须输出到这个路径）。
- 盲测的局限：C_trunc 上次测试中的行为（动作慢、不能抓到）已经为操作者所知，可能识别出该编码；这是对盲测的已知削弱，在分析时说明，不据此改变流程。

## 3. 训练与数据检查（训练后补写，测试前完成）
- 训练数据集：`YKB0123/ykb_red_cube_expert50_rewind_deep20`（70 条，39,711 帧；由 `scripts/build_deep_dataset.py` 生成，去掉纠正数据的 `intervention` 列后与专家 50 合并，原数据集未改动）；
  `scripts/verify_deep_dataset.py`：状态、动作、时间戳、帧序号与来源逐条一致，310 帧解码图像一致，features 与 R、C 的训练集一致（`rewind_offline/deep_dataset_verification.txt`）。
- 位置覆盖（`rewind_offline/coverage_four_models.csv`，方法同右侧测试的覆盖分析）：到测试位置最近的训练样本距离在四个模型之间相同（P1 7.5 px，P2 5.3 px，由专家数据决定）；一个方块宽（17 px）内的样本数
  P1：R 4（其中纠正数据 2），C/C_trunc 2（0），deep 4（2）；P2：R 35（4），C/C_trunc 36（5），deep 37（6）。
- **训练**（已完成，`~/Checkpoints/training_logs/act_red_cube_expert50_rewind_deep20_ft5000.log`，约 4 分钟）：
  `lerobot-train --policy.path=~/Checkpoints/act_red_cube_50_100000 --dataset.repo_id=YKB0123/ykb_red_cube_expert50_rewind_deep20 --policy.optimizer_lr=5e-6 --policy.optimizer_lr_backbone=5e-6 --batch_size=8 --steps=5000 --save_freq=20000 --log_freq=200 --seed=42 --wandb.enable=false --policy.push_to_hub=false`（输出目录和 job 名为 `act_red_cube_expert50_rewind_deep20_ft5000`）。
- **`train_config.json` 对比**（`rewind_offline/compare_training.py` -> `training_comparison.txt`）：deep、C、R、C_trunc 的 117 个键里只有 `dataset.repo_id`、`dataset.root`、`job_name`、`output_dir` 不同，
  其余全部相同（5,000 步、批大小 8、种子 42、lr 与 backbone lr 5e-6、权重衰减 1e-4、梯度裁剪 10、num_workers 4）；四个模型的策略 `config.json` 完全相同，起点都是 `act_red_cube_50_100000`。
- **loss 曲线**（每 200 步记录，`fig_training_loss.png`，`training_loss_curves.csv`）：第 5000 步的训练 loss：deep 0.047、C 0.058、R 0.052、C_trunc 0.046；最后 5 个记录点的平均：0.049 / 0.058 / 0.054 / 0.048。
  deep 与 C_trunc 全程最低，C 最高，R 居中；四条曲线单调下降，没有发散或异常。训练 loss 是在各自的训练集上测的，不是模型好坏的指标，只用来核对训练正常。
- 功能检查：deep checkpoint 可加载，预测有限，在 30 个专家帧上的平均误差 0.82 度（C 0.81 度）。

## 4. 测试位置、次数与顺序
- 位置：**P1 (386, 182)** 和 **P2 (394, 212)**（`results/test_positions.json`；摆放方法、3 px 容差和参考图与重测、C_trunc 测试相同）。
- 每个模型每个位置 8 次，共 **64 次**（V01–V64）；**不中途追加，不提前结束**。
- 顺序（`protocol/four_model_trial_order.csv`，生成脚本 `scripts/make_four_model_order.py`，随机种子 **2660380139**，同 `four_model_order_seed.txt`）：共 **16 个区块**，每块 4 次；**P1 和 P2 的区块依次交替**（每个位置 8 个区块），
  第一个区块是 P1 还是 P2 由种子随机决定，结果是 **P2 开始**：区块位置序列 `P2 P1 P2 P1 P2 P1 P2 P1 P2 P1 P2 P1 P2 P1 P2 P1`；**每个区块是 A–D 四个编码的一个随机排列**，因此每个模型在每个位置恰好 8 次，两个位置在时间上交错。
- 测试在同一天的 10:00–16:00 内完成，尽量一次做完；任何中断只能在区块结束后进行，规则见第 6 节（V32 完成后固定休息 10 分钟）。
  因为位置逐块交替，摆放脚本每个区块内的 4 个 trial 位置相同，位置在区块之间切换，需要重新摆放。
- 摆放：`python scripts/place_cube_overlay.py --order protocol/four_model_trial_order.csv --log results/four_model_placement_log.csv`；运行：`python scripts/run_trial.py V01 --protocol four`（盲测，终端和日志里不出现模型信息）。

## 5. 推理设置、成功标准、失败分类、记录项目
- 推理：temporal ensembling，`n_action_steps=1`，`temporal_ensemble_coeff=0.01`，同步推理，cuda，30 fps，四个模型相同（与重测相同，见 `scripts/run_trial.py` 的 `INFERENCE`）。
- 成功：30 秒内方块落入箱子并已被松开，以方块留在箱子里为准，不看夹爪最终状态（同 `right_side_retest.md`）。
- 失败类型与耗尽 30 秒的归类：沿用 `addendum_02.md`（1 碰到边缘没抓住 / 2 手臂伸不到 / 3 抓起后掉落 / 4 把方块推开）；夹爪合上次数按 `addendum_01.md`。
- 一次运行完整结束（含跑满 30 秒）后必须当场如实记录结果，不得重做；只有与模型无关的原因（硬件故障、崩溃、方块被意外碰动）才重做同一个 trial_id，被中断那次保留并注明。
- 记录项目：录像、关节状态、每一步预测的完整 action chunk、实测方块坐标、夹爪合上次数；**协变量**（由 `run_trial.py` 写入每次试验的 `meta.json`）：首帧桌面灰度、窗户侧灰度、整幅灰度、桌面 R/B、
  相机控制项读回值（auto_exposure、exposure_time_absolute、white_balance_automatic 等，运行中读取）、控制循环周期的均值和最大值、是否在 10:00–16:00 窗口外。
- 环境：自动曝光不锁定，窗帘不拉、灯开，同采集；不控制光照，只记录。

## 6. 中断与偏差
- **试验级中断**：与模型无关的原因（硬件故障、崩溃、方块被意外碰动）才可重做同一 trial_id；被中断那次保留并注明，在偏差日志里写明原因和被中断那次的编号。一次运行完整结束（含跑满 30 秒）后不得重做。
- **测试级中断（休息、设备问题、超过 16:00）只能在区块结束之后进行，不能在区块中间停。** 一个区块 = 4 个 trial（V01–V04、V05–V08……V61–V64）。
  - 区块内遇到设备问题：先按上面的试验级规则重做被中断的 trial，把这个区块做完再停；确实无法在当天做完这个区块时，在偏差日志里记录该区块未完成的 trial，次日从该区块第一个未完成的 trial 继续并标注。
  - 到 16:00 时如果正处在区块中间，把这个区块做完再停（超过 16:00 的 trial 标记为窗口外，但不改变次数）。
- **中断时记录**：完成到第几个区块（最后完成的 trial 编号）和当时的时刻，写入会话记录 `results/four_model_session_notes.txt`，并在偏差日志里引用。
- **第二天继续**：从下一个区块的第一个 trial 继续；**继续前重新运行一次 `scripts/run_trial.py PRACTICE` 并用 `scripts/check_trial_covariates.py` 检查**（协变量写入、控制周期），检查通过后才继续，一次性数据删除。
- **计划中的休息**：第 8 个区块结束后（V32 完成后）休息 10 分钟，让舵机降温，开始和结束的时刻记入会话记录；这是同一天内的休息，不算分日中断，不需要重新 PRACTICE。
- **分析**：主要比较仍用全部 64 次；日期作为一个因素在分析里报告（每个模型每天的次数、成功率，以及各模型的结果在分日之间是否一致，探索性）；若全部在同一天完成则只报告这一点。

## 7. 分析计划
- **主要比较**：deep 对 R，P1+P2 合并（16 对 16），双侧 Fisher 精确检验，显著性水平 0.05；同时报告成功率差和 Newcombe 95% 置信区间，以及各位置的结果（探索性）。
- **次要比较（探索性，不做多重比较校正）**：deep 对 C（16 对 16）；R 对 C（16 对 16）。
- **探索性**：所有两两比较（包括 R 对 C_trunc、deep 对 C_trunc）；成功率按位置；失败类型；夹爪合上次数；抓取前速度曲线、瞄准偏差、向右修正量（方法同 `analyze_retest.py`、`analyze_trunc.py`）；首帧桌面灰度和曝光状态（亮态 ≥ 117.5）与结果的关系；
  本场的 C 与重测的 C（15/16）和截断那一场的 C（8/16）并列，作为场次间漂移的参考（本场四个模型交替，漂移不影响组间比较）。
- 解读：主要比较的 p < 0.05 才称有差异；其余比较只描述，不下结论。样本量（每模型 16 次）只能分辨明显差别。
- 分析脚本在测试后写；解盲前先核对 key 的哈希。

## 8. 赛前检查
1. [已完成] 深回退模型训练完成，第 3 节补全，`git status` 对已跟踪文件无改动。
2. [已完成] `python scripts/run_trial.py PRACTICE` 空跑一次，`python scripts/check_trial_covariates.py`：协变量写入正常（相机控制项运行期间读取、首帧灰度、周期统计），控制周期均值 33.64 ms，在之前 83 个试验的范围（33.46–33.77 ms）内；一次性数据已删除。
3. [已完成] 四个 checkpoint 路径都存在，`sha256sum protocol/four_model_key.json` 等于 `four_model_key.sha256`（核对时不显示编码对应关系）。

## 局限
- 每个模型只训练一个种子；每格 8 次，样本小；只有 P1、P2 两个位置。
- deep 与 R 在数据内容上除"接近过程的长度"外还有其他差别：纠正的人不同时间、不同的触发状态（深回退由空抓触发，回退到约 4 秒前）、数据量（9,810 帧对 9,413 帧）。
- 光照不受控（自动曝光）；盲测被 C_trunc 的行为部分削弱。

## 偏差日志
（测试结束后按时间补记，2026-11-19；测试全部在同一天 13:44–15:40 完成，第 8 个区块后休息一次。）
1. **会话记录未在测试时写**：`results/four_model_session_notes.txt` 没有在测试过程中创建，现根据结果文件和 meta.json 的时间戳事后重建，文件内已注明。休息开始/结束时刻没有记录，
   V32 记录于 14:37:22，V33 摆放于 14:51:59，间隔约 14.6 分钟（≥ 10 分钟的要求满足，但这是推算值）。
2. **两次试验级中断（与模型无关，已按第 6 节重做）**：V01 attempt 1（13:44:27）和 V33 attempt 1（14:52:24），备注都是 "robot did not move"，`return_code = -6`，策略 0 步；
   `run.log` 都是电机总线通信错误（`ConnectionError: Failed to write ... There is no status packet!`，V01 是 `Torque_Enable` id 5，V33 是 `Lock` id 4），发生在启动阶段。
   这两次分别是测试开始后第一次运行和 10 分钟休息后第一次运行。两次都重做了同一 trial_id（attempt 2），被中断那次保留在 `results/four_model_outcomes.csv` 和 `results/four_model_raw/`。
3. **V01、V02 各有两条摆放记录**：V02 的第一条摆放（13:45:05）出现在 V01 重做摆放（13:45:08）之前，原因没有查（可能是 V01 中断后摆放脚本的编号选择，未核实）；分析中取每个 trial 开始前最近的一条摆放记录。V33 同样有两条（对应重做）。所有试验最终摆放误差 ≤ 2.91 px（容差 3 px）。
4. **备注拼写**：V04 备注 "time our"（超时）为笔误，不影响分类。
结果见 `four_model_analysis/report_four_model.md`。

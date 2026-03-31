# VLM 数据采集指南（以抓取为当前实验）

**长期目标**：采集与格式规范服务于 **一个大模型控制机械臂执行多种原子动作**（抓取、放置、倾倒、开关、按压等）。**本文操作示例以抓取（Pick）为主**，因其流水线已最先打通；其它动作将复用 **同一套 universal 配置思想**（多视角、参数 JSON、纠错标签），见总览中的路线图。

> **端到端总览**：[DATA_AND_TRAINING_MASTER_PLAN.md](DATA_AND_TRAINING_MASTER_PLAN.md)

## 〇、通用方案（所有动作最终进同一模型）

**推荐范式**：采集 **`save_frames: -1`**（`universal_vlm_collect` 默认）得到 **全长 HDF5**，转换 **`--mode full`**，用 **`--temporal-stride K`**（未指定 stride 且未指定 `max-timesteps` 时脚本 **默认 K=5**）得到 **固定索引间隔、张数随 episode 变** 的多图样本；可选 **`cache_stride`** 在采集端省盘。详见 **[DATA_AND_TRAINING_MASTER_PLAN.md §2.2](DATA_AND_TRAINING_MASTER_PLAN.md)**。

| 场景 | 采集要点 | 转换要点 |
|------|----------|----------|
| **主线（变长多图）** | `save_frames: -1`，可选 `cache_stride` | `--mode full` + `--temporal-stride K`（或依赖默认 K=5） |
| **固定少量时刻 HDF5** | 如 `save_frames: 5`（`level1_pick_noise_keyframes`） | `--mode full --temporal-stride 1`（用尽 HDF5 内 T 个时刻×3 视角） |
| **极省空间** | `save_frames: 1` 或 `--mode single` 对应采集 | `convert_to_vlm_format.py --mode single` |

```powershell
# Pick 任务 - 通用采集（单物体）
python main.py --config-name level1_pick_noise_universal --headless

# Pick 任务 - 多物体采集（锥形瓶 02/03/04 轮换，推荐）
python main.py --config-name level1_pick_noise_universal_multi_obj --headless

# Pick 任务 - 全仪器采集（锥形瓶/烧杯/量筒/容量瓶，lab_001 中 volume_flask 不存在会自动跳过）
python main.py --config-name level1_pick_noise_universal_all_obj --headless
```

### 多种类设备一次性采集

| 配置 | 物体 | 样本数 | 命令 |
|------|------|--------|------|
| `level1_pick_noise_universal_multi_obj` | 锥形瓶 02/03/04（3 种） | 150 ep | `python main.py --config-name level1_pick_noise_universal_multi_obj --headless --no-video` |
| `level1_pick_noise_universal_all_obj` | 锥形瓶/烧杯/量筒/容量瓶（4 类） | **每类 100 次成功**后切换，全部类完成后结束（`max_episodes` 仅兜底） | `python main.py --config-name level1_pick_noise_universal_all_obj --headless --no-video` |
| `level1_pick_noise_universal_all_obj_fail_200success` | 同上 + 高失败率噪声 | 同上（继承 all_obj 的切换/结束逻辑） | `.\scripts\run_collect_headless.ps1 level1_pick_noise_universal_all_obj_fail_200success --no-video --fast-sim` |

**运行前检查**：若使用 `level1_pick_noise_universal_all_obj` 等多物体配置，可先验证场景中 prim 是否存在：
```powershell
python scripts/list_scene_prims.py
# 或指定场景：python scripts/list_scene_prims.py --usd-path assets/chemistry_lab/lab_001/lab_001.usd
```
lab_001 中 `volume_flask` 不存在，会自动跳过并打印 `[BaseTask] 跳过不存在的 prim`；若所有 prim 均不存在会直接退出。

**扩展其他任务**：新建 `level2_pour_noise_universal.yaml` 等，继承通用配置：

```yaml
defaults:
  - level2_pour          # 或 level1_stir、level1_shake 等
  - universal_vlm_collect
  - _self_

name: Level2_pour_noise_universal
# 覆盖 task_type、controller_type 等
noise:
  enabled: true
```

---

## 一、采集流程

### 1. 混合采集（成功 + 失败，推荐）

同时采集成功与失败样本，用于单模型学习「预测」与「纠错」：

```powershell
cd LabUtopia
conda activate env_isaacsim

# 继承 universal_vlm_collect：save_frames=-1（全长 HDF5）→ 转换 --mode full
python main.py --config-name level1_pick_noise_universal --headless

# 多物体采集（锥形瓶 02/03/04 轮换）
python main.py --config-name level1_pick_noise_universal_multi_obj --headless

# 快速采集（推荐，见 COLLECTION_ACCELERATION.md）
python main.py --config-name level1_pick_noise_universal_fast --headless --no-video

# 其他选项
python main.py --config-name level1_pick_noise --headless          # 全部帧
python main.py --config-name level1_pick_noise_best --headless     # HDF5 T=2；转换 full --temporal-stride 1 → 6 张
python main.py --config-name level1_pick_noise_keyframes --headless # HDF5 T=5；转换 full --temporal-stride 1 → 15 张
python main.py --config-name level1_pick_noise_vlm --headless      # 1 帧（省空间）
```

**save_frames 说明**：`-1`=全部, `1`=仅首帧, `2`=首末帧, `3~N`=首+均匀中间+末

**采集加速**：详见 [COLLECTION_ACCELERATION.md](COLLECTION_ACCELERATION.md)，推荐 `--headless --no-video` 及快速配置。

### 2. 仅成功样本（无噪声）

用于验证默认参数、或补充「纯预测」数据：

```powershell
python main.py --config-name level1_pick_1ep --headless   # 1 ep 测试
python main.py --config-name level1_pick --headless       # 100 ep
```

### 3. 提高失败率（可选）

成功概率过高时使用，目标失败率约 50%：

```powershell
# 单物体 + universal（全长 HDF5）
python main.py --config-name level1_pick_noise_universal_fail --headless --no-video

# 全仪器 + 提高失败率
python main.py --config-name level1_pick_noise_universal_all_obj_fail --headless --no-video

# 多物体 + 提高失败率
python main.py --config-name level1_pick_noise_universal_multi_obj_fail --headless --no-video

# 旧配置（全部帧）
python main.py --config-name level1_pick_noise_fail --headless
```
- **level1_pick_fail**：刁钻物体位置、短步数（无噪声）

**噪声配置**（在 level1_pick 或 level1_pick_noise 中）：
- `noise_scale`: 1.0=默认，2.0=噪声范围翻倍
- `failure_bias_ratio`: 0~1，该比例 episode 使用 noise_scale
- `noise_distribution`: **`edge_bias` U 形分布**（所有资产采集统一使用，更多采样在范围边缘，提高失败率）| `uniform` 均匀分布

---

## 二、输出结构

```
outputs/collect/2026.03.16/14.03.04_Level1_pick_noise/
├── dataset/
│   ├── episode_0000.h5
│   ├── episode_0001.h5
│   └── ...
├── meta/
│   └── episode.jsonl
└── video/           # 若启用
```

每个 `episode_*.h5` 包含：

| 内容 | 说明 | 占空间 |
|------|------|--------|
| camera_1_rgb, camera_2_rgb, camera_3_rgb | 3 相机图像 [T, 256, 256, 3] | **主要** |
| agent_pose | 关节角度 [T, 9] | 小 |
| actions | 动作 [T, 9] | 小 |
| language_instruction | 任务指令 | 小 |
| task_properties | params_used, correction_gt, is_success, **object_type**、回合结束时的 **object_position**（仿真 GT，便于与 `picking_position_delta` 核对）等 | 小 |

**空间占用**：`save_frames:-1` 时 T 随 episode 变；`universal_vlm_collect` 默认 **`cache_stride:20`**（每 20 次 cache 写 1 帧）已显著省盘；若改为 `cache_stride:1` 则单集可达 **数十～数百帧**，磁盘更大。还可选 **`level1_pick_noise_universal_fast`** / **`save_frames:1`** / 转换 **`--mode single`** 进一步省空间。

**执行过程 /「视频」语义与 `cache_stride`**：多帧训练依赖 HDF5 时间维 T。`cache_stride>1` 时只对**每第 N 次** `cache_step` 落盘（与 `agent_pose` 对齐），时序更稀；若希望模型更多从**密集执行过程**学幅度与失败形态，可 **`cache_stride: 1`～`5`**（磁盘允许时）。转换 `full` 时用 `--temporal-stride 1` 可吃满已写入的索引。`scripts/convert_to_vlm_format.py` 会将 HDF5 中的 **CHW `(3,H,W)`** 单帧转为 **HWC** 再存 PNG，与采集格式一致。

---

## 三、转为 VLM 训练格式

```powershell
# 全长 HDF5：full + stride（不写 --temporal-stride / --max-timesteps 时默认 stride=5）
python scripts/convert_to_vlm_format.py outputs/collect/.../dataset --mode full --temporal-stride 5 -o vlm_train.jsonl

# HDF5 内仅 T 个稀疏时刻（如 keyframes）：stride 1 用尽全部索引
python scripts/convert_to_vlm_format.py outputs/collect/.../dataset --mode full --temporal-stride 1 -o vlm_train.jsonl

# 单视角单帧，省空间
python scripts/convert_to_vlm_format.py outputs/collect/.../dataset --mode single
```

**full 模式**：`images` 为列表，instruction 描述 **「共 n_t 个时刻×3 视角」**（n_t 随 episode 变化）；JSONL 可含 `num_timesteps`、`num_images` 等元数据。

**single 模式** 输出：`image` 单路径。

### 3.1 转为 LLaMA-Factory 格式（Qwen2.5-VL 微调）

若使用 LLaMA-Factory 进行 Qwen2.5-VL 微调，需将上述 JSONL 转为 Alpaca/Sharegpt 格式：

```powershell
# 输出到 LLaMA-Factory 的 data 目录
python scripts/convert_to_llamafactory.py vlm_train.jsonl -o /path/to/LLaMA-Factory/data/labutopia_pick
```

输出目录将包含 `train.jsonl` 和 `images/`，可直接供 LLaMA-Factory 使用。详见 [VLM_TRAINING_SERVER_GUIDE.md](VLM_TRAINING_SERVER_GUIDE.md) 与 [LLM_TRAINING_REFERENCE.md](LLM_TRAINING_REFERENCE.md)。

---

## 四、抓住但高度不足的纠正（非重置）

当 Pick 已抓住物体但抬升高度不足（<10cm）时，默认会尝试**非重置纠正**：继续上抬 10cm，最多 60 步。若纠正后达到成功线则记为成功，并写入 `was_lift_corrected: true`、`lift_correction_gt: {after_offset_z: 0.1}`，供 VLM 学习「抓住但不够高 → 继续抬」的纠错模式。

关闭纠正：在配置中设置 `pick.lift_correction_enabled: false`。

**成功/失败判断**：为避免物理抖动导致误判，采用「连续 N 步满足才判成功」：
- `pick.required_success_steps: 15`：主 pick 需连续 15 步物体高于 10cm（原 60 步易误判失败）
- `pick.lift_required_success_steps: 10`：lift correction 阶段需连续 10 步才判成功，避免单帧误判

**picking_position 噪声与纯视觉 correction**：为支持纯视觉部署，采集时对 `picking_position` 也加噪，使 correction 能修正位置误差，将错误参数朝正确参数空间移动。`params_used.picking_position` 为实际使用的抓取位置（GT + 噪声）；`correction_gt.picking_position_delta` 为 **指向当前帧仿真器物体位姿** 的修正量（`object_position[:3] - params_used.picking_position`）。

**与其它参数的区别**：`pre_offset_*` / `euler_deg` 等在失败时可由**成功池**的均值/最近邻估计「正确空间」；但 **`picking_position` 在世界系下对当前场景有唯一物理真值**（即仿真里读到的物体位置），不应与历史成功样本的位置取平均。实现上在写入失败样本前会用 **`_merge_correction_gt_with_sim_picking_gt`** 覆盖池算出的 `picking_position_delta`，保证监督目标为仿真 GT。配置 `noise.picking_position_noise: [-0.02, 0.02]`（x/y/z 各轴 ±2cm），设为 `null` 可禁用。

> **排查「pos 一直不变」**：若仅继承 `level1_pick` 且 `noise.enabled: false`，则根本不会采样 `_episode_noise`，`picking_position` 恒等于 `object_position`（GT），无位置纠错监督。请使用 **`defaults` 含 `universal_vlm_collect`**（或自行设 `noise.enabled: true` + `picking_position_noise`）。同一 **episode 内** `picking_position` 本应为常数（每局 reset 时采样一次噪声）；**跨 episode** 在启用位置噪声时应变化。启动时若 `noise.enabled` 为真但未启用 `picking_position_noise`，控制器会打印 `[Pick][Collect] 提示：...`。

### 4.1 方向与幅值（保守单步标签）

完整几何修正量仍保存在 `correction_gt`（如 `picking_position_delta`、`euler_deg`），同时写入：

- **单位方向 + 模长**：`picking_position_delta_unit` / `picking_position_delta_magnitude`（及欧拉同理），便于分析或单独建模幅值。
- **保守一步 `*_step`**：与 full 同方向，但幅值受 YAML 上限约束（如 `pick.correction_suggested_max_picking_norm`），避免单步过冲。

`scripts/convert_to_vlm_format.py` 构建 **response** 时**优先用 `*_step`** 拼出「修正后参数」，使 VLM 学习**小步、可执行**的纠错；多步 `correction_steps` 内另用 `correction_step_max_picking_norm` 限制每步位置增量。可调小上限以提高稳健性，或设 `null`/≤0 关闭该维度的 step（回退为完整 delta）。

### 4.2 多次 correction + 朝向正确空间

采集时支持「多次 correction 且朝向正确参数空间」：

1. **成功池**：每类物体维护成功样本的 `params_used` 池，用于估计正确参数空间。
2. **correction_gt 朝向正确空间**：失败时，`correction_gt = p_ref - params_used`，其中 `p_ref` 来自成功池（均值或最近点），而非固定 baseline。
3. **多步 correction_steps**：从单次失败生成多步训练样本，每步朝正确空间移动 `alpha` 比例，模拟迭代纠错。

配置（`universal_vlm_collect` 中 `pick` 段）：

| 参数 | 默认 | 说明 |
|------|------|------|
| success_pool_max | 500 | 每类物体最多保留的成功样本数 |
| correction_steps_count | 3 | 失败时生成多步 correction 样本（1=仅单步） |
| correction_alpha | 0.5 | 每步朝正确空间移动的比例 |
| p_ref_method | mean | `mean`=成功样本均值 \| `nearest`=最近成功样本 |
| correction_enrich_unit_magnitude | true | 是否写入 unit/magnitude 与 `*_step` |
| correction_suggested_max_picking_norm | 0.02 | 单步位置修正上限（米），`null` 则不用 step |
| correction_suggested_max_euler_l2_deg | 8.0 | 单步欧拉 L2 上限（度） |
| correction_suggested_max_offset_abs | 0.06 | 标量偏移单步上限 |
| correction_step_max_picking_norm | 0.02 | 多步 `correction_steps` 内每步位置增量上限 |

转换时，`convert_to_vlm_format.py` 会为每个 correction 步生成一条训练样本，同一 episode 的失败可产出多条「当前参数 → 修正量」样本，供模型学习迭代纠错。

---

## 五、单模型推理流程

1. **首次调用**：图像 +「预测抓取参数」→ 模型输出初始参数
2. **执行抓取**：用初始参数执行
3. **若失败**：图像 +「当前参数: {...}，请修正」→ 模型输出修正参数
4. **迭代**：重复 2–3 直到成功或达到最大迭代次数

---

## 六、数据采集完整性检查

以下字段在各类 episode 中均应正确写入，供 `convert_to_vlm_format.py` 使用：

| 场景 | params_used | correction_gt | correction_steps | is_success | object_type |
|------|-------------|---------------|------------------|------------|-------------|
| 主 pick 成功 | ✓ | ✓（noise 时） | - | True | ✓ |
| 主 pick 失败 | ✓ | ✓（成功池或 baseline） | ✓（多步时） | False | ✓ |
| Lift 纠正成功 | ✓ | ✓ + lift_correction_gt | - | True | ✓ |
| Lift 纠正失败 | ✓ | ✓（成功池或 baseline） | ✓（多步时） | False | ✓ |
| 提前返回（object_position=None） | - | - | - | 不写入 | - |

**图像**：pick 执行 + lift correction（若有）期间均调用 `cache_step`，帧数由 `save_frames` 控制。

---

## 七、训练建议

- 成功样本：学习「图像 → 参数」的映射
- 失败样本：学习「图像 + 错误参数 → 修正参数」
- 建议成功与失败样本比例约 1:1 或 2:1，避免失败样本过少
- **object_type**：每条样本记录物体**类别**（如 conical_bottle、beaker），同类资产（conical_bottle02/03/04）自动归为同一类，避免编号拆成多类数据
- **is_success**：response 中包含 `is_success` 字段，供模型学习通过视觉判断原子动作是否成功

---

## 八、数据采集规模建议

### 8.1 每个原子动作应采集多少数据？

| 阶段 | 建议样本量 | 说明 |
|------|------------|------|
| 单动作验证 | 500–1000 条/动作 | 验证 VLM 能否学会该动作的参数预测 |
| 多动作统一模型 | 5000–10000 条 总计 | 10 个原子动作 × 500–1000 条 |
| 生产级 | 10000+ 条 | 覆盖更多场景、材质、失败情况 |

**换算为 episode 数**：每条样本 ≈ 1 episode（成功或失败）。若成功率约 50%，则 1000 条 ≈ 1000 ep；50 ep 单物体 ≈ 50 条，150 ep 多物体 ≈ 150 条。

**建议**：先用 `level1_pick_noise_universal_1ep` 跑通流程，再用 `level1_pick_noise_universal_multi_obj`（150 ep）或 `level1_pick_noise_universal`（50 ep）做小规模采集，验证后再放大到 500–1000 ep/动作。

### 8.2 设备多样性（不同仪器操作）

**当前**：`level1_pick` 仅抓取 `conical_bottle02`（锥形瓶）。

**推荐**：使用多物体配置，每个 episode 随机选不同仪器，提升泛化：

| 配置 | 物体 | 说明 |
|------|------|------|
| level1_pick_noise_universal | 仅 conical_bottle02 | 单物体，快速验证 |
| **level1_pick_noise_universal_multi_obj** | conical_bottle02/03/04 | 3 种锥形瓶，形状/尺寸不同，**推荐** |

**扩展更多仪器**：若场景中有 `beaker2`、`volume_flask`、`graduated_cylinder` 等，可在 `task.obj_paths` 中追加，并确保 `task_utils.py` 中有对应抓取参数（`get_pickz_offset` 等）。不同仪器抓取高度、姿态差异大，数据多样性对 VLM 泛化很重要。

**其他原子动作**：pour、stir、shake、place 等也建议覆盖多种源/目标物体，在各自 controller 中实现 noise 注入和 `params_used` 记录后，继承 `universal_vlm_collect` 即可。

---

## 九、部分 episode 未写入说明

**现象**：`Episode Stats` 显示 `Success = 28/50 written`，但 `_episode_num` 可能为 62，即部分 episode 未写入 HDF5。

**原因**：以下情况会跳过写入（不调用 `write_cached_data`），仅重置并进入下一 episode：

| 原因 | 说明 |
|------|------|
| `object_position=None` | 物体位置无法获取（prim 不可见、被删除或 `get_geometry_center` 失败），打印 `[Pick] 提前返回：object_position=None` |
| 控制器/仿真异常 | `task_controller.step()` 或 `apply_action` 抛异常，main 捕获后跳过写入 |

**统计**：程序会打印 `X skipped (object_position=None or exception)`，便于排查。若跳过较多，可检查场景中 prim 是否存在、物体是否被正确放置。

---

## 十、视频黑屏排查

若保存的视频为全黑，按顺序尝试：

1. **在运行前设置 `ENABLE_CAMERAS=1`**（headless 下必须，建议先手动设置）
   ```powershell
   $env:ENABLE_CAMERAS="1"
   python main.py --config-name level1_pick_noise_universal_all_obj --headless --no-video
   ```

2. **去掉 `--headless`**：带 GUI 运行，确认相机是否正常
   ```powershell
   python main.py --config-name level1_pick_noise_universal_1ep
   ```

3. **使用 `--no-video`**：若仅需 HDF5 数据，可跳过视频；HDF5 中的图像与视频同源，若视频黑则 HDF5 也可能黑，需先解决相机问题。

4. **检查 Isaac Sim 版本与 GPU 驱动**：程序已将首 30 帧作为预热，若仍黑屏可尝试更新驱动或 Isaac Sim。

---

## 十一、视频闪烁排查

若保存的视频出现闪烁，可能原因与对策：

1. **帧率不匹配**：仿真步率与视频 60fps 不一致。已改为 30fps 录制，可减轻闪烁。
2. **使用 `--no-video`**：若仅需 HDF5 数据，可跳过视频；HDF5 中的图像不受录制帧率影响。
3. **headless 下渲染**：headless 模式相机更新可能不稳定，可尝试带 GUI 运行（去掉 `--headless`）对比。

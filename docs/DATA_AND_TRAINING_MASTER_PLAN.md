# LabUtopia 数据采集与训练总览（单一入口）

## 终极目标（文档与工程对齐）

**最终目的**：训练 **一个大模型（VLM）** 根据视觉与语言，**输出可执行参数**，在真实或仿真中 **驱动机械臂连续完成多种实验室原子动作**（抓取、放置、倾倒、开关门/抽屉、按压、摇匀、搅拌等），并在失败时具备 **纠错 / 再执行** 能力。

**当前阶段**：**抓取（Pick）** 作为 **第一条打通的实验线**——用来验证「多视角视频式输入 → 参数 JSON → 仿真执行 → 采集纠错标签 → 转换 → 微调」的全链路。其它动作的采集格式、转换脚本与推理接口将 **按同一套原则扩展**，而不是为 Pick 单独建一套孤立方案。

本文档为 **唯一主线**：从仿真采集 → 格式转换 → VLM 训练。细节见文内链接的专项文档。

---

## 一、端到端流程

```
Isaac Sim 采集 (HDF5)
    → convert_to_vlm_format.py (JSONL + 多图)
    → convert_to_llamafactory.py (可选)
    → LLaMA-Factory + Qwen-VL 等微调
```

| 阶段 | 产出 | 关键脚本/配置 |
|------|------|----------------|
| **1. 采集** | `outputs/collect/.../dataset/episode_*.h5` | `main.py` + Hydra YAML |
| **2. 转 VLM** | `vlm_train.jsonl` + `vlm_images/` | `scripts/convert_to_vlm_format.py` |
| **3. 转训练框架** | `train.jsonl` + `images/` | `scripts/convert_to_llamafactory.py` |
| **4. 训练** | LoRA/全参 checkpoint | LLaMA-Factory，见 `VLM_TRAINING_SERVER_GUIDE.md` |

---

## 二、阶段 1：采集（**当前实验：Pick**；其它动作后续按同范式扩展）

### 2.1 推荐配置（全仪器 + 高失败率 + 按成功数切换）

| 项 | 说明 |
|----|------|
| 配置名 | `level1_pick_noise_universal_all_obj_fail_200success` |
| 切换 | 每类仪器 **100 次成功** 后换下一类；**400 次总成功**（4 类×100）后结束 |
| 噪声 | 约 70% episode：`noise_scale=1.75`（继承 `all_obj_fail`） |
| `max_episodes` | `99999`（实际上由成功数截断） |

**命令（PowerShell，注意多参数由脚本拆行）：**

```powershell
cd LabUtopia
$env:ENABLE_CAMERAS = "1"   # 脚本内已设；headless 黑屏时再确认
.\scripts\run_collect_headless.ps1 level1_pick_noise_universal_all_obj_fail_200success --no-video --fast-sim
```

或直接：

```powershell
python main.py --config-name level1_pick_noise_universal_all_obj_fail_200success --headless --no-video --fast-sim
```

- **加速**：`COLLECTION_ACCELERATION.md`
- **资产与 prim**：`ASSET_INVENTORY.md`；缺失 prim 会跳过（如 `volume_flask` 若场景无则有效类数减少）

### 2.2 帧、HDF5 与「固定采样率 → 变长视频」输入（训练对齐）

**目标**：同一套 **固定时间索引间隔**（相对仿真主循环的步数）采样，**不同动作 / 不同 episode 时长 → 时间维长度 T 可变**，总输入图数 ≈ **T×3**（三视角），供 VLM **变长多图** 训练。

| 环节 | 做法 |
|------|------|
| **采集** | `save_frames: -1`：允许按步堆叠时间维。可选 `cache_stride: K`（仅此时生效）：**每 K 次 `cache_step` 写入 1 帧**（相机与 `agent_pose` 同步），等价在磁盘上已做「固定步频」下采样、**省空间**。 |
| **转换（主路径）** | `python scripts/convert_to_vlm_format.py <dataset> --mode full --temporal-stride K`：在 HDF5 时间维上 **每隔 K 个索引取 1 帧**，再按「每时刻 × 3 视角」展开；**T = ceil(n/K)**，n 随 episode 变 → **变长**。若 **既不传 `--temporal-stride` 也不传 `--max-timesteps`**，脚本 **默认 K=5**。可加 `--max-timesteps T_max` 在 stride 之后 **均匀封顶**，防超长样本 OOM。 |
| **勿重复 stride** | 若采集已设 `cache_stride=K` 且 HDF5 已稀，转换时一般用 `--temporal-stride 1`；若采集密存再转换 `--temporal-stride K`，避免无意 **K×K** 过稀。 |
| **物理时间** | 「固定步频」相对 **仿真主循环步**；若使用 `--fast-sim` 等改变 `physics_step_size`，**同一 K 对应的真实秒数会变**，但 **跨 episode 仍可比较**（同一配置下）。 |

| `collector.save_frames` | `cache_stride` | HDF5 行为 | 转换典型命令 |
|-------------------------|----------------|-----------|----------------|
| `-1`（**universal 默认**） | `20` | 每 20 次 cache 写 1 帧（省盘） | `--mode full --temporal-stride 1`（避免采集与转换再叠乘过稀） |
| `-1` | `1` | 每 cache 步 1 帧（最密） | `convert_to_vlm_format.py --mode full`（未指定 stride/max 时 **默认 stride=5**） |
| `-1` | `K`（其它） | 每 K 次 cache 写 1 帧 | `--mode full --temporal-stride 1`（一般原则同上） |
| `2`～`N`（可选） | - | 采集端均匀取 T 个时刻 | `--mode full --temporal-stride 1`（用尽 HDF5 内 T 个索引） |

在继承 `universal_vlm_collect` 的配置中覆盖 `collector.save_frames` / `collector.cache_stride` 即可（如快速配置 `save_frames: 2`）。

### 2.3 多视角与「过程」语义

- **三相机分开输入**多张图（不要默认拼一张大图），顺序为 **每个时刻：cam1 → cam2 → cam3**（与转换脚本一致）。
- 采集语义（视觉–执行–纠错–再执行）：见 **`LLM_TRAINING_REFERENCE.md` 第一节 §1.1**。

### 2.4 采集检查（精简清单）

| 检查项 | 要求 |
|--------|------|
| 纠错数据 | `noise.enabled: true`，否则大量失败样本无 `correction_gt`，转换会跳过 |
| headless 相机 | `--headless` 时 `ENABLE_CAMERAS=1`（`main.py`/脚本已处理） |
| 物体位姿 | `object_position=None` 的 episode 可能不写入；控制台有 skipped 统计 |

---

## 三、阶段 2：转换为 VLM 格式

**依赖**：`pip install h5py numpy Pillow`

**转换**：统一使用 **`--mode full`**。HDF5 时间维长度 **n** 由采集 `save_frames` / `cache_stride` 决定；`--temporal-stride K` 在 **HDF5 索引**上每隔 K 取一帧，总图数约 **ceil(n/K)×3**。若 **既不指定 `--temporal-stride` 也不指定 `--max-timesteps`**，脚本 **默认 `temporal-stride=5`**，避免全长 OOM；需要密采样时用 `--temporal-stride 1`。

```bash
# 全长 HDF5（save_frames=-1）：默认 stride=5；显式指定 K
python scripts/convert_to_vlm_format.py /path/to/dataset --mode full --temporal-stride 5 -o vlm_train.jsonl

# 封顶：先 stride 再最多 T 个时刻
python scripts/convert_to_vlm_format.py /path/to/dataset --mode full --temporal-stride 5 --max-timesteps 48 -o vlm_train.jsonl

# 仅用均匀下采样到 T 个时刻（不先 stride）：不传 --temporal-stride，传 --max-timesteps
python scripts/convert_to_vlm_format.py /path/to/dataset --mode full --max-timesteps 32 -o vlm_train.jsonl

# 单帧省空间
python scripts/convert_to_vlm_format.py /path/to/dataset --mode single -o vlm_train.jsonl
```

- **变长多图与 Qwen-VL**：可训练；`convert_to_llamafactory.py` 按每条样本的 `images` 列表原样传递；需 **单条总 token < context**（见 `LLM_TRAINING_REFERENCE.md` 变长训练说明），用 `--max-timesteps` 或 stride 控制。
- **指令**：`full` + `temporal_stride≥2` 时 instruction 会注明「按固定间隔采样、动作越长时刻越多」；JSONL 含 `num_timesteps` / `num_images` 便于排查（训练框架可忽略）。

---

## 四、阶段 3：训练（训练机、无仿真）

1. 拷贝 `dataset/` 或已生成的 `vlm_train.jsonl` + `vlm_images/`。
2. 按 **`VLM_TRAINING_SERVER_GUIDE.md`** 安装依赖、`convert_to_llamafactory.py`、LLaMA-Factory 数据集注册与 YAML。
3. **细粒度字段、多动作扩展、Alpaca 列说明**：**`LLM_TRAINING_REFERENCE.md`**。

---

## 五、路线图（通向「一模型多动作」）

### 5.1 单步原子动作（在 Pick 实验线验证后并列扩展）

| 动作 | 场景 | VLM 数据状态 |
|------|------|----------------|
| pick | lab_001 | ✅ 已通（**当前主实验**） |
| place / pour / open / close / press / shake / stir | lab_001 / lab_003 | ❌ 需在 controller 写 `task_properties`，并扩展 `convert_to_vlm_format.py`（**与 Pick 同一目标：服务统一大模型**） |

每动作应对「可操作的多类仪器」分别采集；**每类建议 150–200 ep** 量级（可随算力调整），最终 **混合为同一训练集**，由 `action_type` 区分。

### 5.2 多步复合任务（DeviceOperate、CleanBeaker 等）

- **现状**：一条 episode 为整段轨迹，**无逐步 `params_used` / `correction_gt`**，与当前 VLM 转换脚本 **不对齐**。
- **若要训逐步策略**：需在子步骤边界写 `task_properties`，或拆成多个单步 episode。**不再单独维护缺口分析文档；需要时以本节为准。**

### 5.3 正确参数空间 / 无噪声基线（可选研究）

- 可用 `level1_pick_success_only` 等采成功分布，用于分析噪声范围；**非主线必做**。

---

## 六、文档地图（删减后仍保留）

| 文档 | 用途 |
|------|------|
| **本文** | 总流程 + 命令 + 路线图 |
| `LLM_TRAINING_REFERENCE.md` | 大模型/训练侧：格式、多动作目标、§1.1 流程语义 |
| `VLM_TRAINING_SERVER_GUIDE.md` | 训练机部署步骤 |
| `VLM_DATA_COLLECTION.md` | 采集命令与说明补充 |
| `COLLECTION_ACCELERATION.md` | `--fast-sim`、GPU、快速配置 |
| `ASSET_INVENTORY.md` | 场景 prim 与资产表 |
| `DATA_COLLECTION_CODE_REFERENCE.md` | 代码入口（collector、controller） |

---

## 七、版本提示

- 采集与转换逻辑以仓库内 **`main.py`、`controllers/pick_controller.py`、`scripts/convert_to_vlm_format.py`** 为准；文档仅作索引与约定说明。

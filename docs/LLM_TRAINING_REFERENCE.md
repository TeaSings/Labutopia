# LabUtopia 全原子动作 VLM 训练参考文档（大模型用）

**工程目标**：让 **一个大模型** 在视觉与语言条件下，**控制机械臂执行多种实验室原子动作**（非只做抓取）。抓取（Pick）是当前 **已打通的实验动作**；本文档中的数据格式、多动作 `action_type` 约定与扩展方式，均服务于 **同一统一模型**。

本文档供训练服务器上的大模型 / 工程师阅读，用于理解**所有原子动作**的数据格式与训练流程。

> **流程总览**：`docs/DATA_AND_TRAINING_MASTER_PLAN.md`

---

## 一、任务概述

**目标**：训练一个 **统一 VLM**，根据多视角图像（及失败时的错参提示）预测 / 纠错 **任意注册原子动作** 的可执行参数，从而在闭环中 **驱动真实或仿真机械臂**；**Pick 仅为当前数据与流水线最完整的样例**。

**输入**：
- **主线（`universal_vlm_collect` 默认）**：`save_frames:-1`（可选 `cache_stride`）+ `convert_to_vlm_format.py --mode full --temporal-stride K`；总图约 **T×3**，**T 随 episode 变**；未指定 stride 且未指定 `max-timesteps` 时转换脚本 **默认 K=5**。可加 `--max-timesteps` 封顶。详见 `DATA_AND_TRAINING_MASTER_PLAN.md` §2.2。
- **稀疏 HDF5**（如采集端 `save_frames:5`）：转换时用 **`--mode full --temporal-stride 1`** 用尽 HDF5 内有限个时刻。

**输出**：JSON 格式，含 `action_type` 及该动作的参数字段。

**原则**：多动作统一训练一个大模型，instruction 中显式包含 `action_type`，response 中根据动作类型输出对应参数。

### 1.0 变长多图与训练（Qwen-VL / LLaMA-Factory）

- **数据**：`images` 为列表，**每条样本长度可不同**；`convert_to_llamafactory.py` 会为 Sharegpt 模式生成与图数一致的 `<image>` 前缀。
- **训练**：框架侧通常 **按 batch 内最长序列 padding** 或 dynamic；请 **增大 `cutoff_len`**（如 4096–8192，视显存），使 **最长样本**（最多图 × 每图 visual token + 文本）不被截断。
- **显存**：变长时建议 **`per_device_train_batch_size: 1`**（或 2）+ gradient accumulation；若 OOM，减小 `--max-timesteps` 或提高 `temporal-stride`。
- **元数据**：VLM JSONL 在 `full` 模式下可含 `num_timesteps`、`num_images`、`temporal_stride`（可选），**不参与 loss**，仅调试；Alpaca 转换只使用 `instruction`/`response`/`images`。

### 1.1 视觉–执行–纠错–再执行流程（以 Pick 为样例；其它动作目标一致）

本节以 **Pick** 为例描述「视觉 → 执行 →（失败则）纠错 → 再执行」的**采集与标签语义**；**place / pour 等动作上线后应采用同一套思想**（`params_used`、`correction_gt`、多视角时序），仅参数键与控制器不同。

| 阶段 | 含义 | 实现位置（概念） |
|------|------|------------------|
| **视觉** | 每 episode **3 路 RGB**；`save_frames:-1` + `full` + `temporal_stride` → **T 个时刻 ×3 视角、T 随 episode 变**；稀疏采集时 T 为 HDF5 内时刻数。 | `state['camera_data']` → HDF5 → `convert_to_vlm_format.py` |
| **第一次执行** | 本 episode 确定 **`params_used`**（`pre_offset_x/z`、`after_offset_z`、`euler_deg`、可选带噪 **`picking_position`** 等），驱动 **Pick 原子动作** 完整执行。 | `controllers/pick_controller.py`（`PickTaskController._step_collect` + `PickController.forward`） |
| **纠错（运行时）** | **抓住但高度不足**（已离地约 2 cm 且未到约 10 cm 成功线）：进入 **lift correction**，用 RMP **继续上抬**，相机仍缓存；若连续多步达标则记 **成功**（可带 `was_lift_corrected` / `lift_correction_gt`）。 | 同文件 `_step_collect` 中 `grasped_but_low` 分支 |
| **纠错（标签 / 训练）** | **最终仍失败** 时：据同类 **成功池** 参考 `p_ref`（或噪声 baseline）算 **`correction_gt`**；可选 **`correction_steps`**（`correction_alpha` 比例多步逼近正确空间）。转换后失败样本：**instruction 含当前错参 JSON**，**response 为修正后 JSON**。 | `correction_gt` / `correction_steps` → `task_properties`；`convert_to_vlm_format.py` |
| **再执行** | **同 episode 内**：lift 阶段即「补执行」；**完整 Pick 不会因新 JSON 在同 episode 内自动重跑**（部署时需外层接 VLM）。**下一 episode**：`reset` 后物体重随机、可重新采样噪声，开启新一轮「视觉 + 参数 + 执行」。 | `main.py` 重置循环；部署闭环：看图 → 出参 → 执行 → 失败则带错参再问模型 → 新参再执行 |

**一句话**：视觉贯穿全程 → 按 `params_used` 执行 Pick → 能救则 **lift 纠错** → 救不了则写入 **参数纠错监督**；宏观上 **下一 episode 或部署时** 用新参数「再执行」。

**说明**：当前仿真 **`mode: infer`** 走图像→关节轨迹策略，**未**实现「VLM JSON → Pick → 失败再 VLM 纠错 → 再 Pick」的闭环；该闭环与上表训练语义一致，需在仿真外或扩展 infer 层实现。

---

## 二、原子动作清单与 VLM 就绪状态

| 动作 | action_type | 可操作仪器 | VLM 就绪 | params_used 结构 |
|------|-------------|------------|----------|------------------|
| **pick** | pick | 锥形瓶、烧杯、量筒、容量瓶 | ✅ 已就绪 | pre_offset_x/z, after_offset_z, euler_deg, picking_position |
| **place** | place | 烧杯→目标平台 | ❌ 待扩展 | place_position, release_height, euler_deg（目标） |
| **pour** | pour | 烧杯/锥形瓶/量筒→烧杯 | ❌ 待扩展 | pour_angle, pour_duration, target_position |
| **open** | open | 门、抽屉 | ❌ 待扩展 | handle_grasp_offset, pull_direction, open_angle |
| **close** | close | 门、抽屉 | ❌ 待扩展 | push_position, close_force |
| **press** | press | 按钮 | ❌ 待扩展 | press_position, press_depth |
| **shake** | shake | 烧杯 | ❌ 待扩展 | shake_amplitude, shake_duration |
| **stir** | stir | 烧杯 | ❌ 待扩展 | stir_center, stir_radius, stir_duration |

**当前可训练数据最完整**：**pick**（验证「一模型多动作」流水线的第一站）。其他动作需在 controller 中实现 `set_task_properties`、`params_used`、`correction_gt` 后，扩展 `convert_to_vlm_format.py`，**最终与 pick 样本合并训练同一 VLM**。

**说明**：当前 pick 转换输出不包含 `action_type` 字段（单动作时隐式）。多动作训练时，convert 脚本需扩展为输出 `action_type`，且 instruction 中显式包含「动作类型: pick」等。

---

## 三、统一数据格式

### 3.1 通用 VLM 样本结构（多动作）

每条样本必须包含：

```json
{
  "action_type": "pick",
  "images": ["path1.png", "...", "path_N.png"],
  "instruction": "...",
  "response": "{\"action_type\": \"pick\", \"pre_offset_x\": 0.05, ...}",
  "is_success": true,
  "params_used": {...},
  "object_type": "conical_bottle"
}
```

**action_type**：必填，用于模型区分动作类型并输出对应参数。

### 3.2 各动作 params_used 与 response 格式

#### Pick（已实现）

```json
{
  "action_type": "pick",
  "pre_offset_x": 0.05,
  "pre_offset_z": 0.12,
  "after_offset_z": 0.25,
  "euler_deg": [0, 90, 25],
  "picking_position": [0.28, 0.0, 0.82],
  "is_success": true
}
```

单位：米（m）、度（deg）。

#### Place（目标格式，待实现）

```json
{
  "action_type": "place",
  "place_position": [0.5, 0.0, 0.82],
  "release_height": 0.02,
  "euler_deg": [0, 90, 25],
  "is_success": true
}
```

#### Pour（目标格式，待实现）

```json
{
  "action_type": "pour",
  "pour_angle": 45,
  "pour_duration": 2.0,
  "target_position": [0.3, -0.2, 0.78],
  "is_success": true
}
```

#### Open（目标格式，待实现）

```json
{
  "action_type": "open",
  "handle_grasp_offset": [0.02, 0, 0],
  "pull_direction": [1, 0, 0],
  "open_angle": 90,
  "is_success": true
}
```

#### Close、Press、Shake、Stir（目标格式，待实现）

各动作有独立参数结构，扩展时在 controller 与 convert 脚本中定义。

---

## 四、原始数据（HDF5）格式

### 4.1 目录结构

```
dataset/
├── episode_0000.h5
├── episode_0001.h5
├── ...
└── meta/
    └── episode.jsonl
```

### 4.2 HDF5 内容

| 键名 | 类型 | 说明 |
|------|------|------|
| camera_1_rgb, camera_2_rgb, camera_3_rgb | uint8 [T,H,W,3] | H=W=256 |
| agent_pose | float32 [T,9] | 关节角度 |
| actions | float32 [T,9] | 动作 |
| language_instruction | str | 任务指令 |
| task_properties | JSON str | 见下表 |

### 4.3 task_properties 必含字段（多动作通用）

| 字段 | 类型 | 说明 |
|------|------|------|
| action_type | str | pick / place / pour / open / close / press / shake / stir |
| params_used | dict | 该动作的参数 |
| object_type | str | 物体类型 |
| is_success | bool | 是否成功 |
| correction_gt | dict | 失败时的修正量（可选） |
| correction_steps | list | 多步 correction（可选） |

**当前**：仅 pick 的 HDF5 含完整 task_properties。其他动作采集时需在 controller 中写入。

---

## 五、VLM JSONL 格式（中间格式）

### 5.1 Pick 成功样本

```json
{
  "action_type": "pick",
  "images": ["/path/episode_0000_0.png", "...", "episode_0000_N.png"],
  "instruction": "共 n_t 个时刻×3 视角（共 N 张）…（物体类型: conical_bottle）。动作类型: pick。根据图像预测抓取参数，并判断该原子动作是否成功。输出 JSON: action_type, pre_offset_x, pre_offset_z, after_offset_z, euler_deg, is_success",
  "response": "{\"action_type\": \"pick\", \"pre_offset_x\": 0.05, \"pre_offset_z\": 0.12, \"after_offset_z\": 0.25, \"euler_deg\": [0, 90, 25], \"is_success\": true}",
  "is_success": true,
  "params_used": {...},
  "object_type": "conical_bottle"
}
```

### 5.2 Pick 失败样本（纠错）

```json
{
  "action_type": "pick",
  "images": ["...", "episode_0001_14.png"],
  "instruction": "共 n_t 个时刻×3 视角…。动作类型: pick。当前抓取参数为: {\"pre_offset_x\": 0.08, ...}，执行失败。请修正参数并输出新的 JSON。",
  "response": "{\"action_type\": \"pick\", \"pre_offset_x\": 0.05, ...}",
  "is_success": false,
  "params_used": {...},
  "object_type": "conical_bottle"
}
```

### 5.3 多动作 instruction 模板（扩展用）

- **预测**：`动作类型: {action_type}。根据图像预测{动作名}参数...输出 JSON: action_type, {该动作参数字段}, is_success`
- **纠错**：`动作类型: {action_type}。当前参数为: {params_str}，执行失败。请修正参数并输出新的 JSON。`

### 5.4 图像顺序

**图像顺序**：每个时刻 **cam1 → cam2 → cam3**，共 **T 个时刻** 时列表长度为 **T×3**，**T 随 episode 与 `--temporal-stride` / `--max-timesteps` 变化**。

---

## 六、LLaMA-Factory 格式（Alpaca）

```json
{
  "instruction": "共 n_t 个时刻×3 视角…动作类型: pick...",
  "input": "",
  "output": "{\"action_type\": \"pick\", \"pre_offset_x\": 0.05, ...}",
  "images": ["images/episode_0000_0_0.png", "...", "images/episode_0000_0_14.png"]
}
```

目录结构：
```
labutopia_all/
├── train.jsonl
└── images/
```

---

## 七、数据转换流程

### 7.1 单动作转换（当前仅 Pick）

```bash
python convert_to_vlm_format.py /path/to/dataset --mode full --temporal-stride 5 --output vlm_pick.jsonl
```

**变长 + 固定采样率（与 §1.0 一致）**：采集 `save_frames: -1`（`universal_vlm_collect` 默认），转换使用 **`--mode full`** 与 **`--temporal-stride K`**；未指定 stride 且未指定 `max-timesteps` 时 **默认 K=5**。勿与密存 HDF5 再叠加大 stride 除非有意（见总览 §2.2）。

```bash
python convert_to_vlm_format.py /path/to/dataset --mode full --temporal-stride 5 --output vlm_pick_full.jsonl
python convert_to_vlm_format.py /path/to/dataset --mode full --temporal-stride 5 --max-timesteps 48 --output vlm_pick_full.jsonl
# 仅 max-timesteps、不传 stride：在全长 HDF5 上均匀取 T 个时刻
python convert_to_vlm_format.py /path/to/dataset --mode full --max-timesteps 32 --output vlm_pick_full.jsonl
# 密采样全长：显式 --temporal-stride 1
python convert_to_vlm_format.py /path/to/dataset --mode full --temporal-stride 1 --output vlm_pick_full.jsonl
```

### 7.2 多动作合并

当多个动作均有 VLM 格式数据后：

```bash
# 合并多个 JSONL
cat vlm_pick.jsonl vlm_place.jsonl vlm_pour.jsonl > vlm_all.jsonl

# 转为 LLaMA-Factory 格式
python convert_to_llamafactory.py vlm_all.jsonl -o /path/to/LLaMA-Factory/data/labutopia_all
```

**注意**：`convert_to_llamafactory.py` 会合并图像到同一 `images/` 目录，需确保不同 dataset 的 episode 命名不冲突（如 `pick_episode_0000_0.png`、`place_episode_0000_0.png`）。若冲突，需在转换时加前缀。

### 7.3 扩展 convert_to_vlm_format.py 支持多动作

当 place、pour 等 controller 就绪后，convert 脚本需：

1. 从 `task_properties` 读取 `action_type`
2. 根据 `action_type` 选择 instruction 模板与 response 构建逻辑
3. 输出样本中写入 `action_type` 字段

---

## 八、采集配置与命令

### 8.1 已就绪（Pick）

| 配置 | 物体 | 样本数 | 命令 |
|------|------|--------|------|
| level1_pick_noise_universal | 单锥形瓶 | 50 | `python main.py --config-name level1_pick_noise_universal --headless --no-video` |
| level1_pick_noise_universal_multi_obj | 3 种锥形瓶 | 150 | `python main.py --config-name level1_pick_noise_universal_multi_obj --headless --no-video` |
| level1_pick_noise_universal_all_obj | 4 类仪器 | 每类 **100 成功**切换，全类完成结束 | `python main.py --config-name level1_pick_noise_universal_all_obj --headless --no-video` |
| level1_pick_noise_universal_*_fail | 同上 | 同上 | 高失败率，90% 2x 噪声 |

### 8.2 待扩展（其他动作）

| 动作 | 配置 | 说明 |
|------|------|------|
| place | level1_place_noise_universal | 需 controller 实现 params_used、noise、correction_gt |
| pour | level1_pour_noise_universal | 同上 |
| open | level1_open_noise_universal | 同上 |
| close | level1_close_noise_universal | 同上 |
| press | level1_press_noise_universal | 同上 |
| shake | level1_shake_noise_universal | 同上 |
| stir | level1_stir_noise_universal | 同上 |

**采集原则**：每类仪器 150 ep，继承 `universal_vlm_collect`；**`save_frames` 默认 `-1`（全长）**，可按总览 §2.2 覆盖为更大 `cache_stride` 或更小 `save_frames` 省盘。

---

## 九、LLaMA-Factory 安装与配置

### 9.1 安装

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 9.2 注册数据集

**单动作（Pick）**：
```json
"labutopia_pick": {
  "file_name": "labutopia_pick",
  "formatting": "alpaca",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "images": "images"
  }
}
```

**多动作（合并后）**：
```json
"labutopia_all": {
  "file_name": "labutopia_all",
  "formatting": "alpaca",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "images": "images"
  }
}
```

### 9.3 训练配置

```yaml
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
template: qwen2_vl

dataset: labutopia_pick
dataset_dir: data
max_samples: 5000
cutoff_len: 2048   # 变长多图时酌情调大（如 4096–8192），避免截断
preprocessing_num_workers: 4

stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 64
lora_alpha: 128

output_dir: outputs/labutopia_vlm
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

多动作时：`dataset: labutopia_all`，`max_samples` 按总样本量调整（建议 5000–10000）。

---

## 十、训练执行

```bash
cd LLaMA-Factory
llamafactory train config/labutopia_qwen25vl_lora.yaml
```

多卡：`CUDA_VISIBLE_DEVICES=0,1 llamafactory train config/labutopia_qwen25vl_lora.yaml`

---

## 十一、模型输出格式（推理时）

模型输出 JSON，**必须含 action_type**，其余字段按动作类型：

| action_type | 必含字段 |
|-------------|----------|
| pick | pre_offset_x, pre_offset_z, after_offset_z, euler_deg, is_success |
| place | place_position, release_height, euler_deg, is_success |
| pour | pour_angle, pour_duration, target_position, is_success |
| open | handle_grasp_offset, pull_direction, open_angle, is_success |
| close | push_position, close_force, is_success |
| press | press_position, press_depth, is_success |
| shake | shake_amplitude, shake_duration, is_success |
| stir | stir_center, stir_radius, stir_duration, is_success |

---

## 十二、数据量建议

| 阶段 | 动作 | 样本量 | 说明 |
|------|------|--------|------|
| 验证 | pick | 100–200 | 快速验证流程 |
| 单动作 | pick | 500–600 | 单动作基线 |
| 多动作 | pick + place + pour | 1500–2000 | 3 动作统一 |
| 全量 | 8 动作 | 5000–10000 | 全原子动作统一 |

---

## 十三、完整命令示例（多动作流程）

```bash
# 1. 转换各动作（当前仅 pick 有数据）
python convert_to_vlm_format.py /data/pick/dataset --mode full --temporal-stride 5 -o vlm_pick.jsonl
# 未来：python convert_to_vlm_format.py /data/place/dataset --action-type place -o vlm_place.jsonl

# 2. 合并（多动作时）
cat vlm_pick.jsonl > vlm_all.jsonl
# 未来：cat vlm_pick.jsonl vlm_place.jsonl vlm_pour.jsonl > vlm_all.jsonl

# 3. 转 LLaMA-Factory
python convert_to_llamafactory.py vlm_all.jsonl -o /workspace/LLaMA-Factory/data/labutopia_all

# 4. 训练
cd /workspace/LLaMA-Factory
llamafactory train config/labutopia_qwen25vl_lora.yaml
```

---

## 十四、校验清单

- [ ] HDF5 含 `task_properties`、`action_type`（或可推断）、`params_used`
- [ ] VLM JSONL 每行含 `action_type`、`instruction`、`response`、`images`
- [ ] instruction 中显式包含动作类型
- [ ] response 为 JSON，含 `action_type` 及该动作参数字段
- [ ] LLaMA-Factory `train.jsonl` 路径正确，`dataset_info` 已注册
- [ ] 多动作时样本量均衡，避免单一动作占比过高

---

## 十五、常见错误与处理

| 错误 | 原因 | 处理 |
|------|------|------|
| 转换跳过全部 | 无 params_used | 使用带 noise 的采集配置 |
| 转换仅支持 pick | convert 脚本未扩展 | 其他动作需先扩展 controller 与 convert |
| 多动作 response 解析失败 | 缺少 action_type | 推理端根据 action_type 解析对应字段 |
| OOM | batch 过大 | `per_device_train_batch_size: 1`，`cutoff_len: 1024` |
| 单动作主导 | 样本不均衡 | 按动作分层采样或 oversample 少样本动作 |

---

## 十六、参考文档

- **总览**：`docs/DATA_AND_TRAINING_MASTER_PLAN.md`（采集→转换→训练单入口）
- 采集细节：`docs/VLM_DATA_COLLECTION.md`
- 训练机部署：`docs/VLM_TRAINING_SERVER_GUIDE.md`

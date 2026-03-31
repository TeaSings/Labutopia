# LabUtopia VLM 数据与训练说明（训练服务器部署）

本文档用于在**另一台服务器**上部署训练流程，无需 Isaac Sim 环境。采集在 LabUtopia 仿真机完成，训练在 GPU 服务器完成。

**目的对齐**：训练产出用于 **让大模型驱动机械臂执行多种动作**；当前数据集以 **Pick 实验** 为主，训练脚本与数据集名（如 `labutopia_pick`）可随多动作合并而扩展。

**端到端主线**：请先读 [DATA_AND_TRAINING_MASTER_PLAN.md](DATA_AND_TRAINING_MASTER_PLAN.md)。

---

## 一、数据来源

### 1.1 采集机输出

在 LabUtopia 仿真机上采集后，数据位于：

```
outputs/collect/YYYY.MM.DD/HH.MM.SS_配置名/dataset/
├── episode_0000.h5
├── episode_0001.h5
├── ...
└── meta/episode.jsonl
```

**需要拷贝到训练机的目录**：`dataset/` 及其中的 `episode_*.h5` 文件。

### 1.2 推荐采集配置

| 配置 | 物体 | 样本数 | 用途 |
|------|------|--------|------|
| `level1_pick_noise_universal` | 单锥形瓶 | 50 ep | 快速验证 |
| `level1_pick_noise_universal_multi_obj` | 3 种锥形瓶 | 150 ep | 推荐 |
| `level1_pick_noise_universal_all_obj` | 4 类仪器 | 每类 **100 次成功**后切换，类齐全后结束 | 全量（默认噪声） |
| `level1_pick_noise_universal_all_obj_fail` | 4 类仪器 | 同上切换逻辑 | 高失败率，平衡成功/失败 |

采集命令示例：
```bash
python main.py --config-name level1_pick_noise_universal_multi_obj --headless --no-video
```

**采集与转换对齐**：`universal_vlm_collect` 默认 **`save_frames: -1`**（全长 HDF5）→ 转换 **`--mode full`**，配合 **`--temporal-stride K`**（未指定 stride 且未指定 `max-timesteps` 时脚本 **默认 K=5**）→ **张数随 episode 变**。详见 `DATA_AND_TRAINING_MASTER_PLAN.md` §2.2。

### 1.3 采集语义：视觉–执行–纠错–再执行（Pick）

理解每条 HDF5 与转换后 JSONL 在「预测 / 纠错」上的含义时，请阅读 **[LLM_TRAINING_REFERENCE.md](LLM_TRAINING_REFERENCE.md)** 第一节下的 **「1.1 视觉–执行–纠错–再执行流程（Pick，采集与训练语义）」**（表格 + 部署闭环说明）。

---

## 二、训练机环境要求

| 项目 | 要求 |
|------|------|
| GPU | 建议 1×24GB（A10/4090），Qwen2.5-VL-7B LoRA |
| 系统 | Linux 推荐，Windows 亦可 |
| Python | 3.8+ |
| 依赖 | h5py, numpy, Pillow（仅数据转换）；PyTorch, LLaMA-Factory（训练） |

**无需**：Isaac Sim、Omniverse、CUDA 特殊版本（按 LLaMA-Factory 要求即可）。

---

## 三、数据转换（在训练机上执行）

### 3.1 拷贝必要文件到训练机

从 LabUtopia 项目拷贝到训练机：

1. **数据集**：`outputs/collect/.../dataset/` 整个目录（含 `episode_*.h5`）
2. **转换脚本**：
   - `scripts/convert_to_vlm_format.py`
   - `scripts/convert_to_llamafactory.py`

### 3.2 安装转换依赖

```bash
pip install h5py numpy Pillow
```

### 3.3 转为 VLM 格式（JSONL + 图像）

```bash
# 全长 HDF5（save_frames=-1）
python convert_to_vlm_format.py /path/to/dataset --mode full --temporal-stride 5 --output vlm_train.jsonl

# 稀疏 T 时刻 HDF5（如 save_frames=5）：用尽索引
python convert_to_vlm_format.py /path/to/dataset --mode full --temporal-stride 1 --output vlm_train.jsonl
```

- `--mode single`：1 张/样本，省空间
- `--mode full`：**多时刻×3 视角**；建议采集 `save_frames: -1`；`--temporal-stride K` 控制 HDF5 索引步长（未指定 stride 且未指定 `--max-timesteps` 时 **默认 K=5**）；可加 `--max-timesteps T` **封顶**（总图约 ≤**T×3**）
- 输出：`vlm_train.jsonl`、`vlm_images/`（与 jsonl 同目录）

### 3.4 转为 LLaMA-Factory 格式

```bash
# 输出到 LLaMA-Factory 的 data 目录
python convert_to_llamafactory.py vlm_train.jsonl -o /path/to/LLaMA-Factory/data/labutopia_pick
```

输出目录结构：
```
labutopia_pick/
├── train.jsonl
└── images/
    ├── episode_0000_0.png
    ├── ...
```

---

## 四、LLaMA-Factory 安装与配置

### 4.1 安装

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 4.2 注册数据集

编辑 `LLaMA-Factory/data/dataset_info.json`，添加：

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

确保 `data/labutopia_pick/train.jsonl` 和 `data/labutopia_pick/images/` 存在。

### 4.3 训练配置

创建或复制 `config/labutopia_qwen25vl_lora.yaml`：

```yaml
### 模型
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
template: qwen2_vl

### 数据
dataset: labutopia_pick
dataset_dir: data
max_samples: 5000
cutoff_len: 2048
preprocessing_num_workers: 4

### LoRA
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 64
lora_alpha: 128

### 训练
output_dir: outputs/labutopia_qwen25vl
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

**关键参数说明**：

| 参数 | 说明 | 建议 |
|------|------|------|
| `per_device_train_batch_size` | 每卡 batch | **变长多图**建议 **1** + grad accum；短序列可试 2，OOM 再减 |
| `cutoff_len` | 最大序列长度 | 1024–2048，过长易 OOM |
| `max_samples` | 最大训练样本数 | 按实际数据量调整，可设 `null` 用全量 |
| `num_train_epochs` | 训练轮数 | 3–5 |

---

## 五、训练执行

```bash
cd LLaMA-Factory
llamafactory train config/labutopia_qwen25vl_lora.yaml
```

小规模验证（100 条、1 epoch）：
```yaml
# 在 YAML 中临时添加
max_samples: 100
num_train_epochs: 1
```

多卡训练（示例 2 卡）：
```bash
CUDA_VISIBLE_DEVICES=0,1 llamafactory train config/labutopia_qwen25vl_lora.yaml
```

---

## 六、输出与部署

- **输出目录**：`outputs/labutopia_qwen25vl/`（含 adapter、checkpoint）
- **推理**：使用 LLaMA-Factory 的 `llamafactory chat` 或导出合并后的模型
- **接入 LabUtopia**：将模型路径配置到推理引擎，闭环验证「图像→参数→执行」

---

## 七、常见问题

| 问题 | 处理 |
|------|------|
| OOM | 降低 `per_device_train_batch_size` 为 1，或 `cutoff_len` 为 1024 |
| 图像路径错误 | 确保 `convert_to_llamafactory.py` 使用 `-o` 输出到 LLaMA-Factory `data/`，且默认复制图像 |
| 转换报错「无 task_properties」 | 检查 HDF5 是否来自带 noise 的采集配置（如 `level1_pick_noise_universal`） |
| 数据量少 | 至少 100–200 条用于验证；500+ 条用于正式训练 |

---

## 八、快速检查清单

- [ ] 采集数据 `dataset/` 已拷贝到训练机
- [ ] `convert_to_vlm_format.py`、`convert_to_llamafactory.py` 已拷贝
- [ ] 已生成 `vlm_train.jsonl` 和 `vlm_images/`
- [ ] 已生成 `labutopia_pick/train.jsonl` 和 `labutopia_pick/images/`
- [ ] `dataset_info.json` 已添加 `labutopia_pick`
- [ ] 训练配置 YAML 中 `dataset: labutopia_pick`、`dataset_dir: data`
- [ ] GPU 显存足够（建议 24GB）

---

## 九、参考

- 总览：`docs/DATA_AND_TRAINING_MASTER_PLAN.md`
- 采集：`docs/VLM_DATA_COLLECTION.md`
- 格式与多动作：`docs/LLM_TRAINING_REFERENCE.md`

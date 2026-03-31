# 采集速度加速指南

总流程见 [DATA_AND_TRAINING_MASTER_PLAN.md](DATA_AND_TRAINING_MASTER_PLAN.md)（目标为 **统一大模型驱动多动作机械臂**；下文以 Pick 采集为例）。

## 一、命令行参数


| 参数              | 作用                                     | 加速效果   |
| --------------- | -------------------------------------- | ------ |
| `--headless`    | 无 GUI，减少渲染开销                           | ⭐⭐⭐ 必用 |
| `--no-video`    | 不保存 mp4 视频，仅写 HDF5                     | ⭐⭐ 明显  |
| `--fast-sim`    | 物理步长 1/30s、禁用视口/Motion BVH（仅 headless） | ⭐⭐ 明显  |
| `--backend gpu` | GPU 物理引擎（需支持）                          | ⭐⭐ 明显  |


```powershell
# 推荐采集命令（加速）
python main.py --config-name level1_pick_noise_universal_5obj --headless --no-video

# 仿真加速（物理步长增大、视口/Motion BVH 关闭）
python main.py --config-name level1_pick_noise_universal_all_obj_fail_200success --headless --no-video --fast-sim

# 若有 GPU 物理支持
python main.py --config-name level1_pick_noise_universal_5obj --headless --no-video --backend gpu
```

---

## 二、配置优化

### 1. 快速采集配置

`**level1_pick_noise_universal_fast**`（通用快速）：

- `save_frames: 2`（首+末帧）→ 写入量减少
- `compression: null`（无 gzip）→ 写入更快
- `max_steps: 400`（缩短超时）→ 失败 episode 更快结束

`**level1_pick_noise_universal_all_obj_fail_200success_fast**`（全仪器 200 成功 + 快速）：

- 继承 200success，仅覆盖 `max_steps: 400`
- 配合 `--fast-sim` 使用效果最佳

```powershell
python main.py --config-name level1_pick_noise_universal_fast --headless --no-video

# 全仪器 200 成功 + 仿真加速
.\scripts\run_collect_headless.ps1 level1_pick_noise_universal_all_obj_fail_200success_fast --no-video --fast-sim
```

### 2. 手动调参


| 配置项                     | 默认                 | 加速建议      | 说明                                                             |
| ----------------------- | ------------------ | --------- | -------------------------------------------------------------- |
| `collector.save_frames` | `-1`（universal 默认） | `2` 或 `1` | 越少越省 I/O；全长见总览 §2.2；转换用 `convert_to_vlm_format.py --mode full` |
| `collector.compression` | gzip               | null      | 不压缩，写入快，文件更大                                                   |
| `task.max_steps`        | 800                | 400~500   | 超时更早，失败 episode 更快重置                                           |
| `cameras[].resolution`  | [256,256]          | [128,128] | 分辨率降低，渲染与存储更快（需在配置中覆盖）                                         |


---

## 三、原子动作时长（Pick）

Pick 的 7 个 phase 总时长约 0.1s，已在合理范围。若需进一步压缩，可修改 `controllers/pick_controller.py` 中 `events_dt`，但过短可能导致物理不稳定。

---

## 四、多机并行

多台机器可同时跑不同配置或同一配置，事后合并数据集：

```powershell
# 机器 A
python main.py --config-name level1_pick_noise_universal_5obj --headless --no-video

# 机器 B（同一命令，输出到不同 run_dir）
python main.py --config-name level1_pick_noise_universal_5obj --headless --no-video
```

合并时用 `scripts/merge_dataset.py` 或手动合并 HDF5 + episode.jsonl。

---

## 五、优先级建议

1. **必做**：`--headless --no-video`
2. **推荐**：`--fast-sim`（物理步长 1/30s、禁用视口/Motion BVH）
3. **推荐**：使用 `level1_pick_noise_universal_all_obj_fail_200success_fast` 或 `level1_pick_noise_universal_fast`
4. **可选**：`--backend gpu`、降低分辨率、多机并行


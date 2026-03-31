# 仿真内实时 VLM：初始推断 → 执行 → 纠错

在 Isaac Sim 中运行 **抓取** 任务时，用 HTTP VLM（与 `run_docx/VLM_MODEL_USAGE.md` 一致的 OpenAI 兼容接口）完成：

1. **初始推断（任务①）**：首帧三视角 → 解析 JSON 得到 `pre_offset_*`、`after_offset_z`、`euler_deg`、`picking_position`。
2. **执行**：用 `PickController` 跑完一次完整 pick。
3. **成功判断（任务②）**：用本段过程里缓冲的多帧图（时间 × cam1→cam2→cam3）请求 VLM，解析 **`is_success`**。仅当 **`is_success: true`** 时本 episode 记为**成功**并结束（不再依赖仅物理抬升阈值）。
4. **纠错（任务③）**：若 **`is_success: false`** 且模型未要求停止（**`terminate_correction_loop: false`**），默认**先把机械臂关节插值回本 episode 开始时记录的初始姿态**（再请求纠错 JSON），`pick_controller.reset()` 后**同一 episode 内**再抓。可通过 `vlm_live.recover_to_home_after_failure: false` 关闭回退、直接纠错。**`max_correction_attempts: 0`** 表示不限制纠错轮数，直到模型判成功；设正整数为安全上限。

### 两种「重置」不要混淆

| | **整局重置**（物体碰倒、episode 正常结束等） | **纠错中的关节回退**（`recover_to_home_after_failure`） |
|---|---------------------------------------------|--------------------------------------------------------|
| **触发** | `reset_needed` → `main` 里 `task_controller.reset()` 后 **`task.reset()`** | 任务②失败后，同一 episode 内 |
| **做什么** | `world.reset()`、机器人初始化、物体在 `randomize_object_position` 中重新摆放；旋转在改位置前 **`snapshot_object_xform_rotation`**、改后 **`restore_object_xform_rotation`**（保留 USD 默认朝向，不强制 identity） | **仅**机械臂关节插值到本局开始时记录的关节角；再请求纠错、再 pick |
| **物体姿态** | 重新随机摆放并恢复旋转 | **不**动场景物体（仍可能是上一段尝试后的姿态） |

碰倒检测命中时属于 **整局重置**，与纠错关节回退无关。

## 前置条件

- 本机或 SSH 转发可访问的 VLM 服务，例如：
  - `curl http://127.0.0.1:8000/v1/models` 返回模型列表。
- 配置中 `vlm_live.model` 与 `GET /v1/models` 的 `data[].id` **完全一致**（例如 `Qwen3.5-9B-LabUtopia-lora`）。
- 若服务端启用 `API_KEY`，在 YAML 的 `vlm_live.api_key` 或环境变量 `OPENAI_API_KEY` 中填写（可在 `vlm_api_client` 中扩展读取 env，当前以 YAML 为准）。

## 配置与启动

使用 `config/level1_pick_vlm_live.yaml`：

```bash
cd LabUtopia
python main.py --config-name level1_pick_vlm_live
```

Headless：

```bash
python main.py --config-name level1_pick_vlm_live --headless
```

主要字段：

| 字段 | 含义 |
|------|------|
| `mode` | 必须为 `vlm_live` |
| `collector.type` | 建议 `mock`（不写 HDF5） |
| `vlm_live.base_url` | 含 `/v1` |
| `collector.cache_stride` | 训练采集 `universal_vlm_collect` 中每多少步 `cache_step` 记 1 帧（默认 **20**）；`vlm_live` 未写 `frame_stride` 时，多帧缓冲与此对齐（见 `DATA_AND_TRAINING_MASTER_PLAN.md` §2.2） |
| `vlm_live.frame_stride` | 可选；显式覆盖上述间隔。每隔多少仿真步把三视角各 1 张图追加进缓冲（越大时序越稀、越贴近训练 HDF5 时间采样） |
| `vlm_live.max_correction_attempts` | **0** = 不限制，直到模型 `is_success` 或 `terminate_correction_loop`；正整数 = 纠错安全上限 |
| `vlm_live.recover_to_home_after_failure` | 默认 **true**：任务②失败后先回 episode 初始关节再纠错；**false** 则直接纠错（旧行为） |
| `vlm_live.recovery_joint_alpha` / `recovery_joint_tol` / `recovery_max_steps` | 关节回退：每步混合系数、到位阈值（弧度）、最大步数（超时强制进入纠错） |

## 参数噪声（采集 vs 测试）

- **采集** `mode: collect` 且 `noise.enabled: true` 时，`PickTaskController` 才会对 `pre_offset_*` / `picking_position` 等加 `_episode_noise`（用于失败样本与 correction 训练）。
- **`vlm_live` 与 `infer` 在代码中强制 `_noise_enabled = False`**，即使 YAML 里误开 `noise.enabled`，也不会对抓取参数加噪；`level1_pick_vlm_live.yaml` 亦显式写了 `noise.enabled: false`。
- `vlm_live` 下关节目标完全来自 **VLM 返回的 JSON**（及纠错 API），**不经过**采集噪声逻辑。

## 物体类型如何传给 VLM

- 任务 `state` 需由 `BaseTask.get_basic_state_info(object_path=...)` 提供 **`object_category`**（语义类别，如 `beaker`）与 **`object_name`**（实例名，如 `beaker_2`）。`PickTask` / `SingleObjectTask` 在 `current_obj_path` 有效时会自动带上。
- `pick_controller._vlm_object_type_for_prompt(state)` 会优先用 **`object_category`** 写入三处 API 的「物体类型」；若与实例名不一致则格式化为 `类别（目标物体名: 实例名）`，便于模型区分多实例。
- 若二者皆缺，控制台会打印 **`[Pick][VLM] 警告：state 缺少 object_category 与 object_name`**，且提示词中物体类型退化为 `unknown`。运行日志中「**物体类型(发给 VLM)**」可与初始推断 / 判断 / 纠错行对照检查。

## 实现位置

- `controllers/pick_controller.py`：`mode == "vlm_live"` → `_step_vlm_live`
- `utils/vlm_api_client.py`：HTTP 与提示词
- `utils/vlm_image_utils.py`：CHW → HWC，与采集 HDF5 一致

## 如何确认 `camera_data` 是否正常（与预览黑屏区分）

`main.py` 里黑屏提示针对的是 **`camera_display`**（录屏/窗口）。VLM 使用的是 **`camera_data`**。

运行时加 **`--debug-camera-stats`**：每隔约 120 步打印各相机 `shape / mean / min / max`。  
若 **mean 明显高于 ~5～10**（uint8 图像通常几十以上），说明 **`camera_data` 有正常亮度**；若 mean 也接近 0，则 VLM 同样会收到黑图，需排查渲染或 `ENABLE_CAMERAS`。

在触发「连续多帧预览全黑」警告时，同一帧会额外打印一行 **`[Main] 对照（同一帧）`** 的 `camera_data` 统计（无需再加参数）。

### 冷启动「有时黑、再跑一次就好」

RTX/着色器/相机 pipeline 首次加载较慢，`main.py` 已在 **`task.reset()` 之后、主循环之前**自动做 **`--render-warmup-steps`**（默认 **headless 为 96 步**，非 headless 为 24；设为 **0** 可关闭）。控制台会打印 **`[Main] 渲染预热完成`**。若仍偶发黑图，可加大例如 `--render-warmup-steps 200`，或在配置里调高 **`task.camera_warmup_frames`**（`base_task.check_frame_limits` 中在返回 state 前跳过的帧数，默认 **45**）。

## 为何「纠错后」又出现「初始推断」？

**同一 episode 内**：纠错只会更新参数并 `pick_controller.reset()`，**不会**把 `_vlm_init_done` 置回 `False`，因此**不应**再次调用初始推断。

若你在日志里先看到「已应用第 N 次纠错」，随后又出现 **`[VLM] 初始推断完成`**，通常表示 **上一个 episode 已结束并触发了环境/task 的 `reset`**（例如：抓取仍失败且**无法再发起纠错**——常见原因是 **`frame_stride` / `cache_stride` 较大导致失败当帧缓冲里不足 3 张图**，旧逻辑会直接结束 episode）。当前版本已在失败当帧 **用当前 `camera_data` 三视角补齐缓冲**；**默认多帧采样间隔与训练采集一致**（`level1_pick_vlm_live` 中 `collector.cache_stride: 20`，与 `universal_vlm_collect` 对齐）。需要更密的时间采样可在 YAML 中显式设 `vlm_live.frame_stride: 1`。若仍异常，请看控制台是否打印「缓冲仍不足 3 张」及 **`[Pick][VLM] 多帧缓冲采样`** 一行中的实际 stride。

**若「已应用纠错」下一行立刻出现 `Episode Stats`，且下一局 `本轮前纠错次数` 又回到 0**：常见原因是 **`main.py` 里 `should_reset = … or (task.need_reset() and atomic_done)` 与 `task.max_steps` 超时叠加**：纠错路径里已对 `pick_controller.reset()`，但**下一帧**主循环在 `task.step()` 之前判断 `atomic_done` 时，仍可能把「上一段 pick 已结束」当成原子已完成，与 `task.reset_needed`（已超 `max_steps`）同时为真 → **误判整局重置**。**`vlm_live` 下已改为整局结束仅看 `task_controller.need_reset()`**；`task.max_steps` 不再参与是否 `reset`（仍可在 YAML 里保留大值作记录/其它用途）。另：此前在 `world.is_stopped()` 时误置 `reset_needed` 的逻辑已移除。

## 物体倾倒

- **`vlm_live`**：每步 `_maybe_capture_tipped_baseline` 锁定本局 **初始位姿（R₀）**；`_check_tipped_reset_episode` 在 **关节回退中**、**原子抓取执行中（每步）**、以及 **抓取结束且三视角缓冲就绪之后、调用任务② API 之前** 执行（**物理优先于 VLM**）。已碰倒则不再调用成功判断 API。触发 **整局重置**（`reset_needed` + `_early_return`），与纠错关节回退无关。若肉眼已倒仍不重置，控制台可能打印 **物理相对转角** 与阈值，便于调 `pick.tipped_max_tilt_deg` 或检查 `object_path` 是否对应实际转动的刚体。
- **`collect`**：在 **物理上已判定 pick 未成功**（及 lift 补救用尽失败）之后判倒，再决定是否写失败 HDF5。
- **`infer`**：在 **本步策略推理与成功计数更新之后**，若仍未达到连续抬升成功，再判倒。
- 关闭：`pick.tipped_detection_enabled: false`。阈值与轴：`pick.tipped_max_tilt_deg`、`pick.tipped_local_up_axis`。
- **根 xform 旋转不变**（日志里根旋转角长期为 0°）时：启用 **`pick.tipped_bbox_fallback_enabled`**（默认 true），用 **世界轴对齐包围盒**（`object_utils.get_object_size`，含子树几何）的竖直度 `sz/max(sx,sy)`：初值足够「高瘦」时，若当前值相对初值按 **`tipped_bbox_verticality_factor`** 明显下降则判倒。扁物体可调高 **`tipped_bbox_min_initial_verticality`** 以免误报。

## 限制说明

- **成功条件**以 VLM **任务②** 的 `is_success` 为准，不再用物理 `required_success_steps` 抬升作为 episode 成功判据。
- **`terminate_correction_loop: true`** 时本 episode 记为失败并停止纠错（模型认为无需再试）。
- 每轮 pick 结束至少 **1 次成功判断 API**；若失败再 **1 次纠错 API**，仿真会阻塞在 HTTP 上，属预期。

## 与离线测试脚本的关系

- `run_docx/test_vlm_api_real_collect.py`：用磁盘 `episode_*.h5` 测三种任务。
- 本模式：用**实时仿真图像**走「初始 + **模型判断** + 纠错」闭环，与 `VLM_MODEL_USAGE.md` §5 对齐。

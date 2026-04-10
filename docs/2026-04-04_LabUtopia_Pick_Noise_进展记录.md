# LabUtopia Pick / Place 进展速查

- 更新时间：2026-04-10
- 适用范围：`pick` 与 `place` 当前主线
- 用途：快速回答四个问题
  - 你在原项目基础上到底做了什么
  - `pick` 现在该看哪个配置
  - `place` 现在该看哪个配置
  - 下一步应该切去哪个动作

## 一、先看结论

当前已经有三条可以直接记住的稳定线：

| 动作 | 版本 | 配置文件 | 结果 | 当前结论 |
| --- | --- | --- | --- | --- |
| `pick` | 3 物体正式版 | `config/level1_pick_stratified_all_obj.yaml` | `112/450 written`，`89 skipped`，`24.9% success` | 已达标，固定使用 |
| `pick` | 5 物体泛化版 | `config/level1_pick_stratified_all_obj_5obj.yaml` | `179/750 written`，`107 skipped`，`23.9% success` | 已达标，固定使用 |
| `place` | 当前冻结噪声版 | `config/level1_place_noise_ring_v2.yaml` | `53/183 written`，`29.0% success` | 已达到当前想要的成功率区间，冻结使用 |

当前阶段不要再继续围绕 `pick` 和 `place` 做细碎参数搜索，除非学长提出新的目标。下一步应转入 `pour`。

## 二、你在原项目基础上实际做了什么

这部分只保留真正重要的增量，不再按聊天时间线展开。

### 1. 把 `pick` 从“能跑”推进到“能稳定采集”

- 重新确认了远程真正该用的仓库和环境：
  - 仓库：`/home/ubuntu/teasings_projects/Labutopia`
  - 环境：`labutopia_51`
- 明确了当前问题不再是“系统能不能启动”，而是“如何控制采集分布和成功率”。

### 2. 把关键动作参数从 controller 硬编码提到 YAML

涉及：

- `controllers/pick_controller.py`
- `controllers/place_controller.py`
- `controllers/atomic_actions/place_controller.py`
- `config/level1_pick.yaml`
- `config/level1_place.yaml`

核心变化：

- `pick` 里的 `pre_offset_x / pre_offset_z / after_offset_z / end_effector_euler_deg`
- `place` 里的 `pre_place_z / place_offset_z / release_position_threshold / retreat_offset_* / euler_deg`
- `place` 的 `pick_events_dt / place_events_dt`

现在这些参数都能从配置里直接调，不需要每次改 controller。

### 3. 在任务层补齐了 `pick` 的分层采集逻辑

涉及：

- `tasks/base_task.py`

补齐后的关键能力：

- `task.object_switch_metric`
- `task.object_switch_interval`
- `task.position_switch_interval`
- `task.stratified_collection`
- `task.successes_per_pose`
- `task.episodes_per_pose`

这一步的意义是：

- 不再只能按 success 粗糙切换物体
- 不再每个 episode 都强制换 pose
- 可以真正实现“同一 pose 连续采多局，再换 pose”的 `stratified` 语义

### 4. 把 `pick` 主线整理成稳定配置

涉及：

- `config/level1_pick_stratified_all_obj.yaml`
- `config/level1_pick_stratified_all_obj_5obj.yaml`

结果：

- 3 物体正式版达标
- 5 物体泛化版达标
- `pick` 当前不再作为主要调参对象

### 5. 把 `place` 从 baseline 走到可控噪声版

涉及：

- `config/level1_place.yaml`
- `config/level1_place_1ep.yaml`
- `config/level1_place_noise*.yaml`
- `controllers/place_controller.py`

关键过程：

- 先做 `place` baseline 冒烟，验证 `pick -> place` 链路能稳定跑通
- 再把 `place` 也改成“保留 clean baseline，只在派生配置上加噪”
- 第一轮 `cartesian` 噪声能压低成功率，但很难解释，且容易出现 `z` 向异常
- 最后切换到 `ring` 思路：围绕目标中心按半径采样平面偏移，让噪声结构直接对齐 `xy` 成功判定

结果：

- `config/level1_place_noise_ring_v2.yaml` 成为当前冻结版本

## 三、当前配置速查表

### 1. 现在真正应该记住的 `pick` 配置

| 配置文件 | 用途 | 是否达标 | 是否继续改 |
| --- | --- | --- | --- |
| `config/level1_pick_stratified_all_obj.yaml` | 3 物体正式主配置 | 是 | 先不要动 |
| `config/level1_pick_stratified_all_obj_5obj.yaml` | 5 物体泛化正式配置 | 是 | 先不要动 |
| `config/level1_pick_stratified_all_obj_1ep.yaml` | 3 物体正式版冒烟测试 | 不看成功率，只看能否跑通 | 可以继续用 |
| `config/level1_pick_stratified_all_obj_schedule_debug.yaml` | 关闭噪声的纯调度验证 | 仅用于 debug | 一般不用 |

### 2. 现在真正应该记住的 `place` 配置

| 配置文件 | 用途 | 当前定位 |
| --- | --- | --- |
| `config/level1_place.yaml` | `place` clean baseline | 保留，不继续硬改 |
| `config/level1_place_1ep.yaml` | `place` baseline 冒烟测试 | 继续保留 |
| `config/level1_place_noise_ring_v2.yaml` | `place` 当前冻结噪声版 | 当前主线 |
| `config/level1_place_noise_ring_v2_1ep.yaml` | `place` 当前主线冒烟测试 | 继续保留 |

### 3. 现在已经不该再作为主线继续推进的 `place` 配置

| 配置文件 | 角色 | 当前定位 |
| --- | --- | --- |
| `config/level1_place_noise.yaml` | 第一版 `cartesian` 噪声 | 历史中间阶段 |
| `config/level1_place_noise_hard.yaml` | 过强噪声试探 | 历史极端方案 |
| `config/level1_place_noise_v2.yaml` | 收窄 `z` 后的中间方案 | 历史中间阶段 |
| `config/level1_place_noise_v3.yaml` | 继续回收 `xy` 的中间方案 | 历史中间阶段 |
| `config/level1_place_noise_ring.yaml` | 第一版 ring 噪声 | 历史中间阶段 |

一句话概括就是：

- `pick` 看 `stratified_all_obj`
- `place` 看 `noise_ring_v2`
- 不要再回到 `cartesian` 那几版继续打转

## 四、最终结果速记

### 1. `pick` 3 物体正式版

- 配置：`config/level1_pick_stratified_all_obj.yaml`
- 结果：`112/450 written`
- `skipped`：`89`
- 成功率：`24.9%`
- 当前固定参数：`noise_scale: 2.45`

### 2. `pick` 5 物体泛化版

- 配置：`config/level1_pick_stratified_all_obj_5obj.yaml`
- 第一轮结果：`142/750 written`，`105 skipped`，`18.9% success`
- 第二轮结果：`179/750 written`，`107 skipped`，`23.9% success`
- 最终固定参数：`noise_scale: 2.35`

### 3. `place` 当前冻结版

- baseline：`config/level1_place.yaml`
- 冻结噪声版：`config/level1_place_noise_ring_v2.yaml`
- 关键思路：
  - 保留 baseline
  - 不改学长成功判定
  - 把平面噪声改成 ring 采样
- 当前结果：`53/183 written`，`29.0% success`

## 五、如果你现在只想快速判断该用哪个文件

### 情况 1：我只想复现当前 3 物体 `pick` 正式结果

用：

- `config/level1_pick_stratified_all_obj.yaml`

### 情况 2：我只想复现当前 5 物体 `pick` 正式结果

用：

- `config/level1_pick_stratified_all_obj_5obj.yaml`

### 情况 3：我只想复现当前 `place` 主线结果

用：

- `config/level1_place_noise_ring_v2.yaml`

### 情况 4：我只想先确认 `place` 环境和链路有没有跑通

用：

- `config/level1_place_noise_ring_v2_1ep.yaml`

## 六、当前推荐运行方式

远程环境：

```bash
cd /home/ubuntu/teasings_projects/Labutopia
conda activate labutopia_51
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

`pick` 3 物体正式版：

```bash
python main.py --config-name level1_pick_stratified_all_obj
```

`pick` 5 物体正式版：

```bash
python main.py --config-name level1_pick_stratified_all_obj_5obj
```

`place` 当前冻结版：

```bash
python main.py --config-name level1_place_noise_ring_v2
```

`place` 冒烟：

```bash
python main.py --config-name level1_place_noise_ring_v2_1ep
```

## 七、当前阶段结束语

如果之后你再次忘了配置关系，只记住下面这五句就够了：

1. `pick` 这条线已经完成，3 物体和 5 物体都有冻结版。
2. 3 物体正式版是 `config/level1_pick_stratified_all_obj.yaml`。
3. 5 物体正式版是 `config/level1_pick_stratified_all_obj_5obj.yaml`，最终 `noise_scale = 2.35`。
4. `place` 当前冻结版是 `config/level1_place_noise_ring_v2.yaml`，当前结果约 `29% success`。
5. 下一步应该转去 `pour`，而不是继续在旧 `pick/place` 配置里翻来翻去。

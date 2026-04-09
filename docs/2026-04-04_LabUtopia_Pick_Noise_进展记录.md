# LabUtopia Pick 进展速查

- 更新时间：2026-04-09
- 适用范围：`pick` 这一条线
- 用途：快速回答三个问题
  - 你在原项目基础上到底做了什么
  - 当前应该看哪个配置文件
  - 哪些旧配置已经不该继续用了

## 一、先看结论

`pick` 这条线已经阶段性完成，目前有两个稳定结果：

| 版本 | 配置文件 | 结果 | 当前结论 |
| --- | --- | --- | --- |
| 3 物体正式版 | `config/level1_pick_stratified_all_obj.yaml` | `112/450 written`，`89 skipped`，`24.9% success` | 已达标，固定使用 |
| 5 物体泛化版 | `config/level1_pick_stratified_all_obj_5obj.yaml` | `179/750 written`，`107 skipped`，`23.9% success` | 已达标，固定使用 |

当前阶段不要再继续围绕 `pick` 做新的参数搜索，除非学长提出新的目标。

如果你现在只想找“最后那个成功并且扩到 5 个物体的版本”，直接看：

- `config/level1_pick_stratified_all_obj_5obj.yaml`
- 当前固定值：`noise_scale: 2.35`

如果你想找“先达标、后续所有泛化实验都基于它扩展”的正式主配置，直接看：

- `config/level1_pick_stratified_all_obj.yaml`
- 当前固定值：`noise_scale: 2.45`

## 二、你在原项目基础上实际做了什么

这部分只保留真正重要的增量，不再按时间线展开。

### 1. 把 `pick` 从“能跑”推进到“能稳定做采集”

- 重新确认了远程真正该用的仓库和环境：
  - 仓库：`/home/ubuntu/teasings_projects/Labutopia`
  - 环境：`labutopia_51`
- 明确了当前问题不再是“系统能不能启动”，而是“如何控制采集分布和成功率”。

### 2. 把关键抓取参数从 controller 硬编码提到 YAML

涉及：

- `controllers/pick_controller.py`
- `config/level1_pick.yaml`
- `config/level1_pick_noise_universal.yaml`

核心变化：

- `pre_offset_x`
- `pre_offset_z`
- `after_offset_z`
- `end_effector_euler_deg`

现在这些参数可以直接在配置里调，不需要每次改 controller。

### 3. 在任务层补齐了分层采集逻辑

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

### 4. 把正式配置整理成更容易维护的版本

涉及：

- `config/level1_pick_stratified.yaml`
- `config/level1_pick_stratified_all_obj.yaml`
- `config/level1_pick_noise_uniform_all_obj_pos10_obj150.yaml`

核心变化：

- 正式主配置尽量自包含
- `_1ep` 只做冒烟测试
- `debug` / `schedule_debug` 只做验证
- 不再需要沿着很长的 Hydra 继承链来回追参数

### 5. 拿到了两版正式结果

- `3` 物体正式版已经达标
- `5` 物体泛化版也已经达标
- `skipped` 较多这件事已和学长确认，目前视为正常现象，不是主要优化目标

## 三、当前配置速查表

### 1. 现在真正应该记住的配置

| 配置文件 | 用途 | 是否达标 | 是否继续改 |
| --- | --- | --- | --- |
| `config/level1_pick_stratified_all_obj.yaml` | 3 物体正式主配置 | 是 | 先不要动 |
| `config/level1_pick_stratified_all_obj_5obj.yaml` | 5 物体泛化正式配置 | 是 | 先不要动 |
| `config/level1_pick_stratified_all_obj_1ep.yaml` | 3 物体正式版冒烟测试 | 不看成功率，只看能否跑通 | 可以继续用 |
| `config/level1_pick_stratified_all_obj_debug.yaml` | 小阈值调度快速验证 | 仅用于 debug | 一般不用 |
| `config/level1_pick_stratified_all_obj_schedule_debug.yaml` | 关闭噪声的纯调度验证 | 仅用于 debug | 一般不用 |
| `config/level1_pick_stratified.yaml` | 单物体 stratified 参考配置 | 参考用 | 一般不作为当前主线 |

### 2. 现在已经不该再作为主线继续推进的配置

| 配置文件 | 角色 | 当前定位 |
| --- | --- | --- |
| `config/level1_pick_noise_universal.yaml` | 早期 noise 主线 | 历史中间阶段 |
| `config/level1_pick_noise_uniform_all_obj_pos10_obj150.yaml` | 旧的 uniform 多物体方案 | 历史过渡方案 |
| `config/level1_pick_noise_uniform_all_obj_pos10_obj150_1ep.yaml` | 上面那版的 1ep | 历史过渡方案 |

一句话概括就是：

- 现在看 `stratified_all_obj`
- 不再回到 `noise_universal`
- 更不要再从 `uniform_all_obj_pos10_obj150` 重新起步

## 四、最终结果速记

### 1. 3 物体正式版

- 配置：`config/level1_pick_stratified_all_obj.yaml`
- 结果：`112/450 written`
- `skipped`：`89`
- 成功率：`24.9%`
- 当前固定参数：`noise_scale: 2.45`

这版是“当前正式主线”。

### 2. 5 物体泛化版

- 配置：`config/level1_pick_stratified_all_obj_5obj.yaml`
- 物体集合：
  - `conical_bottle02`
  - `conical_bottle03`
  - `conical_bottle04`
  - `beaker2`
  - `graduated_cylinder_03`
- 第一轮结果：`142/750 written`，`105 skipped`，`18.9% success`
- 第二轮结果：`179/750 written`，`107 skipped`，`23.9% success`
- 最终固定参数：`noise_scale: 2.35`

这版是“扩到 5 个物体后最终达标的版本”。

## 五、如果你现在只想快速判断该用哪个文件

### 情况 1：我只想复现当前 3 物体正式结果

用：

- `config/level1_pick_stratified_all_obj.yaml`

### 情况 2：我只想复现当前 5 物体正式结果

用：

- `config/level1_pick_stratified_all_obj_5obj.yaml`

### 情况 3：我只想先看环境和逻辑有没有跑通

用：

- `config/level1_pick_stratified_all_obj_1ep.yaml`

### 情况 4：我怀疑是调度逻辑有 bug

先用：

- `config/level1_pick_stratified_all_obj_schedule_debug.yaml`

不要一上来就看正式强噪声结果，因为 `skipped` 会打散 written block。

## 六、当前推荐运行方式

远程环境：

```bash
cd /home/ubuntu/teasings_projects/Labutopia
conda activate labutopia_51
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

3 物体正式版：

```bash
python main.py --config-name level1_pick_stratified_all_obj
```

5 物体正式版：

```bash
python main.py --config-name level1_pick_stratified_all_obj_5obj
```

1ep 冒烟：

```bash
python main.py --config-name level1_pick_stratified_all_obj_1ep
```

## 七、当前阶段结束语

如果之后你再次忘了配置关系，只记住下面这四句就够了：

1. `pick` 这条线已经完成，不是当前主要问题。
2. 3 物体正式版是 `config/level1_pick_stratified_all_obj.yaml`。
3. 5 物体正式版是 `config/level1_pick_stratified_all_obj_5obj.yaml`，最终 `noise_scale = 2.35`。
4. 下一步应该转去新任务，例如 `place`，而不是继续在旧 `pick` 配置里翻来翻去。

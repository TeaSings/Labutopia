# LabUtopia Pick Noise 进展记录

- 日期：2026-04-04
- 记录范围：`level1_pick`、`level1_pick_noise_universal` 及其后续配置改造
- 用途：记录最近的改动、当前已经完成的任务

## 一、当前阶段的核心目标

最近这一阶段的工作重点已经从“单纯把 pick 跑通”转成了“围绕噪声采集协议做调整”。当前目标主要有三点：

- 保持 `level1_pick` 基线可稳定运行
- 将带噪声采集的成功率控制在 `20% ~ 40%`
- 将噪声采样方式从 `edge_bias` 改为 `uniform`
- 逐步把采集协议改成 `stratified` 方式：
  - 同一物体在同一 pose 下连续采多局
  - 达到设定阈值后再换 pose
  - 在多物体版本中，再按固定 episode 数切换物体类别

## 二、最近已经确认并完成的事项

### 1. 运行仓库与环境基线已经重新确认

已确认当前远程实际应使用的仓库为：

- `/home/ubuntu/teasings_projects/Labutopia`

当前远程运行基线为：

- `conda activate labutopia_51`
- `OMNI_KIT_ACCEPT_EULA=YES`
- `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`

这一套启动命令能够确保整个项目正确运行起来。

另外，本地仓库当前已经恢复为干净工作区，可以重新走更正常的同步流程：

- 本地建分支、提交、`git push`
- 远程 `git fetch / git checkout / git pull`

### 2. `level1_pick` 基础链路已经跑通

此前已经验证：

- `level1_pick` 的 clean baseline 可以稳定运行
- 采集链路能够正常写出 episode 数据
- `episode_*.h5` 与元数据写入链路不是当前主要阻塞点

这说明当前问题已经不再是“系统是否能跑起来”，而是“在噪声条件下如何组织采样与控制数据分布”。

### 3. `level1_pick_noise_universal` 已得到代表性结果

最近两轮有代表性的结果如下：

| 轮次 | 成功写入 | skipped | 成功率 |
| --- | --- | --- | --- |
| 第 1 轮 | 26 / 50 | 4 | 52.0% |
| 第 2 轮 | 23 / 50 | 11 | 46.0% |

从这两轮结果可以确认：

- 带噪声采集已经能稳定执行到完整统计输出
- 当前成功率仍然偏高于学长后续提出的 `20% ~ 40%` 目标区间

## 三、最近已经做出的代码与配置改动

### 1. 将 pick 默认参数从硬编码提升到 YAML

涉及文件：

- [controllers/pick_controller.py](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/controllers/pick_controller.py)
- [config/level1_pick_noise_universal.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_noise_universal.yaml)

本次改动的核心是把原本写死在 controller 里的默认抓取参数改为从 `cfg.pick` 中读取，包括：

- `pre_offset_x`
- `pre_offset_z`
- `after_offset_z`
- `end_effector_euler_deg`

这样做的意义是：

- 后续调参不需要反复修改 controller 代码
- 实验参数可以直接通过 YAML 管理
- 便于后续记录不同实验配置并复现

### 2. 为任务层增加“按 episode 切换”的能力

涉及文件：

- [tasks/base_task.py](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/tasks/base_task.py)
- [main.py](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/main.py)

新增支持的任务级参数包括：

- `task.object_switch_metric`
- `task.object_switch_interval`
- `task.position_switch_interval`

这部分改动解决了两个原有问题：

1. 原逻辑中，物体切换主要按照 success 计数，不适合“每采集若干次就切换”的新需求  
2. 原逻辑中，物体位置每个 episode 都会重新随机，不适合“固定 10 次再换位置”的需求

改完后可以支持：

- 按 episode 数而不是 success 数切换物体
- 在连续若干个 episode 中保持同一个物体位置
- 到达指定次数后再重新采样物体位置

### 3. 新建独立配置文件，避免继续在旧配置上反复继承叠加

新增文件：

- [config/level1_pick_noise_uniform_all_obj_pos10_obj150.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_noise_uniform_all_obj_pos10_obj150.yaml)
- [config/level1_pick_noise_uniform_all_obj_pos10_obj150_1ep.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_noise_uniform_all_obj_pos10_obj150_1ep.yaml)

这套新配置的目标是把当前学长提出的要求集中写在一个文件里，而不是继续在一批历史配置之间来回覆盖。

这份配置目前明确表达了：

- 使用 `uniform` 噪声，而不是 `edge_bias`
- `noise_scale` 作为后续控制成功率区间的主要调节参数
- 每 `10` 个 episode 切换一次物体位置
- 每 `150` 个 episode 切换一次物体类别
- 每个物体自己的 `position_range` 也集中定义在同一个配置里

### 4. 根据学长新给出的 stratified 配置，补齐了 pose 分层采集逻辑

涉及文件：

- [config/level1_pick_stratified.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_stratified.yaml)
- [tasks/base_task.py](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/tasks/base_task.py)

后来学长给出了一版新的参考配置 `level1_pick_stratified.yaml`。这份配置明确了一个新的采集方向：

- 不再简单按固定 episode 数频繁换 pose
- 而是在同一物体、同一 pose 下，连续积累成功/失败与纠错标签
- 优先按 `successes_per_pose` 控制何时换 pose

在此基础上，本次把原先还没有真正落地的字段补成了可执行逻辑，新增支持：

- `task.stratified_collection`
- `task.successes_per_pose`
- `task.episodes_per_pose`

补齐后的效果是：

- 当启用 `stratified_collection` 时，可以按“累计成功数”或“累计 episode 数”控制 pose 切换
- pose 切换与物体切换被拆成两层逻辑，便于之后单独调整
- 这样就能真正实现“固定 pose 多局采集，再换 pose”的分层采集过程

### 5. 新增多物体 stratified 配置

新增文件：

- [config/level1_pick_stratified_all_obj.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_stratified_all_obj.yaml)
- [config/level1_pick_stratified_all_obj_1ep.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_stratified_all_obj_1ep.yaml)

这套配置是在 `level1_pick_stratified.yaml` 基础上扩展出来的多物体版本，当前语义是：

- 同一物体、同一 pose 下累计 `10` 次成功后换 pose
- 同一物体累计 `150` 个 episode 后切换到下一个物体
- 噪声仍采用更激进的 `uniform` 设定，用于把成功率压到目标区间

这比之前的 `uniform_all_obj_pos10_obj150` 方案更贴近学长现在的意图，因为它把“pose 分层采集”和“object 轮换”真正分开了。

## 四、当前已经达到的阶段性结果

到目前为止，可以明确认为已经完成的阶段性任务有：

### 1. `pick` 任务的最小采集链路已具备继续实验的条件

- clean baseline 能跑
- noise 配置能跑
- 采集统计能出
- 问题已从“无法运行”转向“如何设计合适的数据采集分布”

### 2. 配置驱动式调参路径已经打通

- 以前很多参数只能在 controller 里改
- 现在已经开始转成 YAML 驱动
- 后续更适合做成“一个实验对应一个配置文件”的方式

### 3. 新的采集协议已经形成了明确实现方案

也就是：

- `uniform` 噪声
- 成功率目标 `20% ~ 40%`
- 单物体主线已转向 `stratified` 分层采集
- 多物体版本可在 stratified 基础上继续按 episode 数切换物体

### 4. 当前本地代码已经达到可同步、可验证状态

本次涉及 `stratified` 的核心逻辑已经补到本地代码中，并且已经通过静态校验：

```bash
python3 -m py_compile tasks/base_task.py main.py controllers/pick_controller.py
```

这意味着当前状态已经不是“只停留在讨论和设计”，而是可以进入：

- 本地提交分支
- 推送到 GitHub
- 远程机器直接 `git pull`
- 再做 1ep 冒烟测试和正式运行


## 五、当前仍待进一步验证的部分

以下内容已经设计或写入本地代码，但还需要在远程机器上继续验证：

- 新配置 `level1_pick_stratified_all_obj` 跑起来后，是否真的实现：
  - 同一 pose 下累计 `10` 次成功后再换 pose
  - 同一物体累计 `150` 个 episode 后再换到下一个物体
- 在 `uniform` 噪声下，成功率是否能通过调节 `noise_scale` 稳定落入 `20% ~ 40%`

因此，当前状态更准确地说是：

- 路线已明确
- 代码结构已准备
- 配置方案已独立出来
- stratified 主线已经在本地代码层面接通
- 调度逻辑已经通过小规模 debug run 验证
- 但完整远程正式实验结果还需要继续补充

## 六、2026-04-04 最新补充：调度 debug 已验证通过

本轮又新增了一步非常关键的验证：不是直接在正式强噪声配置上继续猜调度是否出错，而是专门新建了一个“只验证调度”的 debug 配置。

新增文件：

- [config/level1_pick_stratified_all_obj_debug.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_stratified_all_obj_debug.yaml)
- [config/level1_pick_stratified_all_obj_schedule_debug.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_stratified_all_obj_schedule_debug.yaml)
- [temp.py](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/temp.py)

其中：

- `level1_pick_stratified_all_obj_debug.yaml`
  - 用较小阈值验证 object / pose 切换
  - `object_switch_interval: 6`
  - `episodes_per_pose: 3`
- `level1_pick_stratified_all_obj_schedule_debug.yaml`
  - 在上面基础上进一步关闭 `noise`
  - 关闭 `tipped_detection`
  - 目的是尽量减少 `skipped`，让调度顺序以最干净的形式呈现出来

最近这轮 `schedule_debug` 的结果已经说明调度逻辑是正确的：

- `written: 18`
- `Object blocks` 为：
  - `6 / 6 / 6`
- `Pose blocks` 为：
  - `3 / 3`
- `schedule rows` 中：
  - `obj_counter` 按 `0..5` 递增
  - `pose_counter` 按 `0..2` 递增
  - `resampled=True` 只出现在真正换 pose 的 reset 上

这说明当前代码已经可以正确实现：

- 按 episode 每 `3` 局切换 pose
- 按 episode 每 `6` 局切换物体

因此当前结论是：

- `BaseTask` 这条调度主线已经通过验证
- 不需要继续优先修调度代码
- 之后再看到正式强噪声 run 中的 `Object blocks` 不整齐，不能直接判定为调度 bug

这里还需要特别说明一个已经澄清的误区：

- 正式强噪声配置下，会出现一部分 `skipped` episode
- 这些 `skipped` 不会写入 `episode.jsonl`
- 但调度内部计数仍然可能已经前进

因此正式 run 的 written 数据会被“打散”，也就是：

- 底层真实调度可能是连续的
- 但 `episode.jsonl` 里因为缺失了被跳过的 episode，看起来会像块长度不整齐、甚至像是物体切换得更碎

所以正式配置不应该再用“written block 是否完美整齐”作为唯一判断标准。

## 七、当前最合理的下一步

在调度 debug 已通过的前提下，接下来最合理的动作就是直接回到正式配置运行：

- [config/level1_pick_stratified_all_obj.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_stratified_all_obj.yaml)

这一步的目标不再是验证调度逻辑，而是观察正式采集的整体统计结果，重点看：

- 最终成功率是否稳定在 `20% ~ 40%`
- 是否存在异常高的 `skipped`
- 是否出现新的系统性异常

按照当前代码逻辑，停止条件是：

- 每个有效物体 `150` 个 written episode
- 当前有效物体数大概率为 `3`

所以正式 run 的 written 总量大约是：

- `150 × 3 = 450`

如果按“每条约 `20` 秒”粗估：

- 不考虑 `skipped` 时，约 `450 × 20s = 9000s ≈ 2.5 小时`

但由于 collect 模式下停止判断使用的是 `written episode` 数，而不是总尝试次数，所以如果正式 run 中仍有较多 `skipped`，真实墙钟时间会比 `2.5` 小时更长，实操上更合理的预估大约是：

- `2.5 ~ 3.2 小时`

如果后续第 4 个物体也被纳入有效对象，总时长还会进一步增加。

## 八、2026-04-05 最新补充：正式 stratified 多物体结果已达到目标

在远程机器上正式运行：

- [config/level1_pick_stratified_all_obj.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_stratified_all_obj.yaml)

最终结果为：

- `Success = 112/450 written`
- `89 skipped`
- `24.9% success`
- 总运行时长日志显示约 `17713s`，约合 `4.9` 小时

这组结果的意义比较明确：

- 成功率已经稳定落在学长要求的 `20% ~ 40%` 区间内
- 而且不是贴边，而是落在区间内部，当前可以作为这一阶段的正式结果
- 结合此前已经通过的 `schedule_debug`，可以认为目前这条 `stratified + uniform noise + all_obj` 主线已经跑通

另外，学长已经明确反馈：

- 当前 `skipped` 较多属于正常现象

因此目前不再需要继续围绕 `skipped` 做额外优化，也不建议继续微调成功率。当前更合理的做法是：

- 冻结这版正式配置
- 将结果补入汇报材料
- 只在学长提出新目标后再进入下一轮参数调整

基于这个结果，当前阶段可以认为已经完成的新增事项是：

- `level1_pick_stratified_all_obj` 正式配置已经在远程跑出完整结果
- 成功率控制目标已达成
- 调度逻辑与正式统计结果两条线都已经闭环

当前下一步工作的重心，不再是继续改控制参数，而是：

- 整理结果与汇报材料
- 同时把当前正式 YAML 配置整理得更清晰，减少多层继承带来的维护成本

本轮还补做了一项配置整理工作：

- 已将 [config/level1_pick_stratified_all_obj.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_stratified_all_obj.yaml) 改写为自包含版本
- 当前正式运行所依赖的关键参数已经集中写回同一个文件，包括：
  - `task_type / controller_type / mode / usd_path`
  - `task` 中的 `workspace_range / stratified_collection / successes_per_pose / object_switch_interval / obj_paths`
  - `cameras / robot / collector`
  - `pick` 中的默认抓取参数、成功判定参数、correction 相关参数
  - `noise` 中的正式噪声范围、`noise_scale`、`failure_bias_ratio`、`noise_distribution`
- 这样后续如果继续维护或向学长解释当前正式配置，不需要再沿着
  `level1_pick_noise_universal -> level1_pick_stratified -> level1_pick_stratified_all_obj`
  这条链来回追参数
- `level1_pick_stratified_all_obj_1ep.yaml` 与 debug 配置仍然可以继续继承这份主配置，保持调试入口不变

## 九、2026-04-09 最新补充：5 物体泛化版已跑通并固定 noise_scale

在前一阶段 `3` 物体正式配置已经达标之后，又进一步尝试了 5 物体泛化版本：

- [config/level1_pick_stratified_all_obj_5obj.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_stratified_all_obj_5obj.yaml)

这版在原有 `3` 个物体基础上，新增了：

- `conical_bottle03`
- `conical_bottle04`

形成 5 物体集合：

- `conical_bottle02`
- `conical_bottle03`
- `conical_bottle04`
- `beaker2`
- `graduated_cylinder_03`

先前使用较强噪声时，5 物体版跑出的结果是：

- `142/750 written`
- `105 skipped`
- `18.9% success`

这个结果说明：

- 物体扩展后整体难度确实上升了
- 但并不是彻底失控，而是只比目标区间下沿略低

随后仅对噪声强度做了很小的调整：

- `noise_scale: 2.45 -> 2.35`

在保持其余参数不变的情况下，再次运行 5 物体正式配置，结果为：

- `179/750 written`
- `107 skipped`
- `23.9% success`
- 总运行时长日志显示约 `28426s`

这组结果意味着：

- 5 物体版已经重新回到目标区间 `20% ~ 40%`
- 而且回到区间内部，不需要再继续做更激进的参数调整
- 当前最合理的做法是将 5 物体配置固定在 `noise_scale: 2.35`

因此，本轮之后可以认为：

- `3` 物体正式主线已经达标
- `5` 物体泛化版也已经达标
- 当前不需要继续围绕这条 5 物体配置做新一轮参数搜索，除非学长提出新的目标

## 十、当前自定义配置文件用途速查

为了避免后续继续在一批名字接近、继承关系复杂的 YAML 之间来回跳转，当前这条自定义配置链可以按下面理解：

### 1. 当前正式主配置

- [config/level1_pick_stratified_all_obj.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_stratified_all_obj.yaml)
  - 用途：当前正式多物体 `stratified` 采集配置
  - 特点：
    - 自包含
    - `successes_per_pose: 10`
    - `object_switch_interval: 150`
    - 使用 `uniform` 噪声
  - 当前已验证正式结果：
    - `112/450 written`
    - `89 skipped`
    - `24.9% success`

### 2. 正式配置的冒烟测试版本

- [config/level1_pick_stratified_all_obj_1ep.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_stratified_all_obj_1ep.yaml)
  - 用途：正式多物体配置的 `1ep` 冒烟测试
  - 适用场景：
    - 刚改完代码或配置后先确认能否启动
    - 不用于正式统计

### 3. 调度逻辑快速验证版

- [config/level1_pick_stratified_all_obj_debug.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_stratified_all_obj_debug.yaml)
  - 用途：快速检查 object / pose 切换节奏是否生效
  - 特点：
    - 直接继承正式主配置
    - 将 `object_switch_interval` 缩小到 `6`
    - 将 `episodes_per_pose` 设为 `3`
    - 保留 `noise`，因此 written 结果仍可能被 `skipped` 打散
  - 适用场景：
    - 调度逻辑初步联调
    - 不是最干净的验证版

### 4. 调度逻辑纯验证版

- [config/level1_pick_stratified_all_obj_schedule_debug.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_stratified_all_obj_schedule_debug.yaml)
  - 用途：最干净地验证调度逻辑
  - 特点：
    - 直接继承正式主配置
    - `object_switch_interval: 6`
    - `episodes_per_pose: 3`
    - `noise.enabled: false`
    - `pick.tipped_detection_enabled: false`
  - 这版已经验证通过：
    - `Object blocks = 6 / 6 / 6`
    - `Pose blocks = 3 / 3`
  - 适用场景：
    - 当怀疑调度逻辑有问题时，优先跑这版

### 5. 单物体 stratified 主配置

- [config/level1_pick_stratified.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_stratified.yaml)
  - 用途：单物体 `stratified` 采集主配置
  - 特点：
    - 自包含
    - 保留完整 `pick / noise / task / collector` 参数
    - 适合后续若要回到“单物体分层采集”主线时使用

### 6. 旧方案保留配置

- [config/level1_pick_noise_uniform_all_obj_pos10_obj150.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_noise_uniform_all_obj_pos10_obj150.yaml)
  - 用途：更早一版“uniform + 每 10 个 episode 换位置 + 每 150 个 episode 换物体”的历史方案
  - 特点：
    - 已改为自包含
    - 不再是当前主线
    - 与现在的 `stratified` 方案不同，它是按 episode 固定换位置，不是按 `successes_per_pose`
  - 适用场景：
    - 回顾早期实验
    - 与当前 `stratified` 方案做对照

- [config/level1_pick_noise_uniform_all_obj_pos10_obj150_1ep.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_noise_uniform_all_obj_pos10_obj150_1ep.yaml)
  - 用途：上面这套旧方案的 `1ep` 冒烟测试版本

### 7. 5 物体泛化版

- [config/level1_pick_stratified_all_obj_5obj.yaml](/Users/chensanya/Library/CloudStorage/OneDrive-bupt.cn/paper/LabUtopia/config/level1_pick_stratified_all_obj_5obj.yaml)
  - 用途：在当前正式多物体主配置基础上，增加 `conical_bottle03 / conical_bottle04`，测试 pick 泛化能力
  - 特点：
    - 直接继承正式主配置
    - 5 物体集合：`conical_bottle02 / 03 / 04 + beaker2 + graduated_cylinder_03`
    - 当前固定 `noise_scale: 2.35`
  - 当前已验证结果：
    - `179/750 written`
    - `107 skipped`
    - `23.9% success`
  - 结论：
    - 已回到目标区间
    - 当前可以作为 5 物体泛化版的稳定结果

### 8. 当前配置链整理后的原则

目前这条自定义配置链已经整理成下面这个规则：

- 正式主配置尽量自包含，不再挂在 `level1_pick_noise_universal` 上
- `_1ep` 配置只负责把 `max_episodes` 缩到 `1`
- `debug` 配置只负责改动验证逻辑需要的最小字段
- 如果以后新增配置，优先基于当前正式主配置扩展，而不是再回到旧的 `level1_pick_noise_universal` 继承链

## 十一、后续建议记录方式

为了便于下周五汇报，建议后续继续按下面的方式往这份文档补充：

### 每次实验至少记录 4 项

- 使用的配置文件名
- 关键改动参数
- `Episode Stats` 最终结果
- 本轮结论

### 建议按“已验证 / 待验证”区分记录

这样在汇报时可以避免把“已经跑出来的结果”和“已经设计好但还没在远程完整验证的方案”混在一起。

## 十二、下周汇报时可以直接讲的重点

- 目前 `pick` 的 clean baseline 已经稳定，说明系统链路本身不是主要问题
- 带噪声采集已经能够稳定输出结果，当前主要矛盾转为噪声分布与采集协议设计
- 本周最大的变化不是单个参数调优，而是把调参思路从“提高成功率”转成“控制数据难度分布”
- 为了适应新的采集要求，已经开始把实验入口统一到独立 YAML 配置中，并对任务层调度逻辑进行了小范围扩展
- 最近又根据学长新给出的 `stratified` 配置，进一步把 pose 分层采集逻辑补到代码里
- 之后又用单独的 schedule debug 配置验证了调度逻辑本身是正确的
- 现在 `level1_pick_stratified_all_obj` 已经在远程跑出正式结果：`112/450 written`、`89 skipped`、`24.9% success`
- 这说明当前主线配置已经达到“成功率控制在 `20% ~ 40%`”的目标
- 随后又扩展到了 5 物体泛化版，并通过把 `noise_scale` 轻微调到 `2.35`，得到 `179/750 written`、`107 skipped`、`23.9% success`

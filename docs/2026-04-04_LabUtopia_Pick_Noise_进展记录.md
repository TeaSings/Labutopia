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

- `stratified` 逻辑在远程机器上的真实行为是否完全符合预期
- 新配置 `level1_pick_stratified_all_obj` 跑起来后，是否真的实现：
  - 同一 pose 下累计 `10` 次成功后再换 pose
  - 同一物体累计 `150` 个 episode 后再换到下一个物体
- 在 `uniform` 噪声下，成功率是否能通过调节 `noise_scale` 稳定落入 `20% ~ 40%`

因此，当前状态更准确地说是：

- 路线已明确
- 代码结构已准备
- 配置方案已独立出来
- stratified 主线已经在本地代码层面接通
- 但完整远程实验结果还需要继续补充

## 六、后续建议记录方式

为了便于下周五汇报，建议后续继续按下面的方式往这份文档补充：

### 每次实验至少记录 4 项

- 使用的配置文件名
- 关键改动参数
- `Episode Stats` 最终结果
- 本轮结论

### 建议按“已验证 / 待验证”区分记录

这样在汇报时可以避免把“已经跑出来的结果”和“已经设计好但还没在远程完整验证的方案”混在一起。

## 七、下周汇报时可以直接讲的重点

- 目前 `pick` 的 clean baseline 已经稳定，说明系统链路本身不是主要问题
- 带噪声采集已经能够稳定输出结果，当前主要矛盾转为噪声分布与采集协议设计
- 本周最大的变化不是单个参数调优，而是把调参思路从“提高成功率”转成“控制数据难度分布”
- 为了适应新的采集要求，已经开始把实验入口统一到独立 YAML 配置中，并对任务层调度逻辑进行了小范围扩展
- 最近又根据学长新给出的 `stratified` 配置，进一步把 pose 分层采集逻辑补到代码里
- 下一步主要工作将是把这套 stratified 多物体配置同步到远程机器，并验证采集节奏与成功率区间

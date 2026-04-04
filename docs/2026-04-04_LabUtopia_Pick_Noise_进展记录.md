# LabUtopia Pick Noise 进展记录

- 日期：2026-04-04
- 记录范围：`level1_pick`、`level1_pick_noise_universal` 及其后续配置改造
- 用途：记录最近的改动、当前已经完成的任务

## 一、当前阶段的核心目标

最近这一阶段的工作重点已经从“单纯把 pick 跑通”转成了“围绕噪声采集协议做调整”。当前目标主要有三点：

- 保持 `level1_pick` 基线可稳定运行
- 将带噪声采集的成功率控制在 `20% ~ 40%`
- 将噪声采样方式从 `edge_bias` 改为 `uniform`，并支持按固定 episode 数切换物体位置和物体类别

## 二、最近已经确认并完成的事项

### 1. 运行仓库与环境基线已经重新确认

已确认当前远程实际应使用的仓库为：

- `/home/ubuntu/teasings_projects/Labutopia`

当前远程运行基线为：

- `conda activate labutopia_51`
- `OMNI_KIT_ACCEPT_EULA=YES`
- `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`

这一套启动命令能够确保整个项目正确运行起来。

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
- 每 `10` 次换位置
- 每 `150` 次换物体


## 五、当前仍待进一步验证的部分

以下内容已经设计或写入本地代码，但还需要在远程机器上继续验证：

- 新增的 episode 级切换逻辑是否完全符合预期
- 新配置 `level1_pick_noise_uniform_all_obj_pos10_obj150` 跑起来后，是否真的实现：
  - 位置每 10 次切换
  - 物体每 150 次切换
- 在 `uniform` 噪声下，成功率是否能通过调节 `noise_scale` 稳定落入 `20% ~ 40%`

因此，当前状态更准确地说是：

- 路线已明确
- 代码结构已准备
- 配置方案已独立出来
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
- 下一步主要工作将是基于新配置继续在远程机器上验证采集节奏与成功率区间

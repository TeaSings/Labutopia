# Level1 原子动作冻结参数说明

本文只整理 Level1 原子动作。冻结配置来源以 [调参进展记录2026-4-15.md](./调参进展记录2026-4-15.md) 中“当前冻结配置”表为准，并按 Hydra `defaults` 继承链展开后解释最终生效参数。

## 冻结配置索引

| 原子动作 | 冻结配置 | 备注 |
| --- | --- | --- |
| pick | `config/level1_pick_stratified_all_obj.yaml` | 3 物体正式版；另有 `config/level1_pick_stratified_all_obj_5obj.yaml` 作为 5 物体泛化验证版 |
| place | `config/level1_place_noise_ring_v2.yaml` | 使用环形放置噪声 |
| pour | `config/level1_pour_noise_multi_obj_v2.yaml` | 多物体 pour 冻结版 |
| open_door | `config/level1_open_door_noise_multi_obj_v2.yaml` | 多物体开门冻结版 |
| open_drawer | `config/level1_open_drawer_noise_multi_obj_v5.yaml` | 多物体开抽屉冻结版 |
| close_door | `config/level1_close_door_noise_v3.yaml` | 关门冻结版 |
| close_drawer | `config/level1_close_drawer_noise_v14.yaml` | 关抽屉冻结版 |
| press | `config/level1_press_noise_v5.yaml` | 按钮冻结版 |
| shake | `config/level1_shake_noise_v2.yaml` | 摇晃冻结版 |
| stir | `config/level1_stir_noise_v4.yaml` | 搅拌冻结版 |

## 通用参数含义

| 参数 | 含义 |
| --- | --- |
| `task.max_steps` | 单个 episode 最多仿真步数，超过后重置。 |
| `task.camera_warmup_frames` | 采集前给相机和物理状态稳定的预热帧数。 |
| `task.obj_paths` | 当前任务参与采样的物体、物体路径、把手路径、位置范围等。 |
| `object_switch_metric: episode` | 多物体任务按写入 episode 数切换物体，而不是按尝试次数切换。 |
| `object_switch_interval` | 每个物体连续采集的 episode 数。 |
| `events_dt` | primitive 内部各阶段的控制时间步或阶段推进速度，通常影响动作快慢和接触稳定性。 |
| `position_threshold` | 末端或目标位置认为“到达”的距离阈值。 |
| `end_effector_euler_deg` | 末端执行器姿态，单位为角度。 |
| `noise.enabled` | 是否启用每个 episode 的随机扰动。 |
| `noise.noise_distribution` | 默认噪声采样分布。`uniform` 为均匀采样，`edge_bias` 更偏向区间两端，`min_bias` 更偏向下界，`max_bias` 更偏向上界。 |
| `noise.failure_bias_ratio` | 进入失败倾向噪声采样的概率。越接近 1，越频繁采到更强、更容易失败的扰动。 |
| `noise.noise_scale` | 对噪声区间做放缩的倍率，常用于整体增强或减弱噪声。 |
| `*_distribution` | 某一个参数单独覆盖默认噪声分布。 |

除特别说明外，噪声区间表示加到 clean 参数上的 delta。`*_radius_range`、`*_angle_deg` 这类参数表示直接采样半径或角度范围。

## Pick

冻结配置为 `config/level1_pick_stratified_all_obj.yaml`。进展文档同时记录了 `config/level1_pick_stratified_all_obj_5obj.yaml`，用于验证扩展到 5 个物体后的泛化结果。

| 调整项 | 冻结值 | 含义 |
| --- | --- | --- |
| `task.stratified_collection` | `true` | 启用分层采集，固定物体姿态采满一定成功数后再换姿态。 |
| `task.successes_per_pose` | `10` | 每个物体姿态需要写入 10 个成功样本。 |
| `task.object_switch_interval` | `150` | 每个物体采 150 个写入 episode 后切换。 |
| `task.obj_paths` | 3 物体正式版含 `conical_bottle02`、`beaker2`、`graduated_cylinder_03`、`volume_flask` | 控制采样物体与位置范围。 |
| `pick.pre_offset_x` | `0.05` | 抓取前沿 x 方向的预接近偏移。 |
| `pick.pre_offset_z` | `0.12` | 抓取前在物体上方的高度偏移。 |
| `pick.after_offset_z` | `0.25` | 抓住后向上抬起的高度。 |
| `pick.end_effector_euler_deg` | `[0, 90, 25]` | 抓取时的夹爪姿态。 |
| `tipped_detection_enabled` | `true` | 启用物体倾倒检测，避免把倾倒物体样本写入成功数据。 |
| `tipped_max_tilt_deg` | `20` | 物体倾角超过该值认为倾倒。 |
| `required_success_steps` | `15` | 成功条件需要连续保持的帧数。 |
| `lift_required_success_steps` | `10` | 抬起成功需要连续保持的帧数。 |
| `success_pool_max` | `500` | 保留最多 500 个成功参数用于修正标签参考。 |
| `correction_steps_count` | `3` | 生成修正轨迹时使用的修正步数。 |
| `correction_alpha` | `0.5` | 当前参数向成功参考参数靠拢的比例。 |
| `p_ref_method` | `mean` | 使用成功池均值作为参考参数。 |

| 噪声项 | 冻结值 | 含义 |
| --- | --- | --- |
| `noise.pre_offset_x` | `[-0.05, 0.05]` | 扰动横向接近距离，影响是否对准物体。 |
| `noise.pre_offset_z` | `[-0.05, 0.05]` | 扰动下探前高度，影响是否过高或碰撞。 |
| `noise.after_offset_z` | `[-0.05, 0.05]` | 扰动抓取后的抬升高度。 |
| `noise.end_effector_euler_deg` | `[-12, 12]` | 扰动夹爪姿态。 |
| `noise.picking_position_noise` | `[-0.035, 0.035]` | 扰动抓取目标点，三轴都会受影响。 |
| `noise.noise_scale` | `2.45` | 整体放大噪声区间。5 物体泛化版为 `2.35`。 |
| `noise.failure_bias_ratio` | `0.92` | 约 92% episode 使用更偏失败的强噪声。 |
| `noise.noise_distribution` | `uniform` | 在放缩后的区间内均匀采样。 |

## Place

冻结配置为 `config/level1_place_noise_ring_v2.yaml`。

| 调整项 | 冻结值 | 含义 |
| --- | --- | --- |
| `task.max_steps` | `2000` | place 包含先 pick 再放置，允许更长 episode。 |
| `task.obj_paths` | `beaker2` 和 `target_plat` | 待放置物体与目标平台。 |
| `pick.pre_offset_x` | `0.05` | place 前置 pick 阶段的接近 x 偏移。 |
| `pick.pre_offset_z` | `0.05` | pick 阶段的预抓取高度。 |
| `pick.after_offset_z` | `0.15` | 抓起后的抬升高度。 |
| `place.pick_success_min_height_delta` | `0.1` | pick 阶段至少抬起 10 cm 才认为抓取有效。 |
| `place.place_success_xy_threshold` | `0.05` | 放置后 xy 方向离目标小于 5 cm 才成功。 |
| `place.place_success_z_threshold` | `0.05` | 放置后 z 方向离目标小于 5 cm 才成功。 |
| `place.end_effector_euler_deg` | `[0, 90, 20]` | 放置阶段夹爪姿态。 |
| `place.pre_place_z` | `0.2` | 放置前在目标上方悬停的高度。 |
| `place.place_offset_z` | `0.05` | 最终放置点的 z 偏移。 |
| `place.retreat_offset_x` | `-0.15` | 松爪后沿 x 方向后撤。 |
| `place.retreat_offset_z` | `0.15` | 松爪后向上后撤。 |

| 噪声项 | 冻结值 | 含义 |
| --- | --- | --- |
| `noise.place_position_mode` | `radial_ring` | 在目标周围的圆环上采样放置目标偏移。 |
| `noise.place_position_radius_range` | `[0.044, 0.068]` | 放置目标 xy 偏移半径，是主要成功率控制参数。 |
| `noise.place_position_angle_deg` | `[0, 360]` | 圆环噪声角度全方向采样。 |
| `noise.place_position_z_noise` | `[-0.003, 0.003]` | 放置高度微扰。 |
| `noise.pre_place_z` | `[-0.008, 0.008]` | 放置前悬停高度扰动。 |
| `noise.place_offset_z` | `[-0.006, 0.006]` | 最终释放高度扰动。 |
| `noise.end_effector_euler_deg` | `[-6, 6]` | 放置姿态扰动。 |
| `noise.failure_bias_ratio` | `0.0` | 不额外使用失败倾向放缩。 |

## Pour

冻结配置为 `config/level1_pour_noise_multi_obj_v2.yaml`。

| 调整项 | 冻结值 | 含义 |
| --- | --- | --- |
| `task.max_steps` | `1500` | pick 后进入倾倒流程，允许较长轨迹。 |
| `left_pos` | 接收杯位置范围 | pour 目标接收容器的位置采样区域。 |
| `task.obj_paths` | `beaker2`、`conical_bottle02`、`conical_bottle03`、`graduated_cylinder_03` | 多种源容器轮换采集。 |
| `pick.after_offset_z` | `0.5` | 抓起源容器后抬高，给 pour 留出空间。 |
| `pick.success_min_height_delta` | `0.12` | pick 阶段最小抬升高度。 |
| `pick.end_effector_euler_deg` | `[0, 90, 30]` | 抓取源容器姿态。 |
| `pour.stage0_xy_threshold` | `0.08` | 接近 pour 起点时的 xy 到达阈值。 |
| `pour.position_threshold` | `0.006` | pour 阶段位置收敛阈值。 |
| `pour.approach_height_range` | `[0.3, 0.4]` | 倾倒前在目标上方的高度范围。 |
| `pour.pour_height_range` | `[0.1, 0.2]` | 倾倒时源容器相对目标的高度范围。 |
| `pour.end_effector_euler_deg` | `[0, 90, 10]` | 倾倒阶段末端姿态。 |
| `pour.pour_speed` | `-1.0` | 倾倒旋转速度。 |
| `pour.success_distance_buffer` | `0.05` | pour 成功距离阈值的缓冲量。 |
| `pour.pour_rotation_threshold_deg` | `50` | 倾倒角达到该值后认为完成倒液动作。 |
| `pour.return_rotation_threshold_deg` | `30` | 回正旋转的完成阈值。 |
| `pour.return_hold_seconds` | `1.0` | 回正后保持时间。 |

| 噪声项 | 冻结值 | 含义 |
| --- | --- | --- |
| `noise.picking_position_noise` | `[-0.02, 0.02]` | pick 目标点扰动。 |
| `noise.pour_target_position_mode` | `radial_ring` | 在接收杯周围环形采样 pour 目标偏移。 |
| `noise.pour_target_position_radius_range` | `[0.055, 0.095]` | pour 对准误差半径，是主要成功率控制参数。 |
| `noise.pour_target_position_angle_deg` | `[0, 360]` | pour 目标偏移角度全方向采样。 |
| `noise.pour_target_position_z_noise` | `[-0.005, 0.005]` | pour 目标高度扰动。 |
| `noise.approach_height_offset` | `[-0.02, 0.02]` | 倾倒前接近高度扰动。 |
| `noise.pour_height_offset` | `[-0.015, 0.015]` | 倾倒高度扰动。 |
| `noise.pour_speed` | `[-0.2, 0.2]` | 倾倒速度扰动。 |
| `noise.noise_scale` | `1.5` | 整体放大噪声。 |
| `noise.failure_bias_ratio` | `0.8` | 大部分 episode 使用更偏失败的采样。 |
| `noise.noise_distribution` | `edge_bias` | 更容易采到区间边界，制造偏离目标的样本。 |

## Open Door

冻结配置为 `config/level1_open_door_noise_multi_obj_v2.yaml`。

| 调整项 | 冻结值 | 含义 |
| --- | --- | --- |
| `task.operate_type` | `door` | 使用门类开合逻辑。 |
| `task.obj_paths` | `DryingBox_01`、`DryingBox_02`、`MuffleFurnace` | 多个门类物体轮换采集。 |
| `task.object_switch_interval` | `150` | 每个门物体采 150 个写入 episode 后切换。 |
| `open.stage0_offset_x` | `0.08` | 抓把手前第一阶段 x 方向接近偏移。 |
| `open.stage1_offset_x` | `0.015` | 把手附近第二阶段接触偏移。 |
| `open.retreat_offset_x` | `0.06` | 打开后沿 x 方向后撤。 |
| `open.retreat_offset_y` | `0.04` | 打开后沿 y 方向后撤。 |
| `open.door_open_angle_deg` | `50` | 目标开门角度。 |
| `open.end_effector_euler_deg` | `[0, 110, 0]` | 开门时夹爪姿态。 |

| 噪声项 | 冻结值 | 含义 |
| --- | --- | --- |
| `noise.stage0_offset_x` | `[0.01, 0.04]` | 扰动第一阶段接近距离，影响能否稳定抓住把手。 |
| `noise.stage1_offset_x` | `[0.003, 0.018]` | 扰动近把手接触距离。 |
| `noise.retreat_offset_x` | `[-0.04, -0.01]` | 扰动打开后的后撤路径。 |
| `noise.retreat_offset_y` | `[-0.03, -0.005]` | 扰动打开后的侧向后撤路径。 |
| `noise.end_effector_euler_deg` | `[-12, 12]` | 扰动夹爪姿态。 |
| `noise.door_open_angle_deg` | `[-20, 0]` | 减小目标开门角度，制造开门不足样本。 |
| `noise.close_gripper_distance` | `[0.003, 0.012]` | 扰动夹爪闭合距离，影响把手抓取稳定性。 |
| `noise.noise_scale` | `1.6` | 整体放大噪声。 |
| `noise.failure_bias_ratio` | `0.9` | 大部分 episode 使用失败倾向采样。 |
| `noise.noise_distribution` | `edge_bias` | 更偏向采到边界噪声。 |

## Open Drawer

冻结配置为 `config/level1_open_drawer_noise_multi_obj_v5.yaml`。

| 调整项 | 冻结值 | 含义 |
| --- | --- | --- |
| `task.operate_type` | `drawer` | 使用抽屉开合逻辑。 |
| `task.obj_paths` | `Cabinet_01`、`Cabinet_02` | 两个抽屉物体轮换采集。 |
| `open.stage0_offset_x` | `0.08` | 抓把手前第一阶段 x 接近偏移。 |
| `open.stage1_offset_x` | `0.015` | 把手附近第二阶段接触偏移。 |
| `open.retreat_offset_x` | `0.06` | 拉开后 x 方向后撤。 |
| `open.retreat_offset_y` | `0.04` | 拉开后 y 方向后撤。 |
| `open.drawer_pull_offset_x` | `0.04` | 抽屉打开方向的拉动距离。 |
| `open.drawer_retreat_offset_x` | `0.12` | 拉开后的 x 方向退出距离。 |
| `open.drawer_retreat_offset_z` | `0.06` | 拉开后的 z 方向退出高度。 |
| `open.close_gripper_distance` | `0.01` | 抓把手时夹爪闭合距离。 |
| `open.end_effector_euler_deg` | `[90, 90, 0]` | 拉抽屉时末端姿态。 |

| 噪声项 | 冻结值 | 含义 |
| --- | --- | --- |
| `noise.stage0_offset_x` | `[0.0, 0.007]` | 轻微扰动第一阶段接近距离。 |
| `noise.stage1_offset_x` | `[0.0, 0.0035]` | 轻微扰动近把手接触距离。 |
| `noise.drawer_pull_offset_x` | `[-0.011, -0.0025]` | 减小拉抽屉距离，制造打开不足样本。 |
| `noise.drawer_retreat_offset_x` | `[-0.013, -0.004]` | 扰动拉开后的 x 退出距离。 |
| `noise.drawer_retreat_offset_z` | `[-0.006, 0]` | 扰动拉开后的 z 退出高度。 |
| `noise.end_effector_euler_deg` | `[-3, 3]` | 小幅扰动末端姿态。 |
| `noise.close_gripper_distance` | `[0, 0.003]` | 小幅扰动抓把手闭合距离。 |
| `noise.failure_bias_ratio` | `0.0` | 不额外启用失败倾向放缩。 |

## Close Door

冻结配置为 `config/level1_close_door_noise_v3.yaml`。

| 调整项 | 冻结值 | 含义 |
| --- | --- | --- |
| `task.operate_type` | `door` | 使用门类关闭逻辑。 |
| `task.bootstrap_open_with_controller` | `false` | 不在当前 episode 内用 open controller 额外 warmup。 |
| `task.obj_paths` | `DryingBox_03` | 关门冻结版使用的门物体。 |
| `close.door_close_angle_deg` | `50` | nominal 关门旋转量。 |
| `close.end_effector_euler_deg` | `[350, 90, 25]` | 关门推压时的末端姿态。 |
| `close.door_target_distance_threshold` | `0.15` | 门把手离 clean close target 小于该值才判定接近关闭。 |

| 噪声项 | 冻结值 | 含义 |
| --- | --- | --- |
| `noise.end_effector_euler_deg` | `[-18, 18]` | 扰动关门时的推压姿态。 |
| `noise.end_effector_euler_distribution` | `edge_bias` | 姿态噪声更偏向两端，增加推偏概率。 |
| `noise.door_close_angle_deg` | `[-62, -14]` | 对 nominal 关门角度施加负向扰动，制造关闭不足或接触不稳。 |
| `noise.door_close_angle_distribution` | `uniform` | 关门角度扰动均匀采样。 |
| `noise.failure_bias_ratio` | `0.0` | 不额外启用失败倾向放缩。 |

## Close Drawer

冻结配置为 `config/level1_close_drawer_noise_v14.yaml`。

| 调整项 | 冻结值 | 含义 |
| --- | --- | --- |
| `task.operate_type` | `drawer` | 使用抽屉关闭逻辑。 |
| `task.bootstrap_open_with_controller` | `true` | episode 开始时先把抽屉初始化到打开状态。 |
| `task.initial_drawer_open_distance` | `0.16` | 初始打开距离。 |
| `task.obj_paths` | `Cabinet_02` | 关抽屉冻结版使用的抽屉物体。 |
| `close.events_dt` | `[0.0005, 0.002, 0.03, 0.012]` | 关抽屉 primitive 各阶段推进速度。 |
| `close.post_hold_steps` | `165` | 动作结束后额外等待并检查成功条件的窗口。 |
| `close.drawer_closed_open_distance_threshold` | `0.06` | 抽屉剩余打开距离小于 6 cm 才认为足够关闭。 |
| `close.push_distance` | `0.15` | nominal 推抽屉距离。 |
| `close.drawer_approach_offset_x` | `0.1` | 推之前的 x 方向接近偏移。 |
| `close.drawer_push_offset_x` | `0.05` | 推压接触点的 x 方向偏移。 |
| `close.drawer_retreat_offset_x` | `0.1` | 推完后 x 方向退出距离。 |
| `close.drawer_retreat_offset_z` | `0.08` | 推完后 z 方向退出高度。 |
| `close.drawer_retreat_distance_threshold` | `0.08` | 退出阶段完成阈值。 |
| `close.end_effector_euler_deg` | `[90, 90, 0]` | 推抽屉时的末端姿态。 |

| 噪声项 | 冻结值 | 含义 |
| --- | --- | --- |
| `noise.push_distance` | `[-0.128, -0.020]` | 减小推入距离，主要制造关不严样本。 |
| `noise.push_distance_distribution` | `min_bias` | 更偏向较小推入距离。 |
| `noise.drawer_approach_offset_x` | `[-0.018, 0.004]` | 扰动推之前的接近点。 |
| `noise.drawer_push_offset_x` | `[-0.041, 0.0]` | 减小或改变实际推压接触点。 |
| `noise.drawer_push_offset_x_distribution` | `min_bias` | 更偏向较小推压偏移。 |
| `noise.drawer_contact_offset_y` | `[-0.118, 0.118]` | 扰动接触点的左右位置，影响是否推到面板有效区域。 |
| `noise.drawer_contact_offset_y_distribution` | `edge_bias` | 更容易采到左右极端接触点。 |
| `noise.drawer_contact_offset_z` | `[0.001, 0.088]` | 扰动接触点高度。 |
| `noise.drawer_contact_offset_z_distribution` | `max_bias` | 更偏向较高接触点。 |
| `noise.end_effector_euler_deg` | `[-9.1, 9.1]` | 扰动推压姿态。 |
| `noise.end_effector_euler_distribution` | `edge_bias` | 姿态更容易采到区间两端。 |

## Press

冻结配置为 `config/level1_press_noise_v5.yaml`。

| 调整项 | 冻结值 | 含义 |
| --- | --- | --- |
| `task.randomize_button_material` | `false` | 关闭按钮材质随机化，避免 MDL 材质参数报错干扰。 |
| `press.events_dt` | `[0.005, 0.1, 0.005]` | 接近、按压、回撤阶段的推进时间。 |
| `press.initial_offset` | `0.2` | 按按钮前的起始距离。 |
| `press.press_distance` | `0.04` | nominal 按压深度。 |
| `press.end_effector_euler_deg` | `[0, 90, 10]` | 按压时末端姿态。 |
| `press.success_threshold_x` | `0.405` | 按钮位移达到阈值才判定按下成功。 |

| 噪声项 | 冻结值 | 含义 |
| --- | --- | --- |
| `noise.initial_offset` | `[0.035, 0.085]` | 扰动按压前起始距离。 |
| `noise.press_distance` | `[-0.085, -0.035]` | 减小实际按压深度，制造按压不足。 |
| `noise.end_effector_euler_deg` | `[-17, 17]` | 扰动按压姿态。 |
| `noise.target_position_offset_x` | `[-0.065, -0.023]` | 沿 x 方向扰动按钮目标点，主要制造未按到底。 |
| `noise.target_position_offset_y` | `[-0.015, 0.015]` | 横向扰动按钮目标点。 |
| `noise.target_position_offset_z` | `[-0.015, 0.015]` | 高度方向扰动按钮目标点。 |
| `noise.failure_bias_ratio` | `1.0` | 每个 episode 都使用失败倾向噪声。 |
| `noise.noise_distribution` | `edge_bias` | 更偏向采到边界噪声。 |

## Shake

冻结配置为 `config/level1_shake_noise_v2.yaml`。

| 调整项 | 冻结值 | 含义 |
| --- | --- | --- |
| `task.max_steps` | `3000` | shake 轨迹较长，允许更长 episode。 |
| `task.obj_paths` | `beaker_2` | 摇晃对象。 |
| `pick.gripper_distance` | `0.02` | 抓取烧杯时夹爪闭合距离。 |
| `pick.pre_offset_x` | `0.05` | 抓取前 x 方向接近偏移。 |
| `pick.pre_offset_z` | `0.05` | 抓取前高度偏移。 |
| `pick.after_offset_z` | `0.15` | 抓起烧杯后的抬升高度。 |
| `pick.end_effector_euler_deg` | `[0, 90, 30]` | 抓取烧杯姿态。 |
| `shake.shake_distance` | `0.07` | nominal 摇晃幅度。 |
| `shake.end_effector_euler_deg` | `[0, 90, 10]` | 摇晃阶段末端姿态。 |
| `success.min_lift_height` | `0.05` | 烧杯至少抬高 5 cm。 |
| `success.min_shake_span_xy` | `0.045` | xy 摇晃跨度至少达到 4.5 cm。 |
| `success.required_shake_count` | `4` | 至少检测到 4 次有效摇晃。 |
| `success.hold_steps` | `60` | 结束后稳定保持帧数。 |
| `success.max_hold_xy_delta` | `0.0125` | 稳定阶段 xy 漂移上限。 |
| `success.post_hold_max_steps` | `240` | 动作后最多额外等待稳定的步数。 |
| `success.post_hold_settle_steps` | `45` | 额外等待中用于判稳的连续帧数。 |

| 噪声项 | 冻结值 | 含义 |
| --- | --- | --- |
| `noise.pick_gripper_distance` | `[0, 0.005]` | 扰动抓取夹爪距离，影响是否抓稳。 |
| `noise.pick_end_effector_euler_deg` | `[-8, 8]` | 扰动抓取姿态。 |
| `noise.shake_distance` | `[-0.04, -0.01]` | 减小摇晃幅度，制造摇晃次数或跨度不足。 |
| `noise.shake_end_effector_euler_deg` | `[-14, 14]` | 扰动摇晃姿态。 |
| `noise.shake_initial_position_offset_x` | `[-0.015, 0.015]` | 摇晃起始点 x 偏移。 |
| `noise.shake_initial_position_offset_y` | `[-0.04, 0.04]` | 摇晃起始点 y 偏移。 |
| `noise.shake_initial_position_offset_z` | `[-0.03, 0.03]` | 摇晃起始点 z 偏移。 |
| `noise.failure_bias_ratio` | `0.9` | 大部分 episode 使用失败倾向采样。 |
| `noise.noise_distribution` | `edge_bias` | 更偏向采到边界噪声。 |

## Stir

冻结配置为 `config/level1_stir_noise_v4.yaml`。

| 调整项 | 冻结值 | 含义 |
| --- | --- | --- |
| `task.max_steps` | `2400` | stir 轨迹较长，允许更长 episode。 |
| `task.target_position_range_xy` | `[0.075, 0.075]` | 搅拌目标容器的 xy 采样范围。 |
| `pick.object_name` | `glass_rod` | 先抓取玻璃棒。 |
| `pick.pre_offset_x` | `0.1` | 抓玻璃棒前 x 方向接近偏移。 |
| `pick.pre_offset_z` | `0.12` | 抓玻璃棒前高度偏移。 |
| `pick.after_offset_z` | `0.15` | 抓起玻璃棒后的抬升高度。 |
| `pick.gripper_distance` | `0.005` | 抓玻璃棒夹爪闭合距离。 |
| `pick.end_effector_euler_deg` | `[0, 90, 30]` | 抓玻璃棒姿态。 |
| `stir.stir_radius` | `0.009` | nominal 搅拌半径。 |
| `stir.stir_speed` | `3.0` | nominal 搅拌速度。 |
| `stir.center_position_offset` | `[0, 0, 0]` | clean 搅拌中心偏移。 |
| `stir.end_effector_euler_deg` | `[0, 90, -10]` | 搅拌阶段末端姿态。 |
| `success.min_height` | `0.85` | 玻璃棒高度下限。 |
| `success.max_xy_distance` | `0.04` | 玻璃棒离烧杯中心的 xy 最大距离。 |
| `success.hold_steps` | `240` | 成功条件需要保持的帧数。 |

| 噪声项 | 冻结值 | 含义 |
| --- | --- | --- |
| `noise.pick_gripper_distance` | `[0, 0.004]` | 扰动抓玻璃棒夹爪距离。 |
| `noise.pick_after_offset_z` | `[-0.044, -0.01]` | 降低抓起后的抬升高度，制造抓取或高度不足。 |
| `noise.pick_end_effector_euler_deg` | `[-7, 7]` | 扰动抓取姿态。 |
| `noise.stir_radius` | `[-0.0043, -0.0015]` | 减小搅拌半径，影响搅拌轨迹是否有效。 |
| `noise.stir_speed` | `[-1.0, 0.3]` | 扰动搅拌速度，多数情况下偏慢。 |
| `noise.stir_end_effector_euler_deg` | `[-14, 14]` | 扰动搅拌姿态。 |
| `noise.stir_center_offset_x` | `[-0.041, 0.041]` | 搅拌中心 x 偏移。 |
| `noise.stir_center_offset_y` | `[-0.050, 0.050]` | 搅拌中心 y 偏移。 |
| `noise.stir_center_offset_z` | `[-0.017, 0.013]` | 搅拌中心 z 偏移。 |
| `noise.failure_bias_ratio` | `0.9` | 大部分 episode 使用失败倾向采样。 |
| `noise.noise_distribution` | `edge_bias` | 更偏向采到边界噪声。 |

## 使用建议

这些冻结配置的调参目标不是让每个动作“尽量失败”，而是在 clean baseline 可稳定完成的前提下，通过有物理含义的参数扰动把成功率压到可训练的范围。优先解释以下几类参数：

| 类别 | 对应参数 | 解释重点 |
| --- | --- | --- |
| 对准误差 | `*_position_noise`、`*_target_position_*`、`*_center_offset_*` | 控制末端是否对准把手、按钮、容器中心或放置目标。 |
| 接触强度 | `push_distance`、`press_distance`、`drawer_pull_offset_x`、`door_close_angle_deg` | 控制动作是否做够，例如推够、按够、拉够或转够。 |
| 姿态误差 | `end_effector_euler_deg` 及各阶段姿态噪声 | 控制夹爪或工具与物体接触时的角度偏差。 |
| 轨迹稳定性 | `events_dt`、`post_hold_steps`、`hold_steps`、`position_threshold` | 控制动作是否平滑、是否需要在成功状态稳定保持。 |
| 失败样本比例 | `noise_scale`、`failure_bias_ratio`、`noise_distribution` | 控制整体成功率，不改变动作语义。 |

# LabUtopia 数字资产清单

## 一、USD 场景文件

| 路径 | 用途 | 引用配置 |
|------|------|----------|
| `assets/chemistry_lab/lab_001/lab_001.usd` | 主化学实验室场景 | level1_pick, level3_pick, level3_open, level3_TransportBeaker, level3_PourLiquid, level2_*, level1_place/pour/open* |
| `assets/chemistry_lab/lab_003/lab_003.usd` | 另一实验室（加热/搅拌/摇匀） | level3_press, level3_HeatLiquid, level2_StirGlassrod, level2_ShakeBeaker, level2_HeatLiquid, level1_stir/shake/press |
| `assets/chemistry_lab/lab_003/clock.usd` | 时钟场景 | level4_LiquidMixing |
| `assets/chemistry_lab/hard_task/Scene1_hard.usd` | 清洗烧杯困难任务 | level4_CleanBeaker, level4_CleanBeaker7Policy |
| `assets/chemistry_lab/hard_task/lab_004.usd` | 备用场景 | - |
| `assets/navigation_lab/navigation_lab_01/lab.usd` | 导航场景 | level5_Navigation, level5_Mobile_manipulation |
| `assets/robots/Franka.usd` | Franka 机械臂 | 默认机器人 |
| `assets/robots/ridgeback_franka.usd` | 移动底盘+机械臂 | level5 |
| `assets/fetch/fetch.usd` 等 | Fetch 机器人 | - |

**注意**：README 示例中的 `assets/chemistry_lab/pick_task/scene.usd` 不存在，实际使用 `lab_001/lab_001.usd`。

---

## 二、可操作物体（按场景）

### lab_001（主场景）

| Prim 路径 | 类型 | 用途 | task_utils 支持 |
|-----------|------|------|-----------------|
| `/World/conical_bottle02` | 锥形瓶 | Pick, Pour 源 | ✅ pick/pour |
| `/World/conical_bottle03` | 锥形瓶 | Pick, Pour 源 | ✅ pick/pour |
| `/World/conical_bottle04` | 锥形瓶 | Pick 源 | ✅ pick/pour |
| `/World/beaker2` | 烧杯 | Pick, Pour 源, Place 源 | ✅ pick/pour |
| `/World/beaker1` | 烧杯 | Pour 目标 | ✅ pour |
| `/World/beaker3` | 烧杯 | DeviceOperation 第二烧杯 | ✅ (beaker) |
| `/World/graduated_cylinder_03` | 量筒 | Pour 源 | ✅ pick/pour |
| `/World/volume_flask` | 容量瓶 | （代码支持，配置未用） | ✅ pick/pour |
| `/World/target_plat` | 平台 | Place 目标 | - |
| `/World/target_plat2` | 平台 | OpenTransportPour | - |
| `/World/DryingBox_01` | 干燥箱 | Open 门 | - |
| `/World/DryingBox_02` | 干燥箱 | Open 门 | - |
| `/World/DryingBox_03` | 干燥箱 | Close 门 | - |
| `/World/MuffleFurnace` | 马弗炉 | Open 门, 运输目标 | - |
| `/World/Cabinet_01` | 柜子 | Open/Close 抽屉 | - |
| `/World/Cabinet_02` | 柜子 | Close 抽屉 | - |
| `/World/glass_rod` | 玻璃棒 | Stir 工具 | - |
| `/World/test_tube_rack` | 试管架 | Stir 场景 | - |
| `/World/table/surface` | 桌面 | 材质切换 | - |

### lab_003

| Prim 路径 | 类型 | 用途 |
|-----------|------|------|
| `/World/beaker_2` | 烧杯 | Shake, Stir 目标, HeatLiquid |
| `/World/target_beaker` | 烧杯 | Stir 目标 |
| `/World/glass_rod` | 玻璃棒 | Stir 工具 |
| `/World/instrument` | 仪器 | Press 场景 |
| `/World/target_button` | 按钮 | Press 目标 |
| `/World/distractor_button_1/2` | 干扰按钮 | Press |
| `/World/heat_device` | 加热设备 | HeatLiquid |
| `/World/Table1/Desk1/surface/Cube` | 桌面 | HeatLiquid |

### hard_task (Scene1_hard)

| Prim 路径 | 类型 | 用途 |
|-----------|------|------|
| `/World/target_beaker` | 目标烧杯 | CleanBeaker 倾倒目标 |
| `/World/beaker_hard_1` | 烧杯 | CleanBeaker 源 |
| `/World/beaker_hard_2` | 烧杯 | CleanBeaker 源 |
| `/World/target_plat_1` | 平台 | Place 目标 |
| `/World/target_plat_2` | 平台 | Place 目标 |
| `/World/table_hard/...` | 桌面 | 材质表面 |

### LiquidMixing (clock.usd)

| Prim 路径 | 类型 | 用途 |
|-----------|------|------|
| `/World/beaker_4` | 烧杯 | 主烧杯 |
| `/World/beaker_03` ~ `beaker_05` | 烧杯 | 多烧杯混合 |

### navigation_lab

| Prim 路径 | 类型 | 用途 |
|-----------|------|------|
| `/World/beaker` | 烧杯 | Mobile Pick 目标 |
| `/World/Ridgebase` | 移动底盘 | 机器人 |

---

## 三、task_utils 已支持物体（抓取/倾倒参数）

| 物体名 | get_pickz_offset | get_pour_threshold |
|--------|------------------|--------------------|
| conical_bottle02 | 0.03 | 0.03 |
| conical_bottle03 | 0.06 | 0.07 |
| conical_bottle04 | 0.08 | 0.08 |
| beaker2 | 0.02 | 0.02 |
| beaker | 0.02 (pour) | 0.02 |
| graduated_cylinder_01~04 | 0.0 | 0.0 |
| volume_flask | 0.05 | 0.05 |

---

## 四、材质资产

- **lab_001**：`/World/Looks/Material1_plastic_dark_blue` ~ `Material7_stainless`
- **lab_003**：`/World/Looks/Material_1_*` ~ `Material_7_*`，`OmniPBR_Button_*`
- **hard_task**：`Material_1_*` ~ `Material_7_*`，`table_material5`
- 纹理：`SubUSDs/textures/` 下 `*.jpg`、`*.png`（Steel_Stainless、Ash_Planks、lounge_booth_table 等）

---

## 五、VLM 多物体采集可扩展物体

当前 `level1_pick_noise_universal_multi_obj` 使用 **conical_bottle02/03/04**。

**资产归类**：`object_type` 自动归为同类（如 conical_bottle02/03/04→conical_bottle），避免同一类资产因编号被拆成多类数据。可在 obj_paths 中显式指定 `category: conical_bottle`。

**lab_001 中可追加**（若 prim 存在）：
- `beaker2`：烧杯，已有参数
- `volume_flask`：容量瓶，已有参数
- `graduated_cylinder_03`：量筒，已有参数

**需验证**：在 Isaac Sim 中打开 `lab_001.usd`，确认上述 prim 路径存在且可交互。

---

## 六、场景验证（你当前 lab_001 里到底有什么）

**重要**：`ASSET_INVENTORY.md` 中的 prim 列表来自项目设计文档，**不同版本的 lab_001.usd 可能包含不同资产**。若采集时出现 `[Pick] 提前返回：object_position=None`，说明配置中的某些 prim 在场景中不存在。

### 6.1 运行时自动验证

启动采集后，若某 prim 不存在，会打印：

```
[BaseTask] 跳过不存在的 prim: /World/xxx（lab 中可能未包含该资产）
```

据此可知哪些物体被跳过。

### 6.2 手动列出场景 prim

在 **Isaac Sim 环境**（如 `conda activate env_isaacsim`）下运行：

```bash
python scripts/list_scene_prims.py
```

会输出 `/World` 下所有 prim 及 Pick 相关物体的存在性。

### 6.3 已知差异

| 物体 | 状态 | 说明 |
|------|------|------|
| conical_bottle02/03/04 | 通常存在 | level1_pick 默认使用，多数 lab_001 包含 |
| beaker2 | 通常存在 | level3_PourLiquid、level4 等配置使用 |
| graduated_cylinder_03 | 需验证 | 部分 lab 可能只有 01/02/04 |
| **volume_flask** | **常不存在** | `level1_pick_noise_universal_5obj` 已明确不含此物体 |

### 6.4 若 volume_flask 不存在

- **方案 A**：使用 `level1_pick_noise_universal_all_obj`，代码会自动跳过不存在的 prim，实际采集 3 类（conical_bottle、beaker、graduated_cylinder）。
- **方案 B**：使用 `level1_pick_noise_universal_5obj`，显式不含 volume_flask，含 5 物体（3 锥形瓶 + beaker2 + graduated_cylinder_03）。

### 6.5 若需要 volume_flask

需在 Isaac Sim 中手动将容量瓶资产加入 lab_001 场景，并确保 prim 路径为 `/World/volume_flask`。项目未提供现成的 volume_flask 子场景引用路径，需自行从资产库或外部 USD 导入。

---

## 七、其他潜在问题

1. **README 路径过时**：`pick_task/scene.usd` 不存在，应改为 `lab_001/lab_001.usd`。
2. **物体命名不一致**：`beaker2` vs `beaker_2`（lab_001 用 beaker2，lab_003 用 beaker_2）。
3. **graduated_cylinder**：配置中仅 `graduated_cylinder_03` 被使用，01/02/04 未在场景中引用。

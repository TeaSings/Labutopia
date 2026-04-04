import os
import re
from typing import Optional, Tuple, Any

import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from robots.franka.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R

from factories.collector_factory import create_collector

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from .inference_engines.inference_engine_factory import InferenceEngineFactory
from utils.vlm_api_client import VlmApiClient, parse_json_bool
from utils.vlm_image_utils import frame_to_hwc_rgb
class PickTaskController(BaseController):
    """
    Controller for pick-and-place tasks with two operation modes:
    - Collection mode: Gathers training data through demonstrations
    - Inference mode: Executes learned policies for autonomous picking

    「重置」分两类（勿混淆）：
    - **整局重置**：`main` 在 episode 结束时调用 `task.reset()`（含 `world.reset()`、机器人与物体在 `SingleObjectTask.reset`
      / `randomize_object_position` 中重新摆放并恢复物体旋转）。物体碰倒触发的 `_check_tipped_reset_episode` 属于此类。
    - **纠错中的关节回退**（仅 vlm_live，`recover_to_home_after_failure`）：同一 episode 内只把机械臂插值回
      本局开始时记录的关节角，再请求纠错 / 再 pick；**不**重置场景物体位姿，也**不能**代替碰倒后的整局重置。

    Attributes:
        mode (str): Operation mode ("collect" or "infer")
        REQUIRED_SUCCESS_STEPS (int): Number of consecutive steps needed for success
        success_counter (int): Counter for tracking successful steps
    """
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
        self._episode_noise = {}  # 各参数噪声
        self._episode_properties_set = False
        self._early_return = False
        # 抓住但高度不足时尝试非重置纠正：继续上抬
        self._lift_correction_enabled = getattr(getattr(cfg, 'pick', None), 'lift_correction_enabled', True)
        self._lift_correction_active = False
        self._lift_correction_steps = 0
        self._lift_correction_target = None
        self._last_success_was_corrected = False
        # 成功池：按 object_type 存储成功样本的 params_used，用于失败时计算 correction 朝向正确空间
        self._success_pool = {}  # {object_type: [params_used, ...]}
        self._success_pool_max = int(getattr(getattr(cfg, 'pick', None), 'success_pool_max', 500))
        # 多次 correction：失败时生成多步 correction_steps，每步朝正确空间移动
        self._correction_steps_count = int(getattr(getattr(cfg, 'pick', None), 'correction_steps_count', 3))
        self._correction_alpha = float(getattr(getattr(cfg, 'pick', None), 'correction_alpha', 0.5))
        self._p_ref_method = str(getattr(getattr(cfg, 'pick', None), 'p_ref_method', 'mean'))
        # 纠错：完整 delta 保留；另拆出单位方向 + 模长，以及保守一步 step（方向不变、幅值上限），利于训练与真机小步修正
        _pc = getattr(cfg, "pick", None) or {}
        self._correction_enrich_unit_magnitude = bool(getattr(_pc, "correction_enrich_unit_magnitude", True))
        _smp = getattr(_pc, "correction_suggested_max_picking_norm", 0.02)
        self._correction_suggested_max_picking_norm = (
            float(_smp) if _smp is not None and float(_smp) > 0 else None
        )
        _sme = getattr(_pc, "correction_suggested_max_euler_l2_deg", 8.0)
        self._correction_suggested_max_euler_l2_deg = (
            float(_sme) if _sme is not None and float(_sme) > 0 else None
        )
        _smo = getattr(_pc, "correction_suggested_max_offset_abs", 0.06)
        self._correction_suggested_max_offset_abs = (
            float(_smo) if _smo is not None and float(_smo) > 0 else None
        )
        _csp = getattr(_pc, "correction_step_max_picking_norm", 0.02)
        self._correction_step_max_picking_norm = (
            float(_csp) if _csp is not None and float(_csp) > 0 else None
        )
        # 成功判断：连续 N 步物体高于阈值才判成功，避免物理抖动误判
        self.REQUIRED_SUCCESS_STEPS = int(getattr(getattr(cfg, 'pick', None), 'required_success_steps', 15))
        self._lift_success_counter = 0  # lift correction 阶段的连续成功计数
        self._lift_required_steps = int(getattr(getattr(cfg, 'pick', None), 'lift_required_success_steps', 10))
        # 物体倾倒检测：与本局「初始位姿」的世界旋转 R0 对比，相对转角超过阈值则整局重置；无矩阵时回退为竖直轴夹角
        _pick = getattr(cfg, "pick", None) or {}
        self._default_pre_offset_x = float(getattr(_pick, "pre_offset_x", 0.05))
        self._default_pre_offset_z = float(getattr(_pick, "pre_offset_z", 0.12))
        self._default_after_offset_z = float(getattr(_pick, "after_offset_z", 0.25))
        _default_euler = getattr(_pick, "end_effector_euler_deg", [0.0, 90.0, 25.0])
        self._default_euler_deg = np.asarray(_default_euler, dtype=float).reshape(-1)
        if self._default_euler_deg.size != 3:
            self._default_euler_deg = np.array([0.0, 90.0, 25.0], dtype=float)
        self._tipped_detection_enabled = bool(getattr(_pick, "tipped_detection_enabled", True))
        self._tipped_max_tilt_deg = float(getattr(_pick, "tipped_max_tilt_deg", 72.0))
        self._tipped_baseline_delay_steps = int(getattr(_pick, "tipped_baseline_delay_steps", 0))
        _axis = getattr(_pick, "tipped_local_up_axis", [0.0, 0.0, 1.0])
        self._tipped_local_up_axis = np.asarray(_axis, dtype=float).reshape(-1)
        if self._tipped_local_up_axis.size != 3:
            self._tipped_local_up_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        _n = float(np.linalg.norm(self._tipped_local_up_axis))
        if _n > 1e-9:
            self._tipped_local_up_axis = self._tipped_local_up_axis / _n
        else:
            self._tipped_local_up_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        # 根 prim 世界旋转不变、子刚体/网格在倒时：用世界 AABB 的「竖直度」sz/max(sx,sy) 作补充（含子树几何）
        self._tipped_bbox_fallback_enabled = bool(getattr(_pick, "tipped_bbox_fallback_enabled", True))
        self._tipped_bbox_min_initial_verticality = float(
            getattr(_pick, "tipped_bbox_min_initial_verticality", 1.18)
        )
        self._tipped_bbox_verticality_factor = float(
            getattr(_pick, "tipped_bbox_verticality_factor", 0.50)
        )
        self._initial_bbox_sizes: Optional[np.ndarray] = None
        self._initial_bbox_verticality: Optional[float] = None
        self._initial_world_rotation_3x3: Optional[np.ndarray] = None
        self._initial_object_up_world: Optional[np.ndarray] = None
        self._episode_step_idx = 0
        # 噪声配置：支持 picking_position、pre_offset_x/z、after_offset_z、末端姿态
        # picking_position 加噪后支持纯视觉纠正：correction 可把错误参数朝正确参数空间移动
        noise_cfg = getattr(cfg, 'noise', None)
        if noise_cfg and getattr(noise_cfg, 'enabled', False):
            self._noise_enabled = True
            scale = float(getattr(noise_cfg, 'noise_scale', 1.0))
            failure_bias = float(getattr(noise_cfg, 'failure_bias_ratio', 0.0))
            self._noise_distribution = str(getattr(noise_cfg, 'noise_distribution', 'uniform'))
            base = {
                'pre_offset_x': list(getattr(noise_cfg, 'pre_offset_x', [-0.04, 0.04])),
                'pre_offset_z': list(getattr(noise_cfg, 'pre_offset_z', [-0.04, 0.04])),
                'after_offset_z': list(getattr(noise_cfg, 'after_offset_z', [-0.04, 0.04])),
                'euler_deg': list(getattr(noise_cfg, 'end_effector_euler_deg', [-10, 10])),
            }
            # picking_position 噪声：x/y/z 各轴 ±2cm，纯视觉时 correction 可修正位置误差
            # 设为 null 可禁用位置噪声，仅对偏移/姿态加噪
            pos_noise = getattr(noise_cfg, 'picking_position_noise', [-0.02, 0.02])
            self._picking_position_noise_enabled = pos_noise is not None and pos_noise is not False
            if self._picking_position_noise_enabled:
                self._picking_position_noise_range = list(pos_noise) if isinstance(pos_noise, (list, tuple)) else [-0.02, 0.02]
            else:
                self._picking_position_noise_range = [-0.02, 0.02]  # 占位，不会使用
            self._noise_range = base
            self._noise_scale = scale
            self._failure_bias_ratio = failure_bias
        else:
            self._noise_enabled = False

        # 测试/推理/VLM 实时闭环：不使用采集用参数噪声（轨迹参数仅来自策略或 VLM，不经 _episode_noise）
        if getattr(self, "mode", None) in ("vlm_live", "infer"):
            self._noise_enabled = False

        if (
            getattr(self, "mode", None) == "collect"
            and self._noise_enabled
            and not getattr(self, "_picking_position_noise_enabled", False)
        ):
            print(
                "[Pick][Collect] 提示：已开启 noise.enabled，但未启用 picking_position_noise（YAML 中为 null/false）。"
                "此时每局 params_used.picking_position 与物体 GT 一致，HDF5 无 correction_gt.picking_position_delta，"
                "模型难以学习对 picking_position 的纠错。请设 noise.picking_position_noise: [-0.02, 0.02] 或继承 universal_vlm_collect。"
            )

    def _init_collect_mode(self, cfg, robot):
        """
        Initializes components for data collection mode.
        Sets up pick controller, gripper control, and data collector.

        Args:
            cfg: Configuration object containing collection settings
            robot: Robot instance to control
        """
        super()._init_collect_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        )

    def _init_vlm_live_mode(self, cfg, robot):
        """HTTP VLM：初始推断 → pick → 模型任务②判成功 → 否则任务③纠错循环。"""
        collector_cfg = cfg.collector
        self.data_collector = create_collector(
            getattr(collector_cfg, "type", "mock"),
            camera_configs=cfg.cameras,
            save_dir=cfg.multi_run.run_dir,
            max_episodes=cfg.max_episodes,
            compression=getattr(collector_cfg, "compression", None),
            save_frames=getattr(collector_cfg, "save_frames", -1),
            cache_stride=int(getattr(collector_cfg, "cache_stride", 1) or 1),
        )
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008],
        )
        vlm = getattr(cfg, "vlm_live", None) or {}
        _key = str(getattr(vlm, "api_key", "") or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip() or None
        self._vlm_client = VlmApiClient(
            base_url=str(getattr(vlm, "base_url", "http://127.0.0.1:8000/v1")),
            model=str(getattr(vlm, "model", "gpt-3.5-turbo")),
            api_key=_key,
            max_tokens=int(getattr(vlm, "max_tokens", 512)),
            timeout_s=float(getattr(vlm, "timeout_s", 300.0)),
        )
        # max_correction_attempts<=0 表示不限制，直到模型 is_success 或 terminate_correction_loop
        _raw_max = getattr(vlm, "max_correction_attempts", 0)
        self._vlm_max_correction_attempts = int(_raw_max) if _raw_max is not None else 0
        if self._vlm_max_correction_attempts <= 0:
            self._vlm_max_correction_attempts = 10**9
        # 时间采样与训练采集一致：DataCollector 在 save_frames==-1 时用 cache_stride（见 universal_vlm_collect 默认 20）
        collector_cfg = cfg.collector
        _coll_stride = int(getattr(collector_cfg, "cache_stride", 1) or 1)
        _explicit = getattr(vlm, "frame_stride", None)
        if _explicit is not None and str(_explicit).strip() != "":
            self._vlm_frame_stride = max(1, int(_explicit))
        else:
            self._vlm_frame_stride = max(1, _coll_stride)
        print(
            f"[Pick][VLM] 多帧缓冲采样：每 {_coll_stride} 步采 1 组三视角（collector.cache_stride），"
            f"实际使用 stride={self._vlm_frame_stride}"
            + (
                "（来自 vlm_live.frame_stride）"
                if _explicit is not None and str(_explicit).strip() != ""
                else "（未设 vlm_live.frame_stride 时与采集一致）"
            )
        )
        self._vlm_cam_keys = ["camera_1_rgb", "camera_2_rgb", "camera_3_rgb"]
        # 纠错用：任务②失败后仅机械臂关节向本 episode 首次记录的关节插值（非整局 task.reset，不恢复物体姿态）
        self._vlm_home_joint_positions: Optional[np.ndarray] = None
        self._vlm_recovering_to_home = False
        self._vlm_recovery_steps = 0
        self._vlm_recovery_alpha = float(getattr(vlm, "recovery_joint_alpha", 0.4) or 0.4)
        self._vlm_recovery_tol = float(getattr(vlm, "recovery_joint_tol", 0.08) or 0.08)
        self._vlm_recovery_max_steps = int(getattr(vlm, "recovery_max_steps", 500) or 500)
        self._vlm_recover_to_home = bool(getattr(vlm, "recover_to_home_after_failure", True))
        self._reset_vlm_episode_state()

    def _reset_vlm_episode_state(self) -> None:
        self._vlm_params = None
        self._vlm_frame_buffer: list = []
        self._vlm_step_idx = 0
        self._vlm_correction_count = 0
        self._vlm_init_done = False
        self._vlm_home_joint_positions = None
        self._vlm_recovering_to_home = False
        self._vlm_recovery_steps = 0

    def _vlm_normalize_params(self, raw: dict, state: dict) -> dict:
        euler = raw.get("euler_deg", self._default_euler_deg.tolist())
        euler_deg = np.array(euler[:3], dtype=float)
        gt = np.array(state["object_position"][:3], dtype=float)
        if raw.get("picking_position") is not None:
            pp = np.array(raw["picking_position"][:3], dtype=float)
        else:
            pp = gt.copy()
        return {
            "pre_offset_x": float(raw.get("pre_offset_x", self._default_pre_offset_x)),
            "pre_offset_z": float(raw.get("pre_offset_z", self._default_pre_offset_z)),
            "after_offset_z": float(raw.get("after_offset_z", self._default_after_offset_z)),
            "euler_deg": euler_deg,
            "picking_position": pp,
        }

    def _vlm_object_type_for_prompt(self, state: dict) -> str:
        """供 VLM 三处 API 的 object_type 文本：优先语义类别 object_category；与实例名并存时一并写出。"""
        cat = state.get("object_category")
        name = state.get("object_name")
        cat_s = str(cat).strip() if cat is not None else ""
        name_s = str(name).strip() if name is not None else ""
        if not cat_s and not name_s:
            print(
                "[Pick][VLM] 警告：state 缺少 object_category 与 object_name，"
                "VLM 提示词中「物体类型」将使用 unknown（请确认任务 state's get_basic_state_info 传入 object_path）"
            )
            return "unknown"
        if cat_s and name_s:
            name_base = re.sub(r"_?\d+$", "", name_s)
            if cat_s == name_s or cat_s == name_base:
                return cat_s
            return f"{cat_s}（目标物体名: {name_s}）"
        return cat_s or name_s

    def _maybe_buffer_vlm_frames(self, state: dict) -> None:
        # 回退初始关节过程中不追加帧，避免把回退过程采进「过程」缓冲
        if getattr(self, "_vlm_recovering_to_home", False):
            return
        cam = state.get("camera_data") or {}
        self._vlm_step_idx += 1
        if (self._vlm_step_idx - 1) % self._vlm_frame_stride != 0:
            return
        for k in self._vlm_cam_keys:
            if k in cam:
                self._vlm_frame_buffer.append(frame_to_hwc_rgb(cam[k]))

    def _vlm_capture_home_if_needed(self, state: dict) -> None:
        """本 episode 首次有 joint_positions 时记录关节角，供任务②失败后「仅机械臂」回退（非物体倒下后的整局重置）。"""
        if self._vlm_home_joint_positions is not None:
            return
        jp = state.get("joint_positions")
        if jp is None:
            return
        self._vlm_home_joint_positions = np.asarray(jp, dtype=float).copy()
        print("[Pick][VLM] 已记录本 episode 初始关节姿态（失败回退目标）")

    def _vlm_close_to_home(self, state: dict) -> bool:
        if self._vlm_home_joint_positions is None:
            return True
        jp = state.get("joint_positions")
        if jp is None:
            return True
        cur = np.asarray(jp, dtype=float)
        if cur.shape != self._vlm_home_joint_positions.shape:
            return True
        return float(np.max(np.abs(cur - self._vlm_home_joint_positions))) < self._vlm_recovery_tol

    def _vlm_action_toward_home(self, state: dict) -> ArticulationAction:
        cur = np.asarray(state["joint_positions"], dtype=float)
        tgt = self._vlm_home_joint_positions
        if tgt is None or cur.shape != tgt.shape:
            return ArticulationAction(joint_positions=[None] * int(cur.shape[0]))
        alpha = self._vlm_recovery_alpha
        nxt = cur + alpha * (tgt - cur)
        return ArticulationAction(joint_positions=nxt.tolist())

    @staticmethod
    def _vlm_params_nearly_equal(a: dict, b: dict, atol: float = 1e-5, euler_atol_deg: float = 0.02) -> bool:
        """两版 pick 参数是否几乎相同（用于检测纠错 API 是否重复输出）。"""
        try:
            for k in ("pre_offset_x", "pre_offset_z", "after_offset_z"):
                if abs(float(a[k]) - float(b[k])) > atol:
                    return False
            ea = np.asarray(a["euler_deg"], dtype=float).reshape(3)
            eb = np.asarray(b["euler_deg"], dtype=float).reshape(3)
            if float(np.max(np.abs(ea - eb))) > euler_atol_deg:
                return False
            pa = np.asarray(a["picking_position"], dtype=float).reshape(3)
            pb = np.asarray(b["picking_position"], dtype=float).reshape(3)
            return bool(float(np.max(np.abs(pa - pb))) <= atol)
        except Exception:
            return False

    def _vlm_apply_correction_after_failure(self, state: dict) -> Tuple[Any, bool, bool]:
        """任务②失败后调用纠错 API，更新参数并清空缓冲以便再次 pick。"""
        obj = self._vlm_object_type_for_prompt(state)
        lang = self.get_language_instruction() or ""
        if len(self._vlm_frame_buffer) < 3:
            added = self._vlm_append_current_triple(state)
            if len(self._vlm_frame_buffer) < 3:
                print(
                    f"[Pick][VLM] 无法在失败时纠错：缓冲仍不足 3 张（本次仅追加 {added} 张），"
                    "请检查 camera_data 三视角或降低 vlm_live.frame_stride"
                )
                self._last_success = False
                self.reset_needed = True
                return None, True, False
        try:
            p0 = self._vlm_params
            params_dict = {
                "pre_offset_x": float(p0["pre_offset_x"]),
                "pre_offset_z": float(p0["pre_offset_z"]),
                "after_offset_z": float(p0["after_offset_z"]),
                "euler_deg": p0["euler_deg"].tolist(),
                "picking_position": p0["picking_position"].tolist(),
            }
            # 第 N 次应用纠错（1-based）：与 API 提示中「第几次纠错」一致
            attempt_no = int(self._vlm_correction_count) + 1
            raw = self._vlm_client.correct_pick_params(
                obj,
                lang,
                params_dict,
                self._vlm_frame_buffer,
                correction_attempt=attempt_no,
            )
            new_params = self._vlm_normalize_params(raw, state)
            if self._vlm_params_nearly_equal(new_params, p0):
                print(
                    "[Pick][VLM] 警告：纠错 API 返回的参数与「纠错前」几乎相同；"
                    "多为确定性解码+相似图像导致重复输出。可在服务端为纠错请求略提高 temperature，"
                    "或检查 params_dict 是否随每次纠错更新（当前已传入上一段 pick 实际使用的 _vlm_params）。"
                )
            self._vlm_params = new_params
            self._vlm_correction_count += 1
            p = self._vlm_params
            print(
                f"[Pick][VLM] 已应用第 {self._vlm_correction_count} 次纠错 | "
                f"物体类型(发给 VLM): {obj} | "
                f"pre=({p['pre_offset_x']:.4f},{p['pre_offset_z']:.4f},{p['after_offset_z']:.4f}) "
                f"euler={p['euler_deg'].tolist()} pos={p['picking_position'].tolist()} | "
                "同一 episode 内重新执行 pick（仍以模型成功判断为准）"
            )
            self.pick_controller.reset()
            self._vlm_frame_buffer = []
            self._vlm_step_idx = 0
            self.check_success_counter = 0
            self.reset_needed = False
            return None, False, False
        except Exception as e:
            print(f"[Pick][VLM] 纠错请求失败: {e}")
            self._last_failure_reason = str(e)
            self._last_success = False
            self.reset_needed = True
            return None, True, False

    def _vlm_append_current_triple(self, state: dict) -> int:
        """把当前时刻 cam1→cam2→cam3 追加进纠错缓冲（用于 stride 导致张数不足时的补齐）。"""
        cam = state.get("camera_data") or {}
        n = 0
        for k in self._vlm_cam_keys:
            if k in cam:
                self._vlm_frame_buffer.append(frame_to_hwc_rgb(cam[k]))
                n += 1
        return n

    def _vlm_do_init(self, state: dict) -> None:
        cam = state.get("camera_data") or {}
        views = []
        for k in self._vlm_cam_keys:
            if k not in cam:
                raise RuntimeError(f"VLM 需要三视角 {self._vlm_cam_keys}，当前有: {list(cam.keys())}")
            views.append(frame_to_hwc_rgb(cam[k]))
        obj = self._vlm_object_type_for_prompt(state)
        lang = self.get_language_instruction() or ""
        raw = self._vlm_client.initial_pick_params(obj, lang, views)
        self._vlm_params = self._vlm_normalize_params(raw, state)
        self._vlm_init_done = True
        p = self._vlm_params
        print(
            "[VLM] 初始推断完成 | "
            f"物体类型(发给 VLM): {obj} | "
            f"pre=({p['pre_offset_x']:.4f},{p['pre_offset_z']:.4f},{p['after_offset_z']:.4f}) "
            f"euler={p['euler_deg'].tolist()} pos={p['picking_position'].tolist()}"
        )

    def is_atomic_action_complete(self) -> bool:
        """Pick 原子动作（7 个 phase）是否已完整执行完；若在纠正阶段则未完成。"""
        if self.mode == "vlm_live":
            if getattr(self, "_vlm_recovering_to_home", False):
                return False
            return self.pick_controller.is_done()
        if self.mode == "collect" and self._lift_correction_active:
            return False
        if self.mode == "collect":
            return self.pick_controller.is_done()
        return True

    def _sample_noise(self):
        """每个 episode 采样一次随机噪声。
        - noise_scale: 放大范围
        - failure_bias_ratio: 该比例 episode 使用 noise_scale
        - noise_distribution: "uniform" 均匀分布 | "edge_bias" U 形（更多边缘值，提高失败率）
        """
        if not self._noise_enabled:
            self._episode_noise = {}
            return
        r = self._noise_range
        if self._failure_bias_ratio > 0 and np.random.random() < self._failure_bias_ratio:
            scale = self._noise_scale
        else:
            scale = 1.0
        dist = getattr(self, '_noise_distribution', 'uniform')

        def sample_in_range(lo, hi):
            if dist == 'edge_bias':
                # Beta(0.5,0.5) 为 U 形：更多采样在 0 和 1 附近（即 lo 和 hi）
                u = np.random.beta(0.5, 0.5)
                return lo + (hi - lo) * u
            else:
                return np.random.uniform(lo, hi)

        def scaled_range(key):
            lo, hi = r[key][0], r[key][1]
            mid = (lo + hi) / 2
            half = (hi - lo) / 2 * scale
            return mid - half, mid + half

        self._episode_noise = {
            'pre_offset_x': sample_in_range(*scaled_range('pre_offset_x')),
            'pre_offset_z': sample_in_range(*scaled_range('pre_offset_z')),
            'after_offset_z': sample_in_range(*scaled_range('after_offset_z')),
            'euler_deg': np.array([
                sample_in_range(*scaled_range('euler_deg')),
                sample_in_range(*scaled_range('euler_deg')),
                sample_in_range(*scaled_range('euler_deg')),
            ]),
        }
        if getattr(self, '_picking_position_noise_enabled', False):
            pos_lo, pos_hi = self._picking_position_noise_range[0], self._picking_position_noise_range[1]
            pos_half = (pos_hi - pos_lo) / 2 * scale
            pos_mid = (pos_lo + pos_hi) / 2
            pos_range = (pos_mid - pos_half, pos_mid + pos_half)
            self._episode_noise['picking_position'] = np.array([
                sample_in_range(*pos_range),
                sample_in_range(*pos_range),
                sample_in_range(*pos_range),
            ])

    def reset(self):
        super().reset()
        self._early_return = False
        self._lift_correction_active = False
        self._lift_correction_steps = 0
        self._lift_success_counter = 0
        self._last_success_was_corrected = False
        if self.mode == "collect":
            self.pick_controller.reset()
            self._sample_noise()
            self._episode_properties_set = False
        elif self.mode == "vlm_live":
            # 控制器内部清 VLM 缓冲与原子 pick；整局场景/物体由外层 main → task.reset()
            self.pick_controller.reset()
            self._reset_vlm_episode_state()
        else:
            self.inference_engine.reset()
        self.initial_position = None
        self._initial_world_rotation_3x3 = None
        self._initial_object_up_world = None
        self._initial_bbox_sizes = None
        self._initial_bbox_verticality = None
        self._episode_step_idx = 0

    def _get_world_rotation_matrix_for_object(self, object_path: str) -> Optional[np.ndarray]:
        """物体根或常见子 mesh 上当前世界旋转 3×3（与仿真一致）。"""
        ou = self.object_utils
        for suffix in ("", "/mesh", "/Mesh", "/geometry"):
            p = f"{object_path}{suffix}" if suffix else object_path
            Rm = ou.get_world_rotation_matrix_3x3(p)
            if Rm is not None:
                return np.asarray(Rm, dtype=float)
        return None

    @staticmethod
    def _geodesic_rotation_angle_deg(R0: np.ndarray, R1: np.ndarray) -> float:
        """相对旋转角（度）：R0、R1 为本局初始与当前世界旋转，R_rel = R0^T R1。"""
        try:
            R_rel = np.asarray(R0, dtype=float).reshape(3, 3).T @ np.asarray(R1, dtype=float).reshape(3, 3)
            rot = R.from_matrix(R_rel)
            return float(np.degrees(rot.magnitude()))
        except Exception:
            return 0.0

    def _get_object_quat_for_tilt(self, object_path: str) -> Optional[np.ndarray]:
        """尝试根 prim 或常见子 mesh 上的旋转，用于计算物体局部「向上」在世界系中的方向。"""
        ou = self.object_utils
        for suffix in ("", "/mesh", "/Mesh", "/geometry"):
            p = f"{object_path}{suffix}" if suffix else object_path
            try:
                q = ou.get_transform_quat(p, w_first=False)
            except Exception:
                q = None
            if q is not None and len(q) == 4 and np.all(np.isfinite(q)):
                return np.asarray(q, dtype=float)
        return None

    def _object_world_up_direction(self, object_path: str) -> Optional[np.ndarray]:
        """物体局部「竖直」轴在世界系中的单位向量。优先用世界变换矩阵（与 PhysX 一致），否则回退到 xformOp 四元数。"""
        ou = self.object_utils
        for suffix in ("", "/mesh", "/Mesh", "/geometry"):
            p = f"{object_path}{suffix}" if suffix else object_path
            Rm = ou.get_world_rotation_matrix_3x3(p)
            if Rm is None:
                continue
            try:
                rot = R.from_matrix(Rm)
                up = rot.apply(self._tipped_local_up_axis)
                n = float(np.linalg.norm(up))
                if n >= 1e-9:
                    return up / n
            except Exception:
                continue
        q = self._get_object_quat_for_tilt(object_path)
        if q is None:
            return None
        try:
            rot = R.from_quat(q)
            up = rot.apply(self._tipped_local_up_axis)
            n = float(np.linalg.norm(up))
            if n < 1e-9:
                return None
            return up / n
        except Exception:
            return None

    def _try_capture_bbox_baseline(self, object_path: str) -> None:
        """世界轴对齐包围盒（含子 mesh）：竖立细长物体 sz/max(sx,sy) 较大，横躺后明显变小。"""
        try:
            sizes = self.object_utils.get_object_size(object_path=object_path)
            if sizes is None:
                return
            s = np.asarray(sizes, dtype=float).reshape(3)
            if np.any(~np.isfinite(s)):
                return
            mxy = max(float(s[0]), float(s[1]), 1e-9)
            self._initial_bbox_sizes = s.copy()
            self._initial_bbox_verticality = float(s[2]) / mxy
        except Exception:
            return

    def _maybe_capture_tipped_baseline(self, state: dict) -> None:
        """锁定本局初始 R0/up，以及（可选）世界 AABB 竖直度基线。"""
        if not self._tipped_detection_enabled:
            return
        op = state.get("object_path")
        if not op or state.get("object_position") is None:
            return
        if self._episode_step_idx <= self._tipped_baseline_delay_steps:
            return
        if self._tipped_bbox_fallback_enabled and self._initial_bbox_sizes is None:
            self._try_capture_bbox_baseline(str(op))
        if self._initial_world_rotation_3x3 is not None or self._initial_object_up_world is not None:
            return
        R0 = self._get_world_rotation_matrix_for_object(str(op))
        if R0 is not None:
            self._initial_world_rotation_3x3 = R0.copy()
            try:
                rot = R.from_matrix(R0)
                u = rot.apply(self._tipped_local_up_axis)
                nu = float(np.linalg.norm(u))
                if nu >= 1e-9:
                    self._initial_object_up_world = (u / nu).copy()
            except Exception:
                pass
            return
        up = self._object_world_up_direction(str(op))
        if up is not None:
            self._initial_object_up_world = up.copy()

    def _is_object_tipped_by_bbox(self, object_path: str) -> bool:
        """根 xform 不转时仍可用：世界 AABB 随子几何更新，竖直度比初值显著下降则判倒。"""
        if self._initial_bbox_sizes is None or self._initial_bbox_verticality is None:
            return False
        if self._initial_bbox_verticality < self._tipped_bbox_min_initial_verticality:
            return False
        try:
            sizes = self.object_utils.get_object_size(object_path=object_path)
            if sizes is None:
                return False
            s = np.asarray(sizes, dtype=float).reshape(3)
            if np.any(~np.isfinite(s)):
                return False
            mxy = max(float(s[0]), float(s[1]), 1e-9)
            r = float(s[2]) / mxy
            return r < self._initial_bbox_verticality * self._tipped_bbox_verticality_factor
        except Exception:
            return False

    def _is_object_tipped(self, state: dict) -> bool:
        """相对初始：旋转/竖直轴夹角/世界 AABB 竖直度，任一明显异常即判倒。"""
        if not self._tipped_detection_enabled:
            return False
        op = state.get("object_path")
        if not op or state.get("object_position") is None:
            return False
        has_rot = self._initial_world_rotation_3x3 is not None or self._initial_object_up_world is not None
        has_bbox = self._tipped_bbox_fallback_enabled and self._initial_bbox_sizes is not None
        if not has_rot and not has_bbox:
            return False
        lim = float(np.cos(np.radians(self._tipped_max_tilt_deg)))
        tipped = False
        if self._initial_world_rotation_3x3 is not None:
            R_curr = self._get_world_rotation_matrix_for_object(str(op))
            if R_curr is not None:
                ang = self._geodesic_rotation_angle_deg(self._initial_world_rotation_3x3, R_curr)
                tipped = tipped or (ang > self._tipped_max_tilt_deg)
        if self._initial_object_up_world is not None:
            up = self._object_world_up_direction(str(op))
            if up is not None:
                tipped = tipped or (float(np.dot(up, self._initial_object_up_world)) < lim)
        if has_bbox:
            tipped = tipped or self._is_object_tipped_by_bbox(str(op))
        return tipped

    def _check_tipped_reset_episode(self, state: dict, log_prefix: str) -> bool:
        """已建立倾倒基线且当前判定碰倒：结束本 episode，置 `reset_needed` 与 `_early_return`。

        随后 `main` 会走 **整局** `task.reset()`（场景 / 机器人 / 物体位姿与旋转由任务侧恢复），
        与 vlm_live 里「纠错时关节回退 `_vlm_recovering_to_home`」无关：后者仅动机械臂、同一局内继续。
        collect/vlm_live 下碰倒不写 HDF5。
        """
        has_rot = self._initial_world_rotation_3x3 is not None or self._initial_object_up_world is not None
        has_bbox = self._tipped_bbox_fallback_enabled and self._initial_bbox_sizes is not None
        if not has_rot and not has_bbox:
            return False
        if not self._is_object_tipped(state):
            return False
        self.reset_needed = True
        self._last_success = False
        self._early_return = True
        self._last_failure_reason = "object tipped / knocked over"
        print(f"{log_prefix} 检测到物体倾倒，本 episode 重置（不写盘）")
        return True

    def _vlm_log_physical_tilt_metrics(self, state: dict) -> None:
        """VLM 判失败但未走倾倒重置时打一行物理量，便于调参。"""
        if not self._tipped_detection_enabled:
            return
        op = state.get("object_path")
        if not op:
            return
        parts = []
        if self._initial_world_rotation_3x3 is not None:
            R_curr = self._get_world_rotation_matrix_for_object(str(op))
            if R_curr is not None:
                ang = self._geodesic_rotation_angle_deg(self._initial_world_rotation_3x3, R_curr)
                parts.append(f"根旋转角={ang:.2f}°（>{self._tipped_max_tilt_deg}° 为倒）")
        if self._tipped_bbox_fallback_enabled and self._initial_bbox_verticality is not None:
            try:
                sizes = self.object_utils.get_object_size(object_path=str(op))
                if sizes is not None:
                    s = np.asarray(sizes, dtype=float).reshape(3)
                    mxy = max(float(s[0]), float(s[1]), 1e-9)
                    r = float(s[2]) / mxy
                    thr = self._initial_bbox_verticality * self._tipped_bbox_verticality_factor
                    parts.append(
                        f"AABB竖直度 sz/max(sx,sy)={r:.3f}（初值={self._initial_bbox_verticality:.3f}，"
                        f"判倒阈<{thr:.3f}，初值需≥{self._tipped_bbox_min_initial_verticality} 才启用）"
                    )
            except Exception:
                pass
        if not parts:
            return
        print("[Pick][VLM] 物理未判为倾倒：" + " | ".join(parts))

    def step(self, state):
        self._episode_step_idx += 1
        if self.initial_position is None and state.get("object_position") is not None:
            self.initial_position = state["object_position"]
        self.state = state
        if self.mode == "collect":
            return self._step_collect(state)
        if self.mode == "vlm_live":
            return self._step_vlm_live(state)
        return self._step_infer(state)
            
    def _check_success(self):
        """物体是否被成功抬起（z 高于初始位置 10cm）。需连续多步满足才判成功，避免物理抖动误判。"""
        obj_pos = self.state.get('object_position')
        if obj_pos is None or self.initial_position is None:
            return False
        try:
            obj_z = float(obj_pos[2]) if hasattr(obj_pos, '__getitem__') and len(obj_pos) >= 3 else None
            init_z = float(self.initial_position[2]) if hasattr(self.initial_position, '__getitem__') and len(self.initial_position) >= 3 else None
        except (TypeError, IndexError):
            return False
        if obj_z is None or init_z is None:
            return False
        return obj_z > init_z + 0.1

    def _mean_params(self, params_list: list) -> Optional[dict]:
        """从成功样本列表计算参数均值，作为正确空间的参考点"""
        if not params_list:
            return None
        result = {}
        keys = ["picking_position", "pre_offset_x", "pre_offset_z", "after_offset_z", "euler_deg"]
        for k in keys:
            vals = [p.get(k) for p in params_list if p.get(k) is not None]
            if not vals:
                continue
            v0 = vals[0]
            if isinstance(v0, (list, np.ndarray)):
                arr = np.array(vals)
                result[k] = np.mean(arr, axis=0).tolist()
            else:
                result[k] = float(np.mean(vals))
        return result if result else None

    def _nearest_params(self, params_wrong: dict, params_list: list) -> Optional[dict]:
        """从成功样本中找与 params_wrong 距离最近的点"""
        if not params_list:
            return None
        def to_vec(p):
            parts = []
            if "picking_position" in p:
                parts.extend(p["picking_position"][:3])
            parts.extend([
                p.get("pre_offset_x", 0.05), p.get("pre_offset_z", 0.12),
                p.get("after_offset_z", 0.25)
            ])
            if "euler_deg" in p:
                parts.extend(p["euler_deg"][:3])
            return np.array(parts, dtype=float)
        w = to_vec(params_wrong)
        best, best_d = None, float("inf")
        for p in params_list:
            v = to_vec(p)
            d = float(np.linalg.norm(v - w))
            if d < best_d:
                best_d, best = d, p
        return best

    def _params_to_correction_gt(self, p_ref: dict, params: dict) -> dict:
        """计算 correction_gt = p_ref - params（朝向正确空间的修正量）"""
        delta = {}
        for k in ["pre_offset_x", "pre_offset_z", "after_offset_z"]:
            if k in p_ref and k in params:
                delta[k] = float(p_ref[k] - params.get(k, 0))
        if "euler_deg" in p_ref and "euler_deg" in params:
            pr = np.array(p_ref["euler_deg"][:3])
            pa = np.array(params.get("euler_deg", [0, 90, 25])[:3])
            delta["euler_deg"] = (pr - pa).tolist()
        if "picking_position" in p_ref and "picking_position" in params:
            pr = np.array(p_ref["picking_position"][:3])
            pa = np.array(params.get("picking_position", [0, 0, 0])[:3])
            delta["picking_position_delta"] = (pr - pa).tolist()
        return delta

    def _apply_correction(self, params: dict, correction: dict) -> dict:
        """params + correction -> 新 params"""
        out = dict(params)
        for k in ["pre_offset_x", "pre_offset_z", "after_offset_z"]:
            if k in correction:
                out[k] = params.get(k, 0) + correction[k]
        if "euler_deg" in correction:
            base = np.array(params.get("euler_deg", [0, 90, 25])[:3])
            out["euler_deg"] = (base + np.array(correction["euler_deg"])).tolist()
        if "picking_position_delta" in correction and "picking_position" in params:
            base = np.array(params["picking_position"][:3])
            out["picking_position"] = (base + np.array(correction["picking_position_delta"])).tolist()
        return out

    def _get_p_ref_and_correction(self, params_used: dict, object_type: str) -> tuple:
        """获取参考点 p_ref 和 correction_gt。池为空时用 baseline（-noise）。

        注意：返回的 correction_gt 中若含 picking_position_delta，来自成功池 p_ref 时并非当前帧物理真值；
        写入 HDF5 前会由 _merge_correction_gt_with_sim_picking_gt 用 object_position 覆盖。
        """
        pool = self._success_pool.get(object_type, [])
        if pool and self._p_ref_method == "nearest":
            p_ref = self._nearest_params(params_used, pool)
        elif pool:
            p_ref = self._mean_params(pool)
        else:
            p_ref = None
        if p_ref is not None:
            correction_gt = self._params_to_correction_gt(p_ref, params_used)
        else:
            correction_gt = None
        return p_ref, correction_gt

    def _overlay_sim_gt_picking_position_delta(
        self, corr: Optional[dict], params: dict, state: dict
    ) -> dict:
        """picking_position 的真值唯一对应当前仿真物体位姿；用 object_position 覆盖 delta，不用成功池均值。"""
        out = dict(corr) if corr else {}
        gt = state.get("object_position")
        if gt is not None and len(gt) >= 3 and params.get("picking_position") is not None:
            g = np.array(gt[:3], dtype=float)
            pa = np.array(params["picking_position"][:3], dtype=float)
            out["picking_position_delta"] = (g - pa).tolist()
        return out

    def _merge_correction_gt_with_sim_picking_gt(
        self, params_used: dict, correction_gt: Optional[dict], state: dict
    ) -> dict:
        """失败样本写入 HDF5 前：保证 picking_position 的纠错目标为当前帧仿真 GT（与成功池 pre/euler 分离）。"""
        base = dict(correction_gt) if correction_gt else {}
        merged = self._overlay_sim_gt_picking_position_delta(base, params_used, state)
        return self._finalize_correction_with_direction_and_steps(merged)

    @staticmethod
    def _vec_unit_and_norm(v: np.ndarray) -> Tuple[np.ndarray, float]:
        v = np.asarray(v, dtype=float).reshape(-1)
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            z = np.zeros_like(v)
            return z, 0.0
        return v / n, n

    def _clip_vec_preserving_direction(self, v: np.ndarray, max_norm: float) -> np.ndarray:
        v = np.asarray(v, dtype=float).reshape(-1)
        n = float(np.linalg.norm(v))
        if n <= max_norm or n < 1e-12:
            return v.copy()
        return v * (max_norm / n)

    def _finalize_correction_with_direction_and_steps(self, corr: Optional[dict]) -> dict:
        """在保留完整 correction 分量的前提下，增加单位方向、模长与保守一步 step（*_*_step）。

        训练/转换脚本优先用 *_step 作为「单步修正」标签，降低过冲、提高可执行性。
        """
        if not corr:
            return {}
        out = dict(corr)
        if not self._correction_enrich_unit_magnitude:
            return out
        if "picking_position_delta" in out:
            pv = np.asarray(out["picking_position_delta"][:3], dtype=float)
            u, mag = self._vec_unit_and_norm(pv)
            out["picking_position_delta_unit"] = u.tolist()
            out["picking_position_delta_magnitude"] = mag
            if self._correction_suggested_max_picking_norm is not None:
                step = self._clip_vec_preserving_direction(pv, self._correction_suggested_max_picking_norm)
                out["picking_position_delta_step"] = step.tolist()
        if "euler_deg" in out and isinstance(out["euler_deg"], (list, tuple)) and len(out["euler_deg"]) >= 3:
            ev = np.asarray(out["euler_deg"][:3], dtype=float)
            u, mag = self._vec_unit_and_norm(ev)
            out["euler_deg_unit"] = u.tolist()
            out["euler_deg_magnitude"] = mag
            if self._correction_suggested_max_euler_l2_deg is not None:
                step = self._clip_vec_preserving_direction(ev, self._correction_suggested_max_euler_l2_deg)
                out["euler_deg_step"] = step.tolist()
        if self._correction_suggested_max_offset_abs is not None:
            mx = self._correction_suggested_max_offset_abs
            for k in ("pre_offset_x", "pre_offset_z", "after_offset_z"):
                if k in out and isinstance(out[k], (int, float)):
                    v = float(out[k])
                    out[f"{k}_step"] = float(np.clip(v, -mx, mx))
        return out

    def _snapshot_object_position_for_task_props(self, state: dict) -> dict:
        """回合写入 HDF5 前：把当前仿真物体位姿一并写入 task_properties，便于离线核对与重算。

        与 _overlay_sim_gt_picking_position_delta 使用的 state['object_position'] 同源。
        """
        op = state.get("object_position")
        if op is None or len(op) < 3:
            return {}
        return {
            "object_position": [float(op[0]), float(op[1]), float(op[2])],
        }

    def _append_last_params_to_success_pool(self) -> None:
        """将本局实际使用的 params_used 记入成功池，供失败时 p_ref / correction_gt 使用。

        须在「判定成功」且仍持有 _last_params_used / _last_object_type 时调用；
        含主 pick 直接成功与 lift correction 补救成功（二者参数标签一致，均应入池）。
        """
        if not getattr(self, "_last_params_used", None) or not getattr(self, "_last_object_type", None):
            return
        obj_type = self._last_object_type
        if obj_type not in self._success_pool:
            self._success_pool[obj_type] = []
        self._success_pool[obj_type].append(dict(self._last_params_used))
        if len(self._success_pool[obj_type]) > self._success_pool_max:
            self._success_pool[obj_type] = self._success_pool[obj_type][-self._success_pool_max :]

    def _generate_correction_steps(
        self,
        params_used: dict,
        correction_gt: dict,
        object_type: str,
        state: Optional[dict] = None,
    ) -> list:
        """从单次失败生成多步 correction 样本：每步朝正确空间移动 alpha 比例。

        picking_position 每步的 delta 相对当前仿真物体位姿（state），而非成功池 p_ref。
        """
        steps = []
        alpha = self._correction_alpha
        n_steps = max(1, self._correction_steps_count)
        p_ref, _ = self._get_p_ref_and_correction(params_used, object_type)
        if p_ref is None:
            cg = self._merge_correction_gt_with_sim_picking_gt(params_used, correction_gt, state) if state else (correction_gt or {})
            steps.append({"params": dict(params_used), "correction_gt": cg})
            return steps
        params = dict(params_used)
        for _ in range(n_steps):
            corr = self._params_to_correction_gt(p_ref, params)
            if state is not None:
                corr = self._overlay_sim_gt_picking_position_delta(corr, params, state)
            if not corr:
                break
            scaled = {k: (v * alpha if isinstance(v, (int, float)) else [x * alpha for x in v])
                     for k, v in corr.items()}
            # 多步内限制单步位置增量范数，避免与 alpha 叠乘后仍过大
            if self._correction_step_max_picking_norm is not None and "picking_position_delta" in scaled:
                sd = np.asarray(scaled["picking_position_delta"], dtype=float)
                scaled["picking_position_delta"] = self._clip_vec_preserving_direction(
                    sd, self._correction_step_max_picking_norm
                ).tolist()
            scaled = self._finalize_correction_with_direction_and_steps(scaled)
            steps.append({"params": dict(params), "correction_gt": scaled})
            apply_corr = {
                k: scaled[k]
                for k in ("pre_offset_x", "pre_offset_z", "after_offset_z", "euler_deg", "picking_position_delta")
                if k in scaled
            }
            params = self._apply_correction(params, apply_corr)
        return steps

    def _init_infer_mode(self, cfg, robot):
        """
        Initializes components for inference mode.
        Creates inference engine and trajectory controller.

        Args:
            cfg: Configuration object containing model paths and settings
            robot: Robot instance to control
        """
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot
        )
        
        self.inference_engine = InferenceEngineFactory.create_inference_engine(
            cfg, self.trajectory_controller
        )
        
    def _step_collect(self, state):
        """
        Executes one step in collection mode.
        Records demonstrations and manages episode transitions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        # 若物体位置无法获取，立刻结束当前 episode 并重置（不写入数据、不保存 video）
        if state.get('object_position') is None:
            self.reset_needed = True
            self._early_return = True
            print("[Pick] 提前返回：object_position=None")
            return None, True, False

        self._maybe_capture_tipped_baseline(state)

        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
        
        if not self.pick_controller.is_done():
            # 首次进入时 _episode_noise 可能为空（reset 尚未调用），需先采样
            if self._noise_enabled and not self._episode_noise:
                self._sample_noise()
            # picking_position：基准 + 噪声（纯视觉时 correction 可修正位置误差，使错误参数朝正确空间移动）
            gt_position = np.array(state['object_position'][:3])
            picking_position = gt_position.copy()
            if self._noise_enabled and self._episode_noise.get('picking_position') is not None:
                picking_position = gt_position + self._episode_noise['picking_position']
            # 其他参数：无噪声时的默认值
            pre_offset_x = self._default_pre_offset_x
            pre_offset_z = self._default_pre_offset_z
            after_offset_z = self._default_after_offset_z
            euler_deg = self._default_euler_deg.copy()
            if self._noise_enabled:
                n = self._episode_noise
                pre_offset_x = self._default_pre_offset_x + n['pre_offset_x']
                pre_offset_z = self._default_pre_offset_z + n['pre_offset_z']
                after_offset_z = self._default_after_offset_z + n['after_offset_z']
                euler_deg = self._default_euler_deg + n['euler_deg']
            # VLM 训练数据：params_used 含实际使用的参数（含 picking_position），correction_gt 为朝向正确空间的修正量
            if not self._episode_properties_set and hasattr(self.data_collector, 'set_task_properties'):
                n = self._episode_noise if self._noise_enabled else {}
                object_type = self._vlm_object_type_for_prompt(state)
                params_used = {
                    "pre_offset_x": float(pre_offset_x),
                    "pre_offset_z": float(pre_offset_z),
                    "after_offset_z": float(after_offset_z),
                    "euler_deg": euler_deg.tolist(),
                }
                correction_gt = {
                    "pre_offset_x": -n.get('pre_offset_x', 0),
                    "pre_offset_z": -n.get('pre_offset_z', 0),
                    "after_offset_z": -n.get('after_offset_z', 0),
                    "euler_deg": (-n['euler_deg']).tolist() if 'euler_deg' in n else [0, 0, 0],
                }
                params_used["picking_position"] = picking_position.tolist()  # 始终记录，供正确空间分析
                if self._noise_enabled and 'picking_position' in n:
                    correction_gt["picking_position_delta"] = (-n['picking_position']).tolist()
                correction_gt = self._finalize_correction_with_direction_and_steps(correction_gt)
                self._last_params_used = params_used
                self._last_object_type = object_type
                self._last_baseline_correction = dict(correction_gt)
                props = {
                    "params_used": params_used,
                    "object_type": object_type,
                }
                if self._noise_enabled:
                    props["injected_noise"] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in n.items()}
                    props["correction_gt"] = correction_gt
                self.data_collector.set_task_properties(props)
                self._episode_properties_set = True
            action = self.pick_controller.forward(
                picking_position=picking_position,
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name=state['object_name'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians(euler_deg)).as_quat(),
                gripper_position=state['gripper_position'],
                pre_offset_x=pre_offset_x,
                pre_offset_z=pre_offset_z,
                after_offset_z=after_offset_z
            )
            
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            return action, False, False
        
        # Pick 已完成：先根据物理连续抬升判成败，再在未成功时做倾倒判定（写入失败样本 / lift 之前）
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if not self._last_success:
            if self._check_tipped_reset_episode(state, "[Pick][Collect]"):
                return None, True, False
        if self._last_success:
            self._append_last_params_to_success_pool()
            if hasattr(self.data_collector, 'update_task_properties'):
                _snap = self._snapshot_object_position_for_task_props(state)
                self.data_collector.update_task_properties({"is_success": True, **_snap})
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            self.reset_needed = True
            return None, True, True

        # 未成功：若已抓住但高度不足，尝试非重置纠正（继续上抬）
        obj_pos = state.get('object_position')
        grasped_but_low = (
            obj_pos is not None and self.initial_position is not None
            and obj_pos[2] > self.initial_position[2] + 0.02  # 至少抬了 2cm，视为抓住
            and obj_pos[2] <= self.initial_position[2] + 0.1   # 未达 10cm 成功线
        )
        if grasped_but_low and self._lift_correction_enabled and not self._lift_correction_active:
            self._lift_correction_active = True
            self._lift_correction_steps = 0
            self._lift_success_counter = 0
            gripper_pos = np.array(state['gripper_position'][:3])
            self._lift_correction_target = gripper_pos + np.array([0.0, 0.0, 0.1])
            self._lift_correction_orientation = R.from_euler('xyz', np.radians([0.0, 90.0, 25.0])).as_quat()

        if self._lift_correction_active:
            action = self.rmp_controller.forward(
                target_end_effector_position=self._lift_correction_target,
                target_end_effector_orientation=self._lift_correction_orientation
            )
            self._lift_correction_steps += 1
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            if self._check_success():
                self._lift_success_counter += 1
            else:
                self._lift_success_counter = 0
            if self._lift_success_counter >= self._lift_required_steps:
                self._lift_correction_active = False
                self._last_success = True
                self._last_success_was_corrected = True
                # 与主 pick 成功一致：补救成功同样代表「该组噪声参数可达成任务」，应进入成功池
                self._append_last_params_to_success_pool()
                if hasattr(self.data_collector, 'update_task_properties'):
                    _snap = self._snapshot_object_position_for_task_props(state)
                    self.data_collector.update_task_properties({
                        "is_success": True,
                        "was_lift_corrected": True,
                        "lift_correction_gt": {"after_offset_z": 0.1},
                        **_snap,
                    })
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self.reset_needed = True
                return None, True, True
            if self._lift_correction_steps >= self.REQUIRED_SUCCESS_STEPS:
                self._lift_correction_active = False
                if self._check_tipped_reset_episode(state, "[Pick][Collect]"):
                    return None, True, False
                updates = {"is_success": False, "was_lift_corrected_attempted": True}
                if hasattr(self, '_last_params_used') and self._last_params_used and hasattr(self, '_last_object_type'):
                    params_used = self._last_params_used
                    object_type = self._last_object_type
                    p_ref, correction_gt_new = self._get_p_ref_and_correction(params_used, object_type)
                    correction_gt = correction_gt_new if correction_gt_new else getattr(self, '_last_baseline_correction', {})
                    correction_gt = self._merge_correction_gt_with_sim_picking_gt(
                        params_used, correction_gt, state
                    )
                    updates["correction_gt"] = correction_gt
                    if self._correction_steps_count > 1 and self._noise_enabled:
                        correction_steps = self._generate_correction_steps(
                            params_used, correction_gt, object_type, state
                        )
                        if correction_steps:
                            updates["correction_steps"] = correction_steps
                updates.update(self._snapshot_object_position_for_task_props(state))
                if hasattr(self.data_collector, 'update_task_properties'):
                    self.data_collector.update_task_properties(updates)
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = False
                self.reset_needed = True
                return None, True, False
            return action, False, False

        # 失败样本：主 pick 失败当步已判过倾倒；此处为未走 lift 的失败写盘路径
        # 失败样本：用成功池计算 correction_gt 朝向正确空间，并生成多步 correction_steps
        if hasattr(self, '_last_params_used') and self._last_params_used and hasattr(self, '_last_object_type'):
            params_used = self._last_params_used
            object_type = self._last_object_type
            p_ref, correction_gt_new = self._get_p_ref_and_correction(params_used, object_type)
            if correction_gt_new:
                correction_gt = correction_gt_new
            else:
                correction_gt = getattr(self, '_last_baseline_correction', {})
            correction_gt = self._merge_correction_gt_with_sim_picking_gt(
                params_used, correction_gt, state
            )
            correction_steps = []
            if self._correction_steps_count > 1 and self._noise_enabled:
                correction_steps = self._generate_correction_steps(
                    params_used, correction_gt, object_type, state
                )
            updates = {"is_success": False, "correction_gt": correction_gt}
            if correction_steps:
                updates["correction_steps"] = correction_steps
            updates.update(self._snapshot_object_position_for_task_props(state))
            if hasattr(self.data_collector, 'update_task_properties'):
                self.data_collector.update_task_properties(updates)
        else:
            if hasattr(self.data_collector, 'update_task_properties'):
                _snap = self._snapshot_object_position_for_task_props(state)
                self.data_collector.update_task_properties({"is_success": False, **_snap})
        self.data_collector.write_cached_data(state['joint_positions'][:-1])
        self._last_success = False
        self.reset_needed = True
        return None, True, False

    def _step_vlm_live(self, state):
        """初始推断 → pick → VLM 任务②判成功；否则任务③纠错并重试，直到模型判成功或 terminate。

        碰倒：在「关节回退 / 原子抓取 / 抓取结束后调用 VLM 前」均做 `_check_tipped_reset_episode`（物理优先于 VLM）。
        纠错关节回退见 `_vlm_recovering_to_home`（非整局重置）。
        """
        if state.get("object_position") is None:
            self.reset_needed = True
            self._early_return = True
            print("[Pick][VLM] 提前返回：object_position=None")
            return None, True, False

        self._vlm_capture_home_if_needed(state)
        self._maybe_capture_tipped_baseline(state)

        # 以下为「同一 episode 内」纠错：仅机械臂回初始关节，不重置场景物体（与碰倒后的整局 task.reset 不同）
        if self._vlm_recovering_to_home:
            if self._check_tipped_reset_episode(state, "[Pick][VLM]"):
                return None, True, False
            self._vlm_recovery_steps += 1
            if self._vlm_recovery_steps > self._vlm_recovery_max_steps or self._vlm_close_to_home(
                state
            ):
                if self._vlm_recovery_steps > self._vlm_recovery_max_steps:
                    print(
                        f"[Pick][VLM] 回退初始姿态已达上限 {self._vlm_recovery_max_steps} 步，强制进入纠错"
                    )
                self._vlm_recovering_to_home = False
                self._vlm_recovery_steps = 0
                return self._vlm_apply_correction_after_failure(state)
            return self._vlm_action_toward_home(state), False, False

        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0

        self._maybe_buffer_vlm_frames(state)

        if not self._vlm_init_done:
            try:
                self._vlm_do_init(state)
            except Exception as e:
                print(f"[Pick][VLM] 初始推断失败: {e}")
                self._last_failure_reason = str(e)
                self.reset_needed = True
                return None, True, False

        if not self.pick_controller.is_done():
            if self._check_tipped_reset_episode(state, "[Pick][VLM]"):
                return None, True, False
            p = self._vlm_params
            picking_position = np.array(p["picking_position"], dtype=float)
            pre_offset_x = float(p["pre_offset_x"])
            pre_offset_z = float(p["pre_offset_z"])
            after_offset_z = float(p["after_offset_z"])
            euler_deg = np.array(p["euler_deg"], dtype=float)
            action = self.pick_controller.forward(
                picking_position=picking_position,
                current_joint_positions=state["joint_positions"],
                object_size=state["object_size"],
                object_name=state["object_name"],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler("xyz", np.radians(euler_deg)).as_quat(),
                gripper_position=state["gripper_position"],
                pre_offset_x=pre_offset_x,
                pre_offset_z=pre_offset_z,
                after_offset_z=after_offset_z,
            )
            return action, False, False

        # ---- 原子动作已执行完：先由 VLM 任务②判断成功与否；仅 is_success=true 才记为成功并结束 ----
        if len(self._vlm_frame_buffer) < 3:
            self._vlm_append_current_triple(state)
        if len(self._vlm_frame_buffer) < 3:
            print("[Pick][VLM] 无法调用成功判断：camera_data 缓冲不足 3 张")
            self._last_success = False
            self.reset_needed = True
            return None, True, False

        # 物理倾倒优先于 VLM：已碰倒则整局重置，不再调用成功判断 API（也不依赖 judge_api_ok）
        if self._check_tipped_reset_episode(state, "[Pick][VLM]"):
            return None, True, False

        obj = self._vlm_object_type_for_prompt(state)
        lang = self.get_language_instruction() or ""
        judge_api_ok = False
        try:
            judge = self._vlm_client.judge_atomic_pick_success(
                obj, lang, self._vlm_frame_buffer
            )
            vlm_ok = parse_json_bool(judge.get("is_success"))
            vlm_term = parse_json_bool(judge.get("terminate_correction_loop"))
            judge_api_ok = True
            print(
                f"[Pick][VLM] 模型成功判断 | 物体类型(发给 VLM): {obj} | is_success={vlm_ok} "
                f"terminate_correction_loop={vlm_term} | 本轮前纠错次数={self._vlm_correction_count}"
            )
        except Exception as e:
            print(f"[Pick][VLM] 成功判断 API 失败: {e}")
            self._last_failure_reason = str(e)
            vlm_ok = False
            vlm_term = False
            judge_api_ok = False

        if vlm_ok:
            self._last_success = True
            self.reset_needed = True
            print(
                f"[Pick][VLM] 模型判定 is_success=true，episode 成功结束（累计纠错 {self._vlm_correction_count} 次）"
            )
            return None, True, True

        if vlm_term:
            print("[Pick][VLM] 模型判定 terminate_correction_loop=true，结束本 episode（失败，不再纠错）")
            self._last_success = False
            self.reset_needed = True
            return None, True, False

        if judge_api_ok:
            self._vlm_log_physical_tilt_metrics(state)

        # 未成功：继续纠错（直到模型判成功，或达安全上限）
        if self._vlm_correction_count >= self._vlm_max_correction_attempts:
            lim = self._vlm_max_correction_attempts
            lim_s = "无限制(配置0)" if lim > 10**8 else str(lim)
            print(
                f"[Pick][VLM] 已达纠错次数安全上限 ({lim_s})，结束 episode。"
                "可在配置 vlm_live.max_correction_attempts 调大；0 表示不限制。"
            )
            self._last_success = False
            self.reset_needed = True
            return None, True, False

        if self._vlm_recover_to_home and self._vlm_home_joint_positions is not None:
            self.pick_controller.reset()
            self._vlm_recovering_to_home = True
            self._vlm_recovery_steps = 0
            print(
                "[Pick][VLM] 模型判定失败：机械臂回退至本 episode 初始关节（仅关节、同一局内；非碰倒后的整局场景重置），再请求纠错"
            )
            return self._vlm_action_toward_home(state), False, False

        if self._vlm_recover_to_home and self._vlm_home_joint_positions is None:
            print("[Pick][VLM] 未记录初始关节，跳过回退直接纠错")

        return self._vlm_apply_correction_after_failure(state)

    def _step_infer(self, state):
        """
        Executes one step in inference mode.
        Uses inference engine to process observations and generate actions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        if state.get("object_position") is None:
            self.reset_needed = True
            self._early_return = True
            return None, True, False
        self._maybe_capture_tipped_baseline(state)

        language_instruction = self.get_language_instruction()
        state['language_instruction'] = language_instruction
            
        action = self.inference_engine.step_inference(state)
        
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
            
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        # 策略一步内已更新成功计数后，若仍未判成功再判倾倒（与每步物理成败一致）
        if not self._last_success:
            if self._check_tipped_reset_episode(state, "[Pick][Infer]"):
                return None, True, False
        if self._last_success:
            self.reset_needed = True
            return action, True, True
        return action, False, False

    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').replace('  ', ' ').lower()
        self._language_instruction = f"Pick up the {object_name} from the table"
        return self._language_instruction

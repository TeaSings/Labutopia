from typing import Optional
import numpy as np

from isaacsim.core.utils.types import ArticulationAction
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.stir_controller import StirController


class StirTaskController(BaseController):
    def __init__(self, cfg, robot):
        stir_cfg = getattr(cfg, "stir", None)
        pick_cfg = getattr(stir_cfg, "pick", None)
        motion_cfg = getattr(stir_cfg, "motion", None)
        success_cfg = getattr(stir_cfg, "success", None)

        self._pick_events_dt = self._load_sequence(
            pick_cfg, "events_dt", [0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.02], expected_len=7
        )
        self._pick_position_threshold = float(self._get_cfg_value(pick_cfg, "position_threshold", 0.005))
        self._pick_object_name = str(self._get_cfg_value(pick_cfg, "object_name", "glass_rod"))
        pick_gripper_distance = self._get_cfg_value(pick_cfg, "gripper_distance", 0.005)
        self._pick_gripper_distance = None if pick_gripper_distance is None else float(pick_gripper_distance)
        self._pick_pre_offset_x = float(self._get_cfg_value(pick_cfg, "pre_offset_x", 0.1))
        self._pick_pre_offset_z = float(self._get_cfg_value(pick_cfg, "pre_offset_z", 0.12))
        self._pick_after_offset_z = float(self._get_cfg_value(pick_cfg, "after_offset_z", 0.15))
        self._pick_euler_deg = self._load_euler_deg(pick_cfg, "end_effector_euler_deg", [0.0, 90.0, 30.0])

        self._stir_events_dt = self._load_sequence(
            motion_cfg, "events_dt", [0.004, 0.004, 0.005, 0.001, 0.004], expected_len=5
        )
        self._stir_position_threshold = float(self._get_cfg_value(motion_cfg, "position_threshold", 0.005))
        self._stir_radius = float(self._get_cfg_value(motion_cfg, "stir_radius", 0.009))
        self._stir_speed = float(self._get_cfg_value(motion_cfg, "stir_speed", 3.0))
        self._stir_center_position_offset = np.asarray(
            self._load_sequence(motion_cfg, "center_position_offset", [0.0, 0.0, 0.0], expected_len=3), dtype=float
        )
        self._stir_euler_deg = self._load_euler_deg(motion_cfg, "end_effector_euler_deg", [0.0, 90.0, -10.0])

        self._success_min_height = float(self._get_cfg_value(success_cfg, "min_height", 0.85))
        self._success_max_xy_distance = float(self._get_cfg_value(success_cfg, "max_xy_distance", 0.04))
        self._infer_hold_steps = max(1, int(self._get_cfg_value(success_cfg, "hold_steps", 240)))

        noise_cfg = getattr(cfg, "noise", None)
        if noise_cfg and getattr(noise_cfg, "enabled", False):
            self._noise_enabled = True
            self._noise_scale = float(getattr(noise_cfg, "noise_scale", 1.0))
            self._failure_bias_ratio = float(getattr(noise_cfg, "failure_bias_ratio", 0.0))
            self._noise_distribution = str(getattr(noise_cfg, "noise_distribution", "uniform"))
            self._noise_range = {
                "pick_gripper_distance": list(getattr(noise_cfg, "pick_gripper_distance", [0.0, 0.003])),
                "pick_after_offset_z": list(getattr(noise_cfg, "pick_after_offset_z", [-0.03, 0.0])),
                "pick_end_effector_euler_deg": list(getattr(noise_cfg, "pick_end_effector_euler_deg", [-4.0, 4.0])),
                "stir_radius": list(getattr(noise_cfg, "stir_radius", [-0.003, 0.001])),
                "stir_speed": list(getattr(noise_cfg, "stir_speed", [-0.8, 0.4])),
                "stir_end_effector_euler_deg": list(getattr(noise_cfg, "stir_end_effector_euler_deg", [-8.0, 8.0])),
                "stir_center_offset_x": list(getattr(noise_cfg, "stir_center_offset_x", [-0.02, 0.02])),
                "stir_center_offset_y": list(getattr(noise_cfg, "stir_center_offset_y", [-0.02, 0.02])),
                "stir_center_offset_z": list(getattr(noise_cfg, "stir_center_offset_z", [-0.01, 0.01])),
            }
        else:
            self._noise_enabled = False

        self._episode_noise = {}
        self._episode_properties_set = False
        self._stir_params = None
        self._early_return = False
        super().__init__(cfg, robot)

    @staticmethod
    def _get_cfg_value(cfg_section, key, default):
        if cfg_section is None:
            return default
        value = getattr(cfg_section, key, default)
        return default if value is None else value

    @classmethod
    def _load_sequence(cls, cfg_section, key, default, expected_len=None):
        values = cls._get_cfg_value(cfg_section, key, default)
        if isinstance(values, np.ndarray):
            sequence = values.tolist()
        else:
            try:
                sequence = list(values)
            except TypeError:
                sequence = list(default)
        if expected_len is not None and len(sequence) != expected_len:
            return list(default)
        return sequence

    @classmethod
    def _load_euler_deg(cls, cfg_section, key, default):
        values = np.asarray(cls._load_sequence(cfg_section, key, default, expected_len=3), dtype=float).reshape(-1)
        if values.size != 3:
            return np.asarray(default, dtype=float)
        return values

    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)

        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            position_threshold=self._pick_position_threshold,
            events_dt=self._pick_events_dt,
        )

        self.stir_controller = StirController(
            name="stir_controller",
            cspace_controller=self.rmp_controller,
            events_dt=self._stir_events_dt,
            position_threshold=self._stir_position_threshold,
            stir_radius=self._stir_radius,
            stir_speed=self._stir_speed,
        )
        self.gripper_control.release_object()

    def _init_infer_mode(self, cfg, robot):
        super()._init_infer_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            position_threshold=self._pick_position_threshold,
            events_dt=[0.004, 0.002, 0.005, 1.0, 0.05, 0.004, 1.0],
        )
        self._noise_enabled = False
        self.use_stir_model = False
        self.frame_count = 0

    def reset(self):
        super().reset()
        self.gripper_control.release_object()
        self._episode_noise = {}
        self._episode_properties_set = False
        self._stir_params = None
        self._early_return = False
        self.pick_controller.reset(events_dt=self._pick_events_dt)
        if self.mode == "collect":
            self.stir_controller.reset(
                events_dt=self._stir_events_dt,
                position_threshold=self._stir_position_threshold,
                stir_radius=self._stir_radius,
                stir_speed=self._stir_speed,
            )
        else:
            self.inference_engine.reset()
        self.use_stir_model = False
        self.frame_count = 0

    def step(self, state):
        self.state = state
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)

    def _cache_step(self, state):
        if "camera_data" not in state:
            return
        self.data_collector.cache_step(
            camera_images=state["camera_data"],
            joint_angles=state["joint_positions"][:-1],
            language_instruction=self.get_language_instruction(),
        )

    def _sample_noise(self):
        if not self._noise_enabled:
            self._episode_noise = {}
            return

        scale = self._noise_scale if (
            self._failure_bias_ratio > 0 and np.random.random() < self._failure_bias_ratio
        ) else 1.0
        dist = getattr(self, "_noise_distribution", "uniform")

        def sample_in_range(lo, hi):
            if dist == "edge_bias":
                u = np.random.beta(0.5, 0.5)
                return lo + (hi - lo) * u
            return np.random.uniform(lo, hi)

        def scaled_range(key):
            lo, hi = self._noise_range[key][0], self._noise_range[key][1]
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * scale
            return mid - half, mid + half

        self._episode_noise = {
            "pick_gripper_distance": float(sample_in_range(*scaled_range("pick_gripper_distance"))),
            "pick_after_offset_z": float(sample_in_range(*scaled_range("pick_after_offset_z"))),
            "pick_end_effector_euler_deg": np.array(
                [sample_in_range(*scaled_range("pick_end_effector_euler_deg")) for _ in range(3)],
                dtype=float,
            ),
            "stir_radius": float(sample_in_range(*scaled_range("stir_radius"))),
            "stir_speed": float(sample_in_range(*scaled_range("stir_speed"))),
            "stir_end_effector_euler_deg": np.array(
                [sample_in_range(*scaled_range("stir_end_effector_euler_deg")) for _ in range(3)],
                dtype=float,
            ),
            "stir_center_offset_x": float(sample_in_range(*scaled_range("stir_center_offset_x"))),
            "stir_center_offset_y": float(sample_in_range(*scaled_range("stir_center_offset_y"))),
            "stir_center_offset_z": float(sample_in_range(*scaled_range("stir_center_offset_z"))),
        }

    def _snapshot_stir_state_for_task_props(self, state):
        snapshot = {}
        if state.get("object_position") is not None:
            snapshot["object_position"] = [float(x) for x in state["object_position"][:3]]
        if state.get("glass_rod_position") is not None:
            snapshot["glass_rod_position"] = [float(x) for x in state["glass_rod_position"][:3]]
        if state.get("target_position") is not None:
            snapshot["target_position"] = [float(x) for x in state["target_position"][:3]]
        sampled_target_position = state.get("sampled_target_position")
        if sampled_target_position is not None:
            snapshot["sampled_target_position"] = [float(x) for x in sampled_target_position[:3]]
        return snapshot

    def _build_stir_params(self, state):
        target_position = np.asarray(state["target_position"][:3], dtype=float).copy()
        pick_gripper_distance = self._pick_gripper_distance
        pick_after_offset_z = float(self._pick_after_offset_z)
        pick_euler_deg = self._pick_euler_deg.copy()
        stir_radius = float(self._stir_radius)
        stir_speed = float(self._stir_speed)
        stir_euler_deg = self._stir_euler_deg.copy()
        center_position_offset = self._stir_center_position_offset.copy()
        correction_gt = None

        if self._noise_enabled:
            if not self._episode_noise:
                self._sample_noise()
            n = self._episode_noise
            if pick_gripper_distance is not None:
                pick_gripper_distance += float(n["pick_gripper_distance"])
            pick_after_offset_z += float(n["pick_after_offset_z"])
            pick_euler_deg = pick_euler_deg + n["pick_end_effector_euler_deg"]
            stir_radius += float(n["stir_radius"])
            stir_speed += float(n["stir_speed"])
            stir_euler_deg = stir_euler_deg + n["stir_end_effector_euler_deg"]
            center_position_offset = center_position_offset + np.array(
                [
                    float(n["stir_center_offset_x"]),
                    float(n["stir_center_offset_y"]),
                    float(n["stir_center_offset_z"]),
                ],
                dtype=float,
            )
            correction_gt = {
                "pick_gripper_distance": -float(n["pick_gripper_distance"]),
                "pick_after_offset_z": -float(n["pick_after_offset_z"]),
                "pick_end_effector_euler_deg": (-n["pick_end_effector_euler_deg"]).tolist(),
                "stir_radius": -float(n["stir_radius"]),
                "stir_speed": -float(n["stir_speed"]),
                "stir_end_effector_euler_deg": (-n["stir_end_effector_euler_deg"]).tolist(),
                "stir_center_offset_x": -float(n["stir_center_offset_x"]),
                "stir_center_offset_y": -float(n["stir_center_offset_y"]),
                "stir_center_offset_z": -float(n["stir_center_offset_z"]),
            }

        target_position = target_position + center_position_offset
        params_used = {
            "pick_gripper_distance": None if pick_gripper_distance is None else float(pick_gripper_distance),
            "pick_after_offset_z": float(pick_after_offset_z),
            "pick_end_effector_euler_deg": pick_euler_deg.tolist(),
            "stir_radius": float(stir_radius),
            "stir_speed": float(stir_speed),
            "stir_end_effector_euler_deg": stir_euler_deg.tolist(),
            "center_position_offset": [float(x) for x in center_position_offset.tolist()],
            "center_position": [float(x) for x in target_position.tolist()],
            "success_min_height": float(self._success_min_height),
            "success_max_xy_distance": float(self._success_max_xy_distance),
        }
        return params_used, correction_gt

    def _maybe_set_stir_task_properties(self, state, params_used, correction_gt):
        if self._episode_properties_set or not hasattr(self.data_collector, "set_task_properties"):
            return
        props = {
            "action_type": "stir_liquid",
            "params_used": params_used,
            "object_type": str(state.get("object_name") or "glass_rod"),
            "source_object_name": state.get("object_name"),
            "target_object_name": state.get("target_name"),
        }
        props.update(self._snapshot_stir_state_for_task_props(state))
        if self._noise_enabled:
            props["injected_noise"] = {
                key: (value.tolist() if hasattr(value, "tolist") else value)
                for key, value in self._episode_noise.items()
            }
            props["correction_gt"] = correction_gt
        self.data_collector.set_task_properties(props)
        self._episode_properties_set = True

    def _prepare_stir_episode(self, state):
        if self._stir_params is not None:
            return self._stir_params

        params_used, correction_gt = self._build_stir_params(state)
        self._maybe_set_stir_task_properties(state, params_used, correction_gt)
        self.stir_controller.reset(
            events_dt=self._stir_events_dt,
            position_threshold=self._stir_position_threshold,
            stir_radius=params_used["stir_radius"],
            stir_speed=params_used["stir_speed"],
        )
        self._stir_params = {
            "pick_gripper_distance": params_used["pick_gripper_distance"],
            "pick_after_offset_z": params_used["pick_after_offset_z"],
            "pick_end_effector_orientation": R.from_euler(
                "xyz", np.radians(np.asarray(params_used["pick_end_effector_euler_deg"], dtype=float))
            ).as_quat(),
            "stir_end_effector_orientation": R.from_euler(
                "xyz", np.radians(np.asarray(params_used["stir_end_effector_euler_deg"], dtype=float))
            ).as_quat(),
            "center_position": np.asarray(params_used["center_position"], dtype=float),
        }
        return self._stir_params

    def _finalize_episode(self, state, success: bool):
        if self._episode_properties_set and hasattr(self.data_collector, "update_task_properties"):
            self.data_collector.update_task_properties({"is_success": bool(success)})
        self.data_collector.write_cached_data(state["joint_positions"][:-1])

    def _check_collect_success(self, state):
        object_pos = state.get("glass_rod_position")
        target_position = state.get("target_position")
        if object_pos is None:
            self._last_failure_reason = "Glass rod position unavailable"
            return False
        if target_position is None:
            self._last_failure_reason = "Target beaker position unavailable"
            return False
        object_pos = np.asarray(object_pos[:3], dtype=float)
        target_position = np.asarray(target_position[:3], dtype=float)
        if float(object_pos[2]) <= self._success_min_height:
            self._last_failure_reason = (
                f"Glass rod height too low ({float(object_pos[2]):.4f}<={self._success_min_height:.4f})"
            )
            return False
        xy_distance = float(np.linalg.norm(object_pos[:2] - target_position[:2]))
        if xy_distance >= self._success_max_xy_distance:
            self._last_failure_reason = (
                f"Glass rod too far from beaker center ({xy_distance:.4f}>={self._success_max_xy_distance:.4f})"
            )
            return False
        self._last_failure_reason = ""
        return True

    def _step_collect(self, state):
        target_position = state.get("object_position")
        if target_position is None:
            self._last_failure_reason = "Glass rod position unavailable"
            self._early_return = True
            self.data_collector.clear_cache()
            self.reset_needed = True
            self._last_success = False
            return None, True, False

        if not self.pick_controller.is_done():
            stir_params = self._prepare_stir_episode(state)
            self._cache_step(state)
            action = self.pick_controller.forward(
                picking_position=np.asarray(target_position[:3], dtype=float),
                current_joint_positions=state["joint_positions"],
                object_size=state["object_size"],
                object_name=self._pick_object_name,
                gripper_control=self.gripper_control,
                gripper_position=state["gripper_position"],
                end_effector_orientation=stir_params["pick_end_effector_orientation"],
                pre_offset_x=self._pick_pre_offset_x,
                pre_offset_z=self._pick_pre_offset_z,
                after_offset_z=stir_params["pick_after_offset_z"],
                gripper_distances=stir_params["pick_gripper_distance"],
            )
            self.gripper_control.update_grasped_object_position()
            return action, False, False

        if not self.stir_controller.is_done():
            if state.get("target_position") is None:
                self._last_failure_reason = "Target beaker position unavailable"
                self._early_return = True
                self.data_collector.clear_cache()
                self.reset_needed = True
                self._last_success = False
                return None, True, False
            stir_params = self._prepare_stir_episode(state)
            self._cache_step(state)
            action = self.stir_controller.forward(
                center_position=stir_params["center_position"].copy(),
                current_joint_positions=state["joint_positions"],
                gripper_position=state["gripper_position"],
                end_effector_orientation=stir_params["stir_end_effector_orientation"],
            )
            self.gripper_control.update_grasped_object_position()
            return action, False, False

        self.reset_needed = True
        self.gripper_control.release_object()
        self._last_success = self._check_collect_success(state)
        self._finalize_episode(state, self._last_success)
        return None, True, self._last_success

    def _step_infer(self, state):
        if not self.pick_controller.is_done():
            action = self.pick_controller.forward(
                picking_position=state["object_position"],
                current_joint_positions=state["joint_positions"],
                object_size=state["object_size"],
                object_name=self._pick_object_name,
                gripper_control=self.gripper_control,
                gripper_position=state["gripper_position"],
                end_effector_orientation=R.from_euler("xyz", np.radians(self._pick_euler_deg)).as_quat(),
                pre_offset_x=self._pick_pre_offset_x,
                pre_offset_z=self._pick_pre_offset_z,
                after_offset_z=self._pick_after_offset_z,
                gripper_distances=self._pick_gripper_distance,
            )

            final_object_position = state["glass_rod_position"]
            if final_object_position is not None and final_object_position[2] > 0.82:
                self.use_stir_model = True
            self.gripper_control.update_grasped_object_position()
            return action, False, False

        state["language_instruction"] = self.get_language_instruction()
        if self.use_stir_model:
            action = self.inference_engine.step_inference(state)
            self.gripper_control.update_grasped_object_position()
            return action, False, self._check_success()

        return ArticulationAction(), False, False

    def _check_success(self):
        if self._check_collect_success(self.state):
            self.check_success_counter += 1
            if self.check_success_counter > self._infer_hold_steps:
                self._last_success = True
                return True
        else:
            self.check_success_counter = 0
        return False

    def is_atomic_action_complete(self) -> bool:
        if self.mode != "collect":
            return True
        if self.reset_needed:
            return True
        if not self.pick_controller.is_done():
            return False
        if not self.stir_controller.is_done():
            return False
        return True

    def get_language_instruction(self) -> Optional[str]:
        self._language_instruction = "Use the glass rod to stir the liquid."
        return self._language_instruction

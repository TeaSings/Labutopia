import queue
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
from isaacsim.core.utils.types import ArticulationAction

from .atomic_actions.pick_controller import PickController
from .atomic_actions.shake_controller import ShakeController
from .base_controller import BaseController


class ShakeTaskController(BaseController):
    def __init__(self, cfg, robot):
        shake_cfg = getattr(cfg, "shake", None)
        pick_cfg = getattr(shake_cfg, "pick", None)
        motion_cfg = getattr(shake_cfg, "motion", None)
        success_cfg = getattr(shake_cfg, "success", None)

        self._pick_events_dt = self._load_sequence(
            pick_cfg, "events_dt", [0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.02], expected_len=7
        )
        self._pick_position_threshold = float(self._get_cfg_value(pick_cfg, "position_threshold", 0.01))
        self._pick_object_name = str(self._get_cfg_value(pick_cfg, "object_name", "beaker"))
        pick_gripper_distance = self._get_cfg_value(pick_cfg, "gripper_distance", None)
        self._pick_gripper_distance = None if pick_gripper_distance is None else float(pick_gripper_distance)
        self._pick_pre_offset_x = float(self._get_cfg_value(pick_cfg, "pre_offset_x", 0.05))
        self._pick_pre_offset_z = float(self._get_cfg_value(pick_cfg, "pre_offset_z", 0.05))
        self._pick_after_offset_z = float(self._get_cfg_value(pick_cfg, "after_offset_z", 0.15))
        self._pick_euler_deg = self._load_euler_deg(pick_cfg, "end_effector_euler_deg", [0.0, 90.0, 30.0])

        self._shake_events_dt = self._load_sequence(
            motion_cfg,
            "events_dt",
            [0.02, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.015],
            expected_len=10,
        )
        self._shake_distance = float(self._get_cfg_value(motion_cfg, "shake_distance", 0.10))
        self._shake_euler_deg = self._load_euler_deg(motion_cfg, "end_effector_euler_deg", [0.0, 90.0, 10.0])
        self._shake_initial_position_offset = np.asarray(
            self._load_sequence(motion_cfg, "initial_position_offset", [0.0, 0.0, 0.0], expected_len=3),
            dtype=float,
        )

        self._min_lift_height = float(self._get_cfg_value(success_cfg, "min_lift_height", 0.05))
        self._min_shake_span_xy = float(self._get_cfg_value(success_cfg, "min_shake_span_xy", 0.05))
        self._required_shake_count = max(1, int(self._get_cfg_value(success_cfg, "required_shake_count", 5)))
        self._hold_steps_required = max(1, int(self._get_cfg_value(success_cfg, "hold_steps", 60)))
        self._max_hold_xy_delta = float(self._get_cfg_value(success_cfg, "max_hold_xy_delta", 0.01))
        self._post_hold_max_steps = max(
            self._hold_steps_required,
            int(self._get_cfg_value(success_cfg, "post_hold_max_steps", self._hold_steps_required * 3)),
        )
        self._post_hold_settle_steps = max(
            0,
            int(self._get_cfg_value(success_cfg, "post_hold_settle_steps", 0)),
        )

        noise_cfg = getattr(cfg, "noise", None)
        if noise_cfg and getattr(noise_cfg, "enabled", False):
            self._noise_enabled = True
            self._noise_scale = float(getattr(noise_cfg, "noise_scale", 1.0))
            self._failure_bias_ratio = float(getattr(noise_cfg, "failure_bias_ratio", 0.0))
            self._noise_distribution = str(getattr(noise_cfg, "noise_distribution", "uniform"))
            self._noise_range = {
                "pick_gripper_distance": list(getattr(noise_cfg, "pick_gripper_distance", [-0.002, 0.002])),
                "pick_end_effector_euler_deg": list(getattr(noise_cfg, "pick_end_effector_euler_deg", [-4.0, 4.0])),
                "shake_distance": list(getattr(noise_cfg, "shake_distance", [-0.02, 0.01])),
                "shake_end_effector_euler_deg": list(getattr(noise_cfg, "shake_end_effector_euler_deg", [-8.0, 8.0])),
                "shake_initial_position_offset_x": list(
                    getattr(noise_cfg, "shake_initial_position_offset_x", [-0.01, 0.01])
                ),
                "shake_initial_position_offset_y": list(
                    getattr(noise_cfg, "shake_initial_position_offset_y", [-0.02, 0.02])
                ),
                "shake_initial_position_offset_z": list(
                    getattr(noise_cfg, "shake_initial_position_offset_z", [-0.015, 0.015])
                ),
            }
        else:
            self._noise_enabled = False

        self._initial_position = None
        self._shake_positions = []
        self._last_shake_anchor_xy = None
        self._shake_count = 0
        self._hold_positions = queue.Queue(maxsize=self._hold_steps_required)
        self._hold_step = 0
        self._shake_success = False
        self._last_hold_delta = None
        self._early_return = False
        self._awaiting_finalize = False
        self._post_hold_step = 0
        self._hold_target_position = None
        self._episode_noise = {}
        self._episode_properties_set = False
        self._shake_params = None

        if getattr(cfg, "mode", None) != "collect":
            self._noise_enabled = False

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
            events_dt=self._pick_events_dt,
            position_threshold=self._pick_position_threshold,
        )
        self.pick_controller.set_robot_position(np.asarray(self.cfg.robot.position, dtype=float))
        self.shake_controller = ShakeController(
            name="shake_controller",
            cspace_controller=self.rmp_controller,
            events_dt=self._shake_events_dt,
            shake_distance=self._shake_distance,
        )

    def _init_infer_mode(self, cfg, robot):
        super()._init_infer_mode(cfg, robot)

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
            "pick_end_effector_euler_deg": np.array(
                [sample_in_range(*scaled_range("pick_end_effector_euler_deg")) for _ in range(3)],
                dtype=float,
            ),
            "shake_distance": float(sample_in_range(*scaled_range("shake_distance"))),
            "shake_end_effector_euler_deg": np.array(
                [sample_in_range(*scaled_range("shake_end_effector_euler_deg")) for _ in range(3)],
                dtype=float,
            ),
            "shake_initial_position_offset_x": float(sample_in_range(*scaled_range("shake_initial_position_offset_x"))),
            "shake_initial_position_offset_y": float(sample_in_range(*scaled_range("shake_initial_position_offset_y"))),
            "shake_initial_position_offset_z": float(sample_in_range(*scaled_range("shake_initial_position_offset_z"))),
        }

    def _reset_success_trackers(self):
        self._initial_position = None
        self._shake_positions = []
        self._last_shake_anchor_xy = None
        self._shake_count = 0
        self._hold_positions = queue.Queue(maxsize=self._hold_steps_required)
        self._hold_step = 0
        self._shake_success = False
        self._last_hold_delta = None
        self._early_return = False
        self._awaiting_finalize = False
        self._post_hold_step = 0
        self._hold_target_position = None
        self._episode_noise = {}
        self._episode_properties_set = False
        self._shake_params = None

    def reset(self):
        super().reset()
        self._reset_success_trackers()
        self.gripper_control.release_object()
        if self.mode == "collect":
            self.pick_controller.reset(events_dt=self._pick_events_dt)
            self.pick_controller.set_robot_position(np.asarray(self.cfg.robot.position, dtype=float))
            self.shake_controller.reset(events_dt=self._shake_events_dt, shake_distance=self._shake_distance)
        else:
            self.inference_engine.reset()

    def step(self, state):
        self.state = state
        if self._initial_position is None and state.get("object_position") is not None:
            self._initial_position = np.asarray(state["object_position"][:3], dtype=float).copy()
        if self.mode == "collect":
            return self._step_collect(state)
        return self._step_infer(state)

    def _cache_step(self, state):
        if "camera_data" not in state:
            return
        self.data_collector.cache_step(
            camera_images=state["camera_data"],
            joint_angles=state["joint_positions"][:-1],
            language_instruction=self.get_language_instruction(),
        )

    def _finalize_episode(self, state, success: bool):
        if self._episode_properties_set and hasattr(self.data_collector, "update_task_properties"):
            self.data_collector.update_task_properties({"is_success": bool(success)})
        self.data_collector.write_cached_data(state["joint_positions"][:-1])

    def _snapshot_shake_state_for_task_props(self, state):
        snapshot = {}
        if state.get("object_position") is not None:
            snapshot["object_position"] = [float(x) for x in state["object_position"][:3]]
        sampled_object_position = state.get("sampled_object_position")
        if sampled_object_position is not None:
            snapshot["sampled_object_position"] = [float(x) for x in sampled_object_position[:3]]
        return snapshot

    def _build_shake_params(self, state):
        pick_gripper_distance = self._pick_gripper_distance
        pick_euler_deg = self._pick_euler_deg.copy()
        shake_distance = float(self._shake_distance)
        shake_euler_deg = self._shake_euler_deg.copy()
        shake_initial_position_offset = self._shake_initial_position_offset.copy()
        correction_gt = None

        if self._noise_enabled:
            if not self._episode_noise:
                self._sample_noise()
            n = self._episode_noise
            if pick_gripper_distance is not None:
                pick_gripper_distance += float(n["pick_gripper_distance"])
            pick_euler_deg = pick_euler_deg + n["pick_end_effector_euler_deg"]
            shake_distance += float(n["shake_distance"])
            shake_euler_deg = shake_euler_deg + n["shake_end_effector_euler_deg"]
            shake_initial_position_offset = shake_initial_position_offset + np.array(
                [
                    float(n["shake_initial_position_offset_x"]),
                    float(n["shake_initial_position_offset_y"]),
                    float(n["shake_initial_position_offset_z"]),
                ],
                dtype=float,
            )
            correction_gt = {
                "pick_gripper_distance": -float(n["pick_gripper_distance"]),
                "pick_end_effector_euler_deg": (-n["pick_end_effector_euler_deg"]).tolist(),
                "shake_distance": -float(n["shake_distance"]),
                "shake_end_effector_euler_deg": (-n["shake_end_effector_euler_deg"]).tolist(),
                "shake_initial_position_offset_x": -float(n["shake_initial_position_offset_x"]),
                "shake_initial_position_offset_y": -float(n["shake_initial_position_offset_y"]),
                "shake_initial_position_offset_z": -float(n["shake_initial_position_offset_z"]),
            }

        params_used = {
            "pick_gripper_distance": None if pick_gripper_distance is None else float(pick_gripper_distance),
            "pick_end_effector_euler_deg": pick_euler_deg.tolist(),
            "shake_distance": float(shake_distance),
            "shake_end_effector_euler_deg": shake_euler_deg.tolist(),
            "shake_initial_position_offset": [float(x) for x in shake_initial_position_offset.tolist()],
        }
        return params_used, correction_gt

    def _maybe_set_shake_task_properties(self, state, params_used, correction_gt):
        if self._episode_properties_set or not hasattr(self.data_collector, "set_task_properties"):
            return
        props = {
            "action_type": "shake_container",
            "params_used": params_used,
            "object_type": str(state.get("object_category") or state.get("object_name") or "beaker"),
            "source_object_name": state.get("object_name"),
        }
        props.update(self._snapshot_shake_state_for_task_props(state))
        if self._noise_enabled:
            props["injected_noise"] = {
                key: (value.tolist() if hasattr(value, "tolist") else value)
                for key, value in self._episode_noise.items()
            }
            props["correction_gt"] = correction_gt
        self.data_collector.set_task_properties(props)
        self._episode_properties_set = True

    def _prepare_shake_episode(self, state):
        if self._shake_params is not None:
            return self._shake_params
        params_used, correction_gt = self._build_shake_params(state)
        self._maybe_set_shake_task_properties(state, params_used, correction_gt)
        self.shake_controller.reset(events_dt=self._shake_events_dt, shake_distance=params_used["shake_distance"])
        self._shake_params = {
            "pick_gripper_distance": params_used["pick_gripper_distance"],
            "pick_end_effector_orientation": R.from_euler(
                "xyz", np.radians(np.asarray(params_used["pick_end_effector_euler_deg"], dtype=float))
            ).as_quat(),
            "shake_distance": float(params_used["shake_distance"]),
            "shake_end_effector_orientation": R.from_euler(
                "xyz", np.radians(np.asarray(params_used["shake_end_effector_euler_deg"], dtype=float))
            ).as_quat(),
            "shake_initial_position_offset": np.asarray(params_used["shake_initial_position_offset"], dtype=float),
        }
        return self._shake_params

    def _pick_target_name(self, state):
        configured_name = str(self._pick_object_name or "").strip()
        if configured_name:
            return configured_name
        return str(state.get("object_name") or state.get("object_category") or "beaker")

    def _shake_anchor_position(self, state, offset=None):
        gripper_position = state.get("gripper_position")
        if gripper_position is None:
            return None
        base_offset = self._shake_initial_position_offset if offset is None else np.asarray(offset, dtype=float)
        return np.asarray(gripper_position[:3], dtype=float) + base_offset

    def _update_shake_count(self, object_position: np.ndarray) -> bool:
        if self._shake_count >= self._required_shake_count:
            return True
        xy = object_position[:2]
        if self._last_shake_anchor_xy is None:
            self._last_shake_anchor_xy = np.asarray(xy, dtype=float).copy()
            return False
        current_xy = np.asarray(xy, dtype=float)
        dist = float(np.linalg.norm(current_xy - self._last_shake_anchor_xy))
        if dist >= self._min_shake_span_xy:
            self._shake_count += 1
            self._last_shake_anchor_xy = current_xy.copy()
        return self._shake_count >= self._required_shake_count

    def _update_hold_success(self, object_position: np.ndarray) -> bool:
        if self._hold_positions.full():
            self._hold_positions.get()
            self._hold_step = max(0, self._hold_step - 1)
        self._hold_positions.put(object_position[:2].copy())
        self._hold_step += 1

        if self._hold_step < self._hold_steps_required:
            self._last_failure_reason = (
                f"Hold phase too short ({self._hold_step}/{self._hold_steps_required})"
            )
            return False

        arr = np.asarray(list(self._hold_positions.queue), dtype=float)
        delta = arr.max(axis=0) - arr.min(axis=0)
        self._last_hold_delta = delta
        if np.all(delta <= self._max_hold_xy_delta):
            self._shake_success = True
            self._last_failure_reason = ""
            return True

        self._last_failure_reason = (
            "Beaker not stabilized after shake "
            f"(delta_x={delta[0]:.4f}, delta_y={delta[1]:.4f}, "
            f"threshold={self._max_hold_xy_delta:.4f})"
        )
        return False

    def _step_collect(self, state):
        target_position = state.get("object_position")
        if target_position is None:
            self._last_failure_reason = "Target beaker position unavailable"
            self._early_return = True
            self.data_collector.clear_cache()
            self.reset_needed = True
            self._last_success = False
            return None, True, False

        if not self.pick_controller.is_done():
            self._cache_step(state)
            shake_params = self._prepare_shake_episode(state)
            action = self.pick_controller.forward(
                picking_position=np.asarray(target_position[:3], dtype=float),
                current_joint_positions=state["joint_positions"],
                object_size=state["object_size"],
                object_name=self._pick_target_name(state),
                gripper_control=self.gripper_control,
                gripper_position=state["gripper_position"],
                end_effector_orientation=shake_params["pick_end_effector_orientation"],
                pre_offset_x=self._pick_pre_offset_x,
                pre_offset_z=self._pick_pre_offset_z,
                after_offset_z=self._pick_after_offset_z,
                gripper_distances=shake_params["pick_gripper_distance"],
            )
            return action, False, False

        if not self.shake_controller.is_done():
            shake_params = self._prepare_shake_episode(state)
            shake_anchor = self._shake_anchor_position(state, offset=shake_params["shake_initial_position_offset"])
            if shake_anchor is None:
                self._last_failure_reason = "Gripper position unavailable for shake phase"
                self._early_return = True
                self.data_collector.clear_cache()
                self.reset_needed = True
                self._last_success = False
                return None, True, False
            self._hold_target_position = np.asarray(shake_anchor, dtype=float)
            self._cache_step(state)
            action = self.shake_controller.forward(
                current_joint_positions=state["joint_positions"],
                end_effector_orientation=shake_params["shake_end_effector_orientation"],
                initial_position=shake_anchor,
            )
            if self.shake_controller.is_done():
                self._awaiting_finalize = True
            self._last_success = False
            self._evaluate_progress(state, allow_post_hold=False)
            return action, False, False

        self._awaiting_finalize = False
        self._cache_step(state)
        self._post_hold_step += 1
        self._last_success = self._evaluate_progress(state, allow_post_hold=True)
        if self._last_success:
            self._finalize_episode(state, True)
            self.reset_needed = True
            return None, True, True
        if self._post_hold_step < self._post_hold_max_steps:
            if self._hold_target_position is None:
                shake_params = self._prepare_shake_episode(state)
                hold_target = self._shake_anchor_position(state, offset=shake_params["shake_initial_position_offset"])
                if hold_target is not None:
                    self._hold_target_position = np.asarray(hold_target, dtype=float)
            if self._hold_target_position is None:
                return ArticulationAction(), False, False
            shake_params = self._prepare_shake_episode(state)
            hold_action = self.rmp_controller.forward(
                target_end_effector_position=self._hold_target_position,
                target_end_effector_orientation=shake_params["shake_end_effector_orientation"],
            )
            return hold_action, False, False

        if not self._last_failure_reason:
            self._last_failure_reason = "Shake post-hold phase exceeded limit without stabilizing"
        self._finalize_episode(state, False)
        self.reset_needed = True
        return None, True, False

    def _step_infer(self, state):
        state["language_instruction"] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)
        self._last_success = self.is_success()
        if self._last_success:
            self.reset_needed = True
            return action, True, True
        return action, False, False

    def is_success(self):
        return self._evaluate_progress(self.state, allow_post_hold=True)

    def _evaluate_progress(self, state, allow_post_hold: bool) -> bool:
        if self._initial_position is None:
            self._last_failure_reason = "Initial beaker position unavailable"
            return False

        object_position = state.get("object_position")
        if object_position is None:
            self._last_failure_reason = "Current beaker position unavailable"
            return False

        object_position = np.asarray(object_position[:3], dtype=float)
        height_diff = float(object_position[2] - self._initial_position[2])
        if height_diff < self._min_lift_height:
            self._last_failure_reason = (
                f"Beaker lift height too low ({height_diff:.4f}<{self._min_lift_height:.4f})"
            )
            return False

        if not self._update_shake_count(object_position):
            if self._shake_count < self._required_shake_count:
                self._last_failure_reason = (
                    f"Shake count too low ({self._shake_count}/{self._required_shake_count})"
                )
            return False

        if not allow_post_hold:
            self._last_failure_reason = (
                f"Waiting for post-shake stabilization ({self._shake_count}/{self._required_shake_count})"
            )
            return False

        if self._shake_success:
            self._last_failure_reason = ""
            return True

        if self._post_hold_step <= self._post_hold_settle_steps:
            self._last_failure_reason = (
                f"Waiting for post-shake settling ({self._post_hold_step}/{self._post_hold_settle_steps})"
            )
            return False

        return self._update_hold_success(object_position)

    def is_atomic_action_complete(self) -> bool:
        if self.mode != "collect":
            return True
        if self.reset_needed:
            return True
        if not self.pick_controller.is_done():
            return False
        if not self.shake_controller.is_done():
            return False
        if self._awaiting_finalize or self._post_hold_step < self._post_hold_max_steps:
            return False
        return True

    def get_language_instruction(self) -> Optional[str]:
        if self._language_instruction is None:
            return "Pick up the container and shake it"
        return self._language_instruction

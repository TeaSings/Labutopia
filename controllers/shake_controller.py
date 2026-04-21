import queue
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

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

        self._initial_position = None
        self._shake_positions = []
        self._shake_count = 0
        self._hold_positions = queue.Queue(maxsize=self._hold_steps_required)
        self._hold_step = 0
        self._shake_success = False
        self._last_hold_delta = None
        self._early_return = False
        self._awaiting_finalize = False

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

    def _reset_success_trackers(self):
        self._initial_position = None
        self._shake_positions = []
        self._shake_count = 0
        self._hold_positions = queue.Queue(maxsize=self._hold_steps_required)
        self._hold_step = 0
        self._shake_success = False
        self._last_hold_delta = None
        self._early_return = False
        self._awaiting_finalize = False

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

    def _finalize_episode(self, state):
        self.data_collector.write_cached_data(state["joint_positions"][:-1])

    def _pick_target_name(self, state):
        configured_name = str(self._pick_object_name or "").strip()
        if configured_name:
            return configured_name
        return str(state.get("object_name") or state.get("object_category") or "beaker")

    def _shake_anchor_position(self, state):
        gripper_position = state.get("gripper_position")
        if gripper_position is None:
            return None
        return np.asarray(gripper_position[:3], dtype=float) + self._shake_initial_position_offset

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
            action = self.pick_controller.forward(
                picking_position=np.asarray(target_position[:3], dtype=float),
                current_joint_positions=state["joint_positions"],
                object_size=state["object_size"],
                object_name=self._pick_target_name(state),
                gripper_control=self.gripper_control,
                gripper_position=state["gripper_position"],
                end_effector_orientation=R.from_euler("xyz", np.radians(self._pick_euler_deg)).as_quat(),
                pre_offset_x=self._pick_pre_offset_x,
                pre_offset_z=self._pick_pre_offset_z,
                after_offset_z=self._pick_after_offset_z,
            )
            return action, False, False

        if not self.shake_controller.is_done():
            shake_anchor = self._shake_anchor_position(state)
            if shake_anchor is None:
                self._last_failure_reason = "Gripper position unavailable for shake phase"
                self._early_return = True
                self.data_collector.clear_cache()
                self.reset_needed = True
                self._last_success = False
                return None, True, False
            self._cache_step(state)
            action = self.shake_controller.forward(
                current_joint_positions=state["joint_positions"],
                end_effector_orientation=R.from_euler("xyz", np.radians(self._shake_euler_deg)).as_quat(),
                initial_position=shake_anchor,
            )
            if self.shake_controller.is_done():
                self._awaiting_finalize = True
            return action, False, self.is_success()

        self._awaiting_finalize = False
        self._last_success = self.is_success()
        self._finalize_episode(state)
        self.reset_needed = True
        if self._last_success:
            return None, True, True
        if not self._last_failure_reason:
            self._last_failure_reason = "Shake motion completed but success criteria were not met"
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
        if self._initial_position is None:
            self._last_failure_reason = "Initial beaker position unavailable"
            return False

        object_position = self.state.get("object_position")
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

        if self._shake_count < self._required_shake_count:
            xy = object_position[:2]
            self._shake_positions.append(xy)
            if len(self._shake_positions) > 1:
                start_xy = np.asarray(self._shake_positions[0], dtype=float)
                end_xy = np.asarray(self._shake_positions[-1], dtype=float)
                dist = float(np.linalg.norm(end_xy - start_xy))
                if dist >= self._min_shake_span_xy:
                    self._shake_count += 1
                    self._shake_positions = []
            if self._shake_count < self._required_shake_count:
                self._last_failure_reason = (
                    f"Shake count too low ({self._shake_count}/{self._required_shake_count})"
                )
                return False

        if self._shake_success:
            self._last_failure_reason = ""
            return True

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

    def is_atomic_action_complete(self) -> bool:
        if self.mode != "collect":
            return True
        if self.reset_needed:
            return True
        if self._awaiting_finalize:
            return False
        return self.pick_controller.is_done() and self.shake_controller.is_done()

    def get_language_instruction(self) -> Optional[str]:
        if self._language_instruction is None:
            return "Pick up the container and shake it"
        return self._language_instruction

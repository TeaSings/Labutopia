import re
from typing import Optional

import numpy as np
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from robots.franka.rmpflow_controller import RMPFlowController

from controllers.atomic_actions.close_controller import CloseController
from controllers.atomic_actions.open_controller import OpenController
from .base_controller import BaseController
from .inference_engines.inference_engine_factory import InferenceEngineFactory
from .robot_controllers.trajectory_controller import FrankaTrajectoryController


class CloseTaskController(BaseController):
    """Controller for closing doors and drawers in collect or infer mode."""

    def __init__(self, cfg, robot):
        self.operate_type = str(cfg.task.get("operate_type", "door"))
        close_cfg = getattr(cfg, "close", None)
        open_cfg = getattr(cfg, "open", None)
        task_cfg = getattr(cfg, "task", None)
        self._drawer_body_path_by_object = {}
        if self.operate_type == "drawer":
            for obj_cfg in list(getattr(task_cfg, "obj_paths", []) or []):
                obj_path = self._get_cfg_value(obj_cfg, "path", None)
                if obj_path is None:
                    continue
                drawer_body_path = self._get_cfg_value(obj_cfg, "drawer_body_path", f"{obj_path}/drawer_top")
                self._drawer_body_path_by_object[str(obj_path)] = str(drawer_body_path)

        self._close_push_distance = float(self._get_cfg_value(close_cfg, "push_distance", 0.15))
        close_euler_deg = self._load_euler_deg(
            close_cfg,
            "end_effector_euler_deg",
            [350.0, 90.0, 25.0] if self.operate_type == "door" else [90.0, 90.0, 0.0],
        )
        self._close_end_effector_orientation = euler_angles_to_quats(
            close_euler_deg,
            degrees=True,
            extrinsic=False,
        )

        default_warmup_gripper = 0.023 if self.operate_type == "door" else 0.01
        self._warmup_close_gripper_distance = float(
            self._get_cfg_value(open_cfg, "close_gripper_distance", default_warmup_gripper)
        )
        self._bootstrap_open_with_controller = bool(
            self.operate_type == "drawer"
            and self._get_cfg_value(task_cfg, "bootstrap_open_with_controller", True)
        )
        self._bootstrap_open_target_distance = float(
            self._get_cfg_value(task_cfg, "initial_drawer_open_distance", 0.16)
        )
        self._warmup_open_events_dt = self._load_sequence(
            open_cfg,
            "events_dt",
            [0.0025, 0.005, 0.08, 0.004, 0.05, 0.05, 0.01, 0.004],
            expected_len=8,
        )
        self._warmup_open_position_threshold = float(
            self._get_cfg_value(open_cfg, "position_threshold", 0.01)
        )
        self._warmup_stage0_offset_x = float(self._get_cfg_value(open_cfg, "stage0_offset_x", 0.08))
        self._warmup_stage1_offset_x = float(self._get_cfg_value(open_cfg, "stage1_offset_x", 0.015))
        self._warmup_retreat_offset_x = float(self._get_cfg_value(open_cfg, "retreat_offset_x", 0.06))
        self._warmup_retreat_offset_y = float(self._get_cfg_value(open_cfg, "retreat_offset_y", 0.04))
        self._warmup_drawer_pull_offset_x = float(
            self._get_cfg_value(open_cfg, "drawer_pull_offset_x", 0.04)
        )
        self._warmup_drawer_retreat_offset_x = float(
            self._get_cfg_value(open_cfg, "drawer_retreat_offset_x", 0.12)
        )
        self._warmup_drawer_retreat_offset_z = float(
            self._get_cfg_value(open_cfg, "drawer_retreat_offset_z", 0.06)
        )
        warmup_open_euler_deg = self._load_euler_deg(open_cfg, "end_effector_euler_deg", [90.0, 90.0, 0.0])
        self._warmup_open_end_effector_orientation = euler_angles_to_quats(
            warmup_open_euler_deg,
            degrees=True,
            extrinsic=False,
        )

        super().__init__(cfg, robot)
        self.initial_handle_position = None
        self._collect_phase = "close"
        self._warmup_initial_handle_position = None
        self._warmup_open_target_reached = False
        self._warmup_ready_handle_position = None

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

        if self._bootstrap_open_with_controller:
            self.warmup_open_controller = OpenController(
                name="warmup_open_controller",
                cspace_controller=RMPFlowController(
                    name="warmup_open_target_follower_controller",
                    robot_articulation=robot,
                ),
                gripper=robot.gripper,
                events_dt=self._warmup_open_events_dt,
                furniture_type="drawer",
                position_threshold=self._warmup_open_position_threshold,
                stage0_offset_x=self._warmup_stage0_offset_x,
                stage1_offset_x=self._warmup_stage1_offset_x,
                retreat_offset_x=self._warmup_retreat_offset_x,
                retreat_offset_y=self._warmup_retreat_offset_y,
                drawer_pull_offset_x=self._warmup_drawer_pull_offset_x,
                drawer_retreat_offset_x=self._warmup_drawer_retreat_offset_x,
                drawer_retreat_offset_z=self._warmup_drawer_retreat_offset_z,
            )

        self.close_controller = CloseController(
            name="close_controller",
            cspace_controller=RMPFlowController(
                name="target_follower_controller",
                robot_articulation=robot,
            ),
            gripper=robot.gripper,
            furniture_type=self.operate_type,
            door_open_direction="clockwise",
        )

    def _init_infer_mode(self, cfg, robot):
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot,
        )

        self.inference_engine = InferenceEngineFactory.create_inference_engine(
            cfg, self.trajectory_controller
        )

    def reset(self):
        super().reset()
        self._early_return = False
        self.initial_handle_position = None
        self._warmup_initial_handle_position = None
        self._warmup_open_target_reached = False
        self._warmup_ready_handle_position = None
        if self.mode == "collect":
            self._collect_phase = "warmup_open" if self._bootstrap_open_with_controller else "close"
            if self._bootstrap_open_with_controller:
                self.warmup_open_controller.reset(events_dt=self._warmup_open_events_dt)
            self.close_controller.reset()
        else:
            self.inference_engine.reset()

    def is_atomic_action_complete(self) -> bool:
        if self.mode != "collect":
            return True
        if self._collect_phase == "warmup_open" and self._bootstrap_open_with_controller:
            if not self.warmup_open_controller.is_done():
                return False
        if self._collect_phase == "close" and not self.close_controller.is_done():
            return False
        return bool(self.reset_needed)

    def step(self, state):
        self.state = state
        if self.mode == "collect":
            if self._collect_phase == "warmup_open":
                return self._step_collect_warmup_open(state)
            if self.initial_handle_position is None:
                self.initial_handle_position = np.asarray(state["object_position"], dtype=float)
            return self._step_collect(state)

        if self.initial_handle_position is None:
            self.initial_handle_position = np.asarray(state["object_position"], dtype=float)
        return self._step_infer(state)

    def _check_warmup_open_success(self, state):
        if self._warmup_initial_handle_position is None:
            return False

        current_pos = np.asarray(state["object_position"], dtype=float)
        gripper_position = np.asarray(state["gripper_position"], dtype=float)
        handle_move_distance = np.linalg.norm(current_pos - self._warmup_initial_handle_position)
        gripper_to_object_distance = np.linalg.norm(gripper_position - current_pos)
        target_distance = max(0.08, self._bootstrap_open_target_distance * 0.75)
        return handle_move_distance >= target_distance and gripper_to_object_distance > 0.04

    def _is_drawer_already_open_enough(self, state):
        if self.operate_type != "drawer":
            return False

        object_path = str(state.get("object_path", "") or "")
        drawer_body_path = self._drawer_body_path_by_object.get(object_path)
        if not drawer_body_path:
            return False

        local_translation = self.object_utils.get_local_translation(drawer_body_path)
        if local_translation is None:
            return False

        open_distance = abs(float(local_translation[0]))
        return open_distance >= max(0.08, self._bootstrap_open_target_distance * 0.75)

    def _step_collect_warmup_open(self, state):
        if self._warmup_initial_handle_position is None:
            self._warmup_initial_handle_position = np.asarray(state["object_position"], dtype=float)
            if self._is_drawer_already_open_enough(state):
                self._collect_phase = "close"
                self.initial_handle_position = np.asarray(state["object_position"], dtype=float)
                self.close_controller.reset()
                self.check_success_counter = 0
                print(
                    "[CloseDrawerWarmup] "
                    f"drawer already open enough; skipping warmup at {self.initial_handle_position.tolist()}"
                )
                return None, False, False

        if self._check_warmup_open_success(state):
            self._warmup_open_target_reached = True
            self._warmup_ready_handle_position = np.asarray(state["object_position"], dtype=float)

        if self.warmup_open_controller.is_done():
            if self._warmup_open_target_reached:
                self._collect_phase = "close"
                self.initial_handle_position = np.asarray(
                    self._warmup_ready_handle_position
                    if self._warmup_ready_handle_position is not None
                    else state["object_position"],
                    dtype=float,
                )
                self.close_controller.reset()
                self.check_success_counter = 0
                print(
                    "[CloseDrawerWarmup] "
                    f"opened drawer to start close phase at {self.initial_handle_position.tolist()}"
                )
                return None, False, False

            print("[CloseDrawerWarmup] Failed to physically open drawer before close phase")
            self._last_failure_reason = "Warmup open failed before close phase"
            self._last_success = False
            self.reset_needed = True
            self._early_return = True
            return None, True, False

        action = self.warmup_open_controller.forward(
            handle_position=state["object_position"],
            current_joint_positions=state["joint_positions"],
            gripper_position=state["gripper_position"],
            end_effector_orientation=self._warmup_open_end_effector_orientation,
            close_gripper_distance=float(
                state.get("close_gripper_distance", self._warmup_close_gripper_distance)
            ),
        )
        return action, False, False

    def _step_collect(self, state):
        if not self.close_controller.is_done():
            if self.operate_type == "door":
                action = self.close_controller.forward(
                    handle_position=state["object_position"],
                    current_joint_positions=state["joint_positions"],
                    revolute_joint_position=state["revolute_joint_position"],
                    gripper_position=state["gripper_position"],
                    end_effector_orientation=self._close_end_effector_orientation,
                )
            else:
                action = self.close_controller.forward(
                    handle_position=state["object_position"],
                    current_joint_positions=state["joint_positions"],
                    gripper_position=state["gripper_position"],
                    end_effector_orientation=self._close_end_effector_orientation,
                    push_distance=self._close_push_distance,
                )

            if "camera_data" in state:
                self.data_collector.cache_step(
                    camera_images=state["camera_data"],
                    joint_angles=state["joint_positions"][:-1],
                    language_instruction=self.get_language_instruction(),
                )

            if self._check_success(state):
                self.check_success_counter += 1
            else:
                self.check_success_counter = 0

            return action, False, False

        success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if success:
            print("Task success!")
        else:
            print("Task failed!")
            self.print_failure_reason()
        return self._finalize_collect_episode(state, success)

    def _finalize_collect_episode(self, state, is_success):
        self._early_return = False
        if self._collect_phase == "close":
            self.data_collector.write_cached_data(state["joint_positions"][:-1])
        else:
            self.data_collector.clear_cache()

        self._last_success = bool(is_success)
        self.reset_needed = True
        return None, True, bool(is_success)

    def _step_infer(self, state):
        language_instruction = self.get_language_instruction()
        if language_instruction is not None:
            state["language_instruction"] = language_instruction
        else:
            state["language_instruction"] = f"Close the {self.operate_type} of the object"

        action = self.inference_engine.step_inference(state)

        if self._check_success(state):
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0

        success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if success:
            print("Task success!")
            self._last_success = True
            self.reset_needed = True
            return None, True, True

        return action, False, False

    def _check_success(self, state):
        current_pos = np.asarray(state["object_position"], dtype=float)
        gripper_position = np.asarray(state["gripper_position"], dtype=float)
        handle_move_distance = np.linalg.norm(current_pos - self.initial_handle_position)
        gripper_to_object_distance = np.linalg.norm(gripper_position - current_pos)

        if self.operate_type == "drawer":
            handle_moved_enough = handle_move_distance > max(0.13, self._close_push_distance * 0.85)
            gripper_far_enough = gripper_to_object_distance > 0.04
        else:
            handle_moved_enough = current_pos[0] - self.initial_handle_position[0] > 0.08
            gripper_far_enough = gripper_to_object_distance > 0.08

        success = handle_moved_enough and gripper_far_enough
        if success:
            self._last_failure_reason = ""
            return True

        if not handle_moved_enough and not gripper_far_enough:
            self._last_failure_reason = (
                f"Handle moved distance too short ({handle_move_distance:.4f}) and "
                f"gripper too close to object ({gripper_to_object_distance:.4f})"
            )
        elif not handle_moved_enough:
            self._last_failure_reason = f"Handle moved distance too short ({handle_move_distance:.4f})"
        else:
            self._last_failure_reason = f"Gripper too close to object ({gripper_to_object_distance:.4f})"
        return False

    def get_language_instruction(self) -> Optional[str]:
        object_name = re.sub(r"\d+", "", self.state["object_name"]).replace("_", " ").lower()
        self._language_instruction = f"Close the {self.operate_type} of the {object_name}"
        return self._language_instruction

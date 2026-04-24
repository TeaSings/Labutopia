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
        self._close_events_dt = self._load_sequence(
            close_cfg,
            "events_dt",
            [0.0005, 0.002, 0.03, 0.012] if self.operate_type == "drawer" else [0.0025, 0.005, 0.005],
            expected_len=4 if self.operate_type == "drawer" else 3,
        )
        self._post_close_hold_steps = int(self._get_cfg_value(close_cfg, "post_hold_steps", 90))
        self._drawer_close_approach_offset_x = float(
            self._get_cfg_value(close_cfg, "drawer_approach_offset_x", 0.1)
        )
        self._drawer_close_push_offset_x = float(
            self._get_cfg_value(close_cfg, "drawer_push_offset_x", 0.05)
        )
        self._drawer_close_retreat_offset_x = float(
            self._get_cfg_value(close_cfg, "drawer_retreat_offset_x", 0.1)
        )
        self._drawer_close_retreat_offset_z = float(
            self._get_cfg_value(close_cfg, "drawer_retreat_offset_z", 0.08)
        )
        self._drawer_close_retreat_distance_threshold = float(
            self._get_cfg_value(close_cfg, "drawer_retreat_distance_threshold", 0.06)
        )
        self._default_close_door_angle = float(self._get_cfg_value(close_cfg, "door_close_angle_deg", 50.0))
        self._default_close_euler_deg = self._load_euler_deg(
            close_cfg,
            "end_effector_euler_deg",
            [350.0, 90.0, 25.0] if self.operate_type == "door" else [90.0, 90.0, 0.0],
        )
        self._close_end_effector_orientation = euler_angles_to_quats(
            self._default_close_euler_deg,
            degrees=True,
            extrinsic=False,
        )
        self._episode_noise = {}
        self._episode_properties_set = False
        self._last_params_used = None
        self._last_baseline_correction = None
        self._close_params = None
        self._best_success_counter = 0
        self._last_success_progress_reason = ""
        self._post_close_hold_counter = 0

        default_warmup_gripper = 0.023 if self.operate_type == "door" else 0.01
        self._warmup_close_gripper_distance = float(
            self._get_cfg_value(open_cfg, "close_gripper_distance", default_warmup_gripper)
        )
        self._bootstrap_open_with_controller = bool(
            self._get_cfg_value(task_cfg, "bootstrap_open_with_controller", self.operate_type == "drawer")
        )
        self._bootstrap_open_target_distance = float(
            self._get_cfg_value(task_cfg, "initial_drawer_open_distance", 0.16)
        )
        self._warmup_door_open_angle = float(self._get_cfg_value(open_cfg, "door_open_angle_deg", 50.0))
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

        noise_cfg = getattr(cfg, "noise", None)
        if noise_cfg and getattr(noise_cfg, "enabled", False):
            self._noise_enabled = True
            self._noise_scale = float(getattr(noise_cfg, "noise_scale", 1.0))
            self._failure_bias_ratio = float(getattr(noise_cfg, "failure_bias_ratio", 0.0))
            self._noise_distribution = str(getattr(noise_cfg, "noise_distribution", "uniform"))
            self._noise_distribution_by_key = {
                "push_distance": str(getattr(noise_cfg, "push_distance_distribution", self._noise_distribution)),
                "drawer_contact_offset_y": str(
                    getattr(noise_cfg, "drawer_contact_offset_y_distribution", self._noise_distribution)
                ),
                "drawer_contact_offset_z": str(
                    getattr(noise_cfg, "drawer_contact_offset_z_distribution", self._noise_distribution)
                ),
                "end_effector_euler_deg": str(
                    getattr(noise_cfg, "end_effector_euler_distribution", self._noise_distribution)
                ),
                "door_close_angle_deg": str(
                    getattr(noise_cfg, "door_close_angle_distribution", self._noise_distribution)
                ),
            }
            self._noise_range = {
                "push_distance": list(getattr(noise_cfg, "push_distance", [-0.03, 0.03])),
                "drawer_contact_offset_y": list(getattr(noise_cfg, "drawer_contact_offset_y", [0.0, 0.0])),
                "drawer_contact_offset_z": list(getattr(noise_cfg, "drawer_contact_offset_z", [0.0, 0.0])),
                "end_effector_euler_deg": list(getattr(noise_cfg, "end_effector_euler_deg", [-6.0, 6.0])),
                "door_close_angle_deg": list(getattr(noise_cfg, "door_close_angle_deg", [-12.0, 12.0])),
            }
        else:
            self._noise_enabled = False
            self._noise_distribution_by_key = {}

        if getattr(cfg, "mode", None) != "collect":
            self._noise_enabled = False

        super().__init__(cfg, robot)
        self.initial_handle_position = None
        self._collect_phase = "close"
        self._warmup_initial_handle_position = None
        self._warmup_open_target_reached = False
        self._warmup_ready_handle_position = None
        self._closed_reference_handle_position = None
        self._door_close_target_distance_threshold = float(
            self._get_cfg_value(close_cfg, "door_target_distance_threshold", 0.15)
        )

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

    @staticmethod
    def _get_object_type(state):
        obj_category = str(state.get("object_category", "") or "").strip()
        if obj_category:
            return obj_category
        object_name = str(state.get("object_name", "") or "").strip()
        if object_name:
            return re.sub(r"\d+", "", object_name).replace("_", " ").replace("  ", " ").strip().lower()
        return "unknown"

    def _sample_noise(self):
        if not self._noise_enabled:
            self._episode_noise = {}
            return

        if self._failure_bias_ratio > 0 and np.random.random() < self._failure_bias_ratio:
            scale = self._noise_scale
        else:
            scale = 1.0
        dist = getattr(self, "_noise_distribution", "uniform")

        def sample_in_range(lo, hi, distribution):
            if distribution == "edge_bias":
                u = np.random.beta(0.5, 0.5)
            elif distribution == "min_bias":
                u = np.random.beta(0.6, 2.4)
            elif distribution == "max_bias":
                u = np.random.beta(2.4, 0.6)
            else:
                u = np.random.uniform(0.0, 1.0)
            return lo + (hi - lo) * u

        def scaled_range(key):
            lo, hi = self._noise_range[key][0], self._noise_range[key][1]
            mid = (lo + hi) / 2
            half = (hi - lo) / 2 * scale
            return mid - half, mid + half

        def distribution_for(key):
            return self._noise_distribution_by_key.get(key, dist)

        self._episode_noise = {
            "push_distance": float(
                sample_in_range(*scaled_range("push_distance"), distribution_for("push_distance"))
            ),
            "drawer_contact_offset_y": float(
                sample_in_range(
                    *scaled_range("drawer_contact_offset_y"),
                    distribution_for("drawer_contact_offset_y"),
                )
            ),
            "drawer_contact_offset_z": float(
                sample_in_range(
                    *scaled_range("drawer_contact_offset_z"),
                    distribution_for("drawer_contact_offset_z"),
                )
            ),
            "end_effector_euler_deg": np.array(
                [
                    sample_in_range(
                        *scaled_range("end_effector_euler_deg"),
                        distribution_for("end_effector_euler_deg"),
                    )
                    for _ in range(3)
                ],
                dtype=float,
            ),
            "door_close_angle_deg": float(
                sample_in_range(
                    *scaled_range("door_close_angle_deg"),
                    distribution_for("door_close_angle_deg"),
                )
            ),
        }

    @staticmethod
    def _snapshot_close_state_for_task_props(state):
        snapshot = {}
        if state.get("object_position") is not None:
            snapshot["object_position"] = [float(x) for x in state["object_position"][:3]]
        sampled_object_position = state.get("sampled_object_position")
        if sampled_object_position is not None:
            snapshot["sampled_object_position"] = [float(x) for x in sampled_object_position[:3]]
        return snapshot

    def _build_close_params(self):
        push_distance = self._close_push_distance
        drawer_contact_offset_y = 0.0
        drawer_contact_offset_z = 0.0
        euler_deg = self._default_close_euler_deg.copy()
        door_close_angle_deg = float(self._default_close_door_angle)

        correction_gt = None
        if self._noise_enabled:
            if not self._episode_noise:
                self._sample_noise()
            noise = self._episode_noise
            push_distance += float(noise["push_distance"])
            drawer_contact_offset_y += float(noise["drawer_contact_offset_y"])
            drawer_contact_offset_z += float(noise["drawer_contact_offset_z"])
            euler_deg = euler_deg + noise["end_effector_euler_deg"]
            door_close_angle_deg += float(noise["door_close_angle_deg"])
            correction_gt = {
                "push_distance": -float(noise["push_distance"]),
                "drawer_contact_offset_y": -float(noise["drawer_contact_offset_y"]),
                "drawer_contact_offset_z": -float(noise["drawer_contact_offset_z"]),
                "end_effector_euler_deg": (-noise["end_effector_euler_deg"]).tolist(),
                "door_close_angle_deg": -float(noise["door_close_angle_deg"]),
            }

        params_used = {
            "push_distance": float(push_distance),
            "drawer_contact_offset_y": float(drawer_contact_offset_y),
            "drawer_contact_offset_z": float(drawer_contact_offset_z),
            "end_effector_euler_deg": euler_deg.tolist(),
            "door_close_angle_deg": float(door_close_angle_deg),
        }
        return params_used, correction_gt

    def _maybe_set_close_task_properties(self, state, params_used, correction_gt):
        if self._episode_properties_set or not hasattr(self.data_collector, "set_task_properties"):
            return

        props = {
            "action_type": f"close_{self.operate_type}",
            "params_used": params_used,
            "object_type": self._get_object_type(state),
            "source_object_name": state.get("object_name"),
        }
        props.update(self._snapshot_close_state_for_task_props(state))

        if self._noise_enabled:
            props["injected_noise"] = {
                key: (value.tolist() if hasattr(value, "tolist") else value)
                for key, value in self._episode_noise.items()
            }
            props["correction_gt"] = correction_gt

        self.data_collector.set_task_properties(props)
        self._episode_properties_set = True
        self._last_params_used = dict(params_used)
        self._last_baseline_correction = dict(correction_gt) if correction_gt else None

    def _prepare_close_episode(self, state):
        if self._close_params is not None:
            return self._close_params

        params_used, correction_gt = self._build_close_params()
        self._maybe_set_close_task_properties(state, params_used, correction_gt)
        self._close_params = {
            "push_distance": float(params_used["push_distance"]),
            "drawer_contact_offset_y": float(params_used["drawer_contact_offset_y"]),
            "drawer_contact_offset_z": float(params_used["drawer_contact_offset_z"]),
            "door_close_angle_deg": float(params_used["door_close_angle_deg"]),
            "end_effector_orientation": euler_angles_to_quats(
                np.asarray(params_used["end_effector_euler_deg"], dtype=float),
                degrees=True,
                extrinsic=False,
            ),
        }
        return self._close_params

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
                furniture_type=self.operate_type,
                door_open_direction="clockwise",
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
            events_dt=self._close_events_dt,
            furniture_type=self.operate_type,
            door_open_direction="clockwise",
            drawer_approach_offset_x=self._drawer_close_approach_offset_x,
            drawer_push_offset_x=self._drawer_close_push_offset_x,
            drawer_retreat_offset_x=self._drawer_close_retreat_offset_x,
            drawer_retreat_offset_z=self._drawer_close_retreat_offset_z,
            drawer_retreat_distance_threshold=self._drawer_close_retreat_distance_threshold,
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
        self._closed_reference_handle_position = None
        self._best_success_counter = 0
        self._last_success_progress_reason = ""
        self._post_close_hold_counter = 0
        if self.mode == "collect":
            self._episode_properties_set = False
            self._last_params_used = None
            self._last_baseline_correction = None
            self._close_params = None
            self._collect_phase = "warmup_open" if self._bootstrap_open_with_controller else "close"
            if self._bootstrap_open_with_controller:
                self.warmup_open_controller.reset(events_dt=self._warmup_open_events_dt)
            self.close_controller.reset()
            self._sample_noise()
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

    def _read_local_translation(self, object_path):
        stage = getattr(self.object_utils, "_stage", None)
        if stage is None:
            return None

        prim = stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            return None

        translate_attr = prim.GetAttribute("xformOp:translate")
        if not translate_attr.IsValid():
            return None

        value = translate_attr.Get()
        if value is None:
            return None

        return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)

    def _is_drawer_already_open_enough(self, state):
        if self.operate_type != "drawer":
            return False

        object_path = str(state.get("object_path", "") or "")
        drawer_body_path = self._drawer_body_path_by_object.get(object_path)
        if not drawer_body_path:
            return False

        local_translation = self._read_local_translation(drawer_body_path)
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
                    "[CloseWarmup] "
                    f"object already open enough; skipping warmup at {self.initial_handle_position.tolist()}"
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
                    "[CloseWarmup] "
                    f"opened object to start close phase at {self.initial_handle_position.tolist()}"
                )
                return None, False, False

            print("[CloseWarmup] Failed to physically open object before close phase")
            self._last_failure_reason = "Warmup open failed before close phase"
            self._last_success = False
            self.reset_needed = True
            self._early_return = True
            return None, True, False

        if self.operate_type == "door":
            action = self.warmup_open_controller.forward(
                handle_position=state["object_position"],
                current_joint_positions=state["joint_positions"],
                revolute_joint_position=state["revolute_joint_position"],
                gripper_position=state["gripper_position"],
                end_effector_orientation=self._warmup_open_end_effector_orientation,
                angle=self._warmup_door_open_angle,
                close_gripper_distance=float(
                    state.get("close_gripper_distance", self._warmup_close_gripper_distance)
                ),
            )
        else:
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

    def _cache_collect_step(self, state):
        if "camera_data" not in state:
            return
        self.data_collector.cache_step(
            camera_images=state["camera_data"],
            joint_angles=state["joint_positions"][:-1],
            language_instruction=self.get_language_instruction(),
        )

    def _update_success_counter(self, state):
        if self._check_success(state):
            self.check_success_counter += 1
            self._best_success_counter = max(self._best_success_counter, self.check_success_counter)
        else:
            self.check_success_counter = 0
        return self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS

    def _step_collect(self, state):
        if not self.close_controller.is_done():
            close_params = self._prepare_close_episode(state)
            if self.operate_type == "door":
                action = self.close_controller.forward(
                    handle_position=state["object_position"],
                    current_joint_positions=state["joint_positions"],
                    revolute_joint_position=state["revolute_joint_position"],
                    gripper_position=state["gripper_position"],
                    end_effector_orientation=close_params["end_effector_orientation"],
                    angle=close_params["door_close_angle_deg"],
                )
            else:
                action = self.close_controller.forward(
                    handle_position=state["object_position"],
                    current_joint_positions=state["joint_positions"],
                    gripper_position=state["gripper_position"],
                    end_effector_orientation=close_params["end_effector_orientation"],
                    push_distance=close_params["push_distance"],
                    drawer_contact_offset_y=close_params["drawer_contact_offset_y"],
                    drawer_contact_offset_z=close_params["drawer_contact_offset_z"],
                )

            self._cache_collect_step(state)
            self._update_success_counter(state)

            return action, False, False

        success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if not success and self._post_close_hold_counter < self._post_close_hold_steps:
            if self._post_close_hold_counter == 0:
                print(
                    "[ClosePostHold] close primitive done; "
                    f"checking success for up to {self._post_close_hold_steps} extra steps"
                )
            self._post_close_hold_counter += 1
            self._cache_collect_step(state)
            success = self._update_success_counter(state)
            if not success:
                return None, False, False

        if success:
            print("Task success!")
        else:
            print("Task failed!")
            if not self._last_failure_reason and self._best_success_counter > 0:
                self._last_failure_reason = (
                    "Success condition was not held long enough after post-hold "
                    f"({self._best_success_counter}/{self.REQUIRED_SUCCESS_STEPS}, "
                    f"post_hold_steps={self._post_close_hold_steps})"
                )
            elif not self._last_failure_reason:
                self._last_failure_reason = (
                    "Close controller and post-hold finished before success condition was reached"
                )
            self.print_failure_reason()
        return self._finalize_collect_episode(state, success)

    def _finalize_collect_episode(self, state, is_success):
        self._early_return = False
        if self._episode_properties_set and hasattr(self.data_collector, "update_task_properties"):
            updates = {
                "is_success": bool(is_success),
                **self._snapshot_close_state_for_task_props(state),
            }
            if not is_success and self._last_baseline_correction:
                updates["correction_gt"] = dict(self._last_baseline_correction)
            self.data_collector.update_task_properties(updates)

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
            required_handle_move = max(0.13, self._close_push_distance * 0.85)
            required_gripper_distance = 0.04
            handle_moved_enough = handle_move_distance > required_handle_move
            gripper_far_enough = gripper_to_object_distance > required_gripper_distance
        elif self.operate_type == "door":
            handle_moved_enough = handle_move_distance > 0.08
            gripper_far_enough = gripper_to_object_distance > 0.08
            target_position = getattr(self.close_controller, "target_position", None)
            handle_closed_enough = True
            target_distance = None
            if target_position is not None:
                target_distance = np.linalg.norm(current_pos - np.asarray(target_position, dtype=float))
                handle_closed_enough = target_distance < self._door_close_target_distance_threshold
            success = handle_moved_enough and handle_closed_enough and gripper_far_enough
            if success:
                self._last_failure_reason = ""
                return True

            reasons = []
            if not handle_moved_enough:
                reasons.append(f"Handle returned distance too short ({handle_move_distance:.4f}<0.08)")
            if not handle_closed_enough:
                reasons.append(
                    "Handle not close enough to close target "
                    f"({target_distance:.4f}>={self._door_close_target_distance_threshold:.2f})"
                )
            if not gripper_far_enough:
                reasons.append(f"Gripper too close to object ({gripper_to_object_distance:.4f}<0.08)")
            self._last_failure_reason = " and ".join(reasons)
            return False
        else:
            required_handle_move = 0.08
            required_gripper_distance = 0.08
            handle_move_distance = current_pos[0] - self.initial_handle_position[0]
            handle_moved_enough = handle_move_distance > required_handle_move
            gripper_far_enough = gripper_to_object_distance > required_gripper_distance

        success = handle_moved_enough and gripper_far_enough
        if success:
            self._last_failure_reason = ""
            self._last_success_progress_reason = (
                f"handle_move={handle_move_distance:.4f}, "
                f"gripper_distance={gripper_to_object_distance:.4f}"
            )
            return True

        if not handle_moved_enough and not gripper_far_enough:
            self._last_failure_reason = (
                f"Handle moved distance too short ({handle_move_distance:.4f}<{required_handle_move:.4f}) and "
                f"gripper too close to object ({gripper_to_object_distance:.4f}<{required_gripper_distance:.4f})"
            )
        elif not handle_moved_enough:
            self._last_failure_reason = (
                f"Handle moved distance too short ({handle_move_distance:.4f}<{required_handle_move:.4f})"
            )
        else:
            self._last_failure_reason = (
                f"Gripper too close to object ({gripper_to_object_distance:.4f}<{required_gripper_distance:.4f})"
            )
        return False

    def get_language_instruction(self) -> Optional[str]:
        object_name = re.sub(r"\d+", "", self.state["object_name"]).replace("_", " ").lower()
        self._language_instruction = f"Close the {self.operate_type} of the {object_name}"
        return self._language_instruction

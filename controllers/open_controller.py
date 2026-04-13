import re
from typing import Optional
from robots.franka.rmpflow_controller import RMPFlowController
import numpy as np

from controllers.atomic_actions.open_controller import OpenController
from .base_controller import BaseController
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from .inference_engines.inference_engine_factory import InferenceEngineFactory

class OpenTaskController(BaseController):
    """Controller for managing the task of opening a door in collect or infer mode.

    Args:
        cfg: Configuration object containing mode and other parameters.
        robot: Robot articulation instance.
    """

    def __init__(self, cfg, robot):
        open_cfg = getattr(cfg, "open", None)
        self._open_events_dt = self._load_sequence(
            open_cfg, "events_dt", [0.0025, 0.005, 0.08, 0.004, 0.05, 0.05, 0.01, 0.004], expected_len=8
        )
        self._open_position_threshold = float(self._get_cfg_value(open_cfg, "position_threshold", 0.01))
        self._open_stage0_offset_x = float(self._get_cfg_value(open_cfg, "stage0_offset_x", 0.08))
        self._open_stage1_offset_x = float(self._get_cfg_value(open_cfg, "stage1_offset_x", 0.015))
        self._open_retreat_offset_x = float(self._get_cfg_value(open_cfg, "retreat_offset_x", 0.06))
        self._open_retreat_offset_y = float(self._get_cfg_value(open_cfg, "retreat_offset_y", 0.04))
        self._door_open_angle = float(self._get_cfg_value(open_cfg, "door_open_angle_deg", 50.0))
        self._default_open_euler_deg = self._load_euler_deg(open_cfg, "end_effector_euler_deg", [0.0, 110.0, 0.0])
        self._open_end_effector_orientation = euler_angles_to_quats(
            self._default_open_euler_deg,
            degrees=True,
            extrinsic=False,
        )
        self._episode_noise = {}
        self._episode_properties_set = False
        self._last_params_used = None
        self._last_baseline_correction = None
        self._open_params = None

        noise_cfg = getattr(cfg, "noise", None)
        if noise_cfg and getattr(noise_cfg, "enabled", False):
            self._noise_enabled = True
            self._noise_scale = float(getattr(noise_cfg, "noise_scale", 1.0))
            self._failure_bias_ratio = float(getattr(noise_cfg, "failure_bias_ratio", 0.0))
            self._noise_distribution = str(getattr(noise_cfg, "noise_distribution", "uniform"))
            self._noise_range = {
                "stage0_offset_x": list(getattr(noise_cfg, "stage0_offset_x", [-0.02, 0.02])),
                "stage1_offset_x": list(getattr(noise_cfg, "stage1_offset_x", [-0.008, 0.008])),
                "retreat_offset_x": list(getattr(noise_cfg, "retreat_offset_x", [-0.02, 0.02])),
                "retreat_offset_y": list(getattr(noise_cfg, "retreat_offset_y", [-0.02, 0.02])),
                "end_effector_euler_deg": list(getattr(noise_cfg, "end_effector_euler_deg", [-8.0, 8.0])),
                "door_open_angle_deg": list(getattr(noise_cfg, "door_open_angle_deg", [-12.0, 12.0])),
                "close_gripper_distance": list(getattr(noise_cfg, "close_gripper_distance", [-0.006, 0.006])),
            }
        else:
            self._noise_enabled = False

        if getattr(cfg, "mode", None) != "collect":
            self._noise_enabled = False

        super().__init__(cfg, robot)
        self.initial_handle_position = None

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

        def sample_in_range(lo, hi):
            if dist == "edge_bias":
                u = np.random.beta(0.5, 0.5)
                return lo + (hi - lo) * u
            return np.random.uniform(lo, hi)

        def scaled_range(key):
            lo, hi = self._noise_range[key][0], self._noise_range[key][1]
            mid = (lo + hi) / 2
            half = (hi - lo) / 2 * scale
            return mid - half, mid + half

        self._episode_noise = {
            "stage0_offset_x": float(sample_in_range(*scaled_range("stage0_offset_x"))),
            "stage1_offset_x": float(sample_in_range(*scaled_range("stage1_offset_x"))),
            "retreat_offset_x": float(sample_in_range(*scaled_range("retreat_offset_x"))),
            "retreat_offset_y": float(sample_in_range(*scaled_range("retreat_offset_y"))),
            "end_effector_euler_deg": np.array(
                [sample_in_range(*scaled_range("end_effector_euler_deg")) for _ in range(3)],
                dtype=float,
            ),
            "door_open_angle_deg": float(sample_in_range(*scaled_range("door_open_angle_deg"))),
            "close_gripper_distance": float(sample_in_range(*scaled_range("close_gripper_distance"))),
        }

    @staticmethod
    def _snapshot_open_state_for_task_props(state):
        snapshot = {}
        if state.get("object_position") is not None:
            snapshot["object_position"] = [float(x) for x in state["object_position"][:3]]
        sampled_object_position = state.get("sampled_object_position")
        if sampled_object_position is not None:
            snapshot["sampled_object_position"] = [float(x) for x in sampled_object_position[:3]]
        return snapshot

    def _build_open_params(self, state):
        stage0_offset_x = self._open_stage0_offset_x
        stage1_offset_x = self._open_stage1_offset_x
        retreat_offset_x = self._open_retreat_offset_x
        retreat_offset_y = self._open_retreat_offset_y
        euler_deg = self._default_open_euler_deg.copy()
        door_open_angle_deg = float(self._door_open_angle)
        close_gripper_distance = float(state.get("close_gripper_distance", 0.023))

        correction_gt = None
        if self._noise_enabled:
            if not self._episode_noise:
                self._sample_noise()
            n = self._episode_noise
            stage0_offset_x += float(n["stage0_offset_x"])
            stage1_offset_x += float(n["stage1_offset_x"])
            retreat_offset_x += float(n["retreat_offset_x"])
            retreat_offset_y += float(n["retreat_offset_y"])
            euler_deg = euler_deg + n["end_effector_euler_deg"]
            door_open_angle_deg += float(n["door_open_angle_deg"])
            close_gripper_distance += float(n["close_gripper_distance"])
            correction_gt = {
                "stage0_offset_x": -float(n["stage0_offset_x"]),
                "stage1_offset_x": -float(n["stage1_offset_x"]),
                "retreat_offset_x": -float(n["retreat_offset_x"]),
                "retreat_offset_y": -float(n["retreat_offset_y"]),
                "end_effector_euler_deg": (-n["end_effector_euler_deg"]).tolist(),
                "door_open_angle_deg": -float(n["door_open_angle_deg"]),
                "close_gripper_distance": -float(n["close_gripper_distance"]),
            }

        params_used = {
            "stage0_offset_x": float(stage0_offset_x),
            "stage1_offset_x": float(stage1_offset_x),
            "retreat_offset_x": float(retreat_offset_x),
            "retreat_offset_y": float(retreat_offset_y),
            "end_effector_euler_deg": euler_deg.tolist(),
            "door_open_angle_deg": float(door_open_angle_deg),
            "close_gripper_distance": float(close_gripper_distance),
        }
        return params_used, correction_gt

    def _maybe_set_open_task_properties(self, state, params_used, correction_gt):
        if self._episode_properties_set or not hasattr(self.data_collector, "set_task_properties"):
            return

        props = {
            "action_type": "open_door",
            "params_used": params_used,
            "object_type": self._get_object_type(state),
            "source_object_name": state.get("object_name"),
        }
        props.update(self._snapshot_open_state_for_task_props(state))

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

    def _prepare_open_episode(self, state):
        if self._open_params is not None:
            return self._open_params

        params_used, correction_gt = self._build_open_params(state)
        self._maybe_set_open_task_properties(state, params_used, correction_gt)
        self.open_controller.configure_episode_params(
            position_threshold=self._open_position_threshold,
            stage0_offset_x=params_used["stage0_offset_x"],
            stage1_offset_x=params_used["stage1_offset_x"],
            retreat_offset_x=params_used["retreat_offset_x"],
            retreat_offset_y=params_used["retreat_offset_y"],
        )
        self._open_params = {
            "end_effector_orientation": euler_angles_to_quats(
                np.asarray(params_used["end_effector_euler_deg"], dtype=float),
                degrees=True,
                extrinsic=False,
            ),
            "door_open_angle_deg": float(params_used["door_open_angle_deg"]),
            "close_gripper_distance": float(params_used["close_gripper_distance"]),
        }
        return self._open_params

    def _finalize_collect_episode(self, state, is_success):
        if self._episode_properties_set and hasattr(self.data_collector, "update_task_properties"):
            updates = {
                "is_success": bool(is_success),
                **self._snapshot_open_state_for_task_props(state),
            }
            if not is_success and self._last_baseline_correction:
                updates["correction_gt"] = dict(self._last_baseline_correction)
            self.data_collector.update_task_properties(updates)

        if self._episode_properties_set:
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
        else:
            self.data_collector.clear_cache()

        self._last_success = bool(is_success)
        self.reset_needed = True
        return None, True, bool(is_success)
            
    def _init_collect_mode(self, cfg, robot):
        """Initializes the controller for data collection mode.

        Args:
            cfg: Configuration object for collect mode.
            robot: Robot articulation instance.
        """
        super()._init_collect_mode(cfg, robot)
        
        self.open_controller = OpenController(
            name="open_controller",
            cspace_controller=RMPFlowController(
                name="target_follower_controller",
                robot_articulation=robot
            ),
            gripper=robot.gripper,
            events_dt=self._open_events_dt,
            furniture_type=self.cfg.task.get("operate_type", "door"),
            door_open_direction="clockwise",
            position_threshold=self._open_position_threshold,
            stage0_offset_x=self._open_stage0_offset_x,
            stage1_offset_x=self._open_stage1_offset_x,
            retreat_offset_x=self._open_retreat_offset_x,
            retreat_offset_y=self._open_retreat_offset_y,
        )

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

    def reset(self):
        """Resets the controller to its initial state."""
        super().reset()
        self.initial_handle_position = None
        if self.mode == "collect":
            self._episode_properties_set = False
            self._last_params_used = None
            self._last_baseline_correction = None
            self._open_params = None
            self.open_controller.reset(events_dt=self._open_events_dt)
            self._sample_noise()
        else:
            self.inference_engine.reset()

    def is_atomic_action_complete(self) -> bool:
        """Prevent task-level timeout from resetting mid-open before collect finalization."""
        if self.mode != "collect" or not hasattr(self, "open_controller"):
            return True
        return self.open_controller.is_done()

    def step(self, state):
        """Executes one step of the task based on the current state.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        self.state = state
        if self.initial_handle_position is None:
            self.initial_handle_position = state['object_position']
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)

    def _step_collect(self, state):
        """Executes a step in collect mode using the open controller.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        if not self.open_controller.is_done():
            open_params = self._prepare_open_episode(state)
            if self.cfg.task.get("operate_type") == "door":
                action = self.open_controller.forward(
                    handle_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    revolute_joint_position=state['revolute_joint_position'],
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=open_params["end_effector_orientation"],
                    angle=open_params["door_open_angle_deg"],
                    close_gripper_distance=open_params["close_gripper_distance"]
                )
            else:
                action = self.open_controller.forward(
                    handle_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=euler_angles_to_quats([90, 90, 0], degrees=True, extrinsic=False),
                    close_gripper_distance=close_gripper_distance
                )
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
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
        return self._finalize_collect_episode(state, success)

    def _step_infer(self, state):
        """Executes a step in infer mode using the trained policy.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        language_instruction = self.get_language_instruction()
        if language_instruction is not None:
            state['language_instruction'] = language_instruction
        else:
            state['language_instruction'] = "Open the door of the object"
        
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
        """Checks if the task has been successfully completed.

        Args:
            state: Current state of the environment.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        current_pos = state['object_position']
        gripper_position = state['gripper_position']
        
        # Calculate distances
        handle_move_distance = np.linalg.norm(np.array(current_pos) - self.initial_handle_position)
        gripper_to_object_distance = np.linalg.norm(np.array(gripper_position) - np.array(current_pos))
        
        # Check conditions
        handle_moved_enough = handle_move_distance > 0.12
        gripper_far_enough = gripper_to_object_distance > 0.04
        
        success = handle_moved_enough and gripper_far_enough
        if success:
            self._last_failure_reason = ""
            return True
        
        # Update failure reason
        if not handle_moved_enough and not gripper_far_enough:
            self._last_failure_reason = f"Handle moved distance too short ({handle_move_distance:.4f}<0.12) and Gripper too close to object ({gripper_to_object_distance:.4f}<0.04)"
        elif not handle_moved_enough:
            self._last_failure_reason = f"Handle moved distance too short ({handle_move_distance:.4f}<0.12)"
        else:
            self._last_failure_reason = f"Gripper too close to object ({gripper_to_object_distance:.4f}<0.04)"
        
        return False

    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').lower()
        self._language_instruction = f"Open the door of the {object_name}"
        return self._language_instruction

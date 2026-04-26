from scipy.spatial.transform import Rotation as R
import numpy as np
from enum import Enum
from typing import Optional
from utils.task_utils import TaskUtils
import re

from .atomic_actions.pick_controller import PickController
from .atomic_actions.pour_controller import PourController
from .base_controller import BaseController

class Phase(Enum):
    PICKING = "picking"
    POURING = "pouring"
    FINISHED = "finished"

class PickPourTaskController(BaseController):
    def __init__(self, cfg, robot):
        """Initialize the pick and pour task controller.
        
        Args:
            cfg: Configuration object containing controller settings
            robot: Robot instance to control
        """
        super().__init__(cfg, robot)
        self.initial_position = None
        self.initial_size = None
        self.task_utils = TaskUtils.get_instance()
        self.initial_quaternion = None
        self.pour_timer = 0
        self.pour_complete = False
        self.return_complete = False
        self.return_timer = 0
        self.last_error_info = None
        self.current_phase = Phase.PICKING
        self._episode_noise = {}
        self._episode_properties_set = False
        self._last_params_used = None
        self._last_baseline_correction = None
        self._pickpour_params = None
        noise_cfg = getattr(cfg, "noise", None)
        if noise_cfg and getattr(noise_cfg, "enabled", False) and getattr(cfg, "mode", None) == "collect":
            self._noise_enabled = True
            self._noise_scale = float(getattr(noise_cfg, "noise_scale", 1.0))
            self._failure_bias_ratio = float(getattr(noise_cfg, "failure_bias_ratio", 0.0))
            self._noise_distribution = str(getattr(noise_cfg, "noise_distribution", "uniform"))
            self._noise_range = {
                "pick_target_position_xy": list(getattr(noise_cfg, "pick_target_position_xy_noise", [-0.01, 0.01])),
                "pick_target_position_z": list(getattr(noise_cfg, "pick_target_position_z_noise", [-0.005, 0.005])),
                "pour_target_position_xy": list(getattr(noise_cfg, "pour_target_position_xy_noise", [-0.08, 0.08])),
                "pour_target_position_z": list(getattr(noise_cfg, "pour_target_position_z_noise", [-0.01, 0.01])),
                "pick_after_offset_z": list(getattr(noise_cfg, "pick_after_offset_z", [-0.02, 0.02])),
                "gripper_distance": list(getattr(noise_cfg, "gripper_distance", [-0.003, 0.003])),
                "pour_speed": list(getattr(noise_cfg, "pour_speed", [-0.2, 0.2])),
                "pick_end_effector_euler_deg": list(getattr(noise_cfg, "pick_end_effector_euler_deg", [-5.0, 5.0])),
                "pour_end_effector_euler_deg": list(getattr(noise_cfg, "pour_end_effector_euler_deg", [-8.0, 8.0])),
            }
        else:
            self._noise_enabled = False
            
    def _init_collect_mode(self, cfg, robot):
        """Initialize controller for data collection mode."""
        super()._init_collect_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
        )
        
        self.pour_controller = PourController(
            name="pour_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.006, 0.002, 0.009, 0.01, 0.009, 0.01]
        )
        self.active_controller = self.pick_controller

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
            elif dist == "min_bias":
                u = np.random.beta(0.6, 2.4)
            elif dist == "max_bias":
                u = np.random.beta(2.4, 0.6)
            else:
                u = np.random.uniform(0.0, 1.0)
            return lo + (hi - lo) * u

        def scaled_range(key):
            lo, hi = self._noise_range[key][0], self._noise_range[key][1]
            mid = (lo + hi) / 2
            half = (hi - lo) / 2 * scale
            return mid - half, mid + half

        self._episode_noise = {
            "pick_target_position": np.array([
                sample_in_range(*scaled_range("pick_target_position_xy")),
                sample_in_range(*scaled_range("pick_target_position_xy")),
                sample_in_range(*scaled_range("pick_target_position_z")),
            ], dtype=float),
            "pour_target_position": np.array([
                sample_in_range(*scaled_range("pour_target_position_xy")),
                sample_in_range(*scaled_range("pour_target_position_xy")),
                sample_in_range(*scaled_range("pour_target_position_z")),
            ], dtype=float),
            "pick_after_offset_z": float(sample_in_range(*scaled_range("pick_after_offset_z"))),
            "gripper_distance": float(sample_in_range(*scaled_range("gripper_distance"))),
            "pour_speed": float(sample_in_range(*scaled_range("pour_speed"))),
            "pick_end_effector_euler_deg": np.array([
                sample_in_range(*scaled_range("pick_end_effector_euler_deg")) for _ in range(3)
            ], dtype=float),
            "pour_end_effector_euler_deg": np.array([
                sample_in_range(*scaled_range("pour_end_effector_euler_deg")) for _ in range(3)
            ], dtype=float),
        }

    @staticmethod
    def _to_jsonable(value):
        return value.tolist() if hasattr(value, "tolist") else value

    def _build_pickpour_params(self, state):
        params_used = {
            "pick_target_position_offset": [0.0, 0.0, 0.0],
            "pour_target_position_offset": [0.0, 0.0, 0.0],
            "pick_after_offset_z": 0.3,
            "gripper_distance": None,
            "pour_speed": -1.0,
            "pick_end_effector_euler_deg": [0.0, 90.0, 30.0],
            "pour_end_effector_euler_deg": [0.0, 90.0, 15.0],
        }
        correction_gt = None

        if self._noise_enabled:
            if not self._episode_noise:
                self._sample_noise()
            n = self._episode_noise
            pick_offset = n["pick_target_position"]
            pour_offset = n["pour_target_position"]
            base_gripper_distance = float(self.pick_controller.get_gripper_distance(state["object_name"]))

            params_used["pick_target_position_offset"] = pick_offset.tolist()
            params_used["pour_target_position_offset"] = pour_offset.tolist()
            params_used["pick_after_offset_z"] += float(n["pick_after_offset_z"])
            params_used["gripper_distance"] = base_gripper_distance + float(n["gripper_distance"])
            params_used["pour_speed"] += float(n["pour_speed"])
            params_used["pick_end_effector_euler_deg"] = (
                np.asarray(params_used["pick_end_effector_euler_deg"], dtype=float)
                + n["pick_end_effector_euler_deg"]
            ).tolist()
            params_used["pour_end_effector_euler_deg"] = (
                np.asarray(params_used["pour_end_effector_euler_deg"], dtype=float)
                + n["pour_end_effector_euler_deg"]
            ).tolist()
            correction_gt = {
                "pick_target_position_offset": (-pick_offset).tolist(),
                "pour_target_position_offset": (-pour_offset).tolist(),
                "pick_after_offset_z": -float(n["pick_after_offset_z"]),
                "gripper_distance": -float(n["gripper_distance"]),
                "pour_speed": -float(n["pour_speed"]),
                "pick_end_effector_euler_deg": (-n["pick_end_effector_euler_deg"]).tolist(),
                "pour_end_effector_euler_deg": (-n["pour_end_effector_euler_deg"]).tolist(),
            }

        return params_used, correction_gt

    def _maybe_set_pickpour_task_properties(self, state, params_used, correction_gt):
        if self._episode_properties_set or not hasattr(self.data_collector, "set_task_properties"):
            return

        props = {
            "action_type": "pour_liquid",
            "params_used": params_used,
            "source_object_name": state.get("object_name"),
            "object_position": self._to_jsonable(np.asarray(state["object_position"], dtype=float)),
            "target_position": self._to_jsonable(np.asarray(state["target_position"], dtype=float)),
        }
        if self._noise_enabled:
            props["injected_noise"] = {
                key: self._to_jsonable(value)
                for key, value in self._episode_noise.items()
            }
            props["correction_gt"] = correction_gt

        self.data_collector.set_task_properties(props)
        self._episode_properties_set = True
        self._last_params_used = dict(params_used)
        self._last_baseline_correction = dict(correction_gt) if correction_gt else None

    def _prepare_pickpour_episode(self, state):
        if self._pickpour_params is not None:
            return self._pickpour_params

        params_used, correction_gt = self._build_pickpour_params(state)
        self._maybe_set_pickpour_task_properties(state, params_used, correction_gt)
        self._pickpour_params = {
            **params_used,
            "pick_target_position_offset": np.asarray(params_used["pick_target_position_offset"], dtype=float),
            "pour_target_position_offset": np.asarray(params_used["pour_target_position_offset"], dtype=float),
            "pick_end_effector_orientation": R.from_euler(
                'xyz',
                np.radians(params_used["pick_end_effector_euler_deg"]),
            ).as_quat(),
            "pour_end_effector_orientation": R.from_euler(
                'xyz',
                np.radians(params_used["pour_end_effector_euler_deg"]),
            ).as_quat(),
        }
        return self._pickpour_params

    def reset(self):
        """Reset controller state and phase."""
        super().reset()
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.initial_size = None
        self.initial_quaternion = None
        self.pour_timer = 0
        self.pour_complete = False
        self.return_complete = False
        self.return_timer = 0
        self.last_error_info = None
        self._episode_noise = {}
        self._episode_properties_set = False
        self._last_params_used = None
        self._last_baseline_correction = None
        self._pickpour_params = None
        
        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.pick_controller.reset()
            self.pour_controller.reset()
            self._sample_noise()
        else:
            self.inference_engine.reset()

    def _check_phase_success(self):
        """Check if current phase is successful."""
        object_pos = self.state['object_position']
        self.last_error_info = None 
        
        if self.current_phase == Phase.PICKING:
            required_height = self.initial_position[2] + 0.12
            success = object_pos[2] > required_height
            if not success:
                self.last_error_info = {
                    'phase': 'PICKING',
                    'current_height': object_pos[2],
                    'required_height': required_height,
                    'height_diff': object_pos[2] - required_height
                }
            return success
            
        elif self.current_phase == Phase.POURING:
            if self.initial_quaternion is None:
                self.initial_quaternion = self.state['object_quaternion']
                self.last_error_info = {
                    'phase': 'POURING',
                    'error': 'Initial quaternion not set yet'
                }
                return False
                
            current_quat = self.state['object_quaternion']
            
            # First check if we're close enough to target for pouring
            xy_dist = np.linalg.norm(object_pos[:2] - self.state['target_position'][:2])
            pour_threshold = self.task_utils.get_pour_threshold(self.state['object_name'], self.state['object_size']) + 0.05
            
            if xy_dist > pour_threshold:
                self.last_error_info = {
                    'phase': 'POURING',
                    'current_distance': xy_dist,
                    'required_distance': pour_threshold,
                    'distance_diff': xy_dist - pour_threshold
                }
                return False
            
            if not self.pour_complete:
                # print(self.initial_quaternion, current_quat)
                self.pour_complete = self.task_utils.check_rotation_angle(
                    self.initial_quaternion, 
                    current_quat,
                    threshold_degrees=50
                )
                if not self.pour_complete:
                    self.last_error_info = {
                        'phase': 'POURING',
                        'error': 'Pour rotation not complete yet',
                        'pour_complete': self.pour_complete
                    }
                return False
                
            # After pour complete, check if returned to original orientation
            if not self.return_complete:
                rotation_diff = self.task_utils.check_rotation_angle(
                    self.initial_quaternion,
                    current_quat,
                    threshold_degrees=30  # smaller threshold for return position
                )
                if not rotation_diff:
                    self.return_complete = True
                    self.return_timer = 0
                else:
                    self.last_error_info = {
                        'phase': 'POURING',
                        'error': 'Return rotation not complete yet',
                        'return_complete': self.return_complete
                    }
                return False
                
            # Wait for 2 seconds in return position
            if self.return_complete and object_pos[2] > self.initial_position[2] + 0.05:
                self.return_timer += 0.012
                success = self.return_timer >= 1.0
                if not success:
                    self.last_error_info = {
                        'phase': 'POURING',
                        'error': 'Waiting for return timer',
                        'return_timer': self.return_timer,
                        'required_time': 1.0
                    }
                return success
            else:
                self.last_error_info = {
                    'phase': 'POURING',
                    'error': 'Object not in correct position for return timer',
                    'current_height': object_pos[2],
                    'required_height': self.initial_position[2] + 0.05,
                    'return_complete': self.return_complete
                }
                return False
        
        return False

    def _format_failure_reason(self):
        if self.last_error_info is None:
            return f"{self.current_phase.value} phase failed after controller completed"

        phase = self.last_error_info.get('phase', self.current_phase.value)
        if phase == 'PICKING':
            current_height = float(self.last_error_info.get('current_height', 0.0))
            required_height = float(self.last_error_info.get('required_height', 0.0))
            return f"Pick lift height too low ({current_height:.4f}<={required_height:.4f})"

        error = self.last_error_info.get('error')
        if error:
            return f"Pour phase failed: {error}"

        if 'current_distance' in self.last_error_info and 'required_distance' in self.last_error_info:
            current_distance = float(self.last_error_info['current_distance'])
            required_distance = float(self.last_error_info['required_distance'])
            return f"Pour target distance too far ({current_distance:.4f}>{required_distance:.4f})"

        if 'current_height' in self.last_error_info and 'required_height' in self.last_error_info:
            current_height = float(self.last_error_info['current_height'])
            required_height = float(self.last_error_info['required_height'])
            return f"Pour object height too low ({current_height:.4f}<={required_height:.4f})"

        return f"{phase} phase failed after controller completed"

    def _finalize_collect_episode(self, is_success):
        if hasattr(self.data_collector, "update_task_properties"):
            updates = {
                "is_success": bool(is_success),
                "final_object_position": np.asarray(self.state["object_position"], dtype=float).tolist(),
                "target_position": np.asarray(self.state["target_position"], dtype=float).tolist(),
            }
            if not is_success and self._last_baseline_correction:
                updates["correction_gt"] = dict(self._last_baseline_correction)
            self.data_collector.update_task_properties(updates)

        self.data_collector.write_cached_data(self.state['joint_positions'][:-1])
        self._last_success = bool(is_success)
        self.current_phase = Phase.FINISHED
        return None, True, bool(is_success)

    def step(self, state):
        """Execute one step of control.
        
        Args:
            state: Current state dictionary containing sensor data and robot state
            
        Returns:
            Tuple containing action, done flag, and success flag
        """
        self.state = state
        if self.initial_position is None:
            self.initial_position = self.state['object_position']
        if self.initial_size is None:
            self.initial_size = self.state['object_size']
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)

    def _step_collect(self, state):
        """Execute collection mode step."""
        params = self._prepare_pickpour_episode(state)
        success = self._check_phase_success()
        if success:
            if self.current_phase == Phase.PICKING:
                print("Pick task success! Switching to pour...")
                self.current_phase = Phase.POURING
                self.active_controller = self.pour_controller
                return None, False, False
            elif self.current_phase == Phase.POURING:
                print("Pour task success!")
                return self._finalize_collect_episode(True)
        
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.active_controller.is_done():
            action = None
            if self.current_phase == Phase.PICKING:
                pick_kwargs = {}
                if params["gripper_distance"] is not None:
                    pick_kwargs["gripper_distances"] = params["gripper_distance"]
                action = self.pick_controller.forward(
                    picking_position=np.asarray(state['object_position'], dtype=float) + params["pick_target_position_offset"],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=params["pick_end_effector_orientation"],
                    after_offset_z=params["pick_after_offset_z"],
                    **pick_kwargs,
                )
            else:
                action = self.pour_controller.forward(
                    articulation_controller=self.robot.get_articulation_controller(),
                    source_size=self.initial_size,
                    target_position=np.asarray(state['target_position'], dtype=float) + params["pour_target_position_offset"],
                    current_joint_velocities=self.robot.get_joint_velocities(),
                    pour_speed=params["pour_speed"],
                    source_name=state['object_name'],
                    gripper_position=state['gripper_position'],
                    target_end_effector_orientation=params["pour_end_effector_orientation"],
                )
            
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            return action, False, False

        print(f"{self.current_phase.value} task failed!")
        self._last_failure_reason = self._format_failure_reason()
        return self._finalize_collect_episode(False)

    def is_atomic_action_complete(self) -> bool:
        if self.mode != "collect":
            return True
        return self.current_phase == Phase.FINISHED

    def _step_infer(self, state):
        """Execute inference mode step."""
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        language_instruction = self.get_language_instruction()
        if language_instruction is not None:
            state['language_instruction'] = language_instruction
        else:
            state['language_instruction'] = "Pick up the graduated cylinder from the table and pour it into the big beaker"

        action = self.inference_engine.step_inference(state)
        success = self._check_phase_success()
        if success and self.current_phase == Phase.PICKING:
            print("Pick task success! Switching to pour...")
            self.current_phase = Phase.POURING
            self.inference_engine.trajectory_controller.reset()
        elif success and self.current_phase == Phase.POURING:
            print("Pour task success!")
            self._last_success = True
            self.current_phase = Phase.FINISHED
            return None, True, True
               
        return action, False, False

    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        clean_object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').lower()
        if "beaker" in clean_object_name:
            clean_object_name = "small beaker"
        self.language_instruction = f"Pick up the {clean_object_name} from the table and pour it into the big beaker".replace("  ", " ")
        return self.language_instruction

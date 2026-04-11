import re
from typing import Optional
from scipy.spatial.transform import Rotation as R
import numpy as np
from enum import Enum
from utils.task_utils import TaskUtils

from .atomic_actions.pick_controller import PickController
from .atomic_actions.pour_controller import PourController
from .base_controller import BaseController

class Phase(Enum):
    PICKING = "picking"
    POURING = "pouring"
    FINISHED = "finished"

class PourTaskController(BaseController):
    def __init__(self, cfg, robot):
        pick_cfg = getattr(cfg, "pick", None)
        pour_cfg = getattr(cfg, "pour", None)

        self._pick_events_dt = self._load_sequence(
            pick_cfg, "events_dt", [0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02], expected_len=7
        )
        self._pour_events_dt = self._load_sequence(
            pour_cfg, "events_dt", [0.006, 0.002, 0.009, 0.01, 0.009, 0.01], expected_len=6
        )
        self._pick_pre_offset_x = float(self._get_cfg_value(pick_cfg, "pre_offset_x", 0.05))
        self._pick_pre_offset_z = float(self._get_cfg_value(pick_cfg, "pre_offset_z", 0.05))
        self._pick_after_offset_z = float(self._get_cfg_value(pick_cfg, "after_offset_z", 0.5))
        self._pick_success_min_height_delta = float(
            self._get_cfg_value(pick_cfg, "success_min_height_delta", 0.12)
        )
        self._pick_collect_end_effector_orientation = R.from_euler(
            "xyz",
            np.radians(self._load_euler_deg(pick_cfg, "end_effector_euler_deg", [0.0, 90.0, 30.0])),
        ).as_quat()
        self._pick_infer_end_effector_orientation = R.from_euler(
            "xyz",
            np.radians(self._load_euler_deg(pick_cfg, "infer_end_effector_euler_deg", [0.0, 90.0, 15.0])),
        ).as_quat()

        self._pour_position_threshold = float(self._get_cfg_value(pour_cfg, "position_threshold", 0.006))
        self._pour_stage0_xy_threshold = float(self._get_cfg_value(pour_cfg, "stage0_xy_threshold", 0.08))
        self._pour_speed = float(self._get_cfg_value(pour_cfg, "pour_speed", -1.0))
        self._pour_success_distance_buffer = float(
            self._get_cfg_value(pour_cfg, "success_distance_buffer", 0.05)
        )
        self._pour_rotation_threshold_deg = float(
            self._get_cfg_value(pour_cfg, "pour_rotation_threshold_deg", 50.0)
        )
        self._return_rotation_threshold_deg = float(
            self._get_cfg_value(pour_cfg, "return_rotation_threshold_deg", 30.0)
        )
        self._return_height_delta = float(self._get_cfg_value(pour_cfg, "return_height_delta", 0.05))
        self._return_hold_seconds = float(self._get_cfg_value(pour_cfg, "return_hold_seconds", 1.0))
        self._return_timer_dt = float(self._get_cfg_value(pour_cfg, "return_timer_dt", 0.012))
        self._pour_height_range_1 = tuple(
            self._load_sequence(pour_cfg, "approach_height_range", [0.3, 0.4], expected_len=2)
        )
        self._pour_height_range_2 = tuple(
            self._load_sequence(pour_cfg, "pour_height_range", [0.1, 0.2], expected_len=2)
        )
        self._default_pour_euler_deg = self._load_euler_deg(
            pour_cfg, "end_effector_euler_deg", [0.0, 90.0, 10.0]
        )
        self._pour_end_effector_orientation = R.from_euler(
            "xyz",
            np.radians(self._default_pour_euler_deg),
        ).as_quat()

        self._episode_noise = {}
        self._episode_properties_set = False
        self._last_params_used = None
        self._last_baseline_correction = None
        self._pour_phase_params = None

        noise_cfg = getattr(cfg, "noise", None)
        if noise_cfg and getattr(noise_cfg, "enabled", False):
            self._noise_enabled = True
            self._noise_scale = float(getattr(noise_cfg, "noise_scale", 1.0))
            self._failure_bias_ratio = float(getattr(noise_cfg, "failure_bias_ratio", 0.0))
            self._noise_distribution = str(getattr(noise_cfg, "noise_distribution", "uniform"))
            self._pour_target_position_mode = str(
                getattr(noise_cfg, "pour_target_position_mode", "cartesian")
            ).lower()
            legacy_target_position_noise = list(getattr(noise_cfg, "target_position_noise", [-0.04, 0.04]))
            self._noise_range = {
                "pour_target_position_xy": list(
                    getattr(noise_cfg, "pour_target_position_xy_noise", legacy_target_position_noise)
                ),
                "pour_target_position_z": list(
                    getattr(noise_cfg, "pour_target_position_z_noise", [-0.01, 0.01])
                ),
                "pour_target_position_radius": list(
                    getattr(noise_cfg, "pour_target_position_radius_range", [0.04, 0.08])
                ),
                "approach_height_offset": list(getattr(noise_cfg, "approach_height_offset", [-0.03, 0.03])),
                "pour_height_offset": list(getattr(noise_cfg, "pour_height_offset", [-0.02, 0.02])),
                "euler_deg": list(getattr(noise_cfg, "end_effector_euler_deg", [-8.0, 8.0])),
                "pour_speed": list(getattr(noise_cfg, "pour_speed", [-0.2, 0.2])),
            }
            self._pour_target_position_angle_deg = list(
                getattr(noise_cfg, "pour_target_position_angle_deg", [0.0, 360.0])
            )
        else:
            self._noise_enabled = False
            self._pour_target_position_mode = "cartesian"

        if getattr(cfg, "mode", None) != "collect":
            self._noise_enabled = False

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

        if self._pour_target_position_mode == "radial_ring":
            theta_lo, theta_hi = self._pour_target_position_angle_deg[0], self._pour_target_position_angle_deg[1]
            theta_deg = float(np.random.uniform(theta_lo, theta_hi))
            theta_rad = np.deg2rad(theta_deg)
            radius = float(sample_in_range(*scaled_range("pour_target_position_radius")))
            target_position_noise = np.array(
                [
                    radius * np.cos(theta_rad),
                    radius * np.sin(theta_rad),
                    sample_in_range(*scaled_range("pour_target_position_z")),
                ],
                dtype=float,
            )
            self._episode_noise = {
                "pour_target_position": target_position_noise,
                "pour_target_position_radius": radius,
                "pour_target_position_theta_deg": theta_deg,
                "approach_height_offset": float(sample_in_range(*scaled_range("approach_height_offset"))),
                "pour_height_offset": float(sample_in_range(*scaled_range("pour_height_offset"))),
                "euler_deg": np.array(
                    [sample_in_range(*scaled_range("euler_deg")) for _ in range(3)],
                    dtype=float,
                ),
                "pour_speed": float(sample_in_range(*scaled_range("pour_speed"))),
            }
        else:
            self._episode_noise = {
                "pour_target_position": np.array(
                    [
                        sample_in_range(*scaled_range("pour_target_position_xy")),
                        sample_in_range(*scaled_range("pour_target_position_xy")),
                        sample_in_range(*scaled_range("pour_target_position_z")),
                    ],
                    dtype=float,
                ),
                "approach_height_offset": float(sample_in_range(*scaled_range("approach_height_offset"))),
                "pour_height_offset": float(sample_in_range(*scaled_range("pour_height_offset"))),
                "euler_deg": np.array(
                    [sample_in_range(*scaled_range("euler_deg")) for _ in range(3)],
                    dtype=float,
                ),
                "pour_speed": float(sample_in_range(*scaled_range("pour_speed"))),
            }

    @staticmethod
    def _snapshot_pour_state_for_task_props(state):
        snapshot = {}
        if state.get("object_position") is not None:
            snapshot["object_position"] = [float(x) for x in state["object_position"][:3]]
        if state.get("target_position") is not None:
            snapshot["target_position"] = [float(x) for x in state["target_position"][:3]]
        return snapshot

    def _build_pour_params(self, state):
        target_position = np.array(state["target_position"][:3], dtype=float)
        approach_height_range = np.array(self._pour_height_range_1, dtype=float)
        pour_height_range = np.array(self._pour_height_range_2, dtype=float)
        euler_deg = self._default_pour_euler_deg.copy()
        pour_speed = float(self._pour_speed)

        correction_gt = None
        if self._noise_enabled:
            if not self._episode_noise:
                self._sample_noise()
            n = self._episode_noise
            target_position = target_position + n["pour_target_position"]
            approach_height_range = approach_height_range + float(n["approach_height_offset"])
            pour_height_range = pour_height_range + float(n["pour_height_offset"])
            euler_deg = euler_deg + n["euler_deg"]
            pour_speed += float(n["pour_speed"])
            correction_gt = {
                "target_position_delta": (-n["pour_target_position"]).tolist(),
                "approach_height_offset": -float(n["approach_height_offset"]),
                "pour_height_offset": -float(n["pour_height_offset"]),
                "euler_deg": (-n["euler_deg"]).tolist(),
                "pour_speed": -float(n["pour_speed"]),
            }

        params_used = {
            "target_position": target_position.tolist(),
            "approach_height_range": approach_height_range.tolist(),
            "pour_height_range": pour_height_range.tolist(),
            "euler_deg": euler_deg.tolist(),
            "pour_speed": float(pour_speed),
        }
        return params_used, correction_gt

    def _maybe_set_pour_task_properties(self, state, params_used, correction_gt):
        if self._episode_properties_set or not hasattr(self.data_collector, "set_task_properties"):
            return

        props = {
            "action_type": "pour",
            "params_used": params_used,
            "object_type": self._get_object_type(state),
            "source_object_name": state.get("object_name"),
            "target_object_name": state.get("target_name"),
            "pour_target_position_mode": self._pour_target_position_mode,
        }
        sampled_object_position = state.get("sampled_object_position")
        if sampled_object_position is not None:
            props["sampled_object_position"] = [float(x) for x in sampled_object_position[:3]]
        props.update(self._snapshot_pour_state_for_task_props(state))

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

    def _prepare_pour_phase(self, state):
        if self._pour_phase_params is not None:
            return self._pour_phase_params

        params_used, correction_gt = self._build_pour_params(state)
        self._maybe_set_pour_task_properties(state, params_used, correction_gt)
        self.pour_controller.configure_episode_params(
            height_range_1=params_used["approach_height_range"],
            height_range_2=params_used["pour_height_range"],
            reset_random_heights=True,
        )
        self._pour_phase_params = {
            "target_position": np.array(params_used["target_position"], dtype=float),
            "end_effector_orientation": R.from_euler(
                "xyz", np.radians(np.asarray(params_used["euler_deg"], dtype=float))
            ).as_quat(),
            "pour_speed": float(params_used["pour_speed"]),
        }
        return self._pour_phase_params

    def _finalize_collect_episode(self, state, is_success):
        if self._episode_properties_set and hasattr(self.data_collector, "update_task_properties"):
            updates = {
                "is_success": bool(is_success),
                **self._snapshot_pour_state_for_task_props(state),
            }
            if not is_success and self._last_baseline_correction:
                updates["correction_gt"] = dict(self._last_baseline_correction)
            self.data_collector.update_task_properties(updates)

        if self._episode_properties_set:
            self.data_collector.write_cached_data(state["joint_positions"][:-1])
        else:
            self.data_collector.clear_cache()

        self._last_success = bool(is_success)
        self.current_phase = Phase.FINISHED
        return None, True, bool(is_success)
            
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        """Initialize controller for data collection mode."""
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=self._pick_events_dt,
        )

        self.pour_controller = PourController(
            name="pour_controller",
            cspace_controller=self.rmp_controller,
            events_dt=self._pour_events_dt,
            position_threshold=self._pour_position_threshold,
            stage0_xy_threshold=self._pour_stage0_xy_threshold,
            height_range_1=self._pour_height_range_1,
            height_range_2=self._pour_height_range_2,
        )
        self.active_controller = self.pick_controller

    def _init_infer_mode(self, cfg, robot=None):
        super()._init_infer_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=self._pick_events_dt,
        )

    def reset(self):
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
        self._episode_properties_set = False
        self._last_params_used = None
        self._last_baseline_correction = None
        self._pour_phase_params = None
        self.pick_controller.reset(events_dt=self._pick_events_dt)
        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.pour_controller.reset(events_dt=self._pour_events_dt)
            self._sample_noise()
        else:
            self.inference_engine.reset()

    def _check_phase_success(self):
        """Check if current phase is successful."""
        object_pos = self.state['object_position']
        self.last_error_info = None 
        
        if self.initial_position is None:
            raise ValueError("initial_position not set")

        if self.current_phase == Phase.PICKING:
            required_height = self.initial_position[2] + self._pick_success_min_height_delta
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
            pour_threshold = (
                self.task_utils.get_pour_threshold(self.state['object_name'], self.state['object_size'])
                + self._pour_success_distance_buffer
            )
            
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
                    threshold_degrees=self._pour_rotation_threshold_deg
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
                    threshold_degrees=self._return_rotation_threshold_deg
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
            if self.return_complete and object_pos[2] > self.initial_position[2] + self._return_height_delta:
                self.return_timer += self._return_timer_dt
                success = self.return_timer >= self._return_hold_seconds
                if not success:
                    self.last_error_info = {
                        'phase': 'POURING',
                        'error': 'Waiting for return timer',
                        'return_timer': self.return_timer,
                        'required_time': self._return_hold_seconds
                    }
                return success
            else:
                self.last_error_info = {
                    'phase': 'POURING',
                    'error': 'Object not in correct position for return timer',
                    'current_height': object_pos[2],
                    'required_height': self.initial_position[2] + self._return_height_delta,
                    'return_complete': self.return_complete
                }
                return False
        
        return False
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
        success = self._check_phase_success()
        if success:
            if self.current_phase == Phase.PICKING:
                print("Pick task success! Switching to pour...")
                self.current_phase = Phase.POURING
                self.active_controller = self.pour_controller
                return None, False, False
            if self.current_phase == Phase.POURING:
                print("Pour task success!")
                return self._finalize_collect_episode(state, True)

        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.active_controller.is_done():
            action = None
            if self.current_phase == Phase.PICKING:
                action = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=self._pick_collect_end_effector_orientation,
                    pre_offset_x=self._pick_pre_offset_x,
                    pre_offset_z=self._pick_pre_offset_z,
                    after_offset_z=self._pick_after_offset_z,
                )
            else:
                pour_phase_params = self._prepare_pour_phase(state)
                action = self.pour_controller.forward(
                    articulation_controller=self.robot.get_articulation_controller(),
                    source_size=self.initial_size,
                    target_position=pour_phase_params["target_position"],
                    current_joint_velocities=self.robot.get_joint_velocities(),
                    pour_speed=pour_phase_params["pour_speed"],
                    source_name=state['object_name'],
                    gripper_position=state['gripper_position'],
                    target_end_effector_orientation=pour_phase_params["end_effector_orientation"],
                )
                
                if 'camera_data' in state:
                    self.data_collector.cache_step(
                        camera_images=state['camera_data'],
                        joint_angles=state['joint_positions'][:-1],
                        language_instruction=self.get_language_instruction()
                    )
            
            return action, False, False

        if self.current_phase == Phase.PICKING:
            print("Pick task failed before entering pour phase.")
            if self.last_error_info is not None:
                print(f"Phase failure details: {self.last_error_info}")
            self.data_collector.clear_cache()
            self._last_success = False
            self.current_phase = Phase.FINISHED
            return None, True, False

        print("Pour task failed!")
        if self.last_error_info is not None:
            print(f"Phase failure details: {self.last_error_info}")
        return self._finalize_collect_episode(state, False)

    def _step_infer(self, state):
        """Execute inference mode step."""
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.pick_controller.is_done():
            action = None
            action = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=self._pick_infer_end_effector_orientation,
                )
            
        else:
            state['language_instruction'] = self.get_language_instruction()

            action = self.inference_engine.step_inference(state)
        success = self._check_phase_success()
        if success and self.current_phase == Phase.PICKING:
            print("Pick task success! Switching to pour...")
            self.current_phase = Phase.POURING
        elif success and self.current_phase == Phase.POURING:
            print("Pour task success!")
            self._last_success = True
            self.current_phase = Phase.FINISHED
            return None, True, True
               
        return action, False, False

    def get_language_instruction(self) -> Optional[str]:
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').lower()
        self.language_instruction = f"Pick up the {object_name} from the table and pour it into the target".replace("  ", " ")
        return self.language_instruction

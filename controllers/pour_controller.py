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
        self._pour_end_effector_orientation = R.from_euler(
            "xyz",
            np.radians(self._load_euler_deg(pour_cfg, "end_effector_euler_deg", [0.0, 90.0, 10.0])),
        ).as_quat()

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
        self.pick_controller.reset(events_dt=self._pick_events_dt)
        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.pour_controller.reset(events_dt=self._pour_events_dt)
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
            elif self.current_phase == Phase.POURING:
                print("Pour task success!")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.current_phase = Phase.FINISHED
                return None, True, True
        
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
                action = self.pour_controller.forward(
                    articulation_controller=self.robot.get_articulation_controller(),
                    source_size=self.initial_size,
                    target_position=state['target_position'],
                    current_joint_velocities=self.robot.get_joint_velocities(),
                    pour_speed=self._pour_speed,
                    source_name=state['object_name'],
                    gripper_position=state['gripper_position'],
                    target_end_effector_orientation=self._pour_end_effector_orientation,
                )
                
                if 'camera_data' in state:
                    self.data_collector.cache_step(
                        camera_images=state['camera_data'],
                        joint_angles=state['joint_positions'][:-1],
                        language_instruction=self.get_language_instruction()
                    )
            
            return action, False, False

        print(f"{self.current_phase.value} task failed!")
        if self.last_error_info is not None:
            print(f"Phase failure details: {self.last_error_info}")
        self.data_collector.clear_cache()
        self._last_success = False
        self.current_phase = Phase.FINISHED
        return None, True, False

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

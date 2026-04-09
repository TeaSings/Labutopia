import re
from scipy.spatial.transform import Rotation as R
import numpy as np
from enum import Enum

from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .base_controller import BaseController

class Phase(Enum):
    PICKING = "picking"
    PLACING = "placing"
    FINISHED = "finished"

class PlaceTaskController(BaseController):
    def __init__(self, cfg, robot):
        """Initialize the pick and pour task controller.
        
        Args:
            cfg: Configuration object containing controller settings
            robot: Robot instance to control
        """
        pick_cfg = getattr(cfg, "pick", None)
        place_cfg = getattr(cfg, "place", None)

        self._pick_events_dt = self._load_sequence(
            place_cfg, "pick_events_dt", [0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02], expected_len=7
        )
        self._place_events_dt = self._load_sequence(
            place_cfg, "place_events_dt", [0.005, 0.01, 0.08, 0.05, 0.01, 0.1], expected_len=6
        )
        self._pick_pre_offset_x = float(self._get_cfg_value(pick_cfg, "pre_offset_x", 0.05))
        self._pick_pre_offset_z = float(self._get_cfg_value(pick_cfg, "pre_offset_z", 0.05))
        self._pick_after_offset_z = float(self._get_cfg_value(pick_cfg, "after_offset_z", 0.15))
        self._pick_success_min_height_delta = float(
            self._get_cfg_value(place_cfg, "pick_success_min_height_delta", 0.10)
        )
        self._place_success_xy_threshold = float(
            self._get_cfg_value(place_cfg, "place_success_xy_threshold", 0.05)
        )
        self._place_success_z_threshold = float(
            self._get_cfg_value(place_cfg, "place_success_z_threshold", 0.05)
        )
        self._place_position_threshold = float(
            self._get_cfg_value(place_cfg, "position_threshold", 0.01)
        )
        self._place_release_position_threshold = float(
            self._get_cfg_value(place_cfg, "release_position_threshold", 0.02)
        )
        self._place_pre_place_z = float(self._get_cfg_value(place_cfg, "pre_place_z", 0.20))
        self._place_offset_z = float(self._get_cfg_value(place_cfg, "place_offset_z", 0.05))
        self._place_retreat_offset_x = float(self._get_cfg_value(place_cfg, "retreat_offset_x", -0.15))
        self._place_retreat_offset_z = float(self._get_cfg_value(place_cfg, "retreat_offset_z", 0.15))
        self._pick_end_effector_orientation = R.from_euler(
            "xyz",
            np.radians(self._load_euler_deg(pick_cfg, "end_effector_euler_deg", [0.0, 90.0, 30.0])),
        ).as_quat()
        self._place_end_effector_orientation = R.from_euler(
            "xyz",
            np.radians(self._load_euler_deg(place_cfg, "end_effector_euler_deg", [0.0, 90.0, 20.0])),
        ).as_quat()
        super().__init__(cfg, robot)
        self.initial_position = None
        self.initial_size = None
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
        """Initialize controller for data collection mode."""
        super()._init_collect_mode(cfg, robot)
        
        self.place_controller = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=self._place_events_dt,
            position_threshold=self._place_position_threshold,
        )
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=self._pick_events_dt,
        )
        self.active_controller = self.pick_controller

    def _init_infer_mode(self, cfg, robot):
        """Initialize controller for inference mode."""
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=self._pick_events_dt,
        )
        super()._init_infer_mode(cfg, robot)

    def reset(self):
        """Reset controller state and phase."""
        super().reset()
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.initial_size = None
        self.pick_controller.reset(events_dt=self._pick_events_dt)
        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.place_controller.reset(events_dt=self._place_events_dt)
        else:
            self.inference_engine.reset()

    def _check_phase_success(self):
        """Check if current phase is successful based on object position."""
        object_pos = self.state['object_position']
        target_position = self.state['target_position']
        if object_pos is None or target_position is None or self.initial_position is None:
            return False
        
        if self.current_phase == Phase.PICKING:
            return object_pos[2] > self.initial_position[2] + self._pick_success_min_height_delta
        elif self.current_phase == Phase.PLACING:
            success = (
                np.linalg.norm(object_pos[:2] - target_position[:2]) < self._place_success_xy_threshold
                and abs(object_pos[2] - self.initial_position[2]) < self._place_success_z_threshold
            )
            return success


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
                    end_effector_orientation=self._pick_end_effector_orientation,
                    pre_offset_x=self._pick_pre_offset_x,
                    pre_offset_z=self._pick_pre_offset_z,
                    after_offset_z=self._pick_after_offset_z,
                )
            else:
                action = self.place_controller.forward(
                    place_position = state['target_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=self._place_end_effector_orientation,
                    gripper_position=state['gripper_position'],
                    pre_place_z=self._place_pre_place_z,
                    place_offset_z=self._place_offset_z,
                    place_position_threshold=self._place_release_position_threshold,
                    retreat_offset_x=self._place_retreat_offset_x,
                    retreat_offset_z=self._place_retreat_offset_z,
                )
                if 'camera_data' in state:
                    self.data_collector.cache_step(
                        camera_images=state['camera_data'],
                        joint_angles=state['joint_positions'][:-1],
                        language_instruction=self.get_language_instruction()
                    )
            
            return action, False, False

        if success:
            if self.current_phase == Phase.PICKING:
                print("Pick task success! Switching to place...")
                self.current_phase = Phase.PLACING
                self.active_controller = self.place_controller
                return None, False, False
            elif self.current_phase == Phase.PLACING:
                print("Place task success!")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.current_phase = Phase.FINISHED
                return None, True, True
            else:
                print(f"{self.current_phase.value} task failed!")
                self.data_collector.clear_cache()
                self._last_success = False
                self.current_phase = Phase.FINISHED
                return None, True, False
        
        return None, False, False

    def _step_infer(self, state):
        """Execute inference mode step."""
        self.state = state
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success
        
        if self.current_phase == Phase.PICKING:
            action = None
            action = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=self._pick_end_effector_orientation,
                    pre_offset_x=self._pick_pre_offset_x,
                    pre_offset_z=self._pick_pre_offset_z,
                    after_offset_z=self._pick_after_offset_z,
                )    
        else:
            state['language_instruction'] = self.get_language_instruction()
            action = self.inference_engine.step_inference(state)
        return action, False, self.is_success()

    def is_success(self):
        object_pos = self.state["object_position"]
        target_position = self.state['target_position']
        if (
            object_pos is not None
            and target_position is not None
            and self.initial_position is not None
            and np.linalg.norm(object_pos[:2] - target_position[:2]) < self._place_success_xy_threshold
            and abs(object_pos[2] - self.initial_position[2]) < self._place_success_z_threshold
        ):
            self._last_success = True
            self.current_phase = Phase.FINISHED
            return True
        return False

    def get_language_instruction(self) -> str:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').replace('  ',' ').lower()
        self._language_instruction = f"Pick up the {object_name} from the table and place it at the target"
        return self._language_instruction

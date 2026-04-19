from typing import Optional
import numpy as np

from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.press_controller import PressController

class PressTaskController(BaseController):
    def __init__(self, cfg, robot):
        press_cfg = getattr(cfg, "press", None)
        self._press_events_dt = [0.005, 0.1, 0.005]
        self._press_initial_offset = 0.2
        self._press_distance = 0.04
        self._success_threshold_x = 0.405
        self._end_effector_euler_deg = [0.0, 90.0, 10.0]
        if press_cfg is not None:
            self._press_events_dt = [float(v) for v in getattr(press_cfg, "events_dt", self._press_events_dt)]
            self._press_initial_offset = float(getattr(press_cfg, "initial_offset", self._press_initial_offset))
            self._press_distance = float(getattr(press_cfg, "press_distance", self._press_distance))
            self._success_threshold_x = float(getattr(press_cfg, "success_threshold_x", self._success_threshold_x))
            self._end_effector_euler_deg = [
                float(v) for v in getattr(press_cfg, "end_effector_euler_deg", self._end_effector_euler_deg)
            ]
        self._end_effector_orientation = R.from_euler(
            "xyz", np.radians(self._end_effector_euler_deg)
        ).as_quat()
        super().__init__(cfg, robot, use_default_config=False)
        
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        self.press_controller = PressController(
            name="press_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            initial_offset=self._press_initial_offset,
            events_dt=self._press_events_dt,
        )

    def reset(self):
        super().reset()
        if self.mode == "collect":
            self.press_controller.reset(
                initial_offset=self._press_initial_offset,
                events_dt=self._press_events_dt,
            )
        else:
            self.inference_engine.reset()
    
    def step(self, state):
        self.state = state
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _check_success(self):
        final_object_position = self.object_utils.get_object_xform_position(
            object_path=self.cfg.sub_obj_path
        )
        if final_object_position is None:
            self._last_failure_reason = "Button position unavailable"
            return False
        pressed_x = float(final_object_position[0])
        success = pressed_x > self._success_threshold_x
        if not success:
            self._last_failure_reason = (
                f"Button press distance too short ({pressed_x:.4f}<={self._success_threshold_x:.4f})"
            )
        else:
            self._last_failure_reason = ""
        return success

    def _step_collect(self, state):
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
        
        if not self.press_controller.is_done():
            target_position = state.get('object_position')
            if target_position is None:
                self._last_failure_reason = "Target button position unavailable"
                self.data_collector.clear_cache()
                self.reset_needed = True
                self._last_success = False
                return None, True, False
            action = self.press_controller.forward(
                target_position=np.asarray(target_position, dtype=float).copy(),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=self._end_effector_orientation,
                press_distance=self._press_distance,
            )
            
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            return action, False, False
        
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            self.reset_needed = True
            return None, True, True
        else:
            if not self._last_failure_reason:
                self._last_failure_reason = (
                    f"Button not held beyond threshold for long enough "
                    f"({self.check_success_counter}/{self.REQUIRED_SUCCESS_STEPS})"
                )
            self.data_collector.clear_cache()
            self._last_success = False
            self.reset_needed = True
            return None, True, False
        
    def _step_infer(self, state):
        language_instruction = self.get_language_instruction()
        state['language_instruction'] = language_instruction

        action = self.inference_engine.step_inference(state)
        
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
            
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.reset_needed = True
            return action, True, True
        if self.press_controller.is_done() and not self._last_failure_reason:
            self._last_failure_reason = (
                f"Button not held beyond threshold for long enough "
                f"({self.check_success_counter}/{self.REQUIRED_SUCCESS_STEPS})"
            )
        return action, False, False

    def is_atomic_action_complete(self) -> bool:
        if self.mode == "collect":
            return self.press_controller.is_done()
        return True

    def get_language_instruction(self) -> Optional[str]:
        if self._language_instruction is None:
            return "Press the different color button"
        return self._language_instruction

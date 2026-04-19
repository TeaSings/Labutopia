from typing import Optional
import numpy as np

from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.press_controller import PressController

class PressTaskController(BaseController):
    def __init__(self, cfg, robot):
        press_cfg = getattr(cfg, "press", None)
        self._press_events_dt = self._load_sequence(press_cfg, "events_dt", [0.005, 0.1, 0.005], expected_len=3)
        self._press_initial_offset = float(self._get_cfg_value(press_cfg, "initial_offset", 0.2))
        self._press_distance = float(self._get_cfg_value(press_cfg, "press_distance", 0.04))
        self._success_threshold_x = float(self._get_cfg_value(press_cfg, "success_threshold_x", 0.405))
        self._default_press_euler_deg = self._load_euler_deg(
            press_cfg, "end_effector_euler_deg", [0.0, 90.0, 10.0]
        )
        self._episode_noise = {}
        self._episode_properties_set = False
        self._last_params_used = None
        self._last_baseline_correction = None
        self._press_params = None
        self._early_return = False

        noise_cfg = getattr(cfg, "noise", None)
        if noise_cfg and getattr(noise_cfg, "enabled", False):
            self._noise_enabled = True
            self._noise_scale = float(getattr(noise_cfg, "noise_scale", 1.0))
            self._failure_bias_ratio = float(getattr(noise_cfg, "failure_bias_ratio", 0.0))
            self._noise_distribution = str(getattr(noise_cfg, "noise_distribution", "uniform"))
            self._noise_range = {
                "initial_offset": list(getattr(noise_cfg, "initial_offset", [-0.03, 0.03])),
                "press_distance": list(getattr(noise_cfg, "press_distance", [-0.02, 0.02])),
                "end_effector_euler_deg": list(getattr(noise_cfg, "end_effector_euler_deg", [-8.0, 8.0])),
                "target_position_offset_y": list(getattr(noise_cfg, "target_position_offset_y", [-0.02, 0.02])),
                "target_position_offset_z": list(getattr(noise_cfg, "target_position_offset_z", [-0.02, 0.02])),
            }
        else:
            self._noise_enabled = False

        if getattr(cfg, "mode", None) != "collect":
            self._noise_enabled = False
        super().__init__(cfg, robot, use_default_config=False)

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
        return object_name.lower() if object_name else "unknown"

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
            "initial_offset": float(sample_in_range(*scaled_range("initial_offset"))),
            "press_distance": float(sample_in_range(*scaled_range("press_distance"))),
            "end_effector_euler_deg": np.array(
                [sample_in_range(*scaled_range("end_effector_euler_deg")) for _ in range(3)],
                dtype=float,
            ),
            "target_position_offset_y": float(sample_in_range(*scaled_range("target_position_offset_y"))),
            "target_position_offset_z": float(sample_in_range(*scaled_range("target_position_offset_z"))),
        }

    @staticmethod
    def _snapshot_press_state_for_task_props(state):
        snapshot = {}
        if state.get("object_position") is not None:
            snapshot["object_position"] = [float(x) for x in state["object_position"][:3]]
        sampled_object_position = state.get("sampled_object_position")
        if sampled_object_position is not None:
            snapshot["sampled_object_position"] = [float(x) for x in sampled_object_position[:3]]
        return snapshot

    def _build_press_params(self, state):
        initial_offset = float(self._press_initial_offset)
        press_distance = float(self._press_distance)
        euler_deg = self._default_press_euler_deg.copy()
        target_position = np.asarray(state["object_position"][:3], dtype=float).copy()

        correction_gt = None
        target_position_offset_y = 0.0
        target_position_offset_z = 0.0
        if self._noise_enabled:
            if not self._episode_noise:
                self._sample_noise()
            n = self._episode_noise
            initial_offset += float(n["initial_offset"])
            press_distance += float(n["press_distance"])
            euler_deg = euler_deg + n["end_effector_euler_deg"]
            target_position_offset_y = float(n["target_position_offset_y"])
            target_position_offset_z = float(n["target_position_offset_z"])
            target_position[1] += target_position_offset_y
            target_position[2] += target_position_offset_z
            correction_gt = {
                "initial_offset": -float(n["initial_offset"]),
                "press_distance": -float(n["press_distance"]),
                "end_effector_euler_deg": (-n["end_effector_euler_deg"]).tolist(),
                "target_position_offset_y": -target_position_offset_y,
                "target_position_offset_z": -target_position_offset_z,
            }

        params_used = {
            "initial_offset": float(initial_offset),
            "press_distance": float(press_distance),
            "end_effector_euler_deg": euler_deg.tolist(),
            "target_position": [float(x) for x in target_position.tolist()],
            "target_position_offset_y": float(target_position_offset_y),
            "target_position_offset_z": float(target_position_offset_z),
            "success_threshold_x": float(self._success_threshold_x),
        }
        return params_used, correction_gt

    def _maybe_set_press_task_properties(self, state, params_used, correction_gt):
        if self._episode_properties_set or not hasattr(self.data_collector, "set_task_properties"):
            return

        props = {
            "action_type": "press_button",
            "params_used": params_used,
            "object_type": self._get_object_type(state),
            "source_object_name": state.get("object_name"),
        }
        props.update(self._snapshot_press_state_for_task_props(state))

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

    def _prepare_press_episode(self, state):
        if self._press_params is not None:
            return self._press_params

        params_used, correction_gt = self._build_press_params(state)
        self._maybe_set_press_task_properties(state, params_used, correction_gt)
        self.press_controller.reset(
            initial_offset=params_used["initial_offset"],
            events_dt=self._press_events_dt,
        )
        self._press_params = {
            "target_position": np.asarray(params_used["target_position"], dtype=float),
            "press_distance": float(params_used["press_distance"]),
            "end_effector_orientation": R.from_euler(
                "xyz",
                np.radians(np.asarray(params_used["end_effector_euler_deg"], dtype=float)),
            ).as_quat(),
        }
        return self._press_params

    def _finalize_press_episode(self, state, success):
        if self._episode_properties_set and hasattr(self.data_collector, "update_task_properties"):
            self.data_collector.update_task_properties({"is_success": bool(success)})
        self.data_collector.write_cached_data(state["joint_positions"][:-1])
        
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
        self._episode_noise = {}
        self._episode_properties_set = False
        self._last_params_used = None
        self._last_baseline_correction = None
        self._press_params = None
        self._early_return = False
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
                self._early_return = True
                self.data_collector.clear_cache()
                self.reset_needed = True
                self._last_success = False
                return None, True, False
            press_params = self._prepare_press_episode(state)
            action = self.press_controller.forward(
                target_position=press_params["target_position"].copy(),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=press_params["end_effector_orientation"],
                press_distance=press_params["press_distance"],
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
            self._finalize_press_episode(state, True)
            self.reset_needed = True
            return None, True, True
        else:
            if not self._last_failure_reason:
                self._last_failure_reason = (
                    f"Button not held beyond threshold for long enough "
                    f"({self.check_success_counter}/{self.REQUIRED_SUCCESS_STEPS})"
                )
            self._finalize_press_episode(state, False)
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
        if hasattr(self, "press_controller") and self.press_controller.is_done() and not self._last_failure_reason:
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

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
        """Initialize the pick and place task controller.
        
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
        self._default_place_euler_deg = self._load_euler_deg(
            place_cfg, "end_effector_euler_deg", [0.0, 90.0, 20.0]
        )
        self._place_end_effector_orientation = R.from_euler(
            "xyz",
            np.radians(self._default_place_euler_deg),
        ).as_quat()

        self._episode_noise = {}
        self._episode_properties_set = False
        self._last_params_used = None
        self._last_baseline_correction = None

        noise_cfg = getattr(cfg, "noise", None)
        if noise_cfg and getattr(noise_cfg, "enabled", False):
            self._noise_enabled = True
            self._noise_scale = float(getattr(noise_cfg, "noise_scale", 1.0))
            self._failure_bias_ratio = float(getattr(noise_cfg, "failure_bias_ratio", 0.0))
            self._noise_distribution = str(getattr(noise_cfg, "noise_distribution", "uniform"))
            self._place_position_mode = str(getattr(noise_cfg, "place_position_mode", "cartesian")).lower()
            legacy_place_position_noise = list(getattr(noise_cfg, "place_position_noise", [-0.03, 0.03]))
            self._noise_range = {
                "place_position_xy": list(
                    getattr(noise_cfg, "place_position_xy_noise", legacy_place_position_noise)
                ),
                "place_position_z": list(
                    getattr(noise_cfg, "place_position_z_noise", legacy_place_position_noise)
                ),
                "place_position_radius": list(
                    getattr(noise_cfg, "place_position_radius_range", [0.04, 0.07])
                ),
                "pre_place_z": list(getattr(noise_cfg, "pre_place_z", [-0.04, 0.04])),
                "place_offset_z": list(getattr(noise_cfg, "place_offset_z", [-0.03, 0.03])),
                "euler_deg": list(getattr(noise_cfg, "end_effector_euler_deg", [-10.0, 10.0])),
            }
            self._place_position_angle_deg = list(
                getattr(noise_cfg, "place_position_angle_deg", [0.0, 360.0])
            )
        else:
            self._noise_enabled = False
            self._place_position_mode = "cartesian"

        if getattr(cfg, "mode", None) != "collect":
            self._noise_enabled = False

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
        """Sample one place-stage noise packet per episode."""
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

        if self._place_position_mode == "radial_ring":
            theta_lo, theta_hi = self._place_position_angle_deg[0], self._place_position_angle_deg[1]
            theta_deg = float(np.random.uniform(theta_lo, theta_hi))
            theta_rad = np.deg2rad(theta_deg)
            radius = float(sample_in_range(*scaled_range("place_position_radius")))
            place_position_noise = np.array(
                [
                    radius * np.cos(theta_rad),
                    radius * np.sin(theta_rad),
                    sample_in_range(*scaled_range("place_position_z")),
                ],
                dtype=float,
            )
            self._episode_noise = {
                "place_position": place_position_noise,
                "place_position_radius": radius,
                "place_position_theta_deg": theta_deg,
                "pre_place_z": float(sample_in_range(*scaled_range("pre_place_z"))),
                "place_offset_z": float(sample_in_range(*scaled_range("place_offset_z"))),
                "euler_deg": np.array(
                    [sample_in_range(*scaled_range("euler_deg")) for _ in range(3)],
                    dtype=float,
                ),
            }
        else:
            self._episode_noise = {
                "place_position": np.array(
                    [
                        sample_in_range(*scaled_range("place_position_xy")),
                        sample_in_range(*scaled_range("place_position_xy")),
                        sample_in_range(*scaled_range("place_position_z")),
                    ],
                    dtype=float,
                ),
                "pre_place_z": float(sample_in_range(*scaled_range("pre_place_z"))),
                "place_offset_z": float(sample_in_range(*scaled_range("place_offset_z"))),
                "euler_deg": np.array(
                    [sample_in_range(*scaled_range("euler_deg")) for _ in range(3)],
                    dtype=float,
                ),
            }

    @staticmethod
    def _snapshot_place_state_for_task_props(state):
        snapshot = {}
        if state.get("object_position") is not None:
            snapshot["object_position"] = [float(x) for x in state["object_position"][:3]]
        if state.get("target_position") is not None:
            snapshot["target_position"] = [float(x) for x in state["target_position"][:3]]
        return snapshot

    def _build_place_params(self, state):
        """Build the actual place-stage parameters used this episode."""
        target_position = np.array(state["target_position"][:3], dtype=float)
        place_position = target_position.copy()
        pre_place_z = self._place_pre_place_z
        place_offset_z = self._place_offset_z
        euler_deg = self._default_place_euler_deg.copy()

        correction_gt = None
        if self._noise_enabled:
            if not self._episode_noise:
                self._sample_noise()
            n = self._episode_noise
            place_position = target_position + n["place_position"]
            pre_place_z += n["pre_place_z"]
            place_offset_z += n["place_offset_z"]
            euler_deg = euler_deg + n["euler_deg"]
            correction_gt = {
                "place_position_delta": (-n["place_position"]).tolist(),
                "pre_place_z": -float(n["pre_place_z"]),
                "place_offset_z": -float(n["place_offset_z"]),
                "euler_deg": (-n["euler_deg"]).tolist(),
            }

        params_used = {
            "place_position": place_position.tolist(),
            "pre_place_z": float(pre_place_z),
            "place_offset_z": float(place_offset_z),
            "euler_deg": euler_deg.tolist(),
        }
        end_effector_orientation = R.from_euler("xyz", np.radians(euler_deg)).as_quat()
        return params_used, correction_gt, place_position, end_effector_orientation

    def _maybe_set_place_task_properties(self, state, params_used, correction_gt):
        if self._episode_properties_set or not hasattr(self.data_collector, "set_task_properties"):
            return

        props = {
            "action_type": "place",
            "params_used": params_used,
            "object_type": self._get_object_type(state),
            "source_object_name": state.get("object_name"),
            "target_object_name": state.get("target_name"),
            "place_position_mode": self._place_position_mode,
        }
        sampled_object_position = state.get("sampled_object_position")
        if sampled_object_position is not None:
            props["sampled_object_position"] = [float(x) for x in sampled_object_position[:3]]
        props.update(self._snapshot_place_state_for_task_props(state))

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

    def _finalize_collect_episode(self, state, is_success):
        if self._episode_properties_set and hasattr(self.data_collector, "update_task_properties"):
            updates = {
                "is_success": bool(is_success),
                **self._snapshot_place_state_for_task_props(state),
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
        self._episode_properties_set = False
        self._last_params_used = None
        self._last_baseline_correction = None
        self.pick_controller.reset(events_dt=self._pick_events_dt)
        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.place_controller.reset(events_dt=self._place_events_dt)
            self._sample_noise()
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
                params_used, correction_gt, place_position, place_orientation = self._build_place_params(state)
                self._maybe_set_place_task_properties(state, params_used, correction_gt)
                action = self.place_controller.forward(
                    place_position=place_position,
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=place_orientation,
                    gripper_position=state['gripper_position'],
                    pre_place_z=params_used["pre_place_z"],
                    place_offset_z=params_used["place_offset_z"],
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
                return self._finalize_collect_episode(state, True)
            else:
                print(f"{self.current_phase.value} task failed!")
                return self._finalize_collect_episode(state, False)

        if self.current_phase == Phase.PICKING:
            print("Pick task failed before entering place phase.")
            self.data_collector.clear_cache()
            self._last_success = False
            self.current_phase = Phase.FINISHED
            return None, True, False

        if self.current_phase == Phase.PLACING:
            print("Place task failed!")
            return self._finalize_collect_episode(state, False)
        
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

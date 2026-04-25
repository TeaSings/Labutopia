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

class PickPlaceTaskController(BaseController):
    def __init__(self, cfg, robot):
        """Initialize the pick and pour task controller.
        
        Args:
            cfg: Configuration object containing controller settings
            robot: Robot instance to control
        """
        super().__init__(cfg, robot)
        self.initial_position = None
        self.initial_size = None
        self.current_phase = Phase.PICKING
        self._episode_noise = {}
        self._episode_properties_set = False
        self._last_params_used = None
        self._last_baseline_correction = None
        self._pickplace_params = None
        noise_cfg = getattr(cfg, "noise", None)
        if noise_cfg and getattr(noise_cfg, "enabled", False) and getattr(cfg, "mode", None) == "collect":
            self._noise_enabled = True
            self._noise_scale = float(getattr(noise_cfg, "noise_scale", 1.0))
            self._failure_bias_ratio = float(getattr(noise_cfg, "failure_bias_ratio", 0.0))
            self._noise_distribution = str(getattr(noise_cfg, "noise_distribution", "uniform"))
            self._noise_range = {
                "pick_target_position_xy": list(getattr(noise_cfg, "pick_target_position_xy_noise", [-0.01, 0.01])),
                "pick_target_position_z": list(getattr(noise_cfg, "pick_target_position_z_noise", [-0.005, 0.005])),
                "place_target_position_xy": list(getattr(noise_cfg, "place_target_position_xy_noise", [-0.04, 0.04])),
                "place_target_position_z": list(getattr(noise_cfg, "place_target_position_z_noise", [-0.005, 0.005])),
                "pick_pre_offset_x": list(getattr(noise_cfg, "pick_pre_offset_x", [-0.01, 0.01])),
                "pick_pre_offset_z": list(getattr(noise_cfg, "pick_pre_offset_z", [-0.01, 0.01])),
                "pick_after_offset_z": list(getattr(noise_cfg, "pick_after_offset_z", [-0.02, 0.02])),
                "place_offset_z": list(getattr(noise_cfg, "place_offset_z", [-0.02, 0.02])),
                "gripper_distance": list(getattr(noise_cfg, "gripper_distance", [-0.003, 0.003])),
                "pick_end_effector_euler_deg": list(getattr(noise_cfg, "pick_end_effector_euler_deg", [-6.0, 6.0])),
                "place_end_effector_euler_deg": list(getattr(noise_cfg, "place_end_effector_euler_deg", [-6.0, 6.0])),
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
        
        self.place_controller = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
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
            "place_target_position": np.array([
                sample_in_range(*scaled_range("place_target_position_xy")),
                sample_in_range(*scaled_range("place_target_position_xy")),
                sample_in_range(*scaled_range("place_target_position_z")),
            ], dtype=float),
            "pick_pre_offset_x": float(sample_in_range(*scaled_range("pick_pre_offset_x"))),
            "pick_pre_offset_z": float(sample_in_range(*scaled_range("pick_pre_offset_z"))),
            "pick_after_offset_z": float(sample_in_range(*scaled_range("pick_after_offset_z"))),
            "place_offset_z": float(sample_in_range(*scaled_range("place_offset_z"))),
            "gripper_distance": float(sample_in_range(*scaled_range("gripper_distance"))),
            "pick_end_effector_euler_deg": np.array([
                sample_in_range(*scaled_range("pick_end_effector_euler_deg")) for _ in range(3)
            ], dtype=float),
            "place_end_effector_euler_deg": np.array([
                sample_in_range(*scaled_range("place_end_effector_euler_deg")) for _ in range(3)
            ], dtype=float),
        }

    @staticmethod
    def _to_jsonable(value):
        return value.tolist() if hasattr(value, "tolist") else value

    def _build_pickplace_params(self, state):
        base_gripper_distance = float(self.pick_controller.get_gripper_distance(state["object_name"]))
        params_used = {
            "pick_target_position_offset": [0.0, 0.0, 0.0],
            "place_target_position_offset": [0.0, 0.0, 0.0],
            "pick_pre_offset_x": 0.05,
            "pick_pre_offset_z": 0.05,
            "pick_after_offset_z": 0.15,
            "place_pre_z": 0.2,
            "place_offset_z": 0.05,
            "place_position_threshold": 0.02,
            "place_retreat_offset_x": -0.15,
            "place_retreat_offset_z": 0.15,
            "gripper_distance": base_gripper_distance,
            "pick_end_effector_euler_deg": [0.0, 90.0, 30.0],
            "place_end_effector_euler_deg": [0.0, 90.0, 20.0],
        }
        correction_gt = None

        if self._noise_enabled:
            if not self._episode_noise:
                self._sample_noise()
            n = self._episode_noise
            pick_offset = n["pick_target_position"]
            place_offset = n["place_target_position"]
            params_used["pick_target_position_offset"] = pick_offset.tolist()
            params_used["place_target_position_offset"] = place_offset.tolist()
            params_used["pick_pre_offset_x"] += float(n["pick_pre_offset_x"])
            params_used["pick_pre_offset_z"] += float(n["pick_pre_offset_z"])
            params_used["pick_after_offset_z"] += float(n["pick_after_offset_z"])
            params_used["place_offset_z"] += float(n["place_offset_z"])
            params_used["gripper_distance"] += float(n["gripper_distance"])
            params_used["pick_end_effector_euler_deg"] = (
                np.asarray(params_used["pick_end_effector_euler_deg"], dtype=float)
                + n["pick_end_effector_euler_deg"]
            ).tolist()
            params_used["place_end_effector_euler_deg"] = (
                np.asarray(params_used["place_end_effector_euler_deg"], dtype=float)
                + n["place_end_effector_euler_deg"]
            ).tolist()
            correction_gt = {
                "pick_target_position_offset": (-pick_offset).tolist(),
                "place_target_position_offset": (-place_offset).tolist(),
                "pick_pre_offset_x": -float(n["pick_pre_offset_x"]),
                "pick_pre_offset_z": -float(n["pick_pre_offset_z"]),
                "pick_after_offset_z": -float(n["pick_after_offset_z"]),
                "place_offset_z": -float(n["place_offset_z"]),
                "gripper_distance": -float(n["gripper_distance"]),
                "pick_end_effector_euler_deg": (-n["pick_end_effector_euler_deg"]).tolist(),
                "place_end_effector_euler_deg": (-n["place_end_effector_euler_deg"]).tolist(),
            }

        return params_used, correction_gt

    def _maybe_set_pickplace_task_properties(self, state, params_used, correction_gt):
        if self._episode_properties_set or not hasattr(self.data_collector, "set_task_properties"):
            return

        props = {
            "action_type": "transport_beaker",
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

    def _prepare_pickplace_episode(self, state):
        if self._pickplace_params is not None:
            return self._pickplace_params

        params_used, correction_gt = self._build_pickplace_params(state)
        self._maybe_set_pickplace_task_properties(state, params_used, correction_gt)
        self._pickplace_params = {
            **params_used,
            "pick_target_position_offset": np.asarray(params_used["pick_target_position_offset"], dtype=float),
            "place_target_position_offset": np.asarray(params_used["place_target_position_offset"], dtype=float),
            "pick_end_effector_orientation": R.from_euler(
                'xyz',
                np.radians(params_used["pick_end_effector_euler_deg"]),
            ).as_quat(),
            "place_end_effector_orientation": R.from_euler(
                'xyz',
                np.radians(params_used["place_end_effector_euler_deg"]),
            ).as_quat(),
        }
        return self._pickplace_params

    def reset(self):
        """Reset controller state and phase."""
        super().reset()
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.initial_size = None
        self._episode_noise = {}
        self._episode_properties_set = False
        self._last_params_used = None
        self._last_baseline_correction = None
        self._pickplace_params = None
        
        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.pick_controller.reset()
            self.place_controller.reset()
            self._sample_noise()
        else:
            self.inference_engine.reset()

    def _check_phase_success(self):
        """Check if current phase is successful based on object position."""
        object_pos = self.state['object_position']
        target_position = self.state['target_position']
        
        if self.current_phase == Phase.PICKING:
            return object_pos[2] > self.initial_position[2] + 0.1
        elif self.current_phase == Phase.PLACING:
            success = (np.linalg.norm(object_pos[:2] - target_position[:2]) < 0.05 and 
                        abs(object_pos[2] - self.initial_position[2]) < 0.05)
            return success

    def _set_phase_failure_reason(self):
        object_pos = np.asarray(self.state['object_position'], dtype=float)
        target_position = np.asarray(self.state['target_position'], dtype=float)

        if self.current_phase == Phase.PICKING:
            lift_height = object_pos[2] - self.initial_position[2]
            self._last_failure_reason = f"Pick lift height too low ({lift_height:.4f}<0.1000)"
            return

        xy_distance = np.linalg.norm(object_pos[:2] - target_position[:2])
        z_distance = abs(object_pos[2] - self.initial_position[2])
        reasons = []
        if xy_distance >= 0.05:
            reasons.append(f"Object too far from target platform ({xy_distance:.4f}>=0.0500)")
        if z_distance >= 0.05:
            reasons.append(f"Object height not settled ({z_distance:.4f}>=0.0500)")
        self._last_failure_reason = " and ".join(reasons) if reasons else "Pick-place phase failed"

    def _finalize_collect_episode(self, is_success):
        if hasattr(self.data_collector, "update_task_properties"):
            updates = {
                "is_success": bool(is_success),
                "final_object_position": self._to_jsonable(np.asarray(self.state["object_position"], dtype=float)),
                "final_target_position": self._to_jsonable(np.asarray(self.state["target_position"], dtype=float)),
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
        params = self._prepare_pickplace_episode(state)
        success = self._check_phase_success()
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.active_controller.is_done():
            action = None
            if self.current_phase == Phase.PICKING:
                action = self.pick_controller.forward(
                    picking_position=np.asarray(state['object_position'], dtype=float) + params["pick_target_position_offset"],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=params["pick_end_effector_orientation"],
                    pre_offset_x=params["pick_pre_offset_x"],
                    pre_offset_z=params["pick_pre_offset_z"],
                    after_offset_z=params["pick_after_offset_z"],
                    gripper_distances=params["gripper_distance"],
                )
            else:
                action = self.place_controller.forward(
                    place_position=np.asarray(state['target_position'], dtype=float) + params["place_target_position_offset"],
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=params["place_end_effector_orientation"],
                    gripper_position=state['gripper_position'],
                    pre_place_z=params["place_pre_z"],
                    place_offset_z=params["place_offset_z"],
                    place_position_threshold=params["place_position_threshold"],
                    retreat_offset_x=params["place_retreat_offset_x"],
                    retreat_offset_z=params["place_retreat_offset_z"],
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
                print("Pick task success! Switching to pour...")
                self.current_phase = Phase.PLACING
                self.active_controller = self.place_controller
                return None, False, False
            elif self.current_phase == Phase.PLACING:
                print("Pour task success!")
                return self._finalize_collect_episode(True)

        self._set_phase_failure_reason()
        print(f"{self.current_phase.value} task failed!")
        self.print_failure_reason()
        return self._finalize_collect_episode(False)
        
        return None, False, False

    def is_atomic_action_complete(self) -> bool:
        if self.mode != "collect":
            return True
        if self.current_phase == Phase.FINISHED:
            return True
        return bool(self.active_controller.is_done())

    def _step_infer(self, state):
        """Execute inference mode step."""
        self.state = state
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        state['language_instruction'] = self.get_language_instruction()

        action = self.inference_engine.step_inference(state)

        return action, False, self.is_success()

    def is_success(self):
        object_pos = self.state["object_position"]
        target_position = self.state['target_position']
        if (np.linalg.norm(object_pos[:2] - target_position[:2]) < 0.05 and abs(object_pos[2] - self.initial_position[2]) < 0.02
            and np.linalg.norm(self.state["gripper_position"] - object_pos) > 0.05):
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

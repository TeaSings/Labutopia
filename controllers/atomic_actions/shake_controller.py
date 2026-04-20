from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_stage_units

import numpy as np
import typing

class ShakeController(BaseController):
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        shake_distance: float = 0.1,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._events_dt = self._validate_events_dt(events_dt)
        self._cspace_controller = cspace_controller
        self._shake_distance = float(shake_distance) / get_stage_units()
        self._initial_position = None

    @staticmethod
    def _validate_events_dt(events_dt: typing.Optional[typing.List[float]]) -> typing.List[float]:
        if events_dt is None:
            values = [0.02, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.015]
        else:
            values = events_dt
        if not isinstance(values, (np.ndarray, list)):
            raise Exception("events_dt is not a list or numpy array")
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if len(values) != 10:
            raise Exception(f"events_dt length is not 10, got {len(values)}")
        return list(values)

    def forward(
        self,
        current_joint_positions: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        initial_position: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        if initial_position is not None and self._initial_position is None:
            self._initial_position = np.asarray(initial_position, dtype=float).copy()
        if self._initial_position is None:
            self._initial_position = np.array([0.25, 0.0, 1.0], dtype=float)

        if self._event == 0:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self._initial_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 1:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self._initial_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 2:
            target_position = self._initial_position + np.array([0, -self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 3:
            target_position = self._initial_position + np.array([0, self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 4:
            target_position = self._initial_position + np.array([0, -self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 5:
            target_position = self._initial_position + np.array([0, self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 6:
            target_position = self._initial_position + np.array([0, -self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 7:
            target_position = self._initial_position + np.array([0, self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 8:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self._initial_position,
                target_end_effector_orientation=end_effector_orientation
            )
        else:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions = ArticulationAction(
                joint_positions=target_joint_positions,
                joint_velocities=None,
            )
        
        if self._event < len(self._events_dt):
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0

        return target_joint_positions

    def reset(
        self,
        events_dt: typing.Optional[typing.List[float]] = None,
        shake_distance: typing.Optional[float] = None,
        initial_position: typing.Optional[np.ndarray] = None,
    ) -> None:
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._events_dt = self._validate_events_dt(events_dt)
        if shake_distance is not None:
            self._shake_distance = float(shake_distance) / get_stage_units()
        self._initial_position = None if initial_position is None else np.asarray(initial_position, dtype=float).copy()

    def is_done(self) -> bool:
        return self._event >= len(self._events_dt)

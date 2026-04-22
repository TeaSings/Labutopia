import numpy as np
from .base_task import BaseTask


class StirTask(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.glass_rod = self.cfg.obj_path
        self.target_beaker = self.cfg.target_path
        self.glass_rod_mesh = self.cfg.sub_obj_path
        task_cfg = getattr(cfg, "task", None)
        sampling_xy = getattr(task_cfg, "target_position_range_xy", [0.075, 0.075]) if task_cfg is not None else [0.075, 0.075]
        self._target_sampling_xy = np.asarray(list(sampling_xy), dtype=float).reshape(-1)
        if self._target_sampling_xy.size != 2:
            self._target_sampling_xy = np.asarray([0.075, 0.075], dtype=float)
        self._max_steps = int(getattr(task_cfg, "max_steps", 2000)) if task_cfg is not None else 2000
        self._target_base_position = np.asarray([0.24125, -0.31358, 0.77], dtype=float)
        self._sampled_target_position = self._target_base_position.copy()

    def reset(self):
        super().reset()
        self.robot.initialize()

        sampled_target_position = self._target_base_position.copy()
        sampled_target_position[0] += np.random.uniform(-self._target_sampling_xy[0], self._target_sampling_xy[0])
        sampled_target_position[1] += np.random.uniform(-self._target_sampling_xy[1], self._target_sampling_xy[1])
        self._sampled_target_position = sampled_target_position
        self.object_utils.set_object_position(object_path=self.target_beaker, position=sampled_target_position)

        rack_position = np.array([0.28421, 0.30755, 0.82291])
        self.object_utils.set_object_position(object_path="/World/test_tube_rack", position=rack_position)

        self.object_utils.set_object_position(object_path=self.glass_rod, position=rack_position + np.array([-0.01152, -0.1125, 0.03197]))
        self.object_utils.set_object_position(object_path=self.glass_rod_mesh, position=[0, 0, 0])

    def step(self):
        self.frame_idx += 1

        if not self.check_frame_limits(max_steps=self._max_steps):
            return None

        return self.get_basic_state_info(
            object_path=self.glass_rod,
            target_path=self.target_beaker,
            additional_info={
                "target_beaker": self.target_beaker,
                "object_position": self.object_utils.get_object_xform_position(self.glass_rod),
                "glass_rod_position": self.object_utils.get_object_xform_position(object_path=self.cfg.sub_obj_path),
                "sampled_target_position": self._sampled_target_position.copy(),
            },
        )

import numpy as np
from .base_task import BaseTask

import random
from utils.Material_utils import bind_material_to_object

class PressTask(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)

        self.object_utils.set_object_position(object_path=self.cfg.instrument_path, position=np.array([0.73, -0.1, 0.64]))

        self.target_button_path = self.cfg.target_button_path
        self.sub_button_path = self.cfg.sub_obj_path
        self.distractor_button1_path = self.cfg.distractor_button1_path
        self.distractor_button2_path = self.cfg.distractor_button2_path

        self.button_material_paths = self.cfg.button_material_paths

        self.button_types = self.cfg.button_types
        task_cfg = getattr(self.cfg, "task", None)
        self.max_steps = int(getattr(task_cfg, "max_steps", 1000))
        self.randomize_button_material = bool(getattr(task_cfg, "randomize_button_material", True))
                    
    def reset(self):
        super().reset()
        self.robot.initialize()

        self.position1 = np.array([0.40, random.uniform(-0.06, 0.04), 1.1 + np.random.uniform(-0.1, 0.1)])
        self.position2 = self.position1 + np.array([0.0, random.uniform(-0.25, -0.15), 0])
        self.position3 = self.position1 + np.array([0.0, random.uniform(-0.40, -0.30), 0])

        positions = [self.position1, self.position2, self.position3]
        random.shuffle(positions)  

        self.object_utils.set_object_position(object_path=self.target_button_path, position=positions[0])
        self.object_utils.set_object_position(object_path=self.distractor_button1_path, position=positions[1])
        self.object_utils.set_object_position(object_path=self.distractor_button2_path, position=positions[2])

        if self.randomize_button_material:
            random_material_path = random.choice(self.button_material_paths[:self.button_types])
            bind_material_to_object(stage=self.stage,
                                    obj_path=self.cfg.sub_obj_path,
                                    material_path=random_material_path)
        
    def step(self):
        self.frame_idx += 1
        if not self.check_frame_limits(max_steps=self.max_steps):
            return None
            
        object_position = self.object_utils.get_geometry_center(object_path=self.sub_button_path)
        if object_position is None:
            object_position = self.object_utils.get_object_xform_position(object_path=self.target_button_path)
        
        return self.get_basic_state_info(
            object_path=self.sub_button_path,
            additional_info={
                'object_position': object_position,
                'source_object_name': self.target_button_path.split("/")[-1],
            }
        )

from .single_object_task import SingleObjectTask


class OpenCloseTask(SingleObjectTask):
    """Task wrapper for opening doors and drawers."""

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)

    def reset(self):
        """Reset task state and resolve the current handle prim path."""
        super().reset()

        current_obj_cfg = self.obj_configs[self.current_obj_idx] if self.obj_configs else {}
        handle_path = current_obj_cfg.get("handle_path")
        if handle_path is None:
            handle_path = self.cfg.get("handle_path")
        if handle_path is None:
            handle_path = self.current_obj_path + "/handle"
        self.current_sub_obj_path = handle_path

    def step(self):
        """Execute one simulation step and return current task state."""
        self.frame_idx += 1

        if not self.check_frame_limits():
            return None

        object_position = self.object_utils.get_geometry_center(object_path=self.current_sub_obj_path)
        object_size = self.object_utils.get_object_size(object_path=self.current_sub_obj_path)

        close_gripper_distance = self.obj_configs[self.current_obj_idx].get('close_gripper_distance', 0.023)

        if self.cfg.task.get("operate_type") == "door":
            return self.get_basic_state_info(
                object_path=self.current_obj_path,
                additional_info={
                    'object_position': object_position,
                    'object_size': object_size,
                    'revolute_joint_position': self.object_utils.get_revolute_joint_positions(
                        joint_path=self.current_obj_path + "/RevoluteJoint"
                    ),
                    'close_gripper_distance': close_gripper_distance
                }
            )

        return self.get_basic_state_info(
            object_path=self.current_obj_path,
            additional_info={
                'object_position': object_position,
                'object_size': object_size,
                'close_gripper_distance': close_gripper_distance
            }
        )

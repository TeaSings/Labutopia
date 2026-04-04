from .base_task import BaseTask

class SingleObjectTask(BaseTask):
    """
    Base class for handling single target object tasks.
    Suitable for tasks like pick, open, close, etc. that require only one main target object.
    """
    
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        
    def on_task_complete(self, success):
        """
        Handle task completion logic.
        Update object and material indices.
        """
        self.update_object_and_material_indices(success)
        
    def reset(self):
        """
        Reset task state.
        Initialize robot position, apply materials, and place objects.
        """
        super().reset()
        self.robot.initialize()
        
        if self.material_config:
            self.apply_material_to_object(self.material_config.path)
        
        # 若当前物体 prim 无效，尝试下一个直到找到有效物体（setup 已过滤，此处为兜底）
        for _ in range(max(1, len(self.obj_configs))):
            self.current_obj_path = self.place_objects_with_visibility_management(
                self.current_obj_idx, far_distance=10.0
            )
            if self.current_obj_path is not None:
                break
            self.current_obj_idx = (self.current_obj_idx + 1) % max(1, len(self.obj_configs))
        if getattr(self, "debug_collection_schedule", False) and self.current_obj_path is not None:
            sampled = getattr(self, "current_obj_position", None)
            sampled_str = sampled.tolist() if sampled is not None else None
            print(
                "[Task][Schedule] "
                f"object_idx={self.current_obj_idx} "
                f"object={self.current_obj_path.split('/')[-1]} "
                f"obj_counter={self.current_obj_episodes}/{self.episodes_per_obj} "
                f"pose_id={self.current_pose_id} "
                f"pose_counter={self.current_position_counter}/{self.position_switch_interval} "
                f"pose_metric={self.pose_switch_metric} "
                f"resampled={getattr(self, '_last_pose_resampled', False)} "
                f"sampled_pos={sampled_str}"
            )
        
    def step(self):
        """
        Execute one simulation step and return current state.
        
        Returns:
            dict: Dictionary containing current state information
        """
        self.frame_idx += 1
        
        if not self.check_frame_limits():
            return None
            
        return self.get_basic_state_info(object_path=self.current_obj_path) 

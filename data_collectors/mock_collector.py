import os
import numpy as np
from typing import List, Optional
from .data_collector import DataCollector

class MockCollector(DataCollector):
    """Mock data collector for testing purposes - implements same interface as DataCollector but does nothing"""
    
    def __init__(
        self,
        camera_configs: List[dict],
        save_dir="output",
        max_episodes=10,
        max_workers=4,
        compression=None,
        save_frames: int = -1,
        cache_stride: int = 1,
    ):
        """Initialize the mock data collector（签名与 DataCollector 对齐，参数在 mock 中可忽略）。"""
        super().__init__(
            camera_configs,
            save_dir,
            max_episodes,
            max_workers,
            compression,
            save_frames,
            cache_stride,
        )
        
        # Override to not create directories
        self.session_dir = os.path.join(save_dir, "dataset")
        # Don't create the directory in mock mode
        
    def cache_step(self, camera_images: dict = None, joint_angles: np.ndarray = None, language_instruction=None):
        """Mock cache step - does nothing
        
        Args:
            camera_images: Dict of camera name to RGB image {name: np.ndarray}
            joint_angles: Robot joint angles
        """
        # Do nothing - this is a mock collector
        # Override parent method to not actually cache anything
        pass
        
    def write_cached_data(self, final_joint_positions = None):
        """Mock write cached data - does nothing
        
        Args:
            final_joint_positions: Final joint positions
        """
        # Do nothing - this is a mock collector
        # Override parent method to not actually write anything
        self.episode_count += 1
        pass

    def clear_cache(self):
        """与 DataCollector 一致：仅清空缓存，不增加 episode_count（episode_count 仅在 write 后增加）。"""
        for camera_name in self.temp_cameras:
            self.temp_cameras[camera_name] = []
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        self.task_instructions = None
        self._cache_step_index = 0
        
    def close(self):
        """Mock close - does nothing"""
        # Do nothing - this is a mock collector
        # Override parent method to not actually close anything
        pass 
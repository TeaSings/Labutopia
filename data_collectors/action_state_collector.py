import os
import numpy as np
import h5py
from concurrent.futures import ProcessPoolExecutor, Future
from typing import List, Optional
import json

def _write_episode_data_with_properties(episode_path: str, episode_name: str, 
                                        camera_data: dict, agent_pose_data: np.ndarray, 
                                        actions_data: np.ndarray, 
                                        task_properties: dict,
                                        language_instruction: Optional[str] = None, 
                                        compression=None):
    """write episode data with properties
    
    Args:
        episode_path: path of the episode HDF5 file
        episode_name: name of the episode
        camera_data: dictionary of camera name to image data {name: [T, H, W, 3]}
        agent_pose_data: robot joint angles [T, num_joints]
        actions_data: robot actions [T, num_joints]
        task_properties: task unique properties dictionary
        language_instruction: language instruction of the task
        compression: compression method for image data, None means no compression
    """
    
    with h5py.File(episode_path, 'w') as h5_file:
        print(f"Writing episode {episode_name} to {episode_path}")
        
        # use compression and dynamic chunk storage for camera data
        for camera_name, image_data in camera_data.items():
            chunk_size = (min(64, image_data.shape[0]),) + image_data.shape[1:]
            kwargs = {
                'data': image_data,
                'dtype': 'uint8',
                'chunks': chunk_size
            }
            if compression == "gzip":
                kwargs.update({
                    'compression': "gzip",
                    'compression_opts': 5
                })
            h5_file.create_dataset(camera_name, **kwargs)
        
        # store pose and action data without compression (small data size, frequent access)
        h5_file.create_dataset(
            "agent_pose", 
            data=agent_pose_data, 
            dtype='float32', 
            chunks=True
        )
        h5_file.create_dataset(
            "actions", 
            data=actions_data, 
            dtype='float32', 
            chunks=True
        )
        
        # store language instruction (if provided)
        if language_instruction is not None:
            h5_file.create_dataset(
                "language_instruction",
                data=language_instruction,
                dtype=h5py.special_dtype(vlen=str)
            )
        
        # store task properties (as JSON string)
        if task_properties:
            task_properties_json = json.dumps(task_properties, ensure_ascii=False)
            h5_file.create_dataset(
                "task_properties",
                data=task_properties_json,
                dtype=h5py.special_dtype(vlen=str)
            )
        
        print(f"Finished writing episode {episode_name}")


class ActionStateDataCollector:
    """用于存储robot_state和action分开的数据收集器 data collector for storing robot_state and action separately
    
    This collector is suitable for scenarios where action and state are not synchronized, such as navigation tasks.
    Supports storing unique task properties for each episode.
    """
    
    def __init__(self, camera_configs: List[dict], save_dir="output", 
                 max_episodes=10, max_workers=4, compression=None):
        """initialize data collector with multiple processes
        
        Args:
            camera_configs: dictionary of camera configurations, each containing 'name' key
            save_dir (str): root directory for saving data
            max_episodes (int): maximum number of episodes to record
            max_workers (int): maximum number of parallel processes
            compression: compression method for image data, None means no compression
        """
        self.save_dir = save_dir
        self.max_episodes = max_episodes
        self.compression = compression
        self.session_dir = os.path.join(save_dir, "dataset")
        self.mate_dir = os.path.join(self.session_dir, "meta")
        self.episode_file_path = os.path.join(self.mate_dir, "episode.jsonl")
        self.episode_count = 0
        self.camera_configs = camera_configs
        self.task_instructions = None
        
        # create directories
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.mate_dir, exist_ok=True)
        
        # initialize temporary storage dictionary with combined camera names and types
        self.temp_cameras = {}
        for config in camera_configs:
            if '+' in config['image_type']:
                types = config['image_type'].split('+')
                for t in types:
                    self.temp_cameras[f"{config['name']}_{t}"] = []
            else:
                self.temp_cameras[f"{config['name']}_{config['image_type']}"] = []
        
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        self.temp_task_properties = {}
        
        # initialize process pool and tracking variables
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.pending_futures: List[Future] = []
    
    def set_task_properties(self, properties: dict):
        """set the unique task properties for the current episode
        
        Args:
            properties: task properties dictionary, content depends on the specific task
                       for example, navigation task: {"start_position": [x, y, z], "end_position": [x, y, z]}
        """
        self.temp_task_properties = properties
    
    def cache_step(self, camera_images: dict, joint_angles: np.ndarray, 
                   action: np.ndarray, language_instruction: Optional[str] = None):
        """cache the data of each step in the temporary list
        
        Args:
            camera_images: dictionary of camera name to RGB image {name: np.ndarray}
            joint_angles: robot joint angles (state)
            action: robot action
            language_instruction: language instruction of the task
        """
        if self.task_instructions is None and language_instruction is not None:
            self.task_instructions = language_instruction
        
        for camera_name, image in camera_images.items():
            self.temp_cameras[camera_name].append(image)
        
        self.temp_agent_pose.append(joint_angles)
        self.temp_actions.append(action)
        
        if language_instruction is not None:
            self.temp_language_instruction = language_instruction
    
    def write_cached_data(self):
        """write cached data asynchronously using process pool"""
        if self.episode_count >= self.max_episodes:
            self.close()
            return
        
        # convert lists to numpy arrays
        camera_data = {
            name: np.array(images) 
            for name, images in self.temp_cameras.items()
        }
        agent_pose_data = np.array(self.temp_agent_pose)
        actions_data = np.array(self.temp_actions)
        
        # create single episode file path
        episode_name = f"episode_{self.episode_count:04d}"
        episode_path = os.path.join(self.session_dir, f"{episode_name}.h5")
        
        # submit writing task to process pool
        future = self.process_pool.submit(
            _write_episode_data_with_properties,
            episode_path,
            episode_name,
            camera_data,
            agent_pose_data,
            actions_data,
            self.temp_task_properties,
            self.temp_language_instruction,
            self.compression
        )
        self.pending_futures.append(future)
        
        # write episode metadata
        info = {
            "episode_index": self.episode_count,
            "tasks": [self.task_instructions] if self.task_instructions else [],
            "length": len(self.temp_agent_pose),
            "task_properties": self.temp_task_properties
        }
        
        with open(self.episode_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")
        
        # clear cache
        for camera_name in self.temp_cameras:
            self.temp_cameras[camera_name] = []
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        self.temp_task_properties = {}
        
        # increase episode count
        self.episode_count += 1
    
    def clear_cache(self):
        """clear cached data without writing to disk"""
        for camera_name in self.temp_cameras:
            self.temp_cameras[camera_name] = []
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        self.temp_task_properties = {}
        self.task_instructions = None
    
    def close(self, merge=False):
        """close the data collector and optionally merge all episode files
        
        Args:
            merge: whether to merge all episode files into a single file
        """
        # wait for all pending writing operations to complete
        for future in self.pending_futures:
            future.result()
        
        # close process pool
        self.process_pool.shutdown(wait=True)
        
        if merge:
            from glob import glob
            merged_path = os.path.join(self.session_dir, "merged_episodes.hdf5")
            episode_files = sorted(glob(os.path.join(self.session_dir, "episode_*.h5")))
            
            if not episode_files:
                print("No episodes to merge")
                return
            
            with h5py.File(merged_path, 'w') as merged_file:
                # copy each episode file to merged file
                for episode_path in episode_files:
                    episode_name = os.path.splitext(os.path.basename(episode_path))[0]
                    with h5py.File(episode_path, 'r') as episode_file:
                        # create episode group in merged file
                        episode_group = merged_file.create_group(episode_name)
                        
                        # copy all datasets with original compression settings
                        for key in episode_file.keys():
                            episode_file.copy(key, episode_group)
                    
                    # delete single episode file after merging
                    os.remove(episode_path)
            
            os.rename(merged_path, os.path.join(self.session_dir, "episode_data.hdf5"))
            print(f"Successfully merged {len(episode_files)} episodes into {merged_path}")


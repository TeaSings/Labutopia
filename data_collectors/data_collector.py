import os
import numpy as np
import cv2
from datetime import datetime
import json
import h5py
from concurrent.futures import ProcessPoolExecutor, Future
from typing import List, Optional
from glob import glob

def _write_episode_data(episode_path: str, episode_name: str, 
                       camera_data: dict, agent_pose_data: np.ndarray, 
                       actions_data: np.ndarray, task_properties: dict = None,
                       language_instruction: Optional[str] = None, compression=None):
    """Helper function to write episode data in a separate process
    
    Args:
        episode_path: Path to the individual episode HDF5 file
        episode_name: Name of the episode
        camera_data: Dict of camera name to image data {name: [T, H, W, 3]}
        agent_pose_data: Robot joint angles [T, num_joints]
        actions_data: Robot actions [T, num_joints]
        task_properties: Task unique properties dictionary
        language_instruction: Language instruction for the task
        compression: Compression method for image data, None for no compression
    """
    
    with h5py.File(episode_path, 'w') as h5_file:
        print(f"Writing episode {episode_name} to {episode_path}")
        
        # Store camera data with Blosc compression and dynamic chunking
        for camera_name, image_data in camera_data.items():
            if image_data.size == 0:
                continue
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
        
        # Store pose and action data without compression (small size, frequent access)
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
        
        # Store language instruction if provided
        if language_instruction is not None:
            h5_file.create_dataset(
                "language_instruction",
                data=language_instruction,
                dtype=h5py.special_dtype(vlen=str)
            )
        
        # Store task properties (as JSON string)
        if task_properties:
            task_properties_json = json.dumps(task_properties, ensure_ascii=False)
            h5_file.create_dataset(
                "task_properties",
                data=task_properties_json,
                dtype=h5py.special_dtype(vlen=str)
            )
        
        print(f"Finished writing episode {episode_name}")

class DataCollector:
    def __init__(self, camera_configs: List[dict], save_dir="output", 
                 max_episodes=10, max_workers=4, compression=None, save_frames: int = -1,
                 cache_stride: int = 1):
        """Initialize the data collector with multiprocessing support
        
        Args:
            camera_configs: List of camera configuration dicts, each containing 'name' key
            save_dir (str): Root directory for saving data
            max_episodes (int): Maximum number of episodes to record
            max_workers (int): Maximum number of parallel processes
            compression: Compression method for image data, None for no compression
            save_frames: -1=保存全部帧, 1=仅首帧(VLM用,大幅省空间)
            cache_stride: 仅当 save_frames==-1 时生效：每隔 N 次 cache_step 写入一帧（相机与 pose 对齐），
                固定「采样步频」、episode 越长则 T 越大。与 convert --temporal-stride 二选一或组合见文档。
        """
        self.save_dir = save_dir
        self.max_episodes = max_episodes
        self.compression = compression
        self.save_frames = save_frames
        self.cache_stride = max(1, int(cache_stride)) if cache_stride else 1
        self._cache_step_index = 0
        self.session_dir = os.path.join(save_dir, "dataset")
        self.mate_dir = os.path.join(self.session_dir, "meta")
        self.episode_file_path = os.path.join(self.mate_dir, "episode.jsonl")
        self.episode_count = 0
        self.camera_configs = camera_configs
        self.task_instructions = None
        # Create directories
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.mate_dir, exist_ok=True)
        # Initialize temporary storage dictionaries with combined camera name and type
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
        
        # Initialize process pool and tracking variables
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.pending_futures: List[Future] = []
    
    def set_task_properties(self, properties: dict):
        """Set the unique task properties for the current episode
        
        Args:
            properties: Task properties dictionary, content depends on the specific task
                       For example, navigation task: {"start_position": [x, y, z], "end_position": [x, y, z]}
        """
        self.temp_task_properties = properties

    def update_task_properties(self, updates: dict):
        """Merge additional task properties (e.g. is_success) before write.
        
        Args:
            updates: Dict to merge into temp_task_properties
        """
        self.temp_task_properties.update(updates)
        
    def cache_step(self, camera_images: dict, joint_angles: np.ndarray, language_instruction: Optional[str] = None):
        """Cache each step's data in temporary lists
        
        Args:
            camera_images: Dict of camera name to RGB image {name: np.ndarray}
            joint_angles: Robot joint angles
            language_instruction: Language instruction for the task
        """
        self._cache_step_index += 1
        if self.save_frames < 0 and self.cache_stride > 1:
            if (self._cache_step_index - 1) % self.cache_stride != 0:
                return
        if self.task_instructions is None and language_instruction is not None:
            self.task_instructions = language_instruction
        for camera_name, image in camera_images.items():
            if camera_name not in self.temp_cameras:
                print(f"[DataCollector] 警告：camera_data key '{camera_name}' 不在 temp_cameras 中，跳过")
                continue
            lst = self.temp_cameras[camera_name]
            if self.save_frames < 0:
                lst.append(image)
            elif self.save_frames == 1:
                if len(lst) < 1:
                    lst.append(image)
            elif self.save_frames == 2:
                if len(lst) == 0:
                    lst.append(image)
                elif len(lst) == 1:
                    lst.append(image)
                else:
                    lst[-1] = image  # 持续覆盖末帧
            else:
                # save_frames >= 3: 低帧模式，先缓存全部，写入时均匀采样
                lst.append(image)
        self.temp_agent_pose.append(joint_angles)
        if language_instruction is not None:
            self.temp_language_instruction = language_instruction
        
    def write_cached_data(self, final_joint_positions):
        """Write cached data asynchronously using process pool"""
        if self.episode_count >= self.max_episodes:
            self.close()
            return

        if len(self.temp_agent_pose) == 0:
            print("[DataCollector] 跳过写入：无缓存数据（可能为提前返回 episode）")
            return

        # Add the final action
        self.temp_actions = self.temp_agent_pose[1:] + [final_joint_positions]
        
        # Convert lists to numpy arrays，仅保留有数据的相机
        camera_data = {}
        for name, images in self.temp_cameras.items():
            if len(images) > 0:
                camera_data[name] = np.array(images)
            else:
                print(f"[DataCollector] 警告：相机 {name} 无数据，跳过")
        agent_pose_data = np.array(self.temp_agent_pose)
        actions_data = np.array(self.temp_actions)
        # save_frames 限制时，pose/actions 与图像帧数对齐
        if self.save_frames > 0:
            T = len(agent_pose_data)
            K = min(self.save_frames, T)
            if K == 1:
                agent_pose_data = agent_pose_data[:1]
                actions_data = np.array([final_joint_positions])
            elif K == 2:
                agent_pose_data = np.array([agent_pose_data[0], agent_pose_data[-1]])
                actions_data = np.array([self.temp_actions[0], final_joint_positions])
            else:
                # K >= 3: 均匀采样 [首, 中间..., 末]
                indices = [0] + [int((T - 1) * i / (K - 1)) for i in range(1, K - 1)] + [T - 1]
                indices = sorted(set(indices))[:K]
                agent_pose_data = agent_pose_data[indices]
                actions_data = np.array([self.temp_actions[i] for i in indices])
            # camera_data 同样均匀采样
            if self.save_frames >= 3 and T > 0:
                K = min(self.save_frames, T)
                indices = [0] + [int((T - 1) * i / (K - 1)) for i in range(1, K - 1)] + [T - 1]
                indices = sorted(set(indices))[:K]
                camera_data = {
                    name: (arr[indices] if len(arr.shape) == 4 else arr)
                    for name, arr in camera_data.items()
                }
        
        # Create individual episode file path
        episode_name = f"episode_{self.episode_count:04d}"
        episode_path = os.path.join(self.session_dir, f"{episode_name}.h5")
        
        # 同步写入，避免 Windows 下 ProcessPoolExecutor 序列化问题导致 dataset 为空
        try:
            _write_episode_data(
                episode_path,
                episode_name,
                camera_data,
                agent_pose_data,
                actions_data,
                self.temp_task_properties,
                self.temp_language_instruction,
                self.compression
            )
        except Exception as e:
            print(f"[DataCollector] 写入失败 {episode_path}: {e}")
        self.pending_futures.append(None)  # 保持兼容 close() 的 future 遍历

        info = {
            "episode_index": self.episode_count,
            "tasks": [self.task_instructions] if self.task_instructions else [],
            "length": len(self.temp_agent_pose),
            "task_properties": self.temp_task_properties
        }
        
        with open(self.episode_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")
        
        # Clear cache
        for camera_name in self.temp_cameras:
            self.temp_cameras[camera_name] = []
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        self.temp_task_properties = {}
        self._cache_step_index = 0
        
        # Increment episode count
        self.episode_count += 1

    def clear_cache(self):
        """Clear the cached data without writing to disk"""
        for camera_name in self.temp_cameras:
            self.temp_cameras[camera_name] = []
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        self.temp_task_properties = {}
        self.task_instructions = None
        self._cache_step_index = 0
        
    def close(self, merge=False):
        """Close the data collector and merge all episode files"""
        # Wait for all pending writing operations to complete (sync 模式下 future 为 None)
        for future in self.pending_futures:
            if future is not None:
                future.result()
        
        # Shutdown process pool
        self.process_pool.shutdown(wait=True)
        
        if merge:
            merged_path = os.path.join(self.session_dir, "merged_episodes.hdf5")
            episode_files = sorted(glob(os.path.join(self.session_dir, "episode_*.h5")))
            
            if not episode_files:
                print("No episodes to merge")
                return
                
            with h5py.File(merged_path, 'w') as merged_file:
                # Copy each episode file into the merged file
                for episode_path in episode_files:
                    episode_name = os.path.splitext(os.path.basename(episode_path))[0]
                    with h5py.File(episode_path, 'r') as episode_file:
                        # Create episode group in merged file
                        episode_group = merged_file.create_group(episode_name)
                        
                        # Copy all datasets with their original compression settings
                        for key in episode_file.keys():
                            episode_file.copy(key, episode_group)
                    
                    # Remove individual episode file after merging
                    os.remove(episode_path)
            os.rename(merged_path, os.path.join(self.session_dir, "episode_data.hdf5"))
            print(f"Successfully merged {len(episode_files)} episodes into {merged_path}")
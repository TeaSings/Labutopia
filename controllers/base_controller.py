from abc import ABC, abstractmethod 
from typing import Dict, Any, Tuple, Optional
import torch
from omegaconf import OmegaConf
from controllers.inference_engines.inference_engine_factory import InferenceEngineFactory
from controllers.robot_controllers.grapper_manager import Gripper
from controllers.robot_controllers.trajectory_controller import FrankaTrajectoryController
from factories.collector_factory import create_collector
from utils.object_utils import ObjectUtils
from robots.franka.rmpflow_controller import RMPFlowController as FrankaRMPFlowController

class BaseController(ABC):
    """Base class for all controllers in the chemistry lab simulator.
    
    Provides common functionality for robot control, state management,
    and episode tracking.
    """
    
    def __init__(self, cfg, robot, use_default_config=True):
        """Initialize the base controller.
        
        Args:
            cfg: Configuration object containing controller settings
            robot: Robot instance to control
            object_utils: Utility class for object manipulation
        """
        self.cfg = cfg
        self.robot = robot
        self.object_utils = ObjectUtils.get_instance()
        self.reset_needed = False
        self._last_success = False
        self._episode_num = 0
        self.success_count = 0
        self._skipped_count = 0  # 未写入的 episode（object_position=None 或异常）
        self._language_instruction = ""
        self.gripper_control = Gripper()
        self.REQUIRED_SUCCESS_STEPS = 60
        self.check_success_counter = 0
        self.rmp_controller = None
        self._last_failure_reason = ""
        
        self.rmp_controller = FrankaRMPFlowController(
            name="target_follower_controller",
            robot_articulation=robot,
            use_default_config=use_default_config
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        if hasattr(cfg, "mode"):
            self.mode = cfg.mode  # "collect" | "infer" | "vlm_live"
            if self.mode == "collect":
                self._init_collect_mode(cfg, robot)
            elif self.mode == "infer":
                self._init_infer_mode(cfg, robot)
            elif self.mode == "vlm_live":
                self._init_vlm_live_mode(cfg, robot)
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Expected 'collect', 'infer', or 'vlm_live'.")

    @property
    def language_instruction(self) -> Optional[str]:
        """Get the current language instruction for the task.
        
        Returns:
            Optional[str]: The language instruction or None if not set
        """
        return self._language_instruction
    
    @language_instruction.setter
    def language_instruction(self, instruction: Optional[str]):
        """Set the language instruction for the task.
        
        Args:
            instruction: The language instruction to set, or None to clear
        """
        self._language_instruction = instruction
    
    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        This method can be overridden by subclasses to provide dynamic instructions.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        return self._language_instruction
    
    @abstractmethod
    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Execute one step of control.
        
        Args:
            state: Current state dictionary containing sensor data and robot state
            
        Returns:
            Tuple containing:
            - action: Control action to execute
            - done: Whether the episode is complete
            - is_success: Whether the task was completed successfully
        """
        pass
    
    def _init_collect_mode(self, cfg, robot=None):
        """Initialize the controller for collect mode."""
        collector_cfg = cfg.collector
        self.data_collector = create_collector(
            collector_cfg.type,
            camera_configs=cfg.cameras,
            save_dir=cfg.multi_run.run_dir,
            max_episodes=cfg.max_episodes,
            compression=getattr(collector_cfg, 'compression', None),
            save_frames=getattr(collector_cfg, 'save_frames', -1),
            cache_stride=int(getattr(collector_cfg, 'cache_stride', 1) or 1),
        )
    
    def _init_infer_mode(self, cfg, robot=None): 
        """Initialize the controller for infer mode."""
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot,
            use_interpolation=False
        )
        
        self.inference_engine = InferenceEngineFactory.create_inference_engine(
            cfg, self.trajectory_controller
        )

    def _init_vlm_live_mode(self, cfg, robot=None):
        """实时 VLM 闭环（初始推断→执行→失败纠错）。仅 PickTaskController 实现。"""
        raise NotImplementedError("mode vlm_live 仅支持 controller_type: pick（见 PickTaskController）。")
    
    def episode_num(self) -> int:
        """Get the current episode number.
        
        Returns:
            int: Current episode number
        """
        if self.mode == "collect":
            return self.data_collector.episode_count
        return self._episode_num
    
    def print_failure_reason(self) -> None:
        """Print the last failure reason if it exists."""
        if self._last_failure_reason:
            print(f"Failure Reason: {self._last_failure_reason}")
    
    def reset(self) -> None:
        """Reset the controller state between episodes."""
        if self._last_success:
            self.success_count += 1
        skipped = getattr(self, '_early_return', False)
        if skipped:
            self._skipped_count = getattr(self, '_skipped_count', 0) + 1
        self._episode_num += 1
        if self.mode == "collect":
            written = self.data_collector.episode_count
        else:
            written = self._episode_num
        msg = f"Episode Stats: Success = {self.success_count}/{written} written"
        if self._skipped_count > 0:
            msg += f", {self._skipped_count} skipped (object_position=None / tipped / exception)"
        msg += f" ({100*self.success_count/max(1,written):.1f}% success)"
        print(msg)
        self.check_success_counter = 0
        self.reset_needed = False
        self._last_success = False
        self._last_failure_reason = ""
        if self.mode == "collect" or self.mode == "vlm_live":
            self.data_collector.clear_cache()

        
    def close(self) -> None:
        """Clean up resources used by the controller."""
        if self.mode in ("collect", "vlm_live") and hasattr(self, 'data_collector'):
            self.data_collector.close()
        
    def need_reset(self) -> bool:
        """Check if the controller needs to be reset.
        
        Returns:
            bool: True if reset is needed, False otherwise
        """
        return self.reset_needed

    def is_atomic_action_complete(self) -> bool:
        """Check if the current atomic action has fully completed.
        Override in subclasses that use atomic action controllers.
        When False, task timeout should NOT interrupt (wait for action to finish).
        
        Returns:
            bool: True if atomic action is done or N/A, False if still running
        """
        return True

    def is_success(self):
        return self._last_success
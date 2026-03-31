import os
import sys

# 相机/Replicator：必须在 SimulationApp 之前设置。仅 headless 时设会导致「带窗口运行仍全黑」。
# 若需关闭可在外部先设 ENABLE_CAMERAS=0。
if "ENABLE_CAMERAS" not in os.environ:
    os.environ["ENABLE_CAMERAS"] = "1"

import shutil
import argparse

# Parse command line arguments (必须在 isaacsim 导入前完成)
def parse_args():
    parser = argparse.ArgumentParser(description='LabSim Simulation Environment')
    parser.add_argument('--backend', type=str, default='numpy', 
                       choices=['numpy', 'gpu'], 
                       help='Backend choice: numpy (CPU) or gpu')
    parser.add_argument('--headless', action='store_true', 
                       help='Run in headless mode (default is with GUI)')
    parser.add_argument('--no-video', action='store_true', 
                       help='Disable video display and saving')
    parser.add_argument('--fast-sim', action='store_true',
                       help='Accelerate simulation: larger physics step, disable viewport/Motion BVH (headless only)')
    parser.add_argument('--config-name', type=str, default='level3_Heat_Liquid',
                       help='Configuration file name (without .yaml extension)')
    parser.add_argument('--config-dir', type=str, default='config',
                       help='Configuration directory path (default: config)')
    parser.add_argument(
        '--debug-camera-stats',
        action='store_true',
        help='周期性打印 camera_data（VLM/采集用）各视角 shape/mean/min/max，便于与 camera_display 对比',
    )
    parser.add_argument(
        '--render-warmup-steps',
        type=int,
        default=-1,
        help='主循环前仅执行 world.step(render=True) 的步数，用于 RTX/相机/着色器冷启动预热，减轻「首次运行黑屏、再跑一次就好」的现象。'
        ' -1 表示自动：headless 下 96，否则 24；设为 0 关闭',
    )
    return parser.parse_args()

args = parse_args()

from isaacsim import SimulationApp

_black_frame_warned = False
_black_frame_streak = 0
# 连续多少帧仍「全黑」才触发自动恢复（略降低以便尽快尝试修复）
_BLACK_FRAME_WARN_AFTER = 60
# 每次触发黑屏时额外跑的渲染预热步数（优先 render-only，见 _warmup_rendering）
_BLACK_FRAME_RECOVERY_STEPS = 256
_black_frame_recovery_count = 0
_BLACK_FRAME_MAX_RECOVERIES = 8

simulation_config = {
    "headless": args.headless,
    "extra_args": ["--/rtx/raytracing/fractionalCutoutOpacity=true"],
}
if args.headless:
    simulation_config["disable_viewport_updates"] = True
if args.fast_sim and args.headless:
    simulation_config["extra_args"].extend([
        "--/renderer/raytracingMotion/enabled=false",
        "--/renderer/raytracingMotion/enableHydraEngineMasking=false",
        "--/renderer/raytracingMotion/enabledForHydraEngines=''",
    ])
simulation_app = SimulationApp(simulation_config)

import hydra
from omegaconf import OmegaConf
import cv2
import numpy as np

import omni
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
import omni.usd
from isaacsim.core.utils import extensions

extensions.enable_extension("omni.physx.bundle")
extensions.enable_extension("omni.usdphysics.ui")

from factories.robot_factory import create_robot
from utils.object_utils import ObjectUtils
from factories.task_factory import create_task
from factories.controller_factory import create_controller


def _warmup_rendering(
    world,
    n_steps: int,
    simulation_app_ref,
    *,
    quiet: bool = False,
    prefer_render_only: bool = False,
) -> None:
    """冷启动或黑屏恢复时推进渲染管线。

    - `prefer_render_only=False`（启动预热）：每步 `world.step(render=True)`，与主循环一致。
    - `prefer_render_only=True`（中途黑屏修复）：优先 `world.render()` / `scene.render()`，尽量不额外推进物理，
      避免「修黑屏」时仿真状态与 task 帧计数脱节；若无 API 再回退到 `world.step(render=True)`。
    同时尝试 `simulation_app.update()`（Kit/RTX 带窗口时常需）。"""
    if n_steps <= 0:
        return
    for _ in range(n_steps):
        if not simulation_app_ref.is_running():
            return
        stepped = False
        if prefer_render_only:
            for fn in (
                getattr(world, "render", None),
                getattr(getattr(world, "scene", None), "render", None),
            ):
                if callable(fn):
                    try:
                        fn()
                        stepped = True
                        break
                    except Exception:
                        pass
        if not stepped:
            world.step(render=True)
        upd = getattr(simulation_app_ref, "update", None)
        if callable(upd):
            try:
                upd()
            except Exception:
                pass
    if not quiet:
        mode = "优先 render() + step 回退" if prefer_render_only else "world.step(render=True)"
        print(
            f"[Main] 渲染预热完成：{n_steps} 步（{mode}），"
            "缓解冷启动 RTX/相机全黑；若仍黑可增大 --render-warmup-steps。"
        )


def _print_camera_data_stats(camera_data: dict, prefix: str = "[camera_data]") -> None:
    """诊断：camera_data 为任务/VLM 使用的图像（常为 CHW uint8）。"""
    if not camera_data:
        print(f"{prefix} (空)")
        return
    parts = []
    for k in sorted(camera_data.keys()):
        x = np.asarray(camera_data[k])
        parts.append(
            f"{k}: shape={x.shape} dtype={x.dtype} mean={float(np.mean(x)):.2f} "
            f"min={int(np.min(x))} max={int(np.max(x))}"
        )
    print(f"{prefix} " + " | ".join(parts))


def main():
    global _black_frame_warned, _black_frame_streak, _black_frame_recovery_count
    hydra.initialize(config_path=args.config_dir, job_name=args.config_name, version_base="1.1")
    cfg = hydra.compose(config_name=args.config_name)
    os.makedirs(cfg.multi_run.run_dir, exist_ok=True)
    OmegaConf.save(cfg, cfg.multi_run.run_dir + "/config.yaml")

    # Set backend based on command line arguments
    if args.backend == 'gpu':
        world = World(stage_units_in_meters=1, device="cpu")
        physx_interface = omni.physx.get_physx_interface()
        physx_interface.overwrite_gpu_setting(1)
    else:
        world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")
    # 加速仿真：增大物理步长（1/30s 替代默认 1/60s），步数减半，速度约 2x
    if args.fast_sim and args.headless:
        try:
            world.set_physics_step_size(1.0 / 30.0)
            print("[Main] 已启用快速仿真：physics_step_size=1/30s")
        except Exception as e:
            print(f"[Main] 设置 physics_step_size 失败（可忽略）: {e}")
    
    # Override configuration based on command line arguments
    if args.no_video:
        save_video = False
        show_video = False
    else:
        save_video = True
        # headless 模式下不弹窗显示视频，仅保存到文件
        show_video = not args.headless

    robot = create_robot(
        cfg.robot.type,
        position=np.array(cfg.robot.position)
    )
    
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=os.path.abspath(cfg.usd_path), prim_path="/World")
    
    ObjectUtils.get_instance(stage)
    
    task = create_task(
        cfg.task_type,
        cfg=cfg,
        world=world,
        stage=stage,
        robot=robot,
    )
    
    task_controller = create_controller(
        cfg.controller_type,
        cfg=cfg,
        robot=robot,
    )
    
    video_writer = None
    task.reset()
    _debug_cam_step = 0

    rw = args.render_warmup_steps
    if rw < 0:
        # 带窗口时也容易首帧黑，略增默认预热（仍可用 --render-warmup-steps 覆盖）
        rw = 96 if args.headless else 72
    _warmup_rendering(world, rw, simulation_app)
    _black_frame_streak = 0

    while simulation_app.is_running():
        world.step(render=True)
        # 注意：不要在 world.is_stopped() 时设置 task_controller.reset_needed。
        # 暂停/单步或某些帧 is_stopped 为 True 后，下一帧 is_playing 时 should_reset 会因
        # task_controller.need_reset() 立刻整局重置，打断 vlm_live 纠错后的第二次 pick 与任务②判断。

        if world.is_playing():
            # 仅在 controller 完成 或 (task 超时且原子动作已执行完) 时 reset
            # 避免 max_steps 超时中断未完成的原子动作，确保失败样本是「完整执行后失败」
            atomic_done = getattr(task_controller, 'is_atomic_action_complete', lambda: True)()
            # vlm_live：整局结束只认控制器的 reset_needed（VLM 任务②/③、缓冲/API 错误等）。
            # 若仍用 (task.need_reset() and atomic_done)，会在「纠错里已对 pick reset、但本帧尚未再次
            # task.step」时与 max_steps 超时叠加：atomic_done 仍反映「上一段 pick 已结束」→ 误判为
            # 可整局重置，出现「纠错下一行就 Episode Stats」且下一轮 纠错次数=0。
            if getattr(task_controller, "mode", None) == "vlm_live":
                should_reset = task_controller.need_reset()
            else:
                should_reset = task_controller.need_reset() or (task.need_reset() and atomic_done)
            if should_reset:
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    output_dir = os.path.join(cfg.multi_run.run_dir, "video")
                    ep_num = task_controller._episode_num
                    output_path = os.path.join(output_dir, f"episode_{ep_num}.mp4")
                    if os.path.exists(output_path):
                        if getattr(task_controller, '_early_return', False):
                            os.remove(output_path)  # 提前返回：不保存 video
                        else:
                            last_success = task_controller._last_success
                            subdir = "success" if last_success else "failure"
                            dest_dir = os.path.join(output_dir, subdir)
                            os.makedirs(dest_dir, exist_ok=True)
                            dest_path = os.path.join(dest_dir, f"episode_{ep_num}.mp4")
                            shutil.move(output_path, dest_path)
                           
                task_controller.reset()
                # 停止条件：1) max_episodes 达到  2) successes_per_obj 模式下达到目标成功数
                target_successes = None
                if hasattr(cfg, 'task') and hasattr(cfg.task, 'successes_per_obj') and cfg.task.successes_per_obj is not None:
                    n_obj = len(getattr(task, 'obj_configs', []))
                    if n_obj > 0:
                        target_successes = int(cfg.task.successes_per_obj) * n_obj
                should_stop = task_controller.episode_num() >= cfg.max_episodes
                if target_successes is not None and task_controller.success_count >= target_successes:
                    should_stop = True
                if should_stop:
                    task_controller.close()
                    simulation_app.close()
                    cv2.destroyAllWindows()
                    break
                task.reset()
                _black_frame_streak = 0
                _black_frame_recovery_count = 0
                
                continue
                
            state = task.step()
            if state is None:
                continue

            if args.debug_camera_stats:
                _debug_cam_step += 1
                if _debug_cam_step == 1 or _debug_cam_step % 120 == 0:
                    _print_camera_data_stats(
                        state.get("camera_data") or {},
                        f"[Main] debug camera_data (sim_step={_debug_cam_step})",
                    )

            try:
                action, done, is_success = task_controller.step(state)
            except Exception as e:
                print(f"[Main] 控制器/仿真器异常，不写入数据，直接重置: {e}")
                setattr(task_controller, '_early_return', True)
                task_controller.reset_needed = True
                continue

            if action is not None:
                try:
                    robot.get_articulation_controller().apply_action(action)
                except Exception as e:
                    print(f"[Main] apply_action 异常，不写入数据，直接重置: {e}")
                    setattr(task_controller, '_early_return', True)
                    task_controller.reset_needed = True
                    continue
            if done:
                task_controller.print_failure_reason()
                task.on_task_complete(is_success)
                continue

            if save_video or show_video:
                camera_images = []
                for _, image_data in state['camera_display'].items():
                    display_img = cv2.cvtColor(image_data.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    camera_images.append(display_img)
                
                if camera_images:
                    combined_img = np.hstack(camera_images)
                    # 黑屏诊断：仅用于预览/录屏的 camera_display；连续多帧仍黑则先尝试与启动时同类的渲染预热再续跑
                    if np.mean(combined_img) < 10:
                        _black_frame_streak += 1
                    else:
                        _black_frame_streak = 0
                    if _black_frame_streak >= _BLACK_FRAME_WARN_AFTER:
                        if _black_frame_recovery_count < _BLACK_FRAME_MAX_RECOVERIES:
                            _black_frame_recovery_count += 1
                            print(
                                f"[Main] 预览连续 {_BLACK_FRAME_WARN_AFTER} 帧接近全黑，"
                                f"正在执行 {_BLACK_FRAME_RECOVERY_STEPS} 步额外渲染预热以尝试恢复 "
                                f"（第 {_black_frame_recovery_count}/{_BLACK_FRAME_MAX_RECOVERIES} 次自动修复）…"
                            )
                            _warmup_rendering(
                                world,
                                _BLACK_FRAME_RECOVERY_STEPS,
                                simulation_app,
                                quiet=True,
                                prefer_render_only=True,
                            )
                            _black_frame_streak = 0
                            print(
                                "[Main] 预热完成，继续仿真；本帧跳过录屏/窗口预览以免写入全黑帧。"
                                "若仍黑：确认已启用 ENABLE_CAMERAS=1（本脚本默认开启）、增大 --render-warmup-steps，"
                                "或加 --debug-camera-stats 查看 camera_data 是否也为黑。"
                            )
                            continue
                        if not _black_frame_warned:
                            _black_frame_warned = True
                            print(
                                "[Main] 警告：预览仍接近全黑，已达自动恢复次数上限（用于录屏/窗口的 camera_display）。\n"
                                "  · 若使用 --headless，本脚本已在启动前自动设置 ENABLE_CAMERAS=1。\n"
                                "  · 可尝试：去掉 --headless 用 GUI；或在终端先执行 "
                                "`set ENABLE_CAMERAS=1`（PowerShell: `$env:ENABLE_CAMERAS='1'`）再运行。\n"
                                "  · VLM/数据采集中实际使用的是 camera_data；若仅此处全黑而任务正常，可忽略本提示。"
                            )
                            if state.get("camera_data"):
                                _print_camera_data_stats(
                                    state["camera_data"],
                                    "[Main] 对照（同一帧）",
                                )
                            elif args.debug_camera_stats:
                                print("[Main] 对照：本帧 state 无 camera_data 键")
                        _black_frame_streak = 0
                    total_width = 0
                    for idx, img in enumerate(camera_images):
                        label = f"Camera {idx+1} ({cfg.cameras[idx].image_type})"
                        cv2.putText(combined_img, label, (total_width + 2, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                        total_width += img.shape[1]
                    if show_video:
                        cv2.imshow('Camera Views', combined_img)
                        cv2.waitKey(1)
                    if save_video:
                        output_dir = os.path.join(cfg.multi_run.run_dir, "video")
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"episode_{task_controller._episode_num}.mp4")
                        if video_writer is None:
                            height, width = combined_img.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            # 30fps 减少仿真帧率与录制帧率不匹配导致的闪烁
                            video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
                        video_writer.write(combined_img)


if __name__ == "__main__":
    main()

"""
将 LabUtopia 采集的 pick 数据转换为 VLM 训练格式。

模式（与采集 save_frames 对齐，见 docs/DATA_AND_TRAINING_MASTER_PLAN.md §2.2）:
- single: 单视角单帧，省空间
- full: 按 HDF5 时间维 × 3 视角展开多图；建议采集 collector.save_frames=-1（密存）。
       --temporal-stride K：每隔 K 个 HDF5 时间索引取 1 帧（固定索引频率 → 变长）。
       --max-timesteps：均匀封顶时刻数；可与 stride 组合（先 stride 再封顶）。
       若未指定 stride 且未指定 max-timesteps，CLI 默认 stride=5，避免全长爆炸。

用法:
    python scripts/convert_to_vlm_format.py <dataset_dir> [--mode single|full] [--output <path>]
    python scripts/convert_to_vlm_format.py <dataset_dir> --mode full --max-timesteps 32
    python scripts/convert_to_vlm_format.py <dataset_dir> --mode full --temporal-stride 5
    python scripts/convert_to_vlm_format.py <dataset_dir> --mode full --temporal-stride 1
"""

import os
import sys
import json
import argparse
from typing import List, Optional
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import h5py
except ImportError:
    h5py = None


def load_episode(episode_path: str) -> dict:
    """加载单个 episode 的 HDF5 文件"""
    if h5py is None:
        raise ImportError("需要安装 h5py: pip install h5py")
    data = {}
    with h5py.File(episode_path, "r") as f:
        for key in f.keys():
            d = f[key]
            if isinstance(d, h5py.Dataset):
                if d.shape == () and d.dtype == h5py.special_dtype(vlen=str):
                    raw = d[()].decode("utf-8") if isinstance(d[()], bytes) else str(d[()])
                    if key == "task_properties":
                        data[key] = json.loads(raw) if raw else {}
                    else:
                        data[key] = raw
                elif key == "task_properties":
                    raw = d[()].decode("utf-8") if isinstance(d[()], bytes) else str(d[()])
                    data[key] = json.loads(raw) if raw else {}
                else:
                    data[key] = np.array(d)
            else:
                data[key] = d
    return data


CAMERA_KEYS = ["camera_1_rgb", "camera_2_rgb", "camera_3_rgb"]


def _frame_to_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    """将单帧转为 HWC uint8。采集 HDF5 中常见为 (3, H, W) CHW；PIL 需要 (H, W, 3)。"""
    x = np.asarray(frame)
    if x.ndim != 3:
        raise ValueError(f"期望 3D 图像数组，得到 shape={getattr(x, 'shape', None)}")
    # CHW：通道维在前
    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    return np.ascontiguousarray(x, dtype=np.uint8)


def _camera_keys_in_data(data: dict) -> List[str]:
    camera_keys = [k for k in CAMERA_KEYS if k in data]
    if not camera_keys:
        for k, v in data.items():
            if k in ("agent_pose", "actions", "language_instruction", "task_properties"):
                continue
            if isinstance(v, np.ndarray) and len(v.shape) == 4:
                camera_keys = [k]
                break
    return camera_keys


def get_frames(
    data: dict,
    mode: str,
    max_timesteps: Optional[int] = None,
    temporal_stride: Optional[int] = None,
):
    """按模式提取帧。single: 单视角首帧; full: 多时刻×多视角，可选 stride / max_timesteps。"""
    camera_keys = _camera_keys_in_data(data)
    if not camera_keys:
        return None

    if mode == "single":
        cam = camera_keys[0]
        arr = data[cam]
        if len(arr.shape) == 4:
            return [arr[0]]
        return [arr]

    if mode != "full":
        raise ValueError(f"不支持的 mode: {mode}（仅支持 single、full）")

    n = 0
    for cam in camera_keys:
        arr = data.get(cam)
        if arr is not None and len(arr.shape) == 4 and arr.shape[0] > 0:
            n = max(n, arr.shape[0])
    if n < 1:
        return None

    if temporal_stride is not None and temporal_stride >= 1:
        step = max(1, int(temporal_stride))
        indices = list(range(0, n, step))
    elif max_timesteps is not None and max_timesteps > 0 and n > max_timesteps:
        if max_timesteps == 1:
            indices = [0]
        else:
            indices = [0] + [
                int((n - 1) * i / (max_timesteps - 1)) for i in range(1, max_timesteps - 1)
            ] + [n - 1]
            indices = sorted(set(indices))[:max_timesteps]
    else:
        indices = list(range(n))
    # 先按 stride 得到可变长度后，可用 max_timesteps 再均匀封顶（避免极长 episode 撑爆上下文）
    if (
        temporal_stride is not None
        and temporal_stride >= 1
        and max_timesteps is not None
        and max_timesteps > 0
        and len(indices) > max_timesteps
    ):
        L = len(indices)
        if max_timesteps == 1:
            indices = [indices[0]]
        else:
            indices = [
                indices[int((L - 1) * i / (max_timesteps - 1))]
                for i in range(max_timesteps)
            ]

    out = []
    for idx in indices:
        for cam in camera_keys:
            arr = data.get(cam)
            if arr is not None and len(arr.shape) == 4 and arr.shape[0] > idx:
                out.append(arr[idx])
    return out if out else None


def _effective_correction_for_training(correction_gt: dict) -> dict:
    """优先使用保守一步 step（方向与 full 一致、幅值单独上限），否则回退完整 delta。

    与 pick_controller._finalize_correction_with_direction_and_steps 写入的 *_step 字段对齐。
    """
    cg = correction_gt
    eff = {}
    for k in ("pre_offset_x", "pre_offset_z", "after_offset_z"):
        if k not in cg:
            continue
        sk = f"{k}_step"
        eff[k] = cg[sk] if sk in cg else cg[k]
    if "euler_deg" in cg:
        eff["euler_deg"] = cg.get("euler_deg_step", cg["euler_deg"])
    if "picking_position_delta" in cg:
        eff["picking_position_delta"] = cg.get("picking_position_delta_step", cg["picking_position_delta"])
    return eff


def _build_corrected_response(params_used: dict, correction_gt: dict) -> str:
    """根据 params_used + correction_gt 构建 corrected response JSON（单步标签优先用保守 step）。"""
    d = _effective_correction_for_training(correction_gt)
    corrected = {
        "pre_offset_x": params_used.get("pre_offset_x", 0.05) + d.get("pre_offset_x", 0),
        "pre_offset_z": params_used.get("pre_offset_z", 0.12) + d.get("pre_offset_z", 0),
        "after_offset_z": params_used.get("after_offset_z", 0.25) + d.get("after_offset_z", 0),
        "euler_deg": [
            params_used.get("euler_deg", [0, 90, 25])[i] + d.get("euler_deg", [0, 0, 0])[i]
            for i in range(3)
        ],
        "is_success": False,
    }
    if "picking_position" in params_used and "picking_position_delta" in d:
        used_pos = params_used["picking_position"]
        delta = d["picking_position_delta"]
        corrected["picking_position"] = [used_pos[i] + delta[i] for i in range(3)]
    return json.dumps(corrected, ensure_ascii=False)


def _multi_image_mode(mode: str) -> bool:
    return mode == "full"


def _full_temporal_rate_hint(temporal_stride: Optional[int]) -> str:
    if temporal_stride is not None and temporal_stride >= 2:
        return f"（按固定间隔采样：HDF5 时间索引每 {temporal_stride} 步取 1 帧，动作越长则时刻越多）"
    return ""


def build_vlm_sample(
    data: dict,
    episode_name: str,
    output_image_dir: str,
    mode: str,
    max_timesteps: Optional[int] = None,
    temporal_stride: Optional[int] = None,
) -> list:
    """
    构建 VLM 训练样本。成功样本返回 1 条；失败样本若有 correction_steps 则返回多条（多次 correction）。
    single: {image: path}
    full: {images: [...]}，张数为 所选时刻数 × 相机数
    """
    props = data.get("task_properties", {})
    params_used = props.get("params_used", {})
    correction_gt = props.get("correction_gt", {})
    correction_steps = props.get("correction_steps", [])
    is_success = props.get("is_success", True)
    object_type = props.get("object_type", "unknown")

    mt = max_timesteps if mode == "full" else None
    ts = temporal_stride if mode == "full" else None
    frames = get_frames(data, mode, max_timesteps=mt, temporal_stride=ts)
    if not frames:
        return []

    os.makedirs(output_image_dir, exist_ok=True)
    image_paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(output_image_dir, f"{episode_name}_{i}.png")
        arr = _frame_to_hwc_uint8(frame)
        Image.fromarray(arr).save(path)
        image_paths.append(path)

    n_img = len(image_paths)
    n_cams = len(_camera_keys_in_data(data)) or 1
    n_t = n_img // n_cams if n_cams else n_img
    full_rate_hint = _full_temporal_rate_hint(temporal_stride)
    obj_hint = f"（物体类型: {object_type}）" if object_type and object_type != "unknown" else ""
    samples = []

    if is_success:
        if mode == "full" and n_img >= 3:
            instruction = (
                f"共 {n_t} 个时刻×{n_cams} 视角（共 {n_img} 张）{full_rate_hint}，按时间顺序为动作从开始到结束的过程{obj_hint}。"
                f"根据图像预测抓取参数，并判断该原子动作是否成功。输出 JSON: pre_offset_x, pre_offset_z, after_offset_z, euler_deg, is_success"
            )
        else:
            instruction = f"根据图像预测抓取参数，并判断该原子动作是否成功{obj_hint}。输出 JSON 格式: pre_offset_x, pre_offset_z, after_offset_z, euler_deg, is_success"
        response_obj = {**params_used, "is_success": True}
        if props.get("was_lift_corrected") and props.get("lift_correction_gt"):
            lcg = props["lift_correction_gt"]
            response_obj["after_offset_z"] = params_used.get("after_offset_z", 0.25) + lcg.get("after_offset_z", 0.1)
        response = json.dumps(response_obj, ensure_ascii=False)
        out = {
            "instruction": instruction,
            "response": response,
            "is_success": True,
            "params_used": params_used,
            "object_type": object_type,
        }
        if _multi_image_mode(mode):
            out["images"] = image_paths
        else:
            out["image"] = image_paths[0]
        if mode == "full":
            out["num_timesteps"] = n_t
            out["num_images"] = n_img
            if temporal_stride is not None and temporal_stride >= 1:
                out["temporal_stride"] = temporal_stride
            if max_timesteps is not None and max_timesteps >= 1:
                out["max_timesteps_cap"] = max_timesteps
        samples.append(out)
    else:
        steps_to_use = correction_steps if correction_steps else [{"params": params_used, "correction_gt": correction_gt}]
        if not steps_to_use or not steps_to_use[0].get("correction_gt"):
            return []
        for step_idx, step in enumerate(steps_to_use):
            pu = step.get("params", params_used)
            cgt = step.get("correction_gt", correction_gt)
            params_str = json.dumps(pu, ensure_ascii=False)
            if mode == "full" and n_img >= 3:
                instruction = (
                    f"共 {n_t} 个时刻×{n_cams} 视角（共 {n_img} 张）{full_rate_hint}，按时间顺序为动作过程{obj_hint}。"
                    f"当前抓取参数为: {params_str}，执行失败。请修正参数并输出新的 JSON。"
                )
            else:
                prefix = f"{obj_hint} " if obj_hint else ""
                instruction = f"{prefix}当前抓取参数为: {params_str}，执行失败。请修正参数并输出新的 JSON。"
            response = _build_corrected_response(pu, cgt)
            out = {
                "instruction": instruction,
                "response": response,
                "is_success": False,
                "params_used": pu,
                "object_type": object_type,
            }
            if _multi_image_mode(mode):
                out["images"] = image_paths
            else:
                out["image"] = image_paths[0]
            if len(steps_to_use) > 1:
                out["correction_step"] = step_idx
                out["correction_steps_total"] = len(steps_to_use)
            if mode == "full":
                out["num_timesteps"] = n_t
                out["num_images"] = n_img
                if temporal_stride is not None and temporal_stride >= 1:
                    out["temporal_stride"] = temporal_stride
                if max_timesteps is not None and max_timesteps >= 1:
                    out["max_timesteps_cap"] = max_timesteps
            samples.append(out)
    return samples


def convert_dataset(
    dataset_dir: str,
    output_path: str,
    mode: str = "full",
    max_timesteps: Optional[int] = None,
    temporal_stride: Optional[int] = None,
) -> int:
    """转换整个 dataset 目录。mode: single|full；full 时可 max_timesteps / temporal_stride"""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_path).parent
    output_image_dir = output_dir / "vlm_images"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    episodes = sorted(dataset_dir.glob("episode_*.h5"))
    if not episodes:
        print(f"未找到 episode_*.h5 文件: {dataset_dir}")
        return 0

    samples = []
    for ep_path in episodes:
        try:
            data = load_episode(str(ep_path))
            if "task_properties" not in data or not data["task_properties"]:
                continue
            if "params_used" not in data["task_properties"]:
                continue
            built = build_vlm_sample(
                data,
                ep_path.stem,
                str(output_image_dir),
                mode,
                max_timesteps=max_timesteps,
                temporal_stride=temporal_stride,
            )
            for s in built:
                samples.append(s)
        except Exception as e:
            print(f"跳过 {ep_path.name}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    success_count = sum(1 for s in samples if s.get("is_success"))
    print(f"转换完成: {len(samples)} 条，成功 {success_count}，失败 {len(samples) - success_count}")
    extra_parts = []
    if mode == "full":
        if temporal_stride:
            extra_parts.append(f"temporal_stride={temporal_stride}")
        if max_timesteps:
            extra_parts.append(f"max_timesteps={max_timesteps}")
    extra = (", " + ", ".join(extra_parts)) if extra_parts else ""
    print(f"模式: {mode}{extra}")
    print(f"输出: {output_path}")
    print(f"图像: {output_image_dir}")
    return len(samples)


def main():
    parser = argparse.ArgumentParser(description="将 pick 采集数据转为 VLM 训练格式")
    parser.add_argument("dataset_dir", help="dataset 目录")
    parser.add_argument(
        "--mode",
        "-m",
        default="full",
        choices=["single", "full"],
        help="single=单帧单视角；full=多时刻×3视角（建议采集 save_frames=-1）",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        metavar="T",
        help="仅 full：均匀下采样到至多 T 个时刻（总图数≈T×3）；与 --temporal-stride 同用时，先 stride 再封顶",
    )
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=None,
        metavar="K",
        help="仅 full：每隔 K 个 HDF5 帧取一个时刻。若未指定且未指定 --max-timesteps，则默认 K=5",
    )
    parser.add_argument("--output", "-o", default=None, help="输出 JSONL 路径")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"目录不存在: {dataset_dir}")
        sys.exit(1)

    ts = args.temporal_stride
    if args.mode == "full" and ts is None and args.max_timesteps is None:
        ts = 5

    output_path = args.output or str(dataset_dir.parent / "vlm_train.jsonl")
    convert_dataset(
        str(dataset_dir),
        output_path,
        args.mode,
        max_timesteps=args.max_timesteps,
        temporal_stride=ts,
    )


if __name__ == "__main__":
    main()

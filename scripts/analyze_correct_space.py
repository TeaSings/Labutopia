"""
从成功样本中估计正确参数空间（均值、标准差、边界）。

用法:
  python scripts/analyze_correct_space.py outputs/collect/YYYY.MM.DD/HH.MM.SS_Level1_pick_success_only/dataset
  python scripts/analyze_correct_space.py outputs/collect/.../dataset -o correct_space_stats.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    np = None


def load_episode(episode_path: str) -> dict:
    """加载单个 episode 的 HDF5 或从 meta 读取"""
    try:
        import h5py
    except ImportError:
        raise ImportError("需要安装 h5py: pip install h5py")
    data = {}
    with h5py.File(episode_path, "r") as f:
        if "task_properties" in f:
            raw = f["task_properties"][()].decode("utf-8") if hasattr(f["task_properties"][()], "decode") else str(f["task_properties"][()])
            data["task_properties"] = json.loads(raw) if raw else {}
    return data


def load_episodes_from_meta(meta_path: str) -> list:
    """从 meta/episode.jsonl 加载（每行一个 JSON，含 task_properties）"""
    episodes = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            episodes.append(row.get("task_properties", row))
    return episodes


def load_episodes_from_dataset(dataset_dir: str) -> list:
    """从 dataset 目录的 HDF5 文件加载"""
    dataset_dir = Path(dataset_dir)
    episodes = []
    for p in sorted(dataset_dir.glob("episode_*.h5")):
        try:
            d = load_episode(str(p))
            if "task_properties" in d and d["task_properties"]:
                episodes.append(d["task_properties"])
        except Exception as e:
            print(f"跳过 {p.name}: {e}")
    return episodes


def flatten_params_for_stats(params_list: list) -> dict:
    """将 params_used 列表展平为各参数的一维数组，用于统计"""
    result = defaultdict(list)
    for p in params_list:
        for k, v in p.items():
            if v is None:
                continue
            if isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
                for i, x in enumerate(v):
                    suffix = ["x", "y", "z"][i] if k in ("picking_position", "euler_deg") else str(i)
                    result[f"{k}_{suffix}"].append(float(x))
            elif isinstance(v, (int, float)):
                result[k].append(float(v))
    return dict(result)


def compute_stats(arr: list) -> dict:
    """计算均值、标准差、min、max"""
    a = np.array(arr, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)) if len(a) > 1 else 0.0,
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "count": len(a),
    }


def analyze_correct_space(dataset_dir: str, success_only: bool = True) -> dict:
    """
    分析正确参数空间。
    
    Args:
        dataset_dir: dataset 目录（含 episode_*.h5）或 meta 目录的父目录
        success_only: 是否仅分析成功样本
    
    Returns:
        {object_type: {param: {mean, std, min, max, count}}}
    """
    dataset_dir = Path(dataset_dir)
    if (dataset_dir / "meta" / "episode.jsonl").exists():
        episodes = load_episodes_from_meta(str(dataset_dir / "meta" / "episode.jsonl"))
    elif (dataset_dir.parent / "meta" / "episode.jsonl").exists():
        episodes = load_episodes_from_meta(str(dataset_dir.parent / "meta" / "episode.jsonl"))
    elif dataset_dir.is_dir() and any(dataset_dir.glob("episode_*.h5")):
        episodes = load_episodes_from_dataset(str(dataset_dir))
    else:
        episodes = load_episodes_from_dataset(str(dataset_dir))

    if success_only:
        episodes = [e for e in episodes if e.get("is_success", True)]

    if not episodes:
        print("未找到成功样本")
        return {}

    by_object = defaultdict(list)
    for e in episodes:
        obj = e.get("object_type", "unknown")
        pu = e.get("params_used", {})
        if pu:
            by_object[obj].append(pu)

    stats = {}
    for obj_type, params_list in by_object.items():
        flat = flatten_params_for_stats(params_list)
        stats[obj_type] = {}
        for key, arr in flat.items():
            stats[obj_type][key] = compute_stats(arr)

    return stats


def main():
    parser = argparse.ArgumentParser(description="从成功样本估计正确参数空间")
    parser.add_argument("dataset_dir", help="dataset 目录或含 meta 的采集输出目录")
    parser.add_argument("-o", "--output", default="correct_space_stats.json", help="输出 JSON 路径")
    parser.add_argument("--include-failures", action="store_true", help="同时分析失败样本（不推荐）")
    args = parser.parse_args()

    if np is None:
        print("需要安装 numpy")
        sys.exit(1)

    stats = analyze_correct_space(args.dataset_dir, success_only=not args.include_failures)
    if not stats:
        sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"已写入 {args.output}")
    for obj_type, s in stats.items():
        print(f"\n{obj_type}: {len(s)} 个参数")
        for k, v in list(s.items())[:5]:
            print(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}, range=[{v['min']:.4f}, {v['max']:.4f}]")


if __name__ == "__main__":
    main()

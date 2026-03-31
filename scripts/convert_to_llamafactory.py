"""
将 LabUtopia VLM 格式数据转换为 LLaMA-Factory 训练格式。

输入：convert_to_vlm_format.py 输出的 JSONL（含 instruction、response、images；**images 长度可随样本变化**，与 full+temporal-stride 变长训练一致）
输出：Alpaca 或 Sharegpt 格式，供 LLaMA-Factory + Qwen2.5-VL 微调使用。

用法:
    python scripts/convert_to_llamafactory.py vlm_train.jsonl -o /path/to/LLaMA-Factory/data/labutopia_pick
    python scripts/convert_to_llamafactory.py vlm_train.jsonl -o ./llamafactory_data --format sharegpt
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path


def _process_images(
    sample: dict,
    image_dir: Path,
    output_image_dir: Path,
    copy_images: bool,
) -> list:
    """处理图像：复制或解析路径，返回相对 output_image_dir.parent 的路径列表"""
    images = sample.get("images") or ([sample["image"]] if sample.get("image") else [])
    new_paths = []
    for i, img_path in enumerate(images):
        src = Path(img_path) if Path(img_path).is_absolute() else image_dir / img_path
        if not src.exists():
            print(f"警告: 图像不存在 {src}")
            continue
        if copy_images:
            stem = Path(img_path).stem
            dst_name = f"{stem}_{i}.png" if len(images) > 1 else src.name
            dst = output_image_dir / dst_name
            shutil.copy2(src, dst)
            new_paths.append(f"images/{dst_name}")
        else:
            try:
                rel = os.path.relpath(src, output_image_dir.parent)
                new_paths.append(rel)
            except ValueError:
                new_paths.append(str(src))
    return new_paths


def convert_sample_alpaca(sample: dict, new_image_paths: list) -> dict:
    """转换为 LLaMA-Factory Alpaca 格式。columns: instruction, input, output, images"""
    return {
        "instruction": sample.get("instruction", ""),
        "input": "",
        "output": sample.get("response", ""),
        "images": new_image_paths,
    }


def convert_sample_sharegpt(sample: dict, new_image_paths: list) -> dict:
    """转换为 Sharegpt 格式。conversations + images，多图需 <image> token 与数量匹配"""
    instruction = sample.get("instruction", "")
    response = sample.get("response", "")
    image_tokens = "<image>" * len(new_image_paths)
    human_value = f"{image_tokens}{instruction}" if image_tokens else instruction
    return {
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": response},
        ],
        "images": new_image_paths,
    }


def convert_jsonl(
    input_path: str,
    output_dir: str,
    format_type: str = "alpaca",
    copy_images: bool = True,
) -> int:
    """
    转换整个 JSONL 文件。
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image_dir = output_dir / "images"
    output_image_dir.mkdir(exist_ok=True)

    input_parent = input_path.parent
    default_image_dir = input_parent / "vlm_images"

    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"跳过无效行: {e}")
                continue

            img_paths = sample.get("images") or ([sample["image"]] if sample.get("image") else [])
            if not img_paths:
                continue
            first_img = Path(img_paths[0])
            image_dir = first_img.parent if first_img.is_absolute() else default_image_dir

            new_image_paths = _process_images(
                sample, image_dir, output_image_dir, copy_images
            )
            if not new_image_paths:
                continue

            if format_type == "sharegpt":
                out_sample = convert_sample_sharegpt(sample, new_image_paths)
            else:
                out_sample = convert_sample_alpaca(sample, new_image_paths)

            samples.append(out_sample)

    # 输出 JSONL
    output_file = output_dir / "train.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"转换完成: {len(samples)} 条")
    print(f"格式: {format_type}")
    print(f"输出: {output_file}")
    print(f"图像: {output_image_dir}")
    return len(samples)


def main():
    parser = argparse.ArgumentParser(
        description="将 LabUtopia VLM 格式转为 LLaMA-Factory 训练格式"
    )
    parser.add_argument(
        "input_jsonl",
        help="convert_to_vlm_format.py 输出的 JSONL 路径（如 vlm_train.jsonl）",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="输出目录，如 LLaMA-Factory/data/labutopia_pick",
    )
    parser.add_argument(
        "--format", "-f",
        default="alpaca",
        choices=["alpaca", "sharegpt"],
        help="输出格式：alpaca（推荐）或 sharegpt",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="不复制图像，仅使用相对路径（需确保 LLaMA-Factory 能解析）",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        print(f"输入文件不存在: {input_path}")
        sys.exit(1)

    convert_jsonl(
        str(input_path),
        args.output,
        format_type=args.format,
        copy_images=not args.no_copy_images,
    )


if __name__ == "__main__":
    main()

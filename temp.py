from __future__ import annotations

import json
import sys
from pathlib import Path


DEFAULT_JSONL = Path(
    "outputs/collect/2026.04.04/10.47.35_Level1_pick_stratified_all_obj/dataset/meta/episode.jsonl"
)


def object_type_of(row: dict) -> str:
    return row.get("task_properties", {}).get("object_type", "unknown")


def pose_of(row: dict):
    task_properties = row.get("task_properties", {})
    pos = task_properties.get("object_position")
    if pos is None:
        return None
    return tuple(round(float(x), 4) for x in pos[:3])


def load_rows(jsonl_path: Path) -> list[dict]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def print_object_blocks(rows: list[dict]) -> None:
    print("\n[Object blocks]")
    if not rows:
        print("No rows found.")
        return

    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or object_type_of(rows[i]) != object_type_of(rows[start]):
            success_count = sum(
                bool(rows[j].get("task_properties", {}).get("is_success")) for j in range(start, i)
            )
            print(
                f"{start:04d}-{i-1:04d} | {object_type_of(rows[start]):20s} "
                f"| written={i-start:3d} | success={success_count:3d}"
            )
            start = i


def print_pose_blocks(rows: list[dict], max_blocks: int = 30) -> None:
    print(f"\n[First {max_blocks} pose blocks]")
    if not rows:
        print("No rows found.")
        return

    start = 0
    shown = 0
    for i in range(1, len(rows) + 1):
        current_changed = i == len(rows) or (
            object_type_of(rows[i]), pose_of(rows[i])
        ) != (
            object_type_of(rows[start]), pose_of(rows[start])
        )
        if current_changed:
            success_count = sum(
                bool(rows[j].get("task_properties", {}).get("is_success")) for j in range(start, i)
            )
            print(
                f"{start:04d}-{i-1:04d} | {object_type_of(rows[start]):20s} "
                f"| pos={pose_of(rows[start])} | written={i-start:3d} | success={success_count:2d}"
            )
            shown += 1
            start = i
            if shown >= max_blocks:
                break


def main() -> int:
    jsonl_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSONL
    if not jsonl_path.exists():
        print(f"JSONL file not found: {jsonl_path}")
        return 1

    rows = load_rows(jsonl_path)
    print(f"JSONL: {jsonl_path}")
    print(f"written: {len(rows)}")

    print_object_blocks(rows)
    print_pose_blocks(rows, max_blocks=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

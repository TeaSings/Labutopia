from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_JSONL = Path(
    "outputs/collect/2026.04.04/10.47.35_Level1_pick_stratified_all_obj/dataset/meta/episode.jsonl"
)

EPISODE_STATS_RE = re.compile(
    r"Episode Stats: Success = (?P<success_written>\d+)/(?P<written>\d+) written"
    r".*?attempted = (?P<attempted>\d+), success = (?P<success_attempted>\d+)/(?P<attempted_2>\d+)"
    r".*?not_written = (?P<not_written>\d+)"
)
WRITING_EPISODE_RE = re.compile(r"Writing episode (?P<episode>episode_\d+)\b")
FAILURE_REASON_RE = re.compile(r"Failure Reason:\s*(?P<reason>.+?)\s*$")


def object_type_of(row: dict) -> str:
    return row.get("task_properties", {}).get("object_type", "unknown")


def pose_of(row: dict):
    task_properties = row.get("task_properties", {})
    pos = task_properties.get("sampled_object_position")
    if pos is None:
        pos = task_properties.get("object_position")
    if pos is None:
        return None
    return tuple(round(float(x), 4) for x in pos[:3])


def schedule_of(row: dict) -> dict:
    return row.get("task_properties", {}).get("collection_schedule", {})


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


def print_schedule_rows(rows: list[dict], max_rows: int = 30) -> None:
    print(f"\n[First {max_rows} schedule rows]")
    for i, row in enumerate(rows[:max_rows]):
        sched = schedule_of(row)
        print(
            f"{i:04d} | obj={object_type_of(row):20s} "
            f"| obj_idx={sched.get('object_index')} "
            f"| obj_counter={sched.get('object_episode_counter')}/{sched.get('object_switch_interval')} "
            f"| pose_id={sched.get('pose_id')} "
            f"| pose_counter={sched.get('pose_counter')}/{sched.get('pose_switch_interval')} "
            f"| pose_metric={sched.get('pose_switch_metric')} "
            f"| resampled={sched.get('pose_resampled_this_reset')} "
            f"| sampled_pos={pose_of(row)}"
        )


def analyze_jsonl(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        print(f"JSONL file not found: {jsonl_path}")
        return 1

    rows = load_rows(jsonl_path)
    print(f"JSONL: {jsonl_path}")
    print(f"written: {len(rows)}")

    success_count = sum(bool(row.get("task_properties", {}).get("is_success")) for row in rows)
    print(f"success: {success_count}/{len(rows)} ({100 * success_count / max(1, len(rows)):.1f}%)")

    print_object_blocks(rows)
    print_pose_blocks(rows, max_blocks=30)
    print_schedule_rows(rows, max_rows=30)
    return 0


def iter_log_paths(paths: list[Path]) -> list[Path]:
    log_paths: list[Path] = []
    for path in paths:
        if path.is_dir():
            log_paths.extend(sorted(path.rglob("*.log")))
            log_paths.extend(sorted(path.rglob("*.txt")))
        elif path.exists():
            log_paths.append(path)
        else:
            print(f"[warn] path not found: {path}", file=sys.stderr)
    return log_paths


def normalize_reason(reason: str) -> str:
    reason = reason.strip()
    reason = re.sub(r"[-+]?\d+\.\d+", "{n}", reason)
    reason = re.sub(r"(?<![A-Za-z_])[-+]?\d+(?![A-Za-z_])", "{n}", reason)
    reason = re.sub(r"\s+", " ", reason)
    return reason


def parse_episode_stats(line: str) -> dict | None:
    match = EPISODE_STATS_RE.search(line)
    if not match:
        return None
    values = {key: int(value) for key, value in match.groupdict().items()}
    return values


def analyze_logs(paths: list[Path], normalize: bool = True, top: int = 30) -> int:
    log_paths = iter_log_paths(paths)
    if not log_paths:
        print("No log files found.")
        return 1

    reason_counts: Counter[str] = Counter()
    reason_examples: dict[str, str] = {}
    stats_by_file: dict[Path, dict] = {}
    writes_by_file: Counter[Path] = Counter()
    raw_failure_lines = 0
    counted_failure_lines = 0
    duplicate_failure_lines = 0
    task_failed_lines = 0
    task_success_lines = 0
    current_reason_counted = None
    reason_boundary_open = True
    episode_reason_counts: dict[Path, Counter[str]] = defaultdict(Counter)

    for log_path in log_paths:
        current_reason_counted = None
        reason_boundary_open = True
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                lower_line = line.lower()
                if "task success!" in lower_line:
                    task_success_lines += 1
                if "task failed!" in lower_line:
                    task_failed_lines += 1

                if WRITING_EPISODE_RE.search(line):
                    writes_by_file[log_path] += 1

                reason_match = FAILURE_REASON_RE.search(line)
                if reason_match:
                    raw_failure_lines += 1
                    raw_reason = reason_match.group("reason")
                    reason_key = normalize_reason(raw_reason) if normalize else raw_reason.strip()
                    # Many controllers print the same failure reason twice around write/stat lines.
                    if reason_key == current_reason_counted and not reason_boundary_open:
                        duplicate_failure_lines += 1
                        continue
                    reason_counts[reason_key] += 1
                    episode_reason_counts[log_path][reason_key] += 1
                    reason_examples.setdefault(reason_key, raw_reason.strip())
                    current_reason_counted = reason_key
                    reason_boundary_open = False
                    counted_failure_lines += 1

                stats = parse_episode_stats(line)
                if stats is not None:
                    stats_by_file[log_path] = stats
                    reason_boundary_open = True

    print("[Log files]")
    for log_path in log_paths:
        stats = stats_by_file.get(log_path)
        if stats:
            success = stats["success_attempted"]
            attempted = stats["attempted"]
            written = stats["written"]
            not_written = stats["not_written"]
            pct = 100 * success / max(1, attempted)
            print(
                f"- {log_path}: success={success}/{attempted} ({pct:.1f}%), "
                f"written={written}, not_written={not_written}, writes_seen={writes_by_file[log_path]}"
            )
        else:
            print(f"- {log_path}: no Episode Stats line, writes_seen={writes_by_file[log_path]}")

    total_written = sum(stats["written"] for stats in stats_by_file.values())
    total_attempted = sum(stats["attempted"] for stats in stats_by_file.values())
    total_success = sum(stats["success_attempted"] for stats in stats_by_file.values())
    total_not_written = sum(stats["not_written"] for stats in stats_by_file.values())
    if stats_by_file:
        print("\n[Aggregate latest Episode Stats per file]")
        print(
            f"success={total_success}/{total_attempted} "
            f"({100 * total_success / max(1, total_attempted):.1f}%), "
            f"written={total_written}, not_written={total_not_written}"
        )
        print(f"task_success_lines={task_success_lines}, task_failed_lines={task_failed_lines}")

    print("\n[Failure reason counts]")
    print(
        f"raw_failure_reason_lines={raw_failure_lines}, "
        f"counted_episode_reasons={counted_failure_lines}, "
        f"deduped_duplicate_lines={duplicate_failure_lines}"
    )
    if not reason_counts:
        print("No failure reasons found.")
        return 0

    total_reasons = sum(reason_counts.values())
    for reason, count in reason_counts.most_common(top):
        pct = 100 * count / max(1, total_reasons)
        example = reason_examples.get(reason, reason)
        print(f"{count:4d} | {pct:5.1f}% | {reason}")
        if normalize and example != reason:
            print(f"     example: {example}")

    print("\n[Per-file failure reason counts]")
    for log_path in log_paths:
        counts = episode_reason_counts.get(log_path)
        if not counts:
            continue
        print(f"\n{log_path}")
        for reason, count in counts.most_common(top):
            print(f"{count:4d} | {reason}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze LabUtopia output logs for failure reason counts. "
            "Also keeps the old episode.jsonl schedule summary mode."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Log file(s), directory containing logs, or an episode.jsonl file.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Force old episode.jsonl schedule analysis mode.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Do not normalize numeric values in failure reasons.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of failure reason groups to print.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = args.paths or [DEFAULT_JSONL]
    if args.jsonl or (len(paths) == 1 and paths[0].name == "episode.jsonl"):
        return analyze_jsonl(paths[0])
    return analyze_logs(paths, normalize=not args.raw, top=args.top)


if __name__ == "__main__":
    raise SystemExit(main())

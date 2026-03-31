#!/usr/bin/env python3
"""
用本地采集的 episode_*.h5 对 VLM API 跑三种任务（与 VLM_MODEL_USAGE / convert 文案对齐）。

  ① 初始推断：首时刻 cam1→cam2→cam3，共 3 张图
  ② 成功判断：成功回合的过程帧（可用 --max-timesteps 封顶，避免一次请求过大）
  ③ 参数纠错：失败回合的过程帧 + instruction 内嵌 params_used JSON

依赖：h5py、Pillow；需能 import 同目录下 convert_to_vlm_format。

用法（在 LabUtopia 根目录）:
  python scripts/test_vlm_three_tasks_from_collect.py path/to/dataset
  python scripts/test_vlm_three_tasks_from_collect.py path/to/dataset --base-url http://127.0.0.1:8000/v1 --model Qwen3.5-9B-LabUtopia-lora
  python scripts/test_vlm_three_tasks_from_collect.py path/to/dataset --success-ep episode_0005.h5 --fail-ep episode_0001.h5
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

# 同目录 convert_to_vlm_format
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from PIL import Image

import convert_to_vlm_format as cvf

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _post_json(url: str, payload: dict, api_key: str | None) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err}") from e


def _chat(base: str, model: str, api_key: str | None, content: list, max_tokens: int, title: str) -> dict:
    url = base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "do_sample": False,
    }
    print(f"\n--- {title} ---")
    out = _post_json(url, payload, api_key)
    msg = out["choices"][0]["message"]["content"]
    print("回复（前 2000 字符）:\n", msg[:2000])
    if len(msg) > 2000:
        print("... [截断]")
    return out


def _frames_to_data_uris(frames: list) -> list[str]:
    """data:image/png;base64,... 多数 OpenAI 兼容 VLM 接受；file:// 常被服务端禁止。"""
    uris = []
    for fr in frames:
        buf = BytesIO()
        Image.fromarray(cvf._frame_to_hwc_uint8(fr)).save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        uris.append(f"data:image/png;base64,{b64}")
    return uris


def _save_frames_as_file_uris(frames: list, out_dir: Path, prefix: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    uris = []
    for i, fr in enumerate(frames):
        p = (out_dir / f"{prefix}_{i:04d}.png").resolve()
        Image.fromarray(cvf._frame_to_hwc_uint8(fr)).save(p)
        uris.append(p.as_uri())
    return uris


def _mm_content(instruction_body: str, image_uris: list[str]) -> list:
    """OpenAI 风格多模态块：text 内不要写 <image>。

    LLaMA-Factory API（mm_plugin）会把各 image_url 与模板对齐；若在 text 里再写 N 个 <image>，
    会与传入的 N 张图叠成 2N 个占位符，触发 ValueError。
    """
    block: list = [{"type": "text", "text": instruction_body}]
    for u in image_uris:
        block.append({"type": "image_url", "image_url": {"url": u}})
    return block


def _init_frames(data: dict) -> list:
    keys = [k for k in cvf.CAMERA_KEYS if k in data]
    if len(keys) < 3:
        keys = cvf._camera_keys_in_data(data)
    if len(keys) < 3:
        raise RuntimeError(f"需要至少 3 个相机，当前: {keys}")
    keys = keys[:3]
    out = []
    for cam in keys:
        arr = data[cam]
        if arr is None or len(arr.shape) != 4 or arr.shape[0] < 1:
            raise RuntimeError(f"相机 {cam} 无有效帧")
        out.append(arr[0])
    return out


def _task_props(data: dict) -> dict:
    """部分 HDF5/加载路径下 task_properties 可能仍为 JSON 字符串。"""
    p = data.get("task_properties")
    if p is None:
        return {}
    if isinstance(p, dict):
        return p
    if isinstance(p, str):
        try:
            return json.loads(p) if p.strip() else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _find_success_fail(dataset: Path, max_scan: int) -> tuple[Path | None, Path | None]:
    eps = sorted(dataset.glob("episode_*.h5"))
    if not eps:
        return None, None
    ok, bad = None, None
    for i, p in enumerate(eps):
        if i >= max_scan and ok and bad:
            break
        try:
            data = cvf.load_episode(str(p))
        except Exception:
            continue
        props = _task_props(data)
        if props.get("is_success", True):
            if ok is None:
                ok = p
        else:
            if bad is None:
                bad = p
        if ok and bad:
            break
    return ok, bad


def main() -> None:
    ap = argparse.ArgumentParser(description="用本地 HDF5 测 VLM 三种任务")
    ap.add_argument("dataset", type=Path, help="含 episode_*.h5 的 dataset 目录")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI 兼容 base（含 /v1）")
    ap.add_argument(
        "--model",
        default="Qwen3.5-9B-LabUtopia-lora",
        help="与 GET /v1/models 返回的 data[].id 一致",
    )
    ap.add_argument("--api-key", default="", help="Bearer token，无则省略")
    ap.add_argument("--max-timesteps", type=int, default=12, help="任务②③过程图均匀采样时刻上限（防爆上下文）")
    ap.add_argument("--max-scan", type=int, default=400, help="自动查找成功/失败 episode 时最多检查个数")
    ap.add_argument("--success-ep", type=Path, default=None, help="指定成功 episode 文件名或路径")
    ap.add_argument("--fail-ep", type=Path, default=None, help="指定失败 episode 文件名或路径")
    ap.add_argument("--skip-init", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--skip-correct", action="store_true")
    ap.add_argument(
        "--use-file-uri",
        action="store_true",
        help="使用 file:// 绝对路径（仅当 API 明确支持本地文件时）",
    )
    args = ap.parse_args()

    raw_ds = str(args.dataset)
    if "..." in raw_ds or raw_ds.strip() in ('.', '..'):
        sys.exit(
            "路径无效：请勿使用文档里的占位符「...」，请换成 dataset 的完整路径，例如：\n"
            "  outputs\\collect\\2026.03.20\\19.53.55_xxx\\dataset"
        )
    ds = args.dataset.resolve()
    if not ds.is_dir():
        sys.exit(f"目录不存在: {ds}")

    key = args.api_key.strip() or None

    succ_path = args.success_ep
    fail_path = args.fail_ep
    if succ_path and not succ_path.is_absolute():
        succ_path = (ds / succ_path.name).resolve()
    if fail_path and not fail_path.is_absolute():
        fail_path = (ds / fail_path.name).resolve()

    if succ_path is None or fail_path is None:
        auto_ok, auto_bad = _find_success_fail(ds, args.max_scan)
        if succ_path is None:
            succ_path = auto_ok
        if fail_path is None:
            fail_path = auto_bad

    if succ_path is None:
        sys.exit("未找到成功 episode：请指定 --success-ep 或增大 --max-scan")
    if fail_path is None:
        sys.exit("未找到失败 episode：请指定 --fail-ep 或增大 --max-scan")

    data_ok = cvf.load_episode(str(succ_path))
    data_bad = cvf.load_episode(str(fail_path))
    props_ok = _task_props(data_ok)
    props_bad = _task_props(data_bad)
    obj_ok = props_ok.get("object_type", "unknown")
    obj_bad = props_bad.get("object_type", "unknown")

    cam_keys = cvf._camera_keys_in_data(data_ok)
    n_cams = len(cam_keys) or 3

    with tempfile.TemporaryDirectory(prefix="labutopia_vlm_test_") as tmp:
        tdir = Path(tmp)

        def enc_frames(frames: list, prefix: str) -> list[str]:
            if args.use_file_uri:
                return _save_frames_as_file_uris(frames, tdir, prefix)
            return _frames_to_data_uris(frames)

        # ① 初始推断
        if not args.skip_init:
            frames3 = _init_frames(data_ok)
            uris3 = enc_frames(frames3, "init")
            body = (
                f"仅起始时刻的 3 个视角 RGB（cam1→cam2→cam3）。物体类型: {obj_ok}。动作类型: pick。"
                "在尚未执行该原子动作前，根据当前场景图像预测**第一次执行**应使用的可执行参数。"
                "输出 JSON，字段包含: action_type, pre_offset_x, pre_offset_z, after_offset_z, euler_deg, picking_position（如适用）。"
                "**不要**输出 is_success。"
            )
            _chat(
                args.base_url,
                args.model,
                key,
                _mm_content(body, uris3),
                512,
                f"① 初始推断（3 图，来自 {succ_path.name} t=0）",
            )

        # ② 成功判断
        if not args.skip_judge:
            frames = cvf.get_frames(
                data_ok,
                "full",
                max_timesteps=args.max_timesteps,
                temporal_stride=None,
            )
            if not frames:
                print("[跳过] 成功判断：无法从 HDF5 取帧", file=sys.stderr)
            else:
                n_img = len(frames)
                n_t = n_img // n_cams if n_cams else n_img
                obj_hint = f"（物体类型: {obj_ok}）" if obj_ok and obj_ok != "unknown" else ""
                body = (
                    f"共 {n_t} 个时刻×{n_cams} 视角（共 {n_img} 张），按时间顺序为动作从开始到结束的过程{obj_hint}。"
                    "动作类型: pick。请根据图像判断该原子动作是否已经成功完成。"
                    "输出 JSON，仅包含两个布尔字段：is_success、terminate_correction_loop（若已可确定无需再纠错则为 true）。"
                )
                uris = enc_frames(frames, "judge")
                _chat(
                    args.base_url,
                    args.model,
                    key,
                    _mm_content(body, uris),
                    256,
                    f"② 成功判断（{n_img} 图，{succ_path.name}，max_timesteps={args.max_timesteps}）",
                )

        # ③ 参数纠错
        if not args.skip_correct:
            steps = props_bad.get("correction_steps") or []
            if not steps:
                pu = props_bad.get("params_used") or {}
                cg = props_bad.get("correction_gt") or {}
                if not cg:
                    print("[跳过] 参数纠错：无 correction_gt", file=sys.stderr)
                else:
                    steps = [{"params": pu, "correction_gt": cg}]
            step0 = steps[0]
            pu = step0.get("params") or props_bad.get("params_used") or {}
            params_str = json.dumps(pu, ensure_ascii=False)

            frames_b = cvf.get_frames(
                data_bad,
                "full",
                max_timesteps=args.max_timesteps,
                temporal_stride=None,
            )
            if not frames_b:
                print("[跳过] 参数纠错：无法从 HDF5 取帧", file=sys.stderr)
            else:
                n_img = len(frames_b)
                n_t = n_img // n_cams if n_cams else n_img
                obj_hint = f"（物体类型: {obj_bad}）" if obj_bad and obj_bad != "unknown" else ""
                body = (
                    f"共 {n_t} 个时刻×{n_cams} 视角（共 {n_img} 张），按时间顺序为动作过程{obj_hint}。"
                    "动作类型: pick。"
                    f"当前抓取参数为: {params_str}，执行失败。请修正参数并输出新的 JSON，给出全量可执行参数（含 picking_position 如适用）。"
                )
                uris_b = enc_frames(frames_b, "correct")
                _chat(
                    args.base_url,
                    args.model,
                    key,
                    _mm_content(body, uris_b),
                    512,
                    f"③ 参数纠错（{n_img} 图，{fail_path.name}）",
                )

    print("\n完成。默认使用 data:image/png;base64 传图；若 API 支持本地文件可加 --use-file-uri。")


if __name__ == "__main__":
    main()

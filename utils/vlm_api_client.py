"""OpenAI 兼容 VLM HTTP 客户端（与 run_docx/VLM_MODEL_USAGE §4.4-B 一致：text 不写 <image>）。"""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
from typing import Any

import numpy as np

from utils.vlm_image_utils import frame_to_hwc_rgb

try:
    from PIL import Image
except ImportError:
    Image = None


def _hwc_to_data_uri(hwc: np.ndarray) -> str:
    if Image is None:
        raise RuntimeError("需要 Pillow")
    hwc = frame_to_hwc_rgb(hwc)
    img = Image.fromarray(hwc)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def parse_json_bool(val: Any, default: bool = False) -> bool:
    """解析模型 JSON 中的布尔（兼容 true/false 字符串）。"""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "是")
    return default


def extract_first_json_object(text: str) -> dict[str, Any]:
    """从模型回复中抽出第一个 JSON 对象。"""
    t = text.strip()
    if "`" in t:
        t = re.sub(r"^[\s\S]*?```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```[\s\S]*$", "", t)
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError(f"回复中未找到 JSON 对象: {text[:500]!r}")
    return json.loads(m.group(0))


def _post_chat(
    base_v1: str,
    model: str,
    api_key: str | None,
    content: list[dict[str, Any]],
    max_tokens: int,
    timeout_s: float,
) -> str:
    url = base_v1.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "do_sample": False,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err}") from e
    return str(out["choices"][0]["message"]["content"])


class VlmApiClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "Qwen3.5-9B-LabUtopia-lora",
        api_key: str | None = None,
        max_tokens: int = 512,
        timeout_s: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url + "/v1"
        self.model = model
        self.api_key = (api_key or "").strip() or None
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def chat_with_images(self, text: str, images_hwc: list[np.ndarray]) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for im in images_hwc:
            uri = _hwc_to_data_uri(im)
            content.append({"type": "image_url", "image_url": {"url": uri}})
        return _post_chat(
            self.base_url,
            self.model,
            self.api_key,
            content,
            self.max_tokens,
            self.timeout_s,
        )

    def initial_pick_params(
        self,
        object_type: str,
        language_instruction: str,
        three_views_hwc: list[np.ndarray],
    ) -> dict[str, Any]:
        text = (
            f"仅起始时刻的 3 个视角 RGB（cam1→cam2→cam3）。物体类型: {object_type}。动作类型: pick。"
            f"任务说明: {language_instruction}。"
            "在尚未执行该原子动作前，根据当前场景图像预测**第一次执行**应使用的可执行参数。"
            "输出**仅一个** JSON 对象，字段包含: action_type, pre_offset_x, pre_offset_z, after_offset_z, euler_deg, picking_position（世界坐标，3 个数，米）。"
            "**不要**输出 is_success 或 markdown。"
        )
        raw = self.chat_with_images(text, three_views_hwc)
        return extract_first_json_object(raw)

    def correct_pick_params(
        self,
        object_type: str,
        language_instruction: str,
        params_used: dict[str, Any],
        frames_hwc: list[np.ndarray],
        correction_attempt: int = 1,
    ) -> dict[str, Any]:
        n = len(frames_hwc)
        n_t = n // 3 if n % 3 == 0 else max(1, n // 3)
        params_str = json.dumps(params_used, ensure_ascii=False)
        prev_n = max(0, correction_attempt - 1)
        extra = ""
        if correction_attempt >= 2:
            extra = (
                f"这是第 **{correction_attempt}** 次纠错（此前已用不同参数执行 pick 并失败 **{prev_n}** 次）。"
                "JSON 里的「当前抓取参数」就是**上一轮 pick 实际使用**的数值；你必须在**此基础上**做明显调整，"
                "禁止输出与上一段 pick 所用参数**完全相同**的 JSON（除非图像证明已无需再抓）。"
                "若场景与上次相似，请尝试**更小**或**不同方向**的 offset/euler 微调。"
            )
        text = (
            f"共约 {n_t} 个时刻×3 视角（共 {n} 张图），按时间顺序为 pick 动作过程（物体类型: {object_type}）。"
            f"任务说明: {language_instruction}。"
            f"{extra}"
            f"当前抓取参数（上一段 pick 实际使用，执行失败）: {params_str} 。"
            "请修正参数并输出**仅一个** JSON，给出**完整**可执行参数"
            "（pre_offset_x, pre_offset_z, after_offset_z, euler_deg, picking_position，世界坐标米）。"
            "**不要**输出 picking_position_delta、is_success 或 markdown。"
        )
        raw = self.chat_with_images(text, frames_hwc)
        return extract_first_json_object(raw)

    def judge_atomic_pick_success(
        self,
        object_type: str,
        language_instruction: str,
        frames_hwc: list[np.ndarray],
    ) -> dict[str, Any]:
        """任务②：多帧判断 pick 是否已成功（与训练/文档 §5.2 一致）。"""
        n = len(frames_hwc)
        n_t = n // 3 if n % 3 == 0 else max(1, n // 3)
        text = (
            f"共 {n_t} 个时刻×3 视角（共 {n} 张），按时间顺序为动作从开始到结束的过程（物体类型: {object_type}）。"
            f"任务说明: {language_instruction}。动作类型: pick。"
            "请根据图像判断该原子动作是否已经成功完成。"
            "输出**仅一个** JSON，仅包含两个布尔字段：is_success、terminate_correction_loop（若已可确定无需再纠错则为 true）。"
            "**不要**输出 markdown 或其它字段。"
        )
        raw = self.chat_with_images(text, frames_hwc)
        return extract_first_json_object(raw)

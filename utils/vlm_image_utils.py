"""仿真/采集相机数组 → HWC uint8 RGB（与 camera_utils CHW record 对齐）。"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

try:
    from PIL import Image
except ImportError:
    Image = None


def frame_to_hwc_rgb(arr: np.ndarray) -> np.ndarray:
    """单帧转为 HWC uint8 RGB。

    LabUtopia ``camera_utils.process_camera_image`` 的 record 为 **CHW (3,H,W)**。
    """
    x = np.asarray(arr)
    if x.ndim == 1:
        raw = x.tobytes()
        if Image is None:
            raise RuntimeError("需要 Pillow: pip install Pillow")
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.asarray(im, dtype=np.uint8)

    if x.dtype != np.uint8:
        if np.issubdtype(x.dtype, np.floating) and float(np.max(x)) <= 1.0 + 1e-6:
            x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            x = x.astype(np.uint8)

    if x.ndim == 2:
        return np.stack([x, x, x], axis=-1)

    if x.ndim != 3:
        raise ValueError(f"不支持的图像维度: shape={x.shape}, dtype={x.dtype}")

    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    if x.shape[-1] == 4:
        x = x[..., :3]
    if x.shape[-1] != 3:
        flat = x.astype(np.uint8).tobytes()
        if Image is None:
            raise RuntimeError("需要 Pillow")
        im = Image.open(io.BytesIO(flat)).convert("RGB")
        return np.asarray(im, dtype=np.uint8)

    return np.ascontiguousarray(x)

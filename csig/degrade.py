"""Single-image wrapper around the shared CSIG degradation mixture.

The historical API is preserved::

    lq_bgr = degrade(gt_bgr_uint8, np.random.default_rng(0))

Use ``return_metadata=True`` for an exact replay/experiment record.  Spatial and
ISP operators are shared with the GPU training transform; the final encoder is a
real libjpeg/Pillow byte round trip so generated evaluation files carry the same
integer quantisation tables and 4:2:0 layout as the measured data.
"""

from __future__ import annotations

import io
import os
import sys
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from HYPIR.dataset.csig_degradation import (  # noqa: E402
    DEFAULT_CONFIG,
    HUAWEI_MPO_C,
    HUAWEI_MPO_Y,
    DegradationConfig,
    TensorDegrader,
    sample_degradation,
)


def _encode_ordinary(bgr: np.ndarray, quality: int) -> np.ndarray:
    params = [
        int(cv2.IMWRITE_JPEG_QUALITY), int(quality),
        int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420),
    ]
    ok, encoded = cv2.imencode(".jpg", bgr, params)
    if not ok:
        raise RuntimeError("OpenCV/libjpeg failed to encode the degraded image")
    result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if result is None:
        raise RuntimeError("OpenCV/libjpeg failed to decode its own JPEG output")
    return result


def _encode_huawei(rgb: np.ndarray) -> np.ndarray:
    stream = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(
        stream,
        format="JPEG",
        qtables=[list(HUAWEI_MPO_Y), list(HUAWEI_MPO_C)],
        subsampling=2,
        optimize=False,
    )
    decoded = np.asarray(Image.open(io.BytesIO(stream.getvalue())).convert("RGB"))
    return np.ascontiguousarray(decoded[:, :, ::-1])


def degrade(
    gt_bgr: np.ndarray,
    rng: np.random.Generator,
    device: str = "cuda",
    *,
    profile: str = "auto",
    profile_hint: Optional[str] = None,
    config: DegradationConfig = DEFAULT_CONFIG,
    return_metadata: bool = False,
    jpeg_quality: Optional[int] = None,
):
    """Degrade one BGR uint8 image, optionally returning strict-JSON metadata."""

    if not isinstance(gt_bgr, np.ndarray) or gt_bgr.dtype != np.uint8:
        raise ValueError("gt_bgr must be a numpy uint8 array")
    if gt_bgr.ndim != 3 or gt_bgr.shape[2] != 3:
        raise ValueError(f"gt_bgr must have shape HxWx3, got {gt_bgr.shape}")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be numpy.random.Generator")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    metadata = sample_degradation(
        rng,
        profile=profile,
        profile_hint=profile_hint,
        config=config,
        jpeg_quality=jpeg_quality,
    )
    rgb = np.ascontiguousarray(gt_bgr[:, :, ::-1])
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device) / 255.0
    # Real byte encoding below replaces the tensor JPEG approximation.
    out = TensorDegrader().apply(tensor, [metadata], apply_jpeg=False)
    rgb_out = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    bgr_out = np.ascontiguousarray(rgb_out[:, :, ::-1])

    if metadata["jpeg_qtable_profile"] == "huawei_mpo_2026":
        lq = _encode_huawei(rgb_out)
        metadata["jpeg_backend"] = "pillow_custom_qtable"
    else:
        quality = int(metadata["jpeg_quality"])
        lq = _encode_ordinary(bgr_out, quality)
        metadata["jpeg_backend"] = "libjpeg"
    return (lq, metadata) if return_metadata else lq


__all__ = ["DEFAULT_CONFIG", "DegradationConfig", "degrade"]

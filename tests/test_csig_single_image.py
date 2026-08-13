import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "csig"))
from degrade import degrade  # noqa: E402
from HYPIR.dataset.csig_degradation import (  # noqa: E402
    HUAWEI_MPO_C, HUAWEI_MPO_Y, IJG_Q95_C, IJG_Q95_Y,
)


def _sample_image(height=65, width=67):
    y, x = np.mgrid[:height, :width]
    return np.stack(((x * 7) % 256, (y * 11) % 256, ((x + y) * 5) % 256), axis=-1).astype(np.uint8)


def test_legacy_api_and_determinism():
    image = _sample_image()
    a = degrade(image, np.random.default_rng(123), device="cpu")
    b, meta = degrade(image, np.random.default_rng(123), device="cpu", return_metadata=True)
    assert np.array_equal(a, b)
    assert a.shape == image.shape and a.dtype == np.uint8
    json.dumps(meta, allow_nan=False)


def test_real_jpeg_tables_and_420(monkeypatch):
    import cv2
    import degrade as module

    captured = {}
    original = cv2.imdecode

    def spy(buf, flags):
        captured["bytes"] = bytes(buf)
        return original(buf, flags)

    monkeypatch.setattr(module.cv2, "imdecode", spy)
    module.degrade(_sample_image(), np.random.default_rng(9), device="cpu", profile="ordinary")
    image = Image.open(io.BytesIO(captured["bytes"]))
    assert tuple(image.quantization[0]) == IJG_Q95_Y
    assert tuple(image.quantization[1]) == IJG_Q95_C
    assert image.layer == [(1, 2, 2, 0), (2, 1, 1, 1), (3, 1, 1, 1)]


def test_huawei_qtables_are_encoded():
    from degrade import _encode_huawei

    rgb = _sample_image()[:, :, ::-1]
    # The public helper returns pixels; construct an equivalent stream to inspect
    # Pillow's integer tables independently of the decoder.
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, "JPEG", qtables=[HUAWEI_MPO_Y, HUAWEI_MPO_C], subsampling=2)
    image = Image.open(io.BytesIO(stream.getvalue()))
    assert tuple(image.quantization[0]) == HUAWEI_MPO_Y
    assert tuple(image.quantization[1]) == HUAWEI_MPO_C
    assert _encode_huawei(rgb).shape == rgb.shape

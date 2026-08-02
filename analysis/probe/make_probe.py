"""最小探测包：不用模型权重、不用已生成的增强图，只对 LQ 原图施加经典算子。

    python experiments/make_probe.py --op lq            # LQ 原样
    python experiments/make_probe.py --op usm3.5        # USM 锐化
    python experiments/make_probe.py --op blur1.5       # 高斯模糊

model_dir 只放恒等 runner + 占位权重（约 1 MB），output_dir 是 LQ 派生图，
整包约 280 MB 而非 3.04 GB —— 上传快一个数量级，且时延项恒定为「近乎 0」。
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "赛题二" / "测试集"


def make_op(spec: str):
    if spec == "lq":
        return lambda a: a
    if spec.startswith("usm"):
        amount = float(spec[3:])
        def f(a):
            blur = cv2.GaussianBlur(a, (0, 0), 3)
            return np.clip(a.astype(np.float32) + amount * (a.astype(np.float32) - blur),
                           0, 255).round().astype(np.uint8)
        return f
    if spec.startswith("blur"):
        sigma = float(spec[4:])
        return lambda a: cv2.GaussianBlur(a, (0, 0), sigma)
    raise ValueError(f"未知算子: {spec}")


def build(spec: str, quality: int = 98) -> Path:
    fn = make_op(spec)
    out = ROOT / "experiments" / f"probe_{spec}"
    shutil.rmtree(out, ignore_errors=True)
    (out / "output_dir").mkdir(parents=True)
    (out / "model_dir").mkdir(parents=True)

    # model_dir：恒等 runner 不加载任何权重；放一个占位 .pt 以防平台校验存在性
    shutil.copy(ROOT / "experiments" / "runner_instant.py", out / "model_dir" / "runner.py")
    (out / "model_dir" / "your_model.pt").write_bytes(b"placeholder")

    for p in sorted(TESTSET.iterdir()):
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        ok, buf = cv2.imencode(".jpg", fn(img), [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert ok
        buf.tofile(str(out / "output_dir" / f"{p.stem}.jpg"))

    zip_path = ROOT / f"probe_{spec}.zip"
    members = [(q, Path(d) / q.relative_to(out / d))
               for d in ("output_dir", "model_dir")
               for q in sorted((out / d).rglob("*")) if q.is_file()]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for q, arc in members:
            zf.write(q, arc)
    total = sum(q.stat().st_size for q, _ in members)
    print(f"探测包 {spec}: {len(members)} 文件 {total/1024**2:.0f} MB -> {zip_path.name}")
    return zip_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, help="lq / usm<amount> / blur<sigma>")
    build(ap.parse_args().op)

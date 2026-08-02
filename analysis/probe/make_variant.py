"""产出评分探测实验用的提交包。

    python experiments/make_variant.py --variant 1   # (B, R) 只换图片为 LQ 原图
    python experiments/make_variant.py --variant 3   # (A, Z) 只换 runner 为恒等

不改动仓库里的 output_dir/ 与 model_dir/，全部在 experiments/build_<n>/ 下组装。
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


def build(variant: int) -> Path:
    out = ROOT / "experiments" / f"build_{variant}"
    shutil.rmtree(out, ignore_errors=True)
    (out / "output_dir").mkdir(parents=True)
    (out / "model_dir").mkdir(parents=True)

    # model_dir：权重/tokenizer/config 一律沿用正式版，只在实验 3 换 runner.py
    for item in ("hypir_weights.pth", "config.yaml", "tokenizer"):
        src = ROOT / "model_dir" / item
        dst = out / "model_dir" / item
        (shutil.copytree if src.is_dir() else shutil.copy)(src, dst)
    runner_src = (ROOT / "experiments" / "runner_instant.py") if variant == 3 else (ROOT / "model_dir" / "runner.py")
    shutil.copy2(runner_src, out / "model_dir" / "runner.py")

    # output_dir：实验 1 用 LQ 原图（文件名保持一致），其余沿用正式增强图。
    # 用 copy 而非 copy2：测试集里有文件的 mtime 早于 1980，zipfile 会直接拒收。
    img_src = TESTSET if variant == 1 else (ROOT / "output_dir")
    for p in sorted(img_src.iterdir()):
        shutil.copy(p, out / "output_dir" / f"{p.stem}.jpg")

    zip_path = ROOT / f"my_work_exp{variant}.zip"
    members = [(p, Path(d) / p.relative_to(out / d))
               for d in ("output_dir", "model_dir")
               for p in sorted((out / d).rglob("*")) if p.is_file()]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for p, arc in members:
            zf.write(p, arc)
    total = sum(p.stat().st_size for p, _ in members)
    print(f"变体 {variant}: {len(members)} 个文件 ({total / 1024 ** 3:.2f} GB) -> {zip_path}")
    return zip_path


def build_blend(alpha: float, quality: int = 98) -> Path:
    """画质轴变体：output_dir 换成 LQ 与增强图的像素混合，runner 保持正式版。

    这样只动画质、不动时延，且混合是纯图像运算（不用重跑 37 分钟的推理）。
    JPEG 质量沿用正式提交的 98，避免引入第二个变量。
    """
    tag = "a%03d" % round(alpha * 100)
    out = ROOT / "experiments" / f"build_{tag}"
    shutil.rmtree(out, ignore_errors=True)
    (out / "output_dir").mkdir(parents=True)
    (out / "model_dir").mkdir(parents=True)
    for item in ("hypir_weights.pth", "config.yaml", "tokenizer", "runner.py"):
        src = ROOT / "model_dir" / item
        dst = out / "model_dir" / item
        (shutil.copytree if src.is_dir() else shutil.copy)(src, dst)

    for p in sorted(TESTSET.iterdir()):
        lq = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        en = cv2.imdecode(np.fromfile(str(ROOT / "output_dir" / f"{p.stem}.jpg"), np.uint8), cv2.IMREAD_COLOR)
        mix = np.clip((1.0 - alpha) * lq.astype(np.float32) + alpha * en.astype(np.float32), 0, 255)
        ok, buf = cv2.imencode(".jpg", mix.round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert ok
        buf.tofile(str(out / "output_dir" / f"{p.stem}.jpg"))

    zip_path = ROOT / f"my_work_{tag}.zip"
    members = [(q, Path(d) / q.relative_to(out / d))
               for d in ("output_dir", "model_dir")
               for q in sorted((out / d).rglob("*")) if q.is_file()]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for q, arc in members:
            zf.write(q, arc)
    total = sum(q.stat().st_size for q, _ in members)
    print(f"混合 alpha={alpha}: {len(members)} 个文件 ({total / 1024 ** 3:.2f} GB) -> {zip_path}")
    return zip_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, choices=[1, 3])
    ap.add_argument("--alpha", type=float, help="画质轴：out = (1-a)*LQ + a*增强，用于感知分标定")
    a = ap.parse_args()
    if a.alpha is not None:
        build_blend(a.alpha)
    else:
        build(a.variant)

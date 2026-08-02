"""通用探测包组装器：{图片} x {runner}，一律不带权重。

    python experiments/make_combo.py --images model  --runner instant   # 模型图 + 恒等(≈0ms)
    python experiments/make_combo.py --images model  --runner t90       # 模型图 + 定时 90ms
    python experiments/make_combo.py --images lq     --runner t20       # LQ 图 + 定时 20ms

图片来源：model=仓库 output_dir/ 的模型增强图；lq=已提交 probe_lq 的那批（保证逐字节可比）。
model_dir 只放 runner.py + 11 字节占位 .pt，故包体只有图片的大小。
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_SRC = {
    "model": ROOT / "output_dir",
    "lq": ROOT / "experiments" / "probe_lq" / "output_dir",
}


def runner_source(spec: str) -> str:
    if spec == "instant":
        return (ROOT / "experiments" / "runner_instant.py").read_text()
    if spec.startswith("t"):
        ms = float(spec[1:])
        src = (ROOT / "experiments" / "runner_timed.py").read_text()
        return src.replace('os.environ.get("CSIG_TARGET_MS", "182.0")',
                           'os.environ.get("CSIG_TARGET_MS", "%.1f")' % ms)
    raise ValueError(f"未知 runner: {spec}")


def build(images: str, runner: str) -> Path:
    tag = f"{images}_{runner}"
    out = ROOT / "experiments" / f"combo_{tag}"
    shutil.rmtree(out, ignore_errors=True)
    (out / "output_dir").mkdir(parents=True)
    (out / "model_dir").mkdir(parents=True)

    for p in sorted(IMG_SRC[images].iterdir()):
        shutil.copy(p, out / "output_dir" / f"{p.stem}.jpg")
    (out / "model_dir" / "runner.py").write_text(runner_source(runner))
    (out / "model_dir" / "your_model.pt").write_bytes(b"placeholder")

    zip_path = ROOT / f"combo_{tag}.zip"
    members = [(q, Path(d) / q.relative_to(out / d))
               for d in ("output_dir", "model_dir")
               for q in sorted((out / d).rglob("*")) if q.is_file()]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for q, arc in members:
            zf.write(q, arc)
    print(f"combo_{tag}: {len(members)} 文件 {zip_path.stat().st_size / 1e6:.0f} MB")
    return zip_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, choices=sorted(IMG_SRC))
    ap.add_argument("--runner", required=True, help="instant / t<毫秒>，如 t90")
    a = ap.parse_args()
    build(a.images, a.runner)

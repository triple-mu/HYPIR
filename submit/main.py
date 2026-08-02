"""赛题二批量推理驱动：对输入文件夹里的每张图跑 model_dir/runner.py 的 Runner，写出增强结果。

提交产物结构（文件名为 MMDDHHMM_描述.zip，便于把线上分数对回具体版本）：
    08011842_cudaop_87ms_v100img.zip
    ├── output_dir/          本脚本产出，文件名与输入一一对应
    │   ├── case1.jpg
    │   └── ...
    └── model_dir/
        ├── runner.py        评测入口
        ├── csig_ops.py      算子后端分发
        ├── config.yaml      推理超参（含 op_backend）
        ├── custom_op/       预编译的 CUDA 扩展（cuda 后端）
        ├── hypir_weights.pth
        └── tokenizer/

用法：
    python main.py --input-dir 赛题二/测试集            # 推理，结果写到 output_dir/
    python main.py --input-dir 赛题二/测试集 --limit 3   # 只跑前 3 张，快速验证
    python main.py --pack --desc cudaop_87ms            # 打包 -> MMDDHHMM_cudaop_87ms.zip
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_key(path: Path) -> list:
    """case2 排在 case10 前面，让长任务的进度日志可读。"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path.stem)]


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    """读图 -> (1,3,H,W) fp32 [-1,1]。

    解码后先把 uint8 传上卡（36MB）再在 GPU 上转 fp32，比在 CPU 上转好再传 fp32（151MB）
    快约 20 倍（实测 9ms vs 182ms @ 12MP）。
    """
    import cv2
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None, f"无法解码: {path}"
    t = torch.from_numpy(bgr).to(device)  # (H,W,3) uint8 BGR
    t = t.flip(-1).permute(2, 0, 1)[None].float()  # (1,3,H,W) fp32 RGB [0,255]
    return t.div_(127.5).sub_(1.0)


def save_image(path: Path, x: torch.Tensor, quality: int) -> None:
    """x: (1,3,H,W) fp32 [-1,1] -> 写盘。量化在 GPU 上做，只回传 uint8。"""
    import cv2
    arr = ((x[0] + 1) * 127.5).round_().clamp_(0, 255).to(torch.uint8)
    bgr = arr.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()  # (H,W,3) uint8 BGR
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(path.suffix, bgr, params)
    assert ok, f"编码失败: {path}"
    buf.tofile(str(path))


def run(args: argparse.Namespace) -> None:
    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    files = sorted((p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural_key)
    if args.limit:
        files = files[: args.limit]
    assert files, f"{in_dir} 下没有图像"
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [(p, out_dir / f"{p.stem}{args.ext}") for p in files]
    if args.skip_existing:
        todo = [(p, o) for p, o in todo if not o.exists()]
        print(f"共 {len(files)} 张，跳过已完成 {len(files) - len(todo)} 张")
    print(f"待处理 {len(todo)} 张: {in_dir} -> {out_dir}")
    if not todo:
        return

    sys.path.insert(0, str(Path(args.model_dir).resolve()))
    from runner import Runner

    t0 = time.time()
    runner = Runner(args.model_dir)
    print(f"模型加载完成 ({time.time() - t0:.1f}s, device={runner.device}, "
          f"dtype={runner.weight_dtype})")

    t_start = time.time()
    for i, (src, dst) in enumerate(todo, 1):
        t = time.time()
        lq = load_image(src, runner.device)
        out = runner.enhance(lq)
        save_image(dst, out, args.quality)
        eta = (time.time() - t_start) / i * (len(todo) - i)
        print(f"[{i}/{len(todo)}] {src.name} {tuple(lq.shape[2:])} -> {dst.name} "
              f"({time.time() - t:.1f}s, 剩余约 {eta / 60:.1f} 分钟)", flush=True)
    print(f"全部完成，共 {len(todo)} 张，耗时 {(time.time() - t_start) / 60:.1f} 分钟")


def pack(args: argparse.Namespace) -> None:
    """把 output_dir/ 与 model_dir/ 打包成提交用的 zip。

    内容（JPEG、fp16 权重）都已接近不可压缩，用 ZIP_STORED 省掉几分钟的无效压缩。
    """
    import zipfile
    # 默认文件名带时间戳与描述，便于把线上分数对回具体是哪一版
    zip_path = Path(args.zip) if args.zip else REPO_ROOT / (
        time.strftime("%m%d%H%M") + (f"_{args.desc}" if args.desc else "") + ".zip")
    # __pycache__ 是本机 Python 版本的字节码，评测机版本不同时无用；dist-info 是 pip 元数据。
    # 两者都不该进提交包（custom_op/ 里的 _C*.so 要进，它是 cuda 后端的全部内容）
    skip = ("__pycache__", ".dist-info", ".pytest_cache")
    members = []
    for d in (Path(args.output_dir), Path(args.model_dir)):
        assert d.is_dir(), f"缺少目录: {d}"
        members += [(p, Path(d.name) / p.relative_to(d))
                    for p in sorted(d.rglob("*"))
                    if p.is_file() and not any(s in str(p.relative_to(d)) for s in skip)]
    assert members, "没有可打包的文件"

    total = sum(p.stat().st_size for p, _ in members)
    print(f"打包 {len(members)} 个文件 ({total / 1024 ** 3:.2f} GB) -> {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for p, arc in members:
            zf.write(p, arc)
    print(f"完成: {zip_path} ({zip_path.stat().st_size / 1024 ** 3:.2f} GB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="赛题二批量推理驱动")
    parser.add_argument("--input-dir", type=str, default=str(REPO_ROOT / "赛题二" / "测试集"))
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "output_dir"))
    parser.add_argument("--model-dir", type=str, default=str(REPO_ROOT / "model_dir"))
    parser.add_argument("--ext", type=str, default=".jpg", help="输出扩展名")
    parser.add_argument("--quality", type=int, default=98, help="JPEG 质量（仅 .jpg 有效）")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0 表示全部）")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已存在的输出，便于断点续跑")
    parser.add_argument("--pack", action="store_true", help="打包提交 zip（不推理）")
    parser.add_argument("--desc", type=str, default="", help="打包文件名的描述后缀，便于追溯")
    parser.add_argument("--zip", type=str, default="", help="显式指定 zip 路径；留空则用 MMDDHHMM_desc.zip")
    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    if _args.pack:
        pack(_args)
    else:
        run(_args)

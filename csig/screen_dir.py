"""对一个散图目录跑三闸门筛选，产出 file_list。

用于那些「已经是散文件」的源：ShopSign、CASIA-10K、Wikimedia 下载结果等。
（4KLSDB 在 parquet 里、PD12M 要边下边筛，各有专门脚本。）

三闸门的标定依据见 ../README.md 第三节。核心是：训练 GT 必须在**原生分辨率上真正锐利**，
否则叠上我们的退化就是双重模糊，模型会学不到该学的东西。

用法:
    python screen_dir.py <图片目录> <输出list路径> [--min-long 3000] [--workers 12]
    python screen_dir.py <图片目录> <输出list路径> --copy-to <目录>   # 顺便把过闸的拷出来
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csig_data import hf_max, ijg_quality, source_bpp

MIN_LONG = 3000
MIN_SHORT = 512
HF_MIN = 0.005      # 谱截止：保留 ~92% 真原生，误收 ~15% 假高清，误收 0% 模糊图
IJG_MIN = 93        # 量化表：仅对 IJG 族生效，厂商自研表跳过
BPP_MIN = 2.0       # 仅适用于普通单帧编码；ISO gain-map MPO 单独识别


def check(args):
    """返回 (path, 是否通过, 各闸门结果) —— 返回细节便于统计各闸门的拦截率。"""
    path, min_long = args
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    try:
        im = Image.open(path)
        if max(im.size) < min_long or min(im.size) < MIN_SHORT:
            return path, False, "尺寸"
        if hf_max(np.asarray(im.convert("L"))) < HF_MIN:
            return path, False, "谱截止"
        q = getattr(im, "quantization", None)
        if q:
            iq = ijg_quality(q[0])
            if iq is not None and iq < IJG_MIN:
                return path, False, "量化表"
        bpp, is_gain_map_mpo = source_bpp(path, im)
        if not is_gain_map_mpo and bpp < BPP_MIN:
            return path, False, "bpp"
        return path, True, "通过"
    except Exception:
        return path, False, "读取失败"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_list")
    ap.add_argument("--min-long", type=int, default=MIN_LONG)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--copy-to", default="")
    a = ap.parse_args()

    fs = []
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        fs += glob.glob(os.path.join(a.src, "**", "*." + ext), recursive=True)
    fs = sorted(set(fs))
    print("扫到 %d 张" % len(fs), flush=True)

    kept, reasons = [], {}
    with ProcessPoolExecutor(a.workers) as ex:
        for p, ok, why in ex.map(check, [(f, a.min_long) for f in fs], chunksize=8):
            reasons[why] = reasons.get(why, 0) + 1
            if ok:
                kept.append(p)

    if a.copy_to:
        os.makedirs(a.copy_to, exist_ok=True)
        pref = os.path.basename(a.src.rstrip("/"))[:8] or "img"
        moved = []
        for i, f in enumerate(kept):
            dst = os.path.join(a.copy_to, "%s_%05d.jpg" % (pref, i))
            shutil.copy(f, dst)
            moved.append(dst)
        kept = moved

    with open(a.out_list, "w") as fh:
        fh.write("\n".join(kept) + "\n")
    print("拦截明细: %s" % reasons, flush=True)
    print("通过 %d / %d (%.1f%%) -> %s" %
          (len(kept), len(fs), 100.0 * len(kept) / max(len(fs), 1), a.out_list), flush=True)


if __name__ == "__main__":
    main()

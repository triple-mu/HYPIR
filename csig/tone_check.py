"""真实退化的色调变换到底是什么样：逐通道拟合 GT -> 真实 LQ 的仿射。

目视发现合成样本有明显偏色（逐通道独立采样的仿射，通道间增益最多差 10%），
而真实 LQ 是整体发灰而非偏某个色相。这个脚本量清楚：真实退化的三个通道
增益/偏置到底差多少，好决定 AFFINE 该逐通道随机还是全局共享。

顺带用**大幅模糊后**再拟合，把模糊本身对亮度统计的影响剔掉：
先把 GT 按真实 LQ 的等效强度模糊，再逐通道最小二乘，这样拟合到的才是纯色调项。

    python csig/tone_check.py --val <验证集目录>
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def resample(x, s):
    h, w = x.shape[-2:]
    small = F.interpolate(x, size=(max(int(h / s), 8), max(int(w / s), 8)),
                          mode="bicubic", align_corners=False)
    return F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)


def fit_affine(src, dst):
    """逐通道最小二乘 dst ≈ a*src + b，返回 (a[3], b[3], 残差 RMSE)。"""
    a, b, res = [], [], []
    for c in range(3):
        s = src[0, c].flatten().double()
        d = dst[0, c].flatten().double()
        A = torch.stack([s, torch.ones_like(s)], 1)
        sol = torch.linalg.lstsq(A, d[:, None]).solution[:, 0]
        a.append(float(sol[0])); b.append(float(sol[1]))
        res.append(float(((A @ sol - d) ** 2).mean().sqrt()))
    return a, b, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--crop", type=int, default=1024)
    a_ = ap.parse_args()
    dev = torch.device("cuda")
    S = a_.crop

    print("拟合 真实LQ ≈ a*blur(GT) + b（先按等效强度模糊，剔除模糊对统计的影响）\n")
    print("%-7s %-6s %22s %22s %8s" % ("图", "等效", "逐通道增益 a (R,G,B)", "逐通道偏置 b*255", "残差"))
    print("-" * 74)
    scales = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
    for lqp in sorted(glob.glob(os.path.join(a_.val, "*_lq.jpg"))):
        base = os.path.basename(lqp).replace("_lq.jpg", "")
        gtp = glob.glob(os.path.join(a_.val, base + "_gt.*"))[0]
        gt_f = cv2.imread(gtp)[:, :, ::-1].copy()
        lq_f = cv2.imread(lqp)[:, :, ::-1].copy()
        H, W = gt_f.shape[:2]
        y, x = (H - S) // 2, (W - S) // 2
        gt = torch.from_numpy(gt_f[y:y+S, x:x+S]).permute(2,0,1)[None].float().to(dev)/255.
        lq = torch.from_numpy(lq_f[y:y+S, x:x+S]).permute(2,0,1)[None].float().to(dev)/255.
        # 用像素域残差挑等效强度（比频谱拟合更直接，且模糊与色调联合最优）
        best, berr, bfit = None, 1e18, None
        for s in scales:
            g = resample(gt, s)
            aa, bb, rr = fit_affine(g, lq)
            e = float(np.mean(rr))
            if e < berr:
                best, berr, bfit = s, e, (aa, bb)
        aa, bb = bfit
        print("%-7s %5.1fx  %22s %22s %8.4f"
              % (base, best,
                 "%.3f %.3f %.3f" % tuple(aa),
                 "%+5.1f %+5.1f %+5.1f" % tuple(v*255 for v in bb),
                 berr))
    print("\n参考：v2 的 AFFINE_A=(1.00,1.10) 逐通道独立采样、AFFINE_B=(-14/255,0) 逐通道独立")
    print("      若上面三通道的 a 彼此接近，说明真实退化的色调是**全局**的，")
    print("      逐通道独立采样会造出真实数据里不存在的随机色偏。")


if __name__ == "__main__":
    main()

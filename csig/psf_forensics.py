"""直接把 GT -> 真实LQ 的传递函数解出来，看真实退化到底是什么变换。

之前所有分析都是间接的（FR/NR 标量、径向频谱），从没直接看过这个映射本身。
维纳反解 K(f) = FFT(LQ)·conj(FFT(GT)) / (|FFT(GT)|^2 + eps)，反变换回空域就是
真实的点扩散函数（PSF），形状直接回答：高斯？盒式？带负瓣（重采样的 sinc）？
各向异性？空间是否变化？

前置检查是对齐——GT 和 LQ 若有亚像素偏移，解出来的 PSF 会被平移和展宽污染。

    python csig/psf_forensics.py --val <验证集目录> --out <输出目录>
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def to_t(a, dev):
    return torch.from_numpy(a).permute(2, 0, 1)[None].float().to(dev) / 255.


def align_shift(gt, lq):
    """相位相关求整数+亚像素平移量。"""
    g = gt.mean(1)[0].double()
    l = lq.mean(1)[0].double()
    G, L = torch.fft.fft2(g), torch.fft.fft2(l)
    R = G * L.conj()
    R = R / (R.abs() + 1e-12)
    c = torch.fft.ifft2(R).real
    idx = int(torch.argmax(c))
    H, W = c.shape
    dy, dx = idx // W, idx % W
    if dy > H // 2: dy -= H
    if dx > W // 2: dx -= W
    return dy, dx, float(c.max())


def _hann2d(h, w, dev):
    """二维 Hann 窗。FFT 反解假定循环卷积，不加窗的话图像边界的跳变会被当成
    高频注入传递函数，把 PSF 污染出振铃。"""
    wy = torch.hann_window(h, periodic=False, dtype=torch.float64, device=dev)
    wx = torch.hann_window(w, periodic=False, dtype=torch.float64, device=dev)
    return wy[:, None] * wx[None, :]


def estimate_psf(gt, lq, ksize=21, eps=1e-3):
    """维纳反解传递函数，返回中心 ksize x ksize 的 PSF（每通道）。"""
    psfs = []
    win = _hann2d(gt.shape[-2], gt.shape[-1], gt.device)
    for c in range(3):
        g = gt[0, c].double()
        l = lq[0, c].double()
        g = (g - g.mean()) * win
        l = (l - l.mean()) * win
        G, L = torch.fft.fft2(g), torch.fft.fft2(l)
        K = (L * G.conj()) / (G.abs() ** 2 + eps * (G.abs() ** 2).mean())
        k = torch.fft.fftshift(torch.fft.ifft2(K).real)
        H, W = k.shape
        r = ksize // 2
        psfs.append(k[H // 2 - r:H // 2 + r + 1, W // 2 - r:W // 2 + r + 1].cpu().numpy())
    return np.stack(psfs)


def psf_report(k, tag):
    """能量集中度、各向异性、负瓣占比。"""
    kk = k.mean(0)
    kk = kk / kk.sum()
    r = kk.shape[0] // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    w = np.abs(kk)
    cy = (yy * w).sum() / w.sum(); cx = (xx * w).sum() / w.sum()
    vy = ((yy - cy) ** 2 * w).sum() / w.sum()
    vx = ((xx - cx) ** 2 * w).sum() / w.sum()
    neg = kk[kk < 0].sum() / np.abs(kk).sum()
    c1 = kk[r, r]
    c3 = kk[r - 1:r + 2, r - 1:r + 2].sum()
    print("  %-8s 等效sigma %.2f(y) %.2f(x)  各向异性 %.2f  中心权重 %.3f  3x3内 %.3f  负瓣占比 %.1f%%"
          % (tag, np.sqrt(max(vy, 0)), np.sqrt(max(vx, 0)),
             np.sqrt(max(vy, 1e-9) / max(vx, 1e-9)), c1, c3, 100 * neg))
    return kk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--ksize", type=int, default=41)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    S = a.crop

    for lqp in sorted(glob.glob(os.path.join(a.val, "*_lq.jpg"))):
        base = os.path.basename(lqp).replace("_lq.jpg", "")
        gtp = glob.glob(os.path.join(a.val, base + "_gt.*"))[0]
        gt_f = cv2.imread(gtp)[:, :, ::-1].copy()
        lq_f = cv2.imread(lqp)[:, :, ::-1].copy()
        assert gt_f.shape == lq_f.shape, (gt_f.shape, lq_f.shape)
        H, W = gt_f.shape[:2]
        print("\n=== %s  %dx%d ===" % (base, W, H))

        # 全图解一次（用户要求：crop 的边界效应会污染 FFT 反解），
        # 再在三个位置各解一次看空间是否变化
        pos = [("全图", None, None),
               ("左上", H // 4 - S // 2, W // 4 - S // 2),
               ("中心", (H - S) // 2, (W - S) // 2),
               ("右下", 3 * H // 4 - S // 2, 3 * W // 4 - S // 2)]
        psf_mid = None
        for tag, y, x in pos:
            if y is None:
                gt, lq = to_t(gt_f, dev), to_t(lq_f, dev)
            else:
                y = max(0, min(y, H - S)); x = max(0, min(x, W - S))
                gt = to_t(gt_f[y:y + S, x:x + S], dev)
                lq = to_t(lq_f[y:y + S, x:x + S], dev)
            dy, dx, pk = align_shift(gt, lq)
            k = estimate_psf(gt, lq, a.ksize)
            kk = psf_report(k, tag)
            if tag == "全图":
                psf_mid = (k, kk, gt, lq)
            print("           对齐: dy=%d dx=%d 相关峰=%.3f  |  逐通道中心权重 %.3f %.3f %.3f"
                  % (dy, dx, pk, k[0][a.ksize//2, a.ksize//2],
                     k[1][a.ksize//2, a.ksize//2], k[2][a.ksize//2, a.ksize//2]))

        k, kk, gt, lq = psf_mid
        # PSF 剖面（水平/垂直/对角）与常见核对照
        r = a.ksize // 2
        prof_h = kk[r, :] / kk[r, r]
        prof_v = kk[:, r] / kk[r, r]
        print("  PSF 水平剖面(归一化到中心，中心 +-8):")
        print("    ", " ".join("%+.3f" % v for v in prof_h[r-8:r+9]))
        print("  PSF 垂直剖面:")
        print("    ", " ".join("%+.3f" % v for v in prof_v[r-8:r+9]))
        # 能量随半径的累积，直接看核有多宽
        yy, xx = np.mgrid[-r:r+1, -r:r+1]
        rad = np.sqrt(yy**2 + xx**2)
        tot = np.abs(kk).sum()
        acc = [float(np.abs(kk)[rad <= t].sum() / tot) for t in (1, 2, 3, 5, 8, 12, 16, 20)]
        print("  |PSF| 能量累积 (r<=1,2,3,5,8,12,16,20):", " ".join("%.2f" % v for v in acc))

        # 残差（对齐后、去掉最佳全局仿射）的性质
        res = (lq - gt)[0].mean(0)
        print("  残差 GT-LQ: 均值 %+.4f  标准差 %.4f  |  高频占比 %.2f"
              % (float(res.mean()), float(res.std()),
                 float((res - F.avg_pool2d(res[None,None], 8, 8, count_include_pad=False)
                        .repeat_interleave(8,-1).repeat_interleave(8,-2)[0,0][:res.shape[0],:res.shape[1]]).std()
                       / res.std())))

        # 存图：PSF 放大 + 残差
        vis = kk - kk.min()
        vis = (vis / max(vis.max(), 1e-12) * 255).astype(np.uint8)
        vis = cv2.resize(vis, (a.ksize * 16, a.ksize * 16), interpolation=cv2.INTER_NEAREST)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
        rv = res.cpu().numpy()
        rv = np.clip((rv - rv.mean()) / (4 * rv.std()) * 127 + 128, 0, 255).astype(np.uint8)
        rv = cv2.resize(rv, (a.ksize * 16, a.ksize * 16))
        cv2.imwrite(os.path.join(a.out, "%s_psf.png" % base), vis)
        cv2.imwrite(os.path.join(a.out, "%s_residual.png" % base),
                    cv2.applyColorMap(rv, cv2.COLORMAP_BONE))


if __name__ == "__main__":
    main()

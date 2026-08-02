"""频域最优 LSI 滤波器（径向各向同性）的上界评估。
g*(f) = E[Re(G conj(L))] / E[|L|^2]，在验证集上按径向频段聚合，做留一 case 交叉验证。
这是任何"固定参数的线性去模糊/反卷积"能达到的理论上限。
"""
import numpy as np
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CROPS = HERE + "/crops/"
CASES = [f"{c}_{i}" for c in ("case1", "case2", "case3") for i in range(5)]
S = 1024
_yy, _xx = np.mgrid[:S, :S]
_RR = np.sqrt((_yy - S / 2) ** 2 + (_xx - S / 2) ** 2).astype(np.int32)
NB = _RR.max() + 1
_CNT = np.bincount(_RR.ravel(), minlength=NB)
_WIN = np.outer(np.hanning(S), np.hanning(S)).astype(np.float32)


def stats(name):
    lq = np.load(f"{CROPS}{name}_lq.npy").astype(np.float32)
    gt = np.load(f"{CROPS}{name}_gt.npy").astype(np.float32)
    num = np.zeros(NB)
    den = np.zeros(NB)
    for c in range(3):
        L = np.fft.fftshift(np.fft.fft2(lq[:, :, c] * _WIN))
        G = np.fft.fftshift(np.fft.fft2(gt[:, :, c] * _WIN))
        num += np.bincount(_RR.ravel(), (L.conj() * G).real.ravel(), minlength=NB)
        den += np.bincount(_RR.ravel(), (np.abs(L) ** 2).ravel(), minlength=NB)
    return num, den


def gain(cases):
    n = np.zeros(NB)
    d = np.zeros(NB)
    for c in cases:
        a, b = stats(c)
        n += a
        d += b
    return n / np.maximum(d, 1e-9)


def apply_gain(img, g):
    """把径向增益曲线做成 2D 频域掩码并应用（图任意尺寸，按归一化频率插值）。"""
    H, W = img.shape[:2]
    yy, xx = np.mgrid[:H, :W]
    fy = (yy - H / 2) / H
    fx = (xx - W / 2) / W
    rr = np.sqrt(fy ** 2 + fx ** 2) * S  # 映射回 1024 网格的径向 bin
    M = np.interp(rr, np.arange(NB), g).astype(np.float32)
    M = np.fft.ifftshift(M)
    out = np.empty_like(img, np.float32)
    x = img.astype(np.float32)
    for c in range(3):
        out[:, :, c] = np.fft.ifft2(np.fft.fft2(x[:, :, c]) * M).real
    return np.clip(out, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import metrics
    base = json.load(open(HERE + "/baseline.json"))
    g_all = gain(CASES)
    np.save(HERE + "/gain_all.npy", g_all)
    print("全量拟合的最优径向增益 g*(f):")
    for i in [0, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512]:
        print(f"  r={i:4d} (f={i/S:.3f} cyc/px)  g*={g_all[i]:.4f}")

    for tag, mk in [("in-fit", lambda h: CASES),
                    ("LOCV", lambda h: [n for n in CASES if not n.startswith(h)])]:
        dp, ds = [], []
        for hold in ["case1", "case2", "case3"]:
            g = gain(mk(hold))
            for n in [x for x in CASES if x.startswith(hold)]:
                lq = np.load(f"{CROPS}{n}_lq.npy")
                gt = np.load(f"{CROPS}{n}_gt.npy")
                o = apply_gain(lq, g)
                dp.append(metrics.psnr(o, gt) - base[n]["psnr"])
                ds.append(metrics.ssim(o, gt) - base[n]["ssim"])
        print(f"{tag}: dPSNR={np.mean(dp):+.4f}  dSSIM={np.mean(ds):+.5f}")

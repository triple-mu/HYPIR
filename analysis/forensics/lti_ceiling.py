"""空域最小二乘最优 K x K 线性核（含常数项），双向：
  forward : LQ ~ K * GT   -> LTI 退化模型的上界（能不能用一个卷积核合成 LQ）
  inverse : GT ~ K * LQ   -> 确定性线性复原的上界（不用生成先验能挣多少 dB）
用 FFT 算自相关/互相关构造正规方程，全图无子采样，in-sample 最优。
"""
import sys, numpy as np, cv2

D = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"


def load(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    assert im is not None, p
    return im.astype(np.float64)


def fit_kernel(x, y, K):
    """min ||conv(x,k)+c - y||^2, k is KxK. 返回 (k, c)。x,y: HxW float64"""
    H, W = x.shape
    r = K // 2
    fh, fw = H + K, W + K
    Fx = np.fft.rfft2(x, (fh, fw))
    Fy = np.fft.rfft2(y, (fh, fw))
    Rxx = np.fft.irfft2(Fx * np.conj(Fx), (fh, fw))   # Rxx[d] = sum x[p]x[p-d]
    Rxy = np.fft.irfft2(np.conj(Fx) * Fy, (fh, fw))   # Rxy[d] = sum x[p] y[p+d]
    n = H * W
    sx, sy = x.sum(), y.sum()
    idx = [(i, j) for i in range(-r, r + 1) for j in range(-r, r + 1)]
    m = len(idx)
    A = np.empty((m + 1, m + 1))
    b = np.empty(m + 1)
    for a, (i1, j1) in enumerate(idx):
        for bb, (i2, j2) in enumerate(idx):
            A[a, bb] = Rxx[(i1 - i2) % fh, (j1 - j2) % fw]
        A[a, m] = sx
        A[m, a] = sx
        b[a] = Rxy[(-i1) % fh, (-j1) % fw]
    A[m, m] = n
    b[m] = sy
    sol = np.linalg.solve(A + np.eye(m + 1) * 1e-6 * A[m // 2, m // 2], b)
    k = sol[:m].reshape(K, K)
    return k, sol[m]


def psnr(a, b, bd=64):
    d = a[bd:-bd, bd:-bd] - b[bd:-bd, bd:-bd]
    return 10 * np.log10(255.0 ** 2 / np.mean(d ** 2))


for case in (1, 2, 3):
    gtp = f"{D}/case{case}_gt." + ("png" if case in (1, 2) else "jpg")
    gt = load(gtp)
    lq = load(f"{D}/case{case}_lq.jpg")
    if case == 3:          # 去掉上下黑边（analysis_tone t16：73-76 行）
        gt, lq = gt[96:-96], lq[96:-96]
    print(f"=== case{case}  shape={gt.shape}  PSNR(LQ,GT)={np.mean([psnr(lq[...,c],gt[...,c]) for c in range(3)]):.3f} dB")
    for K in (1, 9, 21, 31):
        pf, pi = [], []
        for c in range(3):
            g, l = gt[..., c], lq[..., c]
            kf, cf = fit_kernel(g, l, K)      # forward: 用 GT 预测 LQ
            pred = cv2.filter2D(g, -1, kf[::-1, ::-1], borderType=cv2.BORDER_REFLECT) + cf
            pf.append(psnr(pred, l))
            ki, ci = fit_kernel(l, g, K)      # inverse: 用 LQ 预测 GT
            pred = cv2.filter2D(l, -1, ki[::-1, ::-1], borderType=cv2.BORDER_REFLECT) + ci
            pi.append(psnr(pred, g))
            if K == 31 and c == 1:
                np.save(f"/tmp/kinv_case{case}.npy", ki)
                np.save(f"/tmp/kfwd_case{case}.npy", kf)
        print(f"  K={K:2d}  forward LQ~K*GT: {np.mean(pf):7.3f} dB   |   inverse GT~K*LQ: {np.mean(pi):7.3f} dB")
    sys.stdout.flush()

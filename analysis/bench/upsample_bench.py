"""Upsample2D 的四种等价实现对拍与计时。

nsys 显示 6 个 `ConvTranspose` 走的是 `sm70_xmma_dgrad_implicit_gemm_indexed`，V100 上合计
4.98 ms/iter（占 GPU 时间 5.9%）。dgrad 是 stride>1 的索引式 gather 路径，效率不如 fprop，
所以值得试试能不能换成普通卷积。

四种实现（数学上全部等价于原始的 `nearest x2 + conv3x3`）：
    convt     现状：折叠成 ConvTranspose2d(k=4, s=2, p=1)，dgrad
    nearest   折叠前的原始形式，fprop 但空间工作量 4x
    sub2      子像素：pad 1 圈后做 k=2、输出 4*Co 的 fprop，再按子格写回。MAC 与 convt 相同
    sub3      子像素：k=3 嵌 2x2 有效抽头 + pixel_shuffle。单次卷积不用切片，但 MAC 是 2.25x

用法：
    python csig_bench/upsample_bench.py            # 先 CPU float64 验等价，再 GPU 计时
    python csig_bench/upsample_bench.py --layout nchw
"""

import argparse
import time

import torch
from torch.nn import functional as F

# 生产形状 (Cin=Cout, H)，来自 512x512 下的一次真实 infer
SHAPES = [("unet", 1280, 8), ("unet", 1280, 16), ("unet", 640, 32),
          ("vae", 512, 64), ("vae", 512, 128), ("vae", 256, 256)]

_CONVT_TAPS = {0: (2,), 1: (1, 2), 2: (0, 1), 3: (0,)}


def fold_convt(w3: torch.Tensor) -> torch.Tensor:
    """3x3 conv + nearest 上采样 -> ConvTranspose2d(k=4,s=2,p=1) 权重 (Ci, Co, 4, 4)。"""
    v = torch.zeros(w3.shape[0], w3.shape[1], 4, 4, dtype=w3.dtype, device=w3.device)
    for kp, sp in _CONVT_TAPS.items():
        for kq, sq in _CONVT_TAPS.items():
            v[:, :, kp, kq] = w3[:, :, sp, :][:, :, :, sq].sum((2, 3))
    return v.transpose(0, 1).contiguous()


def fold_sub2(v: torch.Tensor) -> torch.Tensor:
    """ConvTranspose 权重 (Ci,Co,4,4) -> 子像素 k=2 权重 (4*Co, Ci, 2, 2)。

    out[2i+a] = Σ_d V[k] · in[i-d]，k = 2d+a+1；a=0 -> d∈{0,1}，a=1 -> d∈{-1,0}。
    输入两侧各 pad 1 后取 slice s=a，tap u 读到 x[i+a+u-1]，故 d = 1-a-u。
    """
    ci, co = v.shape[:2]
    w = torch.zeros(4 * co, ci, 2, 2, dtype=v.dtype, device=v.device)
    for a in range(2):
        for b in range(2):
            g = a * 2 + b
            for u in range(2):
                for t in range(2):
                    w[g * co:(g + 1) * co, :, u, t] = v[:, :, 2 * (1 - a - u) + a + 1,
                                                        2 * (1 - b - t) + b + 1].T
    return w


def fold_sub3(v: torch.Tensor) -> torch.Tensor:
    """-> 子像素 k=3 权重 (4*Co, Ci, 3, 3)，通道序按 pixel_shuffle 的 c*4 + a*2 + b 排。

    k=3、padding=1 时 tap t 读到 x[i+t-1]，四个子格共用同一对齐：a=0 用 t∈{0,1}，
    a=1 用 t∈{1,2}，余下一列填 0。多算 2.25 倍乘加，换来单次卷积 + pixel_shuffle。
    """
    ci, co = v.shape[:2]
    w = torch.zeros(co * 4, ci, 3, 3, dtype=v.dtype, device=v.device)
    for a in range(2):
        for b in range(2):
            for t in range(2):
                for s in range(2):
                    # tap 位置 t+a 读到 x[i+t+a-1]，即 d = 1-a-t
                    w[a * 2 + b::4, :, t + a, s + b] = v[:, :, 2 * (1 - a - t) + a + 1,
                                                         2 * (1 - b - s) + b + 1].T
    return w


def make_impls(w3: torch.Tensor, bias: torch.Tensor, mf):
    """返回 {名字: 可调用}，权重都按 mf 预布局好（生产里是加载期折叠，不计入时延）。"""
    v = fold_convt(w3)
    w2 = fold_sub2(v)
    ws = fold_sub3(v)
    if mf is torch.channels_last:
        w3, v, w2, ws = (t.contiguous(memory_format=mf) for t in (w3, v, w2, ws))
    # sub2 的通道序是 g-major（每组连续 Co 个），sub3 是 c-major（pixel_shuffle 要求 c*4+g）
    b_g = bias.repeat(4)
    b_c = bias.repeat_interleave(4)
    co = bias.numel()

    def convt(x):
        return F.conv_transpose2d(x, v, bias, stride=2, padding=1)

    def nearest(x):
        return F.conv2d(F.interpolate(x, scale_factor=2.0, mode="nearest"), w3, bias, padding=1)

    def sub2(x):
        y = F.conv2d(F.pad(x, (1, 1, 1, 1)), w2, b_g)
        n, _, h, _ = x.shape
        out = torch.empty((n, co, 2 * h, 2 * h), dtype=x.dtype, device=x.device,
                          memory_format=mf or torch.contiguous_format)
        for a in range(2):
            for b in range(2):
                g = a * 2 + b
                out[:, :, a::2, b::2] = y[:, g * co:(g + 1) * co, a:a + h, b:b + h]
        return out

    def sub3(x):
        return F.pixel_shuffle(F.conv2d(x, ws, b_c, padding=1), 2)

    return {"convt": convt, "nearest": nearest, "sub2": sub2, "sub3": sub3}


def check_equivalence() -> None:
    """CPU float64：四种实现必须互相一致到机器精度。GPU 上 fp16/TF32 会把真 bug 藏掉。"""
    torch.manual_seed(0)
    print("== CPU float64 等价性 ==")
    for ci, co, h in ((3, 5, 7), (8, 8, 6), (4, 6, 9)):
        x = torch.randn(1, ci, h, h, dtype=torch.float64)
        w3 = torch.randn(co, ci, 3, 3, dtype=torch.float64)
        bias = torch.randn(co, dtype=torch.float64)
        impls = make_impls(w3, bias, None)
        ref = impls["nearest"](x)
        errs = {k: float((fn(x) - ref).abs().max()) for k, fn in impls.items()}
        ok = all(e < 1e-12 for e in errs.values())
        print(f"  Ci={ci} Co={co} H={h}  {'OK ' if ok else '失败 '}"
              + "  ".join(f"{k}={v:.2e}" for k, v in errs.items()))
        assert ok, errs


def main() -> None:
    ap = argparse.ArgumentParser(description="Upsample2D 四实现基准")
    ap.add_argument("--layout", default="nhwc", choices=["nhwc", "nchw"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()

    if not args.skip_check:
        check_equivalence()
    if not torch.cuda.is_available():
        return

    mf = torch.channels_last if args.layout == "nhwc" else None
    torch.backends.cudnn.benchmark = True
    print(f"\n== {torch.cuda.get_device_name(0)}  fp16  {args.layout} ==")
    names = ["convt", "nearest", "sub2", "sub3"]
    print(f"{'形状':24s}" + "".join(f"{n + ' ms':>11s}" for n in names) + f"{'最优':>10s}")
    total = dict.fromkeys(names, 0.0)
    for tag, c, h in SHAPES:
        x = torch.randn(1, c, h, h, device="cuda", dtype=torch.float16)
        if mf:
            x = x.contiguous(memory_format=mf)
        w3 = torch.randn(c, c, 3, 3, device="cuda", dtype=torch.float16) * 0.02
        bias = torch.randn(c, device="cuda", dtype=torch.float16) * 0.02
        impls = make_impls(w3, bias, mf)
        ms = {}
        for n in names:
            fn = impls[n]
            for _ in range(10):
                fn(x)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(args.iters):
                fn(x)
            torch.cuda.synchronize()
            ms[n] = (time.perf_counter() - t) / args.iters * 1e3
            total[n] += ms[n]
        best = min(ms, key=ms.get)
        print(f"{tag + f' C={c} {h}->{2 * h}':24s}"
              + "".join(f"{ms[n]:11.3f}" for n in names) + f"{best:>10s}")
    print(f"{'合计':24s}" + "".join(f"{total[n]:11.3f}" for n in names)
          + f"{min(total, key=total.get):>10s}")
    ref = total["convt"]
    print("\n相对现状(convt)：" + "  ".join(f"{n} {ref / total[n]:.2f}x" for n in names))


if __name__ == "__main__":
    main()

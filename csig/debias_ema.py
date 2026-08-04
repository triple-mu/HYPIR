"""给 EMA 快照做 bias correction。

HYPIR/utils/ema.py:15-17 用当前参数 θ₀ 初始化 EMA 且**没有 bias correction**，
于是 EMA_t = (1-d^t)·(真正的滑动平均) + d^t·θ₀。

官方从零训 LoRA 时这无害：θ₀ 是 B=0 的恒等映射，掺进去不改变什么。
但我们是从**官方 HYPIR_sd2.pth 续训**的，θ₀ 就是「官方 LoRA + TAESD」这个已知的坏点
（arm B：CLIP-IQA 0.6279 / MANIQA 0.4567，都明显差于官方 + SD-VAE）。
decay=0.999 时 EMA@1000 里有 0.999^1000 = 36.8% 是这个坏点，
EMA@750 是 47.2%、EMA@1250 是 28.6%。

去偏后 θ_corr = (EMA_t − d^t·θ₀)/(1 − d^t)，它仍是真实迭代点 θ₁..θ_t 的凸组合
（权重全正、和为 1），与 EMA 同类，不引入新的外插风险。

    python csig/debias_ema.py <ema_state_dict.pth> <theta0.pth> <step> <out.pth> [decay]
"""
import sys

import torch


def main():
    p_ema, p_theta0, step, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    decay = float(sys.argv[5]) if len(sys.argv) > 5 else 0.999

    ema = torch.load(p_ema, map_location="cpu", weights_only=True)
    t0 = torch.load(p_theta0, map_location="cpu", weights_only=True)
    missing = set(ema) - set(t0)
    assert not missing, "θ₀ 缺 %d 个 key，例如 %s" % (len(missing), list(missing)[:2])

    w = decay ** step                       # θ₀ 的残留权重
    corrected = {k: (ema[k].float() - w * t0[k].float()) / (1.0 - w) for k in ema}
    torch.save(corrected, out)
    print("step=%d decay=%.4f -> θ₀ 残留 %.2f%%，%d 个张量 -> %s"
          % (step, decay, 100 * w, len(corrected), out))


if __name__ == "__main__":
    main()

"""在两份 LoRA 权重之间线性插值：out = (1-lam)*A + lam*B。

用来在失真-感知轴上滑动。实测 EMA 快照与 raw 快照分别落在这条轴的两侧
（EMA 的 LPIPS/NIQE 好、raw 的 MUSIQ/MANIQA/CLIP-IQA 好），中间可能有更好的折中点。

注意：LoRA 的增量是 B·A 两个矩阵的乘积，分别插值 A 和 B **不等于**插值它们的乘积，
有交叉项。但两份权重来自同一次训练、相隔几百步，彼此很近，近似成立 ——
而且 EMA 本身干的就是同一件事（对 A、B 各自做指数平均）。

    python csig/blend_lora.py <A.pth> <B.pth> <lam> <out.pth>
"""
import sys

import torch


def main():
    pa, pb, lam, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    a = torch.load(pa, map_location="cpu", weights_only=True)
    b = torch.load(pb, map_location="cpu", weights_only=True)
    assert set(a) == set(b), "两份权重的 key 不一致：%d vs %d" % (len(a), len(b))
    blended = {k: (1 - lam) * a[k].float() + lam * b[k].float() for k in a}
    torch.save(blended, out)
    print("%d 个张量，lam=%.2f -> %s" % (len(blended), lam, out))


if __name__ == "__main__":
    main()

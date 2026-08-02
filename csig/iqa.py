"""画质指标封装。

只用 TOPIQ-FR（有参考）+ MANIQA（无参考）。依据 FINDINGS §4：用三组已知感知分的图反查，
只有这两个能复现实测排序（模型 2.655 > LQ 1.000 > USM3.5 0.820）；LPIPS / DISTS / MUSIQ /
TOPIQ-NR / NIMA / NIQE / BRISQUE / HyperIQA 全都把过锐化的 USM3.5 排在 LQ 之上，
PSNR/SSIM 更是完全反向。
"""

import torch

# FINDINGS §4 拟合出的代理映射。注意它是 3 方程 3 未知量的精确解，零验证价值，
# 只能当排序代理用，不能当绝对分数信。
_W_FR, _W_NR, _BIAS = 13.674, 4.477, -5.731

_CACHE = {}


def get_metrics(device="cuda"):
    if device not in _CACHE:
        import pyiqa
        _CACHE[device] = (pyiqa.create_metric("topiq_fr", device=device),
                          pyiqa.create_metric("maniqa", device=device))
    return _CACHE[device]


def to01(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1]，指标模型要的输入域。"""
    return (x + 1.0).mul_(0.5).clamp_(0.0, 1.0)


@torch.no_grad()
def score(out: torch.Tensor, gt: torch.Tensor, device="cuda"):
    """out/gt: [1,3,H,W] in [-1,1]。返回 (topiq_fr, maniqa, proxy)。"""
    fr, nr = get_metrics(device)
    o, g = to01(out.float().clone()), to01(gt.float().clone())
    v_fr = float(fr(o, g))
    v_nr = float(nr(o))
    return v_fr, v_nr, _W_FR * v_fr + _W_NR * v_nr + _BIAS

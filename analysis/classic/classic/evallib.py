"""赛题二经典算子评测库：加载验证集 crop、计算 PSNR/SSIM/LPIPS/NIQE。"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

VAL_DIR = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集/"
TEST_DIR = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集/"
CACHE = "/home/ubuntu/workspace/contest/CSIG-2026/csig_bench/classicE/cache"
os.makedirs(CACHE, exist_ok=True)

CASES = ["case1", "case2", "case3"]
_GT_NAME = {"case1": "case1_gt.png", "case2": "case2_gt.png", "case3": "case3_gt.jpg"}


def load_full(case, kind):
    """返回 BGR uint8 全图。"""
    p = VAL_DIR + (_GT_NAME[case] if kind == "gt" else f"{case}_lq.jpg")
    return cv2.cvtColor(np.asarray(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)


def crop_set(size=1024, n_off=3):
    """返回 [(tag, lq_crop, gt_crop)]；每张图取 n_off 个 crop（中心 + 偏移）。"""
    f = f"{CACHE}/crops_{size}_{n_off}.npz"
    if os.path.exists(f):
        d = np.load(f)
        return [(k, d[k + "_lq"], d[k + "_gt"]) for k in d["tags"]]
    out, store = [], {}
    for c in CASES:
        lq, gt = load_full(c, "lq"), load_full(c, "gt")
        H, W = lq.shape[:2]
        cy, cx = H // 2, W // 2
        offs = [(0, 0), (-H // 4, -W // 4), (H // 4, W // 4), (-H // 4, W // 4), (H // 4, -W // 4)][:n_off]
        for i, (dy, dx) in enumerate(offs):
            y = int(np.clip(cy + dy - size // 2, 0, H - size))
            x = int(np.clip(cx + dx - size // 2, 0, W - size))
            tag = f"{c}_{i}"
            a, b = lq[y:y + size, x:x + size].copy(), gt[y:y + size, x:x + size].copy()
            out.append((tag, a, b))
            store[tag + "_lq"], store[tag + "_gt"] = a, b
    np.savez(f, tags=np.array([t for t, _, _ in out]), **store)
    return out


# ---------------- metrics ----------------
_lpips_net = None


def _lpips():
    global _lpips_net
    if _lpips_net is None:
        import lpips
        import torch
        _lpips_net = lpips.LPIPS(net="alex").cuda().eval()
        for p in _lpips_net.parameters():
            p.requires_grad_(False)
    return _lpips_net


def psnr(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return 10 * np.log10(255.0 ** 2 / max(np.mean(d * d), 1e-12))


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return structural_similarity(a, b, channel_axis=2, data_range=255)


def lpips_d(a, b, tile=1024):
    """a,b BGR uint8；大图分块平均。"""
    import torch
    net = _lpips()
    H, W = a.shape[:2]
    vals, ws = [], []
    with torch.no_grad():
        for y in range(0, H, tile):
            for x in range(0, W, tile):
                ha, wa = min(tile, H - y), min(tile, W - x)
                if ha < 64 or wa < 64:
                    continue
                pa = a[y:y + ha, x:x + wa][:, :, ::-1].astype(np.float32) / 127.5 - 1
                pb = b[y:y + ha, x:x + wa][:, :, ::-1].astype(np.float32) / 127.5 - 1
                ta = torch.from_numpy(np.ascontiguousarray(pa.transpose(2, 0, 1)))[None].cuda()
                tb = torch.from_numpy(np.ascontiguousarray(pb.transpose(2, 0, 1)))[None].cuda()
                vals.append(net(ta, tb).item())
                ws.append(ha * wa)
    return float(np.average(vals, weights=ws))


def niqe(a):
    from basicsr.metrics.niqe import calculate_niqe
    return float(calculate_niqe(a, crop_border=0, input_order="HWC", convert_to="y"))


def evaluate(fn, crops, with_lpips=True, with_niqe=False, base=None):
    """对每个 crop 应用 fn，返回相对 LQ 的指标增量（均值）。"""
    rows = []
    t0 = time.perf_counter()
    npx = 0
    for tag, lq, gt in crops:
        out = fn(lq) if fn is not None else lq
        npx += lq.shape[0] * lq.shape[1]
        rows.append((tag, out, gt, lq))
    dt = (time.perf_counter() - t0) / max(npx, 1) * 12e6 * 1000  # ms per 12MP

    def agg(imgs):
        p = np.mean([psnr(o, g) for _, o, g, _ in imgs])
        s = np.mean([ssim(o, g) for _, o, g, _ in imgs])
        l = np.mean([lpips_d(o, g) for _, o, g, _ in imgs]) if with_lpips else 0.0
        n = np.mean([niqe(o) for _, o, g, _ in imgs]) if with_niqe else 0.0
        return dict(psnr=p, ssim=s, lpips=l, niqe=n)

    cur = agg(rows)
    if base is None:
        base = agg([(t, l, g, l) for t, _, g, l in rows])
    return dict(
        d_psnr=cur["psnr"] - base["psnr"],
        d_ssim=cur["ssim"] - base["ssim"],
        d_lpips=cur["lpips"] - base["lpips"],
        d_niqe=cur["niqe"] - base["niqe"],
        cost_ms=dt,
        abs=cur,
    ), base


def per_case(fn, crops):
    """逐 crop 的 PSNR/SSIM 增量，用于检查是否有个案倒退。"""
    r = []
    for tag, lq, gt in crops:
        o = fn(lq) if fn is not None else lq
        r.append((tag, psnr(o, gt) - psnr(lq, gt), ssim(o, gt) - ssim(lq, gt)))
    return r

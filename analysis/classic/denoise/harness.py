"""评测工具：PSNR/SSIM/LPIPS/NIQE，全部相对未处理 LQ 计算增量。"""
import os, sys, time
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
VAL = '/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集/'
sys.path.insert(0, '/home/ubuntu/workspace/contest/CSIG-2026/BasicSR')

_PAIRS = [('case1_lq.jpg', 'case1_gt.png'),
          ('case2_lq.jpg', 'case2_gt.png'),
          ('case3_lq.jpg', 'case3_gt.jpg')]


def load_bgr(path):
    im = Image.open(path).convert('RGB')
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def center_crop(img, s):
    h, w = img.shape[:2]
    y, x = (h - s) // 2, (w - s) // 2
    return img[y:y + s, x:x + s].copy()


def load_pairs(crop=1024, ncrop=1):
    """返回 [(lq, gt), ...]；crop=None 表示全图。ncrop>1 时每图取多个 crop。"""
    out = []
    for lqn, gtn in _PAIRS:
        lq, gt = load_bgr(VAL + lqn), load_bgr(VAL + gtn)
        assert lq.shape == gt.shape, (lq.shape, gt.shape)
        if crop is None:
            out.append((lq, gt))
            continue
        h, w = lq.shape[:2]
        if ncrop == 1:
            cs = [((h - crop) // 2, (w - crop) // 2)]
        else:
            rng = np.random.RandomState(0)
            cs = [((h - crop) // 2, (w - crop) // 2)]
            for _ in range(ncrop - 1):
                cs.append((rng.randint(0, h - crop), rng.randint(0, w - crop)))
        for (y, x) in cs:
            out.append((lq[y:y + crop, x:x + crop].copy(),
                        gt[y:y + crop, x:x + crop].copy()))
    return out


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0


def ssim(a, b):
    """标准 gaussian-window SSIM，各通道平均（与常见评测一致）。"""
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    vals = []
    for c in range(a.shape[2]):
        x, y = a[..., c], b[..., c]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sxx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x ** 2
        syy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y ** 2
        sxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        m = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / \
            ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
        vals.append(m[5:-5, 5:-5].mean())
    return float(np.mean(vals))


_lpips_net = None


def lpips_dist(a, b):
    global _lpips_net
    import torch, lpips as L
    if _lpips_net is None:
        _lpips_net = L.LPIPS(net='alex').cuda().eval()
    with torch.no_grad():
        t = lambda im: (torch.from_numpy(
            cv2.cvtColor(im, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        ).float().cuda() / 127.5 - 1.0)
        return float(_lpips_net(t(a), t(b)).item())


_niqe_fn = None


def niqe(img):
    global _niqe_fn
    if _niqe_fn is None:
        from basicsr.metrics.niqe import calculate_niqe
        _niqe_fn = calculate_niqe
    return float(_niqe_fn(img, crop_border=0, input_order='HWC', convert_to='y'))

"""方向探针：模糊 vs 锐化，哪边提升 PSNR/SSIM。"""
import numpy as np
import cv2
import harness


def gblur(s):
    def f(x):
        k = int(2 * round(3 * s) + 1)
        return cv2.GaussianBlur(x, (k, k), s)
    return f


def usm(s, amt):
    def f(x):
        k = int(2 * round(3 * s) + 1)
        xf = x.astype(np.float32)
        bl = cv2.GaussianBlur(xf, (k, k), s)
        return np.clip(xf + amt * (xf - bl), 0, 255).astype(np.uint8)
    return f


print("=== 高斯模糊 ===")
for s in [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]:
    a, _ = harness.evaluate(gblur(s))
    print(f"blur sigma={s:.1f}  dPSNR={a['psnr']:+.4f}  dSSIM={a['ssim']:+.5f}")

print("=== USM 温和档 ===")
for s in [1.0, 2.0, 3.0]:
    for amt in [0.1, 0.2, 0.35, 0.5, 0.8]:
        a, _ = harness.evaluate(usm(s, amt))
        print(f"usm sigma={s:.1f} amt={amt:.2f}  dPSNR={a['psnr']:+.4f}  dSSIM={a['ssim']:+.5f}")

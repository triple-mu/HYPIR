"""TAESD 与 SD VAE 的 latent 兼容性 + 各段画质代价。

换解码器只改输出图像，风险低；换编码器等于改 UNet 的输入分布，必须先证明
两者的 latent 空间能对上——这个脚本就是量那件事。

runner 里流动的是**未缩放**的 SD latent：
    z = sample_latent(vae.encode_moments(x))      # 未缩放
    z = z*0.18215 -> UNet -> /0.18215
    out = vae.decode(z)
而 TAESD（diffusers AutoencoderTiny）的 latent 是另一套尺度，先拟合线性换算。

    python csig/taesd_probe.py --model-dir <包含 hypir_weights.pth 的 model_dir>
"""
import argparse
import glob
import importlib.util
import os
import sys

import cv2
import numpy as np
import torch

CSIG = os.environ.get("CSIG", "/root/.cache/huggingface/csig")
VAL = os.path.join(CSIG, "data/csig_bench/赛题二/验证集")


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10 * np.log10(1.0 / max(mse, 1e-20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(CSIG, "export/model_dir"))
    ap.add_argument("--crops", type=int, default=12)
    a = ap.parse_args()

    sys.path.insert(0, a.model_dir)
    spec = importlib.util.spec_from_file_location("pkgrunner", os.path.join(a.model_dir, "runner.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["pkgrunner"] = m
    spec.loader.exec_module(m)
    import csig_ops as ops

    r = m.Runner(a.model_dir)
    dev, dt = r.device, r.weight_dtype

    from diffusers import AutoencoderTiny
    tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dt).to(dev).eval()

    # 取真实赛题图的 crop，别用随机噪声——latent 统计对内容敏感
    crops = []
    for gtp in sorted(glob.glob(os.path.join(VAL, "*_gt.*")))[:3]:
        img = cv2.imread(gtp)[:, :, ::-1].copy()
        H, W = img.shape[:2]
        rng = np.random.RandomState(0)
        for _ in range(a.crops):
            y, x = rng.randint(0, H - 512), rng.randint(0, W - 512)
            crops.append(img[y:y + 512, x:x + 512])
    X = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().to(dev) / 255.
    X = X * 2 - 1                      # runner 的 VAE 吃 [-1,1]
    print(f"样本: {X.shape[0]} 个 512x512 真实赛题 crop\n")

    with torch.no_grad():
        xin = X.to(dt)
        if r._mf:
            xin = xin.contiguous(memory_format=r._mf)
        z_sd = ops.sample_latent(r.vae.encode_moments(xin), torch.zeros(
            X.shape[0], 4, 64, 64, device=dev, dtype=dt))     # noise=0 取均值，去掉随机性
        z_tae = tae.encode(xin).latents

        # 线性拟合 z_tae ~ a*z_sd + b
        zs, zt = z_sd.float().flatten(), z_tae.float().flatten()
        A = float(((zs - zs.mean()) * (zt - zt.mean())).mean() / ((zs - zs.mean()) ** 2).mean())
        B = float(zt.mean() - A * zs.mean())
        corr = float(((zs - zs.mean()) * (zt - zt.mean())).mean()
                     / (zs.std() * zt.std()))
        print("=== latent 兼容性 ===")
        print(f"  z_sd  (未缩放)  均值 {zs.mean():+.4f}  标准差 {zs.std():.4f}")
        print(f"  z_tae           均值 {zt.mean():+.4f}  标准差 {zt.std():.4f}")
        print(f"  拟合 z_tae = {A:.5f} * z_sd + {B:+.5f}   相关系数 {corr:.4f}")
        print(f"  （SD 的 scaling_factor 是 0.18215，对照看拟合出的斜率）\n")

        # 各条重建路径的 PSNR（对 GT 图）
        gt = ((X + 1) / 2).clamp(0, 1)
        rec = {}
        rec["SD enc + SD dec"] = r.vae.decode(z_sd).float()
        rec["SD enc + TAESD dec"] = tae.decode((z_sd * A + B).to(dt)).sample.float()
        rec["TAESD enc + SD dec"] = r.vae.decode(((z_tae - B) / A).to(dt)).float()
        rec["TAESD enc + TAESD dec"] = tae.decode(z_tae).sample.float()
        print("=== 自编码往返重建（不经 UNet），对 GT 的 PSNR ===")
        base = None
        for k, v in rec.items():
            img = ((v + 1) / 2).clamp(0, 1)
            p_gt = psnr(img, gt)
            if base is None:
                base = img
                print(f"  {k:<24} {p_gt:6.2f} dB   (基准)")
            else:
                print(f"  {k:<24} {p_gt:6.2f} dB   与基准 {psnr(img, base):6.2f} dB")


if __name__ == "__main__":
    main()

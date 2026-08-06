"""三项 loss 的目标端探针：TAESD 解码器的可达集，以及训练目标里够不到的部分。

回答四个问题：
  A. 往返天花板：dec(enc(gt)) 离 gt 有多远（L2 / LPIPS / 感知分三种口径）
  B. 「预补偿」余量：对固定 gt 直接梯度下降优化 z，dec(z) 能比 dec(enc(gt)) 好多少？
     余量大 -> 用像素 GT 当 L2/LPIPS 目标有意义；余量小 -> 是在罚 G 够不到的目标
  C. USM：解码器能不能表达 USMSharp 后的高频；训练目标 usm(gt) 相对评估参考 gt 的天花板损失
  D. 频带：推理端 wavelet_reconstruction 把输出低频整体换成 LQ 低频，
     训练侧三项 loss 都罚在未替换的原始输出上 —— 有多少 L2 是罚在评估时被丢掉的频带里

    python csig/taesd_headroom.py --part all
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from HYPIR.dataset.utils import USMSharp
from HYPIR.utils.common import wavelet_decomposition, wavelet_reconstruction

REALSR = "/home/ubuntu/workspace/contest/CSIG-2026/OSEDiff/preset/datasets/benchmark_realsr"


def psnr(a, b):
    """a/b in [0,1]"""
    return 10 * torch.log10(1.0 / ((a - b) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-20))


def load_triples(n, device):
    """RealSR：LR 128 / HR 512 / OSEDiff 输出 512，与 DIV2K_V2_val 的几何一致（x4）。"""
    from PIL import Image
    names = sorted(os.listdir(os.path.join(REALSR, "test_HR")))[:n]
    def rd(sub, nm):
        a = np.asarray(Image.open(os.path.join(REALSR, sub, nm)).convert("RGB"), dtype=np.float32) / 255.
        return torch.from_numpy(a).permute(2, 0, 1)
    hr = torch.stack([rd("test_HR", nm) for nm in names]).to(device)
    lr = torch.stack([rd("test_LR", nm) for nm in names]).to(device)
    out = torch.stack([rd("results_osediff", nm) for nm in names]).to(device)
    return names, hr, lr, out


def build_tae(device, dtype=torch.float32):
    from diffusers import AutoencoderTiny
    t = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype).to(device)
    return t.eval().requires_grad_(False)


def rt(tae, x01, dtype=None):
    """x01 in [0,1] -> TAESD 往返 -> [0,1]。与 HYPIR/utils/taesd.py 的往返逐位等价
    （wrapper 的仿射 A/B 在 enc 后除、dec 前乘，往返里精确抵消）。"""
    dtype = dtype or next(tae.parameters()).dtype
    x = (x01 * 2 - 1).to(dtype)
    with torch.no_grad():
        z = tae.encode(x).latents
        y = tae.decode(z).sample.float()
    return ((y + 1) / 2).clamp(0, 1)


# ---------------------------------------------------------------- B: 预补偿余量
def latent_opt(tae, target01, steps, lr, lam_l2, lam_lpips, net_lpips=None, init=None):
    """对固定 target 直接优化 z，最小化 lam_l2*MSE + lam_lpips*LPIPS（口径同训练）。
    返回 (最优 dec(z) in [0,1], 每 100 步的 (psnr, lpips) 轨迹)。"""
    tgt = (target01 * 2 - 1)
    with torch.no_grad():
        z0 = tae.encode(tgt).latents if init is None else init
    z = z0.clone().float().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    traj = []
    for i in range(steps + 1):
        y = tae.decode(z).sample.float()
        loss = 0.
        if lam_l2:
            loss = loss + lam_l2 * F.mse_loss(y, tgt)
        if lam_lpips:
            loss = loss + lam_lpips * net_lpips(y.clamp(-1, 1), tgt).mean()
        if i % 100 == 0:
            with torch.no_grad():
                y01 = ((y + 1) / 2).clamp(0, 1)
                p = psnr(y01, target01).mean().item()
                l = net_lpips(y.clamp(-1, 1), tgt).mean().item() if net_lpips is not None else float("nan")
                traj.append((i, p, l))
        if i == steps:
            break
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        y01 = ((tae.decode(z).sample.float() + 1) / 2).clamp(0, 1)
    return y01, traj


def band_split(err):
    """把误差图按推理端同一套 wavelet（levels=5）拆成 高频/低频，返回 (LF 占 MSE 比例)。"""
    hf, lf = wavelet_decomposition(err, levels=5)
    e = (err ** 2).mean()
    return (lf ** 2).mean().item() / e.item(), (hf ** 2).mean().item() / e.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.02)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    names, hr, lr128, osed = load_triples(a.n, dev)
    tae = build_tae(dev)
    import lpips as lpips_pkg
    net_lpips = lpips_pkg.LPIPS(net="vgg").to(dev).eval().requires_grad_(False)

    def LP(a01, b01):
        return net_lpips(a01 * 2 - 1, b01 * 2 - 1).mean().item()

    usm = USMSharp().to(dev)
    hr_usm = usm(hr.clone())          # 训练里的 GT：batch_transform.py:160-162

    print(f"样本 {len(names)} 张 RealSR 512x512 真实照片（LR 128 / HR 512，x4，与赛题几何一致）\n")

    if a.part in ("all", "A"):
        print("=== A. TAESD 往返天花板（假设 G 完美，只剩解码器误差）===")
        for tag, tgt in (("gt", hr), ("usm(gt)", hr_usm)):
            r = rt(tae, tgt)
            print(f"  dec(enc({tag})) vs {tag:8s}: PSNR {psnr(r, tgt).mean():6.2f} dB  LPIPS {LP(r, tgt):.4f}")
        r_bf = rt(build_tae(dev, torch.bfloat16), hr)
        print(f"  （bf16 往返，训练实际口径）      : PSNR {psnr(r_bf, hr).mean():6.2f} dB")
        print(f"  usm(gt) vs gt（训练目标 vs 评估参考）: PSNR {psnr(hr_usm, hr).mean():6.2f} dB  LPIPS {LP(hr_usm, hr):.4f}")
        print()

    if a.part in ("all", "B"):
        print("=== B. 预补偿余量：直接对 z 做梯度下降，看能比 enc(gt) 好多少 ===")
        for tag, tgt in (("gt", hr), ("usm(gt)", hr_usm)):
            base = rt(tae, tgt)
            p0, l0 = psnr(base, tgt).mean().item(), LP(base, tgt)
            y, traj = latent_opt(tae, tgt, a.steps, a.lr, 1.0, 0.0, net_lpips)
            p1 = psnr(y, tgt).mean().item()
            print(f"  目标={tag:8s} 纯 L2 目标: enc(gt) {p0:6.2f} dB -> 优化后 {p1:6.2f} dB  (余量 {p1-p0:+.2f} dB)")
            print(f"        轨迹 " + "  ".join(f"{i}:{p:.2f}dB" for i, p, _ in traj))
            y2, traj2 = latent_opt(tae, tgt, a.steps, a.lr, 1.0, 5.0, net_lpips)
            p2, l2v = psnr(y2, tgt).mean().item(), LP(y2, tgt)
            print(f"        训练口径 (1*L2+5*LPIPS): PSNR {p0:.2f}->{p2:.2f} dB   LPIPS {l0:.4f}->{l2v:.4f}")
            print(f"        轨迹 " + "  ".join(f"{i}:({p:.2f}dB,{l:.4f})" for i, p, l in traj2))
        print()

    if a.part in ("all", "C"):
        print("=== C. USM 高频的可表达性 ===")
        # USM 残差本身的能量与频带
        d = hr_usm - hr
        lf_f, hf_f = band_split(d)
        print(f"  USM 残差 RMS {d.pow(2).mean().sqrt():.5f}（{d.abs().max():.3f} 峰值），"
              f"按推理端 wavelet 拆：低频 {lf_f*100:.1f}% / 高频 {hf_f*100:.1f}%")
        # 解码器能表达多少 USM 残差：优化 z 逼近 usm(gt)，看残差里 USM 部分被复现了多少
        y, _ = latent_opt(tae, hr_usm, a.steps, a.lr, 1.0, 0.0, net_lpips)
        rec = y - rt(tae, hr)           # 优化解相对「往返 gt」的增量
        num = (rec * d).mean().item()
        den = (d * d).mean().item()
        print(f"  优化解相对 dec(enc(gt)) 的增量 与 USM 残差 的投影系数: {num/den:.3f} "
              f"(1.0=完全表达, 0=完全表达不出)")
        print(f"  最优 dec(z) vs usm(gt): {psnr(y, hr_usm).mean():.2f} dB;  同一张图 vs gt: {psnr(y, hr).mean():.2f} dB")
        print()

    if a.part in ("all", "D"):
        print("=== D. 频带错配：推理端 wavelet 把低频整体换成 LQ 的 ===")
        ref = F.interpolate(lr128, scale_factor=4, mode="bicubic").clamp(0, 1)   # enhancer/base.py:88,102
        for tag, x in (("OSEDiff 输出(真实 SR 模型)", osed), ("dec(enc(gt))(纯解码器误差)", rt(tae, hr))):
            e = x - hr
            lf_f, hf_f = band_split(e)
            xw = wavelet_reconstruction(x, ref)
            print(f"  {tag}")
            print(f"     训练侧 loss 看到的误差: MSE 里 低频 {lf_f*100:5.1f}% / 高频 {hf_f*100:5.1f}%")
            print(f"     PSNR 未过 wavelet {psnr(x, hr).mean():6.2f} dB -> 过 wavelet {psnr(xw, hr).mean():6.2f} dB")
            print(f"     LPIPS 未过 {LP(x, hr):.4f} -> 过 {LP(xw, hr):.4f}")
        # 低频被替换后，G 在低频上做任何事都不影响评估：量一下 wavelet 前后输出差多少
        xw = wavelet_reconstruction(rt(tae, hr), ref)
        print(f"  wavelet 前后输出本身的差异（=被丢弃的那部分 G 输出）: "
              f"{psnr(xw, rt(tae, hr)).mean():.2f} dB")
        print()

    if a.part in ("all", "H"):
        print("=== H. 只用 L2 精修 z（不带 LPIPS，便宜 10 倍）能不能同样把 LPIPS 打下来 ===")
        tgt = (hr * 2 - 1)
        with torch.no_grad():
            z0 = tae.encode(tgt).latents
        for lr in (0.02, 0.05):
            z = z0.clone().float().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=lr)
            log = []
            for i in range(101):
                y = tae.decode(z).sample.float()
                if i in (0, 5, 10, 20, 50, 100):
                    with torch.no_grad():
                        y01 = ((y + 1) / 2).clamp(0, 1)
                        log.append((i, psnr(y01, hr).mean().item(), LP(y01, hr)))
                if i == 100:
                    break
                loss = F.mse_loss(y, tgt)
                opt.zero_grad(); loss.backward(); opt.step()
            print(f"  纯 L2, lr={lr}: " + "  ".join(f"{i}步 {p:.2f}dB/{l:.4f}" for i, p, l in log))
        print()

    if a.part in ("all", "G"):
        print("=== G. 余量的收敛速度与解的 z 统计（判断能否在线做）===")
        tgt = (hr * 2 - 1)
        with torch.no_grad():
            z0 = tae.encode(tgt).latents
        for lr in (0.02, 0.05, 0.1):
            z = z0.clone().float().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=lr)
            log = []
            for i in range(101):
                y = tae.decode(z).sample.float()
                if i in (0, 5, 10, 20, 30, 50, 100):
                    with torch.no_grad():
                        y01 = ((y + 1) / 2).clamp(0, 1)
                        log.append((i, psnr(y01, hr).mean().item(), LP(y01, hr)))
                if i == 100:
                    break
                loss = F.mse_loss(y, tgt) + 5.0 * net_lpips(y.clamp(-1, 1), tgt).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            print(f"  lr={lr}: " + "  ".join(f"{i}步 {p:.2f}dB/{l:.4f}" for i, p, l in log))
            with torch.no_grad():
                d = (z - z0)
                print(f"        z0 std {z0.float().std():.3f}  |z*-z0| RMS {d.pow(2).mean().sqrt():.3f} "
                      f"({d.pow(2).mean().sqrt()/z0.float().std()*100:.1f}% of z std)  z* std {z.std():.3f}")
        print()

    if a.part in ("all", "F"):
        print("=== F. 三项 loss 对 G 施加的「力」：dL/dx 的逐像素 RMS（同一个 x 上比较）===")
        x = (osed * 2 - 1).clone().requires_grad_(True)   # 用真实 SR 模型输出当代表点
        g = (hr * 2 - 1)
        g_rt = (rt(tae, hr) * 2 - 1)
        for tag, tgt, lam_l2, lam_lp in (
            ("L2 (target=gt)", g, 1.0, 0.0),
            ("L2 (target=dec(enc(gt)))", g_rt, 1.0, 0.0),
            ("5*LPIPS (target=gt)", g, 0.0, 5.0),
            ("5*LPIPS (target=dec(enc(gt)))", g_rt, 0.0, 5.0),
        ):
            if x.grad is not None:
                x.grad = None
            loss = 0.
            if lam_l2:
                loss = loss + lam_l2 * F.mse_loss(x, tgt)
            if lam_lp:
                loss = loss + lam_lp * net_lpips(x, tgt).mean()
            loss.backward()
            print(f"  {tag:32s} loss {float(loss):8.4f}   |dL/dx| RMS {x.grad.pow(2).mean().sqrt():.3e}")
        print()

    if a.part in ("all", "I"):
        print("=== I. 赛题口径：各种 d_real / 训练目标候选值多少感知分 ===")
        import pyiqa
        fr = pyiqa.create_metric("topiq_fr", device=dev)
        nr = pyiqa.create_metric("maniqa", device=dev)
        def score(x):
            vf, vn = [], []
            with torch.no_grad():
                for i in range(len(x)):
                    xi = x[i:i + 1].clamp(0, 1)
                    vf.append(float(fr(xi, hr[i:i + 1])))
                    vn.append(float(nr(xi)))
                    torch.cuda.empty_cache()
            return float(np.mean(vf)), float(np.mean(vn))
        ref = F.interpolate(lr128, scale_factor=4, mode="bicubic").clamp(0, 1)
        z20, _ = latent_opt(tae, hr, 20, 0.02, 1.0, 5.0, net_lpips)
        z20u, _ = latent_opt(tae, hr_usm, 20, 0.02, 1.0, 5.0, net_lpips)
        cases = {
            "gt 本身（绝对天花板）": hr,
            "usm(gt)（训练里的 GT / L2+LPIPS 目标）": hr_usm,
            "dec(enc(gt))": rt(tae, hr),
            "dec(enc(usm(gt)))  <- 现在 drt 的 d_real": rt(tae, hr_usm),
            "dec(z20)  z 精修 20 步, 目标 gt": z20,
            "dec(z20)  z 精修 20 步, 目标 usm(gt)": z20u,
            "wavelet(dec(enc(gt)), ref)": wavelet_reconstruction(rt(tae, hr), ref),
            "bicubic 上采样 LQ（下界参照）": ref,
            "OSEDiff 输出（真实模型参照）": osed,
        }
        for k, v in cases.items():
            f_, n_ = score(v)
            print(f"  {k:42s} TOPIQ-FR {f_:.4f}  MANIQA {n_:.4f}  感知分 {13.674*f_+4.477*n_-5.731:7.4f}")
        print()

    if a.part in ("all", "E"):
        print("=== E. 换成赛题口径：各个「够不到的目标」值多少感知分 ===")
        import pyiqa
        fr = pyiqa.create_metric("topiq_fr", device=dev)
        nr = pyiqa.create_metric("maniqa", device=dev)
        def score(x):
            with torch.no_grad():
                v_fr = float(fr(x.clamp(0, 1), hr).mean())
                v_nr = float(nr(x.clamp(0, 1)).mean())
            return v_fr, v_nr, 13.674 * v_fr + 4.477 * v_nr - 5.731
        ref = F.interpolate(lr128, scale_factor=4, mode="bicubic").clamp(0, 1)
        cases = {
            "gt 本身（绝对天花板）": hr,
            "usm(gt)（训练目标）": hr_usm,
            "dec(enc(gt))（G 完美时可达）": rt(tae, hr),
            "dec(enc(usm(gt)))（G 完美地学到训练目标）": rt(tae, hr_usm),
            "wavelet(dec(enc(gt)), ref)（推理端实际输出）": wavelet_reconstruction(rt(tae, hr), ref),
            "wavelet(dec(enc(usm(gt))), ref)": wavelet_reconstruction(rt(tae, hr_usm), ref),
            "OSEDiff 输出（对照，真实模型）": osed,
        }
        for k, v in cases.items():
            f_, n_, p_ = score(v)
            print(f"  {k:44s} TOPIQ-FR {f_:.4f}  MANIQA {n_:.4f}  感知分 {p_:7.4f}")
        print()


if __name__ == "__main__":
    main()

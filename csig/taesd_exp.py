"""TAESD 换 VAE 的完整实验：4 种组合 x (速度 | 画质)。

速度依赖 GPU 架构（必须在 V100 上量，评测机是 sm70）；画质与 GPU 无关，
所以两边分开跑，最后用同一个公式合成分数：

    综合分 = 感知分 x 时延^-0.204

组合（编码器 x 解码器）：SD/SD（现状）、SD/TAESD、TAESD/SD、TAESD/TAESD。
四种都要测——latent 探针显示 TAESD 的编解码是共同适配的，混搭比配对更差
（往返 PSNR：SD+TAE 28.87，TAE+SD 24.77，TAE+TAE 29.94）。

    python csig/taesd_exp.py --mode speed    # V100 上跑
    python csig/taesd_exp.py --mode quality  # 有评估数据的机器上跑
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

CSIG = os.environ.get("CSIG", "/root/.cache/huggingface/csig")
VAL = os.path.join(CSIG, "data/csig_bench/赛题二/验证集")
PROMPT_UNUSED = ""      # runner 的 prompt 在 __init__ 已烧进去

CONFIGS = ["SD/SD", "SD/TAESD", "TAESD/SD", "TAESD/TAESD"]


def load_runner(model_dir):
    sys.path.insert(0, model_dir)
    spec = importlib.util.spec_from_file_location("pkgrunner", os.path.join(model_dir, "runner.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["pkgrunner"] = m
    spec.loader.exec_module(m)
    return m, m.Runner(model_dir)


def build_variants(r, ops, dt):
    """返回 {名字: (encode_fn, decode_fn)}，两端统一用「未缩放 SD latent」做接口。"""
    from diffusers import AutoencoderTiny
    tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dt).to(r.device).eval()
    # 探针实测的换算：z_tae = A*z_sd + B
    A, B = 0.16668, 0.01702

    def enc_sd(x):
        return ops.sample_latent(r.vae.encode_moments(x), r._tile_noise)

    def enc_tae(x):
        return (tae.encode(x).latents - B) / A

    def dec_sd(z):
        return r.vae.decode(z)

    def dec_tae(z):
        return tae.decode((z * A + B).to(dt)).sample

    return tae, {"SD/SD": (enc_sd, dec_sd), "SD/TAESD": (enc_sd, dec_tae),
                 "TAESD/SD": (enc_tae, dec_sd), "TAESD/TAESD": (enc_tae, dec_tae)}


def patch(r, enc, dec, dt):
    def _tile(x):
        z = enc(x)
        z = r._forward_generator(z.to(dt), r._text_embed)
        return dec(z.to(dt)).float()
    r._infer_tile = _tile


def run_speed(a):
    m, r = load_runner(a.model_dir)
    import csig_ops as ops
    dt = r.weight_dtype
    tae, variants = build_variants(r, ops, dt)
    from torch.nn.attention import sdpa_kernel, SDPBackend

    x = torch.zeros(1, 3, 512, 512, dtype=dt, device=r.device)
    if r._mf:
        x = x.contiguous(memory_format=r._mf)

    def bench(fn, n=50):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1000

    out = {}
    print("GPU:", torch.cuda.get_device_name(0), "| 后端:", ops.BACKEND, "\n")
    print("%-14s %10s %10s %10s %10s" % ("组合", "encode", "UNet", "decode", "合计"))
    print("-" * 58)
    with torch.no_grad(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        z_ref = variants["SD/SD"][0](x)
        z1 = r._forward_generator(z_ref.to(dt), r._text_embed)
        t_unet = bench(lambda: r._forward_generator(z_ref.to(dt), r._text_embed))
        for name in CONFIGS:
            enc, dec = variants[name]
            t_e = bench(lambda: enc(x))
            t_d = bench(lambda: dec(z1.to(dt)))
            patch(r, enc, dec, dt)
            t_all = bench(lambda: r._infer_tile(x))
            out[name] = dict(encode=t_e, unet=t_unet, decode=t_d, total=t_all)
            print("%-14s %9.2f %10.2f %10.2f %10.2f" % (name, t_e, t_unet, t_d, t_all))
    base = out["SD/SD"]["total"]
    print("\n%-14s %10s %12s" % ("组合", "相对现状", "速度带来的分数"))
    print("-" * 40)
    for name in CONFIGS:
        sp = base / out[name]["total"]
        out[name]["speedup"] = sp
        out[name]["score_gain"] = sp ** 0.204 - 1
        print("%-14s %9.2fx %11.1f%%" % (name, sp, 100 * out[name]["score_gain"]))
    json.dump(out, open(a.out, "w"), indent=1)
    print("\n已存:", a.out)


def run_quality(a):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import iqa
    m, r = load_runner(a.model_dir)
    import csig_ops as ops
    dt = r.weight_dtype
    tae, variants = build_variants(r, ops, dt)

    def load(p):
        return torch.from_numpy(cv2.imread(p)[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda() / 255.

    sets = {}
    pairs = []
    for lqp in sorted(glob.glob(os.path.join(VAL, "*_lq.jpg"))):
        b = os.path.basename(lqp).replace("_lq.jpg", "")
        g = glob.glob(os.path.join(VAL, b + "_gt.*"))
        if g:
            pairs.append((lqp, g[0]))
    sets["真实验证对"] = pairs

    p0 = os.path.join(CSIG, "data/eval_p0")
    if os.path.isdir(p0):
        meta = json.load(open(os.path.join(p0, "meta.json")))["items"][:a.p0]
        sets["P0(合成,%d张)" % len(meta)] = [(os.path.join(p0, "lq", it["name"]),
                                             os.path.join(p0, "gt", it["name"])) for it in meta]

    res = {}
    for sname, items in sets.items():
        print(f"\n=== {sname} ===")
        print("%-14s %8s %8s %9s %10s %9s" % ("组合", "FR", "NR", "p_out", "保留增益", "vs SD/SD"))
        print("-" * 62)
        ref_imgs = None
        for name in CONFIGS:
            enc, dec = variants[name]
            patch(r, enc, dec, dt)
            outs, fr, nr, po, pl, pg = [], [], [], [], [], []
            for lqp, gtp in items:
                lq, gt = load(lqp), load(gtp)
                with torch.no_grad():
                    o = r.enhance(lq * 2 - 1)
                o = ((o.float() + 1) / 2).clamp(0, 1)
                outs.append(o)
                f, n, p = iqa.score(o * 2 - 1, gt * 2 - 1)
                _, _, l = iqa.score(lq * 2 - 1, gt * 2 - 1)
                _, _, g = iqa.score(gt * 2 - 1, gt * 2 - 1)
                fr.append(f); nr.append(n); po.append(p); pl.append(l); pg.append(g)
            ret = (np.mean(po) - np.mean(pl)) / max(np.mean(pg) - np.mean(pl), 1e-9)
            if ref_imgs is None:
                ref_imgs = outs
                dpsnr = "—"
            else:
                v = np.mean([10 * np.log10(1.0 / max(float(((x - y) ** 2).mean()), 1e-20))
                             for x, y in zip(outs, ref_imgs)])
                dpsnr = "%.2f dB" % v
            res.setdefault(sname, {})[name] = dict(fr=float(np.mean(fr)), nr=float(np.mean(nr)),
                                                   p_out=float(np.mean(po)), retained=float(ret))
            print("%-14s %8.4f %8.4f %9.3f %9.1f%% %10s"
                  % (name, np.mean(fr), np.mean(nr), np.mean(po), 100 * ret, dpsnr))
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("\n已存:", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["speed", "quality"], required=True)
    ap.add_argument("--model-dir", default=os.path.join(CSIG, "export/model_dir"))
    ap.add_argument("--p0", type=int, default=40, help="P0 集取前 N 张")
    ap.add_argument("--out", default="taesd_exp.json")
    a = ap.parse_args()
    (run_speed if a.mode == "speed" else run_quality)(a)

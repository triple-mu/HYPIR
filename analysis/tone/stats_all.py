"""逐张统计：亮度/饱和度/梯度/暗部亮部/拉普拉斯方差/径向谱 + JPEG 段解析。"""
import os, sys, json, struct
import numpy as np
from PIL import Image
import cv2

TEST = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
VAL = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"
OUT = "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone"


def parse_jpeg(path):
    """手工扫 JPEG 段：量化表 + APP 标记 + 采样因子。非 JPEG 返回 None。"""
    d = open(path, "rb").read()
    if d[:2] != b"\xff\xd8":
        return {"format": "PNG" if d[:4] == b"\x89PNG" else "other", "filesize": len(d)}
    i, qt, apps, samp = 2, {}, [], None
    while i < len(d) - 1:
        if d[i] != 0xFF:
            i += 1
            continue
        m = d[i + 1]
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m == 0xDA or m == 0xD9:
            break
        if i + 4 > len(d):
            break
        L = struct.unpack(">H", d[i + 2:i + 4])[0]
        seg = d[i + 4:i + 2 + L]
        if m == 0xDB:  # DQT
            p = 0
            while p < len(seg):
                pq, tq = seg[p] >> 4, seg[p] & 15
                p += 1
                n = 64 * (2 if pq else 1)
                tbl = np.frombuffer(seg[p:p + n], dtype=">u2" if pq else "u1").astype(float)
                qt[tq] = tbl
                p += n
        elif m in (0xC0, 0xC1, 0xC2):  # SOF
            nc = seg[5]
            samp = [(seg[6 + 3 * k], seg[7 + 3 * k] >> 4, seg[7 + 3 * k] & 15) for k in range(nc)]
        elif 0xE0 <= m <= 0xEF:
            tag = seg[:12].split(b"\x00")[0][:12].decode("latin1", "replace")
            apps.append((f"APP{m-0xE0}", tag, L - 2))
        i += 2 + L
    r = {"format": "JPEG", "filesize": len(d), "apps": apps,
         "samp": samp, "chroma_sub": None}
    if samp and len(samp) == 3:
        r["chroma_sub"] = f"{samp[0][1]}x{samp[0][2]}"
    for k, t in qt.items():
        r[f"q{k}_dc"] = float(t[0])
        r[f"q{k}_mean"] = float(t.mean())
        r[f"q{k}_max"] = float(t.max())
    r["n_qt"] = len(qt)
    return r


def radial_psd(gray):
    """中心 2048 方块的径向功率谱，返回按 cyc/px 分箱的功率。"""
    h, w = gray.shape
    s = min(2048, h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    g = gray[y0:y0 + s, x0:x0 + s].astype(np.float64)
    g = g - g.mean()
    win = np.hanning(s)
    g *= win[:, None] * win[None, :]
    F = np.fft.rfft2(g)
    P = (F.real ** 2 + F.imag ** 2)
    fy = np.fft.fftfreq(s)[:, None]
    fx = np.fft.rfftfreq(s)[None, :]
    fr = np.sqrt(fy ** 2 + fx ** 2)
    nb = 128
    idx = np.clip((fr / 0.5 * nb).astype(int), 0, nb)
    prof = np.bincount(idx.ravel(), P.ravel(), minlength=nb + 1)[:nb]
    cnt = np.bincount(idx.ravel(), minlength=nb + 1)[:nb]
    prof = prof / np.maximum(cnt, 1)
    freq = (np.arange(nb) + 0.5) / nb * 0.5
    return freq, prof, cnt


def stats_one(path):
    r = parse_jpeg(path)
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    r["W"], r["H"] = im.size
    r["megapix"] = a.shape[0] * a.shape[1] / 1e6
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    r["lum_mean"] = float(gray.mean())
    r["lum_std"] = float(gray.std())
    for p in (1, 5, 50, 95, 99):
        r[f"lum_p{p}"] = float(np.percentile(gray, p))
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    r["sat_mean"] = float(hsv[..., 1].mean())
    r["sat_p90"] = float(np.percentile(hsv[..., 1], 90))
    # 暗部/亮部截断（任一通道触界 & 全通道触界）
    r["frac_zero_any"] = float((a == 0).any(2).mean())
    r["frac_zero_all"] = float((a == 0).all(2).mean())
    r["frac_255_any"] = float((a == 255).any(2).mean())
    r["frac_255_all"] = float((a == 255).all(2).mean())
    r["frac_lum_lt2"] = float((gray < 2).mean())
    r["frac_lum_gt253"] = float((gray > 253).mean())
    gf = gray.astype(np.float32)
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    r["grad_mean"] = float(mag.mean())
    r["grad_p99"] = float(np.percentile(mag, 99))
    lap = cv2.Laplacian(gf, cv2.CV_32F)
    r["lap_var"] = float(lap.var())
    # 归一化清晰度：拉普拉斯方差 / 局部对比度，抵消内容影响
    r["lap_var_norm"] = float(lap.var() / max(r["lum_std"] ** 2, 1e-6))
    freq, prof, _ = radial_psd(gray)
    tot = prof[1:].sum()
    def band(lo, hi):
        m = (freq >= lo) & (freq < hi)
        return float(prof[m].sum() / max(tot, 1e-12))
    r["psd_e_0125_025"] = band(0.125, 0.25)
    r["psd_e_025_0375"] = band(0.25, 0.375)
    r["psd_e_0375_05"] = band(0.375, 0.5)
    r["psd_e_gt025"] = band(0.25, 0.5)
    # 关键判据：高频/中频功率比（带限图这个比值应远小）
    r["psd_ratio_hi_mid"] = float(r["psd_e_gt025"] / max(r["psd_e_0125_025"], 1e-12))
    # 谱截止：功率降到 f=0.125 处的 1/1000 时的频率
    ref = prof[(freq >= 0.11) & (freq < 0.14)].mean()
    thr = ref / 1000.0
    below = np.where((freq > 0.14) & (prof < thr))[0]
    r["psd_f_cut1e3"] = float(freq[below[0]]) if len(below) else 0.5
    r["psd_prof"] = [float(x) for x in prof]
    return r


def main():
    items = []
    for i in range(1, 101):
        p = f"{TEST}/case{i}.jpg"
        s = stats_one(p)
        s["name"] = f"test_case{i}"
        s["split"] = "test"
        s["idx"] = i
        items.append(s)
        print(f"test {i} done", flush=True)
    for i in (1, 2, 3):
        for kind in ("lq", "gt"):
            cands = [f"{VAL}/case{i}_{kind}.png", f"{VAL}/case{i}_{kind}.jpg"]
            p = [c for c in cands if os.path.exists(c)][0]
            s = stats_one(p)
            s["name"] = f"val_case{i}_{kind}"
            s["split"] = f"val_{kind}"
            s["idx"] = i
            items.append(s)
            print(f"val {i} {kind} done", flush=True)
    with open(f"{OUT}/stats_all.json", "w") as f:
        json.dump(items, f)
    print("WROTE", f"{OUT}/stats_all.json")


if __name__ == "__main__":
    main()

"""不依赖 EXIF 的「是否走过退化管线」判别器。
纹理条件化高频比 HFR = 在最有纹理的 10% 个 256x256 tile 上，
  E[0.20<=f<0.40 cyc/px] / E[0.03<=f<0.08 cyc/px]
低频归一化抵消内容/对比度差异；高频带完全落在 LQ 的截止之外。
"""
import glob, os, json, numpy as np, cv2

def hfr(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None: return None
    g = g.astype(np.float32)
    H, W = g.shape; T = 256; s = 256
    win = np.outer(np.hanning(T), np.hanning(T)).astype(np.float32)
    f = np.fft.fftfreq(T); R = np.hypot(*np.meshgrid(f, f, indexing='ij'))
    mh = (R >= 0.20) & (R < 0.40); ml = (R >= 0.03) & (R < 0.08)
    tiles = []
    for y in range(0, H - T + 1, s):
        for x in range(0, W - T + 1, s):
            t = g[y:y+T, x:x+T]
            tiles.append((float(t.std()), y, x))
    tiles.sort(reverse=True)
    sel = tiles[:max(1, len(tiles)//10)]
    eh = el = 0.0
    for _, y, x in sel:
        t = g[y:y+T, x:x+T]
        P = np.abs(np.fft.fft2((t - t.mean()) * win))**2
        eh += P[mh].mean(); el += P[ml].mean()
    return float(eh / el)

D = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二"
res = {}
for c in (1,2,3):
    res[f"val{c}_lq"] = hfr(f"{D}/验证集/case{c}_lq.jpg")
    res[f"val{c}_gt"] = hfr(f"{D}/验证集/case{c}_gt." + ("png" if c in (1,2) else "jpg"))
for i in range(1, 101):
    res[f"test{i}"] = hfr(f"{D}/测试集/case{i}.jpg")
json.dump(res, open("/home/ubuntu/workspace/contest/CSIG-2026/verdict/hfr.json","w"), indent=0)

NAT = {1,2,3,4,5,38,64,65,66}
print("验证集:")
for c in (1,2,3):
    print(f"  case{c}  LQ={res[f'val{c}_lq']:.5f}   GT={res[f'val{c}_gt']:.5f}   GT/LQ={res[f'val{c}_gt']/res[f'val{c}_lq']:.1f}x")
nat = sorted((res[f"test{i}"], i) for i in NAT)
oth = sorted((res[f"test{i}"], i) for i in range(1,101) if i not in NAT)
print("\n9 张原生(带EXIF/MPF)，升序:"); print("  " + "  ".join(f"c{i}:{v:.5f}" for v,i in nat))
print(f"\n91 张重编码: min={oth[0][0]:.5f}(c{oth[0][1]})  p50={oth[45][0]:.5f}  p90={oth[81][0]:.5f}  max={oth[-1][0]:.5f}(c{oth[-1][1]})")
print("  最高的 8 张: " + "  ".join(f"c{i}:{v:.5f}" for v,i in oth[-8:][::-1]))
for th in (0.010,0.015,0.020,0.030,0.040):
    tp = sum(1 for v,i in nat if v>=th); fp = sum(1 for v,i in oth if v>=th)
    print(f"  阈值 {th:.3f}: 9张中命中 {tp}/9，91张误报 {fp}")

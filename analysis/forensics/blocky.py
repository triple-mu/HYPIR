"""判别器候选 B：平坦区 8x8 块边界效应。
原生相机 JPEG（华为粗量化表）在平坦区留下 8x8 网格；被退化管线糊过之后再 q95 编码，
平坦区不产生任何块边界。这是像素域特征，不依赖任何元数据。
"""
import numpy as np, cv2, json
D = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二"

def blockiness(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None: return None
    g = g.astype(np.float32); H, W = g.shape
    H -= H % 32; W -= W % 32; g = g[:H, :W]
    # 选最平坦的 30% 的 32x32 块（排除纯死黑）
    b = g.reshape(H//32, 32, W//32, 32).transpose(0,2,1,3).reshape(-1,32,32)
    sd = b.reshape(len(b),-1).std(1)
    ok = np.where((sd > 0.3) & (sd < np.quantile(sd[sd>0.3], 0.30)))[0]
    if len(ok) < 20: return None
    sub = b[ok]
    dh = np.abs(np.diff(sub, axis=2))          # (n,32,31)
    dv = np.abs(np.diff(sub, axis=1))
    ph = np.arange(31) % 8                      # 列 j 与 j+1 之间；块边界在 j%8==7
    edge_h = dh[:, :, ph == 7].mean(); in_h = dh[:, :, ph != 7].mean()
    pv = np.arange(31) % 8
    edge_v = dv[:, pv == 7, :].mean(); in_v = dv[:, pv != 7, :].mean()
    return float(edge_h/max(in_h,1e-6)), float(edge_v/max(in_v,1e-6))

res = {}
for c in (1,2,3):
    res[f"val{c}_lq"] = blockiness(f"{D}/验证集/case{c}_lq.jpg")
    res[f"val{c}_gt"] = blockiness(f"{D}/验证集/case{c}_gt." + ("png" if c in (1,2) else "jpg"))
for i in range(1,101):
    res[f"test{i}"] = blockiness(f"{D}/测试集/case{i}.jpg")
json.dump(res, open("/home/ubuntu/workspace/contest/CSIG-2026/verdict/blocky.json","w"))
NAT = {1,2,3,4,5,38,64,65,66}
sc = lambda v: max(v) if v else float('nan')
print("验证集:")
for c in (1,2,3):
    print(f"  case{c} LQ={res[f'val{c}_lq']}  GT={res[f'val{c}_gt']}")
nat = sorted((sc(res[f"test{i}"]), i) for i in NAT)
oth = sorted((sc(res[f"test{i}"]), i) for i in range(1,101) if i not in NAT)
print("\n9 原生 max(h,v): " + " ".join(f"c{i}:{v:.2f}" for v,i in nat))
print(f"91 重编: min={oth[0][0]:.2f} p50={oth[45][0]:.2f} p90={oth[81][0]:.2f} max={oth[-1][0]:.2f}(c{oth[-1][1]})")
print("91 最高 8: " + " ".join(f"c{i}:{v:.2f}" for v,i in oth[-8:][::-1]))
for th in (1.2,1.3,1.5,1.8,2.0,2.5):
    print(f"  阈值 {th}: 9张命中 {sum(1 for v,_ in nat if v>=th)}/9  91张误报 {sum(1 for v,_ in oth if v>=th)}")

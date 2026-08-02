import numpy as np, cv2, json
D = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二"
T = 256
win = np.outer(np.hanning(T), np.hanning(T)).astype(np.float32)
f = np.fft.fftfreq(T); R = np.hypot(*np.meshgrid(f, f, indexing='ij'))
NB = 64; edges = np.linspace(0, 0.5, NB + 1)
masks = [(R >= edges[i]) & (R < edges[i+1]) for i in range(NB)]

def spec(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None: return None
    g = g.astype(np.float32); H, W = g.shape
    tiles = sorted(((float(g[y:y+T,x:x+T].std()), y, x)
                    for y in range(0, H-T+1, T) for x in range(0, W-T+1, T)), reverse=True)
    acc = np.zeros(NB)
    sel = tiles[:max(1, len(tiles)//10)]
    for _, y, x in sel:
        t = g[y:y+T, x:x+T]
        P = np.abs(np.fft.fft2((t - t.mean()) * win))**2
        for i, m in enumerate(masks): acc[i] += P[m].mean()
    return (acc / len(sel)).tolist()

out = {}
for c in (1,2,3):
    out[f"val{c}_lq"] = spec(f"{D}/验证集/case{c}_lq.jpg")
    out[f"val{c}_gt"] = spec(f"{D}/验证集/case{c}_gt." + ("png" if c in (1,2) else "jpg"))
for i in range(1, 101):
    out[f"test{i}"] = spec(f"{D}/测试集/case{i}.jpg")
json.dump({"edges": edges.tolist(), "spec": out}, open("/home/ubuntu/workspace/contest/CSIG-2026/verdict/spec.json","w"))
print("ok", len(out))

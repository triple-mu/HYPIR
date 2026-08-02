"""P0 离线评估集：按测试集内容配比造 held-out 的 (LQ, GT) 配对。

为什么必须做：现有离线评估只有验证集那 3 张，全是建筑/远景城市；而测试集里
植被/花卉 26%、招牌文字 6%、器物 5%、街景 4% —— 验证集覆盖为零。
拿 3 张建筑图调出来的参数，在那 41% 的图上没有任何保证。

零重叠保证：本脚本只用 EVAL_SHARDS 里的分片，训练集构建时会显式排除它们。
"""
import io, os, re, sys, glob, json
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csig_data import hf_max, ijg_quality
from degrade import degrade

EVAL_SHARDS = ["00118", "00119"]          # 专供评估，绝不进训练
CROP = 512
SEED = 20260802                            # 固定，保证评估集可复现

# 测试集实测内容配比（逐张目视 + caption 统计）
MIX = [
    ("building",   0.38, r"\b(building|architecture|facade|skyscraper|tower|apartment|window|balcony|roof)\b"),
    ("vegetation", 0.26, r"\b(plant|flower|leaf|leaves|tree|foliage|garden|branch|petal|blossom|grass)\b"),
    ("skyline",    0.15, r"\b(skyline|cityscape|city view|downtown|aerial|panorama|horizon)\b"),
    ("signage",    0.06, r"\b(sign|signage|shop|store|storefront|billboard|text|letter)\b"),
    ("object",     0.05, r"\b(statue|sculpture|boat|ship|monument|fountain|lamp|bench)\b"),
    ("street",     0.04, r"\b(street|road|sidewalk|alley|crosswalk|pavement)\b"),
]
DROP = re.compile(r"\b(portrait|man|woman|person|people|girl|boy|face|model|couple|child|kid|selfie)\b", re.I)
TOTAL = 240


def main(src, out):
    want = {n: max(1, int(round(TOTAL * f))) for n, f, _ in MIX}
    pats = {n: re.compile(p, re.I) for n, _, p in MIX}
    got = {n: [] for n in want}
    files = [f for f in sorted(glob.glob(os.path.join(src, "*.parquet")))
             if any(s in os.path.basename(f) for s in EVAL_SHARDS)]
    assert files, "没找到评估用分片，先把 %s 下下来" % EVAL_SHARDS
    print("评估分片 %d 个: %s" % (len(files), [os.path.basename(f) for f in files]), flush=True)

    for f in files:
        t = pq.read_table(f, columns=["hr", "caption", "cogvlm_caption"])
        for row in t.to_pylist():
            if all(len(got[n]) >= want[n] for n in want):
                break
            cap = (row.get("caption") or "") + " " + (row.get("cogvlm_caption") or "")
            if DROP.search(cap):
                continue
            cat = next((n for n, _, _ in MIX if pats[n].search(cap) and len(got[n]) < want[n]), None)
            if cat is None:
                continue
            b = row["hr"]["bytes"]
            if not b:
                continue
            try:
                im = Image.open(io.BytesIO(b))
                if min(im.size) < CROP or max(im.size) < 3000:
                    continue
                if hf_max(np.asarray(im.convert("L"))) < 0.005:
                    continue
                q = getattr(im, "quantization", None)
                if q:
                    iq = ijg_quality(q[0])
                    if iq is not None and iq < 93:
                        continue
                if len(b) * 8 / (im.width * im.height) < 2.0:
                    continue
                got[cat].append(np.asarray(im.convert("RGB"))[:, :, ::-1])   # -> BGR
            except Exception:
                continue

    os.makedirs(os.path.join(out, "gt"), exist_ok=True)
    os.makedirs(os.path.join(out, "lq"), exist_ok=True)
    import cv2
    rng = np.random.default_rng(SEED)
    meta = []
    for cat, imgs in got.items():
        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            y, x = (h - CROP) // 2, (w - CROP) // 2      # 中心裁，去掉位置随机性
            gt = np.ascontiguousarray(img[y:y + CROP, x:x + CROP])
            lq = degrade(gt, rng)                         # 固定 rng 序列 -> 可复现
            name = "%s_%03d.png" % (cat, i)
            cv2.imwrite(os.path.join(out, "gt", name), gt)
            cv2.imwrite(os.path.join(out, "lq", name), lq)
            meta.append({"name": name, "cat": cat})
    json.dump({"seed": SEED, "eval_shards": EVAL_SHARDS, "items": meta},
              open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("各类产出:", {k: len(v) for k, v in got.items()}, flush=True)
    print("目标配比:", want, flush=True)
    print("合计 %d 对 -> %s" % (len(meta), out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

"""从 125 个分片里分层抽样候选（几何闸门），输出 sample.csv 供 EXIF 头探针使用。"""
import pyarrow.parquet as pq, pandas as pd, numpy as np, glob, sys
K = int(sys.argv[1]) if len(sys.argv)>1 else 80
rng = np.random.default_rng(20260802)
out=[]
for f in sorted(glob.glob("meta/metadata/*.parquet")):
    t = pq.read_table(f, columns=["url","caption","width","height","mime_type","source"]).to_pandas()
    L=t[["width","height"]].max(1); S=t[["width","height"]].min(1); ar=L/S
    c = t[(t.mime_type=="image/jpeg")&(L>=3400)&(ar>=1.28)&(ar<=1.40)].copy()
    c["shard"]=f[-11:-8]
    if len(c)>K: c = c.iloc[rng.choice(len(c), K, replace=False)]
    out.append(c)
d = pd.concat(out, ignore_index=True)
d.to_csv("sample.csv", index=False)
print(len(d), d.shard.nunique())

import pyarrow.parquet as pq, pandas as pd, numpy as np, glob, collections, re, sys
files = sorted(glob.glob("meta/metadata/*.parquet"))
res = collections.Counter(); src = collections.Counter(); srcc = collections.Counter()
tot = 0; cand = 0
cn_pat = re.compile(r"\b(china|chinese|beijing|shanghai|hangzhou|shenzhen|guangzhou|wenzhou|nanjing|suzhou|chengdu|xian|hong kong|taiwan|taipei|macau|tibet|yunnan|sichuan|zhejiang|jiangsu|guangdong|fujian|pagoda|temple)\b", re.I)
cn_hits = collections.Counter(); cn_hits_c = collections.Counter()
rows_out = []
for f in files:
    t = pq.read_table(f, columns=["url","caption","width","height","mime_type","source"]).to_pandas()
    tot += len(t)
    src.update(t.source.value_counts().to_dict())
    L = t[["width","height"]].max(1); S = t[["width","height"]].min(1)
    ar = L/S
    m = (t.mime_type=="image/jpeg") & (L>=3400) & (ar>=1.28) & (ar<=1.40)
    c = t[m]
    cand += len(c)
    srcc.update(c.source.value_counts().to_dict())
    res.update(zip(c.width, c.height))
    h = t.caption.str.contains(cn_pat, na=False)
    cn_hits.update(t.caption[h].str.lower().str.findall(cn_pat).explode().value_counts().to_dict())
    hc = c.caption.str.contains(cn_pat, na=False)
    cn_hits_c.update(c.caption[hc].str.lower().str.findall(cn_pat).explode().value_counts().to_dict())
print("total rows", tot, "candidates(no caption filter)", cand, cand/tot)
print("\n== source (全库 top30) ==")
for k,v in src.most_common(30): print(f"{v:8d} {v/tot*100:5.2f}%  {k}")
print("\n== source (候选 top30) ==")
for k,v in srcc.most_common(30): print(f"{v:8d} {v/cand*100:5.2f}%  {k}")
print("\n== 候选分辨率 top60 ==")
for (w,h),v in res.most_common(60): print(f"{v:8d}  {w}x{h}")
print("\n== caption 中国关键词命中（全库） ==")
for k,v in cn_hits.most_common(30): print(f"{v:8d} {k}")
print("\n== caption 中国关键词命中（候选） ==")
for k,v in cn_hits_c.most_common(30): print(f"{v:8d} {k}")

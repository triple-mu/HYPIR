"""Google HDR+ burst 数据集：匿名 GCS，取每个 burst 的 final.jpg（真计算摄影管线成品）。"""
import json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
API="https://storage.googleapis.com/storage/v1/b/hdrplusdata/o"
PFX="20171106/results_20171023/"
OUT=sys.argv[1]
def get(url):
    with urllib.request.urlopen(url, timeout=60) as r: return json.load(r)
bursts, tok = [], None
while True:
    u=f"{API}?prefix={PFX}&delimiter=/&maxResults=1000"+(f"&pageToken={tok}" if tok else "")
    d=get(u); bursts += [p.split("/")[-2] for p in d.get("prefixes",[])]
    tok=d.get("nextPageToken")
    if not tok: break
print(f"共 {len(bursts)} 个 burst", flush=True)
os.makedirs(OUT, exist_ok=True)
def one(b):
    dst=os.path.join(OUT, b+".jpg")
    if os.path.exists(dst) and os.path.getsize(dst)>10000: return 0
    try:
        urllib.request.urlretrieve(f"https://storage.googleapis.com/hdrplusdata/{PFX}{b}/final.jpg", dst); return 1
    except Exception: return -1
with ThreadPoolExecutor(16) as ex: r=list(ex.map(one, bursts))
print(f"新下 {r.count(1)}  已有 {r.count(0)}  失败 {r.count(-1)}", flush=True)

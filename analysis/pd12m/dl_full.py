import pandas as pd, requests, os, sys, time
from concurrent.futures import ThreadPoolExecutor
d = pd.read_csv("fullset.csv")
S = requests.Session(); S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=2))
def g(t):
    k, u = t
    p = f"full/{k:04d}.jpg"
    if os.path.exists(p) and os.path.getsize(p) > 1000: return 0
    try:
        r = S.get(u, timeout=180); open(p, "wb").write(r.content); return len(r.content)
    except Exception as e: print("ERR", k, e); return -1
t0=time.time()
with ThreadPoolExecutor(24) as ex:
    tot = sum(max(0,x) for x in ex.map(g, enumerate(d.url)))
print("bytes", tot/1e9, "GB", time.time()-t0, "s")

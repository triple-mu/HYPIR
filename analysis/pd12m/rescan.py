import sys, csv, time, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,"/home/ubuntu/workspace/contest/CSIG-2026/Moebius/tools")
import pd12m_sort as P
P._polys()
d = pd.read_csv("sample.csv")
rows = list(zip(d.url, d.width, d.height, d.caption))
t0=time.time()
with open("exif_head2.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=P.COLS, extrasaction="ignore", quoting=csv.QUOTE_ALL); w.writeheader()
    with ThreadPoolExecutor(64) as ex:
        for r in ex.map(P.probe_head, rows): w.writerow(r)
print("done", time.time()-t0)

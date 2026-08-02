from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor
import sys
def g(i):
    for _ in range(3):
        try:
            return hf_hub_download("Spawning/PD12M", f"metadata/pd12m.{i:03d}.parquet",
                                   repo_type="dataset",
                                   local_dir="/home/ubuntu/workspace/contest/CSIG-2026/pd12m_recon/meta")
        except Exception as e:
            err = e
    print("FAIL", i, err); return None
with ThreadPoolExecutor(16) as ex:
    r = list(ex.map(g, range(125)))
print("ok", sum(x is not None for x in r))

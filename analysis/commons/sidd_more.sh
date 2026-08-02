#!/bin/bash
for i in 0 1 2 3; do
  /home/ubuntu/miniconda3/envs/torch/bin/python - "$i" <<'PY' &
import sys, os
sys.path.insert(0,'/home/ubuntu/workspace/CSIG-2026-src')
from rzip import R, entries, fetch
import re
r = R("http://130.63.97.225/share/SIDD_Medium_Srgb.zip")
es = [e for e in entries(r) if re.search(r"GT_SRGB.*\.PNG$", e[0])]
k = int(sys.argv[1])
os.makedirs('/home/ubuntu/workspace/CSIG-2026-src/samp/sidd', exist_ok=True)
sel = es[k::4][::max(1,len(es)//4//4)][:4]
for e in sel:
    fn = '/home/ubuntu/workspace/CSIG-2026-src/samp/sidd/' + os.path.basename(e[0])
    if os.path.exists(fn): continue
    try:
        open(fn,'wb').write(fetch(r,e)); print('ok', e[0], flush=True)
    except Exception as ex: print('err', ex, flush=True)
PY
done
wait
echo DONE

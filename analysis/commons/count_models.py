import sys, json, time
sys.path.insert(0, '/home/ubuntu/workspace/contest/CSIG-2026/commons')
from wq import hits

models = json.load(open('/home/ubuntu/workspace/contest/CSIG-2026/commons/models.json'))
res = {}
for brand, lst in models.items():
    res[brand] = []
    for qid, label in lst:
        n = hits(f'haswbstatement:P4082={qid}')
        time.sleep(0.30)
        if n:
            res[brand].append([qid, label, n])
    res[brand].sort(key=lambda x: -x[2])
    tot = sum(x[2] for x in res[brand])
    print(f'{brand}: models_with_files={len(res[brand])}/{len(lst)} total_files={tot}', flush=True)
    for x in res[brand][:8]:
        print('    ', x[0], x[1], x[2], flush=True)
json.dump(res, open('/home/ubuntu/workspace/contest/CSIG-2026/commons/p4082_counts.json', 'w'), ensure_ascii=False, indent=0)

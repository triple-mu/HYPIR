import sys, json, time
sys.path.insert(0, '/home/ubuntu/workspace/contest/CSIG-2026/commons')
from wq import api
import commons_harvest as ch

def h(q):
    for _ in range(4):
        try:
            d = api({'action': 'query', 'list': 'search', 'srsearch': q, 'srnamespace': 6,
                     'srlimit': 1, 'srinfo': 'totalhits', 'srprop': ''})
        except Exception:
            time.sleep(3); continue
        time.sleep(0.25)
        if d is None or 'error' in d: continue
        if 'timed out' in json.dumps(d.get('warnings', {})): return 'TO'
        return d['query']['searchinfo']['totalhits']
    return 'FAIL'

def union(base):
    a, b, ab = h(base + ' filew:>2999'), h(base + ' fileh:>2999'), h(base + ' filew:>2999 fileh:>2999')
    if any(isinstance(x, str) for x in (a, b, ab)): return None
    return a + b - ab

PROV = ['Anhui', 'Beijing', 'Chongqing', 'Fujian', 'Gansu', 'Guangdong', 'Guangxi', 'Guizhou',
        'Hainan', 'Hebei', 'Heilongjiang', 'Henan', 'Hubei', 'Hunan', 'Inner Mongolia', 'Jiangsu',
        'Jiangxi', 'Jilin', 'Liaoning', 'Ningxia', 'Qinghai', 'Shaanxi', 'Shandong', 'Shanghai',
        'Shanxi', 'Sichuan', 'Tianjin', 'Tibet', 'Xinjiang', 'Yunnan', 'Zhejiang',
        'Hong Kong', 'Macau', 'Taiwan']
tot_all = tot_free = 0
print('%-18s %9s %9s' % ('province', '>=3000px', 'free-lic'))
res = {}
for p in PROV:
    base = 'deepcat:"%s" %s %s' % (p, ch.CN_PHONES, ch.JPG)
    u, f = union(base), union(base + ' ' + ch.FREE_LIC)
    res[p] = (u, f)
    if u is None:
        print('%-18s %9s %9s   <-- deepcat 超时，不可信' % (p, 'TO', 'TO')); continue
    tot_all += u; tot_free += f
    print('%-18s %9d %9d' % (p, u, f), flush=True)
print('%-18s %9d %9d   (省份间有重叠，为上界)' % ('SUM', tot_all, tot_free))
json.dump(res, open('/home/ubuntu/workspace/contest/CSIG-2026/commons/prov.json', 'w'))

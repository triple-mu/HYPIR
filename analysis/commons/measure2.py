import sys, json, time
sys.path.insert(0, '/home/ubuntu/workspace/contest/CSIG-2026/commons')
from wq import api
import commons_harvest as ch

def h(q):
    for _ in range(3):
        try:
            d = api({'action': 'query', 'list': 'search', 'srsearch': q, 'srnamespace': 6,
                     'srlimit': 1, 'srinfo': 'totalhits', 'srprop': ''})
        except Exception:
            time.sleep(2); continue
        time.sleep(0.3)
        if d is None or 'error' in d: continue
        if 'timed out' in json.dumps(d.get('warnings', {})): return 'DEEPCAT_TIMEOUT'
        return d['query']['searchinfo']['totalhits']
    return 'FAIL'

def union(base):
    a, b, ab = h(base + ' filew:>2999'), h(base + ' fileh:>2999'), h(base + ' filew:>2999 fileh:>2999')
    if any(isinstance(x, str) for x in (a, b, ab)): return None
    return a + b - ab

print('%-22s %10s %10s %10s %7s' % ('bucket', 'all_jpg', '>=3000px', 'free-lic', 'free%'))
for name, cat in list(ch.BRAND_CAT.items()) + [('ALL-Chinese', ch.CN_PHONES)]:
    tot = h(cat + ' ' + ch.JPG)
    u = union(cat + ' ' + ch.JPG)
    f = union(cat + ' ' + ch.JPG + ' ' + ch.FREE_LIC)
    print('%-22s %10s %10s %10s %6.1f%%' % (name, tot, u, f, 100.0 * f / max(u, 1)), flush=True)

print()
print('%-22s %10s %10s %10s' % ('geo x CNphone', '>=3000px', 'free-lic', 'sample'))
for g in ['Zhejiang', 'Hangzhou', 'Wenzhou', 'Ningbo', 'Shanghai', 'Jiangsu', 'Guangzhou',
          'Shenzhen', 'Beijing', 'Fujian', 'Chengdu', 'Hong Kong', 'Sichuan', 'Yunnan', 'Shandong']:
    base = 'deepcat:"%s" %s %s' % (g, ch.CN_PHONES, ch.JPG)
    u, f = union(base), union(base + ' ' + ch.FREE_LIC)
    print('%-22s %10s %10s' % (g, u, f), flush=True)

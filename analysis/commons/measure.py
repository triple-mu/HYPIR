import sys, json, time
sys.path.insert(0, '/home/ubuntu/workspace/contest/CSIG-2026/commons')
from wq import api

def h(q):
    for _ in range(3):
        d = api({'action': 'query', 'list': 'search', 'srsearch': q, 'srnamespace': 6,
                 'srlimit': 1, 'srinfo': 'totalhits', 'srprop': ''})
        time.sleep(0.3)
        if d is None or 'error' in d: continue
        w = json.dumps(d.get('warnings', {}))
        if 'timed out' in w: return ('DEEPCAT_TIMEOUT', None)
        return (d['query']['searchinfo']['totalhits'], None)
    return ('FAIL', None)

def union3(base):
    a = h(base + ' filew:>2999')[0]
    b = h(base + ' fileh:>2999')[0]
    ab = h(base + ' filew:>2999 fileh:>2999')[0]
    if any(isinstance(x, str) for x in (a, b, ab)): return (a, b, ab, 'ERR')
    return (a, b, ab, a + b - ab)

BRAND_CATS = {
    'Huawei(+Honor)': 'deepcat:"Photos taken with Huawei mobile phones"',
    'Xiaomi(+Redmi/Poco)': 'deepcat:"Photos taken with Xiaomi mobile phones"',
    'Oppo': 'deepcat:"Photos taken with Oppo mobile phones"',
    'Vivo': 'deepcat:"Taken with Vivo mobile phones"',
    'Realme': 'deepcat:"Taken with Realme mobile phones"',
    'OnePlus': 'deepcat:"Taken with OnePlus mobile phones"',
    'Meizu': 'deepcat:"Photos taken with Meizu mobile phones"',
    'ZTE': 'deepcat:"Taken with ZTE mobile phones"',
    'ALL-Chinese': 'deepcat:"Photos taken with Chinese mobile phones"',
}
LIC_OK = 'haswbstatement:P275=Q6938433|P275=Q20007257|P275=Q14947546|P275=Q19125117|P275=Q18810333|P275=Q30942811'
JPG = ' filemime:image/jpeg'

print('=== A) brand x resolution (long side >=3000 = filew>2999 UNION fileh>2999) ===')
print(f'{"brand":24} {"total":>8} {"w>2999":>8} {"h>2999":>8} {"both":>8} {"UNION":>8} {"union+nonSA-lic":>16}')
rows = {}
for k, cat in BRAND_CATS.items():
    tot = h(cat + JPG)[0]
    a, b, ab, u = union3(cat + JPG)
    la = h(cat + JPG + ' filew:>2999 ' + LIC_OK)[0]
    lb = h(cat + JPG + ' fileh:>2999 ' + LIC_OK)[0]
    lab = h(cat + JPG + ' filew:>2999 fileh:>2999 ' + LIC_OK)[0]
    lu = la + lb - lab if not any(isinstance(x, str) for x in (la, lb, lab)) else 'ERR'
    rows[k] = dict(total=tot, w=a, hh=b, both=ab, union=u, lic_union=lu)
    print(f'{k:24} {tot:>8} {a:>8} {b:>8} {ab:>8} {u:>8} {str(lu):>16}', flush=True)

print()
print('=== B) geography x Chinese-phone ===')
CN = BRAND_CATS['ALL-Chinese']
geo = ['Hangzhou', 'Wenzhou', 'Zhejiang', 'Ningbo', 'Shanghai', 'Jiangsu', 'Suzhou, Jiangsu',
       'Guangzhou', 'Shenzhen', 'Beijing', 'Chengdu', "Xi'an", 'Chongqing', 'Fujian', 'Hong Kong']
print(f'{"geo":22} {"geo_alone":>10} {"x CNphone":>10} {"xCN+jpg+>=3000":>15} {"+nonSA lic":>11}')
georows = {}
for g in geo:
    ga = h(f'deepcat:"{g}"')[0]
    gc = h(f'deepcat:"{g}" ' + CN)[0]
    if isinstance(gc, str):
        print(f'{g:22} {str(ga):>10} {str(gc):>10}'); continue
    a, b, ab, u = union3(f'deepcat:"{g}" ' + CN + JPG)
    la = h(f'deepcat:"{g}" ' + CN + JPG + ' filew:>2999 ' + LIC_OK)[0]
    lb = h(f'deepcat:"{g}" ' + CN + JPG + ' fileh:>2999 ' + LIC_OK)[0]
    lab = h(f'deepcat:"{g}" ' + CN + JPG + ' filew:>2999 fileh:>2999 ' + LIC_OK)[0]
    lu = la + lb - lab if not any(isinstance(x, str) for x in (la, lb, lab)) else 'ERR'
    georows[g] = dict(geo_alone=ga, x_cn=gc, x_cn_3000=u, lic=lu)
    print(f'{g:22} {str(ga):>10} {str(gc):>10} {str(u):>15} {str(lu):>11}', flush=True)

json.dump({'brand': rows, 'geo': georows}, open('/home/ubuntu/workspace/contest/CSIG-2026/commons/measure.json', 'w'), indent=1)

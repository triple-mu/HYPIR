import sys, json, time
sys.path.insert(0, '/home/ubuntu/workspace/contest/CSIG-2026/commons')
from wq import api

BRANDS = ['Huawei', 'HUAWEI', 'Xiaomi', 'Redmi', 'Poco', 'Oppo', 'OPPO', 'Vivo', 'vivo',
          'Honor', 'HONOR', 'OnePlus', 'Realme', 'realme', 'Meizu', 'ZTE', 'Nubia', 'Hisense', 'Lenovo', 'Smartisan']
out = {}
for b in BRANDS:
    pref, cont, got = 'Taken with ' + b, None, []
    while True:
        p = {'action': 'query', 'list': 'allcategories', 'acprefix': pref, 'aclimit': 500, 'acprop': 'size'}
        if cont: p['acfrom'] = cont
        d = api(p)
        cs = d['query']['allcategories']
        got += [(c['category'], c['files']) for c in cs]
        cont = d.get('continue', {}).get('accontinue')
        time.sleep(0.3)
        if not cont: break
    out[b] = got
json.dump(out, open('/home/ubuntu/workspace/contest/CSIG-2026/commons/takenwith_cats.json', 'w'), ensure_ascii=False, indent=0)
for b, got in out.items():
    tot = sum(n for _, n in got)
    print(f'{b}: {len(got)} cats, {tot} files')

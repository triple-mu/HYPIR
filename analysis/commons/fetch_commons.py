import hashlib, urllib.parse, urllib.request, os, sys, concurrent.futures as cf
NAMES = [l.strip() for l in open(sys.argv[1]) if l.strip()]
OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
def url(n):
    n = n.replace(' ', '_')
    m = hashlib.md5(n.encode('utf-8')).hexdigest()
    return f"upload.wikimedia.org/wikipedia/commons/{m[0]}/{m[0:2]}/{urllib.parse.quote(n)}"
def go(n):
    fn = os.path.join(OUT, hashlib.md5(n.encode()).hexdigest()[:10] + ".jpg")
    if os.path.exists(fn): return n, 'skip'
    u = "https://wsrv.nl/?url=" + urllib.parse.quote(url(n), safe='') + "&output=jpg&q=100"
    try:
        b = urllib.request.urlopen(u, timeout=180).read()
        if len(b) < 20000: return n, 'tiny ' + str(len(b))
        open(fn, 'wb').write(b); return n, f'ok {len(b)}'
    except Exception as e: return n, 'err ' + str(e)[:60]
with cf.ThreadPoolExecutor(6) as ex:
    for n, s in ex.map(go, NAMES): print(s, '|', n[:60], flush=True)

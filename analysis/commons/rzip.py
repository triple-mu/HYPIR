"""用 HTTP Range 从远程 zip 里抽取少量文件（不下整包）。"""
import sys, struct, zlib, io, re, urllib.request

class R:
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method='HEAD')
        r = urllib.request.urlopen(req, timeout=60)
        self.size = int(r.headers['Content-Length'])
        self.final = r.geturl()
    def get(self, start, end):
        req = urllib.request.Request(self.final, headers={'Range': f'bytes={start}-{end}'})
        return urllib.request.urlopen(req, timeout=180).read()

def entries(r):
    tail = r.get(max(0, r.size - 200000), r.size - 1)
    i = tail.rfind(b'PK\x06\x06')          # zip64 EOCD
    if i >= 0:
        cd_size = struct.unpack('<Q', tail[i+40:i+48])[0]
        cd_off  = struct.unpack('<Q', tail[i+48:i+56])[0]
    else:
        i = tail.rfind(b'PK\x05\x06')
        cd_size = struct.unpack('<I', tail[i+12:i+16])[0]
        cd_off  = struct.unpack('<I', tail[i+16:i+20])[0]
    cd = r.get(cd_off, cd_off + cd_size - 1)
    out, p = [], 0
    while p + 46 <= len(cd) and cd[p:p+4] == b'PK\x01\x02':
        meth = struct.unpack('<H', cd[p+10:p+12])[0]
        csz, usz = struct.unpack('<II', cd[p+20:p+28])
        nl, el, cl = struct.unpack('<HHH', cd[p+28:p+34])
        lho = struct.unpack('<I', cd[p+42:p+46])[0]
        name = cd[p+46:p+46+nl].decode('utf-8', 'replace')
        ex = cd[p+46+nl:p+46+nl+el]
        if 0xFFFFFFFF in (csz, usz, lho):    # zip64 extra
            q = 0
            while q + 4 <= len(ex):
                hid, hsz = struct.unpack('<HH', ex[q:q+4]); body = ex[q+4:q+4+hsz]; k = 0
                if hid == 1:
                    if usz == 0xFFFFFFFF: usz = struct.unpack('<Q', body[k:k+8])[0]; k += 8
                    if csz == 0xFFFFFFFF: csz = struct.unpack('<Q', body[k:k+8])[0]; k += 8
                    if lho == 0xFFFFFFFF: lho = struct.unpack('<Q', body[k:k+8])[0]; k += 8
                q += 4 + hsz
        out.append((name, meth, csz, usz, lho))
        p += 46 + nl + el + cl
    return out

def fetch(r, e):
    name, meth, csz, usz, lho = e
    hdr = r.get(lho, lho + 29)
    nl, el = struct.unpack('<HH', hdr[26:30])
    off = lho + 30 + nl + el
    raw = r.get(off, off + csz - 1)
    return zlib.decompress(raw, -15) if meth == 8 else raw

if __name__ == '__main__':
    url, pat, n, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    import os; os.makedirs(outdir, exist_ok=True)
    r = R(url)
    es = [e for e in entries(r) if re.search(pat, e[0]) and e[3] > 0]
    print('total matching', len(es), 'zip size', r.size)
    step = max(1, len(es) // n)
    for e in es[::step][:n]:
        b = fetch(r, e)
        fn = os.path.join(outdir, os.path.basename(e[0]))
        open(fn, 'wb').write(b)
        print('  ', e[0], len(b))

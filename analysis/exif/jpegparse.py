"""手写 JPEG 段解析器：不依赖 PIL，直接扫 marker。"""
import os, sys, struct, json, hashlib

ZIGZAG = [
     0, 1, 8,16, 9, 2, 3,10,
    17,24,32,25,18,11, 4, 5,
    12,19,26,33,40,48,41,34,
    27,20,13, 6, 7,14,21,28,
    35,42,49,56,57,50,43,36,
    29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,
    53,60,61,54,47,55,62,63]

MARKER_NAMES = {
    0xC0:'SOF0',0xC1:'SOF1',0xC2:'SOF2',0xC3:'SOF3',0xC5:'SOF5',0xC6:'SOF6',0xC7:'SOF7',
    0xC9:'SOF9',0xCA:'SOF10',0xCB:'SOF11',0xCD:'SOF13',0xCE:'SOF14',0xCF:'SOF15',
    0xC4:'DHT',0xCC:'DAC',0xD8:'SOI',0xD9:'EOI',0xDA:'SOS',0xDB:'DQT',0xDC:'DNL',
    0xDD:'DRI',0xDE:'DHP',0xDF:'EXP',0xFE:'COM',
}
for i in range(16):
    MARKER_NAMES[0xE0+i] = 'APP%d' % i


def parse(path):
    with open(path,'rb') as f:
        d = f.read()
    r = {
        'path': path, 'name': os.path.basename(path), 'filesize': len(d),
        'md5': hashlib.md5(d).hexdigest(),
        'magic': d[:4].hex(),
        'qtables': {},      # id -> list of 64 (natural order)
        'qtables_zz': {},   # id -> list of 64 (zigzag / stream order)
        'qprec': {},
        'apps': [],         # (marker_int, name, length, first16hex, ident)
        'dht': [],          # (class, id, total_codes)
        'sof': None,
        'progressive': False,
        'dri': None,
        'n_scans': 0,
        'scans': [],
        'markers_order': [],
        'com': [],
        'trailing': 0,
        'eoi_offset': None,
    }
    if d[:2] != b'\xff\xd8':
        r['error'] = 'not a JPEG SOI'
        return r
    i = 2
    n = len(d)
    while i < n:
        # find next marker
        if d[i] != 0xFF:
            i += 1
            continue
        # skip fill bytes
        j = i
        while j < n and d[j] == 0xFF:
            j += 1
        if j >= n: break
        m = d[j]
        i = j + 1
        if m == 0x00 or (0xD0 <= m <= 0xD7):
            continue
        name = MARKER_NAMES.get(m, 'M_%02X' % m)
        if m == 0xD9:
            r['markers_order'].append(name)
            r['eoi_offset'] = i
            # trailing bytes after EOI
            r['trailing'] = n - i
            break
        if m == 0xD8:
            r['markers_order'].append(name)
            continue
        if i + 2 > n: break
        seglen = struct.unpack('>H', d[i:i+2])[0]
        seg = d[i+2:i+seglen]
        r['markers_order'].append(name)

        if m == 0xDB:  # DQT
            p = 0
            while p < len(seg):
                pq = seg[p] >> 4; tq = seg[p] & 15; p += 1
                if pq == 0:
                    vals = list(seg[p:p+64]); p += 64
                else:
                    vals = list(struct.unpack('>64H', seg[p:p+128])); p += 128
                nat = [0]*64
                for k in range(64):
                    nat[ZIGZAG[k]] = vals[k]
                r['qtables'][tq] = nat
                r['qtables_zz'][tq] = vals
                r['qprec'][tq] = pq
        elif m == 0xC4:  # DHT
            p = 0
            while p < len(seg):
                tc = seg[p] >> 4; th = seg[p] & 15; p += 1
                counts = list(seg[p:p+16]); p += 16
                tot = sum(counts)
                p += tot
                r['dht'].append({'class': tc, 'id': th, 'ncodes': tot, 'counts': counts})
        elif m in (0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF):
            prec = seg[0]
            h, w = struct.unpack('>HH', seg[1:5])
            nc = seg[5]
            comps = []
            for c in range(nc):
                cid = seg[6+c*3]; hv = seg[7+c*3]; tq = seg[8+c*3]
                comps.append({'id': cid, 'h': hv>>4, 'v': hv&15, 'tq': tq})
            r['sof'] = {'marker': name, 'prec': prec, 'w': w, 'h': h, 'comps': comps}
            r['progressive'] = m in (0xC2, 0xC6, 0xCA, 0xCE)
        elif m == 0xDD:
            r['dri'] = struct.unpack('>H', seg[:2])[0]
        elif 0xE0 <= m <= 0xEF:
            ident = seg.split(b'\x00',1)[0][:32]
            try: idents = ident.decode('latin-1')
            except: idents = repr(ident)
            r['apps'].append({'marker': m, 'name': name, 'len': seglen,
                              'first16': seg[:16].hex(), 'ident': idents,
                              'offset': i-2, 'data': seg.hex() if seglen < 200000 else None})
        elif m == 0xFE:
            r['com'].append(seg[:200].hex())
        elif m == 0xDA:
            ns = seg[0]
            sc = []
            for c in range(ns):
                sc.append({'cs': seg[1+c*2], 'td': seg[2+c*2]>>4, 'ta': seg[2+c*2]&15})
            p = 1+ns*2
            ss, se, ahal = seg[p], seg[p+1], seg[p+2]
            r['scans'].append({'ncomp': ns, 'comps': sc, 'ss': ss, 'se': se,
                               'ah': ahal>>4, 'al': ahal&15, 'offset': i-2})
            r['n_scans'] += 1
            # skip entropy data
            k = i + seglen
            while k < n-1:
                if d[k] == 0xFF:
                    nb = d[k+1]
                    if nb == 0x00 or (0xD0 <= nb <= 0xD7) or nb == 0xFF:
                        k += 1; continue
                    break
                k += 1
            i = k
            continue
        i += seglen
    return r


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(json.dumps(parse(p), ensure_ascii=False))

import os, sys, struct, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegexif import *

TEST = '/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集'
VAL  = '/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集'

def magic(data):
    if data[:2] == b'\xff\xd8': return 'JPEG'
    if data[:8] == b'\x89PNG\r\n\x1a\n': return 'PNG'
    return 'OTHER:' + data[:8].hex()

rows = []
for i in range(1, 101):
    p = os.path.join(TEST, 'case%d.jpg' % i)
    data = open(p, 'rb').read()
    segs = scan_jpeg(data)
    apps = []
    sof = None
    dqts = None
    coms = []
    exif_tiff = None
    xmp = []
    mpf = None
    icc_len = 0
    for name, off, L, pl in segs:
        if name.startswith('APP'):
            # 识别标识串
            ident = pl.split(b'\x00')[0][:32]
            apps.append((name, L, ident))
            if name == 'APP1' and pl[:6] == b'Exif\x00\x00':
                exif_tiff = pl[6:]
            if name == 'APP1' and pl[:29] == b'http://ns.adobe.com/xap/1.0/\x00':
                xmp.append(pl[29:])
            if name == 'APP1' and pl[:35] == b'http://ns.adobe.com/xmp/extension/\x00':
                xmp.append(('EXT', pl[35:]))
            if name == 'APP2' and pl[:4] == b'MPF\x00':
                mpf = pl[4:]
            if name == 'APP2' and pl[:11] == b'ICC_PROFILE':
                icc_len += L
        elif name.startswith('SOF'):
            if sof is None:
                sof = (name, parse_sof(pl))
        elif name == 'DQT':
            d = parse_dqt(pl)
            dqts = (dqts or []) + d
        elif name == 'COM':
            coms.append(pl)
    rows.append(dict(case=i, path=p, size=len(data), segs=segs, apps=apps, sof=sof,
                     dqts=dqts, coms=coms, exif_tiff=exif_tiff, xmp=xmp, mpf=mpf, icc_len=icc_len))

# 汇总: 哪些有 EXIF
withexif = [r for r in rows if r['exif_tiff']]
print('== 100 张测试图 APP 段汇总 ==')
print('有 EXIF APP1 的: %d 张 -> %s' % (len(withexif), [r['case'] for r in withexif]))
print('有 XMP 的: %s' % [r['case'] for r in rows if r['xmp']])
print('有 MPF 的: %s' % [r['case'] for r in rows if r['mpf']])
print('有 ICC 的: %s' % [r['case'] for r in rows if r['icc_len']])
print('有 COM 的: %s' % [r['case'] for r in rows if r['coms']])
print()
print('== 逐张 APP 段签名 / DQT 指纹 / SOF ==')
for r in rows:
    t0 = r['dqts'][0][2] if r['dqts'] else None
    t1 = r['dqts'][1][2] if r['dqts'] and len(r['dqts']) > 1 else None
    fp = 'dc0=%d m0=%.1f mx0=%d | dc1=%d m1=%.1f mx1=%d' % (
        t0[0], sum(t0)/64, max(t0), t1[0], sum(t1)/64, max(t1)) if t1 else 'na'
    sofn, s = r['sof']
    sub = ','.join('%d:%dx%d' % (c[0], c[1], c[2]) for c in s['comps'])
    appstr = ' '.join('%s(%d,%s)' % (a[0], a[1], a[2].decode('latin1')) for a in r['apps'])
    print('case%-4d %5.2fMB %s %dx%d [%s] %s || %s' % (
        r['case'], r['size']/1e6, sofn, s['w'], s['h'], sub, fp, appstr))

json_out = '/home/ubuntu/workspace/contest/CSIG-2026/Moebius/forensics/rows.json'
import pickle
pickle.dump([{k: v for k, v in r.items() if k != 'segs'} for r in rows],
            open('/home/ubuntu/workspace/contest/CSIG-2026/Moebius/forensics/rows.pkl', 'wb'))

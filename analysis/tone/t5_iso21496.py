import struct, sys
def parse_iso(b):
    """b = APP2 payload 去掉 'urn:iso:std:iso:ts:21496:-1\\0' 后的字节。"""
    o=0
    minv, wrv = struct.unpack('>HH', b[0:4]); o=4
    fl = b[4]; o=5
    is_mc = (fl>>7)&1; use_base_cs = (fl>>6)&1
    def rat(signed=False):
        nonlocal o
        n = struct.unpack('>i' if signed else '>I', b[o:o+4])[0]
        d = struct.unpack('>I', b[o+4:o+8])[0]; o+=8
        return n/d if d else float('nan'), n, d
    bh = rat(); ah = rat()
    print(f"  min_version={minv} writer_version={wrv} flags=0x{fl:02x} is_multichannel={is_mc} use_base_colour_space={use_base_cs}")
    print(f"  base_hdr_headroom      = {bh[0]:.6f} stops  ({bh[1]}/{bh[2]})")
    print(f"  alternate_hdr_headroom = {ah[0]:.6f} stops  ({ah[1]}/{ah[2]})  -> linear {2**ah[0]:.4f}x")
    nch = 3 if is_mc else 1
    for c in range(nch):
        gmin=rat(True); gmax=rat(True); gam=rat(); boff=rat(True); aoff=rat(True)
        print(f"  ch{c}: gain_map_min={gmin[0]:+.6f} gain_map_max={gmax[0]:+.6f} gamma={gam[0]:.6f} "
              f"base_offset={boff[0]:.6f} alt_offset={aoff[0]:.6f}")
        print(f"        -> linear gain range [{2**gmin[0]:.4f}x, {2**gmax[0]:.4f}x]")
    print(f"  consumed {o} of {len(b)} bytes")

d=open(sys.argv[1],'rb').read()
i=0
while True:
    j=d.find(b'urn:iso:std:iso:ts:21496', i)
    if j<0: break
    L=struct.unpack('>H', d[j-2:j])[0]
    pay=d[j:j+L-2]
    body=pay[28:]
    print(f"\nAPP2 ISO21496 @file {j-4}, payload {len(pay)}B, metadata {len(body)}B")
    if len(body)>=5: parse_iso(body)
    else: print("  (marker only, no metadata)")
    i=j+1
print(f"\n--- tail after gain-map EOI ---")
print("bytes 1149183..1149283:", d[1149183:1149283])
print("hex:", d[1149183:1149243].hex())

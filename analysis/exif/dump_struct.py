import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegmeta import MARKER_NAMES

def dump(fp, limit=200):
    data = open(fp, "rb").read()
    n = len(data)
    print("=== %s  (%d bytes) ===" % (os.path.basename(fp), n))
    i = 0
    cnt = 0
    while i < n - 1 and cnt < limit:
        if data[i] != 0xFF:
            print("  @%d  NON-MARKER byte %02x -- resync" % (i, data[i]))
            i += 1
            continue
        m = data[i + 1]
        name = MARKER_NAMES.get(m, "0x%02X" % m)
        if m in (0x00, 0xFF):
            i += 1
            continue
        if m == 0xD8:
            print("@%-9d SOI" % i); i += 2; cnt += 1; continue
        if m == 0xD9:
            print("@%-9d EOI" % i); i += 2; cnt += 1; continue
        if 0xD0 <= m <= 0xD7:
            i += 2; continue
        ln = struct.unpack(">H", data[i + 2:i + 4])[0]
        pl = data[i + 4:i + 2 + ln]
        ident = pl[:24]
        printable = "".join(chr(c) if 32 <= c < 127 else "." for c in ident)
        print("@%-9d %-6s len=%-7d ident=%r" % (i, name, ln, printable))
        i = i + 2 + ln
        cnt += 1
        if m == 0xDA:
            j = i
            while j < n - 1:
                if data[j] == 0xFF and data[j+1] != 0x00 and not (0xD0 <= data[j+1] <= 0xD7) and data[j+1] != 0xFF:
                    break
                j += 1
            print("            [entropy data %d bytes -> next marker @%d = FF%02X]" % (j - i, j, data[j+1] if j < n-1 else 0))
            i = j

for fp in sys.argv[1:]:
    dump(fp)
    print()

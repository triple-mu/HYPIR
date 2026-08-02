"""MPO 文件里主图占多少字节：找第一个 EOI(FFD9) 之后的第二个 SOI(FFD8FF)。"""
import os
import numpy as np

T = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
NATIVE = [1, 2, 3, 4, 5, 38, 64, 65, 66]


def primary_bytes(path):
    d = open(path, "rb").read()
    # 从头扫段，跳过 SOS 后的熵编码数据，找真正的 EOI
    i = 2
    while i < len(d) - 1:
        if d[i] != 0xFF:
            i += 1
            continue
        m = d[i + 1]
        if m == 0xD9:
            return i + 2, len(d)
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m == 0xDA:  # SOS -> 熵数据，逐字节找非填充的 FFD9
            i += 2 + int.from_bytes(d[i + 2:i + 4], "big")
            while i < len(d) - 1:
                if d[i] == 0xFF and d[i + 1] not in (0x00,) and not (0xD0 <= d[i + 1] <= 0xD7):
                    if d[i + 1] == 0xD9:
                        return i + 2, len(d)
                    break
                i += 1
            continue
        i += 2 + int.from_bytes(d[i + 2:i + 4], "big")
    return len(d), len(d)


print(f"{'case':<8}{'总字节':>12}{'主图字节':>12}{'副图字节':>12}{'主图 bpp':>11}")
pb = []
for i in NATIVE:
    p = f"{T}/case{i}.jpg"
    a, tot = primary_bytes(p)
    b = 8 * a / (3072 * 4096)
    pb.append(b)
    print(f"case{i:<4}{tot:>12,}{a:>12,}{tot-a:>12,}{b:>11.2f}")
print(f"\n主图 bpp: 均值 {np.mean(pb):.2f} 范围 {min(pb):.2f}~{max(pb):.2f}")
print("对照：91 张重编码 bpp 0.63~2.66（中位 1.28），验证集 LQ bpp 0.94~1.04")

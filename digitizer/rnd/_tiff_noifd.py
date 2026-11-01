"""TIFF без IFD (недописан): разжать полосы LZW (TIFF, MSB, early change) подряд с байта 8; длины полос и ширина строки по
автокорреляции. tiff_lzw.py SRC OUT_PREFIX"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np


def decode(b, bitpos):
    table = [bytes([i]) for i in range(256)] + [b"", b""]
    out = bytearray(); width = 9; prev = None; nbits = len(b) * 8
    while bitpos + width <= nbits:
        i = bitpos >> 3; off = bitpos & 7
        chunk = int.from_bytes(b[i:i + 3].ljust(3, b"\0"), "big")
        code = (chunk >> (24 - off - width)) & ((1 << width) - 1)
        bitpos += width
        if code == 256:
            del table[258:]; width = 9; prev = None; continue
        if code == 257:
            return out, bitpos, "EOI"
        if prev is None:
            if code > 255:
                return out, bitpos, f"плохой код {code} после clear"
            entry = table[code]
        elif code < len(table):
            entry = table[code]; table.append(prev + entry[:1])
        elif code == len(table):
            entry = prev + prev[:1]; table.append(entry)
        else:
            return out, bitpos, f"плохой код {code} при таблице {len(table)}"
        out += entry; prev = entry
        if len(table) == (1 << width) - 1 and width < 12:
            width += 1
    return out, bitpos, "конец файла"


SRC, OUT = Path(sys.argv[1]), sys.argv[2]
b = SRC.read_bytes()
pos = 8 * 8; strips = []; t0 = time.time()
while pos < len(b) * 8 - 16:
    s, pos2, why = decode(b, pos)
    strips.append((pos // 8, len(s), why, s))
    if why != "EOI" or not s:
        break
    pos = ((pos2 + 7) // 8) * 8                       # следующая полоса — с границы байта
    while pos // 8 < len(b) and b[pos // 8] != 0x80:  # пропуск выравнивания до clear code
        pos += 8
print(f"полос {len(strips)} за {time.time() - t0:.0f} с; длины первых: {[x[1] for x in strips[:6]]}; последняя: {strips[-1][:3]}")
print("до байта", pos // 8, "из", len(b))
raw = b"".join(x[3] for x in strips)
a = np.frombuffer(raw, np.uint8).astype(np.float32)
print("разжато байт", len(raw))
seg = a[:min(len(a), 4_000_000)] - a[:4_000_000].mean()
best = []
for rb in range(600, 30000):
    if rb >= len(seg) // 4:
        break
    c = float(np.dot(seg[:-rb][:1_000_000], seg[rb:][:1_000_000]))
    best.append((c, rb))
best.sort(reverse=True)
print("лучшие длины строки (байт):", [r for _, r in best[:12]])
np.save(OUT + "_raw.npy", np.frombuffer(raw, np.uint8))

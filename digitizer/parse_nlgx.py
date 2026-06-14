"""
Парсер структуры .nlgx (NeuraLOG) — это TIFF-контейнер (II*\\0) с приватными тегами.
Цель: перечислить ВСЕ IFD-теги, типы, размеры, и для приватных тегов (>=34768)
показать сырые байты / интерпретации (double, ascii, ints), чтобы найти:
  - калибровку Depth Axis (depth<->y), Scale Axis (value<->x) в пикселях
  - точки пиксельной трассы кривой
Использование: python parse_nlgx.py <file.nlgx> [--dump-tag N] [--raw-tag N]
"""
import sys, struct

TIFF_TYPES = {
    1: ("BYTE", 1, "B"),
    2: ("ASCII", 1, "s"),
    3: ("SHORT", 2, "H"),
    4: ("LONG", 4, "I"),
    5: ("RATIONAL", 8, "II"),
    6: ("SBYTE", 1, "b"),
    7: ("UNDEFINED", 1, "B"),
    8: ("SSHORT", 2, "h"),
    9: ("SLONG", 4, "i"),
    10: ("SRATIONAL", 8, "ii"),
    11: ("FLOAT", 4, "f"),
    12: ("DOUBLE", 8, "d"),
    13: ("IFD", 4, "I"),
    16: ("LONG8", 8, "Q"),
    17: ("SLONG8", 8, "q"),
    18: ("IFD8", 8, "Q"),
}

# Известные стандартные TIFF-теги для контекста
STD_TAGS = {
    256: "ImageWidth", 257: "ImageLength", 258: "BitsPerSample",
    259: "Compression", 262: "PhotometricInterpretation", 273: "StripOffsets",
    274: "Orientation", 277: "SamplesPerPixel", 278: "RowsPerStrip",
    279: "StripByteCounts", 282: "XResolution", 283: "YResolution",
    284: "PlanarConfiguration", 296: "ResolutionUnit", 305: "Software",
    306: "DateTime", 320: "ColorMap", 322: "TileWidth", 323: "TileLength",
    324: "TileOffsets", 325: "TileByteCounts", 338: "ExtraSamples",
    339: "SampleFormat", 347: "JPEGTables", 530: "YCbCrSubSampling",
    532: "ReferenceBlackWhite",
}


def parse_tiff(data):
    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        raise ValueError(f"Not a TIFF: first bytes {data[:4]!r}")
    magic = struct.unpack(endian + "H", data[2:4])[0]
    assert magic == 42, f"bad magic {magic}"
    first_ifd = struct.unpack(endian + "I", data[4:8])[0]
    print(f"TIFF endian={'little' if endian=='<' else 'big'} magic={magic} first_ifd_off={first_ifd}")

    ifds = []
    off = first_ifd
    ifd_no = 0
    seen = set()
    while off != 0 and off not in seen:
        seen.add(off)
        n = struct.unpack(endian + "H", data[off:off+2])[0]
        print(f"\n=== IFD #{ifd_no} @ {off}: {n} entries ===")
        entries = []
        p = off + 2
        for i in range(n):
            tag, typ, count = struct.unpack(endian + "HHI", data[p:p+8])
            value_off = p + 8
            tname, tsize, _ = TIFF_TYPES.get(typ, (f"?type{typ}", 1, None))
            total = tsize * count
            if total <= 4:
                raw = data[value_off:value_off+4][:total]
                inline = True
                ptr = value_off
            else:
                ptr = struct.unpack(endian + "I", data[value_off:value_off+4])[0]
                raw = data[ptr:ptr+total]
                inline = False
            entries.append((tag, typ, count, ptr, raw, inline))
            p += 12
        next_off = struct.unpack(endian + "I", data[p:p+4])[0]
        ifds.append((off, entries, next_off))

        for tag, typ, count, ptr, raw, inline in entries:
            std = STD_TAGS.get(tag, "")
            tname = TIFF_TYPES.get(typ, (f"?type{typ}",))[0]
            mark = "PRIV" if tag >= 34768 else ("std" if std else "")
            desc = describe(endian, typ, count, raw)
            print(f"  tag {tag:>6} {mark:<4} {std:<22} type={tname:<9} count={count:<6} "
                  f"off={ptr:<8} {'inline' if inline else 'ptr   '} | {desc}")
        off = next_off
        ifd_no += 1
    return endian, ifds


def describe(endian, typ, count, raw, maxshow=8):
    tname, tsize, fmt = TIFF_TYPES.get(typ, (None, 1, None))
    if typ == 2:  # ASCII
        try:
            s = raw.split(b"\x00")[0].decode("latin1")
        except Exception:
            s = repr(raw[:40])
        return f'ascii="{s}"'
    if fmt is None:
        return f"raw={raw[:16].hex()}"
    try:
        if typ in (5, 10):  # rational
            vals = struct.unpack(endian + (fmt * count), raw[:8*count])
            pairs = [f"{vals[i]}/{vals[i+1]}" for i in range(0, len(vals), 2)]
            return "rat=" + ",".join(pairs[:maxshow])
        vals = struct.unpack(endian + (fmt * count), raw[:tsize*count])
        shown = vals[:maxshow]
        more = "..." if count > maxshow else ""
        if typ == 12:  # double
            return "dbl=[" + ", ".join(f"{v:.6g}" for v in shown) + more + "]"
        if typ == 11:
            return "flt=[" + ", ".join(f"{v:.6g}" for v in shown) + more + "]"
        return f"{tname}=[" + ", ".join(str(v) for v in shown) + more + "]"
    except Exception as e:
        return f"raw={raw[:16].hex()} ({e})"


def dump_tag(data, endian, ifds, target, raw_mode=False):
    """Подробный дамп одного тега: все значения + hex."""
    for off, entries, nxt in ifds:
        for tag, typ, count, ptr, raw, inline in entries:
            if tag != target:
                continue
            tname, tsize, fmt = TIFF_TYPES.get(typ, (None, 1, None))
            print(f"\n### TAG {tag} type={tname} count={count} off={ptr} bytes={tsize*count}")
            full = raw if inline else data[ptr:ptr+tsize*count]
            if raw_mode or fmt is None:
                # hex dump
                for i in range(0, min(len(full), 512), 16):
                    chunk = full[i:i+16]
                    hexs = " ".join(f"{b:02x}" for b in chunk)
                    asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
                    print(f"  {ptr+i:>8}: {hexs:<48} {asc}")
                if len(full) > 512:
                    print(f"  ... ({len(full)} bytes total)")
                return
            if typ == 2:
                print("  " + repr(full.split(b'\x00')[0].decode('latin1', 'replace')))
                return
            vals = struct.unpack(endian + fmt*count, full[:tsize*count])
            for i, v in enumerate(vals):
                print(f"  [{i}] {v}")
                if i > 200:
                    print(f"  ... {count} total")
                    break


if __name__ == "__main__":
    path = sys.argv[1]
    data = open(path, "rb").read()
    print(f"FILE {path}  size={len(data)}")
    endian, ifds = parse_tiff(data)

    if "--dump-tag" in sys.argv:
        t = int(sys.argv[sys.argv.index("--dump-tag")+1])
        dump_tag(data, endian, ifds, t, raw_mode=False)
    if "--raw-tag" in sys.argv:
        t = int(sys.argv[sys.argv.index("--raw-tag")+1])
        dump_tag(data, endian, ifds, t, raw_mode=True)

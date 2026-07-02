"""
write_nlgx.py — ЗАПИСЬ .nlgx (обратное к extract_nlgx). Цель проекта: сдаём nlgx+bck,
поэтому оцифровщик должен УМЕТЬ ПИСАТЬ nlgx из вычисленной трассы+калибровки,
минуя UI NeuraLOG.

Подход: generic round-trip TIFF.
  read_full()  — захватить КАЖДЫЙ тег каждого IFD как (tag,type,count,raw_bytes)
                 + strip-данные (8-байтная заглушка-картинка, тег 273/279).
  write_full() — пере-сериализовать: header, затем для каждого IFD его директория
                 + внешние данные (значения >4 байт и strip), с пересчётом смещений.
  Спец-обработка: тег 273 (StripOffsets) и next-IFD-pointer пересчитываются;
  остальные значения переносятся как есть.

Проверка: extract(write(read(orig))) семантически == extract(orig); diff байт.
Редактирование: set_tag(ifds, ifd_idx, tag, type, values) — заменить трассу/калибровку.

bck = nlgx с 1 отличающимся байтом (флаг) — write_bck() ставит его.
"""
import struct, sys
from extract_nlgx import read_ifds, val, extract, FMT, NULL

TYSIZE = {t: sz for t, (f, sz) in FMT.items()}


def read_full(data):
    """Полный разбор: list[ifd], ifd = {'entries':[(tag,typ,count,raw)], 'strip':bytes, 'strip_tag':int}."""
    assert data[:2] == b"II"
    e = "<"
    ifds = []
    off = struct.unpack(e + "I", data[4:8])[0]
    seen = set()
    while off and off not in seen:
        seen.add(off)
        n = struct.unpack(e + "H", data[off:off+2])[0]
        entries = []
        p = off + 2
        strip_off = strip_cnt = None
        for _ in range(n):
            tag, typ, count = struct.unpack(e + "HHI", data[p:p+8])
            sz = TYSIZE.get(typ, 1)
            total = sz * count
            if total <= 4:
                raw = data[p+8:p+8+total]
            else:
                ptr = struct.unpack(e + "I", data[p+8:p+12])[0]
                raw = data[ptr:ptr+total]
            if tag == 273:  # StripOffsets
                strip_off = struct.unpack(e + "I", data[p+8:p+12])[0] if total <= 4 else None
                strip_off = struct.unpack(e + "I", (raw + b"\0\0\0\0")[:4])[0]
            if tag == 279:  # StripByteCounts
                strip_cnt = struct.unpack(e + "I", (raw + b"\0\0\0\0")[:4])[0]
            entries.append([tag, typ, count, raw])
            p += 12
        nxt = struct.unpack(e + "I", data[p:p+4])[0]
        strip = data[strip_off:strip_off+strip_cnt] if (strip_off is not None and strip_cnt) else b""
        ifds.append({"entries": entries, "strip": strip})
        off = nxt
    return ifds


def write_full(ifds, e="<"):
    """Сериализовать ifds обратно в байты nlgx. Раскладка: header, затем по IFD
       директория + его внешние данные (strip + значения >4Б)."""
    # сначала вычислим размеры и смещения каждого IFD
    cursor = 8  # после header
    layout = []
    for ifd in ifds:
        ents = ifd["entries"]
        n = len(ents)
        dir_size = 2 + 12 * n + 4
        dir_off = cursor
        data_off = dir_off + dir_size
        # внешние данные: strip + значения >4 байт
        ext = []  # (label, bytes)
        if ifd["strip"]:
            ext.append(("strip", ifd["strip"]))
        for idx, (tag, typ, count, raw) in enumerate(ents):
            total = TYSIZE.get(typ, 1) * count
            if total > 4:
                ext.append((idx, raw))
        # выровнять каждое внешнее значение на 2 (TIFF word-align)
        data_size = 0
        ext_off = {}
        for label, b in ext:
            if data_size % 2:
                data_size += 1
            ext_off[label] = data_off + data_size
            data_size += len(b)
        layout.append({"dir_off": dir_off, "data_off": data_off,
                       "ext_off": ext_off, "ext": ext})
        cursor = data_off + data_size
    total_len = cursor

    buf = bytearray(total_len)
    buf[0:4] = b"II" + struct.pack(e + "H", 42)
    struct.pack_into(e + "I", buf, 4, layout[0]["dir_off"])

    for i, ifd in enumerate(ifds):
        lo = layout[i]
        ents = ifd["entries"]
        p = lo["dir_off"]
        struct.pack_into(e + "H", buf, p, len(ents)); p += 2
        for idx, (tag, typ, count, raw) in enumerate(ents):
            total = TYSIZE.get(typ, 1) * count
            struct.pack_into(e + "HHI", buf, p, tag, typ, count)
            if tag == 273:  # StripOffsets -> новый offset strip
                struct.pack_into(e + "I", buf, p+8, lo["ext_off"].get("strip", 0))
            elif total <= 4:
                buf[p+8:p+8+total] = raw + b"\x00" * (4 - total)
            else:
                struct.pack_into(e + "I", buf, p+8, lo["ext_off"][idx])
            p += 12
        nxt = layout[i+1]["dir_off"] if i+1 < len(layout) else 0
        struct.pack_into(e + "I", buf, p, nxt)
        # внешние данные
        for label, b in lo["ext"]:
            o = lo["ext_off"][label]
            buf[o:o+len(b)] = b
    return bytes(buf)


def find_ifd(ifds, pred):
    """Индексы IFD по предикату над dict тегов."""
    out = []
    for i, ifd in enumerate(ifds):
        tags = {t: (typ, c, raw) for t, typ, c, raw in ifd["entries"]}
        if pred(tags):
            out.append(i)
    return out


def set_tag(ifds, ifd_idx, tag, typ, values):
    """Заменить/добавить значение тега. values: list или скаляр; typ — TIFF-тип."""
    fmt, sz = FMT[typ]
    if not isinstance(values, (list, tuple)):
        values = [values]
    if typ == 2:  # ASCII
        raw = (values[0].encode("latin1") if isinstance(values[0], str) else values[0])
        if not raw.endswith(b"\x00"):
            raw += b"\x00"
        count = len(raw)
    else:
        raw = struct.pack("<" + fmt * len(values), *values)
        count = len(values)
    ents = ifds[ifd_idx]["entries"]
    for ent in ents:
        if ent[0] == tag:
            ent[1], ent[2], ent[3] = typ, count, raw
            return
    ents.append([tag, typ, count, raw])
    ents.sort(key=lambda x: x[0])


def write_bck(nlgx_bytes, off=5489, byte=0x12):
    """УСТАРЕЛО — НЕ ИСПОЛЬЗОВАТЬ (анализ 02.07): «флаг off=5489» был артефактом одного файла;
    в других nlgx этот offset попадает ВНУТРЬ данных тега 35490 (молча портит трассу).
    Реальный .bck NeuraLOG = предыдущее сохранение → пишите точную копию nlgx."""
    b = bytearray(nlgx_bytes)
    if off < len(b):
        b[off] = byte
    return bytes(b)


# ---- самопроверка round-trip ----
def _roundtrip(path):
    orig = open(path, "rb").read()
    ifds = read_full(orig)
    out = write_full(ifds)
    # семантическая сверка через extract
    import tempfile, os
    tmp = path + ".__rt.nlgx"
    open(tmp, "wb").write(out)
    a = extract(path); b = extract(tmp)
    os.remove(tmp)
    same_len = len(orig) == len(out)
    ndiff = sum(1 for x, y in zip(orig, out) if x != y) + abs(len(orig)-len(out))
    da_ok = a["depth_axis"] == b["depth_axis"]
    sa_ok = a["scale_axes"] == b["scale_axes"]
    cv_ok = ([c["xs"] for c in a["curves"]] == [c["xs"] for c in b["curves"]] and
             [c["segments"] for c in a["curves"]] == [c["segments"] for c in b["curves"]])
    print(f"{path.split(chr(92))[-1]}")
    print(f"  len orig={len(orig)} out={len(out)} same_len={same_len} byte_diffs={ndiff}")
    print(f"  semantic: depth_axis={da_ok} scale_axes={sa_ok} curves(xs+seg)={cv_ok}")
    return da_ok and sa_ok and cv_ok


if __name__ == "__main__":
    for p in sys.argv[1:]:
        _roundtrip(p)

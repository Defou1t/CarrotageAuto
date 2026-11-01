r"""_final_exif.py — `_final_run.py` с разворотом скана по EXIF-ориентации (01.11, §6.352).

`auto.imaging.load_rgb` читает скан `Image.open` без учёта тега EXIF Orientation, а эталон / рамка nlgx — в развёрнутой системе
(яркость скана в точках эталона: развёрнутый 70–109, сырой 182–183 при медиане листа 216). В архиве 12 таких сканов: ориентация
6 / 8 (90°) — рамка вне скана, падение или мусор; 3 (180°) — лист идёт молча по перевёрнутому скану. Здесь `load_rgb`
подменяется только в своём процессе (живой `auto/` не трогается, пока идут A/B §6.350–6.351: их повторы читают сканы тем же
`load_rgb`); у сканов без тега — прежний `load_rgb` бит-в-бит.

Второй дефект (01.11, круг 3): 17 TIFF семейства БВп (LZW, строка на полосу) — `OSError: decoder error -2`, libtiff «Invalid strip
byte count 0, strip 0»: первые 8 полос не записаны (смещение 0, длина 0), остальные целы. Если прежний `load_rgb` падает на таком
TIFF — в копии в памяти пустые полосы указывают на первую непустую (8 верхних строк поля дублируются, геометрия рамки та же).

  _final_exif.py --out F:/nds/output/final_2026-10/FINAL --sheets F:/nds/output/taskS/final_exif_sheets.txt --shard 0/1
      [--add-failed "decoder|MemoryError"]  — добавить листы, чей последний статус в `--out/_log_*.tsv` содержит одну из строк
      [--requeue E:/Carrotagki_auto/_quarantine/2026-11-01_exif]   — прежние каталоги этих листов в `--out` — в карантин (с описью),
                                                                     иначе `_final_run` их пропустит как сделанные
"""
import sys, io, time, shutil, struct
from pathlib import Path
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
import numpy as np
from PIL import Image, ImageOps
from auto import imaging

_load_rgb = imaging.load_rgb


def fix_zero_strips(b):
    """Копия TIFF (II, несжатый IFD0), где полосы с длиной 0 указывают на первую непустую; → (байты, сколько исправлено)."""
    ifd = struct.unpack_from("<I", b, 4)[0]; n = struct.unpack_from("<H", b, ifd)[0]
    ent = {}
    for e in range(n):
        tag, ty, cnt, val = struct.unpack_from("<HHII", b, ifd + 2 + 12 * e)
        ent[tag] = (ty, cnt, val, ifd + 2 + 12 * e + 8)

    def arr(tag):
        ty, cnt, val, here = ent[tag]; w = 4 if ty == 4 else 2; fmt = "<I" if ty == 4 else "<H"
        base = val if cnt * w > 4 else here
        return base, fmt, w, [struct.unpack_from(fmt, b, base + i * w)[0] for i in range(cnt)]
    bo, fo, wo, offs = arr(273); bc, fc, wc, cnts = arr(279)
    good = next(i for i, c in enumerate(cnts) if c > 0)
    bb = bytearray(b); nz = 0
    for i, c in enumerate(cnts):
        if c == 0:
            struct.pack_into(fo, bb, bo + i * wo, offs[good]); struct.pack_into(fc, bb, bc + i * wc, cnts[good]); nz += 1
    return bytes(bb), nz


def load_rgb(path):
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        if im.getexif().get(0x0112, 1) != 1:
            return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    try:
        return _load_rgb(path)
    except OSError:
        b = Path(path).read_bytes()
        if b[:4] != b"II*\0":
            raise
        fb, nz = fix_zero_strips(b)
        if not nz:
            raise
        with Image.open(io.BytesIO(fb)) as im:
            return np.asarray(im.convert("RGB"))


imaging.load_rgb = load_rgb


def add_failed(out, sheets, pat):
    """Список `sheets` + листы, чей последний статус в `out/_log_*.tsv` содержит одну из строк `pat` (через |) → путь нового списка."""
    st = {}
    for lg in sorted(Path(out).glob("_log_*.tsv")):
        for line in lg.read_text(encoding="utf-8").splitlines():
            p = line.split("\t")
            if len(p) >= 3:
                st[p[0]] = p[2]
    base = Path(sheets).read_text(encoding="utf-8").split()
    add = sorted(n for n, s in st.items() if any(x in s for x in pat.split("|")) and n not in base)
    new = Path(sheets).with_name(Path(sheets).stem + "_run.txt")
    new.write_text("\n".join(base + add) + "\n", encoding="utf-8")
    print(f"★ к {len(base)} листам списка добавлено {len(add)} со статусом «{pat}»: {new}")
    return str(new)


def requeue(out, sheets, q):
    import _final_run
    for n in Path(sheets).read_text(encoding="utf-8").split():
        d = Path(out) / _final_run.sheet_dir(Path(n))
        if not d.is_dir():
            continue
        dst = Path(q) / d.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        size = sum(f.stat().st_size for f in d.iterdir()) / 2**20
        shutil.move(str(d), str(dst))
        with open(Path(q).parent / "_manifest.tsv", "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{d}\t{dst}\t{size:.1f} МБ\tвыдача / падение итоговой оцифровки по "
                     f"скану без разворота EXIF или с пустыми полосами TIFF (§6.352) — переделана `_final_exif.py`\n")
        print(f"  в карантин: {d.name} ({size:.1f} МБ)")


if __name__ == "__main__":
    if "--add-failed" in sys.argv:
        i = sys.argv.index("--add-failed"); pat = sys.argv[i + 1]; del sys.argv[i:i + 2]
        j = sys.argv.index("--sheets") + 1
        sys.argv[j] = add_failed(sys.argv[sys.argv.index("--out") + 1], sys.argv[j], pat)
    if "--requeue" in sys.argv:
        i = sys.argv.index("--requeue"); q = sys.argv[i + 1]; del sys.argv[i:i + 2]
        requeue(sys.argv[sys.argv.index("--out") + 1], sys.argv[sys.argv.index("--sheets") + 1], q)
    import _final_run
    _final_run.main()

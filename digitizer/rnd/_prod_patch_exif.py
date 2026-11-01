"""§6.352: прод-правка `auto.imaging.load_rgb` — (1) разворот скана по тегу EXIF Orientation (ImageOps.exif_transpose);
(2) TIFF с незаписанными полосами (длина 0) — пустые полосы в копии в памяти указывают на первую непустую.
⚠ ПРИМЕНЯТЬ ТОЛЬКО ПОСЛЕ FINAL DONE И КОНЦА A/B §6.350–6.351 (их повторы читают сканы тем же load_rgb — смешанная база).
(1) У сканов без тега (2731 из 2743) exif_transpose отдаёт копию — массив бит-в-бит; меняются только 12 листов §6.352.
(2) ~22 TIFF Бильче-Волыци (`decoder error -2`): прежний повтор §6.108 с LOAD_TRUNCATED_IMAGES на Pillow 12 (libtiff) падает так же —
лист терялся целиком; у них не записаны первые 8 полос, остальное цело. Прочие OSError — прежний повтор с обрезанными.
После применения — кэши трасс 7 листов (1) (`tcache_*`) собраны по неразвёрнутому скану: пересобрать или исключить; листы (2)
в кэшах отсутствуют (падали) — новые листы стенда, база A/B пересчитывается.

  _prod_patch_exif.py [корень с auto/ — по умолчанию F:/nds/Auto]"""
import sys
from pathlib import Path
ROOT = sys.argv[1] if len(sys.argv) > 1 else "F:/nds/Auto"

NEW = r'''def _fill_zero_strips(b):
    """Копия TIFF (II, IFD0), где полосы с длиной 0 указывают на первую непустую → (байты, сколько заполнено); 0 — не тот случай."""
    import struct
    try:
        ifd = struct.unpack_from("<I", b, 4)[0]
        if ifd < 8:
            return b, 0
        n = struct.unpack_from("<H", b, ifd)[0]
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
    except (struct.error, KeyError, StopIteration):
        return b, 0
    bb = bytearray(b); nz = 0
    for i, c in enumerate(cnts):
        if c == 0:
            struct.pack_into(fo, bb, bo + i * wo, offs[good]); struct.pack_into(fc, bb, bc + i * wc, cnts[good]); nz += 1
    return bytes(bb), nz


def load_rgb(path):
    """RGB-массив скана, развёрнутый по тегу EXIF Orientation. Битый файл читается, но ГРОМКО (§6.108, §6.352).

    ⚠ EXIF (§6.352, 01.11): рамка и эталон nlgx — в системе скана, РАЗВЁРНУТОГО по тегу Orientation; 12 сканов архива несут
    тег 3 / 6 / 8 — без разворота 90° падали или давали мусор, 180° шли молча по перевёрнутому скану. У сканов без тега
    exif_transpose отдаёт копию — массив бит-в-бит.
    ⚠⚠ ПУСТЫЕ ПОЛОСЫ TIFF (§6.108 → §6.352). ~22 TIFF Бильче-Волыци (LZW, строка на полосу) — `decoder error -2`,
    `TIFFFillStrip: Invalid strip byte count 0`: НЕ ЗАПИСАНЫ ПЕРВЫЕ 8 ПОЛОС (смещение 0, длина 0), остальное цело — это не
    обрезанный хвост. Прежний повтор с LOAD_TRUNCATED_IMAGES на Pillow 12 (libtiff) падает так же, и лист терялся целиком.
    Теперь пустые полосы в копии в памяти указывают на первую непустую: верхние строки поля дублируются, геометрия рамки та же.
    ⚠ Прочие OSError — повтор с обрезанными: у картинки может не хватать низа, трасса выйдет короче эталона — это дефект
    ДАННЫХ, а не ведения; поэтому флаг включается ТОЛЬКО на повторе, снимается в `finally` (он глобальный на процесс) и
    печатается предупреждение."""
    import io
    from PIL import Image, ImageFile, ImageOps
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as im:
            return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except OSError as e:
        with open(path, "rb") as fh:
            b = fh.read()
        fb, nz = _fill_zero_strips(b) if b[:4] == b"II*\0" else (b, 0)
        if nz:
            with Image.open(io.BytesIO(fb)) as im:
                arr = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
            print(f"⚠ {path}: {nz} полос TIFF с длиной 0 ({type(e).__name__}: {str(e)[:40]}) — указаны на первую непустую, "
                  f"{arr.shape[1]}×{arr.shape[0]}: верхние строки скана — копия первой записанной.")
            return arr
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            arr = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = False
        print(f"⚠⚠ {path}: ФАЙЛ ОБРЕЗАН ({type(e).__name__}: {str(e)[:60]}) — прочитан ЧАСТИЧНО, "
              f"{arr.shape[1]}×{arr.shape[0]}. Низа скана может не хватать: короткая трасса на "
              f"этом листе — дефект ДАННЫХ, а не ведения.")
        return arr
'''

p = Path(ROOT + "/auto/imaging.py"); t = p.read_text(encoding="utf-8")
i = t.index("def load_rgb(path):\n")
j = t.index("        return arr\n", i) + len("        return arr\n")
old = t[i:j]
assert t.count("def load_rgb(path):\n") == 1 and "ImageOps" not in old and "LOAD_TRUNCATED_IMAGES = True" in old, "load_rgb уже правлен?"
p.write_text(t[:i] + NEW + t[j:], encoding="utf-8")
print("ok")

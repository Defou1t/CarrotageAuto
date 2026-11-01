"""§6.352: прод-правка `auto.imaging.load_rgb` — разворот скана по тегу EXIF Orientation (ImageOps.exif_transpose).
⚠ ПРИМЕНЯТЬ ТОЛЬКО ПОСЛЕ FINAL DONE И КОНЦА A/B §6.350–6.351 (их повторы читают сканы тем же load_rgb — смешанная база).
У сканов без тега (2725 из 2743) exif_transpose отдаёт копию — массив бит-в-бит; меняются только 12 листов §6.352.
После применения — кэши трасс этих листов (`tcache_*`) собраны по неразвёрнутому скану: пересобрать или исключить.

  _prod_patch_exif.py [корень с auto/ — по умолчанию F:/nds/Auto]"""
import sys
from pathlib import Path
ROOT = sys.argv[1] if len(sys.argv) > 1 else "F:/nds/Auto"


def sub(path, pairs):
    p = Path(path); t = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert t.count(a) == 1, (path, a[:80])
        t = t.replace(a, b)
    p.write_text(t, encoding="utf-8")


sub(ROOT + "/auto/imaging.py", [
    ("    обрезанным незаметно.\"\"\"\n    from PIL import Image, ImageFile\n",
     "    обрезанным незаметно.\n"
     "    ⚠ EXIF (§6.352, 01.11): рамка и эталон nlgx — в системе скана, РАЗВЁРНУТОГО по тегу Orientation; 12 сканов архива несут\n"
     "    тег 3 / 6 / 8 — без разворота 90° падали или давали мусор, 180° шли молча по перевёрнутому скану.\"\"\"\n"
     "    from PIL import Image, ImageFile, ImageOps\n"),
    ("        return np.asarray(Image.open(path).convert(\"RGB\"))\n",
     "        return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert(\"RGB\"))\n"),
    ("            arr = np.asarray(Image.open(path).convert(\"RGB\"))\n",
     "            arr = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert(\"RGB\"))\n"),
])
print("ok")

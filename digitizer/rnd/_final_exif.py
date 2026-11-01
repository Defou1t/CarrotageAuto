r"""_final_exif.py — `_final_run.py` с разворотом скана по EXIF-ориентации (01.11, §6.352).

`auto.imaging.load_rgb` читает скан `Image.open` без учёта тега EXIF Orientation, а эталон / рамка nlgx — в развёрнутой системе
(яркость скана в точках эталона: развёрнутый 70–109, сырой 182–183 при медиане листа 216). В архиве 12 таких сканов: ориентация
6 / 8 (90°) — рамка вне скана, падение или мусор; 3 (180°) — лист идёт молча по перевёрнутому скану. Здесь `load_rgb`
подменяется только в своём процессе (живой `auto/` не трогается, пока идут A/B §6.350–6.351: их повторы читают сканы тем же
`load_rgb`); у сканов без тега — прежний `load_rgb` бит-в-бит.

  _final_exif.py --out F:/nds/output/final_2026-10/FINAL --sheets F:/nds/output/taskS/final_exif_sheets.txt --shard 0/1
      [--requeue E:/Carrotagki_auto/_quarantine/2026-11-01_exif]   — прежние каталоги этих листов в `--out` — в карантин (с описью),
                                                                     иначе `_final_run` их пропустит как сделанные
"""
import sys, time, shutil
from pathlib import Path
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
import numpy as np
from PIL import Image, ImageOps
from auto import imaging

_load_rgb = imaging.load_rgb


def load_rgb(path):
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        if im.getexif().get(0x0112, 1) == 1:
            return _load_rgb(path)
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


imaging.load_rgb = load_rgb

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
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{d}\t{dst}\t{size:.1f} МБ\tвыдача итоговой оцифровки по скану без "
                     f"разворота EXIF (§6.352) — переделана `_final_exif.py`\n")
        print(f"  в карантин: {d.name} ({size:.1f} МБ)")


if __name__ == "__main__":
    if "--requeue" in sys.argv:
        i = sys.argv.index("--requeue"); q = sys.argv[i + 1]; del sys.argv[i:i + 2]
        requeue(sys.argv[sys.argv.index("--out") + 1], sys.argv[sys.argv.index("--sheets") + 1], q)
    import _final_run
    _final_run.main()

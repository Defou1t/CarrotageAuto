r"""_tiff_lossless.py — СЖАТИЕ СКАНОВ TIFF БЕЗ ПОТЕРЬ (deflate) С ПОПИКСЕЛЬНОЙ СВЕРКОЙ (аудит 26.09, §6.223).

В архиве 150 TIFF без сжатия (`raw`) или со слабым `packbits` — 67 ГБ; deflate без потерь уменьшает их ≈ вдвое. Для каждого
файла: сжатая копия рядом (`*.deflate.tmp`) → сверка пикселей ДВУМЯ декодерами (PIL и OpenCV — так читают пайплайн и стенды)
→ только при полном совпадении и выигрыше в размере оригинал уходит в карантин (тот же том, мгновенно), копия встаёт на его
место под тем же именем. Иначе копия удаляется, оригинал не тронут. Повторный запуск пропускает уже сжатые (идемпотентно).
По умолчанию — сухой прогон.

  _tiff_lossless.py [--limit N] [--apply] [--quarantine E:/Carrotagki_auto/_quarantine/tiff_originals]
"""
import os, sys, argparse, time, shutil
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("--roots", nargs="+", default=[r"E:/Carrotagki_auto/projects/Archive", r"E:/Carrotagki_auto/data/archive_incomplete"])
ap.add_argument("--quarantine", default=r"E:/Carrotagki_auto/_quarantine/tiff_originals")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--smallest-first", action="store_true")
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()
try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

todo = []
for R in a.roots:
    for dp, dn, fn in os.walk(R):
        for f in fn:
            if f.lower().endswith((".tif", ".tiff")):
                p = os.path.join(dp, f)
                try:
                    c = Image.open(p).info.get("compression", "raw")
                except Exception:
                    continue
                if c in ("raw", "packbits"):
                    todo.append((os.path.getsize(p), p, c))
todo.sort(reverse=not a.smallest_first)
if a.limit:
    todo = todo[:a.limit]
print(f"★ к сжатию: {len(todo)} файлов, {sum(t[0] for t in todo)/2**30:.2f} ГБ; режим {'ПРИМЕНИТЬ' if a.apply else 'сухой'}")
Q = Path(a.quarantine); man = Q / "_manifest_tiff.tsv"
saved = done = bad = 0
for sz, p, c in todo:
    t = time.time(); tmp = p + ".deflate.tmp"
    try:
        im = Image.open(p); im.load()
        kw = {}
        if im.info.get("dpi"):
            kw["dpi"] = im.info["dpi"]
        im.save(tmp, format="TIFF", compression="tiff_deflate", **kw)
        a0 = np.asarray(im); del im
        im2 = Image.open(tmp); im2.load(); a1 = np.asarray(im2); del im2
        same_pil = a0.shape == a1.shape and a0.dtype == a1.dtype and np.array_equal(a0, a1)
        del a0, a1
        c0 = cv2.imread(p, cv2.IMREAD_UNCHANGED); c1 = cv2.imread(tmp, cv2.IMREAD_UNCHANGED)
        same_cv = c0 is not None and c1 is not None and c0.shape == c1.shape and np.array_equal(c0, c1)
        del c0, c1
        nsz = os.path.getsize(tmp)
    except Exception as e:
        print(f"  ⛔ {Path(p).name[:60]}: {type(e).__name__}: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        bad += 1; continue
    ok = same_pil and same_cv and nsz < sz
    print(f"  {'★' if ok else '⛔'} {Path(p).name[:60]:<62} {c:<8} {sz/2**20:7.0f} → {nsz/2**20:7.0f} МБ  "
          f"PIL {'=' if same_pil else '≠'} CV {'=' if same_cv else '≠'}  [{time.time()-t:.0f} с]")
    if ok and a.apply:
        dst = Q / Path(*Path(p).resolve().parts[1:])
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(p, dst)                 # оригинал — в карантин (тот же том)
        os.rename(tmp, p)                 # сжатая копия — на его место, то же имя
        with open(man, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{p}\t{dst}\t{sz/2**20:.1f} → {nsz/2**20:.1f} МБ\t{c} → deflate, пиксели PIL и CV равны\n")
        saved += sz - nsz; done += 1
    else:
        os.remove(tmp)
        bad += (not ok)
print(f"★ ИТОГО: сжато {done}, отказов {bad}, выигрыш {saved/2**30:.2f} ГБ (оригиналы — в карантине до очистки владельцем)")

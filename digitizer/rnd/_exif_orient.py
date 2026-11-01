"""EXIF-ориентация сканов архива: у каких листов скан повёрнут тегом (PIL.Image.open его не применяет).
  exif_orient.py [имя.nlgx ...]   — без аргументов: все листы архива; печать — листы с ориентацией ≠ 1 и размер против рамки."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer")
from collections import Counter
from pathlib import Path
from PIL import Image
from extract_nlgx import extract
from dataset_build import find_image
Image.MAX_IMAGE_PIXELS = None
SRC = sorted(Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx"), key=lambda q: q.name)
want = set(sys.argv[1:])
cnt = Counter(); bad = []
for q in SRC:
    if want and q.name not in want:
        continue
    img = find_image(q)
    if not img:
        cnt["нет скана"] += 1; continue
    try:
        with Image.open(img) as im:
            o = im.getexif().get(0x0112, 1); W, H = im.size
    except Exception as e:
        cnt["не картинка"] += 1; continue
    cnt[o] += 1
    if o != 1:
        da = extract(str(q)).get("depth_axis") or {}
        by = da.get("bottom_y") or 0
        bad.append((q.name, Path(img).suffix, o, W, H, by, "вне скана" if by > H else "влезает"))
print("ориентация:", cnt.most_common())
for b in bad:
    print(*b)

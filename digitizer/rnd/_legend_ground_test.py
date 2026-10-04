r"""_legend_ground_test.py — ПОДПИСИ ЛЕГЕНДЫ С РАМКАМИ (ЛОКАЛЬНАЯ МОДЕЛЬ) И ОБРАЗЕЦ ЛИНИИ СЛЕВА ОТ ПОДПИСИ ПО СКАНУ (04.10, §6.282).

Шапка листа → qwen2.5-vl (LM Studio) → JSON записей легенды {caption, bbox}; рамки рисуются на шапке; слева от каждой подписи
на её высоте ищется горизонтальный ран туши 30–260 px — образец линии: толщина (высота рана), цвет, разрывы.

  _legend_ground_test.py --sheet "2003.05.21_Rybal_149_BKZ1_KB_(2234-2748).nlgx"
"""
import sys, argparse, json, re, base64, io, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from extract_nlgx import extract, NULL

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--sheet", nargs="+", required=True)
ap.add_argument("--model", default="qwen2.5-vl-7b-instruct")
ap.add_argument("--url", default="http://localhost:1234/v1/chat/completions")
ap.add_argument("--max-side", type=int, default=1600)
ap.add_argument("--out", default=r"F:/nds/output/visual_2026-10-03/N_name_style")
a = ap.parse_args()
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
IM = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/img/*"):
    IM.setdefault(q.stem, q)
font = ImageFont.truetype("arial.ttf", 14)


def sample_left(gray, rgb, bx, thr=60):
    """образец линии слева от подписи: на высоте центра подписи ±(высота/2), левее x1 на 10…300 px — самый длинный
    горизонтальный ран туши; → толщина, цвет, доля разрывов, длина"""
    x1, y1, x2, y2 = bx
    yc = int((y1 + y2) / 2); h = max(6, int((y2 - y1) * 0.8))
    best = None
    for xa, xb in ((max(0, int(x1) - 300), max(0, int(x1) - 4)), (int(x2) + 4, min(gray.shape[1], int(x2) + 400))):
        if xb - xa < 30:
            continue
        r_ = _sample_in(gray, rgb, xa, xb, yc, h, thr)
        if r_ and (best is None or r_["length"] > best["length"]):
            best = dict(r_, side="слева" if xa < x1 else "справа")
    return best


def _sample_in(gray, rgb, xa, xb, yc, h, thr):
    reg = gray[max(0, yc - h):yc + h, xa:xb].astype(np.int16)
    paper = np.percentile(gray[max(0, yc - 40):yc + 40, ::4], 90)
    ink = (paper - reg) > thr
    if not ink.any():
        return None
    rowsum = ink.sum(1)
    r = int(np.argmax(rowsum))
    if rowsum[r] < 25:
        return None
    # толщина: подряд строки около r, где тушь в той же колонне ≥ 60% от лучшей строки
    cols = np.flatnonzero(ink[r])
    c0, c1 = cols.min(), cols.max()
    band = ink[:, c0:c1 + 1]
    frac = band.mean(1)
    on = frac >= 0.6 * frac[r]
    top = r
    while top > 0 and on[top - 1]:
        top -= 1
    bot = r
    while bot < len(on) - 1 and on[bot + 1]:
        bot += 1
    thick = bot - top + 1
    seg = ink[top:bot + 1, c0:c1 + 1].any(0)
    gap = 1.0 - float(seg.mean())
    yy = max(0, yc - h) + r; xx = xa + c0 + int(np.argmax(reg[r, c0:c1 + 1] * -1))
    px = rgb[yy, xa + c0 + (c1 - c0) // 2].astype(int)
    return dict(thick=int(thick), length=int(c1 - c0 + 1), gap=round(gap, 2), c1=int(px[0] - px[2]), c2=int(px[1] - (px[0] + px[2]) // 2))


for sh in a.sheet:
    q = SRC[sh]
    mt = extract(str(q))
    starts = [c["top_y"] + next(i for i, x in enumerate(c["xs"]) if x != NULL) for c in mt["curves"]
              if sum(1 for x in c["xs"] if x != NULL) >= 50 and not c["name"].startswith("DA")]
    im = Image.open(IM[q.stem]).convert("RGB")
    y1 = min(starts) + 40
    hy0 = max(0, y1 - 2600)
    head = im.crop((0, hy0, im.width, y1))
    s_ = min(1.0, (1_000_000 / (head.width * head.height)) ** 0.5)
    W_ = max(28, int(head.width * s_) // 28 * 28); H_ = max(28, int(head.height * s_) // 28 * 28)
    small = head.resize((W_, H_), Image.LANCZOS)
    sx, sy = head.width / W_, head.height / H_
    buf = io.BytesIO(); small.save(buf, format="PNG")
    prompt = ("Look at this header of a scanned well-log chart. Find every legend entry: a short horizontal line sample followed by a "
              "written caption that names a curve (a probe code made of letters A, M, N and numbers, or a curve name with units). "
              "Read each caption exactly as written in the image (do not invent captions). Return ONLY JSON: a list of objects "
              "with keys caption and bbox, where bbox is [x1, y1, x2, y2] of the caption text in pixels of this image.")
    body = {"model": a.model, "temperature": 0, "max_tokens": 1200,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                      {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}]}]}
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(a.url, data=json.dumps(body).encode(),
                                                                 headers={"Content-Type": "application/json"}), timeout=900).read())
    txt = r["choices"][0]["message"].get("content") or ""
    mm = re.search(r"\[.*\]", txt, flags=re.S)
    ents = json.loads(mm.group(0)) if mm else []
    print(f"{sh}: записей {len(ents)} за {time.time() - t0:.0f} с")
    gray = np.asarray(head.convert("L")); rgb = np.asarray(head)
    dr = ImageDraw.Draw(head)
    for e in ents:
        try:
            bx = [e["bbox"][0] * sx, e["bbox"][1] * sy, e["bbox"][2] * sx, e["bbox"][3] * sy]
        except Exception:
            continue
        st = sample_left(gray, rgb, bx)
        e["sample"] = st
        dr.rectangle([tuple(bx[:2]), tuple(bx[2:])], outline=(255, 0, 0), width=3)
        dr.text((bx[0], bx[3] + 2), f"{st}" if st else "образца нет", fill=(255, 0, 0), font=font)
        print(f"   {e.get('caption', '')[:50]:50s} образец: {st}")
    if head.width > 1600:
        head = head.resize((1600, int(head.height * 1600 / head.width)))
    head.save(Path(a.out) / f"L_легенда_{q.stem[:40]}.png".replace(" ", "_").replace(",", ""))

r"""Трек 1 + АПСКЕЙЛ: извлечь-вычесть на апскейленном ×SCALE бандле → зазор толщины MGZ/MPZ растёт
(2.2 vs 1.4 → ~4.4 vs 2.8) → prefer_thick надёжнее, меньше свопов на пересечениях. Трасса обратно ÷SCALE.
Бандл-кроп без рамки → dark-only порог + CC-фильтр меток (без structure_mask)."""
import sys
import numpy as np, cv2
sys.path.insert(0, r"F:\nds\Auto")
import auto.imaging as im, auto.meta as M, auto.frame as F
from auto.config import DEFAULT
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
SC = r"F:\nds\Auto\Yatskivska_1_MK_3130_3760_200_D1.jpg"; NL = SC.replace('.jpg', '.nlgx')
OUT = r"C:\Users\Defou1t\AppData\Local\Temp\claude\F--nds-Auto\5bbdb336-3b32-4e7a-b006-ae7aeb0160e5\scratchpad"
p = DEFAULT.cv
rgb = im.load_rgb(SC); H, W = rgb.shape[:2]
m = M.parse_filename(SC, r"F:\nds\Auto\mnemonics.json")
fr = F.frame_from_nlgx(NL, m, rgb=rgb)
ppm = fr.px_per_m
def yof(d): return int(fr.top_y + (d - fr.top_depth) * ppm)
SCALE = 2
CX0, CX1 = 60, 340                              # x-окно бандла (без правого поля/бордюра)
TY, BY = fr.top_y, fr.bottom_y

# апскейл кропа бандла ×SCALE (gray-V достаточно: MK чёрный)
gray = im.value_channel(rgb)[TY:BY, CX0:CX1]
up = cv2.resize(gray, (gray.shape[1] * SCALE, gray.shape[0] * SCALE), interpolation=cv2.INTER_LANCZOS4)
UH, UW = up.shape
fg = (up < p.dark_v).astype(np.uint8)           # dark-only (грид светлый, рамки в кропе нет)
# CC-фильтр: убрать короткие (метки глубины) — высота < 150*SCALE
bridged = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7 * SCALE)))
n, lab, st, _ = cv2.connectedComponentsWithStats(bridged, 8)
keep = np.zeros_like(fg)
for i in range(1, n):
    if st[i, cv2.CC_STAT_HEIGHT] >= 150 * SCALE:
        keep[lab == i] = 1
fg = (keep & fg).astype(np.uint8)
DT = cv2.distanceTransform(fg, cv2.DIST_L2, 5)


def runs_at(fg, DT, y):
    out = []
    row = fg[y]
    xs = np.nonzero(row)[0]
    if not len(xs):
        return out
    cuts = np.nonzero(np.diff(xs) > 2 * SCALE)[0] + 1
    for seg in np.split(xs, cuts):
        a, b = int(seg[0]), int(seg[-1]); dt = float(DT[y, a:b + 1].max())
        out.append({"a": a, "b": b, "c": (a + b) / 2.0, "dt": dt, "spike": (b - a + 1) > 4 * dt + 6 * SCALE})
    return out


def vtx(r, base):
    return (r["a"] if abs(r["a"] - base) > abs(r["b"] - base) else r["b"]) if r["spike"] else r["c"]


def trace_one(fg, DT, seed_y, seed_x, prefer_thick, slmax=26.0 * SCALE, maxjump=70.0 * SCALE, band=85.0 * SCALE):
    rr = {y: runs_at(fg, DT, y) for y in range(UH)}
    st = {"x": seed_x, "sl": 0.0, "anc": seed_x}
    tr, trun = {}, {}

    def march(ys):
        st["x"] = seed_x; st["sl"] = 0.0; st["anc"] = seed_x
        for y in ys:
            runs = rr.get(y) or []
            if not runs:
                st["x"] += float(np.clip(st["sl"], -slmax, slmax)); st["sl"] *= 0.7; continue
            pred = st["x"] + float(np.clip(st["sl"], -slmax, slmax))
            cands = [r for r in runs if r["a"] - 3 * SCALE <= pred <= r["b"] + 3 * SCALE]
            if not cands:
                cands = [r for r in runs if abs(r["c"] - pred) <= maxjump or abs(r["c"] - st["anc"]) <= band]
            if not cands:
                st["x"] += float(np.clip(st["sl"], -slmax, slmax)); st["sl"] *= 0.7; continue
            r = min(cands, key=lambda r: abs(r["c"] - pred) - (5.0 * r["dt"] if prefer_thick else 0.0))
            nx = vtx(r, st["anc"])
            st["sl"] = float(np.clip(0.7 * st["sl"] + 0.3 * (nx - st["x"]), -slmax, slmax))
            st["x"] = nx; tr[y] = nx; trun[y] = (r["a"], r["b"])
            st["anc"] = 0.995 * st["anc"] + 0.005 * r["c"]
    march(range(seed_y, UH))
    march(range(seed_y - 1, -1, -1))
    return tr, trun


ys_dt, xs_dt = np.where(DT == DT.max())
seedy = int(ys_dt[len(ys_dt) // 2]); seedx = int(xs_dt[len(xs_dt) // 2])
mgz, mgz_run = trace_one(fg, DT, seedy, seedx, prefer_thick=True)
fg2 = fg.copy()
for y, (a, b) in mgz_run.items():
    fg2[y, max(0, a - 2 * SCALE):min(UW, b + 2 * SCALE + 1)] = 0
DT2 = cv2.distanceTransform(fg2, cv2.DIST_L2, 5)
nn, lab2, st2, cen2 = cv2.connectedComponentsWithStats(fg2, 8)
if nn > 1:
    big = 1 + int(np.argmax([st2[i, cv2.CC_STAT_HEIGHT] for i in range(1, nn)]))
    sy2 = int(cen2[big][1]); xs2 = np.where(lab2[sy2] == big)[0]; sx2 = int(xs2[len(xs2)//2]) if len(xs2) else int(cen2[big][0])
else:
    sy2, sx2 = seedy, seedx
mpz, _ = trace_one(fg2, DT2, sy2, sx2, prefer_thick=False)
known = sorted(mpz)
if len(known) >= 2:
    kx = [mpz[y] for y in known]
    for y in sorted(mgz):
        if y not in mpz and known[0] <= y <= known[-1]:
            mpz[y] = float(np.interp(y, known, kx))
# обратно в ОРИГИНАЛ: x_orig = x_up/SCALE + CX0 ; y_orig = y_up/SCALE + TY
def to_orig(tr):
    return {int(round(y / SCALE)) + TY: x / SCALE + CX0 for y, x in tr.items()}
mgzo, mpzo = to_orig(mgz), to_orig(mpz)
common = sorted(set(mgzo) & set(mpzo))
print(f"АПСКЕЙЛ×{SCALE}: MGZ DT(up)={DT.max():.1f} | MGZ точек={len(mgzo)} MPZ={len(mpzo)} | "
      f"MGZ левее {100*np.mean([mgzo[y]<=mpzo[y] for y in common]):.0f}%")

COLM, COLP = (220, 0, 0), (0, 110, 230)
def draw(ov, ox, oy):
    for d, col in ((mgzo, COLM), (mpzo, COLP)):
        for y, x in d.items():
            xi = int(x)
            if 0 <= y - oy < ov.shape[0] and 0 <= xi - ox < ov.shape[1]:
                ov[y - oy, max(0, xi - ox - 1):xi - ox + 2] = col
try:
    FONT = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 24)
except Exception:
    FONT = ImageFont.load_default()
full = rgb.copy(); draw(full, 0, 0)
Image.fromarray(full[TY:BY, 0:340]).save(f"{OUT}/MK_id_full_native.png"); print("сохранён MK_id_full_native.png")
for d in (3134, 3192, 3308, 3368, 3482, 3688):
    yc = yof(d); y0, y1 = yc - 130, yc + 130
    crop = rgb[y0:y1, 70:300].copy(); draw(crop, 70, y0)
    im_c = Image.fromarray(crop).resize((230 * 3, 260 * 3), Image.NEAREST)
    ImageDraw.Draw(im_c).text((6, 4), f"{d}м MGZ=красн MPZ=син ап×{SCALE}", fill=(0, 0, 0), font=FONT)
    im_c.save(f"{OUT}/MK_id_{d}.png"); print(f"  сохранён MK_id_{d}.png")

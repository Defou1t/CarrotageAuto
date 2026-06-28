r"""Трек 1 ЯДРО: ИЗВЛЕЧЬ-ВЫЧЕСТЬ. MGZ (толстый доминант) трассируем одиночным 2D-обходом ПЕРВЫМ
(по макс-DT), вычитаем его чернила, MPZ остаётся один → трассируем остаток. Свопы исчезают
(MGZ/MPZ — разные объекты по построению), спайки ведутся (каждая линия одна). Налегание→окклюзия MPZ."""
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
TY, BY, XL, XR = fr.top_y, fr.bottom_y, 80, 320
ppm = fr.px_per_m
def yof(d): return int(fr.top_y + (d - fr.top_depth) * ppm)


def clean_fg():
    fg = (im.dark_mask(rgb, p) & ~im.structure_mask(rgb, p)).astype(np.uint8)
    sub = np.zeros_like(fg); sub[TY:BY, XL:XR] = fg[TY:BY, XL:XR]
    bridged = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bridged, 8)
    keep = np.zeros_like(sub)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_HEIGHT] >= 150:
            keep[lab == i] = 1
    return (keep & sub).astype(np.uint8)


def runs_at(fg, DT, y):
    out = []
    for a, b, c in im.row_runs(fg[y, XL:XR], gap=2):
        a += XL; b += XL; dt = float(DT[y, a:b + 1].max())
        out.append({"a": a, "b": b, "c": (a + b) / 2.0, "dt": dt, "spike": (b - a + 1) > 4 * dt + 6})
    return out


def vtx(r, base):
    return (r["a"] if abs(r["a"] - base) > abs(r["b"] - base) else r["b"]) if r["spike"] else r["c"]


def trace_one(fg, DT, seed_y, seed_x, prefer_thick, slmax=26.0, maxjump=70.0, band=85.0):
    """Одиночный 2D-обход от сида вверх и вниз. prefer_thick: при выборе тянуться к ТОЛСТОМУ рану."""
    rr = {y: runs_at(fg, DT, y) for y in range(TY, BY)}
    st = {"x": seed_x, "sl": 0.0, "anc": seed_x}
    tr, trun = {}, {}

    def march(ys):
        st["x"] = seed_x; st["sl"] = 0.0; st["anc"] = seed_x
        for y in ys:
            runs = rr.get(y) or []
            if not runs:
                st["x"] += float(np.clip(st["sl"], -slmax, slmax)); st["sl"] *= 0.7; continue
            pred = st["x"] + float(np.clip(st["sl"], -slmax, slmax))
            cands = [r for r in runs if r["a"] - 3 <= pred <= r["b"] + 3]                 # связность
            if not cands:
                cands = [r for r in runs if abs(r["c"] - pred) <= maxjump or abs(r["c"] - st["anc"]) <= band]
            if not cands:
                st["x"] += float(np.clip(st["sl"], -slmax, slmax)); st["sl"] *= 0.7; continue
            r = min(cands, key=lambda r: abs(r["c"] - pred) - (2.5 * r["dt"] if prefer_thick else 0.0))
            nx = vtx(r, st["anc"])
            st["sl"] = float(np.clip(0.7 * st["sl"] + 0.3 * (nx - st["x"]), -slmax, slmax))
            st["x"] = nx; tr[y] = nx; trun[y] = (r["a"], r["b"])
            st["anc"] = 0.995 * st["anc"] + 0.005 * r["c"]
    march(range(seed_y, BY))
    march(range(seed_y - 1, TY - 1, -1))
    return tr, trun


fg = clean_fg()
DT = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
# сид MGZ = глобально самый ТОЛСТЫЙ пиксель (точно MGZ)
ys_dt, xs_dt = np.where(DT[TY:BY] == DT[TY:BY].max())
seedy = TY + int(ys_dt[len(ys_dt) // 2]); seedx = int(xs_dt[len(xs_dt) // 2])
print(f"сид MGZ: y={seedy} x={seedx} DT={DT[seedy, seedx]:.1f}")
mgz, mgz_run = trace_one(fg, DT, seedy, seedx, prefer_thick=True)

# вычесть MGZ: пиксели его ранов ±2 → убрать
fg2 = fg.copy()
for y, (a, b) in mgz_run.items():
    fg2[y, max(XL, a - 2):min(XR, b + 3)] = 0
fg2 = (fg2 & (im.dark_mask(rgb, p).astype(np.uint8))).astype(np.uint8)   # на всякий
DT2 = cv2.distanceTransform(fg2, cv2.DIST_L2, 5)
# сид MPZ = самый длинный остаточный CC (тонкая линия)
n, lab, stt, cen = cv2.connectedComponentsWithStats(fg2, 8)
if n > 1:
    big = 1 + int(np.argmax([stt[i, cv2.CC_STAT_HEIGHT] for i in range(1, n)]))
    seedy2 = int(cen[big][1]); xs2 = np.where(lab[seedy2] == big)[0]
    seedx2 = int(xs2[len(xs2) // 2]) if len(xs2) else int(cen[big][0])
else:
    seedy2, seedx2 = seedy, seedx
mpz, _ = trace_one(fg2, DT2, seedy2, seedx2, prefer_thick=False)
nraw = len(mpz)
# ИНТЕРПОЛЯЦИЯ MPZ сквозь ОККЛЮЗИЮ (где MGZ накрыл MPZ): MPZ непрерывна, идёт за MGZ → линейно
known = sorted(mpz)
if len(known) >= 2:
    kx = [mpz[y] for y in known]
    for y in sorted(mgz):
        if y not in mpz and known[0] <= y <= known[-1]:
            mpz[y] = float(np.interp(y, known, kx))
print(f"MGZ точек={len(mgz)} | MPZ точек={len(mpz)} (видимых {nraw} + интерп {len(mpz)-nraw})")
common = sorted(set(mgz) & set(mpz))
print(f"MGZ левее MPZ: {100*np.mean([mgz[y]<=mpz[y] for y in common]):.0f}% | "
      f"MGZ DT(мед)={np.median([DT[y,int(mgz[y])] for y in list(mgz)[::5]]):.1f} "
      f"MPZ DT(мед)={np.median([DT[y,int(mpz[y])] for y in list(mpz)[::5]]):.1f}")

COLM, COLP = (220, 0, 0), (0, 110, 230)
def draw(ov, ox, oy):
    for d, col in ((mgz, COLM), (mpz, COLP)):
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
for d in (3186, 3192, 3220, 3308, 3368, 3740):
    yc = yof(d); y0, y1 = yc - 130, yc + 130
    crop = rgb[y0:y1, 70:300].copy(); draw(crop, 70, y0)
    im_c = Image.fromarray(crop).resize((230 * 3, 260 * 3), Image.NEAREST)
    ImageDraw.Draw(im_c).text((6, 4), f"{d}м MGZ=красн MPZ=син", fill=(0, 0, 0), font=FONT)
    im_c.save(f"{OUT}/MK_id_{d}.png"); print(f"  сохранён MK_id_{d}.png")

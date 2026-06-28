r"""2D-МУЛЬТИ-ОБХОД штриха для MK (2 зонда). Ведёт 2 линии по СВЯЗНОСТИ РАНОВ (ран перекрывает
предсказание = физическая непрерывность штриха → СПАЙК доводится до ВЕРШИНЫ), идентичность по
DT-толщине (тонкий спайк→MPZ, толстое тело→MGZ), наложение→дубль. dark-only (рамки нет)."""
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


fg = clean_fg()
DT = cv2.distanceTransform(fg, cv2.DIST_L2, 5)


def runs_at(y):
    out = []
    for a, b, c in im.row_runs(fg[y, XL:XR], gap=2):
        a += XL; b += XL; dt = float(DT[y, a:b + 1].max())
        out.append((a, b, (a + b) / 2.0, dt, (b - a + 1) > 4 * dt + 6))   # a,b,center,dt,spike
    return out


def vtx(run, base):
    a, b, c, dt, spike = run
    return (a if abs(a - base) > abs(b - base) else b) if spike else c   # спайк→дальний край, иначе центр


def trace_bundle(slmax=26.0, dtlam=10.0, maxjump=70.0, jumppen=60.0, mergepen=8.0, anc_a=0.995):
    rows = list(range(TY, BY)); rr = [runs_at(y) for y in rows]
    seed = next((i for i, r in enumerate(rr) if len(r) >= 2), int(np.argmax([len(r) for r in rr])))
    sr = sorted(rr[seed], key=lambda r: -r[3])[:2]            # толще DT = L0 (MGZ)
    sr = (sr + [sr[-1]] * 2)[:2]
    L = [{"x": r[2], "sl": 0.0, "dt": r[3], "anc": r[2], "tr": {rows[seed]: r[2]}} for r in sr]

    band = 90.0

    def coast(ln):
        ln["x"] += float(np.clip(ln["sl"], -slmax, slmax)); ln["sl"] *= 0.7   # затухание, чтобы не улетать

    def march(order):
        for i in order:
            y = rows[i]; runs = rr[i]
            if not runs:
                for ln in L:
                    coast(ln)
                continue
            pred = [ln["x"] + float(np.clip(ln["sl"], -slmax, slmax)) for ln in L]
            # валидность рана для линии: накрывает предсказание | близко к предсказанию | близко к ЯКОРЮ (ре-захват)
            C = np.full((2, len(runs)), 1e9); pairs = []
            for a in range(2):
                for b, r in enumerate(runs):
                    overlap = r[0] - 3 <= pred[a] <= r[1] + 3
                    jd = abs(r[2] - pred[a]); ad = abs(r[2] - L[a]["anc"])
                    if overlap or jd <= maxjump or ad <= band:
                        C[a, b] = dtlam * abs(L[a]["dt"] - r[3]) + 0.12 * jd + (0.0 if overlap else jumppen)
                        pairs.append((C[a, b], a, b))
            # 1) уникальное жадное назначение (split: 2 рана → 2 линии)
            done = set(); taken = set()
            for c, a, b in sorted(pairs):
                if a in done or b in taken:
                    continue
                done.add(a); taken.add(b)
                r = runs[b]; nx = vtx(r, L[a]["anc"])
                L[a]["sl"] = float(np.clip(0.7 * L[a]["sl"] + 0.3 * (nx - L[a]["x"]), -slmax, slmax))
                L[a]["x"] = nx; L[a]["tr"][y] = nx
                L[a]["dt"] = 0.85 * L[a]["dt"] + 0.15 * r[3]
                L[a]["anc"] = anc_a * L[a]["anc"] + (1 - anc_a) * r[2]
            # 2) не назначенные: сесть на ЛУЧШИЙ валидный ран (дубль с занятым = НАЛОЖЕНИЕ) или coast
            for a in range(2):
                if a in done:
                    continue
                valid = [(C[a, b], b) for b in range(len(runs)) if C[a, b] < 1e9]
                if valid:
                    b = min(valid)[1]; r = runs[b]; nx = vtx(r, L[a]["anc"])
                    L[a]["sl"] = float(np.clip(0.7 * L[a]["sl"] + 0.3 * (nx - L[a]["x"]), -slmax, slmax))
                    L[a]["x"] = nx; L[a]["tr"][y] = nx
                    L[a]["dt"] = 0.85 * L[a]["dt"] + 0.15 * r[3]
                    L[a]["anc"] = anc_a * L[a]["anc"] + (1 - anc_a) * r[2]
                else:
                    coast(L[a])
    march(range(seed, len(rows)))
    for ln, r in zip(L, sr):
        ln["x"] = r[2]; ln["sl"] = 0.0; ln["anc"] = r[2]; ln["dt"] = r[3]
    march(range(seed - 1, -1, -1))
    return L


L = trace_bundle()
# глобальная метка: толще DT = MGZ (красный)
def medt(ln):
    ys = sorted(ln["tr"]); return float(np.median([DT[y, int(ln['tr'][y])] for y in ys[::5]])) if ys else 0
if medt(L[1]) > medt(L[0]):
    L = [L[1], L[0]]
common = sorted(set(L[0]["tr"]) & set(L[1]["tr"]))
dup = 100 * np.mean([abs(L[0]['tr'][y] - L[1]['tr'][y]) < 1 for y in common])
print(f"MGZ DT={medt(L[0]):.1f} MPZ DT={medt(L[1]):.1f} | точки={[len(l['tr']) for l in L]} | "
      f"MGZ левее {100*np.mean([L[0]['tr'][y]<=L[1]['tr'][y] for y in common]):.0f}% | дубль {dup:.0f}%")

COLM, COLP = (220, 0, 0), (0, 110, 230)
def draw(ov, ox, oy):
    for ln, col in ((L[0], COLM), (L[1], COLP)):
        for y, x in ln["tr"].items():
            xi = int(x)
            if 0 <= y - oy < ov.shape[0] and 0 <= xi - ox < ov.shape[1]:
                ov[y - oy, max(0, xi - ox - 1):xi - ox + 2] = col
try:
    FONT = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 24)
except Exception:
    FONT = ImageFont.load_default()
full = rgb.copy(); draw(full, 0, 0)
Image.fromarray(full[TY:BY, 0:340]).save(f"{OUT}/MK_id_full_native.png"); print("сохранён MK_id_full_native.png")
for d in (3186, 3220, 3308, 3368, 3740, 3750):
    yc = yof(d); y0, y1 = yc - 130, yc + 130
    crop = rgb[y0:y1, 70:300].copy(); draw(crop, 70, y0)
    im_c = Image.fromarray(crop).resize((230 * 3, 260 * 3), Image.NEAREST)
    ImageDraw.Draw(im_c).text((6, 4), f"{d}м MGZ=красн MPZ=син", fill=(0, 0, 0), font=FONT)
    im_c.save(f"{OUT}/MK_id_{d}.png"); print(f"  сохранён MK_id_{d}.png")

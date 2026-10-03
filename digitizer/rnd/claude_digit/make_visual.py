r"""make_visual.py — ПАПКА ПРИМЕРОВ ДЛЯ ВИЗУАЛЬНОГО РАЗБОРА (03.10, по просьбе заказчика).

Разделы (имена файлов — с буквой раздела):
  A — моя оцифровка «своими глазами» против эталона и прода (Lokachi 27 целиком, Shebel 700 — частокол);
  B — мера честности на склонах: прод идёт по туши, но нечестен при «3 px по строке» и честен «на плоскости» (§6.258),
      и пограничные случаи, которые медиана пропускает;
  C — случайные окна поля: выдача прода поверх скана (проверка глазами);
  D — притяжка к центру штриха (§6.259) — где помогала;
  E — имена по шапке (§6.260/6.261): шапка + окно с метками, ответы прода / эталона / мои / моделей;
  F — куда уходит лучшая трасса у кривых без честного кандидата (§6.257).
Пишет PNG и index.html (картинки внутри, JPEG) + README.md.

  python make_visual.py --out F:/nds/output/visual_2026-10-03
"""
import sys, json, argparse, hashlib, pickle, random, base64, io, subprocess
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from extract_nlgx import extract, NULL

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=r"F:/nds/output/visual_2026-10-03")
ap.add_argument("--seed", type=int, default=5)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
TS = Path(r"F:/nds/output/taskS")
HERE = Path(__file__).parent
ARCH = Path(r"F:\nds\projects\Archive")
SRC = {q.name: q for q in ARCH.glob("*/wlg/*.nlgx")}
IMGS = {}
for q in ARCH.glob("*/img/*"):
    if q.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
        IMGS.setdefault(q.stem, q)
try:
    FONT = ImageFont.truetype("arial.ttf", 16); FONTB = ImageFont.truetype("arialbd.ttf", 17); FTAG = ImageFont.truetype("arialbd.ttf", 18)
except Exception:
    FONT = FONTB = FTAG = ImageFont.load_default()
GREEN, RED, BLUE, ORANGE = (0, 170, 0), (225, 0, 0), (20, 60, 230), (255, 140, 0)
ENTRIES = []          # (раздел, файл, подпись)


def dense_pts(pts, gap=200):
    pts = sorted((int(round(y)), float(x)) for y, x in pts)
    out = {}
    for (y0, x0), (y1, x1) in zip(pts, pts[1:]):
        out[y0] = x0
        if 0 < y1 - y0 <= gap:
            for yy in range(y0 + 1, y1):
                out[yy] = x0 + (x1 - x0) * (yy - y0) / (y1 - y0)
    if pts:
        out[pts[-1][0]] = pts[-1][1]
    return out


def dense_curve(c):
    return dense_pts([(c["top_y"] + i, x) for i, x in enumerate(c["xs"]) if x != NULL])


def key_of(stem):
    return f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"


def prod_curves(stem):
    pd = TS / "rp_vc" / "N" / key_of(stem)
    got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
    return {c["name"]: dense_curve(c) for c in extract(str(got))["curves"]} if got else {}


def truth_curves(sheet):
    return {c["name"]: dense_curve(c) for c in extract(str(SRC[sheet]))["curves"]
            if not c["name"].startswith("DA") and sum(1 for x in c["xs"] if x != NULL) >= 50}


def plane_d(tr, gt, rows):
    from scipy.spatial import cKDTree
    ys = np.array(sorted(tr), float); xs = np.array([tr[int(y)] for y in ys], float)
    P = [np.stack([ys, xs], 1)]
    for i in range(len(ys) - 1):
        dy = ys[i + 1] - ys[i]; dx = xs[i + 1] - xs[i]
        if dy <= 30 and abs(dx) > 1:
            n = int(np.ceil(abs(dx))); t = np.arange(1, n) / n
            P.append(np.stack([ys[i] + dy * t, xs[i] + dx * t], 1))
    d, _ = cKDTree(np.concatenate(P)).query(np.array([[y, gt[y]] for y in rows], float))
    return d


def stats(tr, gt, y0=None, y1=None):
    rows = [y for y in gt if y in tr and (y0 is None or y0 <= y < y1)]
    if len(rows) < 10:
        return None
    e = np.array([abs(tr[y] - gt[y]) for y in rows]); d = plane_d(tr, gt, rows)
    return dict(n=len(rows), med=float(np.median(e)), in3=float(np.mean(e <= 3)), pmed=float(np.median(d)), pin3=float(np.mean(d <= 3)))


def caption_img(img, lines, legend=None):
    w = max(img.width, 1000)
    meas = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    wrapped = []                                   # (текст, жирный) — длинные строки переносятся по словам
    for i, l in enumerate(lines):
        f = FONTB if i == 0 else FONT
        cur = ""
        for word in l.split(" "):
            t = (cur + " " + word).strip()
            if cur and meas.textlength(t, font=f) > w - 16:
                wrapped.append((cur, i == 0)); cur = word
            else:
                cur = t
        wrapped.append((cur, i == 0))
    h = 10 + 22 * len(wrapped) + (26 if legend else 0)
    can = Image.new("RGB", (w, img.height + h), "white")
    dr = ImageDraw.Draw(can)
    y = 6
    for l, bold in wrapped:
        dr.text((8, y), l, fill=(0, 0, 0), font=FONTB if bold else FONT); y += 22
    if legend:
        x = 8
        for col, txt in legend:
            dr.line([(x, y + 10), (x + 30, y + 10)], fill=col, width=4); x += 36
            dr.text((x, y + 1), txt, fill=(0, 0, 0), font=FONT); x += dr.textlength(txt, font=FONT) + 24
    can.paste(img, (0, h))
    return can


def window(img_path, y0, y1, x0, x1, zoom, layers, marks=None):
    """layers: [(dense dict, цвет, толщина, стиль)] стиль: line | dots | dash"""
    im = Image.open(img_path).convert("RGB")
    x0 = max(0, int(x0)); x1 = min(im.width, int(x1)); y0 = max(0, int(y0)); y1 = min(im.height, int(y1))
    cr = im.crop((x0, y0, x1, y1))
    cr = cr.resize((int(cr.width * zoom), int(cr.height * zoom)), Image.LANCZOS)
    dr = ImageDraw.Draw(cr)
    for tr, col, wdt, style in layers:
        pts = [((x - x0) * zoom, (y - y0) * zoom) for y, x in sorted(tr.items()) if y0 <= y < y1]
        if style == "dots":
            for p in pts[::max(1, int(2 / zoom) or 1)]:
                dr.ellipse([(p[0] - wdt, p[1] - wdt), (p[0] + wdt, p[1] + wdt)], fill=col)
        elif style == "dash":
            for i in range(0, len(pts) - 1, 6):
                dr.line(pts[i:i + 4], fill=col, width=wdt)
        elif len(pts) > 1:
            dr.line(pts, fill=col, width=wdt)
    for (x, y, txt, col) in (marks or []):
        px, py = (x - x0) * zoom, (y - y0) * zoom
        dr.rectangle([px + 8, py - 12, px + 14 + dr.textlength(txt, font=FTAG), py + 12], fill="white", outline=col, width=2)
        dr.text((px + 11, py - 11), txt, fill=col, font=FTAG)
        dr.line([(px, py), (px + 8, py)], fill=col, width=2)
    return cr


def save(sec, name, img, cap):
    p = OUT / name
    img.save(p)
    ENTRIES.append((sec, name, cap))
    print(f"  {name}")


def f1(v):
    return "—" if v is None else f"{v:.1f}"


# ═════════════════════════ A — моя оцифровка ═════════════════════════
print("A — моя оцифровка")
LOK = "Lokachi_027_(2009.10.20)_RK_0780-0864_M200_D_1"
lok_img = IMGS[LOK]; lok_sheet = LOK + ".nlgx"
T = truth_curves(lok_sheet); P = prod_curves(LOK)
MINE = {"GK1 DA1 SA1": dense_pts(json.load(open(HERE / "gk_all.json"))), "NGK1 DA1 SA2": dense_pts(json.load(open(HERE / "ngk_all.json")))}
for nm, ru, wins in (("GK1 DA1 SA1", "ГК", [(4180, 4520), (5440, 5780), (6440, 6780), (7680, 8020)]),
                     ("NGK1 DA1 SA2", "НГК", [(4440, 4780), (5440, 5780), (6940, 7280), (7680, 8020)])):
    for k, (y0, y1) in enumerate(wins, 1):
        gt, me, pr = T[nm], MINE[nm], P.get(nm, {})
        xs = [v for d in (gt, me, pr) for y, v in d.items() if y0 <= y < y1]
        img = window(lok_img, y0, y1, min(xs) - 40, max(xs) + 40, 1.6,
                     [(gt, GREEN, 3, "line"), (pr, RED, 2, "dots"), (me, BLUE, 2, "line")])
        sm, sp = stats(me, gt, y0, y1), stats(pr, gt, y0, y1)
        cap = [f"Lokachi 27, РК 2009 — {ru}, строки {y0}–{y1} (окно {k} из 4)",
               f"|Δx| по строке, медиана: я {f1(sm and sm['med'])} px, прод {f1(sp and sp['med'])} px;  "
               f"строк в 3 px на плоскости: я {sm['pin3']:.0%}, прод {sp['pin3']:.0%}" if sm and sp else ""]
        save("A", f"A{'1' if ru == 'ГК' else '2'}{k}_lokachi_{'GK' if ru == 'ГК' else 'NGK'}_{y0}.png",
             caption_img(img, cap, [(GREEN, "эталон эксперта"), (RED, "прод"), (BLUE, "моя оцифровка (вершины на глаз)")]), " | ".join(cap))
sm_all = {nm: stats(MINE[nm], T[nm]) for nm in MINE}; sp_all = {nm: stats(P[nm], T[nm]) for nm in MINE}
SHEB = "Shebel_700_1987_11_19_RK_1588_2388_200_D_1"
Ts = truth_curves(SHEB + ".nlgx"); Ps = prod_curves(SHEB)
me = dense_pts(json.load(open(HERE / "sheb_pts_a.json")))
gt = Ts["NGK1 DA1 SA2"]; pr = Ps.get("NGK1 DA1 SA2", {})
img = window(IMGS[SHEB], 19590, 19810, 1250, 1760, 2.0, [(gt, GREEN, 3, "line"), (pr, RED, 2, "dots"), (me, BLUE, 2, "line")])
sm, sp = stats(me, gt, 19600, 19800), stats(pr, gt, 19600, 19800)
cap = ["Shebel 700, РК 1987 — НГК, частокол, строки 19600–19800",
       f"|Δx| медиана: я {sm['med']:.1f} px, прод {sp['med']:.1f} px; на плоскости в 3 px: я {sm['pin3']:.0%}, прод {sp['pin3']:.0%} "
       f"— прод здесь ведёт соседнюю кривую"]
save("A", "A3_shebel_NGK_zigzag.png", caption_img(img, cap, [(GREEN, "эталон"), (RED, "прод"), (BLUE, "моя оцифровка")]), " | ".join(cap))

# ═════════════════════════ B — мера на склонах ═════════════════════════
print("B — мера на склонах")
G = pickle.load(open(TS / "tol_d2_gained_field.pkl", "rb"))
rng = random.Random(a.seed)
pick = rng.sample(G, 10)
for k, (sh, gname, oname, mraw, mpl) in enumerate(pick, 1):
    stem = SRC[sh].stem
    gt = truth_curves(sh)[gname]; pr = prod_curves(stem)[oname]
    rows = sorted(y for y in gt if y in pr)
    e = np.array([abs(pr[y] - gt[y]) for y in rows]); d = plane_d(pr, gt, rows)
    good = ((e > 3) & (d <= 3)).astype(float)
    W = 260
    cs = np.concatenate([[0], np.cumsum(good)])
    i0 = int(np.argmax(cs[W:] - cs[:-W])) if len(rows) > W else 0
    y0 = rows[i0]; y1 = y0 + W
    xs = [v for dd in (gt, pr) for y, v in dd.items() if y0 <= y < y1]
    img = window(IMGS[stem], y0, y1, min(xs) - 50, max(xs) + 50, 1.8, [(gt, GREEN, 3, "line"), (pr, RED, 2, "dots")])
    st_all = stats(pr, gt); st_w = stats(pr, gt, y0, y1)
    cap = [f"{stem[:60]} — эталон «{gname.split()[0]}», выдача «{oname.split()[0]}», строки {y0}–{y1}",
           f"вся кривая: |Δx| медиана {mraw:.1f} px → НЕЧЕСТНА по строке; на плоскости медиана {mpl:.1f} px → честна; "
           f"доля строк в 3 px: по строке {st_all['in3']:.0%}, на плоскости {st_all['pin3']:.0%}",
           f"в окне: по строке в 3 px {st_w['in3']:.0%}, на плоскости {st_w['pin3']:.0%}"]
    save("B", f"B{k:02d}_{stem[:28]}_{gname.split()[0]}.png".replace(" ", "_").replace(",", ""),
         caption_img(img, cap, [(GREEN, "эталон"), (RED, "выдача прода")]), " | ".join(cap))

# ═════════════════════════ C — случайные окна поля ═════════════════════════
print("C — случайные окна поля")
COLS = [(225, 0, 0), (20, 60, 230), (200, 0, 200), (0, 150, 150), (230, 120, 0), (120, 60, 0), (0, 0, 0), (150, 150, 0)]
names = [l.strip() for l in open(TS / "wellmap_sheets.txt", encoding="utf-8") if l.strip()]
rng = random.Random(3); rng.shuffle(names)
made = 0
for sh in names:
    if made >= 12:
        break
    stem = SRC[sh].stem if sh in SRC else None
    if not stem or stem not in IMGS:
        continue
    Gt = truth_curves(sh); Wp = {k: v for k, v in prod_curves(stem).items() if v and not k.startswith("DA")}
    if not Gt or not Wp:
        continue
    ys = sorted(set().union(*[set(g) for g in Gt.values()]))
    y0 = rng.randint(ys[0], max(ys[0], ys[-1] - 1600)); y1 = y0 + 1600
    xs = [x for g in list(Gt.values()) + list(Wp.values()) for y, x in g.items() if y0 <= y < y1]
    if not xs:
        continue
    layers = [(g, GREEN, 2, "dash") for g in Gt.values()]
    marks, txt = [], []
    for j, (k, w) in enumerate(sorted(Wp.items())):
        col = COLS[j % len(COLS)]
        layers.append((w, col, 1, "dots"))
        pts = [(y, x) for y, x in sorted(w.items()) if y0 <= y < y1]
        if pts:
            yy, xx = pts[len(pts) // 5]
            marks.append((xx, yy, k.split()[0], col))
        s_ = stats(w, Gt[k], y0, y1) if k in Gt else None
        txt.append(f"{k.split()[0]}: " + (f"|Δx| мед. {s_['med']:.0f} px" if s_ else "нет эталона в окне"))
    img = window(IMGS[stem], y0, y1, min(xs) - 60, max(xs) + 60, 0.55, layers, marks)
    cap = [f"{stem[:70]} — строки {y0}–{y1} (случайное окно)", "выдача против эталона того же имени: " + "; ".join(txt)]
    save("C", f"C{made + 1:02d}_{stem[:34]}.png".replace(" ", "_").replace(",", ""),
         caption_img(img, cap, [(GREEN, "эталон (пунктир)"), (RED, "выдача прода: цвет на кривую, подпись — имя")]), " | ".join(cap))
    made += 1

# ═════════════════════════ D — притяжка к центру штриха ═════════════════════════
print("D — притяжка")
from PIL import Image as _I
gray = np.asarray(_I.open(lok_img).convert("L"), np.int16)
paper = np.percentile(gray[:, ::3], 90, axis=1)


def snap(tr, R=8, thr=30):
    out = {}
    for y, xm in tr.items():
        row = gray[y]; c = int(round(xm)); lo, hi = max(0, c - R), min(len(row), c + R + 1)
        lim = paper[y] - thr; ink = row[lo:hi] < lim
        if not ink.any():
            out[y] = xm; continue
        idx = np.flatnonzero(ink) + lo; p0 = int(idx[np.argmin(np.abs(idx - xm))]); l = r = p0
        while l > 0 and row[l - 1] < lim and xm - l < 3 * R:
            l -= 1
        while r < len(row) - 1 and row[r + 1] < lim and r - xm < 3 * R:
            r += 1
        out[y] = (l + r) / 2.0
    return out


for nm, ru, (y0, y1) in (("GK1 DA1 SA1", "ГК", (5440, 5780)), ("NGK1 DA1 SA2", "НГК", (6940, 7280))):
    gt, pr = T[nm], P[nm]; ps = snap(pr)
    xs = [v for d in (gt, pr) for y, v in d.items() if y0 <= y < y1]
    img = window(lok_img, y0, y1, min(xs) - 40, max(xs) + 40, 1.6, [(gt, GREEN, 3, "line"), (pr, RED, 2, "dots"), (ps, ORANGE, 2, "dots")])
    s1, s2 = stats(pr, gt), stats(ps, gt)
    cap = [f"Lokachi 27 — {ru}, притяжка выдачи к центру штриха (§6.259), строки {y0}–{y1}",
           f"вся кривая: |Δx| медиана прод {s1['med']:.1f} → с притяжкой {s2['med']:.1f} px; строк в 3 px {s1['in3']:.0%} → {s2['in3']:.0%}",
           "⚠ на всём поле притяжка ВРЕДИТ (−27): на сплетениях «ближайший ран» — чужая тушь или сетка; закрыта"]
    save("D", f"D{1 if ru == 'ГК' else 2}_lokachi_{'GK' if ru == 'ГК' else 'NGK'}_snap.png",
         caption_img(img, cap, [(GREEN, "эталон"), (RED, "прод"), (ORANGE, "прод + притяжка")]), " | ".join(cap))
del gray

# ═════════════════════════ E — имена по шапке ═════════════════════════
print("E — имена")
VN = Path(r"F:/nds/output/vlm_names")
meta = json.load(open(VN / "meta.json", encoding="utf-8"))
ANS = {}
for tag, f in (("gemma-4", "answers_google_gemma-4-26b-a4b.json"), ("Qwen2.5-VL", "answers_qwen2.5-vl-7b-instruct.json"),
               ("Claude", "answers_claude.json")):
    if (VN / f).exists():
        ANS[tag] = {k: (v.get("answer") or {}) for k, v in json.load(open(VN / f, encoding="utf-8")).items()}
for j in ["05", "07", "25", "13", "02", "06"]:
    m = meta[j]
    hd = Image.open(VN / f"{j}_header.png").convert("RGB"); wn = Image.open(VN / f"{j}_window.png").convert("RGB")
    s = 900 / hd.height
    hd = hd.resize((max(1, int(hd.width * s)), 900), Image.LANCZOS)
    s2 = min(1.0, 900 / wn.height, 1100 / wn.width)
    wn = wn.resize((int(wn.width * s2), int(wn.height * s2)), Image.LANCZOS)
    can = Image.new("RGB", (hd.width + wn.width + 10, 900), "white")
    can.paste(hd, (0, 0)); can.paste(wn, (hd.width + 10, 0))
    rows = []
    for lab in sorted(m["labels"], key=lambda t: int(t[1:])):
        tr = m["truth"][lab]
        parts = [f"{lab}: эталон {tr.split()[0] if tr else '—'}", f"прод {m['labels'][lab].split()[0]}"]
        for tag in ("Claude", "gemma-4", "Qwen2.5-VL"):
            v = ANS.get(tag, {}).get(j, {}).get(lab)
            if v:
                parts.append(f"{tag} {str(v).split()[0]}")
        rows.append(", ".join(parts))
    cap = [f"{m['sheet'][:70]} [{m['kind']}] — слева шапка (легенда), справа окно с метками кривых выдачи"] + rows
    save("E", f"E_{j}_{m['sheet'][:30]}.png".replace(" ", "_").replace(",", ""), caption_img(can, cap), " | ".join(cap))

# ═════════════════════════ F — уходы лучшей трассы ═════════════════════════
print("F — уходы")
tmp = OUT / "_dep_tmp"
r = subprocess.run([sys.executable, "-W", "ignore", str(HERE.parent / "_dep_viz.py"), "--every", "60", "--offset", "7", "--out", str(tmp)],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
for line in r.stdout.splitlines():
    if ".png:" not in line:
        continue
    fn, rest = line.split(".png:", 1)
    p = tmp / (fn + ".png")
    if not p.exists():
        continue
    img = Image.open(p).convert("RGB")
    cap = [f"{fn[3:60]}", rest.strip()[:160]]
    save("F", f"F_{fn[:40]}.png".replace(" ", "_").replace(",", ""),
         caption_img(img, cap, [(GREEN, "эталон своей кривой"), (RED, "лучшая трасса (прод или декодер)"), ((0, 90, 255), "другие кривые эталона"),
                                ((255, 0, 255), "строка ухода")]), " | ".join(cap))
for q in tmp.glob("*.png"):
    q.unlink()
tmp.rmdir()

# ═════════════════════════ индекс ═════════════════════════
SEC = {
    "A": ("A. Моя оцифровка «своими глазами» (без скриптов проекта)",
          f"Вершины ломаной ставил я сам по вырезкам скана с сеткой, как эксперт. Lokachi 27 — обе кривые целиком (ГК 671 вершина, "
          f"НГК 699). Вся кривая: ГК — |Δx| медиана я {sm_all['GK1 DA1 SA1']['med']:.1f} / прод {sp_all['GK1 DA1 SA1']['med']:.1f} px, "
          f"на плоскости в 3 px я {sm_all['GK1 DA1 SA1']['pin3']:.0%} / прод {sp_all['GK1 DA1 SA1']['pin3']:.0%}; "
          f"НГК — {sm_all['NGK1 DA1 SA2']['med']:.1f} / {sp_all['NGK1 DA1 SA2']['med']:.1f} px, {sm_all['NGK1 DA1 SA2']['pin3']:.0%} / "
          f"{sp_all['NGK1 DA1 SA2']['pin3']:.0%}. На чистом листе прод по туши точнее меня; на частоколе Shebel я держу свою кривую, "
          f"прод уходит на соседнюю."),
    "B": ("B. Мера честности на склонах (§6.258)",
          "Случайные 10 кривых поля, которые прод провёл по туши, но которые нечестны при нынешней мере «3 px по строке» и честны "
          "«на плоскости» (расстояние до линии трассы). Окно — где таких строк больше всего. На крутом склоне центр штриха в строке "
          "отходит от эталона на 4–8 px, хотя по вертикали это меньше строки. Поле: 1192 → 1617 безымянных, 861 → 1165 именных."),
    "C": ("C. Случайные окна поля — проверка глазами",
          "12 случайных листов поля, случайное окно 1600 строк. Эталон — зелёный пунктир, выдача прода — цветные точки, подпись — имя "
          "кривой выдачи. Что видно: где прод идёт по своей кривой — точен; провалы — кривая выдачи на линии сетки, перепутанные имена, "
          "россыпь на частом зигзаге."),
    "D": ("D. Притяжка к центру штриха (§6.259) — закрыта",
          "На листе Lokachi притяжка выдачи к центру ближайшего рана туши убирала смещение 6 → 1 px. На всём поле — −27 честных "
          "(вариант «только тёмная тушь» +5, на непросмотренных листах +1). Один лист — не выборка."),
    "E": ("E. Имена по шапке листа (§6.260/6.261)",
          "Слева — шапка (легенда зондов, образцы линий, цвета), справа — окно с метками #1…#K на кривых выдачи. Под картинкой — для каждой "
          "метки имя эталона (истина), имя прода и ответы: мой (Claude, где есть) и локальных моделей. Итог пилота на 50 листах: "
          "прод 37% на перепутанных, gemma-4 39%, Qwen2.5-VL 30%; я — 6 из 9 на трудных листах."),
    "F": ("F. Куда уходит лучшая трасса у кривых без честного кандидата (§6.257)",
          "Для кривой эталона (зелёная) — лучшая из всех трасс прода и декодера (красная) и момент ухода (пурпурная риска слева). "
          "Уходов у касания с другой кривой ≈ 10%; чаще трасса уходит в фон или на кривую далеко в стороне."),
}
md = ["# Примеры для визуального разбора — 03.10.2026", "",
      "Открыть `index.html` — все картинки с подписями на одной странице. Ниже — список файлов по разделам.", ""]
html = ["<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Визуальный разбор 03.10</title><style>body{font-family:Arial,sans-serif;margin:16px;background:#fafafa;color:#111}"
        "h1{font-size:22px}h2{font-size:19px;margin-top:34px;border-bottom:2px solid #ccc}p.d{max-width:1100px}"
        "figure{margin:14px 0 26px 0}img{max-width:100%;height:auto;border:1px solid #ddd;background:#fff}"
        "figcaption{font-size:13px;color:#333;max-width:1200px}nav a{margin-right:14px}</style></head><body>",
        "<h1>Примеры для визуального разбора — 03.10.2026</h1><nav>" +
        "".join(f"<a href='#{s}'>{SEC[s][0].split('.')[0]}</a>" for s in SEC) + "</nav>"]
for s in SEC:
    title, desc = SEC[s]
    md += [f"## {title}", "", desc, ""]
    html.append(f"<h2 id='{s}'>{title}</h2><p class='d'>{desc}</p>")
    for sec, fn, cap in ENTRIES:
        if sec != s:
            continue
        md.append(f"- `{fn}` — {cap}")
        im = Image.open(OUT / fn).convert("RGB")
        if im.width > 1400:
            im = im.resize((1400, int(im.height * 1400 / im.width)), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=78)
        html.append(f"<figure><img src='data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}' alt='{fn}'>"
                    f"<figcaption>{fn}</figcaption></figure>")
    md.append("")
html.append("</body></html>")
(OUT / "README.md").write_text("\n".join(md), encoding="utf-8")
(OUT / "index.html").write_text("\n".join(html), encoding="utf-8")
print(f"★ готово: {len(ENTRIES)} картинок, {OUT}")

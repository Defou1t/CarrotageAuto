r"""_label_offset.py — ГДЕ ЭТАЛОН ОТНОСИТЕЛЬНО ЦЕНТРА ШТРИХА ПО КРУТИЗНЕ (02.10, к §6.257).

§6.257: карта декодера на крутых строках (> 2 px/строку) видит лишь 0.51–0.65 против 0.81–0.92 на пологих. Цель обучения —
узкий гаусс (sigma 1.5) у x эталона в КАЖДОЙ строке, а эталон — линейная интерполяция вершин полилинии (шаг 6–23 строки).
Если на крутых участках x эталона систематически не в центре рана туши, сеть получает противоречивые цели (тушь есть,
а цель — «нет») и учится гасить отклик на склонах. Здесь по строкам эталона: ПОЛНЫЙ ран туши (вход декодера `_band`,
порог 20), содержащий x эталона или ближайший в ±R px; смещение «центр рана − эталон» и ширина рана — по корзинам
крутизны. Только чтение, выдачу не меняет.

  _label_offset.py --every 9
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M, imaging as im, rowdec as RD
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--every", type=int, default=9)
ap.add_argument("--radius", type=int, default=12)
ap.add_argument("--thr", type=int, default=20)
ap.add_argument("--rowstep", type=int, default=2)
a = ap.parse_args()
TS = Path(a.ts); P = DEFAULT.cv
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
BINS = ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 1e9))
OFF = {b: [] for b in BINS}; WID = {b: [] for b in BINS}; MISS = {b: [0, 0] for b in BINS}
VERT = {b: [] for b in BINS}           # ★ смещение по вертикали: на крутом склоне ошибка по x ≈ наклон × сдвиг по y
SIG = []                                # ★ (лист, знаковый наклон dx/dy, знаковое «центр − эталон») — систематический сдвиг по y?
for sh in lst("wellmap_sheets.txt")[::a.every] + lst("holdoutA_sheets.txt")[::a.every]:
    q = SRC.get(sh)
    if not q:
        continue
    stem = q.stem; key = f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    f = Path(a.cache) / f"{key}.pkl"
    if not f.exists():
        continue
    v = pickle.load(open(f, "rb"))
    G = {c["name"]: dense(c) for c in extract(str(q))["curves"]
         if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    if not G:
        continue
    rgb = im.load_rgb(v["image"])
    dark = RD._band(rgb, P, 0, rgb.shape[1])
    H, W = dark.shape
    for g, gt in G.items():
        for y in sorted(gt)[::a.rowstep]:
            if not (1 <= y < H - 1):
                continue
            xt = gt[y]
            sl = abs(gt.get(y + 1, xt) - gt.get(y - 1, xt)) / 2.0
            b = next(bb for bb in BINS if bb[0] <= sl < bb[1])
            MISS[b][1] += 1
            c = int(round(xt))
            if not (0 <= c < W):
                MISS[b][0] += 1; continue
            row = dark[y] >= a.thr
            lo, hi = max(0, c - a.radius), min(W, c + a.radius + 1)
            if not row[lo:hi].any():
                MISS[b][0] += 1; continue
            # ближайший к x эталона пиксель туши в окне, затем ПОЛНЫЙ ран вокруг него
            idx = np.flatnonzero(row[lo:hi]) + lo
            p0 = int(idx[np.argmin(np.abs(idx - xt))])
            l = p0
            while l > 0 and row[l - 1]:
                l -= 1
            r = p0
            while r < W - 1 and row[r + 1]:
                r += 1
            OFF[b].append((l + r) / 2.0 - xt); WID[b].append(r - l + 1)
            if r - l + 1 <= 120:
                SIG.append((sh, (gt.get(y + 1, xt) - gt.get(y - 1, xt)) / 2.0, (l + r) / 2.0 - xt))
    del rgb, dark
print(f"★ ЭТАЛОН ПРОТИВ ЦЕНТРА ПОЛНОГО РАНА ТУШИ (каждый {a.every}-й лист поля и сорта A, каждая {a.rowstep}-я строка эталона):")
print("   наклон px/строку | строк | туши нет в ±R | |центр − эталон|: медиана, 75%, доля > 3 px | ширина рана: медиана")
for b in BINS:
    o = np.abs(np.array(OFF[b])); w = np.array(WID[b])
    if not len(o):
        continue
    print(f"   {b[0]:>4}–{b[1] if b[1] < 1e8 else '∞':<4} | {MISS[b][1]:>8} | {MISS[b][0] / max(1, MISS[b][1]):.3f} | "
          f"{np.median(o):.1f}, {np.percentile(o, 75):.1f}, {np.mean(o > 3):.2f} | {np.median(w):.0f}")

# ★ 03.10: систематический сдвиг эталона по вертикали. Если эталон сдвинут на d строк вниз, то в строке y он показывает x(y − d),
#   а центр штриха — x(y): центр − эталон ≈ d · (dx/dy). Регрессия по крутым строкам (|наклон| 1–20) без свободного члена
#   и со свободным; по листам — разброс d.
S = np.array([(sl, off) for _, sl, off in SIG if 1.0 <= abs(sl) <= 20.0])
if len(S):
    sl, off = S[:, 0], S[:, 1]
    d0 = float((sl * off).sum() / (sl * sl).sum())
    A = np.stack([np.ones_like(sl), sl], 1); c, d1 = np.linalg.lstsq(A, off, rcond=None)[0]
    # устойчиво: медиана off/sl
    dm = float(np.median(off / sl))
    print(f"\n★ СДВИГ ЭТАЛОНА ПО ВЕРТИКАЛИ (строки с |наклон| 1–20, {len(S)}): МНК без члена d = {d0:+.3f} строки; "
          f"со свободным: d = {d1:+.3f}, сдвиг по x {c:+.2f} px; медиана (центр − эталон)/наклон = {dm:+.3f}")
    for lo_, hi_ in ((1, 2), (2, 4), (4, 8), (8, 20)):
        m = (np.abs(sl) >= lo_) & (np.abs(sl) < hi_)
        if m.any():
            print(f"   |наклон| {lo_}–{hi_}: строк {int(m.sum())}; медиана (центр − эталон)/наклон {np.median(off[m] / sl[m]):+.3f}; "
                  f"при наклоне > 0: медиана смещения {np.median(off[m & (sl > 0)]):+.2f} px, при < 0: {np.median(off[m & (sl < 0)]):+.2f} px")
    per = {}
    for shn, slv, ofv in SIG:
        if 1.0 <= abs(slv) <= 20.0:
            per.setdefault(shn, []).append(ofv / slv)
    dd = np.array([np.median(v) for v in per.values() if len(v) >= 200])
    if len(dd):
        print(f"   по листам ({len(dd)} с ≥ 200 крутых строк): медиана d {np.median(dd):+.3f}, квартили {np.percentile(dd, 25):+.3f}…"
              f"{np.percentile(dd, 75):+.3f}; |d| > 0.3 у {np.mean(np.abs(dd) > 0.3):.0%}")

r"""_drift_anatomy.py — АНАТОМИЯ УХОДА ТРАССЫ С ЭТАЛОНА НА ЧУЖУЮ ЛИНИЮ (§6.219, 25.09).

Выборка та же, что в `_ruling_share.py` (§6.218): не взятые кривые многокривых треков (выдача `--dir`, по умолчанию прод RA),
для каждой — ближайшая по медиане кривая выдачи трека; строки УХОДА — где в выдаче есть значение (не мост `dense`), есть
эталон кривой и трасса дальше 3 px от ВСЕХ эталонов трека. Три разреза, веса — строки ухода:

  A. ЧТО ПОД ТРАССОЙ: эталон другого трека (±3 px) / копия своего эталона со сдвигом (отрезок ≥ 40 строк: IQR сдвига ≤ 4 px,
     эталон после снятия линейного тренда не ровный, σ ≥ 3 px; отдельно сдвиг ≈ ширине шкалы ±10% — перевынос) /
     вертикаль-линейка (сплошная или пунктирная, как в `_ruling_share`) / другая тушь / туши нет.
     Цвет — ТОЛЬКО на треках, где эталоны трека разного преобладающего цвета (на одноцветных совпадение цвета тривиально):
     преобладающий цвет туши под трассой против преобладающего цвета своего эталона.
  B. КАК НАЧАЛСЯ ОТРЕЗОК УХОДА (≥ 20 строк). Каждая строка выдачи получает состояние: СВОЙ эталон (≤ 3 px) / СОСЕДНИЙ эталон
     трека / УХОД / НЕТ ЭТАЛОНА. Начало ухода классифицируется от ПОСЛЕДНЕЙ строки на линии (свой или соседний эталон):
     «прыжок» — строки подряд (≤ 2) и сдвиг относительно покинутой линии изменился > 3 px за шаг; «сползание» — ≤ 3 px;
     «через пропуск выдачи» — между ними > 2 строк без значения; «трасса начинается не на эталоне»; «эталон начинается,
     трасса уже на другой линии»; «продолжение ухода после пропуска».
  C. РАЗЛИЧИМЫ ЛИ ЛИНИИ ПО СТИЛЮ на отрезке: сплошность (доля строк с тушью в ±3 px — одинаковый радиус у трассы и
     эталона) и толщина ПОПЕРЁК линии (горизонтальный ран / √(1+наклон²), строки с |наклон| ≤ 2 px/строку).

⚠ Поправки ревизии агентом (25.09) к первым редакциям `_ink_identity.py` / `_drift_onset.py`: двойной счёт строк (две не взятые
кривые трека на одну выдачу — 27% веса) снят: строка (выдача, y) отдаётся кривой с меньшей медианой; начало ухода — от линии,
которую трасса ФАКТИЧЕСКИ покинула (прежде «с начала» на 90% было «ушла с соседнего эталона»); цвет — только на
разноцветных треках; эталон притягивается к туши в ±3 px (эталон — ломаная эксперта, на 1–2 px мимо штриха).
"""
import sys, argparse, pickle, hashlib, json, math
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from dataset_build import find_image
from auto import meta as M, imaging as im, trace2d as T2
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--rows", default="pick_learn_honest2.pkl.dump")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--dir", default="ab_slot/RA")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--frac", type=float, default=0.6)
ap.add_argument("--frac-dot", type=float, default=0.3)
ap.add_argument("--block", type=int, default=1000)
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--dump", default="")
a = ap.parse_args()
TS = Path(a.ts)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def st(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    return float(np.median([abs(tr[y] - gt[y]) for y in com])), len(com) / max(1, len(gt))


def match(rows, cols, ok):
    pair = {}

    def try_(r, seen):
        for c in cols:
            if not ok.get((r, c)) or c in seen:
                continue
            seen.add(c)
            if c not in pair or try_(pair[c], seen):
                pair[c] = r
                return True
        return False
    for r in rows:
        try_(r, set())
    return {r: c for c, r in pair.items()}


RAW = {}


def read_out(root, sh):
    stem = Path(sh).stem
    pd = TS / root / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    if not got:
        return None
    cs = [c for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"]
    RAW.clear()
    for c in cs:
        RAW[c["name"]] = {c["top_y"] + i for i, x in enumerate(c["xs"]) if x != NULL}
    return {c["name"]: dense(c) for c in cs}


def dil(m, r=2):
    """Горизонтальное расширение без заворота через край (вместо np.roll)."""
    o = m.copy()
    for s in range(1, r + 1):
        o[:, s:] |= m[:, :-s]
        o[:, :-s] |= m[:, s:]
    return o


trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q
sheets = sorted({r[0] for r in trk})[a.offset::a.every]
A = Counter(); AC = Counter(); B = Counter(); BN = Counter(); C = Counter(); D = Counter(); E = Counter(); REC = []
ncurves = 0
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    W = read_out(a.dir, sh)
    if W is None:
        continue
    G = extract(str(q))
    gts = {c["name"]: dense(c) for c in G["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    ax = {d["name"]: d["axis_idx"] for d in G.get("curve_desc", [])}
    swid = {s["idx"]: max(1, s["x_right"] - s["x_left"]) for s in G.get("scale_axes", [])}
    tm = smap.get(sh, {})
    bytr = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None:
            bytr[t].append(nm)
    todo = []
    for t, ns in bytr.items():
        if (sh, t) not in KOF or len(ns) < 2:
            continue
        Wt = [k for k in W if tm.get(k) == t and W[k]]
        if not Wt:
            continue
        ok = {(g, k): HON(*st(W[k], gts[g])) for g in ns for k in Wt}
        mt = match(ns, Wt, ok)
        cand = defaultdict(list)
        for g in ns:
            if g in mt:
                continue
            best = None
            for k in Wt:
                m, c = st(W[k], gts[g])
                if m is not None and (best is None or m < best[0]):
                    best = (m, k)
            if best and best[0] > 3:
                cand[best[1]].append((best[0], g))
        for kname, lst in cand.items():
            w = W[kname]; raw = RAW.get(kname, set())
            claimed = set()
            for med, g in sorted(lst):
                gt = gts[g]
                offall = {y for y in w if y in raw and y in gt
                          and all(not (y in gts[o] and abs(w[y] - gts[o][y]) <= 3) for o in ns)}
                mine = offall - claimed          # ★ строка (выдача, y) — одной кривой, с меньшей медианой
                claimed |= mine
                if len(mine) >= 30:
                    todo.append((t, g, kname, offall, mine, ns))
    if not todo:
        continue
    img = find_image(q)
    if not img:
        continue
    stem = Path(sh).stem
    pdir = TS / a.dir / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    pj = next(iter(pdir.glob("*_pick.json")), None)
    SRCK = {}
    if pj:
        pk = json.loads(pj.read_text(encoding="utf-8"))
        tdec = {t["track"]: t["dec"] for t in pk.get("tracks", [])}
        for sl in pk.get("slots", []):
            SRCK[sl["name"]] = sl.get("src") or (("dec" if sl["base"] == "prod" else "prod") if sl["flip"] else sl["base"])
        for k in W:
            if k not in SRCK:
                SRCK[k] = "dec" if tdec.get(tm.get(k)) else "prod"
    rgb = im.load_rgb(str(img))
    blk = T2._color_fg(rgb, "black", DEFAULT.cv)
    H, Wd = blk.shape
    sm = im.structure_mask(rgb, DEFAULT.cv)
    COL = {cn: (cm & ~sm) for cn, cm in im.color_channels(rgb, DEFAULT.cv).items()}
    ink = blk.copy()
    for cm in COL.values():
        ink |= cm
    nb = (H + a.block - 1) // a.block
    rul = np.zeros((nb, Wd), bool)
    for b in range(nb):
        mb = blk[b * a.block:(b + 1) * a.block].mean(0)
        rul[b] = mb >= a.frac_dot
    rul2 = dil(rul); ink2 = dil(ink)
    # ★ высота связной компоненты туши (для «другой туши»: длинная линия или короткий объект)
    import cv2
    _nl, _lab, _sts, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    comp_h = _sts[:, cv2.CC_STAT_HEIGHT]
    sax = {s_["idx"]: (s_["x_left"], s_["x_right"]) for s_ in G.get("scale_axes", [])}

    def snap(y, x, r=3):
        """Ближайший пиксель туши в строке в ±r px от x, иначе None."""
        if not (0 <= y < H):
            return None
        xi = int(round(x)); lo = max(0, xi - r); hi = min(Wd, xi + r + 1)
        if lo >= hi:
            return None
        idx = np.flatnonzero(ink[y, lo:hi])
        if not len(idx):
            return None
        return lo + int(idx[np.argmin(np.abs(lo + idx - xi))])

    def colour(y, x):
        if x is None:
            return None
        lo = max(0, x - 1); hi = min(Wd, x + 2)
        best = max(((int(m[y, lo:hi].sum()), cn) for cn, m in COL.items()), default=(0, None))
        if best[0] > 0:
            return best[1]
        return "black" if blk[y, lo:hi].any() else None

    def hrun(y, x):
        l = r = x
        row = ink[y]
        while l > 0 and row[l - 1]:
            l -= 1
        while r < Wd - 1 and row[r + 1]:
            r += 1
        return r - l + 1

    # преобладающий цвет каждой кривой эталона трека (по притянутым точкам, каждая 5-я строка)
    domc = {}
    for t, g, kname, offall, mine, ns in todo:
        for o in ns:
            if o in domc:
                continue
            cc = Counter(colour(y, snap(y, gts[o][y])) for y in sorted(gts[o])[::5] if 0 <= y < H)
            cc.pop(None, None)
            domc[o] = cc.most_common(1)[0][0] if cc else None
    for t, g, kname, offall, mine, ns in todo:
        ncurves += 1
        src = SRCK.get(kname, "?")
        w = W[kname]; gt = gts[g]; raw = RAW.get(kname, set())
        others = [o for o in gts if o not in ns]
        swd = swid.get(ax.get(g, -1), 0)
        xl, xr = sax.get(ax.get(g, -1), (None, None))
        ysg = sorted(gt)
        for y1, y2 in zip(ysg, ysg[1:]):
            if y2 - y1 == 1:
                st_ = abs(gt[y2] - gt[y1])
                E["эталон: шаг > 30 px за строку"] += st_ > 30; E["эталон: шаг > 10 px за строку"] += st_ > 10
                E["эталон: строк-шагов"] += 1
        multicol = len({domc.get(o) for o in ns if domc.get(o)}) >= 2
        # состояния строк выдачи
        ys = sorted(y for y in w if y in raw)
        stt = {}
        for y in ys:
            if y in offall:
                stt[y] = ("off", None)
            elif y in gt and abs(w[y] - gt[y]) <= 3:
                stt[y] = ("own", g)
            else:
                o = next((o for o in ns if o != g and y in gts[o] and abs(w[y] - gts[o][y]) <= 3), None)
                stt[y] = ("nb", o) if o else ("nogt", None)
        # отрезки ухода: подряд идущие строки выдачи в состоянии off, пропуск ≤ 5 строк без значения
        segs = []; cur = []
        for i, y in enumerate(ys):
            if stt[y][0] == "off" and cur and y - cur[-1][1] <= 6:
                cur.append((i, y))
            else:
                if cur:
                    segs.append(cur)
                cur = [(i, y)] if stt[y][0] == "off" else []
        if cur:
            segs.append(cur)
        lab = {}
        for sg in segs:
            rr = [y for _, y in sg]
            wt = sum(1 for y in rr if y in mine)
            if not wt:
                continue
            # A: копия эталона со сдвигом
            if len(rr) >= 40:
                d = np.array([w[y] - gt[y] for y in rr]); gx = np.array([gt[y] for y in rr], float)
                yy = np.array(rr, float)
                res = gx - np.polyval(np.polyfit(yy, gx, 1), yy)
                iqr = np.percentile(d, 75) - np.percentile(d, 25)
                wv = np.array([w[y] for y in rr], float)
                wres = wv - np.polyval(np.polyfit(yy, wv, 1), yy)
                cor = float(np.corrcoef(wres, res)[0, 1]) if res.std() > 0 and wres.std() > 0 else 0.0
                if res.std() >= 3 and cor >= 0.8 and not (iqr <= 4 and abs(np.median(d)) > 3):
                    for y in rr:
                        lab[y] = "копия эталона по форме (corr ≥ 0.8, сдвиг гуляет)"
                if iqr <= 4 and res.std() >= 3 and abs(np.median(d)) > 3:
                    md = abs(float(np.median(d)))
                    cat = ("копия эталона со сдвигом ≈ ширине шкалы (перевынос)" if swd and abs(md - swd) <= 0.1 * swd
                           else "копия эталона со сдвигом (дубль/повтор)")
                    for y in rr:
                        lab[y] = cat
            D["отрезок < 20 строк" if len(rr) < 20 else "отрезок ≥ 20 строк"] += wt
            if len(rr) < 20:
                continue
            # B: начало отрезка
            i0, y0 = sg[0]
            if i0 == 0:
                kind, frm = "трасса начинается не на эталоне", "—"
            else:
                yb = ys[i0 - 1]; sb, ob = stt[yb]
                frm = {"own": "со своего", "nb": "с соседнего", "off": "—", "nogt": "—"}[sb]
                if sb == "off":
                    kind = "продолжение ухода после пропуска"
                elif sb == "nogt":
                    kind = "эталон начинается, трасса уже на другой линии"
                elif y0 - yb > 2:
                    kind = "через пропуск выдачи (> 2 строк)"
                else:
                    L = gts[ob]
                    if y0 in L and yb in L:
                        dch = abs((w[y0] - L[y0]) - (w[yb] - L[yb]))
                        kind = "прыжок (сдвиг от линии > 3 px за шаг)" if dch > 3 else "сползание (≤ 3 px за шаг)"
                        if dch > 3:
                            jx = abs(w[y0] - w[yb])
                            jb = ("прыжок трассы ≤ 10 px" if jx <= 10 else "10–30 px" if jx <= 30
                                  else "30–100 px" if jx <= 100 else "> 100 px")
                            E[("jump", jb)] += wt; E["прыжков, строк"] += wt
                            # ★ тушь ПОКИНУТОЙ линии в строках y0..y0+3 (±3 px): линия продолжалась — селектор её бросил;
                            #   туши нет — разрыв линии, трасса зацепилась за чужую в пропуске
                            live = sum(1 for yy in range(y0, y0 + 4) if yy in L and snap(yy, L[yy]) is not None)
                            big = "прыжок > 30 px" if jx > 30 else "прыжок ≤ 30 px"
                            E[("live", f"{big}: покинутая линия ПРОДОЛЖАЛАСЬ (тушь ≥ 3 из 4 строк)" if live >= 3
                               else f"{big}: у покинутой линии РАЗРЫВ")] += wt
                            # эталон в той же точке: сколько сам сдвинулся за шаг
                            E[("jump_gt", "эталон тоже сдвинулся > 10 px" if abs(L[y0] - L[yb]) > 10 else "эталон почти на месте")] += wt
                    else:
                        kind = "покинутая линия кончилась"
            B[kind] += wt; B[(src, kind)] += wt; BN[kind] += 1
            B[("from", frm)] += wt
            B["всего"] += wt; B[(src, "всего")] += wt; BN["всего"] += 1
            # C: стиль
            yv = [y for y in rr if 0 <= y < H]
            if len(yv) >= 20:
                pt = [snap(y, w[y]) for y in yv]; pg = [snap(y, gt[y]) for y in yv]
                ct = np.mean([p is not None for p in pt]); cg = np.mean([p is not None for p in pg])
                tt, tg = [], []
                for j, y in enumerate(yv):
                    for lst, pp, line in ((tt, pt, w), (tg, pg, gt)):
                        if pp[j] is None or (y - 5) not in line or (y + 5) not in line:
                            continue
                        s = (line[y + 5] - line[y - 5]) / 10.0
                        if abs(s) <= 2:
                            lst.append(hrun(y, pp[j]) / math.sqrt(1 + s * s))
                mt_ = float(np.median(tt)) if len(tt) >= 10 else 0.0
                mg_ = float(np.median(tg)) if len(tg) >= 10 else 0.0
                if abs(ct - cg) >= 0.25:
                    sty = "различимы по сплошности (|Δ| ≥ 0.25)"
                elif mt_ and mg_ and max(mt_, mg_) / min(mt_, mg_) >= 1.5:
                    sty = "различимы по толщине поперёк (≥ 1.5×)"
                elif mt_ and mg_:
                    sty = "по стилю НЕ различимы"
                else:
                    sty = "толщину не измерить (мало строк)"
                C[sty] += wt; C["всего"] += wt
                REC.append((sh, g, src, y0, len(rr), wt, kind, frm, ct, cg, mt_, mg_))
        # A: метки строк (только строки этой кривой)
        dgc = domc.get(g)
        for y in sorted(mine):
            xi = int(round(w[y]))
            if not (0 <= y < H and 0 <= xi < Wd):
                cat = "вне листа"
            elif any(y in gts[o] and abs(w[y] - gts[o][y]) <= 3 for o in others):
                cat = "эталон другого трека"
            elif y in lab:
                cat = lab[y]
            elif rul2[y // a.block, xi]:
                cat = "вертикаль-линейка"
            elif ink2[y, xi]:
                cat = "другая тушь"
                if xl is not None:
                    E[("scale", "в пределах шкалы своей кривой" if xl - 3 <= w[y] <= xr + 3 else "ВНЕ шкалы своей кривой")] += 1
                sx = snap(y, w[y], 2)
                if sx is not None:
                    hh = int(comp_h[_lab[y, sx]])
                    E[("comp", "компонента ≥ 1000 строк" if hh >= 1000 else "300–1000 строк" if hh >= 300
                       else "50–300 строк" if hh >= 50 else "< 50 строк (текст/шум)")] += 1
                if multicol and dgc:
                    ct_ = colour(y, snap(y, w[y]))
                    if ct_:
                        AC["разноцв. трек: тушь того же цвета, что эталон" if ct_ == dgc
                           else "разноцв. трек: тушь ДРУГОГО цвета"] += 1
            else:
                cat = "туши нет"
            A[cat] += 1; A[(src, cat)] += 1; A[(src, "всего")] += 1
            dd = abs(w[y] - gt[y])
            D["3–6 px" if dd <= 6 else "6–15 px" if dd <= 15 else "15–50 px" if dd <= 50 else "> 50 px"] += 1
    del rgb, blk, ink, ink2, COL, sm
    if si % 50 == 0:
        print(f"  … {si}/{len(sheets)}, кривых {ncurves}", file=sys.stderr)

tot = sum(v for k, v in A.items() if isinstance(k, str))
print(f"\n★★ НЕ ВЗЯТЫЕ КРИВЫЕ МНОГОКРИВЫХ ТРЕКОВ: {ncurves}; строк ухода (без двойного счёта) {tot}")
print("A. ЧТО ПОД ТРАССОЙ:")
CATS = [k for k, _ in sorted(((k, v) for k, v in A.items() if isinstance(k, str)), key=lambda kv: -kv[1])]
for k in CATS:
    print(f"   {k}: {A[k]} ({100*A[k]/max(1,tot):.1f}%)")
for src in ("prod", "dec", "?"):
    n = A.get((src, "всего"), 0)
    if n:
        print(f"   {src}: строк {n} ({100*n/max(1,tot):.0f}%): " +
              ", ".join(f"{c} {100*A.get((src, c), 0)/n:.0f}%" for c in CATS if A.get((src, c))))
print("   расстояние трасса−эталон: " + ", ".join(f"{k} {100*D[k]/max(1,tot):.0f}%" for k in ("3–6 px", "6–15 px", "15–50 px", "> 50 px")))
print("   строки ухода в отрезках: " + ", ".join(f"{k} {100*D[k]/max(1,tot):.0f}%" for k in ("отрезок < 20 строк", "отрезок ≥ 20 строк")))
for grp in ("scale", "comp"):
    n_ = sum(v for k, v in E.items() if isinstance(k, tuple) and k[0] == grp)
    print(f"   «другая тушь» — {'шкала' if grp == 'scale' else 'высота связной компоненты под трассой'} (строк {n_}): " +
          ", ".join(f"{k[1]} {100*v/max(1,n_):.0f}%" for k, v in sorted(((k, v) for k, v in E.items() if isinstance(k, tuple) and k[0] == grp), key=lambda kv: -kv[1])))
nc = sum(AC.values())
if nc:
    print(f"   цвет (только «другая тушь» на треках с эталонами разного цвета, строк {nc}): " +
          ", ".join(f"{k} {100*v/nc:.0f}%" for k, v in AC.most_common()))
T = B["всего"]
print(f"B. НАЧАЛО ОТРЕЗКОВ УХОДА ≥ 20 строк: {BN['всего']} отрезков, {T} строк")
for k, v in sorted(((k, v) for k, v in B.items() if isinstance(k, str) and k != "всего"), key=lambda kv: -kv[1]):
    print(f"   {k}: {100*v/max(1,T):.0f}% строк ({BN[k]} отр.)")
print("   покинутая линия: " + ", ".join(f"{k[1]} {100*v/max(1,T):.0f}%" for k, v in B.items() if isinstance(k, tuple) and k[0] == "from"))
for src in ("prod", "dec"):
    n = B.get((src, "всего"), 0)
    if n:
        print(f"   {src} ({100*n/max(1,T):.0f}% строк): " + "; ".join(
            f"{k.split(' (')[0]} {100*v/n:.0f}%" for (s2, k), v in sorted(
                ((k, v) for k, v in B.items() if isinstance(k, tuple) and k[0] == src and k[1] != "всего"), key=lambda kv: -kv[1])))
nj = E["прыжков, строк"]
print(f"   величина прыжка трассы в момент ухода (строк отрезков {nj}): " + ", ".join(
    f"{k[1]} {100*v/max(1,nj):.0f}%" for k, v in sorted(((k, v) for k, v in E.items() if isinstance(k, tuple) and k[0] == "jump"), key=lambda kv: -kv[1])))
print("   тушь покинутой линии в момент прыжка: " + ", ".join(
    f"{k[1]} {100*v/max(1,nj):.0f}%" for k, v in sorted(((k, v) for k, v in E.items() if isinstance(k, tuple) and k[0] == "live"), key=lambda kv: -kv[1])))
print("   покинутая линия в тот же шаг: " + ", ".join(
    f"{k[1]} {100*v/max(1,nj):.0f}%" for k, v in E.items() if isinstance(k, tuple) and k[0] == "jump_gt"))
print(f"   для сравнения — шаги самих эталонов: > 10 px {100*E['эталон: шаг > 10 px за строку']/max(1,E['эталон: строк-шагов']):.2f}%, "
      f"> 30 px {100*E['эталон: шаг > 30 px за строку']/max(1,E['эталон: строк-шагов']):.3f}% строк")
TC = C["всего"]
print(f"C. СТИЛЬ НА ОТРЕЗКАХ (строк {TC}): " + "; ".join(
    f"{k} {100*v/max(1,TC):.0f}%" for k, v in sorted(((k, v) for k, v in C.items() if k != "всего"), key=lambda kv: -kv[1])))
if a.dump:
    pickle.dump(REC, open(a.dump, "wb"))

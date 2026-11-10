r"""_sample_review.py — случайная выборка листов: скан + эталон эксперта + выдача прода, вердикт по каждой кривой; разбор промахов (03.11).

Выдача — повтор прода (`--mode`, по умолчанию NSE19) ВНЕ ФОЛДА (`taskS/rp_lvl/<mode>`: модели не видели скважину листа) — честная картина для новых сканов.
Итоговая оцифровка (`final_2026-10/FINAL`) по листам корпуса завышена (модели учены на всём корпусе) — её вердикт рядом, для сверки.
Мера — приёмки (plane2 в значениях): функции скопированы из `_name_cost_prod.py` (он исполняется при импорте); полистные счёты
сверяются с его дампом `rp_lvl_NS_<mode>.pkl` — расхождение = ошибка копии, прогон падает.

Причина промаха кривой эталона (первая подходящая):
  верно / верно под другим именем (1:1 внутри трека, как безымянный счёт) / нет кривой с этим именем / уровень масштаба (в пикселях
  честно, в значениях нет) / нет общей глубины / короткая (≤ 3 px, покрытие < 0.9) / близко (3–6 px) / на другой кривой эталона /
  не та линия.

  _sample_review.py --pop          — все листы поле + вне + сорт A → <out>/pop_curves.tsv, pop_summary.txt
  _sample_review.py --sample 30    — случайные листы (--seed) → <out>/NN_*.jpg, diag.json, ВЫБОРКА.html
"""
import sys, argparse, pickle, hashlib, json, base64, io, math
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from extract_nlgx import extract, NULL, depth_of
from _multi_replica_probe import dense
from auto import meta as M
import decode_levels as DL

TS = Path(r"F:/nds/output/taskS")
POOLS = ["pools", "pools_gate", "pools_wide", "pools_more", "pools_div", "pools_heldout", "pools_all"]
ARCH = Path(r"F:\nds\projects\Archive")
MODE = "NSE19"                                    # --mode: какой повтор прода показывать (каталог rp_lvl/<MODE>, дамп rp_lvl_NS_<MODE>.pkl)
OOF = TS / "rp_lvl" / MODE
FINAL = Path(r"F:/nds/output/final_2026-10/FINAL")
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9
CATS = ["верно", "верно под другим именем", "нет кривой с этим именем", "уровень масштаба", "нет общей глубины",
        "короткая (покрытие < 0.9)", "близко (3–6 px)", "на другой кривой эталона", "не та линия"]

# ── мера приёмки: копия `_name_cost_prod.py` (plane2, --levels) ──────────────────────────────────────────────────────────────
_TREES = {}


def _plane(tr, com, gt):
    from scipy.spatial import cKDTree
    k = id(tr)
    if k not in _TREES:
        ys = np.array(sorted(tr), float); xs = np.array([tr[int(y)] for y in ys], float)
        P = [np.stack([ys, xs], 1)]
        if len(ys) > 1:
            dy = np.diff(ys); dx = np.diff(xs)
            for i in np.flatnonzero((dy <= 30) & (np.abs(dx) > 1) & (np.abs(dx) <= 3000)):
                n = int(np.ceil(abs(dx[i]))); t = np.arange(1, n) / n
                P.append(np.stack([ys[i] + dy[i] * t, xs[i] + dx[i] * t], 1))
        _TREES[k] = (cKDTree(np.concatenate(P)), tr)
    d, _ = _TREES[k][0].query(np.array([[y, gt[y]] for y in com], float))
    return d


def lv_rows(c):
    out = {}
    for y0, y1, lv in c.get("segments") or []:
        for y in range(int(y0), int(y1) + 1):
            out[y] = int(lv)
    return out


def relevel(tr, lw, lg, chain):
    if len(chain) < 2:
        return tr
    out = {}
    for y, x in tr.items():
        lo, lt = lw.get(y, 0), lg.get(y, 0)
        if lo == lt:
            out[y] = x; continue
        if lo >= len(chain) or lt >= len(chain):
            out[y] = x + 1e4; continue
        so, sg = chain[lo], chain[lt]
        v = so["v_left"] + (x - so["x_left"]) * (so["v_right"] - so["v_left"]) / ((so["x_right"] - so["x_left"]) or 1)
        out[y] = sg["x_left"] + (v - sg["v_left"]) * (sg["x_right"] - sg["x_left"]) / ((sg["v_right"] - sg["v_left"]) or 1)
    return out


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = _plane(tr, com, gt); d2 = _plane(gt, com, tr)
    return max(float(np.median(d)), float(np.median(d2))), len(com) / max(1, len(gt))


def match(rows, cols, ok, pairs_out):
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
    n_ = sum(1 for r in rows if try_(r, set()))
    pairs_out.extend((r, c) for c, r in pair.items())
    return n_


# ── окружение ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def sheet_dir(nm):
    st = Path(nm).stem
    return f"{st[:40]}_{hashlib.md5(st.encode('utf-8')).hexdigest()[:8]}"


def load_tracks():
    """слот → трек из пуловых дампов (3.3 ГБ на HDD) — один раз, дальше из `taskS/_track_map.pkl`"""
    cache = TS / "_track_map.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    TRACK, seen = {}, set()
    for root in POOLS:
        for f in sorted((TS / root).glob("*.pkl")):
            if f.stem in seen:
                continue
            seen.add(f.stem)
            d = pickle.load(open(f, "rb"))
            TRACK[d["name"]] = {s["name"]: s["track"] for s in d["slots"]}
    pickle.dump(TRACK, open(cache, "wb"))
    return TRACK


def eval_sets():
    lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    FLD = set(lst("wellmap_sheets.txt")); HLD = set(lst("holdoutA_sheets.txt"))
    EVO = set(json.loads((TS / "line_choice_v6e" / "line_choice_folds.json").read_text(encoding="utf-8"))) - FLD
    return {"поле": FLD, "вне": EVO, "сорт A": HLD}


def curves(src, got):
    mt = extract(str(src)); mw = extract(str(got))
    gts = {c["name"]: dense(c) for c in mt["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    wr = {c["name"]: dense(c) for c in mw["curves"] if M.mnem_root(c["name"]) != "DA"}
    return mt, mw, gts, wr


def diagnose(src, got, tr_map):
    """→ (записи по кривым эталона, лишние кривые выдачи, (именных, безымянных, кривых) как у `_name_cost_prod.py`)"""
    _TREES.clear()
    mt, mw, gts, wr = curves(src, got)
    LG = {c["name"]: lv_rows(c) for c in mt["curves"] if c["name"] in gts}
    CH = {c["name"]: DL.build_family(mt, c) for c in mt["curves"] if c["name"] in gts}
    LW = {c["name"]: lv_rows(c) for c in mw["curves"] if c["name"] in wr}
    E = {}

    def e(w, g):
        if (w, g) not in E:
            E[(w, g)] = err(relevel(wr[w], LW.get(w, {}), LG.get(g, {}), CH.get(g, [])), gts[g])
        return E[(w, g)]
    named = {g: g in wr and HON(*e(g, g)) for g in gts}
    pairs, fre = [], 0
    for t in set(tr_map.get(k) for k in gts) | set(tr_map.get(k) for k in wr):
        if t is None:
            continue
        G = [k for k in gts if tr_map.get(k) == t]; W = [k for k in wr if tr_map.get(k) == t]
        if G and W:
            fre += match(G, W, {(g, w): HON(*e(w, g)) for g in G for w in W}, pairs)
    partner = dict(pairs)
    recs = []
    for g, gt in gts.items():
        d, c = e(g, g) if g in wr else (None, 0.0)
        r = {"curve": g, "track": tr_map.get(g), "rows": len(gt), "y0": min(gt), "y1": max(gt),
             "d": None if d is None else round(d, 2), "cov": round(c, 3)}
        if named[g]:
            cat = "верно"
        elif g in partner:
            cat = "верно под другим именем"; r["as"] = partner[g]
        elif g not in wr:
            cat = "нет кривой с этим именем"
        else:
            dp, cp = err(wr[g], gt)
            if HON(dp, cp):
                cat = "уровень масштаба"; r["d_px"] = round(dp, 2)
            elif d is None:
                cat = "нет общей глубины"
            elif d <= 3:
                cat = "короткая (покрытие < 0.9)"
            elif d <= 6:
                cat = "близко (3–6 px)"
            else:
                oth = [g2 for g2 in gts if g2 != g and HON(*e(g, g2))]
                cat = "на другой кривой эталона" if oth else "не та линия"
                if oth:
                    r["on"] = oth[0]
        r["cat"] = cat
        if cat != "верно":                       # ближайшая кривая выдачи (любое имя) — для зума
            bw = [(e(w, g)[0], w) for w in wr if e(w, g)[0] is not None]
            if bw:
                r["best"] = min(bw)[1]; r["best_d"] = round(min(bw)[0], 2)
        recs.append(r)
    used = set(partner.values()) | {g for g in gts if named[g]}
    extra = [w for w in wr if w not in used and w not in gts]
    return recs, extra, (sum(named.values()), fre, len(gts))


# ── разложение всех промахов ─────────────────────────────────────────────────────────────────────────────────────────────────
def population(out):
    TRACK = load_tracks(); S = eval_sets(); dump = pickle.load(open(TS / f"rp_lvl_NS_{MODE}.pkl", "rb"))[MODE]
    SRC = {q.name: q for q in ARCH.glob("*/wlg/*.nlgx")}
    rows, bad, n = [], 0, 0
    agg = {k: Counter() for k in list(S) + ["поле+вне"]}
    for part, SS in S.items():
        for nm in sorted(SS):
            got = next(iter(sorted((OOF / sheet_dir(nm)).glob("*_auto.nlgx"))), None)
            if got is None or nm not in SRC:
                continue
            recs, extra, cnt = diagnose(SRC[nm], got, TRACK.get(nm, {}))
            n += 1
            if nm in dump and tuple(dump[nm]) != tuple(cnt):
                bad += 1; print(f"  ⛔ расхождение с дампом: {nm} {cnt} против {dump[nm]}")
            for r in recs:
                agg[part][r["cat"]] += 1
                if part != "сорт A":
                    agg["поле+вне"][r["cat"]] += 1
                rows.append([part, nm, r["curve"], str(r["track"]), r["cat"], str(r["d"]), str(r["cov"]), str(r.get("best_d", ""))])
            agg[part]["лишних кривых выдачи"] += len(extra)
            if n % 200 == 0:
                print(f"  {n} листов")
    (out / "pop_curves.tsv").write_text("\n".join("\t".join(r) for r in
                                        [["часть", "лист", "кривая", "трек", "причина", "d", "cov", "best_d"]] + rows) + "\n",
                                        encoding="utf-8")
    L = [f"★ листов {n}, сверка с дампом {MODE}: {'СОШЛОСЬ' if not bad else f'⛔ {bad} расхождений'}"]
    for part in ["поле+вне", "поле", "вне", "сорт A"]:
        c = agg[part]; tot = sum(v for k, v in c.items() if k in CATS)
        L += ["", f"## {part}: кривых эталона {tot}"]
        L += [f"  {k:28s} {c[k]:5d}  {c[k] / max(tot, 1):6.1%}" for k in CATS]
        L += [f"  {'лишних кривых выдачи':28s} {c['лишних кривых выдачи']:5d}"]
    (out / "pop_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if bad:
        sys.exit(1)


# ── выборка с картинками ─────────────────────────────────────────────────────────────────────────────────────────────────────
GT_C, OUR_C = (0, 190, 0, 150), (255, 0, 255, 235)
# цвет на имя кривой: эталон — толстая полупрозрачная, выдача — тонкая того же цвета (одно имя = один цвет)
PAL = [(0, 170, 0), (255, 120, 0), (230, 0, 230), (0, 170, 230), (150, 90, 0), (120, 60, 255), (0, 140, 120), (220, 40, 90)]


def _col(names):
    return {nm: PAL[i % len(PAL)] for i, nm in enumerate(names)}


def _font(sz):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("arial.ttf", sz)
    except OSError:
        return ImageFont.load_default()


def _lines(dr, tr, fx, fy, col, w, step):
    ys = sorted(tr)[::step]
    run = []
    for y in ys:
        if run and (y - run[-1][2] > 3 * step or abs(tr[y] - tr[run[-1][2]]) > 150):   # разрыв или перевынос — не соединять
            if len(run) > 1:
                dr.line([(a, b) for a, b, _ in run], fill=col, width=w)
            run = []
        run.append((fx(tr[y]), fy(y), y))
    if len(run) > 1:
        dr.line([(a, b) for a, b, _ in run], fill=col, width=w)


def _jpg(im, q=78):
    b = io.BytesIO(); im.convert("RGB").save(b, "JPEG", quality=q); return b.getvalue()


def render(rgb, mt, mw, gts, wr, recs, title, odir, tag):
    from PIL import Image, ImageDraw
    H0, W0 = rgb.shape[:2]
    gx = [x for t in gts.values() for x in t.values()]
    ox = [x for t in wr.values() for x in t.values() if 0 <= x < W0]
    x0 = int(max(0, min(gx + ([np.percentile(ox, 1)] if ox else [])) - 120))
    x1 = int(min(W0, max(gx + ([np.percentile(ox, 99)] if ox else [])) + 120))
    y0 = int(max(0, min(min(t) for t in gts.values()) - 100)); y1 = int(min(H0, max(max(t) for t in gts.values()) + 100))
    W, H = x1 - x0, y1 - y0
    TW, TH = 1900, 1300
    s = min(1.0, math.sqrt(TW * TH / (H * W)))
    P = max(1, math.ceil(H * s / TH)); ph = math.ceil(H * s / P)
    base = Image.fromarray(np.ascontiguousarray(rgb[y0:y1, x0:x1])).resize((max(1, round(W * s)), max(1, round(H * s))),
                                                                           Image.LANCZOS).convert("RGBA")
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    fx = lambda x: (x - x0) * s; fy = lambda y: (y - y0) * s; st = max(1, int(1 / s))
    C = _col(list(gts) + [w for w in wr if w not in gts])
    for nm, g in gts.items():
        _lines(dr, g, fx, fy, C[nm] + (110,), max(4, round(7 * s)), st)
    for nm, w in wr.items():
        _lines(dr, {y: x for y, x in w.items() if -50 <= x < W0 + 50}, fx, fy, C[nm] + (255,), max(1, round(2 * s)), st)
    img = Image.alpha_composite(base, ov)
    pw = img.size[0]; HDR = 64 + 22 * math.ceil(len(recs) / 3)
    can = Image.new("RGB", (P * (pw + 14) + 14, ph + HDR + 26), "white"); cd = ImageDraw.Draw(can)
    f1, f2, f3 = _font(22), _font(17), _font(15)
    cd.text((14, 8), title, fill="black", font=f1)
    cd.text((14, 36), "толстая полупрозрачная — эталон эксперта;  тонкая того же цвета — наша оцифровка под тем же именем",
            fill=(90, 90, 90), font=f2)
    for i, r in enumerate(recs):
        ok = r["cat"] == "верно"; X = 14 + (i % 3) * 600; Y = 60 + 22 * (i // 3)
        cd.rectangle((X, Y + 4, X + 14, Y + 16), fill=C[r["curve"]])
        cd.text((X + 20, Y), f"{'+' if ok else '×'} {r['curve']}: {r['cat']}"
                + (f" ({r['d']} px)" if r.get("d") is not None and not ok else ""), fill=(0, 120, 0) if ok else (200, 0, 0), font=f2)
    for p in range(P):
        a, b = p * ph, min(img.size[1], (p + 1) * ph)
        if a >= b:
            continue
        X = 14 + p * (pw + 14)
        can.paste(img.crop((0, a, pw, b)), (X, HDR + 20))
        ya, yb = y0 + a / s, y0 + b / s
        da, db = depth_of(mt, ya), depth_of(mt, yb)
        cd.text((X, HDR + 2), f"{da:.0f}–{db:.0f} м" if np.isfinite(da) and np.isfinite(db) else f"стр. {ya:.0f}", fill="black", font=f3)
        for nm, t in list(gts.items()) + list(wr.items()):
            yy = [y for y in t if ya <= y < ya + 40 / s]
            if yy:
                ours = t is not gts.get(nm)
                cd.text((X + fx(t[yy[0]]) + 6, HDR + 22 + (18 if ours else 0)), nm.split()[0] + (" (наша)" if ours else ""),
                        fill=tuple(int(v * 0.75) for v in C[nm]), font=f3)
    files = {"overview": f"{tag}_overview.jpg"}
    (odir / files["overview"]).write_bytes(_jpg(can))
    # зумы промахов: где разошлись сильнее всего и типичное место
    for r in recs:
        if r["cat"] == "верно":
            continue
        g = gts[r["curve"]]; w = wr.get(r["curve"]) if r["curve"] in wr else wr.get(r.get("best", ""), None)
        com = sorted(y for y in g if w is not None and y in w)
        lvl = ""
        if r["cat"] == "уровень масштаба":            # зум туда, где разошлись уровни масштаба, а не пиксели
            lg = lv_rows(next(c for c in mt["curves"] if c["name"] == r["curve"]))
            lw = lv_rows(next(c for c in mw["curves"] if c["name"] == r["curve"]))
            mis = [y for y in com if lg.get(y, 0) != lw.get(y, 0)]
            if mis:
                runs, st_ = [], 0
                for i in range(1, len(mis) + 1):
                    if i == len(mis) or mis[i] - mis[i - 1] > 5:
                        runs.append((mis[st_], mis[i - 1])); st_ = i
                a_, b_ = max(runs, key=lambda t: t[1] - t[0]); m_ = (a_ + b_) // 2
                lvl = f"; уровень эталона {lg.get(m_, 0)}, наш {lw.get(m_, 0)}; расходятся {len(mis)} строк из {len(com)}"
                cen = {"начало расхождения уровня": a_, "середина расхождения уровня": m_}
        if not lvl and len(com) > 50:
            dv = np.array([abs(w[y] - g[y]) for y in com]); k = min(len(com), 200)
            sm = np.convolve(dv, np.ones(k) / k, "same")
            cen = {"худшее": com[int(np.argmax(sm))], "типичное": com[int(np.argsort(sm)[len(sm) // 2])]}
        elif not lvl:
            ys = sorted(g); cen = {"середина": ys[len(ys) // 2]}
        for lab, c in cen.items():
            za, zb = max(0, c - 350), min(H0, c + 350)
            xs = [g[y] for y in g if za <= y < zb] + ([w[y] for y in w if za <= y < zb and 0 <= w[y] < W0] if w else [])
            xa, xb = int(max(0, min(xs) - 110)), int(min(W0, max(xs) + 110))
            if xb - xa > 1100:
                cx = int(np.median([g[y] for y in g if za <= y < zb])); xa, xb = max(0, cx - 550), min(W0, cx + 550)
            zi = Image.fromarray(np.ascontiguousarray(rgb[za:zb, xa:xb])).convert("RGBA")
            zo = Image.new("RGBA", zi.size, (0, 0, 0, 0)); zd = ImageDraw.Draw(zo)
            for nm, t in gts.items():
                _lines(zd, {y: x for y, x in t.items() if za <= y < zb}, lambda x: x - xa, lambda y: y - za, C[nm] + (110,), 6, 1)
            for nm, t in wr.items():
                _lines(zd, {y: x for y, x in t.items() if za <= y < zb}, lambda x: x - xa, lambda y: y - za, C[nm] + (255,), 2, 1)
            zc = Image.new("RGB", (zi.size[0], zi.size[1] + 30), "white"); zc.paste(Image.alpha_composite(zi, zo), (0, 30))
            dz = depth_of(mt, c)
            ImageDraw.Draw(zc).text((6, 6), f"{r['curve']} — {r['cat']}, {lab}" + (" место" if not lvl else "")
                                    + (f", {dz:.1f} м" if np.isfinite(dz) else "") + lvl, fill="black", font=f2)
            fn = f"{tag}_{''.join(ch if ch.isalnum() else '_' for ch in r['curve'])}_{lab}.jpg"
            (odir / fn).write_bytes(_jpg(zc)); files.setdefault("zoom", []).append([r["curve"], lab, fn])
    return files


def sample(out, n, seed):
    from PIL import Image
    import _final_exif                            # load_rgb с разворотом EXIF и починкой пустых полос TIFF (§6.352)
    from dataset_build import find_image
    TRACK = load_tracks(); S = eval_sets(); SRC = {q.name: q for q in ARCH.glob("*/wlg/*.nlgx")}
    dump = pickle.load(open(TS / f"rp_lvl_NS_{MODE}.pkl", "rb"))[MODE]
    part = {nm: p for p, SS in S.items() for nm in SS}
    uni = sorted(nm for nm in part if nm in SRC and any((OOF / sheet_dir(nm)).glob("*_auto.nlgx")))
    order = np.random.default_rng(seed).permutation(len(uni))
    print(f"★ листов с выдачей вне фолда: {len(uni)}; seed {seed}")
    picked, skipped, meta = [], [], []
    for i in order:
        if len(picked) >= n:
            break
        nm = uni[i]; src = SRC[nm]; img = find_image(src)
        if img is None:
            skipped.append((nm, "нет скана")); continue
        with Image.open(img) as im:
            if im.size[0] * im.size[1] > 400e6:
                skipped.append((nm, f"скан {im.size[0]}×{im.size[1]} — слишком велик для показа")); continue
        picked.append(nm)
        got = next(iter(sorted((OOF / sheet_dir(nm)).glob("*_auto.nlgx"))))
        recs, extra, cnt = diagnose(src, got, TRACK.get(nm, {}))
        assert nm not in dump or tuple(dump[nm]) == tuple(cnt), f"расхождение с дампом {nm}: {cnt} против {dump[nm]}"
        fin = next(iter(sorted((FINAL / sheet_dir(nm)).glob("*_auto.nlgx"))), None)
        fcnt = diagnose(src, fin, TRACK.get(nm, {}))[2] if fin else None
        mt, mw, gts, wr = curves(src, got)
        if not gts:                               # у эталона нет плотных кривых (только пустые / NULL) — показывать нечего
            picked.pop(); skipped.append((nm, "у эталона нет кривых с точками")); continue
        rgb = _final_exif.load_rgb(str(img))
        tag = f"{len(picked):02d}"
        title = (f"{tag}. {nm}  ({part[nm]}, скв. {src.parent.parent.name})  — верно {cnt[0]} из {cnt[2]}"
                 + (f";  в итоговой выдаче {fcnt[0]} из {fcnt[2]}" if fcnt else ""))
        files = render(rgb, mt, mw, gts, wr, recs, title, out, tag)
        del rgb
        meta.append({"tag": tag, "sheet": nm, "part": part[nm], "well": src.parent.parent.name, "scan": str(img), "oof": cnt,
                     "final": fcnt, "curves": recs, "extra": extra, "files": files, "out_nlgx": str(got),
                     "final_nlgx": str(fin) if fin else None, "truth_nlgx": str(src)})
        print(f"  {tag} {nm}: верно {cnt[0]}/{cnt[2]}, безымянных {cnt[1]}; итог {fcnt}; "
              + ", ".join(f"{r['curve']}={r['cat']}" for r in recs if r["cat"] != "верно"))
    for nm, why in skipped:
        print(f"  ⚠ пропущен {nm}: {why}")
    (out / "diag.json").write_text(json.dumps({"seed": seed, "skipped": skipped, "sheets": meta}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    html(out, meta, skipped, seed)


def html(out, meta, skipped, seed):
    b64 = lambda f: base64.b64encode((out / f).read_bytes()).decode()
    tot = sum(m["oof"][2] for m in meta); ok = sum(m["oof"][0] for m in meta); un = sum(m["oof"][1] for m in meta)
    cats = Counter(r["cat"] for m in meta for r in m["curves"])
    rv = out / "review.json"                      # разбор причин глазами (workflow), ключ «тег|кривая»
    REV = json.loads(rv.read_text(encoding="utf-8")) if rv.exists() else {}
    H = ["<!doctype html><html lang=ru><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
         "<title>Случайная выборка оцифровки</title><style>body{font:15px/1.45 system-ui,sans-serif;margin:16px;background:#fff;"
         "color:#111}img{max-width:100%;border:1px solid #ccc}table{border-collapse:collapse}td,th{border:1px solid #ccc;"
         "padding:3px 8px;text-align:left}.ok{color:#080}.bad{color:#c00}h2{margin-top:2.2em}</style>",
         f"<h1>Случайная выборка: {len(meta)} листов (seed {seed})</h1>",
         f"<p>Оцифровка — прод на листах, чью скважину модели не видели (повтор {MODE} вне фолда): так алгоритм поведёт себя на новых "
         "сканах. На картинках: толстая полупрозрачная линия — эталон эксперта, тонкая того же цвета — наша кривая под тем же "
         "именем. Мера приёмки: медиана расстояния на плоскости в обе стороны ≤ 3 px, покрытие ≥ 0.9, в значениях шкалы. "
         "«В итоговой выдаче» — тот же лист в итоговой оцифровке архива: её модели видели весь корпус, по листам корпуса она лучше "
         "честной.</p>",
         f"<p><b>Верно с именем: {ok} из {tot} кривых ({ok / max(tot, 1):.0%})</b>; верная линия без учёта имени: {un} "
         f"({un / max(tot, 1):.0%}).</p><table><tr><th>причина</th><th>кривых</th></tr>"
         + "".join(f"<tr><td>{k}</td><td>{cats[k]}</td></tr>" for k in CATS if cats[k]) + "</table>"]
    rf = out / "review_full.json"
    if rf.exists():
        pat = json.loads(rf.read_text(encoding="utf-8"))["verified"]["patterns"]
        H.append("<h2>Разбор ошибок и порядок исправления</h2>" + "".join(f"<p>{p}</p>" for p in pat.split("\n") if p.strip()))
    if skipped:
        H.append("<p>Пропущены при выборе: " + "; ".join(f"{a} — {b}" for a, b in skipped) + "</p>")
    for m in meta:
        H.append(f"<h2>{m['tag']}. {m['sheet']} <small>({m['part']}, скв. {m['well']})</small></h2>")
        H.append(f"<p>Верно {m['oof'][0]} из {m['oof'][2]}" + (f"; в итоговой выдаче {m['final'][0]} из {m['final'][2]}"
                                                                if m["final"] else "") + "</p><table>")
        for r in m["curves"]:
            c = "ok" if r["cat"] == "верно" else "bad"
            H.append(f"<tr><td>{r['curve']}</td><td class={c}>{r['cat']}" + (f" → {r['as']}" if r.get("as") else "")
                     + (f" (стоит на {r['on']})" if r.get("on") else "") + "</td><td>"
                     + (f"{r['d']} px, покрытие {r['cov']:.0%}" if r.get("d") is not None else "") + "</td><td>"
                     + (f"<b>{REV[k]['cause']}</b>" + (" — <b>ошибка эталона</b>" if REV[k].get("truth_suspect") else "")
                        + f": {REV[k]['note']}" if (k := f"{m['tag']}|{r['curve']}") in REV else "") + "</td></tr>")
        if m["extra"]:
            H.append(f"<tr><td colspan=3>лишние в выдаче: {', '.join(m['extra'])}</td></tr>")
        H.append("</table>")
        H.append(f"<p><img src='data:image/jpeg;base64,{b64(m['files']['overview'])}'></p>")
        for cv, lab, fn in m["files"].get("zoom", []):
            H.append(f"<p><img src='data:image/jpeg;base64,{b64(fn)}' style='max-width:48%;margin-right:1%'></p>")
    (out / "ВЫБОРКА.html").write_text("\n".join(H) + "\n", encoding="utf-8")
    print(f"★ {out / 'ВЫБОРКА.html'}: верно {ok}/{tot}, безымянных {un}; " + ", ".join(f"{k} {v}" for k, v in cats.most_common()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"F:/nds/output/sample_2026-11-03")
    ap.add_argument("--pop", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--mode", default="NSE19", help="повтор прода: rp_lvl/<mode>, дамп rp_lvl_NS_<mode>.pkl")
    a = ap.parse_args()
    MODE = a.mode; OOF = TS / "rp_lvl" / MODE
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.pop:
        population(out)
    if a.sample:
        sample(out, a.sample, a.seed)
    if a.html_only:                               # пересобрать страницу из diag.json (+ review.json), без отрисовки
        D = json.loads((out / "diag.json").read_text(encoding="utf-8"))
        html(out, D["sheets"], D["skipped"], D["seed"])

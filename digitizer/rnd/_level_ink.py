r"""_level_ink.py — УРОВЕНЬ МАСШТАБА ПО СКАНУ ВОКРУГ ЛИНИИ (03.10, §6.266; критерии — в ROADMAP до обучения).

Ответ заказчика (03.10): «большие значения всегда в масштабе, который левее»; штрих 5× «обычно такой же»; «иногда пунктир,
который идёт от завершения масштаба до начала (справа налево)»; «каждую линию отдельно смотрю». ⇒ уровень — по скану вокруг
самой линии, а не по трассе (§6.263–6.265).

scan  — кривые кэша `_level_bench.py` с цепочкой масштабов (поле + сорт A). Каждая 8-я строка протяжённости трассы выдачи:
        - профиль туши поперёк полосы шкал (объединение x всех уровней ± 5% ширины): 128 корзин, максимум затемнения по
          8 строкам блока (горизонтальный пунктир перехода виден как тушь во многих корзинах);
        - занятость полосы ДРУГИМИ трассами выдачи того же листа (128 корзин) — чья тушь чужая;
        - тушь в точках «близнеца»: то же значение на соседнем уровне (k → k+1 и k+1 → k, до уровня 3);
        - уровень выдачи (прод) и уровень эталона;
        - метка — уровень В ЗНАЧЕНИЯХ: единственный k, при котором x выдачи, пересчитанный с уровня k на уровень эталона,
          ложится в 5 px от эталона (несколько или ни одного — без метки; уровень эталона ≥ 4 — без метки).
        Пишет `--out/<md5 листа>.pkl`, с подхватом (готовые листы пропускает), `--workers` процессов пониженного приоритета.
train — torch (embedded-python ComfyUI): 1D свёртка по позициям — вход 1×1 → 64 канала, 12 остаточных блоков (норма по каналам,
        GELU, ядро 5, расширения 1…32 дважды — поле ±252 позиции ≈ ±2000 строк), выход — уровни 0…3 с маской k < длины цепочки.
        Входные каналы по позиции: профиль туши и занятость — в абсолютной сетке полосы и в сетке смещений от трассы выдачи
        (±64 корзины), позиция трассы, тушь близнецов, уровень прода, константы цепочки (длина, отношение диапазонов, сдвиг,
        семейство). Поле — 5 фолдов по скважинам, предсказание вне фолда; сорт A — модель на всём поле. Уровни → на все строки
        по ближайшей позиции → enforce_min_run 25 (та же логика, по пробегам) → именные честные в значениях (функция
        `_level_bench.py`), парный знаковый тест по листам против прода.

  _level_ink.py scan --workers 4
  _level_ink.py train --steps 4000
"""
import sys, argparse, pickle, hashlib, re, os, time
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

RES = {"GZ", "OGZ", "BK", "MBK", "BKZ", "BMK", "IK", "IKA", "IKR", "PZ", "MGZ", "MPZ", "BKP", "LL", "LLD", "LLS"}
KMAX = 4                                    # уровни 0…3


def group(root):
    r = re.sub(r"^BKZ_", "", root)
    if r in RES:
        return 0
    if r == "GK":
        return 1
    if r in ("NGK", "NNK", "NNKB", "NNKM", "NKTB", "NKTM"):
        return 2
    if r.startswith("TM"):
        return 3
    return 4


def val(s, x):
    """пиксель → значение по шкале s (как в honest)"""
    return s["v_left"] + (x - s["x_left"]) * (s["v_right"] - s["v_left"]) / ((s["x_right"] - s["x_left"]) or 1)


def pix(s, v):
    """значение → пиксель по шкале s (как в honest)"""
    return s["x_left"] + (v - s["v_left"]) * (s["x_right"] - s["x_left"]) / ((s["v_right"] - s["v_left"]) or 1)


def seg_levels(segs, rows):
    """уровни строк rows по сегментам (вне сегментов — 0; поздний сегмент сверху)"""
    out = np.zeros(len(rows), np.int16)
    for y0, y1, lv in segs or []:
        if lv:
            out[(rows >= y0) & (rows <= y1)] = lv
    return out


# ───────────────────────────── scan ─────────────────────────────

def _low_priority():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL
    except Exception:
        pass


def scan_curve(cv, gray, paper, OT, P):
    H, W = gray.shape
    st, nb, tol = P["stride"], P["nb"], P["tol"]
    half = st // 2
    ch = cv["chain"]; K = len(ch)
    xl = min(min(s["x_left"], s["x_right"]) for s in ch); xr = max(max(s["x_left"], s["x_right"]) for s in ch)
    w = max(8.0, xr - xl); m = 0.05 * w
    bx0 = int(max(0, np.floor(xl - m))); bx1 = int(min(W, np.ceil(xr + m)))
    ncol = bx1 - bx0
    ty, tx, gy, gx = cv["ty"], cv["tx"], cv["gy"], cv["gx"]
    if len(ty) < 50 or ncol < nb:
        return None
    y0 = int(max(half, ty[0])); y1 = int(min(H - st + half, ty[-1]))
    if y1 - y0 < st * 10:
        return None
    Pp = np.arange(y0, y1 + 1, st, dtype=np.int64)
    n = len(Pp)
    # трасса выдачи (плотная, как в мере) и эталон в позициях
    j = np.clip(np.searchsorted(ty, Pp), 0, len(ty) - 1); has = ty[j] == Pp
    xo = np.where(has, tx[j], np.nan).astype(np.float64)
    jg = np.clip(np.searchsorted(gy, Pp), 0, len(gy) - 1); hg = gy[jg] == Pp
    g = np.where(hg, gx[jg], np.nan).astype(np.float64)
    lt = seg_levels(cv["lt"], Pp); lw = seg_levels(cv["lw"], Pp)
    # близнецы: если трасса на k — тот же отсчёт на k+1; если на k+1 — на k
    TWX = []
    for k in range(min(K, KMAX) - 1):
        TWX.append(pix(ch[k + 1], val(ch[k], xo)))
        TWX.append(pix(ch[k], val(ch[k + 1], xo)))
    NT = 2 * (KMAX - 1)
    tw = np.zeros((n, NT), np.uint8); twin = np.zeros((n, NT), np.uint8)
    ink = np.zeros((n, nb), np.uint8)
    edges = np.linspace(0, ncol, nb + 1).astype(np.int64)[:-1]
    CH = 512
    for c0 in range(0, n, CH):
        c1 = min(n, c0 + CH); nbk = c1 - c0
        r0 = int(Pp[c0]) - half; r1 = r0 + nbk * st
        G = gray[r0:r1, bx0:bx1].astype(np.int16)
        D = np.clip(paper[r0:r1, None] - G, 0, 255).astype(np.uint8)
        Dk = D.reshape(nbk, st, ncol).max(1)
        ink[c0:c1] = np.maximum.reduceat(Dk, edges, axis=1)
        for t, xc in enumerate(TWX):
            xs_ = xc[c0:c1]
            ok = np.isfinite(xs_) & (xs_ >= bx0 + 3) & (xs_ <= bx1 - 4)
            ci = np.round(np.where(ok, xs_, bx0 + 3) - bx0).astype(np.int64)
            idx = np.clip(ci[:, None] + np.arange(-3, 4)[None, :], 0, ncol - 1)
            v = Dk[np.arange(nbk)[:, None], idx].max(1)
            tw[c0:c1, t] = np.where(ok, v, 0); twin[c0:c1, t] = ok
    # занятость полосы другими трассами выдачи
    occ = np.zeros((n, nb), np.uint8)
    for nm, oy, ox in OT:
        if nm == cv["name"] or oy[-1] < Pp[0] - half or oy[0] > Pp[-1] + half:
            continue
        k = np.clip(np.searchsorted(oy, Pp), 1, len(oy) - 1)
        kn = np.where(np.abs(oy[k] - Pp) <= np.abs(oy[k - 1] - Pp), k, k - 1)
        near = np.abs(oy[kn] - Pp) <= half
        b = np.floor((ox[kn] - bx0) / ncol * nb).astype(np.int64)
        sel = near & (b >= 0) & (b < nb)
        occ[np.flatnonzero(sel), b[sel]] = 1
    # метка в значениях
    Kc = min(K, KMAX)
    E = np.full((n, Kc), np.inf)
    for t in np.unique(lt):
        if t >= K:
            continue
        mm = lt == t
        for k in range(Kc):
            xt = xo[mm] if k == t else pix(ch[t], val(ch[k], xo[mm]))
            E[mm, k] = np.abs(xt - g[mm])
    okk = E <= tol
    one = (okk.sum(1) == 1) & has & hg & (lt < KMAX)
    lab = np.full(n, -1, np.int8); lab[one] = np.argmax(okk[one], axis=1)
    s0, s1 = ch[0], ch[1]
    r0_ = (s0["v_right"] - s0["v_left"]) or 1.0
    f = abs((s1["v_right"] - s1["v_left"]) / r0_)
    shift = float(abs(s1["v_left"] - s0["v_right"]) < 1e-6 * max(1.0, abs(s0["v_right"])))
    u = np.where(has, (xo - bx0) / ncol, np.nan).astype(np.float32)
    return dict(ci=cv["ci"], set=cv["set"], sheet=cv["sheet"], name=cv["name"], root=cv["root"], K=K, f=f, shift=shift,
                grp=group(cv["root"]), band=(bx0, bx1), P=Pp.astype(np.int32), ink=ink, occ=np.packbits(occ, axis=1),
                u=u, has=has, tw=tw, twin=twin, lw=lw.astype(np.int8), lt=lt.astype(np.int8), lab=lab)


def v4_extra(res, rgb, gray, paper, P):
    """§6.271: длина вертикального рана туши по корзинам (блок 8 строк, максимум), цвет самого тёмного пикселя корзины
    (R − B, G − (R + B)/2, /2 в int8), цвет туши под трассой выдачи (±3 px). Тушь — затемнение > vthr."""
    st, nb, thr = P["stride"], P["nb"], P["vthr"]
    half = st // 2
    bx0, bx1 = res["band"]; ncol = bx1 - bx0
    Pp = res["P"].astype(np.int64); n = len(Pp)
    r0 = int(Pp[0]) - half; nr = n * st
    edges = np.linspace(0, ncol, nb + 1).astype(np.int64)
    RLk = np.zeros((n, ncol), np.uint16)
    chroma = np.zeros((n, nb, 2), np.int8); tchroma = np.zeros((n, 2), np.int8)
    xo = np.where(res["has"], res["u"].astype(np.float64) * ncol + bx0, np.nan)
    CH = 512
    M = np.zeros((nr, ncol), bool)
    for c0 in range(0, n, CH):
        c1 = min(n, c0 + CH); nbk = c1 - c0
        a0 = r0 + c0 * st; a1 = a0 + nbk * st
        D = np.clip(paper[a0:a1, None] - gray[a0:a1, bx0:bx1].astype(np.int16), 0, 255)
        M[c0 * st:c1 * st] = D > thr
        Dk = D.reshape(nbk, st, ncol); arow = Dk.argmax(1); dmax = Dk.max(1)
        ii = np.arange(nbk)
        for b in range(nb):
            e0, e1 = edges[b], edges[b + 1]
            jb = dmax[:, e0:e1].argmax(1); col = e0 + jb
            dv = dmax[ii, col]; row = arow[ii, col]
            px = rgb[a0 + ii * st + row, bx0 + col].astype(np.int16)
            ok = dv > thr
            chroma[c0:c1, b, 0] = np.where(ok, np.clip((px[:, 0] - px[:, 2]) // 2, -127, 127), 0)
            chroma[c0:c1, b, 1] = np.where(ok, np.clip((px[:, 1] - (px[:, 0] + px[:, 2]) // 2) // 2, -127, 127), 0)
        xs_ = xo[c0:c1]
        okt = np.isfinite(xs_)
        ci = np.round(np.where(okt, xs_, bx0) - bx0).astype(np.int64)
        idx = np.clip(ci[:, None] + np.arange(-3, 4)[None, :], 0, ncol - 1)
        win = dmax[ii[:, None], idx]; jj = win.argmax(1)
        col = idx[ii, jj]; dv = dmax[ii, col]; row = arow[ii, col]
        px = rgb[a0 + ii * st + row, bx0 + col].astype(np.int16)
        okt = okt & (dv > thr)
        tchroma[c0:c1, 0] = np.where(okt, np.clip((px[:, 0] - px[:, 2]) // 2, -127, 127), 0)
        tchroma[c0:c1, 1] = np.where(okt, np.clip((px[:, 1] - (px[:, 0] + px[:, 2]) // 2) // 2, -127, 127), 0)
    for c in range(ncol):
        col = M[:, c]
        if not col.any():
            continue
        d = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
        stt = np.flatnonzero(d == 1); enn = np.flatnonzero(d == -1)
        rl = np.zeros(nr, np.uint16)
        lens = np.minimum(enn - stt, 400).astype(np.uint16)
        rid = np.cumsum(d[:-1] == 1)
        rl[col] = lens[rid[col] - 1]
        RLk[:, c] = rl.reshape(n, st).max(1)
    vrun = np.maximum.reduceat(RLk, edges[:-1], axis=1).astype(np.int32)
    res["vrun"] = (np.minimum(vrun, 400) * 255 // 400).astype(np.uint8)
    res["chroma"] = chroma; res["tchroma"] = tchroma
    return res


def scan_sheet(job):
    sheet, img, got, curves, P = job
    from PIL import Image
    from extract_nlgx import extract, NULL
    Image.MAX_IMAGE_PIXELS = None
    outp = Path(P["out"]) / (hashlib.md5(sheet.encode("utf-8")).hexdigest()[:16] + ".pkl")
    if outp.exists():
        return sheet, "есть", 0, 0.0
    t0 = time.time()
    gray = np.asarray(Image.open(img).convert("L"))
    H = gray.shape[0]
    paper = np.empty(H, np.int16)
    for r in range(0, H, 2048):
        paper[r:r + 2048] = np.percentile(gray[r:r + 2048, ::4], 90, axis=1)
    OT = []
    for c in extract(str(got))["curves"]:
        pts = [(c["top_y"] + i, x) for i, x in enumerate(c["xs"]) if x != NULL]
        if len(pts) >= 2:
            OT.append((c["name"], np.array([p[0] for p in pts], np.int64), np.array([p[1] for p in pts], np.float64)))
    res = [scan_curve(cv, gray, paper, OT, P) for cv in curves]
    if P.get("v4"):
        rgb = np.asarray(Image.open(img).convert("RGB"))
        res = [None if r is None else v4_extra(r, rgb, gray, paper, P) for r in res]
        del rgb
    tmp = outp.with_suffix(".tmp")
    pickle.dump(dict(sheet=sheet, curves=res), open(tmp, "wb"), protocol=4)
    os.replace(tmp, outp)
    return sheet, "готов", sum(r is not None for r in res), time.time() - t0


def main_scan(a):
    import multiprocessing as mp
    Path(a.out).mkdir(parents=True, exist_ok=True)
    CUR = pickle.load(open(a.cache, "rb"))
    IMGS = {}
    for q in Path(r"F:\nds\projects\Archive").glob("*/img/*"):
        if q.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            IMGS.setdefault(q.stem, q)
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    by = defaultdict(list)
    for i, cv in enumerate(CUR):
        if len(cv["chain"]) >= 2:
            by[cv["sheet"]].append(dict(ci=i, **{k: cv[k] for k in ("set", "sheet", "name", "root", "chain", "gy", "gx", "lt",
                                                                     "ty", "tx", "lw")}))
    P = dict(stride=a.stride, nb=a.nb, tol=a.tol, out=a.out, v4=a.v4, vthr=a.vthr)
    jobs, miss = [], Counter()
    for sh, cs in sorted(by.items()):
        q = SRC.get(sh)
        if not q or q.stem not in IMGS:
            miss["нет скана"] += 1; continue
        stem = q.stem
        pd = Path(a.dir) / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
        got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
        if not got:
            miss["нет выдачи"] += 1; continue
        if (Path(a.out) / (hashlib.md5(sh.encode("utf-8")).hexdigest()[:16] + ".pkl")).exists():
            miss["уже есть"] += 1; continue
        jobs.append((sh, str(IMGS[stem]), str(got), cs, P))
    if a.limit:
        jobs = jobs[:a.limit]
    del CUR
    print(f"листов с цепочками {len(by)}; к скану {len(jobs)}; пропуск {dict(miss)}")
    t0 = time.time(); done = 0
    with mp.Pool(a.workers, initializer=_low_priority) as pool:
        for sh, st, nc, dt in pool.imap_unordered(scan_sheet, jobs):
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} листов, {time.time() - t0:.0f} с (последний {dt:.1f} с, кривых {nc})")
    print("скан готов")


# ───────────────────────────── train ─────────────────────────────

def honest(cv, lo):
    """как в `_level_bench.py`: lo — None (уровни выдачи), {} (всё 0) или dict строка→уровень (наследуется последний)"""
    gy, gx, ty, tx, ch = cv["gy"], cv["gx"], cv["ty"], cv["tx"], cv["chain"]
    com, ig, it = np.intersect1d(gy, ty, return_indices=True)
    if len(com) < 30:
        return False
    x = tx[it].astype(np.float64); g = gx[ig].astype(np.float64)
    lt = seg_levels(cv["lt"], com)
    if lo is None:
        lw = seg_levels(cv["lw"], com)
    elif not lo:
        lw = np.zeros(len(com), np.int16)
    else:
        ks = np.array(sorted(lo), np.int64); vs = np.array([lo[k] for k in ks], np.int16)
        pos = np.searchsorted(ks, com, side="right") - 1
        lw = np.where(pos >= 0, vs[np.clip(pos, 0, len(vs) - 1)], 0).astype(np.int16)
    if len(ch) >= 2:
        diff = lw != lt
        for o in np.unique(lw[diff]):
            for t in np.unique(lt[diff & (lw == o)]):
                m = diff & (lw == o) & (lt == t)
                if o >= len(ch) or t >= len(ch):
                    x[m] = 1e4; continue
                so, sg = ch[o], ch[t]
                v = so["v_left"] + (x[m] - so["x_left"]) * (so["v_right"] - so["v_left"]) / ((so["x_right"] - so["x_left"]) or 1)
                x[m] = sg["x_left"] + (v - sg["v_left"]) * (sg["x_right"] - sg["x_left"]) / ((sg["v_right"] - sg["v_left"]) or 1)
    e = np.abs(x - g)
    return bool(np.median(e) <= 3.0 and len(com) / max(1, len(gy)) >= 0.9)


def min_run_runs(starts, levels, ends, min_run):
    """enforce_min_run из auto/refine.py на пробегах (строки сплошные): тот же порядок — первый короткий пробег с соседом
    уходит к более длинному соседу (при равенстве — к предыдущему), затем слияние и повтор."""
    R = [[int(s), int(e), int(l)] for s, e, l in zip(starts, ends, levels)]

    def merge(R):
        out = []
        for r in R:
            if out and out[-1][2] == r[2]:
                out[-1][1] = r[1]
            else:
                out.append(r)
        return out
    R = merge(R)
    changed = True
    while changed:
        changed = False
        for k2, (s, e, lv) in enumerate(R):
            if e - s + 1 >= min_run:
                continue
            pv = R[k2 - 1] if k2 > 0 else None; nx = R[k2 + 1] if k2 + 1 < len(R) else None
            if pv is None and nx is None:
                continue
            if pv is None:
                to = nx[2]
            elif nx is None:
                to = pv[2]
            else:
                to = pv[2] if (pv[1] - pv[0] + 1) >= (nx[1] - nx[0] + 1) else nx[2]
            if to != lv:
                R[k2][2] = to; R = merge(R); changed = True
                break
    return R


def main_train(a):
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CUR = pickle.load(open(a.cache, "rb"))
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    well = lambda sh: SRC[sh].parent.parent.name if sh in SRC else sh
    D = []
    for f in sorted(Path(a.out).glob("*.pkl")):
        for c in pickle.load(open(f, "rb"))["curves"]:
            if c is not None:
                D.append(c)
    field_wells = sorted({well(c["sheet"]) for c in D if c["set"] == "поле"})
    FOLD = {wl: i % a.folds for i, wl in enumerate(field_wells)}
    nb = D[0]["ink"].shape[1]; NT = D[0]["tw"].shape[1]
    # склейка в общий массив позиций
    off = np.cumsum([0] + [len(c["P"]) for c in D])
    N = int(off[-1])
    cat = lambda k: np.concatenate([c[k] for c in D])
    t_ink = torch.from_numpy(cat("ink")).to(dev)                                   # [N, nb] uint8
    t_occ = torch.from_numpy(np.unpackbits(cat("occ"), axis=1)[:, :nb].copy()).to(dev)
    u = cat("u"); hs = cat("has")
    t_u = torch.from_numpy(np.nan_to_num(u, nan=0.0).astype(np.float32)).to(dev)
    t_has = torch.from_numpy(hs.astype(np.float32)).to(dev)
    t_tw = torch.from_numpy(cat("tw")).to(dev); t_twin = torch.from_numpy(cat("twin")).to(dev)
    V4 = bool(a.v4) and all("vrun" in c for c in D)        # ★ §6.271: длина вертикальных ранов и цвет туши
    if a.v4 and not V4:
        sys.exit("⛔ --v4: в скане нет полей vrun/chroma — нужен скан с --v4")
    if V4:
        t_vrun = torch.from_numpy(cat("vrun")).to(dev)
        t_chr = torch.from_numpy(np.concatenate([c["chroma"].reshape(len(c["P"]), -1) for c in D])).to(dev)
        t_tch = torch.from_numpy(cat("tchroma")).to(dev)
    t_lw = torch.from_numpy(np.minimum(cat("lw"), KMAX - 1).astype(np.int64)).to(dev)
    t_lab = torch.from_numpy(cat("lab").astype(np.int64)).to(dev)
    cid = np.repeat(np.arange(len(D)), np.diff(off))
    CONST = np.array([[float(c["K"] == 2), float(c["K"] == 3), float(c["K"] >= 4), np.log(max(1e-3, c["f"])), c["shift"]]
                      + [float(c["grp"] == g_) for g_ in range(5)] for c in D], np.float32)
    KC = torch.from_numpy(np.array([min(c["K"], KMAX) for c in D], np.int64)).to(dev)
    t_const = torch.from_numpy(CONST).to(dev); t_cid = torch.from_numpy(cid).to(dev)
    n_lab = Counter(cat("lab").tolist())
    print(f"кривых {len(D)} (поле {sum(c['set'] == 'поле' for c in D)}), позиций {N}, меток {dict(sorted(n_lab.items()))}, "
          f"скважин поля {len(field_wells)}; устройство {dev}")
    OFS = torch.arange(-nb // 2, nb // 2, device=dev)
    DIL = tuple(int(x) for x in a.dil.split(",")) * 2               # ★ §6.269: расширения (дважды); по умолчанию 1…32 — как v2
    CIN = nb * 5 + 2 + 2 * NT + KMAX + CONST.shape[1] + ((nb * 2 + nb * 2 + 2) if V4 else 0)

    def feats(gi, valid):
        """gi [B, L] глобальные индексы позиций, valid [B, L] → [B, CIN, L]"""
        ink = t_ink[gi].float() / 255.0; occ = t_occ[gi].float()
        hv = t_has[gi] * valid
        b = torch.clamp((t_u[gi] * nb).long(), 0, nb - 1)
        oh = F.one_hot(b, nb).float() * hv[..., None]
        rel = b[..., None] + OFS
        rv = ((rel >= 0) & (rel < nb)).float() * hv[..., None]
        rel = rel.clamp(0, nb - 1)
        rink = torch.gather(ink, 2, rel) * rv; rocc = torch.gather(occ, 2, rel) * rv
        sc = [t_u[gi][..., None] * hv[..., None], hv[..., None], t_tw[gi].float() / 255.0, t_twin[gi].float(),
              F.one_hot(t_lw[gi], KMAX).float() * valid[..., None], t_const[t_cid[gi]] * valid[..., None]]
        if V4:
            vr = t_vrun[gi].float() / 255.0 * valid[..., None]
            rvr = torch.gather(vr, 2, rel) * rv
            sc += [vr, rvr, t_chr[gi].float() / 127.0 * valid[..., None], t_tch[gi].float() / 127.0 * hv[..., None]]
        x = torch.cat([ink * valid[..., None], occ * valid[..., None], oh, rink, rocc] + sc, dim=2)
        return x.transpose(1, 2).contiguous()

    class CNorm(nn.Module):
        def __init__(s, ch):
            super().__init__(); s.ln = nn.LayerNorm(ch)

        def forward(s, x):
            return s.ln(x.transpose(1, 2)).transpose(1, 2)

    class Net(nn.Module):
        def __init__(s, cin, ch=64, dil=(1, 2, 4, 8, 16, 32) * 2, k=5):
            super().__init__()
            s.inp = nn.Conv1d(cin, ch, 1)
            s.blocks = nn.ModuleList([nn.Sequential(CNorm(ch), nn.GELU(), nn.Conv1d(ch, ch, k, dilation=d, padding=d * (k // 2)))
                                      for d in dil])
            s.out = nn.Sequential(CNorm(ch), nn.GELU(), nn.Conv1d(ch, KMAX, 1))

        def forward(s, x):
            h = s.inp(x)
            for b in s.blocks:
                h = h + b(h)
            return s.out(h)

    def mask_logits(lo, kc):
        """lo [B, KMAX, L], kc [B] — уровни ≥ длины цепочки запрещены"""
        m = torch.arange(KMAX, device=dev)[None, :, None] >= kc[:, None, None]
        return lo.masked_fill(m, -1e4)

    def fit(tr_idx, tag, seed=0):
        torch.manual_seed(seed); rs = np.random.RandomState(seed)
        tr_idx = [i for i in tr_idx if (D[i]["lab"] >= 0).any()]
        ln = np.array([len(D[i]["P"]) for i in tr_idx], np.float64)
        pr = ln / ln.sum()
        net = Net(CIN, dil=DIL).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
        L = a.crop; ar = torch.arange(L, device=dev)
        t0 = time.time(); run = 0.0
        for step in range(a.steps):
            pick = rs.choice(len(tr_idx), size=a.batch, p=pr)
            ci = [tr_idx[p] for p in pick]
            st = np.array([off[c] + rs.randint(0, max(1, len(D[c]["P"]) - L + 1)) for c in ci], np.int64)
            en = np.array([off[c + 1] for c in ci], np.int64)
            st_t = torch.from_numpy(st).to(dev); en_t = torch.from_numpy(en).to(dev)
            gi = st_t[:, None] + ar[None, :]
            valid = (gi < en_t[:, None]).float()
            gi = torch.minimum(gi, en_t[:, None] - 1)
            x = feats(gi, valid)
            y = torch.where(valid > 0, t_lab[gi], torch.full_like(gi, -1))
            if not bool((y >= 0).any()):
                continue
            kc = KC[torch.from_numpy(np.array(ci, np.int64)).to(dev)]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                lo = net(x)
            lo = mask_logits(lo.float(), kc)
            loss = F.cross_entropy(lo, y, ignore_index=-1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sch.step()
            run = 0.98 * run + 0.02 * float(loss.detach()) if step else float(loss.detach())
            if (step + 1) % 500 == 0:
                print(f"    {tag}: шаг {step + 1}/{a.steps}, потеря {run:.4f}, {time.time() - t0:.0f} с")
        return net

    @torch.no_grad()
    def predict(nets, idx):
        """→ {i: вероятности уровней [KMAX, n]} — среднее по моделям; пачки по длине, округлённой вверх до 1024 позиций
        (немного разных форм — быстро; на краях вход дополнен нулями, как короткие куски при обучении)"""
        for net in nets:
            net.eval()
        out = {}
        by = defaultdict(list)
        for i in idx:
            by[-(-len(D[i]["P"]) // 1024) * 1024].append(i)
        for Lb, ids in by.items():
            bs = max(1, 32768 // Lb)
            for b0 in range(0, len(ids), bs):
                ib = ids[b0:b0 + bs]
                st_t = torch.from_numpy(np.array([off[i] for i in ib], np.int64)).to(dev)
                en_t = torch.from_numpy(np.array([off[i + 1] for i in ib], np.int64)).to(dev)
                gi = st_t[:, None] + torch.arange(Lb, device=dev)[None, :]
                valid = (gi < en_t[:, None]).float()
                gi = torch.minimum(gi, en_t[:, None] - 1)
                x = feats(gi, valid)
                kc = KC[torch.from_numpy(np.array(ib, np.int64)).to(dev)]
                pr = 0
                for net in nets:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        lo = net(x)
                    pr = pr + torch.softmax(mask_logits(lo.float(), kc), dim=1)
                pr = (pr / len(nets)).cpu().numpy()
                for j, i in enumerate(ib):
                    out[i] = pr[j, :, :len(D[i]["P"])].astype(np.float32)
        for net in nets:
            net.train()
        return out

    PROB = {}
    fi = [i for i, c in enumerate(D) if c["set"] == "поле"]
    T0 = time.time()
    META = dict(cin=CIN, nb=nb, kmax=KMAX, nt=NT, stride=a.stride, ch=64, dil=list(DIL), k=5, tol=a.tol, v4=V4, vthr=a.vthr,
                min_run=a.min_run, steps=a.steps, seeds=a.seeds)
    SD = Path(a.save_dir) if a.save_dir else None
    if SD:
        SD.mkdir(parents=True, exist_ok=True)
        import json as _json
        (SD / "level_ink_folds.json").write_text(_json.dumps(
            {D[i]["sheet"]: FOLD[well(D[i]["sheet"])] for i in fi}, ensure_ascii=False, indent=0), encoding="utf-8")
    for k in range(a.folds):
        tr = [i for i in fi if FOLD[well(D[i]["sheet"])] != k]; te = [i for i in fi if FOLD[well(D[i]["sheet"])] == k]
        nets = [fit(tr, f"фолд {k}, зерно {s_}", seed=1000 * k + s_) for s_ in range(a.seeds)]
        if SD:
            torch.save(dict(META, sds=[n_.state_dict() for n_ in nets], fold=k), SD / f"level_ink_f{k}.pt")
        PROB.update(predict(nets, te))
        print(f"  фолд {k}: обучение {len(tr)} кривых, проверка {len(te)}, моделей {len(nets)}; {time.time() - T0:.0f} с")
    nets = [fit(fi, f"всё поле, зерно {s_}", seed=9000 + s_) for s_ in range(a.seeds)]
    torch.save(dict(META, sds=[n_.state_dict() for n_ in nets], fold=None), a.model_out)
    if SD:
        torch.save(dict(META, sds=[n_.state_dict() for n_ in nets], fold=None), SD / "level_ink_all.pt")
    PROB.update(predict(nets, [i for i, c in enumerate(D) if c["set"] == "сорт A"]))
    print(f"  всё поле: моделей {len(nets)}; {time.time() - T0:.0f} с")
    PRED = {i: pr.argmax(0).astype(np.int16) for i, pr in PROB.items()}
    # точность по меткам
    for sn in ("поле", "сорт A"):
        ok = tot = 0; okp = 0
        for i, c in enumerate(D):
            if c["set"] != sn or i not in PRED:
                continue
            m = c["lab"] >= 0
            ok += int((PRED[i][m] == c["lab"][m]).sum()); okp += int((np.minimum(c["lw"][m], KMAX - 1) == c["lab"][m]).sum())
            tot += int(m.sum())
        print(f"   {sn}: верных уровней по меткам — модель {ok / max(1, tot):.3f}, прод {okp / max(1, tot):.3f} (позиций {tot})")
    # мера
    half = a.stride // 2
    LV = {}
    for i, c in enumerate(D):
        if i not in PRED:
            continue
        Pp = c["P"].astype(np.int64); p = PRED[i]
        ch_ = np.flatnonzero(np.diff(np.concatenate([[-1], p])) != 0)
        starts = Pp[ch_] - half; starts[0] = Pp[0] - half
        ends = np.concatenate([Pp[ch_[1:]] - half - 1, [Pp[-1] - half + a.stride - 1]])
        R = min_run_runs(starts, p[ch_], ends, a.min_run)
        LV[c["ci"]] = {r[0]: r[2] for r in R}
    C = Counter(); PER = defaultdict(lambda: [0, 0]); FH = Counter()
    for i, cv in enumerate(CUR):
        h0 = honest(cv, None)
        h1 = honest(cv, LV[i]) if (len(cv["chain"]) >= 2 and i in LV) else h0
        C[(cv["set"], 0)] += h0; C[(cv["set"], 1)] += h1
        PER[(cv["set"], cv["sheet"])][0] += h0; PER[(cv["set"], cv["sheet"])][1] += h1
        if len(cv["chain"]) >= 2:
            fm = re.sub(r"^BKZ_", "", cv["root"])
            FH[(cv["set"], fm, 0)] += h0; FH[(cv["set"], fm, 1)] += h1
    pickle.dump(dict(pred={D[i]["ci"]: (D[i]["P"], PRED[i]) for i in PRED}, levels=LV,
                     prob={D[i]["ci"]: PROB[i].astype(np.float16) for i in PROB}), open(a.dump, "wb"))
    rng = np.random.default_rng(0)
    for sn in ("поле", "сорт A"):
        d = np.array([v1 - v0 for (s, _), (v0, v1) in PER.items() if s == sn]); nz = d[d != 0]
        p = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)) >= abs(d.sum()))) if len(nz) else 1.0
        print(f"★ {sn}: именных честных в значениях — прод {C[(sn, 0)]}, модель {C[(sn, 1)]} (Δ {C[(sn, 1)] - C[(sn, 0)]:+d}, "
              f"листов ↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {p:.4f})")
        fams = sorted({k[1] for k in FH if k[0] == sn}, key=lambda f_: -FH[(sn, f_, 0)])
        print("   по семействам (прод/модель): " + "; ".join(f"{f_} {FH[(sn, f_, 0)]}/{FH[(sn, f_, 1)]}" for f_ in fams[:14]))


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "train"])
    ap.add_argument("--cache", default=r"F:/nds/output/taskS/level_bench.pkl")
    ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_vc/N")
    ap.add_argument("--out", default=r"F:/nds/output/taskS/level_ink")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--nb", type=int, default=128)
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--dil", default="1,2,4,8,16,32", help="расширения одного прохода (стек повторяется дважды)")
    ap.add_argument("--save-dir", default="", help="сохранить модели фолдов, модель на всём поле и карту лист → фолд")
    ap.add_argument("--v4", action="store_true", help="§6.271: скан — длина вертикальных ранов и цвет туши; обучение — эти каналы")
    ap.add_argument("--vthr", type=int, default=100)
    ap.add_argument("--min-run", type=int, default=25)
    ap.add_argument("--dump", default=r"F:/nds/output/taskS/level_ink_pred.pkl")
    ap.add_argument("--model-out", default=r"F:/nds/output/taskS/level_ink_model.pt")
    a = ap.parse_args()
    if a.cmd == "scan":
        main_scan(a)
    else:
        main_train(a)


if __name__ == "__main__":
    main()

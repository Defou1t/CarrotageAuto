r"""_cand_fail.py — КУДА УХОДИТ ЛУЧШАЯ ТРАССА КРИВОЙ БЕЗ ЧЕСТНОГО КАНДИДАТА (02.10, к §6.256).

§6.236: у кривых без честного кандидата (ни одна трасса обоих путей не честна) лучшая ОДНА трасса держит 0.30 строк эталона,
объединение всех — 0.62. Здесь — куда эта лучшая трасса уходит с кривой. Контекст декодера (§6.256) лечит уход на ЧУЖУЮ
кривую (пересечение, касание); уход в фон и обрывы — другое. Для лучшей трассы (больше всего строк в 3 px) строки эталона:
  «верно»  — трасса в 3 px от своей кривой;
  «чужая»  — в 3 px от ДРУГОЙ эталонной кривой листа;
  «рядом»  — в 3–10 px от своей кривой (пик кривой срезан, вершины эталона редки — §6.251, допуск толщины);
  «фон»    — трасса есть, но ни у одной кривой эталона (сетка, текст, шум, неоцифрованная кривая);
  «нет»    — у трассы нет точки (мост ≤ 30 строк, как в `hon` §6.236).
И уходы (отрезок «верно» ≥ 20 строк, затем ≥ 20 строк не «верно»): у касания ли (другая кривая эталона в 10 px от своей
в строке ухода) и куда (чужая / фон / нет). Только чтение кэша, выдачу не меняет.

  _cand_fail.py --cache F:/nds/output/taskS/tcache
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--dump", default="")
a = ap.parse_args()
TS = Path(a.ts)
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SETS = {"поле": lst("wellmap_sheets.txt"), "сорт A": lst("holdoutA_sheets.txt")}
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}


def arr(d, n):
    out = np.full(n, np.nan, np.float32)
    if d:
        ys = np.fromiter(d.keys(), int, len(d)); xs = np.fromiter(d.values(), float, len(d))
        ok = ys < n
        out[ys[ok]] = xs[ok]
    return out


def bridged(t, n, gap=30):
    """точки трассы + линейная штопка разрывов ≤ gap строк (покрытие как в `hon` §6.236)"""
    ys = np.asarray(t[0], int); xs = np.asarray(t[1], float)
    o = np.argsort(ys); ys, xs = ys[o], xs[o]
    ok = ys < n; ys, xs = ys[ok], xs[ok]
    raw = np.full(n, np.nan, np.float32); raw[ys] = xs
    br = raw.copy()
    if len(ys) > 1:
        d = np.diff(ys)
        for i in np.flatnonzero((d > 1) & (d <= gap)):
            yy = np.arange(ys[i] + 1, ys[i + 1])
            br[yy] = xs[i] + (xs[i + 1] - xs[i]) * (yy - ys[i]) / d[i]
    return raw, br


C = {k: Counter() for k in SETS}
FR = {k: [] for k in SETS}
DEP = {k: Counter() for k in SETS}
FD = {k: [] for k in SETS}
STEEP = {k: {"с": [], "без": []} for k in SETS}
BADST = {k: [] for k in SETS}
ROWS = []
for sn, names in SETS.items():
    for sh in names[::a.every]:
        q = SRC.get(sh)
        if not q:
            continue
        stem = q.stem; key = f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
        f = Path(a.cache) / f"{key}.pkl"
        if not f.exists():
            continue
        v = pickle.load(open(f, "rb"))
        Gd = {c["name"]: dense(c) for c in extract(str(q))["curves"]
              if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        if not Gd:
            continue
        T = [t for _, t in v["traces"]] + ([t for _, t in v["alt"]] if v.get("alt") else [])
        n = 1 + max([max(g) for g in Gd.values()] + [int(np.max(t[0])) for t in T if len(t[0])])
        G = {k: arr(g, n) for k, g in Gd.items()}
        TB = [bridged(t, n) for t in T]
        for g, gt in G.items():
            r = ~np.isnan(gt)
            nr = int(r.sum())
            if nr < 50:
                continue
            C[sn]["кривых"] += 1
            gi = gt.copy(); idx = np.flatnonzero(r)
            slope = np.abs(np.gradient(gt[idx])) if len(idx) > 2 else np.zeros(len(idx))
            steep = slope > 2.0                                   # строк с наклоном > 2 px/строку
            best, bgood, honest = None, -1, False
            for raw, br in TB:
                e = np.abs(raw[r] - gt[r]); com = ~np.isnan(e)
                if com.sum() >= 30 and np.median(e[com]) <= 3.0 and (~np.isnan(br[r])).mean() >= 0.9:
                    honest = True; break
                good = int(np.sum(np.abs(br[r] - gt[r]) <= 3.0))
                if good > bgood:
                    best, bgood = br, good
            if honest:
                C[sn]["с честным кандидатом"] += 1
                STEEP[sn]["с"].append(float(steep.mean()))
                continue
            STEEP[sn]["без"].append(float(steep.mean()))
            C[sn]["без честного кандидата"] += 1
            others = [o[r] for k2, o in G.items() if k2 != g]
            x = best[r] if best is not None else np.full(nr, np.nan, np.float32)
            gr = gt[r]
            st = np.full(nr, 3, np.int8)                         # 0 верно, 1 чужая, 2 фон, 3 нет, 4 рядом
            pres = ~np.isnan(x)
            st[pres] = 2
            st[pres & (np.abs(x - gr) <= 10.0)] = 4
            if others:
                O = np.vstack(others)
                near_o = np.nanmin(np.where(np.isnan(O), np.inf, np.abs(O - x[None, :])), axis=0) <= 3.0
                st[pres & near_o] = 1
                near10 = np.nanmin(np.where(np.isnan(O), np.inf, np.abs(O - x[None, :])), axis=0) <= 10.0
                contact = np.nanmin(np.where(np.isnan(O), np.inf, np.abs(O - gr[None, :])), axis=0) <= 10.0
            else:
                contact = np.zeros(nr, bool); near10 = np.zeros(nr, bool)
            st[pres & (np.abs(x - gr) <= 3.0)] = 0
            bad = pres & (st != 0)
            BADST[sn].append((float(steep[bad].mean()) if bad.any() else np.nan,
                              float((bad & (st == 2) & near10).sum()) / nr))
            fr = np.bincount(st, minlength=5) / nr
            fd = np.abs(x - gr)[st == 2]
            FD[sn].append(float(np.median(fd)) if len(fd) else np.nan)
            FR[sn].append(fr)
            # уходы: ≥ 20 строк «верно», затем ≥ 20 строк не «верно»
            okr = st == 0
            i = 0
            while i < nr:
                if not okr[i]:
                    i += 1; continue
                j = i
                while j < nr and okr[j]:
                    j += 1
                if j - i >= 20 and j < nr:
                    k = j
                    while k < nr and not okr[k]:
                        k += 1
                    if k - j >= 20:
                        tail = st[j:min(k, j + 50)]
                        where = ("чужая", "фон", "нет", "рядом")[int(np.bincount(tail, minlength=5)[1:].argmax())]
                        at = bool(contact[max(0, j - 5):j + 5].any())
                        DEP[sn][(where, "у касания" if at else "не у касания")] += 1
                i = j
            ROWS.append(dict(set=sn, sheet=sh, name=g, fam=M.mnem_root(g), fr=fr.tolist()))
if a.dump:
    pickle.dump(ROWS, open(a.dump, "wb"))
for sn in SETS:
    c = C[sn]
    print(f"★ {sn}: кривых {c['кривых']}, с честным кандидатом {c['с честным кандидатом']}, без — {c['без честного кандидата']}")
    if FR[sn]:
        F = np.array(FR[sn])
        NM = ("верно", "чужая", "фон", "нет", "рядом")
        print("   лучшая трасса, доли строк эталона (среднее по кривым): " +
              ", ".join(f"{nm} {F[:, i].mean():.2f}" for i, nm in enumerate(NM)))
        print("   медианы: " + ", ".join(f"{nm} {np.median(F[:, i]):.2f}" for i, nm in enumerate(NM)))
        fd = np.array([d for d in FD[sn] if d == d])
        print(f"   «фон»: расстояние до своей кривой — медиана по кривым {np.median(fd):.0f} px (квартили "
              f"{np.percentile(fd, 25):.0f}–{np.percentile(fd, 75):.0f})")
        dom = Counter(NM[int(np.argmax(f[1:]) + 1)] for f in F)
        ss = STEEP[sn]
        print(f"   доля крутых строк эталона (> 2 px/строку): с кандидатом — медиана {np.median(ss['с']):.2f}, "
              f"без — {np.median(ss['без']):.2f}")
        bs = np.array([b[0] for b in BADST[sn] if b[0] == b[0]]); b10 = np.array([b[1] for b in BADST[sn]])
        print(f"   у лучшей трассы: доля крутых среди неверных строк — медиана {np.median(bs):.2f}; «фон», который в 10 px "
              f"от другой кривой, — среднее {b10.mean():.2f} строк эталона")
        print("   главный вид потерь кривой (больше всего строк): " + ", ".join(f"{k} {v}" for k, v in dom.most_common()))
    d = DEP[sn]; tot = sum(d.values())
    if tot:
        print(f"   уходов с кривой (≥ 20 верных строк → ≥ 20 неверных): {tot}")
        for (w, at), m in sorted(d.items(), key=lambda kv: -kv[1]):
            print(f"      → {w:5s} {at:13s} {m:5d} ({100 * m / tot:.0f}%)")
        print(f"   уходов на кривую: {tot / max(1, C[sn]['без честного кандидата']):.1f}")

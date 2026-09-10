r"""_assemble_real.py — СБОРКА БЕЗ ОРАКУЛА: сколько из потолка 50.8% берёт настоящий решатель (§6.197).

ОТКУДА ВОПРОС. §6.192 намерил ПОТОЛОК сборки — 50.8% против 27.1% у прода на тех же кривых, — но
узлы там разрешал ОРАКУЛ по эталону. §6.197 показал, что весь оставшийся запас (+509 кривых) лежит
именно там, где линия U1 неверна, а тушь под кривой есть. Остался единственный вопрос, от которого
зависит, стоит ли это писать в прод: **сколько из потолка возьмёт решатель, который эталона не
видит.** Возьмёт мало — направление закрыто числом, а не мнением.

ЧТО ДЕЛАЕТ (то же поле и те же листы, что `_assemble_ceiling.py --seed 0`):
  1. раны строк маской ПРОДА (§6.100);
  2. сегменты трекером, РЕЗАННЫЕ в узлах (второй ран ближе `--near` ⇒ продолжение неоднозначно);
  3. ★ РЕШАТЕЛЬ БЕЗ ЭТАЛОНА: жадно строит K цепочек — K берётся из РАСКЛАДКИ (§6.194: совпадает с
     оцифрованным на 99.5%), сегмент присоединяется по непрерывности наклона, занятые не делятся;
  4. счёт честных — БЕЗЫМЯННЫЙ 1:1 внутри трека (§6.143), как считает ветка.

⚠ Сравнение печатается на ОДНИХ И ТЕХ ЖЕ кривых: прод, декодер, сборка. Иначе числа несравнимы.

  <ComfyUI>\python_embeded\python.exe _assemble_real.py --cap 32 --shard 0/4
  <ComfyUI>\python_embeded\python.exe _assemble_real.py --sum
"""
import sys, argparse, pickle, hashlib
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
from auto import meta as M, imaging as im, trace2d as T
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--rows", default="pick_learn_honest2.pkl.dump")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--prod", default="ab_wellmap/A")
ap.add_argument("--dec", default="ab_rdhonest/B")
ap.add_argument("--near", type=float, default=12.0)
ap.add_argument("--xt", type=float, default=8.0)
ap.add_argument("--gap", type=int, default=20)
ap.add_argument("--maxw", type=int, default=60)
ap.add_argument("--minseg", type=int, default=20)
ap.add_argument("--joingap", type=int, default=400, help="строк разрыва, через которые цепочка сшивается")
ap.add_argument("--joinx", type=float, default=25.0, help="допуск по x на стыке цепочки")
ap.add_argument("--mink", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cap", type=int, default=32)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--solver", default="greedy", choices=["greedy","rank"])
ap.add_argument("--tag", default="asmreal")
ap.add_argument("--sum", action="store_true")
a = ap.parse_args()
TS = Path(a.ts)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9

if a.sum:
    agg, sh = Counter(), 0
    for p in sorted(TS.glob(f"{a.tag}_*of*.pkl")):
        d = pickle.load(open(p, "rb"))
        agg.update(d["cnt"]); sh += d["sheets"]
    n = agg["кривых"]
    print(f"★★ СБОРКА БЕЗ ОРАКУЛА (листов {sh}, кривых {n})")
    print("| счёт | честных | доля |")
    for k in ("прод", "декодер", "★ сборка БЕЗ оракула"):
        print(f"| {k} | {agg[k]} | {100*agg[k]/max(1,n):.1f}% |")
    print(f"\n★ для сравнения потолок той же сборки с ОРАКУЛОМ (§6.192): 50.8%")
    sys.exit(0)

i, n = (int(x) for x in a.shard.split("/"))
trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]): int(r[3]["n_prod"]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q
_all = sorted({s for (s, t), K in KOF.items() if K >= a.mink})
cand = [_all[k] for k in np.random.default_rng(a.seed).permutation(len(_all))[:a.cap]]
mine = [s for k, s in enumerate(cand) if k % n == i]
print(f"★ ШАРД {i}/{n}: листов {len(mine)} из {len(cand)}")


def stats(tr, gt):
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
    return [r for r in rows if try_(r, set())]


def segments(rows, Y0):
    active, done = [], []
    for idx, xs in enumerate(rows):
        y = idx + Y0
        for s in active:
            k = list(s["pts"])[-30:]
            s["pred"] = (s["last_x"] if len(k) < 5 else
                         s["pts"][k[-1]] + (s["pts"][k[-1]] - s["pts"][k[0]]) /
                         max(1, k[-1] - k[0]) * (y - k[-1]))
        cnd = sorted((abs(s["pred"] - x), si, xi)
                     for si, s in enumerate(active) for xi, x in enumerate(xs)
                     if abs(s["pred"] - x) <= a.xt)
        us, ux = set(), set()
        for d0, si, xi in cnd:
            if si in us or xi in ux:
                continue
            if sum(1 for x in xs if abs(active[si]["pred"] - x) <= a.near) > 1:
                us.add(si)              # узел: не гадаем, режем
                continue
            us.add(si); ux.add(xi)
            active[si]["pts"][y] = xs[xi]
            active[si]["last_x"] = xs[xi]; active[si]["last_y"] = y
        for xi, x in enumerate(xs):
            if xi not in ux:
                active.append(dict(pts={y: x}, last_x=x, last_y=y, pred=x))
        keep = []
        for s in active:
            (keep if y - s["last_y"] <= a.gap else done).append(s)
        active = keep
    return [s["pts"] for s in done + active if len(s["pts"]) >= a.minseg]


def solve_rank(segs, rows, Y0, K):
    """★ ВТОРОЙ РЕШАТЕЛЬ — ПО РАНГУ x, а не по длине. На каждой строке раны упорядочены слева
    направо; сегмент наследует медианный ранг своих ранов, и цепочка k собирается из сегментов
    ранга k. Это прямая проверка предпосылки §5.2 («ordered» листы, 44%): если кривые не меняются
    местами, раздача по порядку законна и никакой склейки по наклону не нужно.
    ⚠ Ранг считается среди ранов ТОЙ ЖЕ строки — величина локальная и проду доступная."""
    rank = {}
    for idx, xs in enumerate(rows):
        for r, x in enumerate(xs):
            rank[(idx + Y0, x)] = r
    out = defaultdict(dict)
    for s in segs:
        rr = [rank.get((y, x)) for y, x in s.items()]
        rr = [v for v in rr if v is not None]
        if not rr:
            continue
        k = int(np.median(rr))
        if k < K:
            out[k].update(s)
    return [out[k] for k in sorted(out)]


def solve(segs, K):
    """★ РЕШАТЕЛЬ БЕЗ ЭТАЛОНА. Жадно: самый длинный свободный сегмент начинает цепочку, дальше к ней
    присоединяется сегмент, чьё начало ближе всего к линейной экстраполяции конца. Занятый сегмент
    второй цепочке не достаётся — это и есть ограничение 1:1, известное заранее."""
    free = sorted(range(len(segs)), key=lambda k: -len(segs[k]))
    used, chains = set(), []
    for _ in range(K):
        st = next((k for k in free if k not in used), None)
        if st is None:
            break
        used.add(st)
        ch = dict(segs[st])
        for _ in range(200):                       # сколько угодно стыков, но не бесконечно
            ys = sorted(ch)
            tail = ys[-30:]
            sl = ((ch[tail[-1]] - ch[tail[0]]) / max(1, tail[-1] - tail[0])) if len(tail) > 5 else 0.0
            best, bd = None, 1e9
            for k in free:
                if k in used:
                    continue
                s2 = segs[k]
                y2 = min(s2)
                if not (0 < y2 - ys[-1] <= a.joingap):
                    continue
                pred = ch[ys[-1]] + sl * (y2 - ys[-1])
                d = abs(pred - s2[y2])
                if d < bd:
                    bd, best = d, k
            if best is None or bd > a.joinx:
                break
            used.add(best)
            ch.update(segs[best])
        chains.append(ch)
    return chains


cnt, sheets = Counter(), 0
for si, sh in enumerate(mine, 1):
    q = SRC.get(sh)
    img = find_image(q) if q else None
    if not img:
        continue
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    tracks = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None and KOF.get((sh, t), 0) >= a.mink:
            tracks[t].append(nm)
    if not tracks:
        continue
    try:
        rgb = im.load_rgb(str(img))
    except Exception:
        continue
    H, W = rgb.shape[:2]
    p = DEFAULT.cv
    fgm = np.zeros((H, W), bool)
    for c in ["black", "red", "orange", "green", "blue"]:
        fgm |= T._color_fg(rgb, c, p)
    ink = fgm & ~im.structure_mask(rgb, p)
    sheets += 1
    stem = Path(sh).stem
    dn = f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    outs = {}
    for tag, root in (("прод", a.prod), ("декодер", a.dec)):
        d0 = Path(a.ts) / root / dn
        g0 = next(iter(sorted(d0.glob("*_auto.nlgx"))), None) if d0.is_dir() else None
        outs[tag] = ({c["name"]: dense(c) for c in extract(str(g0))["curves"]
                      if M.mnem_root(c["name"]) != "DA"} if g0 else {})

    for t, ns in tracks.items():
        allx = [x for nm in ns for x in gts[nm].values()]
        X0, X1 = max(0, int(min(allx)) - 40), min(W, int(max(allx)) + 40)
        ys = sorted(set().union(*[set(gts[nm]) for nm in ns]))
        Y0, Y1 = ys[0], ys[-1]
        rows = []
        for y in range(Y0, Y1 + 1):
            r = ink[y, X0:X1]
            if not r.any():
                rows.append([]); continue
            dd = np.diff(r.astype(np.int8))
            st_ = list(np.flatnonzero(dd == 1) + 1) + ([0] if r[0] else [])
            en_ = list(np.flatnonzero(dd == -1) + 1) + ([len(r)] if r[-1] else [])
            st_, en_ = sorted(st_), sorted(en_)
            rows.append(sorted(X0 + (s + e) / 2.0 for s, e in zip(st_, en_) if e - s <= a.maxw))
        segs = segments(rows, Y0)
        K = KOF.get((sh, t), len(ns))
        chains = solve_rank(segs, rows, Y0, K) if a.solver == "rank" else solve(segs, K)
        # ── мостик по цепочке, как в потолке: решатель отдаёт опорные строки, между ними интерполяция
        asm = []
        for ch in chains:
            if len(ch) < 30:
                continue
            kk = sorted(ch)
            allr = list(range(kk[0], kk[-1] + 1))
            iv = np.interp(allr, kk, [ch[y] for y in kk])
            asm.append({y: float(v) for y, v in zip(allr, iv)})
        cnt["кривых"] += len(ns)
        okA = {(g, j): HON(*stats(asm[j], gts[g])) for g in ns for j in range(len(asm))}
        cnt["★ сборка БЕЗ оракула"] += len(match(ns, list(range(len(asm))), okA))
        for tag in ("прод", "декодер"):
            W_ = [k for k in outs[tag] if tm.get(k) == t]
            ok = {(g, w): HON(*stats(outs[tag][w], gts[g])) for g in ns for w in W_}
            cnt[tag] += len(match(ns, W_, ok))
    if si % 3 == 0 or si == len(mine):
        print(f"  {si}/{len(mine)}  кривых {cnt['кривых']}, сборка {cnt['★ сборка БЕЗ оракула']}, "
              f"прод {cnt['прод']}")

out = TS / f"{a.tag}_{i}of{n}.pkl"
pickle.dump(dict(cnt=dict(cnt), sheets=sheets), open(out, "wb"))
print(f"★ ГОТОВО: листов {sheets}, кривых {cnt['кривых']} → {out.name}")
for k in ("прод", "декодер", "★ сборка БЕЗ оракула"):
    print(f"   {k}: {cnt[k]}")

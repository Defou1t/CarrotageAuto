r"""_assemble_ceiling.py — ПОТОЛОК СБОРКИ: если узлы разрешать ИДЕАЛЬНО, сколько кривых соберётся (§6.191).

ОТКУДА ВОПРОС. §6.191 показал: решений на лист — десятки (медиана 54 узла), а 90.6% материала
однозначно. Отсюда предложение — собирать штрихи объектами, РЕЗАТЬ их в узлах и разрешать узлы
глобально. Прежде чем писать решатель, надо узнать его ПОТОЛОК: сколько честных кривых вышло бы,
если бы каждый узел разрешался правильно. Не выйдет почти ничего — значит дело не в узлах, а в
самих сегментах, и решатель не окупится.

ЧТО СЧИТАЕТСЯ (ровно то же, что §6.190 назвал правильной постановкой):
  1. раны в строке — маской ПРОДА (§6.100), минус структура;
  2. штрихи трекером с памятью и наклоном; штрих РЕЖЕТСЯ, как только рядом с предсказанием есть
     второй ран ближе `--near` — то есть сегмент никогда не проходит через узел;
  3. ОРАКУЛ раздаёт сегменты кривым эталона (1:1, каждый сегмент максимум одной кривой);
  4. счёт честных — критерий ветки: med ≤ 3px И cov ≥ 0.9, без мостика и с мостиком.

⚠⚠ ЭТО ПОТОЛОК, А НЕ РЕЗУЛЬТАТ. Оракул знает ответ; настоящий решатель будет ниже. Смысл числа
один: ВЕРХНЯЯ граница того, что даст правильная постановка. Сравнивать надо с продом и декодером
НА ТЕХ ЖЕ ЛИСТАХ — стенд печатает все три.

  <ComfyUI>\python_embeded\python.exe _assemble_ceiling.py --cap 40 --shard 0/4
  <ComfyUI>\python_embeded\python.exe _assemble_ceiling.py --sum
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
ap.add_argument("--near", type=float, default=12.0, help="узел: второй ран ближе этого ⇒ режем")
ap.add_argument("--xt", type=float, default=8.0, help="допуск продолжения по x")
ap.add_argument("--gap", type=int, default=20, help="строк без туши, после которых штрих закрыт")
ap.add_argument("--maxw", type=int, default=60)
ap.add_argument("--minseg", type=int, default=20, help="сегмент короче — выбрасывается")
ap.add_argument("--cap", type=int, default=40)
ap.add_argument("--mink", type=int, default=3, help="брать листы с треком K ≥ этого")
ap.add_argument("--seed", type=int, default=0, help="зерно СЛУЧАЙНОЙ выборки листов")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--tag", default="asmceil")
ap.add_argument("--sum", action="store_true")
a = ap.parse_args()
TS = Path(a.ts)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9

if a.sum:
    agg, sheets = Counter(), 0
    for p in sorted(TS.glob(f"{a.tag}_*of*.pkl")):
        d = pickle.load(open(p, "rb"))
        agg.update(d["cnt"]); sheets += d["sheets"]
    print(f"★★ ПОТОЛОК СБОРКИ (листов {sheets}, кривых {agg['кривых']})")
    print("| счёт | честных | доля |")
    for k in ("прод", "декодер", "★ сборка без мостика", "★ сборка с мостиком"):
        print(f"| {k} | {agg[k]} | {100*agg[k]/max(1,agg['кривых']):.1f}% |")
    print(f"\n★ сегментов на кривую: {agg['сегментов']/max(1,agg['кривых']):.1f}; "
          f"кривых, которым не досталось ни одного: {agg['без сегментов']}")
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
# ★★ ПОЛЕ: листы с треком K ≥ `--mink`, ВЫБРАННЫЕ СЛУЧАЙНО С ФИКСИРОВАННЫМ ЗЕРНОМ.
# ⚠ Здесь стояло `sorted(...)[:cap]` — первые по алфавиту. Это не выборка: первыми оказались
#   листы `100003Q*`, где под эталоном прод-тушь есть лишь на 35-48% строк (на случайных K≥3 —
#   80-100%). Замер по ним показал ноль и означал бы «сборка не работает», хотя означал только
#   «взяты худшие листы». Тот же класс ошибки, что §6.118 (шардинг по размеру).
_all = sorted({sh for (sh, t), K in KOF.items() if K >= a.mink})
_rng = np.random.default_rng(a.seed)
cand = [_all[k] for k in _rng.permutation(len(_all))[:a.cap]]
mine = [s for k, s in enumerate(cand) if k % n == i]
print(f"★ ШАРД {i}/{n}: листов {len(mine)} из {len(cand)}")


def runs_of(ink, y, X0, X1):
    r = ink[y, X0:X1]
    if not r.any():
        return []
    d = np.diff(r.astype(np.int8))
    st = list(np.flatnonzero(d == 1) + 1) + ([0] if r[0] else [])
    en = list(np.flatnonzero(d == -1) + 1) + ([len(r)] if r[-1] else [])
    st, en = sorted(st), sorted(en)
    return sorted(X0 + (s + e) / 2.0 for s, e in zip(st, en) if e - s <= a.maxw)


def segments(rows, Y0):
    """Штрихи трекером; РЕЖЕМ в узле — там, где рядом с предсказанием есть второй ран."""
    active, done = [], []
    for idx, xs in enumerate(rows):
        y = idx + Y0
        for s in active:
            k = list(s["pts"])[-30:]
            s["pred"] = (s["last_x"] if len(k) < 5 else
                         s["pts"][k[-1]] + (s["pts"][k[-1]] - s["pts"][k[0]]) /
                         max(1, k[-1] - k[0]) * (y - k[-1]))
        cand_ = sorted((abs(s["pred"] - x), si, xi)
                       for si, s in enumerate(active) for xi, x in enumerate(xs)
                       if abs(s["pred"] - x) <= a.xt)
        us, ux = set(), set()
        for d0, si, xi in cand_:
            if si in us or xi in ux:
                continue
            # ★★ УЗЕЛ: рядом с предсказанием есть ВТОРОЙ ран ⇒ продолжение неоднозначно. Не гадаем,
            #    а закрываем сегмент: 90.6% материала однозначно и без этих мест (§6.191).
            near2 = sum(1 for x in xs if abs(active[si]["pred"] - x) <= a.near)
            if near2 > 1:
                us.add(si)
                continue
            us.add(si); ux.add(xi)
            active[si]["pts"][y] = xs[xi]
            active[si]["last_x"] = xs[xi]
            active[si]["last_y"] = y
        for xi, x in enumerate(xs):
            if xi not in ux:
                active.append(dict(pts={y: x}, last_x=x, last_y=y, pred=x))
        keep = []
        for s in active:
            (keep if y - s["last_y"] <= a.gap else done).append(s)
        active = keep
    return [s["pts"] for s in done + active if len(s["pts"]) >= a.minseg]


def match(rows, cols, ok):
    """Максимальное 1:1 внутри трека — механика `_name_cost_prod.py` дословно."""
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


def stats(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


cnt, done_sheets = Counter(), 0
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
    done_sheets += 1

    # ── выдачи прода и декодера на тех же кривых ────────────────────────────────────────────
    dn = f"{Path(sh).stem[:40]}_{hashlib.md5(Path(sh).stem.encode('utf-8')).hexdigest()[:8]}"
    outs = {}
    for tag, root in (("прод", a.prod), ("декодер", a.dec)):
        d = Path(a.ts) / root / dn
        g = next(iter(sorted(d.glob("*_auto.nlgx"))), None) if d.is_dir() else None
        outs[tag] = ({c["name"]: dense(c) for c in extract(str(g))["curves"]
                      if M.mnem_root(c["name"]) != "DA"} if g else {})

    for t, ns in tracks.items():
        allx = [x for nm in ns for x in gts[nm].values()]
        X0, X1 = max(0, int(min(allx)) - 40), min(W, int(max(allx)) + 40)
        ys = sorted(set().union(*[set(gts[nm]) for nm in ns]))
        Y0, Y1 = ys[0], ys[-1]
        rows = [runs_of(ink, y, X0, X1) for y in range(Y0, Y1 + 1)]
        segs = segments(rows, Y0)
        cnt["сегментов"] += len(segs)
        used = set()
        for nm in ns:
            g = gts[nm]
            cnt["кривых"] += 1
            # ── ОРАКУЛ: берём сегменты, ложащиеся на эту кривую, каждый — не более чем одной ──
            mine_ = []
            for k, s in enumerate(segs):
                if k in used:
                    continue
                com = [y for y in s if y in g]
                if len(com) < 10:
                    continue
                if float(np.median([abs(s[y] - g[y]) for y in com])) <= 3.0:
                    mine_.append(k)
            for k in mine_:
                used.add(k)
            if not mine_:
                cnt["без сегментов"] += 1
            asm = {}
            for k in mine_:
                asm.update(segs[k])
            m, c = stats(asm, g)
            if HON(m, c):
                cnt["★ сборка без мостика"] += 1
            if asm:                       # ── с мостиком: линейно между покрытыми строками ──
                kk = sorted(asm)
                allr = [y for y in sorted(g) if kk[0] <= y <= kk[-1]]
                iv = np.interp(allr, kk, [asm[y] for y in kk])
                br = {y: float(v) for y, v in zip(allr, iv)}
                m2, c2 = stats(br, g)
                if HON(m2, c2):
                    cnt["★ сборка с мостиком"] += 1
        # ⚠⚠ ВЕДУЩИЙ СЧЁТ — БЕЗЫМЯННЫЙ 1:1 (§6.143). Здесь стояло сравнение ПО ИМЕНИ
        # (`outs[tag].get(nm)`), а сборка считалась фактически безымянно — и стенд печатал
        # «50.8% против 27.1% у прода», сравнивая РАЗНЫЕ величины. На безымянном счёте прод на
        # этих же листах даёт около 50%, то есть преимущества у сборки не было.
        for tag in ("прод", "декодер"):
            W_ = [k for k in outs[tag] if tm.get(k) == t]
            ok = {(g2, w): HON(*stats(outs[tag][w], gts[g2])) for g2 in ns for w in W_}
            cnt[tag] += len(match(ns, W_, ok))
    if si % 5 == 0 or si == len(mine):
        print(f"  {si}/{len(mine)}  кривых {cnt['кривых']}, сборка без мостика {cnt['★ сборка без мостика']}")

out = TS / f"{a.tag}_{i}of{n}.pkl"
pickle.dump(dict(cnt=dict(cnt), sheets=done_sheets), open(out, "wb"))
print(f"★ ГОТОВО: листов {done_sheets}, кривых {cnt['кривых']} → {out.name}")
for k in ("прод", "декодер", "★ сборка без мостика", "★ сборка с мостиком", "без сегментов"):
    print(f"   {k}: {cnt[k]}")

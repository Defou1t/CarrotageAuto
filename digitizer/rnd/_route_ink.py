r"""_route_ink.py — ТУШЬ ПОД КАЖДЫМ ИЗ ДВУХ КАНДИДАТОВ СЛОТА: последний непроверенный признак B1a (17.09).

ОТКУДА ВОПРОС. Выбор пути по слоту (`_route_learn.py`, §6.201) закрыт четырежды по ПРИЗНАКАМ
ГЕОМЕТРИИ трасс (длина, дрожание, согласие путей, разброс): держанно +4 из +189 потолка, и «+4»
оказалось удачей раскладки (20 случайных раскладок: 1099…1109). В описи остался ОДИН непроверенный
признак — **тушь ПОД каждым кандидатом**: своя ли это тушь по ширине и цвету. Идея: неверный
кандидат — это трасса, перескочившая на СОСЕДНЮЮ линию; вдоль неё меняется ширина штриха или его
цвет, у верной — нет. Геометрия этого не видит (обе трассы гладкие), растр — может.

ЧТО ДЕЛАЕТ. Один проход по растрам поля 1123 листов: для каждого кандидата (слот × путь A/B) из
`route_learn_cache.pkl` берётся его плотная трасса из замороженной выдачи (A = `ab_wellmap/A`,
B = `ab_rdhonest/B`) и по растру считаются признаки туши:
  cov     — доля строк, где под трассой (±2 px) есть тушь (paper − V ≥ 90, вне сетки — как V3/§6.133);
  wid_med — медианная ширина штриха под трассой, px;
  wid_dev — доля строк с тушью, где ширина отличается от медианы больше чем на max(1, 0.5·медианы);
  col_dom — доля строк с тушью, чей цветовой класс (black/red/orange/green/blue) совпадает с модальным;
  n_col   — число классов с долей ≥ 0.1 (2+ = трасса идёт по туши разных цветов);
  gap_max — самый длинный участок без туши, долей от длины трассы;
  n_rows  — строк трассы (для веса).
Кэш → `route_ink_cache.pkl`: {(sheet, track, slot, tag): [7 признаков]}. Дальше `_route_learn.py --ink`
добавляет их к признакам пары и повторяет ТУ ЖЕ перекрёстную оценку с теми же тремя контролями.

⚠ Эталон сюда не входит (§6.146): всё считается из трассы кандидата и растра.
⚠ Шардируется по листам (`--shard i/n`), результат складывается `--merge`.

  <ComfyUI>\python_embeded\python.exe _route_ink.py --shard 0/4
  <ComfyUI>\python_embeded\python.exe _route_ink.py --merge
"""
import sys, argparse, pickle, hashlib, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from extract_nlgx import extract
from _multi_replica_probe import dense
from dataset_build import find_image
from auto import meta as M
from auto import imaging as im
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--cachef", default="route_learn_cache.pkl")
ap.add_argument("--a", default="ab_wellmap/A")
ap.add_argument("--b", default="ab_rdhonest/B")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--out", default="route_ink_cache.pkl")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--merge", action="store_true")
ap.add_argument("--cap", type=int, default=0)
a = ap.parse_args()
TS = Path(a.ts)
DELTA = 90.0                 # §6.133 / V3: тушь = бумага строки − V ≥ 90
HALF = 15                    # полуширина окна вокруг трассы, px
NEAR = (0, -1, 1, -2, 2)     # порядок поиска туши около трассы

if a.merge:
    parts = sorted(TS.glob(f"{Path(a.out).stem}_*of*.pkl"))
    out = {}
    for p in parts:
        out.update(pickle.load(open(p, "rb")))
    pickle.dump(out, open(TS / a.out, "wb"))
    print(f"слито {len(parts)} шардов → {len(out)} кандидатов → {TS / a.out}")
    sys.exit(0)

TR = pickle.load(open(TS / a.cachef, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q
for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
    SRC.setdefault(q.name, q)


def rd(root, sh):
    stem = Path(sh).stem
    pd = TS / root / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    if not got:
        return {}
    return {c["name"]: dense(c) for c in extract(str(got))["curves"]
            if M.mnem_root(c["name"]) != "DA"}


def ink_feats(tr, dark, cls, H, W):
    """→ [cov, wid_med, wid_dev, col_dom, n_col, gap_max, n_rows] по одной трассе."""
    ys = np.array(sorted(y for y in tr if 0 <= y < H), int)
    if len(ys) < 10:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, float(len(ys))]
    xs = np.clip(np.round([tr[y] for y in ys]).astype(int), HALF, W - HALF - 1)
    off = np.arange(-HALF, HALF + 1)
    idx = xs[:, None] + off[None, :]
    Dk = dark[ys[:, None], idx]                              # (n, 2·HALF+1) bool
    c = np.full(len(ys), -1, int)
    for o in NEAR:                                           # ближайшая тушь около трассы
        col = HALF + o
        sel = (c < 0) & Dk[:, col]
        c[sel] = col
    has = c >= 0
    cov = float(has.mean())
    if not has.any():
        return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, float(len(ys))]
    # ширина штриха: тушь, принадлежащая тому же рану, что пиксель c (ран = равный cumsum(~Dk))
    blk = np.cumsum(~Dk, axis=1)
    cc = np.where(has, c, 0)
    same = Dk & (blk == blk[np.arange(len(ys)), cc][:, None])
    wid = same.sum(1)[has].astype(float)
    wmed = float(np.median(wid))
    wdev = float(np.mean(np.abs(wid - wmed) > max(1.0, 0.5 * wmed)))
    # цвет: класс пикселя под трассой (0 black, 1 red, 2 orange, 3 green, 4 blue)
    kk = cls[ys[has], xs[has] + (c[has] - HALF)]
    cnt = np.bincount(kk, minlength=5).astype(float) / has.sum()
    col_dom = float(cnt.max())
    n_col = float((cnt >= 0.1).sum())
    # самый длинный участок без туши (по строкам трассы)
    gap, best, run = ~has, 0, 0
    for g in gap:
        run = run + 1 if g else 0
        best = max(best, run)
    return [cov, wmed, wdev, col_dom, n_col, best / len(ys), float(len(ys))]


i, n = map(int, a.shard.split("/"))
sheets = sorted({r["sheet"] for r in TR})
mine = sheets[len(sheets) * i // n:len(sheets) * (i + 1) // n]
if a.cap:
    mine = mine[:a.cap]
by = defaultdict(list)
for r in TR:
    by[r["sheet"]].append(r)
print(f"★ ШАРД {i}/{n}: листов {len(mine)} из {len(sheets)}")
out, miss, t0 = {}, Counter(), time.time()
for k, sh in enumerate(mine, 1):
    q = SRC.get(sh)
    img = find_image(q) if q else None
    if not img:
        miss["нет растра"] += 1
        continue
    try:
        rgb = im.load_rgb(str(img))
        H, W = rgb.shape[:2]
        V = im.value_channel(rgb)
        paper = np.empty(H, np.float32)
        for y0 in range(0, H, 4096):
            y1 = min(H, y0 + 4096)
            paper[y0:y1] = np.percentile(V[y0:y1, ::4], 90, axis=1)
        dark = (paper[:, None] - V.astype(np.float32)) >= DELTA
        dark &= ~im.structure_mask(rgb, DEFAULT.cv)
        cm = im.color_channels(rgb, DEFAULT.cv)
        cls = np.zeros((H, W), np.uint8)
        for j, nm in enumerate(("red", "orange", "green", "blue"), 1):
            cls[cm[nm]] = j
        WA, WB = rd(a.a, sh), rd(a.b, sh)
        for r in by[sh]:
            for s, tag in r["have"]:
                tr = (WA if tag == "A" else WB).get(s)
                if tr is None:
                    miss["кандидат без трассы"] += 1
                    continue
                out[(sh, r["track"], s, tag)] = ink_feats(tr, dark, cls, H, W)
        del rgb, V, dark, cls, cm                      # мемо-кэш imaging умирает вместе с массивом
    except Exception as e:
        miss[f"{type(e).__name__}"] += 1
        print(f"  ⚠ {sh[:50]}: {type(e).__name__}: {str(e)[:80]}")
    if k % 25 == 0:
        print(f"  … {k}/{len(mine)}  кандидатов {len(out)}  {time.time()-t0:.0f}с")
tag = f"{Path(a.out).stem}_{i}of{n}.pkl"
pickle.dump(out, open(TS / tag, "wb"))
print(f"★ ГОТОВО: листов {len(mine)-sum(miss.values())}, пропуски {dict(miss)}, кандидатов {len(out)} → {TS / tag}")

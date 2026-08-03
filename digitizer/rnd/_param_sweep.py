r"""_param_sweep.py — СВИП ЧЕТЫРЁХ ПАРАМЕТРОВ ПРОД-ТРАССИРОВЩИКА НА ЕГО ЖЕ КОДЕ (§6.101 шаг 1).

ЗАЧЕМ. `trace2d.trace_line` имеет четыре константы, ни одна из которых не свипалась на прод-коде:
  `band_pad=8`   — допуск вокруг размаха ЛИНИИ (не шага!), задаёт окно поиска ранов;
  `slmax=30`     — кламп скорости (и предсказание, и коаст, и `_extend_ends`);
  `wide_run=14`  — ран шире этого = спайк ⇒ берём ВЕРШИНУ выноса, а не центр;
  `jump_limit`   — предохранитель прыжка в ветке «нет рана под предсказанием»; сейчас None
                   (прыжок НЕ ограничен ничем — §6.19/§6.57).
⚠ §6.99 свипал «band_pad» на СВОЁМ стенде, где это был предел шага на строку; §6.100 это отозвал —
у прода такого параметра нет. Здесь свипается НАСТОЯЩИЙ `band_pad` настоящей функции.

КАК. Тот же стенд, что и `_start_probe.py`: линия восстанавливается из пула, гоняется НАСТОЯЩАЯ
`trace_line`. Калибровка стенда проверена ПО СОСТАВУ (а не по доле, как в отозванном §6.97):
из 174 прод-честных кривых стенд воспроизводит 169 (97.1%), Jaccard 0.949.
⚠ ОГОВОРКА, КОТОРУЮ НАДО ЦИТИРОВАТЬ ВМЕСТЕ С ЧИСЛАМИ: в пуле нет `line.x_lo/x_hi`, поэтому размах
линии восстановлен по её СОБСТВЕННОЙ трассе. Трасса лежит внутри [x_lo−8, x_hi+8], значит
восстановленный размах не шире настоящего +8px с каждой стороны, но может быть УЖЕ. Для `band_pad`
это смещает нуль отсчёта ⇒ вывод про band_pad обязателен к перепроверке A/B на выдаче.

РЕЖИМЫ:
  --grid oat   (умолч.) — по одному параметру от дефолта: 1 + 6 + 5 + 5 + 5 = 22 сочетания;
  --grid pair  — сетка по двум лучшим осям, задаётся --pair "band_pad=8,12,16,24 slmax=30,60,120";
  --grid file  — сочетания из json-файла --combos.

  <ComfyUI>\python_embeded\python.exe _param_sweep.py [--shard 0/8] [--grid oat] [--limit N]
"""
import sys, argparse, pickle, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from _multi_replica_probe import dense
from auto import imaging as im, trace2d as T, meta as M
from auto.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div"])
ap.add_argument("--shard", default="0/1")
ap.add_argument("--grid", default="oat", choices=["oat", "pair", "file"])
ap.add_argument("--pair", default="", help='напр. "band_pad=8,16,24,40 slmax=30,60,120"')
ap.add_argument("--combos", default="", help="json-список словарей для --grid file")
ap.add_argument("--limit", type=int, default=0, help="взять только N листов (замер скорости)")
ap.add_argument("--tag", default="oat")
ap.add_argument("--out", default=r"F:\nds\output\taskS\param_sweep")
# ★ КЭШ МАСОК. Замер 01.08: 8 шардов на одном HDD дают CPU 14% — упирается в ЧТЕНИЕ СКАНОВ, а не
# в счёт. Маска цвета (`_color_fg`) кладётся упакованной по битам И СЖАТОЙ: 76.8 → 2.3 МБ на
# крупном YULIIV (маска разреженная, сжимается в 33 раза), запись 0.4 с. Это меньше самого JPEG,
# так что кэш живёт рядом с данными на F и заменяет собой декодирование 20 Мпикс + dark/color/
# structure. Кэшируется РОВНО массив, уходящий в `trace_line`: путь прод-кода не меняется ни на
# байт, и это проверено — 0 расхождений из 644 (кэш против прогона по картинке).
ap.add_argument("--fgcache", default=r"F:\nds\output\taskS\fgcache", help="пусто = не кэшировать")
ap.add_argument("--dump-every", type=int, default=10, help="дописывать результат каждые N листов")
ap.add_argument("--shard-mode", default="block", choices=["block", "stride"])
ap.add_argument("--cache-only", action="store_true",
                help="только прогреть кэш масок (фаза 1, мало процессов — упирается в диск)")
a = ap.parse_args()
SH_I, SH_N = (int(v) for v in a.shard.split("/"))
P = Config().cv
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9

# ── СОЧЕТАНИЯ ────────────────────────────────────────────────────────────────────────────────
BASE = dict(band_pad=8, slmax=30.0, wide_run=14, jump_limit=None)
# ⚠ Дефолт ОБЯЗАН быть в списке первым: все дельты считаются от него на ОДНОЙ И ТОЙ ЖЕ выборке,
# иначе сравнение поедет на кривых, где стенд упал (§6.71 «ноль различий — проверь, что сравниваешь»).
OAT = {"band_pad": [4, 12, 16, 24, 40, 80],
       "slmax": [10.0, 20.0, 45.0, 60.0, 120.0],
       "wide_run": [8, 10, 20, 30, 50, 10 ** 6],
       "jump_limit": [10, 20, 40, 80, 160]}


def name_of(c):
    d = [f"{k}={c[k]}" for k in ("band_pad", "slmax", "wide_run", "jump_limit") if c[k] != BASE[k]]
    return "base" if not d else " ".join(d)


if a.grid == "oat":
    combos = [dict(BASE)] + [dict(BASE, **{k: v}) for k, vs in OAT.items() for v in vs]
elif a.grid == "pair":
    axes = {}
    for part in a.pair.split():
        k, vs = part.split("=")
        axes[k] = [float(v) if k == "slmax" else int(v) for v in vs.split(",")]
    combos = [dict(BASE)]
    keys = list(axes)
    import itertools
    for vals in itertools.product(*(axes[k] for k in keys)):
        c = dict(BASE, **dict(zip(keys, vals)))
        if c not in combos:
            combos.append(c)
else:
    combos = [dict(BASE)] + [dict(BASE, **c) for c in json.load(open(a.combos, encoding="utf-8"))]
NAMES = [name_of(c) for c in combos]
print(f"сочетаний {len(combos)}: {', '.join(NAMES)}")


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


WELL = {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WELL[q.name] = wlg.parent.name
for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
    WELL.setdefault(q.name, "Semeguniv")
WLG = {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WLG[q.name] = q
for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
    WLG.setdefault(q.name, q)

# ⚠⚠ СПИСОК ЛИСТОВ СТРОИТСЯ ПО ИМЕНАМ ФАЙЛОВ, А НЕ ЧТЕНИЕМ ПУЛОВ. Унаследованный от `_start_probe`
# цикл `pickle.load` по всем пяти каталогам читал 2.35 ГБ РАДИ ОДНОГО ПОЛЯ `name` — на 16 шардах
# это 38 ГБ с одного HDD ДО начала счёта, и шарды по полчаса не выходили из стартового цикла
# (CPU 14%, ни одного чекпойнта). `_pool_oracle.py` кладёт дамп как `{nlgx.stem[:60]}.pkl`, поэтому
# имя файла — готовый ключ дедупликации, а сам пул читается ровно один раз и ровно тогда, когда
# лист считается. Совпадение ключа с `d["name"]` проверяется при чтении и говорится вслух.
files, seen = [], set()
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        if f.stem in seen:
            continue
        seen.add(f.stem); files.append((f, f.stem))
# ★ ШАРД БЛОКОМ, А НЕ ЧЕРЕЗ ШАГ. Сканы одной скважины лежат в одном каталоге, а имена листов
# начинаются со скважины ⇒ соседние по списку листы физически рядом на диске. Шаг `i % N`
# раскидывал каждый процесс по ВСЕМУ архиву, и 16 читателей гоняли головку HDD впустую (замер:
# 19% CPU при 10 МБ чтения на лист). Блок сохраняет локальность. `stride` оставлен для сверки.
files.sort(key=lambda q: q[1])
if a.shard_mode == "block":
    lo = len(files) * SH_I // SH_N; hi = len(files) * (SH_I + 1) // SH_N
    files = files[lo:hi]
else:
    files = [q for i, q in enumerate(files) if i % SH_N == SH_I]
if a.limit:
    files = files[:a.limit]
print(f"шард {SH_I}/{SH_N} ({a.shard_mode}): листов {len(files)} "
      f"(всего уникальных дампов {len(seen)})")

CDIR = Path(a.fgcache) if a.fgcache else None
if CDIR:
    CDIR.mkdir(parents=True, exist_ok=True)


def masks(name, n, img, colors):
    """Маски нужных цветов листа. Из кэша, если он есть; иначе считаем и кладём в кэш.
    ⚠ Кэш хранит РОВНО `T._color_fg(...)` — тот массив, который уходит в `trace_line`."""
    cf = CDIR / f"{name}.npz" if CDIR else None
    if cf is not None and cf.exists():
        try:
            z = np.load(cf)
            H, W = int(z["shape"][0]), int(z["shape"][1])
            got = {c: np.unpackbits(z[f"m_{c}"], count=H * W).reshape(H, W).astype(bool)
                   for c in colors if f"m_{c}" in z}
            if len(got) == len(colors):
                return got
        except Exception as e:
            print(f"  ⚠ кэш {name[:36]} битый ({type(e).__name__}), пересчитываю")
    rgb = im.load_rgb(str(img))
    out = {}
    for c0 in colors:
        try:
            out[c0] = T._color_fg(rgb, c0, P)
        except Exception:
            out[c0] = T._color_fg(rgb, "black", P)
    if cf is not None:
        try:
            np.savez_compressed(cf.with_suffix(".tmp.npz"), shape=np.array(rgb.shape[:2]),
                                **{f"m_{c}": np.packbits(m) for c, m in out.items()})
            cf.with_suffix(".tmp.npz").replace(cf)
        except Exception as e:
            print(f"  ⚠ кэш {name[:36]} не записан: {type(e).__name__}: {e}")
    del rgb
    return out


Path(a.out).mkdir(parents=True, exist_ok=True)
DST = Path(a.out) / f"{a.tag}_{SH_I}of{SH_N}.pkl"
rows = []
t0 = time.time()
for k, (pf, stem) in enumerate(files, 1):
    d = pickle.load(open(pf, "rb"))
    name = d["name"]
    if not name.startswith(stem[:40]):
        print(f"  ⚠ имя дампа {stem[:40]!r} расходится с полем name {name[:40]!r}")
    n = WLG.get(name)
    img = find_image(n) if n else None
    if not img:
        continue
    try:
        G = extract(str(n))
        raws = {c["name"]: c for c in G["curves"]
                if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        gts = {nm: dense(c) for nm, c in raws.items()}
        if not gts or not d["lines"]:
            continue
        fgc = masks(name, n, img, {ln["color"] or "black" for ln in d["lines"]})
    except Exception as e:
        print(f"  ПАДЕНИЕ {name[:40]}: {type(e).__name__}: {e}")
        continue
    if a.cache_only:
        del fgc
        if k % a.dump_every == 0:
            sp = (time.time() - t0) / k
            print(f"  прогрев {k}/{len(files)}, {sp:.1f} с/лист, "
                  f"осталось ~{sp * (len(files) - k) / 60:.0f} мин")
        continue

    for nm, gt in gts.items():
        # линия прода, ближайшая к кривой по медиане трассы (та, что её вела бы) — как в _start_probe
        best, bd = None, 1e18
        for ln in d["lines"]:
            m, c = err(ln["tr"], gt)
            v = m if m is not None else 1e9
            if v < bd:
                best, bd = ln, v
        if best is None:
            continue
        xs_ = np.array(list(best["tr"].values()), float)
        ys_ = np.array(list(best["tr"].keys()), int)
        if len(xs_) < 30:
            continue
        fg = fgc.get(best["color"] or "black")
        L = SimpleNamespace(x_lo=float(xs_.min()), x_hi=float(xs_.max()),
                            y0=int(ys_.min()), y1=int(ys_.max()),
                            x_center=float(best["x_center"]), color=best["color"],
                            behavior=best["behavior"], track_index=best["track"],
                            confidence=best.get("conf", 1.0))
        res = []
        for c in combos:
            try:
                tr = T.trace_line(fg, L, None, P, **c)
            except Exception:
                tr = {}
            m, cv = err(tr, gt)
            res.append((1 if HON(m, cv) else 0,
                        float(m) if m is not None else float("nan"), float(cv)))
        rows.append(dict(sheet=name, well=WELL.get(name, "?"), curve=nm,
                         pool=1 if HON(*err(best["tr"], gt)) else 0,
                         span=float(xs_.max() - xs_.min()), res=res))
    del fgc
    if k % a.dump_every == 0:
        # ⚠ Дамп ПРОМЕЖУТОЧНЫЙ и атомарный: свип идёт часами, и сводку надо уметь снимать на ходу,
        # не дожидаясь конца (и не теряя всё, если шард упадёт).
        pickle.dump({"names": NAMES, "combos": combos, "rows": rows, "done": k,
                     "total": len(files)}, open(DST.with_suffix(".tmp"), "wb"))
        DST.with_suffix(".tmp").replace(DST)
        sp = (time.time() - t0) / k
        print(f"  {k}/{len(files)} листов, кривых {len(rows)}, {sp:.1f} с/лист, "
              f"осталось ~{sp * (len(files) - k) / 60:.0f} мин")

dst = DST
pickle.dump({"names": NAMES, "combos": combos, "rows": rows, "done": len(files),
             "total": len(files)}, open(dst, "wb"))
print(f"\nкривых {len(rows)}, прод-пул {sum(r['pool'] for r in rows)}")
for i, nm in enumerate(NAMES):
    h = sum(r["res"][i][0] for r in rows)
    print(f"  {nm:<22} {h:>4}")
print(f"записано → {dst}  ({(time.time()-t0)/60:.1f} мин)")

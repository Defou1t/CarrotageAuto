r"""_start_probe.py — СКОЛЬКО СТОИТ СТАРТОВАЯ ТОЧКА. Прогон НАСТОЯЩЕГО прод-трассировщика.

⚠ ЗАЧЕМ (§6.101). Моё грубое ведение нашло 53 кривые, которых нет у прода, — но оно стартовало с
ИСТИННОЙ первой точки эксперта, а прод выбирает старт сам (ран, ближайший к `line.x_center`). Пока
эти два фактора не разделены, «+53 кривые» цитировать нельзя.

КАК РАЗДЕЛЕНО. Берётся `trace2d.trace_line` — тот самый код, что работает в проде, — и запускается
ДВАЖДЫ на одной и той же маске и одной и той же линии:
  свой   — `x_center` линии как есть (что делает прод);
  ★ старт — `x_center` подменён на ПЕРВУЮ ТОЧКУ ЭКСПЕРТА (подсказка).
Всё остальное (полоса по размаху линии, кламп скорости, предпочтение перекрывающего рана, вершина
широкого рана) идентично, потому что это одна и та же функция.
⚠ `x_center` у прода служит и стартом, и ориентиром для вершины широкого рана (`base`), поэтому замер
меряет ВЛИЯНИЕ ОРИЕНТИРА в целом, а не только первой строки. Это надо назвать честно.

Линия восстанавливается из пула: `x_lo/x_hi/y0/y1` — размах её собственной трассы, `x_center` и цвет —
из дампа. Это та линия, которую прод реально вёл.

  <ComfyUI>\python_embeded\python.exe _start_probe.py [--shard 0/8]
"""
import sys, argparse, pickle
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
ap.add_argument("--shard-mode", default="block", choices=("block", "stride"))
ap.add_argument("--limit", type=int, default=0, help="взять первые N листов шарда (проверка)")
ap.add_argument("--out", default=r"F:\nds\output\taskS\start_probe")
a = ap.parse_args()
SH_I, SH_N = (int(v) for v in a.shard.split("/"))
P = Config().cv
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


# ⚠⚠ СПИСОК ЛИСТОВ СТРОИТСЯ ПО ИМЕНАМ ФАЙЛОВ, А НЕ ЧТЕНИЕМ ПУЛОВ. Прежний цикл делал `pickle.load`
# по всем пяти каталогам (2.35 ГБ) РАДИ ОДНОГО ПОЛЯ `name`, а рабочий цикл ниже читал тот же дамп
# ВТОРОЙ раз. На 16 шардах это 38 ГБ с одного HDD ДО начала счёта, и шарды по полчаса не выходили из
# стартового цикла. `_pool_oracle.py` кладёт дамп как `{nlgx.stem[:60]}.pkl`, поэтому имя файла —
# готовый ключ дедупликации, а сам пул читается ровно один раз и ровно тогда, когда лист считается.
# Совпадение ключа с `d["name"]` проверяется при чтении и говорится вслух (см. рабочий цикл).
files, seen = [], set()
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        if f.stem in seen:
            continue
        seen.add(f.stem); files.append((f, f.stem))
# ★ ШАРД БЛОКОМ, А НЕ ЧЕРЕЗ ШАГ. Сканы одной скважины лежат в одном каталоге, а имена листов
# начинаются со скважины ⇒ соседние по списку листы физически рядом на диске. Шаг `i % N` раскидывал
# каждый процесс по ВСЕМУ архиву, и читатели гоняли головку HDD впустую. Блок сохраняет локальность.
# ⚠ `stride` оставлен ДЛЯ СВЕРКИ и потому НЕ сортирует: он обязан выбирать ровно те же листы, что
# выбирал прежний код, иначе сверять новую редакцию со старой нечем.
if a.shard_mode == "block":
    files.sort(key=lambda q: q[1])
    lo = len(files) * SH_I // SH_N; hi = len(files) * (SH_I + 1) // SH_N
    files = files[lo:hi]
else:
    files = [q for i, q in enumerate(files) if i % SH_N == SH_I]
if a.limit:
    files = files[:a.limit]
print(f"шард {SH_I}/{SH_N} ({a.shard_mode}): листов {len(files)} "
      f"(всего уникальных дампов {len(seen)})")

WLG = {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WLG[q.name] = q
for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
    WLG.setdefault(q.name, q)

rows = []
for k, (pf, stem) in enumerate(files, 1):
    d = pickle.load(open(pf, "rb"))          # единственное чтение пула
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
        rgb = im.load_rgb(str(img))
        fgc = {}
        for ln in d["lines"]:
            c0 = ln["color"] or "black"
            if c0 not in fgc:
                try:
                    fgc[c0] = T._color_fg(rgb, c0, P)
                except Exception:
                    fgc[c0] = T._color_fg(rgb, "black", P)
    except Exception as e:
        print(f"  ПАДЕНИЕ {name[:40]}: {type(e).__name__}: {e}")
        continue

    for nm, gt in gts.items():
        ys = sorted(gt)
        x0 = gt[ys[0]]
        # линия прода, ближайшая к кривой по медиане трассы (та, что её вела бы)
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
        base = SimpleNamespace(x_lo=float(xs_.min()), x_hi=float(xs_.max()),
                               y0=int(ys_.min()), y1=int(ys_.max()),
                               x_center=float(best["x_center"]), color=best["color"],
                               behavior=best["behavior"], track_index=best["track"],
                               confidence=best.get("conf", 1.0))
        hint = SimpleNamespace(**{**base.__dict__, "x_center": float(x0)})
        out = {}
        for tag, L in (("own", base), ("hint", hint)):
            try:
                tr = T.trace_line(fg, L, None, P)
            except Exception:
                tr = {}
            m, c = err(tr, gt)
            out[tag] = (1 if HON(m, c) else 0, m if m is not None else float("nan"))
        rows.append(dict(sheet=name, curve=nm, own=out["own"][0], hint=out["hint"][0],
                         med_own=out["own"][1], med_hint=out["hint"][1],
                         pool=1 if HON(*err(best["tr"], gt)) else 0))
    del rgb, fgc
    if k % 10 == 0:
        print(f"  {k}/{len(files)} листов, кривых {len(rows)}")

Path(a.out).mkdir(parents=True, exist_ok=True)
dst = Path(a.out) / f"rows_{SH_I}of{SH_N}.pkl"
pickle.dump(rows, open(dst, "wb"))
o = sum(r["own"] for r in rows); h = sum(r["hint"] for r in rows); p = sum(r["pool"] for r in rows)
print(f"\nкривых {len(rows)}: свой старт {o}, ★подсказанный {h} ({h-o:+d}), трасса из пула {p}")
print(f"записано → {dst}")

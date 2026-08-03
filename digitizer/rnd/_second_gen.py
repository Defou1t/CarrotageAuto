r"""_second_gen.py — ВТОРОЙ ГЕНЕРАТОР ТРАСС В ПУЛ (§6.101, шаг 2 очереди).

ЗАЧЕМ. §6.101 показал по составу: прод-трассировщик и моё грубое ведение ДОПОЛНЯЮТ друг друга —
пересечение честных лишь 49%, и у грубого ведения 53 кривые, которых у прода нет вовсе. У прода уже
есть механизм «много трасс + отбор» (пул из 6597 трасс и раскладка поверх), поэтому второе ведение
не нужно ВМЕСТО прода: его надо добавить ГЕНЕРАТОРОМ КАНДИДАТОВ в тот же пул.

⚠⚠ ЧЕМ ЭТОТ ЗАМЕР ЧЕСТНЕЕ §6.101. Там `self` искал раны в окне ±150px вокруг ИСТИННОГО размаха кривой
и стартовал с ИСТИННОЙ первой точки — обоих этих сведений у прода нет. Здесь генератор пользуется
ТОЛЬКО тем, что есть у линии: `x_lo/x_hi/y0/y1/x_center/color`. Старт — ран, ближайший к `x_center`
(правило прода). Окно — размах ЛИНИИ плюс `--pad`.
⚠ И считается не только потолок: память проекта прямо предупреждает, что **гибрид обязан уважать 1:1**
(одна трасса — один слот). Поэтому рядом с «потолком пула» всегда печатается ОПТИМУМ 1:1 — то, что
отбор вообще может достать при действующем контракте выдачи.

ЧЕМ ГЕНЕРАТОР ОТЛИЧАЕТСЯ ОТ ПРОДА (в этом весь смысл — иначе он даст те же трассы):
  прод  — предсказание по скорости `clip(v,±slmax)`, предпочтение ПЕРЕКРЫВАЮЩЕГО рана,
          вершина широкого рана, коаст по инерции через разрыв;
  наш   — ближайший по центру ран к СВОЕМУ предыдущему выбору, без скорости, без вершины,
          на разрыве держит позицию. Ровно `self` из `_drift_metric.py`, но без подглядывания.

  <ComfyUI>\python_embeded\python.exe _second_gen.py [--shard 0/12] [--pad 8 40 150]
"""
import sys, argparse, pickle, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
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
ap.add_argument("--pad", type=int, nargs="+", default=[8, 40, 150],
                help="ширина окна вокруг размаха ЛИНИИ для второго генератора")
ap.add_argument("--fgcache", default=r"F:\nds\output\taskS\fgcache")
ap.add_argument("--out", default=r"F:\nds\output\taskS\second_gen")
ap.add_argument("--dump-every", type=int, default=10)
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


def trace_simple(fg, L, pad):
    """Второе ведение: ближайший по центру ран к своему предыдущему выбору. Никакого эталона."""
    H, W = fg.shape
    lo = max(0, int(L["x_lo"]) - pad); hi = min(W, int(L["x_hi"]) + pad + 1)
    if hi <= lo:
        return {}
    base = float(L["x_center"]); x = None
    tr = {}
    for y in range(max(0, int(L["y0"])), min(H, int(L["y1"]) + 1)):
        rr = im.row_runs(fg[y, lo:hi])
        if not rr:
            continue                       # разрыв — держим позицию, строку не пишем
        rr = [(p + lo, q + lo, c + lo) for p, q, c in rr]
        ref = base if x is None else x
        r = min(rr, key=lambda t: 0.0 if t[0] <= ref <= t[1]
                else min(abs(t[0] - ref), abs(t[1] - ref)))
        x = float(r[2]); tr[y] = x
    return tr


CDIR = Path(a.fgcache) if a.fgcache else None


def masks(name, img, colors):
    cf = CDIR / f"{name}.npz" if CDIR else None
    if cf is not None and cf.exists():
        try:
            z = np.load(cf)
            H, W = int(z["shape"][0]), int(z["shape"][1])
            got = {c: np.unpackbits(z[f"m_{c}"], count=H * W).reshape(H, W).astype(bool)
                   for c in colors if f"m_{c}" in z}
            if len(got) == len(colors):
                return got
        except Exception:
            pass
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
        except Exception:
            pass
    del rgb
    return out


def best_and_opt(traces, gts):
    """(потолок, оптимум 1:1) честных кривых для набора трасс. Потолок — лучшая трасса на кривую
    без ограничений; оптимум — жадное 1:1 по возрастанию ошибки (как в `_pool_oracle.py`)."""
    pairs, cap = [], 0
    for nm, gd in gts.items():
        best = None
        for ti, tr in enumerate(traces):
            m, c = err(tr, gd)
            if m is None:
                continue
            pairs.append((m, c, nm, ti))
            if best is None or (c >= 0.9, -m) > (best[1] >= 0.9, -best[0]):
                best = (m, c)
        cap += bool(best and HON(*best))
    pairs.sort(key=lambda q: (q[1] < 0.9, q[0]))
    got, used, opt = set(), set(), 0
    for m, c, nm, ti in pairs:
        if nm in got or ti in used:
            continue
        got.add(nm); used.add(ti); opt += HON(m, c)
    return cap, opt


WLG = {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WLG[q.name] = q
for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
    WLG.setdefault(q.name, q)

files, seen = [], set()
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        if f.stem in seen:
            continue
        seen.add(f.stem); files.append(f)
files.sort(key=lambda f: f.stem)
lo_, hi_ = len(files) * SH_I // SH_N, len(files) * (SH_I + 1) // SH_N
files = files[lo_:hi_]
print(f"шард {SH_I}/{SH_N}: листов {len(files)}, окна {a.pad}")

Path(a.out).mkdir(parents=True, exist_ok=True)
DST = Path(a.out) / f"rows_{SH_I}of{SH_N}.pkl"
rows, t0 = [], time.time()
for k, pf in enumerate(files, 1):
    d = pickle.load(open(pf, "rb"))
    name = d["name"]
    if not name.startswith(pf.stem[:40]):     # список строится по именам файлов ⇒ ключ надо сверять
        print(f"  ⚠ имя дампа {pf.stem[:40]!r} расходится с полем name {name[:40]!r}")
    n = WLG.get(name)
    img = find_image(n) if n else None
    if not img or not d["lines"]:
        continue
    try:
        G = extract(str(n))
        gts = {c["name"]: dense(c) for c in G["curves"]
               if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        if not gts:
            continue
        fgc = masks(name, img, {ln["color"] or "black" for ln in d["lines"]})
    except Exception as e:
        print(f"  ПАДЕНИЕ {name[:40]}: {type(e).__name__}: {e}")
        continue

    # линии восстанавливаются из ИХ ЖЕ трасс (как в `_start_probe`/`_param_sweep`): у пула нет
    # x_lo/x_hi/y0/y1. ⚠ Это ограничение стенда, и его надо цитировать вместе с числом.
    lines = []
    for ln in d["lines"]:
        xs_ = np.array(list(ln["tr"].values()), float)
        ys_ = np.array(list(ln["tr"].keys()), int)
        if len(xs_) < 30:
            continue
        lines.append(dict(x_lo=xs_.min(), x_hi=xs_.max(), y0=ys_.min(), y1=ys_.max(),
                          x_center=float(ln["x_center"]), color=ln["color"] or "black"))
    prod = [ln["tr"] for ln in d["lines"] if len(ln["tr"]) >= 30]
    if not lines or not prod:
        continue
    rec = dict(sheet=name, curves=len(gts), nprod=len(prod))
    rec["prod_cap"], rec["prod_opt"] = best_and_opt(prod, gts)
    for pad in a.pad:
        mine = [t for t in (trace_simple(fgc[L["color"]], L, pad) for L in lines) if len(t) >= 30]
        rec[f"mine{pad}_cap"], rec[f"mine{pad}_opt"] = best_and_opt(mine, gts)
        rec[f"uni{pad}_cap"], rec[f"uni{pad}_opt"] = best_and_opt(prod + mine, gts)
        rec[f"n{pad}"] = len(mine)
    rows.append(rec)
    del fgc
    if k % a.dump_every == 0:
        pickle.dump({"rows": rows, "pad": a.pad, "done": k, "total": len(files)},
                    open(DST.with_suffix(".tmp"), "wb"))
        DST.with_suffix(".tmp").replace(DST)
        sp = (time.time() - t0) / k
        print(f"  {k}/{len(files)} листов, кривых {sum(r['curves'] for r in rows)}, "
              f"{sp:.1f} с/лист, осталось ~{sp*(len(files)-k)/60:.0f} мин")

pickle.dump({"rows": rows, "pad": a.pad, "done": len(files), "total": len(files)}, open(DST, "wb"))
tot = lambda k: sum(r.get(k, 0) for r in rows)
print(f"\nлистов {len(rows)}, кривых {tot('curves')}")
print(f"  прод:      потолок {tot('prod_cap')}, оптимум1:1 {tot('prod_opt')}")
for pad in a.pad:
    print(f"  pad={pad:<4} моё: потолок {tot(f'mine{pad}_cap')}, оптимум {tot(f'mine{pad}_opt')}; "
          f"★ объединение: потолок {tot(f'uni{pad}_cap')}, ★оптимум {tot(f'uni{pad}_opt')}")
print(f"записано → {DST}")

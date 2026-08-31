r"""_supervision_audit.py — СКОЛЬКО СУПЕРВИЗИИ В КОРПУСЕ И РАЗРЕШИМА ЛИ ЗАДАЧА ПО ПОЛОЖЕНИЮ.

Нынешний ранжировщик раскладки обучается на 68 160 РУКОДЕЛЬНЫХ ПАР (14 признаков). При этом эталон
даёт плотную разметку «строка → x» на каждую кривую: это принципиально другой объём супервизии,
и он не используется для обучения вовсе — только для оценки.

Считаем три вещи, все по эталону, без пайплайна:
  1. ОБЪЁМ: сколько всего размеченных точек (кривая × строка) лежит в корпусе;
  2. РАЗРЕШИМОСТЬ ПО ПОЛОЖЕНИЮ: на какой доле строк две кривые ОДНОГО ТРЕКА сходятся ближе 3px —
     там положение не различает их в принципе, и нужен другой сигнал (тон, цвет, ширина штриха);
  3. ТРИВИАЛЬНОСТЬ: доля строк, где на треке кривая одна и выбирать не из чего.
⇒ Даёт потолок построчного декодера ДО того, как он построен: если неразрешимых строк единицы
процентов, задача «предсказать x всех K кривых на строке» хорошо поставлена.

  <ComfyUI>\python_embeded\python.exe _supervision_audit.py --shard 0/6
  <ComfyUI>\python_embeded\python.exe _supervision_audit.py --sum
"""
import sys, pickle, argparse, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--shard", default="0/6")
ap.add_argument("--out", default=r"F:/nds/output/taskS")
ap.add_argument("--sum", action="store_true")
a = ap.parse_args()
OUT = Path(a.out)
NEAR = 3.0


def collect(i, n):
    seen, files = set(), []
    for root in a.pools:
        for f in sorted(Path(root).glob("*.pkl")):
            if f.stem not in seen:
                seen.add(f.stem); files.append(f)
    mine = files[len(files) * i // n:len(files) * (i + 1) // n]
    print(f"★ ШАРД {i}/{n}: листов {len(mine)}")
    pts = alone = ambig = multi = 0
    kdist = defaultdict(int)
    for f in mine:
        d = pickle.load(open(f, "rb"))
        tr = {s["name"]: s["track"] for s in d["slots"]}
        byt = defaultdict(list)
        for nm, g in d["gts"].items():
            if g:
                byt[tr.get(nm)].append(g)
                pts += len(g)
        for t, curves in byt.items():
            if t is None:
                continue
            kdist[len(curves)] += 1
            if len(curves) == 1:
                alone += len(curves[0]); continue
            rows = defaultdict(list)
            for g in curves:
                for y, x in g.items():
                    rows[y].append(x)
            for y, xs in rows.items():
                if len(xs) == 1:
                    alone += 1; continue
                multi += len(xs)
                xs = sorted(xs)
                if any(xs[q + 1] - xs[q] <= NEAR for q in range(len(xs) - 1)):
                    ambig += len(xs)
    p = OUT / f"superv_{i}of{n}.json"
    p.write_text(json.dumps(dict(pts=pts, alone=alone, multi=multi, ambig=ambig,
                                 kdist={str(k): v for k, v in kdist.items()})), encoding="utf-8")
    print(f"точек {pts}, одиночных {alone}, в компании {multi}, неразрешимых {ambig} → {p}")


def summarise():
    fs = sorted(OUT.glob("superv_*of*.json"))
    if not fs:
        sys.exit("нет дампов")
    den = fs[0].stem.split("of")[1]
    fs = [f for f in fs if f.stem.endswith("of" + den)]
    T = defaultdict(int); kd = defaultdict(int)
    for f in fs:
        d = json.loads(f.read_text(encoding="utf-8"))
        for k in ("pts", "alone", "multi", "ambig"):
            T[k] += d[k]
        for k, v in d["kdist"].items():
            kd[int(k)] += v
    print(f"шардов {len(fs)} из {den}" + ("  ⚠ НЕПОЛНЫЙ" if len(fs) < int(den) else ""))
    print(f"\n★★ ОБЪЁМ СУПЕРВИЗИИ: {T['pts']:,} размеченных точек «кривая × строка»")
    print(f"   для сравнения: нынешний ранжировщик учится на 68 160 парах "
          f"⇒ в {T['pts']/68160:,.0f} раз меньше")
    tot = T["alone"] + T["multi"]
    print(f"\nстрок-кривых всего {tot:,}")
    print(f"   на треке кривая ОДНА (выбирать не из чего): {T['alone']:,} "
          f"({100*T['alone']/max(1,tot):.1f}%)")
    print(f"   в компании других:                          {T['multi']:,} "
          f"({100*T['multi']/max(1,tot):.1f}%)")
    print(f"   ★ из них НЕРАЗРЕШИМЫ по положению (соседи ближе {NEAR:.0f}px): {T['ambig']:,} "
          f"({100*T['ambig']/max(1,tot):.1f}% всех строк)")
    print(f"\nсколько кривых на треке: " +
          ", ".join(f"{k}→{v}" for k, v in sorted(kd.items())[:8]))
    print(f"\n⇒ если неразрешимых единицы процентов, построчная задача «предсказать x всех K кривых»")
    print(f"  хорошо поставлена, и потолок такого декодера — {100-100*T['ambig']/max(1,tot):.1f}%.")


if a.sum:
    summarise()
else:
    i, n = (int(v) for v in a.shard.split("/"))
    collect(i, n)

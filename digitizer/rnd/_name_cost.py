r"""_name_cost.py — ЦЕНА ИМЕНИ: сколько честных кривых теряется на ПОДПИСИ, а не на геометрии.

Метрика ветки засчитывает кривую, только если трасса совпала с эталоном ТОЙ ЖЕ мнемоники. Значит
лист, где все линии прослежены верно, но две подписи переставлены местами, считается браком дважды —
хотя геометрически файл безупречен. Сколько это стоит, не считалось НИ РАЗУ (DIRECTIONS.md §3.1).

Считаем на одних и тех же выданных трассах два числа:
  1. С ИМЕНЕМ (как сейчас) — написанное под именем N сверяется с эталоном имени N;
  2. БЕЗ ИМЕНИ — внутри ОДНОГО ТРЕКА ищем максимальное паросочетание 1:1 между написанными трассами
     и эталонными кривыми: пара засчитывается, если med≤3px и cov≥0.9, кто бы как ни назывался.
Разность и есть цена имени. Она — верхняя граница выигрыша от любой работы с ПОРЯДКОМ/подписью
и нижняя граница того, что уже умеет ведение.

⚠ Трек — единица сравнения не случайно: переставлять подписи между треками нельзя, это разные
дорожки бланка. Ограничение делает оценку КОНСЕРВАТИВНОЙ.
⚠ `written` в пуловых дампах собран с `slot_model=""` (§4: пулы намеренно жадные), поэтому число
относится к раскладке ПРАВИЛОМ, а не к проду. Для прода нужен отдельный прогон.

  <ComfyUI>\python_embeded\python.exe _name_cost.py --shard 0/6
  <ComfyUI>\python_embeded\python.exe _name_cost.py --sum
"""
import sys, pickle, argparse, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--shard", default="0/6")
ap.add_argument("--out", default=r"F:/nds/output/taskS")
ap.add_argument("--sum", action="store_true")
# ★ §6.129: какие мнемоники оракул меняет местами — это указывает на НЕДОСТАЮЩИЙ признак
ap.add_argument("--confusion", action="store_true", help="копить пары (написано → правда)")
a = ap.parse_args()
OUT = Path(a.out)


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def match(rows, cols, ok):
    """Максимальное паросочетание (аугментирующие пути). Размеры — единицы, хватает с запасом."""
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

    n = 0
    for r in rows:
        if try_(r, set()):
            n += 1
    return n


def collect(i, n):
    seen, files = set(), []
    for root in a.pools:
        for f in sorted(Path(root).glob("*.pkl")):
            if f.stem not in seen:
                seen.add(f.stem); files.append(f)
    mine = files[len(files) * i // n:len(files) * (i + 1) // n]
    print(f"★ ШАРД {i}/{n}: листов {len(mine)} из {len(files)}")
    rows, done, conf = [], 0, []
    for f in mine:
        d = pickle.load(open(f, "rb"))
        done += 1
        track = {s["name"]: s["track"] for s in d["slots"]}
        gts, wr = d["gts"], d["written"]
        # 1. с именем
        named = sum(1 for nm, gt in gts.items()
                    if nm in wr and HON(*err(wr[nm], gt)))
        # 2. без имени, внутри трека
        free = 0
        tracks = set(track.get(nm) for nm in gts) | set(track.get(nm) for nm in wr)
        for t in tracks:
            if t is None:
                continue
            G = [nm for nm in gts if track.get(nm) == t]
            W = [nm for nm in wr if track.get(nm) == t]
            if not G or not W:
                continue
            ok = {}
            for g in G:
                for w in W:
                    ok[(g, w)] = HON(*err(wr[w], gts[g]))
            free += match(G, W, ok)
            if a.confusion:
                # пары, которые ЧИНИТ перестановка: правда g достаётся написанному под ДРУГИМ именем
                for g in G:
                    for w in W:
                        if ok[(g, w)] and g != w and not HON(*err(wr.get(g, {}), gts[g])):
                            conf.append((w, g))
        rows.append(dict(sheet=d["name"], named=named, free=free,
                         curves=len(gts), written=len(wr)))
    p = OUT / f"name_cost_{i}of{n}.json"
    p.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    if a.confusion:
        (OUT / f"name_conf_{i}of{n}.json").write_text(
            json.dumps(conf, ensure_ascii=False), encoding="utf-8")
    print(f"★ СВЕРКА: обработано {done} + пропущено 0 = {done} против {len(mine)}   "
          f"{'★ СОШЛОСЬ' if done == len(mine) else '⛔ НЕ СОШЛОСЬ'}")
    print(f"листов {len(rows)} → {p}")


def summarise():
    fs = sorted(OUT.glob("name_cost_*of*.json"))
    if not fs:
        sys.exit("нет дампов")
    den = fs[0].stem.split("of")[1]
    fs = [f for f in fs if f.stem.endswith("of" + den)]
    rows = [r for f in fs for r in json.loads(f.read_text(encoding="utf-8"))]
    print(f"шардов {len(fs)} из {den}" + ("  ⚠⚠ НЕПОЛНЫЙ" if len(fs) < int(den) else ""))
    named = sum(r["named"] for r in rows)
    free = sum(r["free"] for r in rows)
    curves = sum(r["curves"] for r in rows)
    diff = [r["free"] - r["named"] for r in rows]
    up = sum(1 for v in diff if v > 0)
    print(f"листов {len(rows)}, экспертных кривых {curves}")
    print(f"\n  честных С ИМЕНЕМ (как считает ветка):  {named}  ({100*named/curves:.1f}%)")
    print(f"  честных БЕЗ ИМЕНИ (1:1 внутри трека):  {free}  ({100*free/curves:.1f}%)")
    print(f"\n★★ ЦЕНА ИМЕНИ: {free-named:+d} кривых "
          f"({100*(free-named)/max(1,named):+.1f}% к нынешнему счёту), "
          f"листов с разницей {up}")
    print(f"⇒ столько даёт ИДЕАЛЬНАЯ подпись при НЕИЗМЕННОЙ геометрии — верхняя граница всей работы")
    print(f"  с порядком и раскладкой, и она же нижняя граница того, что ведение УЖЕ умеет.")
    print(f"⚠ Считано по `written` пуловых дампов (раскладка ПРАВИЛОМ, §4), не по проду.")


if a.sum:
    summarise()
else:
    i, n = (int(v) for v in a.shard.split("/"))
    collect(i, n)

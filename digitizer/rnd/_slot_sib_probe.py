r"""_slot_sib_probe.py — ВЫНИМАЕТСЯ ЛИ СИГНАЛ «ПОРЯДОК ЗОНДОВ В СЕМЕЙСТВЕ» ДОБАВКОЙ К СКОРУ.

§6.129: раскладка путает БРАТСКИЕ зонды одного семейства (GZ1…GZ5 — около трети всех путаниц), и ни
один из 14 признаков не кодирует, который из зондов перед нами. Конвенция «индекс растёт вместе с x»
держится примерно в половине групп (вшестеро чаще случайного на группах из 4).

Прежде чем строить признак, переобучать вес и гнать A/B — тот же дешёвый ход, которым §6.115.2
проверял `npts`: прибавить сигнал ПРЯМО К СКОРУ и посмотреть на пулах. Не вынимается добавкой —
переобучение тем более не оправдано.

Добавка: пара (слот, трасса) получает `bonus`, если ранг трассы по x среди кандидатов трека совпадает
с рангом слота по индексу зонда внутри его семейства. Направление конвенции определяется ПО ЛИСТУ
(берётся то, которое согласует больше пар) — глобального направления нет (§6.129: 53% против 20%).

  <ComfyUI>\python_embeded\python.exe _slot_sib_probe.py --bonus 0 0.25 0.5 1 2
"""
import sys, argparse, pickle, re, collections
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from auto import slot_model as SM

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--model", default="slot_model_g250.npz")
ap.add_argument("--bonus", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0])
ap.add_argument("--shard", default="0/6")
ap.add_argument("--out", default=r"F:/nds/output/taskS")
a = ap.parse_args()
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


class FakeLine:
    __slots__ = ("color", "behavior", "x_center")

    def __init__(self, color, behavior, x_center):
        self.color, self.behavior, self.x_center = color, behavior, x_center


def fam_idx(nm):
    t = nm.split()[0]
    m = re.match(r"^([A-Za-z_]+)(\d+)$", t)
    return (m.group(1), int(m.group(2))) if m else (t, None)


def sib_term(slots, pairs):
    """+1 паре, если ранг трассы по x совпадает с рангом слота по индексу зонда. Направление
    конвенции выбирается по листу — тем, которое согласует больше пар."""
    fam = {}
    for s in slots:
        f, ix = fam_idx(s["name"])
        if ix is not None:
            fam.setdefault((s["track"], f), []).append((ix, s["name"]))
    rank_slot = {}
    for key, items in fam.items():
        if len(items) < 2:
            continue
        for r, (_, nm) in enumerate(sorted(items)):
            rank_slot[nm] = (r, len(items), key)
    by_track = collections.defaultdict(list)
    for k, (nm, L, tr) in enumerate(pairs):
        by_track[getattr(L, "_track", None) or nm].append(k)
    out = np.zeros(len(pairs))
    xs = {k: pairs[k][1].x_center for k in range(len(pairs))}
    for key in {v[2] for v in rank_slot.values()}:
        ks = [k for k in range(len(pairs)) if rank_slot.get(pairs[k][0], (0, 0, None))[2] == key]
        if not ks:
            continue
        uniq = sorted({xs[k] for k in ks})
        # ⚠⚠ НАПРАВЛЕНИЕ — ОДНО НА ВСЕХ, И ТОЛЬКО ПРЯМОЕ. Первая редакция брала max(fwd, bwd) НА
        # КАЖДУЮ ПАРУ и тем поощряла кандидата у ЛЮБОГО края — добавка переставала различать
        # порядок. Выбрать направление по листу нельзя: без эталона его неоткуда узнать, а
        # конвенция «индекс растёт вместе с x» — большинство (53% против 20%, §6.129).
        for k in ks:
            r, n, _ = rank_slot[pairs[k][0]]
            xr = uniq.index(xs[k])
            frac_x = xr / max(1, len(uniq) - 1)
            frac_s = r / max(1, n - 1)
            out[k] = 1.0 - abs(frac_x - frac_s)
    return out


def main():
    i, n = (int(v) for v in a.shard.split("/"))
    w = SM.load(a.model)
    seen, files = set(), []
    for root in a.pools:
        for f in sorted(Path(root).glob("*.pkl")):
            if f.stem not in seen:
                seen.add(f.stem); files.append(f)
    mine = files[len(files) * i // n:len(files) * (i + 1) // n]
    print(f"★ ШАРД {i}/{n}: листов {len(mine)}, добавки {a.bonus}")
    T = collections.Counter()
    for f in mine:
        d = pickle.load(open(f, "rb"))
        gts = d["gts"]
        by_track = {}
        for ln in d["lines"]:
            by_track.setdefault(ln["track"], []).append(
                (FakeLine(ln["color"], ln["behavior"], ln["x_center"]), ln["tr"]))
        slots = [dict(name=s["name"], track=s["track"], color=s["color"], cls=s["class"])
                 for s in d["slots"]]
        if not slots:
            continue
        X, pairs = SM.rows(slots, by_track)
        if not len(X):
            continue
        base = SM.predict(w, X)
        sib = sib_term(slots, pairs)
        for b in a.bonus:
            sc = base + b * sib
            got, used = {}, set()
            for k in sorted(range(len(sc)), key=lambda q: -sc[q]):
                nm, L, tr = pairs[k]
                if nm in got or id(L) in used:
                    continue
                got[nm] = tr; used.add(id(L))
            T[b] += sum(1 for nm, tr in got.items()
                        if nm in gts and HON(*err(tr, gts[nm])))
    p = Path(a.out) / f"sib_probe_{i}of{n}.pkl"
    pickle.dump(dict(T), open(p, "wb"))
    for b in a.bonus:
        print(f"  bonus {b:<5} честных {T[b]}")
    print(f"→ {p}")


main()

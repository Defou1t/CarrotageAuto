r"""_slot_scale.py — БАЗОВЫЕ ШКАЛЫ СЛОТОВ И ТИПИЧНЫЕ ЗНАЧЕНИЯ СЕМЕЙСТВ ДЛЯ ВЫБОРА ТРАССЫ (18.10, §6.311).

Одна и та же линия в разных слотах трека даёт разные значения: шкалы слотов (ГЗ «0…20 Ом·м», ПС «−50…+50 мВ») различны, а
значения метода обычно лежат в своём диапазоне. Для признака правдоподобия в `_line_choice_assign.py --vfeat`:
- по каждому листу (все nlgx архива): базовая шкала каждого слота (v_left, v_right первой оси цепочки);
- по каждому семейству (корень имени без BKZ_): медиана и межквартильный размах значений эталона (по уровням эталона) — по
  листам НЕ сорта A (листы сорта A в приор не идут).
Пишет `--out` = {"scale": {лист: {слот: (vl, vr)}}, "prior": {семейство: (медиана, IQR, q05, q95)}}.

  _slot_scale.py --workers 4
"""
import sys, argparse, pickle, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
from collections import defaultdict
import numpy as np


def one(q):
    from extract_nlgx import extract, NULL
    import decode_levels as DL
    from auto import meta as M
    try:
        mt = extract(str(q))
    except Exception:
        return q.name, {}, {}
    sc, vals = {}, defaultdict(list)
    for c in mt["curves"]:
        root = M.mnem_root(c["name"])
        if root == "DA":
            continue
        fam = DL.build_family(mt, c)
        if not fam:
            continue
        sc[c["name"]] = (float(fam[0]["v_left"]), float(fam[0]["v_right"]))
        lv = DL.gt_levels(c)
        xs = [(c["top_y"] + i, x) for i, x in enumerate(c["xs"]) if x != NULL][::25]
        if len(xs) >= 10:
            maps = [DL.scale_map(s) for s in fam]
            vals[re.sub(r"^BKZ_", "", root)] += [maps[min(lv.get(y, 0), len(fam) - 1)](x) for y, x in xs]
    return q.name, sc, dict(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=r"F:/nds/output/taskS/slot_scale.pkl")
    a = ap.parse_args()
    import multiprocessing as mp
    TS = Path(r"F:/nds/output/taskS")
    HOLD = {l.strip() for l in (TS / "holdoutA_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    qs = sorted(Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx"))
    SC, V = {}, defaultdict(list)
    WELL = {q.name: q.parent.parent.name for q in qs}
    reg = lambda w: (re.sub(r"[_\s\-]*\d.*$", "", w) or w).lower()
    VW = defaultdict(list)                                  # §6.311б: (семейство, скважина) → значения
    with mp.Pool(a.workers) as pool:
        for n, sc, vals in pool.imap_unordered(one, qs, chunksize=8):
            SC[n] = sc
            if n not in HOLD:
                for f, v in vals.items():
                    V[f] += v
                    VW[(f, WELL[n])] += v
    PR = {}
    for f, v in V.items():
        v = np.array(v, float); v = v[np.isfinite(v)]
        if len(v) >= 200:
            q05, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
            PR[f] = (float(q50), float(max(q75 - q25, 1e-6)), float(q05), float(q95))
    st = lambda v: (lambda q: (float(q[2]), float(max(q[3] - q[1], 1e-6)), float(q[0]), float(q[4])))(np.percentile(v, [5, 25, 50, 75, 95]))
    # приор района: все скважины района (для прода) и без своей скважины (для замера, по листу)
    byreg = defaultdict(lambda: defaultdict(list))
    for (f, w), v in VW.items():
        byreg[reg(w)][f].append((w, np.array(v, float)))
    REGT = {r: {f: st(np.concatenate([v for _, v in L])) for f, L in fs.items() if sum(len(v) for _, v in L) >= 200}
            for r, fs in byreg.items()}
    PREG = {}
    for n in SC:
        w = WELL[n]; r = reg(w); out = {}
        for f, L in byreg.get(r, {}).items():
            other = [v for w2, v in L if w2 != w]
            if len(other) >= 2 and sum(len(v) for v in other) >= 200:
                out[f] = st(np.concatenate(other))
        PREG[n] = out
    pickle.dump(dict(scale=SC, prior=PR, prior_reg=PREG, reg_table=REGT, reg_of={n: reg(WELL[n]) for n in SC}), open(a.out, "wb"))
    print(f"приор района (без своей скважины): листов с ним {sum(1 for v in PREG.values() if v)}, районов в таблице {len(REGT)}")
    print(f"листов {len(SC)}, семейств с приором {len(PR)}")
    for f in ("GZ", "OGZ", "SP", "PS", "GK", "NGK", "BK", "IK", "TM", "ZAT", "TAU"):
        if f in PR:
            print(f"  {f}: медиана {PR[f][0]:.2f}, IQR {PR[f][1]:.2f}, 5–95% {PR[f][2]:.2f}…{PR[f][3]:.2f}")


if __name__ == "__main__":
    main()

r"""_pair_value_order.py — §6.366 РАЗВЕДКА: держится ли порядок ЗНАЧЕНИЙ в парах кривых одного трека по эталону (14.11).

Для пар кривых эталона одного трека (одна ось DA) — на общих строках доля строк, где значение A > значения B (значения — по шкале уровня
масштаба эталона на строке: сегменты уровня + цепочка шкал). Пара упорядочена по имени (A < B по строке имени). По парам семейств —
доля пар, где порядок держится на ≥ 90% строк, и где обратный (≤ 10%). Физика: Т2 > Т1 (АК), МПЗ ≥ МГЗ в проницаемых, ZAT большей задержки
меньше счёт.

  _pair_value_order.py
"""
import sys, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
from collections import defaultdict
import numpy as np
import _sample_review as SR
from extract_nlgx import extract
from _multi_replica_probe import dense
from auto import meta as M
import decode_levels as DL


def vals(c, m):
    tr = dense(c); ch = DL.build_family(m, c)
    if not ch or len(tr) < 200:
        return None
    lv = SR.lv_rows(c)
    out = {}
    for y, x in tr.items():
        s = ch[min(int(lv.get(y, 0)), len(ch) - 1)]
        w = (s["x_right"] - s["x_left"]) or 1.0
        out[y] = s["v_left"] + (x - s["x_left"]) * (s["v_right"] - s["v_left"]) / w
    return out


def main():
    TR = SR.load_tracks(); S = SR.eval_sets(); sheets = set().union(*S.values())
    SRC = {q.name: q for q in SR.ARCH.glob("*/wlg/*.nlgx")}
    by = defaultdict(list)
    for nm in sorted(sheets):
        if nm not in SRC:
            continue
        try:
            m = extract(str(SRC[nm]))
        except Exception:
            continue
        tm = TR.get(nm, {})
        C = [(c["name"], c) for c in m["curves"] if M.mnem_root(c["name"]) != "DA"]
        V = {}
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                (na, ca), (nb, cb) = sorted([C[i], C[j]], key=lambda t: t[0])
                if tm.get(na) is None or tm.get(na) != tm.get(nb) or na.split()[1] != nb.split()[1]:
                    continue
                fa, fb = M.mnem_root(na), M.mnem_root(nb)
                key = (re.sub(r"\d+$", "", na.split()[0]), re.sub(r"\d+$", "", nb.split()[0]))
                for n_, c_ in ((na, ca), (nb, cb)):
                    if n_ not in V:
                        V[n_] = vals(c_, m)
                va, vb = V[na], V[nb]
                if not va or not vb:
                    continue
                com = [y for y in va if y in vb]
                if len(com) < 200:
                    continue
                f = float(np.mean([va[y] > vb[y] for y in com]))
                by[(fa, fb)].append((f, na.split()[0], nb.split()[0]))
    print("★ доля строк «значение A > значение B» (A — меньшее имя пары); пары семейств с ≥ 15 парами")
    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(v) < 15:
            continue
        f = np.array([x[0] for x in v])
        print(f"   {k[0]:>8s} > {k[1]:<8s} пар {len(f):4d}: ≥ 0.9 {np.mean(f >= 0.9):5.1%}, ≤ 0.1 {np.mean(f <= 0.1):5.1%}, медиана {np.median(f):.2f}")


if __name__ == "__main__":
    main()

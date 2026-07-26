r"""seq_why.py — ПОЧЕМУ селектор проиграл на двух листах валидации.

Обе деградации одинаковы при oracle и при nl+npts, значит дело не в отборе, а в самой трассе.
Печатаем ПОКРИВНО: какой трассе назначена кривая, её med и cov до и после, и что именно
перевело кривую через порог честности (med<=3 И cov>=0.9).
"""
import sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.argv = ["_pick_gate.py"]
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")

import _pick_gate as G
from pathlib import Path

ROOT = Path(r"F:\nds\output\taskS\pick_gate")
SHEETS = ["Semeguniv_20_BKZ_3100_3510_200_D1", "Stanulska_2_BKZ_115_620_200_D1"]
G.a.bridge = 20
G.a.dedup_tol = 50


def assign(picked, GM, raw):
    """Повтор назначения 1-к-1 из G.score, но с возвратом подробностей по каждой кривой."""
    pairs = []
    for nm, gm in GM.items():
        for ti, tr in enumerate(picked):
            com = [y for y in tr if y in gm]
            if len(com) < 30:
                continue
            dd = np.array([abs(tr[y] - gm[y]) for y in com])
            pairs.append((float(np.median(dd)), len(com) / len(gm), nm, ti))
    pairs.sort(key=lambda q: (q[1] < 0.9, q[0]))
    got, used = {}, set()
    for med, cov, nm, ti in pairs:
        if nm in got or ti in used:
            continue
        got[nm] = (med, cov, ti); used.add(ti)
    out = {}
    for nm in GM:
        if nm not in got:
            out[nm] = ("НЕ НАЗНАЧЕНА", None, None, None, False)
            continue
        med, cov, ti = got[nm]
        lk = G.leaked(picked[ti], raw[nm])
        out[nm] = ("ok", med, cov, ti, (not lk) and med <= 3 and cov >= 0.9)
    return out


for sh in SHEETS:
    print(f"\n{'='*100}\n{sh}")
    st = {}
    for tag, d in (("жадный", "holdout"), ("селектор", "seq_hold")):
        dd = pickle.load(open(ROOT / d / f"{sh}.pkl", "rb"))
        cand = [G.bridge(t, 20) for t in dd["traces"] if len(t) >= G.MINPTS]
        K = max(1, dd["K"] or len(dd["GM"]))
        st[tag] = (assign(list(cand), dd["GM"], dd["raw"]), cand, dd["GM"])
        print(f"  {tag}: трасс {len(cand)}, кривых {len(dd['GM'])}, K={K}")

    names = list(st["жадный"][2])
    print(f"\n  {'кривая':<26}{'med жадн':>10}{'cov жадн':>10}{'чест':>6}   "
          f"{'med сел':>10}{'cov сел':>10}{'чест':>6}   что изменилось")
    for nm in names:
        g = st["жадный"][0][nm]; s = st["селектор"][0][nm]
        def fmt(r):
            if r[0] != "ok":
                return f"{'—':>10}{'—':>10}{'нет':>6}"
            return f"{r[1]:>10.1f}{r[2]:>10.2f}{('ДА' if r[4] else 'нет'):>6}"
        note = ""
        if g[0] == "ok" and s[0] == "ok":
            if g[4] and not s[4]:
                if s[1] > 3 and g[1] <= 3:
                    note = f"★ТОЧНОСТЬ: med {g[1]:.1f}->{s[1]:.1f} перешла 3px"
                elif s[2] < 0.9 <= g[2]:
                    note = f"★ПОКРЫТИЕ: cov {g[2]:.2f}->{s[2]:.2f} упало ниже 0.9"
                else:
                    note = "★утечка/иное"
            elif s[4] and not g[4]:
                note = "выигрыш"
        elif g[0] == "ok" and s[0] != "ok":
            note = "★трасса пропала из назначения"
        print(f"  {nm[:25]:<26}{fmt(g)}   {fmt(s)}   {note}")

    # длины трасс: не укоротил ли селектор ведение
    for tag in ("жадный", "селектор"):
        cand = st[tag][1]
        L = sorted((len(t) for t in cand), reverse=True)
        print(f"  длины трасс {tag:<9}: {L[:8]}")

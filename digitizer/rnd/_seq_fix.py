r"""seq_why2.py — лечатся ли две деградации.

Гипотезы:
  Semeguniv_20/GZ41  — потеря покрытия 1.00->0.83 при УЛУЧШЕНИИ точности. Если дело в РАЗРЫВЕ,
                       мостик подлиннее его закроет и честность вернётся. Если в ОБРЫВЕ (трасса
                       кончается раньше) — мостик не поможет.
  Stanulska_2/GZ31   — med 1.5->3.2 при пороге 3. Если ошибка сосредоточена в узком куске строк,
                       это локальная экскурсия; если размазана — систематический сдвиг.
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
G.a.dedup_tol = 50


def best_for(cand, gm):
    """Трасса из cand, лучше всех накрывающая gm (по медиане |dx| при >=30 общих строк)."""
    best = None
    for tr in cand:
        com = [y for y in tr if y in gm]
        if len(com) < 30:
            continue
        med = float(np.median([abs(tr[y] - gm[y]) for y in com]))
        r = (med, len(com) / len(gm), tr)
        if best is None or (r[1] >= 0.9, -r[0]) > (best[1] >= 0.9, -best[0]):
            best = r
    return best


def load(d, sh, bridge):
    dd = pickle.load(open(ROOT / d / f"{sh}.pkl", "rb"))
    return [G.bridge(t, bridge) for t in dd["traces"] if len(t) >= G.MINPTS], dd["GM"]


print("=" * 100)
print("Semeguniv_20 / GZ41 — мостик против потери покрытия")
sh, cur = "Semeguniv_20_BKZ_3100_3510_200_D1", "GZ41 DA1 SA5"
for bridge in (20, 50, 100, 200, 500):
    line = f"  мостик {bridge:>4}: "
    for tag, d in (("жадный", "holdout"), ("селектор", "seq_hold")):
        cand, GM = load(d, sh, bridge)
        med, cov, tr = best_for(cand, GM[cur])
        honest = med <= 3 and cov >= 0.9
        line += f"{tag} med {med:>5.1f} cov {cov:>4.2f} {'ЧЕСТНАЯ' if honest else 'нет    '}   "
    print(line)

cand, GM = load("seq_hold", sh, 20)
gm = GM[cur]
med, cov, tr = best_for(cand, gm)
ys_gm = np.array(sorted(gm)); ys_tr = np.array(sorted(tr))
miss = np.setdiff1d(ys_gm, ys_tr)
print(f"\n  строк у эксперта {len(ys_gm)}, у трассы селектора на них {len(ys_gm)-len(miss)}, "
      f"НЕ покрыто {len(miss)}")
if len(miss):
    b = np.split(miss, np.where(np.diff(miss) > 1)[0] + 1)
    print(f"  непокрытые куски (всего {len(b)}): "
          + ", ".join(f"{s[0]}..{s[-1]} ({len(s)} строк)" for s in b[:8]))
    print(f"  самый длинный кусок: {max(len(s) for s in b)} строк")
    print(f"  диапазон трассы {ys_tr.min()}..{ys_tr.max()}, эксперта {ys_gm.min()}..{ys_gm.max()}")

print("\n" + "=" * 100)
print("Stanulska_2 / GZ31 — где сидит ошибка")
sh, cur = "Stanulska_2_BKZ_115_620_200_D1", "GZ31 DA1 SA4"
for tag, d in (("жадный", "holdout"), ("селектор", "seq_hold")):
    cand, GM = load(d, sh, 20)
    gm = GM[cur]
    med, cov, tr = best_for(cand, gm)
    com = np.array(sorted(set(tr) & set(gm)))
    err = np.array([abs(tr[y] - gm[y]) for y in com])
    q = np.percentile(err, [50, 75, 90, 95, 99])
    print(f"  {tag:<9} med {med:>5.2f} cov {cov:.2f}  общих строк {len(com)}  "
          f"p50/75/90/95/99 = {q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f}/{q[3]:.1f}/{q[4]:.1f}  max {err.max():.0f}")
    bad = err > 3
    print(f"            строк с ошибкой >3px: {bad.sum()} из {len(com)} ({100*bad.mean():.0f}%)")
    if bad.any():
        yb = com[bad]
        blocks = np.split(yb, np.where(np.diff(yb) > 5)[0] + 1)
        blocks = sorted(blocks, key=len, reverse=True)
        print(f"            кусков: {len(blocks)}, крупнейшие: "
              + ", ".join(f"{b[0]}..{b[-1]} ({len(b)})" for b in blocks[:4]))

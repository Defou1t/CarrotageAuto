r"""_piece_union.py — ПОТОЛОК СБОРКИ КРИВОЙ ИЗ КУСКОВ ТРАСС (§6.236, 27.09).

У 1662 кривых поля+сорта A (§6.228) нет ни одной честной трассы. Ведут ли их КУСКАМИ разные линии (трасса идёт по кривой,
уходит на соседа, а кривую подхватывает другая трасса)? Для каждой кривой без честного кандидата: доля строк эталона, где
ХОТЯ БЫ ОДНА трасса (любого пути) в 3 px; число кусков (непрерывных отрезков ≥ 50 строк, где держит одна трасса), нужных для
покрытия жадно; и то же, если брать только трассы своего трека. ≥ 0.9 объединением — потолок сборки из кусков; «штопка» как
задача выбора по узлам — только если кусков мало.

  _piece_union.py --cache F:/nds/output/taskS/tcache --every 3
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--every", type=int, default=3)
a = ap.parse_args()
SRC = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx"):
    SRC.setdefault(q.stem, q)


def un(t):
    return dict(zip(t[0].tolist(), t[1].tolist()))


def hon(tr, gt, bridge=30):
    com = [y for y in gt if y in tr]
    if len(com) < 30 or np.median([abs(tr[y] - gt[y]) for y in com]) > 3.0:
        return False
    rs = np.array(sorted(tr)); ok = 0
    for y in gt:
        if y in tr:
            ok += 1; continue
        i = np.searchsorted(rs, y)
        if 0 < i < len(rs) and rs[i] - rs[i - 1] <= bridge:
            ok += 1
    return ok / len(gt) >= 0.9


def pieces(gy, H):
    """H: (n_traces, n_rows) bool — трасса держит строку. Жадно: кусок = самый длинный непрерывный отрезок одной трассы
    от первой непокрытой строки; с допуском разрыва 30 строк внутри куска. Возвращает (число кусков, доля покрытия)."""
    n = H.shape[1]; i = 0; k = 0; cov = 0
    while i < n:
        if not H[:, i].any():
            i += 1; continue
        best = i
        for t in np.flatnonzero(H[:, i]):
            j = i; gap = 0
            while j + 1 < n and (H[t, j + 1] or gap < 30):
                gap = 0 if H[t, j + 1] else gap + 1
                j += 1
            j -= gap
            best = max(best, j)
        cov += int(H[:, i:best + 1].any(0).sum()); k += 1; i = best + 1
    return k, cov / max(1, n)


R = []
files = sorted(Path(a.cache).glob("*.pkl"))[::a.every]
for f in files:
    v = pickle.load(open(f, "rb"))
    q = SRC.get(Path(v["frame_nlgx"]).stem)
    if not q:
        continue
    G = {c["name"]: dense(c) for c in extract(str(q))["curves"]
         if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    T = [un(t) for _, t in v["traces"]] + ([un(t) for _, t in v["alt"]] if v.get("alt") else [])
    for g, gt in G.items():
        if any(hon(t, gt) for t in T):
            continue
        gy = np.array(sorted(gt)); gx = np.array([gt[y] for y in gy])
        H = np.array([[y in t and abs(t[y] - x) <= 3 for y, x in zip(gy, gx)] for t in T]) if T else np.zeros((0, len(gy)), bool)
        if not len(H):
            R.append((0, 0.0, 0.0, M.mnem_root(g))); continue
        union = float(H.any(0).mean()); best1 = float(H.mean(1).max())
        k, c = pieces(gy, H)
        R.append((k, union, best1, M.mnem_root(g)))
u = np.array([r[1] for r in R]); b = np.array([r[2] for r in R]); k = np.array([r[0] for r in R])
print(f"★ КРИВЫХ БЕЗ ЧЕСТНОГО КАНДИДАТА: {len(R)} (каждый {a.every}-й лист кэша)")
print(f"   лучшая ОДНА трасса держит в 3 px: медиана {np.median(b):.2f} строк эталона")
print(f"   ОБЪЕДИНЕНИЕ всех трасс держит: медиана {np.median(u):.2f}; ≥ 0.9 у {100*np.mean(u >= 0.9):.0f}% ({int(np.sum(u >= 0.9))}), "
      f"≥ 0.7 у {100*np.mean(u >= 0.7):.0f}%, < 0.3 у {100*np.mean(u < 0.3):.0f}%")
m = u >= 0.9
if m.any():
    kc = Counter(int(x) if x < 6 else 6 for x in k[m])
    print("   из покрытых объединением ≥ 0.9 — кусков нужно: " + ", ".join(f"{x if x < 6 else '6+'}: {v}" for x, v in sorted(kc.items())))

r"""_strat_width_grid.py — оффлайн-грид правила выбора на дампе моментов ухода.
Дёшево (без трассировки) ищем форму cost, потом лучшие несколько гоняем на стенде.

Замер показал: в МОМЕНТ ухода свой ран ШИРЕ приора (med 10 при приоре 5), а ложный —
ТОНКИЙ (med 2). Значит штраф за ширину должен быть АСИММЕТРИЧНЫМ: узкий = подозрительный,
широкий = скорее пересечение/спайк своей кривой. Но награда за ширину нуждается в ПОТОЛКЕ:
ран в 200px — это рамка/горизонтальная линовка, а не штрих.
"""
import sys, pickle, itertools
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np

with open(r"F:\nds\output\taskS\width_depart.pkl", "rb") as f:
    DUMP = pickle.load(f)
print(f"моментов ухода: {len(DUMP)}")

G = [d[0].astype(np.float64) for d in DUMP]
W = [d[1].astype(np.float64) for d in DUMP]
P = [float(d[2]) for d in DUMP]
OI = [d[3] for d in DUMP]


def acc(rule):
    ok = 0
    for g, w, p, oi in zip(G, W, P, OI):
        if int(np.argmin(rule(g, w, p))) == oi:
            ok += 1
    return 100.0 * ok / len(DUMP)


print("\n=== cost = gap + bn*max(0, p-w) - rw*min(w, cap) ===")
print(f"{'bn':>5}{'rw':>6}{'cap':>6}{'  попаданий':>12}")
best = []
for bn, rw, cap in itertools.product((0.0, 1.0, 3.0, 6.0), (0.0, 1.0, 2.0, 3.0, 5.0),
                                     (10, 20, 40, 1e9)):
    a = acc(lambda g, w, p, bn=bn, rw=rw, cap=cap:
            g + bn * np.maximum(0.0, p - w) - rw * np.minimum(w, cap))
    best.append((a, bn, rw, cap))
for a, bn, rw, cap in sorted(best, reverse=True)[:12]:
    print(f"{bn:>5.1f}{rw:>6.1f}{cap:>6.0f}{a:>11.1f}%")

print("\n=== относительный вариант: gap + bn*max(0,1-w/p)*S - rw*min(w,cap) ===")
b2 = []
for bn, rw, cap, S in itertools.product((5.0, 10.0, 20.0), (0.0, 1.0, 2.0, 3.0),
                                        (10, 20, 40), (1.0,)):
    a = acc(lambda g, w, p, bn=bn, rw=rw, cap=cap:
            g + bn * np.maximum(0.0, 1.0 - w / max(p, 1.0)) * 10.0 - rw * np.minimum(w, cap))
    b2.append((a, bn, rw, cap))
for a, bn, rw, cap in sorted(b2, reverse=True)[:8]:
    print(f"{bn:>5.1f}{rw:>6.1f}{cap:>6.0f}{a:>11.1f}%")

print("\n=== жёсткий фильтр: отбросить w < f*p, затем ближайший по зазору ===")
for f_ in (0.4, 0.6, 0.8, 1.0, 1.2):
    def rule(g, w, p, f_=f_):
        okm = w >= f_ * p
        return np.where(okm, g, g + 1e6)
    print(f"   w >= {f_:.1f}*p : {acc(rule):5.1f}%")

print("\n=== контроль: сколько кандидатов и где вообще свой ===")
nc = np.array([len(g) for g in G])
print(f"   кандидатов med {np.median(nc):.0f}  p90 {np.percentile(nc,90):.0f}")
print(f"   доля, где свой ран — САМЫЙ ШИРОКИЙ: "
      f"{100.0*np.mean([int(np.argmax(w))==oi for w,oi in zip(W,OI)]):.1f}%")
print(f"   доля, где свой ран — САМЫЙ БЛИЗКИЙ : "
      f"{100.0*np.mean([int(np.argmin(g))==oi for g,oi in zip(G,OI)]):.1f}%")

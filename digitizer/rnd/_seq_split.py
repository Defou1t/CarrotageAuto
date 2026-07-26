r"""_seq_split.py — РАЗЛОЖЕНИЕ ВЫИГРЫША на «правки прода» и «обученный селектор», три набора.

§6.71 показал на третьем наборе, что заявленный в §6.66 выигрыш складывается из двух разных
вещей: правок трассировки 24-26.07 (§6.52/§6.58/§6.65) и собственно селектора — потому что жадные
кэши собирались ДО этих правок, а селекторные ПОСЛЕ. Здесь то же разложение делается для ВСЕХ
трёх наборов, по три кэша на каждый:

    старый жадный  (24.07, прежний код)   -> новый жадный (сегодня)  = ВКЛАД ПРАВОК ПРОДА
    новый  жадный  (сегодня)              -> новый seq    (сегодня)  = ВКЛАД СЕЛЕКТОРА

⚠ Второе сравнение — единственное, что можно называть «выигрышем селектора»: обе стороны собраны
одним кодом, режим запиннен в стенде (`cfg.cv.seq_model = ""`), отпечатки сверяются.

  python _seq_split.py
"""
import sys, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.argv = ["_pick_gate.py"]
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
import _pick_gate as G
from _decoder_data import train_sheets
from pathlib import Path

R = Path(r"F:\nds\output\taskS\pick_gate")
G.a.bridge = 20
G.a.dedup_tol = 50
TRAIN = {Path(str(s)).stem for s in train_sheets(25)}

SETS = [("гейт",      "cache",   "gate_g2",  "gate_s2"),
        ("валидация", "holdout", "hold_g2",  "hold_s2"),
        ("третий",    "wide",    "wide60",   "seq_wide60")]
_C = {}


def honest(d, name, pick):
    key = (d, name)
    if key not in _C:
        dd = pickle.load(open(R / d / f"{name}.pkl", "rb"))
        _C[key] = ([G.bridge(t, G.a.bridge) for t in dd["traces"] if len(t) >= G.MINPTS],
                   max(1, dd["K"] or len(dd["GM"])), dd["GM"], dd["raw"])
    cand, K, GM, raw = _C[key]
    return G.score(G.PICKERS[pick](cand, K, G.a), GM, raw)


def pair(da, db, names, pick):
    sa = sb = up = dn = 0
    worst = []
    for n in names:
        x = honest(da, n, pick)[0]; y = honest(db, n, pick)[0]
        sa += x; sb += y; up += y > x; dn += y < x
        if y < x:
            worst.append((n, x, y))
    return sa, sb, up, dn, worst


for pick in ("nl+npts", "oracle"):
    tag = "ПРОД §6.49" if pick == "nl+npts" else "ПОТОЛОК (отбора нет)"
    print(f"\n{'#'*100}\n### {pick} — {tag}\n")
    print(f"{'набор':<12}{'листов':>7}{'чист':>6} | {'правки прода':>22} | {'СЕЛЕКТОР':>22}")
    T = dict(n=0, clean=0, a1=0, b1=0, u1=0, d1=0, a2=0, b2=0, u2=0, d2=0)
    degraded = []
    for lbl, dold, dnew, dseq in SETS:
        missing = [d for d in (dold, dnew, dseq) if not (R / d).is_dir()]
        if missing:
            print(f"{lbl:<12}  — кэши не готовы: {', '.join(missing)}")
            continue
        names = sorted(set.intersection(*[{f.stem for f in (R / d).glob("*.pkl")}
                                          for d in (dold, dnew, dseq)]))
        clean = sum(1 for n in names if n not in TRAIN)
        a1, b1, u1, d1, _ = pair(dold, dnew, names, pick)
        a2, b2, u2, d2, w2 = pair(dnew, dseq, names, pick)
        degraded += [(lbl, *w) for w in w2]
        print(f"{lbl:<12}{len(names):>7}{clean:>6} | {a1:>5} -> {b1:<5} ({b1-a1:+3d}) "
              f"вверх {u1:<2} вниз {d1:<2} | {a2:>5} -> {b2:<5} ({b2-a2:+3d}) "
              f"вверх {u2:<2} вниз {d2:<2}")
        for k, v in zip(("n", "clean", "a1", "b1", "u1", "d1", "a2", "b2", "u2", "d2"),
                        (len(names), clean, a1, b1, u1, d1, a2, b2, u2, d2)):
            T[k] += v
    if T["n"]:
        pc = lambda a, b: f"{100*(b-a)/a:+.0f}%" if a else "—"
        print(f"{'-'*100}\n{'ИТОГО':<12}{T['n']:>7}{T['clean']:>6} | "
              f"{T['a1']:>5} -> {T['b1']:<5} ({T['b1']-T['a1']:+3d}) вверх {T['u1']:<2} вниз {T['d1']:<2}"
              f" | {T['a2']:>5} -> {T['b2']:<5} ({T['b2']-T['a2']:+3d}) вверх {T['u2']:<2} вниз {T['d2']:<2}")
        print(f"{'':<25} | правки прода {pc(T['a1'], T['b1']):>8}"
              f"        | СЕЛЕКТОР {pc(T['a2'], T['b2']):>8}")
        if degraded:
            print(f"\n  деградации от селектора ({len(degraded)}):")
            for lbl, n, x, y in degraded:
                print(f"    [{lbl:<9}] {x} -> {y}   {n[:62]}")
    print()

# ── сверка отпечатков ───────────────────────────────────────────────────────────────────────
print(f"\n{'#'*100}\n### ОТПЕЧАТКИ СБОРКИ (§6.71)")
for lbl, dold, dnew, dseq in SETS:
    if all((R / d).is_dir() for d in (dnew, dseq)):
        print(f"\n{lbl}:")
        G.check_stamps([R / dold, R / dnew, R / dseq])

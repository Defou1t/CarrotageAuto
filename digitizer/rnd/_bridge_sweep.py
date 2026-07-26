r"""_bridge_sweep.py — устойчив ли вывод о селекторе к величине МОСТИКА.

⚠ ЧТО ТАКОЕ МОСТИК И ГДЕ ОН ЖИВЁТ. `_pick_gate.bridge` линейно заполняет разрывы трассы длиной
<= N строк ПЕРЕД сравнением с эталоном. В ПРОДЕ ЕГО НЕТ: `auto/emit.py:201` пишет пропуски как
NULL в xs, то есть выданный файл содержит дырки, а метрика их закрывает. Обоснование §6.53
разумное (эксперт хранит кривую полилинией с шагом вершин 6-23 строки, и разрыв короче шага в его
представлении не существует), но это ПОМОЩЬ МЕТРИКЕ, а не свойство продукта.

⇒ Поэтому вывод обязан быть устойчив к N. Особенно важен N=0 — метрика без всякой помощи, самая
консервативная оценка. Если при N=0 преимущество селектора исчезает, значит он выигрывает не
качеством ведения, а тем, что оставляет разрывы удобной длины.

Все пары — сегодняшний код, отпечатки сверены (§6.71):
    гейт  gate_g2/gate_s2 · валидация hold_g2/hold_s2 · третий wide60/seq_wide60

  python _bridge_sweep.py
"""
import sys, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.argv = ["_pick_gate.py"]
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
import _pick_gate as G
from pathlib import Path

R = Path(r"F:\nds\output\taskS\pick_gate")
G.a.dedup_tol = 50
SETS = [("гейт", "gate_g2", "gate_s2"), ("валидация", "hold_g2", "hold_s2"),
        ("третий", "wide60", "seq_wide60")]
BRIDGES = [0, 5, 10, 20, 50, 100, 200]

raw = {}
for _, dg, ds in SETS:
    for d in (dg, ds):
        raw.setdefault(d, {f.stem: pickle.load(open(f, "rb")) for f in sorted((R / d).glob("*.pkl"))})


def total(d, names, bridge, pick):
    G.a.bridge = bridge
    s = 0
    for n in names:
        dd = raw[d][n]
        cand = [G.bridge(t, bridge) for t in dd["traces"] if len(t) >= G.MINPTS]
        s += G.score(G.PICKERS[pick](cand, max(1, dd["K"] or len(dd["GM"])), G.a),
                     dd["GM"], dd["raw"])[0]
    return s


for pick in ("nl+npts", "oracle"):
    print(f"\n{'#'*92}\n### {pick} — {'ПУТЬ 2 (§6.75: НЕ отгрузка)' if pick == 'nl+npts' else 'потолок'}\n")
    print(f"{'мостик':>7} | " + "".join(f"{lb:>22}" for lb, _, _ in SETS) + f"{'ИТОГО':>22}")
    for b in BRIDGES:
        line = f"{b:>7} | "
        ta = tb = 0
        for lb, dg, ds in SETS:
            names = sorted(set(raw[dg]) & set(raw[ds]))
            x = total(dg, names, b, pick); y = total(ds, names, b, pick)
            ta += x; tb += y
            line += f"{x:>7} ->{y:>4} ({y-x:+3d})  "
        pc = f"{100*(tb-ta)/ta:+.0f}%" if ta else "—"
        line += f"{ta:>6} ->{tb:>4} ({tb-ta:+3d}, {pc})"
        print(line + ("   ← §6.72" if b == 20 else ("   ← БЕЗ ПОМОЩИ МЕТРИКЕ" if b == 0 else "")))

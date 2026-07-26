r"""_seq_wide60.py — ЧИСТЫЙ замер селектора: обе стороны собраны ОДНИМ И ТЕМ ЖЕ прод-кодом.

⚠⚠ ЗАЧЕМ ПОНАДОБИЛСЯ. Замер §6.66 сравнивал кэши, собранные В РАЗНОЕ ВРЕМЯ РАЗНЫМ КОДОМ:
жадные (`cache`/`holdout`/`wide`) — 24.07, селекторные (`seq_*`) — 25-26.07. Между этими датами
в прод легли §6.52 (порог сетки, «0 → 19 честных» на тёмных листах), §6.58 (подхват), §6.63
(`trace_flagged` по умолчанию) и §6.65 (цвет ловил бумагу). То есть часть «выигрыша селектора» —
это правки прода. Прямой признак: на 7 листах у жадного кэша НОЛЬ трасс, а у селекторного 1-5;
только на них набегает +10 честных кривых из +34.

ЗДЕСЬ обе стороны пересобраны сегодняшним кодом по одним и тем же 60 листам:
    wide60      — жадный выбор,  сегодняшний прод
    seq_wide60  — селектор,      сегодняшний прод
Ни один из 60 листов не входит в обучающие 25 (§6.70), поэтому это ещё и полностью held-out.

ТРИ СРАВНЕНИЯ, и третье — самое важное для доверия к прежним цифрам:
  1. seq_wide60 против seq_wide  — сошлась ли пересборка селектора (20 общих листов);
  2. ★ wide60 против seq_wide60  — ЧИСТЫЙ выигрыш селектора;
  3. ⚠ wide против wide60        — сколько давал ОДИН ТОЛЬКО дрейф прод-кода, без селектора.

  python _seq_wide60.py
"""
import sys, pickle
import numpy as np
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
_C = {}


def honest(d, name, pick):
    """(честных, кривых, med, трасс) — с кэшированием разбора pkl."""
    key = (d, name)
    if key not in _C:
        dd = pickle.load(open(R / d / f"{name}.pkl", "rb"))
        _C[key] = ([G.bridge(t, G.a.bridge) for t in dd["traces"] if len(t) >= G.MINPTS],
                   max(1, dd["K"] or len(dd["GM"])), dd["GM"], dd["raw"], len(dd["traces"]))
    cand, K, GM, raw, ntr = _C[key]
    h, ncur, med = G.score(G.PICKERS[pick](cand, K, G.a), GM, raw)
    return h, ncur, med, ntr


def compare(da, db, pick, title, note=""):
    names = sorted({f.stem for f in (R / da).glob("*.pkl")} & {f.stem for f in (R / db).glob("*.pkl")})
    if not names:
        print(f"\n{title}: НЕТ ОБЩИХ ЛИСТОВ ({da} / {db})")
        return
    rows = []
    for n in names:
        a = honest(da, n, pick); b = honest(db, n, pick)
        rows.append((n, a[0], b[0], a[1], a[2], b[2], a[3], b[3]))
    sa = sum(r[1] for r in rows); sb = sum(r[2] for r in rows)
    up = [r for r in rows if r[2] > r[1]]; dn = [r for r in rows if r[2] < r[1]]
    pct = f"{100*(sb-sa)/sa:+.0f}%" if sa else "—"
    print(f"\n{'='*98}\n{title}   {note}")
    print(f"  листов {len(rows)}, кривых {sum(r[3] for r in rows)}   "
          f"{da} {sa} -> {db} {sb}   ({sb-sa:+d}, {pct})   вверх {len(up)}, ВНИЗ {len(dn)}")
    fm = lambda v: "—" if v != v else f"{v:.1f}"
    for n, a, b, nc, ma, mb, ta, tb in sorted(rows, key=lambda r: r[2] - r[1]):
        if a != b:
            print(f"   {'ВНИЗ ' if b < a else 'вверх'} {b-a:+d}  {a}->{b} из {nc}"
                  f"  med {fm(ma)}->{fm(mb)}  трасс {ta}->{tb}   {n[:56]}")
    return sa, sb


names60 = sorted({f.stem for f in (R / "wide60").glob("*.pkl")}
                 & {f.stem for f in (R / "seq_wide60").glob("*.pkl")})
print(f"листов пересобрано обеими сторонами: {len(names60)}; "
      f"из них в обучающей выборке: {len(set(names60) & TRAIN)}")

for pick in ("nl+npts", "oracle"):
    tag = "ПРОД §6.49" if pick == "nl+npts" else "ПОТОЛОК (отбора нет)"
    print(f"\n\n{'#'*98}\n### КРИТЕРИЙ {pick} — {tag}")
    compare("seq_wide", "seq_wide60", pick, "1. СВЕРКА ПЕРЕСБОРКИ СЕЛЕКТОРА",
            "(расхождение = прод изменился с 26.07 01:13)")
    compare("wide", "wide60", pick, "2. ⚠ ДРЕЙФ ПРОД-КОДА БЕЗ СЕЛЕКТОРА",
            "(жадный 24.07 против жадного сегодня)")
    compare("wide60", "seq_wide60", pick, "3. ★ ЧИСТЫЙ ВЫИГРЫШ СЕЛЕКТОРА",
            "(обе стороны — сегодняшний код, 0 листов из обучения)")

r"""_relatch_verify.py — НЕЗАВИСИМАЯ ПРОВЕРКА выигрыша стратегии «ширина» (_strat_width.py).

ЗАЧЕМ ОТДЕЛЬНЫЙ ФАЙЛ. Агент-скептик для этой стратегии не отработал (лимит сессии), а сама
стратегия смешивает в одном трассировщике ЧЕТЫРЕ независимые правки, из которых только одна
относится к «перезахватыванию»:
  1. в ветке else: ЗАЗОР до рана (gap) вместо расстояния до ЦЕНТРА рана;
  2. в ветке else: предпочтение ШИРОКОГО рана (rw=100, cap=20 — это вырожденный предел
     «бери любой ран шириной >=20px, зазор лишь тайбрейк»; сам приор ширины замером признан
     ИНЕРТНЫМ — bn=0 даёт те же числа);
  3. wide_mode='clipall': на ЛЮБОМ широком ране (>=14px) точка берётся как clip(pred,a,b),
     а не «дальний край от базлайна». ⚠ Это действует и в ветке cont, т.е. на 76% строк —
     к перезахватыванию отношения не имеет вовсе;
  4. коаст по пустым строкам с записью предсказанных значений (тот самый механизм, который
     скептик по _strat_coast.py разоблачил нуль-контролем: cov растёт по построению).

Смешанные правки нельзя вести в прод одним куском: непонятно, что именно работает, а §6.14
уже показал, как «лишнее ветвление без выигрыша» попадает в код. Здесь каждая правка
включается ОТДЕЛЬНЫМ флагом, и меряется её собственный вклад.

★ ГЕЙТ: V0 (все флаги выключены) обязан дать базу бит-в-бит (68.3 / 0.99 / 3 из 24).

  python _relatch_verify.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
import numpy as np
import _relatch_bench as BE

SLMAX, WIDE = BE.SLMAX, BE.WIDE_RUN


def make(else_rule="center", wide_mode="edge", coast=False, wmin=20.0, decay=0.7):
    """Копия BE.trace с ОДИНОЧНО включаемыми правками.

    else_rule : 'center' — как в проде (ближайший ЦЕНТР рана к предсказанию);
                'gap'    — ближайший по ЗАЗОРУ (расстояние до отрезка рана, а не до центра);
                'wide'   — сначала ширина (min(w,wmin) максимальна), зазор — тайбрейк.
    wide_mode : 'edge'    — как в проде (дальний край от базлайна = вершина выноса);
                'clipall' — ближайшая к предсказанию точка внутри рана.
    coast     : писать ли предсказанные значения на строках БЕЗ чернил (метрический риск).
    """
    def tracer(rec, csr, H):
        base = rec["base"]
        x = None; v = 0.0; tr = {}
        for y in range(max(0, rec["y0"]), min(H, rec["y1"] + 1)):
            A, B, Cc = BE._runs_at(csr, y)
            if not len(A):
                if x is not None:
                    if coast:
                        v *= decay
                    x = x + float(np.clip(v, -SLMAX, SLMAX))
                    if coast:
                        tr[y] = float(x)
                continue
            if x is None:
                k = int(np.argmin(np.abs(Cc - base)))
                x = float(Cc[k]); v = 0.0; tr[y] = x; continue
            pred = x + float(np.clip(v, -SLMAX, SLMAX))
            cont = np.nonzero((A - 2 <= pred) & (pred <= B + 2))[0]
            if len(cont):
                k = int(cont[np.argmin(np.abs(Cc[cont] - pred))])
            else:
                gap = np.maximum(np.maximum(A - pred, pred - B), 0.0).astype(np.float64)
                if else_rule == "center":
                    k = int(np.argmin(np.abs(Cc - pred)))
                elif else_rule == "gap":
                    k = int(np.argmin(gap))
                else:                                   # 'wide'
                    w = (B - A).astype(np.float64)
                    k = int(np.argmin(-np.minimum(w, wmin) * 1e6 + gap))
            a, bb, c = int(A[k]), int(B[k]), float(Cc[k])
            if (bb - a) >= WIDE:
                nx = (float(min(max(pred, a), bb)) if wide_mode == "clipall"
                      else float(bb if abs(bb - base) >= abs(a - base) else a))
            else:
                nx = c
            v = 0.6 * v + 0.4 * (nx - x); x = float(nx); tr[y] = float(nx)
        BE._extend_ends(tr, csr, H, SLMAX)
        return tr
    return tracer


VARIANTS = [
    ("V0 БАЗА (гейт: обязан совпасть)", dict()),
    ("V1 только clipall (не перезахват!)", dict(wide_mode="clipall")),
    ("V2 только gap вместо центра",       dict(else_rule="gap")),
    ("V3 только ШИРИНА в else",           dict(else_rule="wide")),
    ("V4 ШИРИНА + clipall",               dict(else_rule="wide", wide_mode="clipall")),
    ("V5 ШИРИНА + clipall + коаст (=W5)", dict(else_rule="wide", wide_mode="clipall", coast=True)),
    ("V6 ШИРИНА wmin=14",                 dict(else_rule="wide", wmin=14.0)),
    ("V7 ШИРИНА wmin=40",                 dict(else_rule="wide", wmin=40.0)),
]

base_rows = None
res = []
for nm, kw in VARIANTS:
    rows = BE.run_strategy(tracer=make(**kw))
    r = BE.report(nm, rows)
    if base_rows is None:
        base_rows = rows
        ref = {(q["well"], q["curve"]): q for q in rows}
    else:
        # ★ ГЛАВНАЯ ЛОВУШКА ПРОЕКТА: не куплен ли выигрыш в med падением ПОКРЫТИЯ.
        drop = [(q["well"], q["curve"], ref[(q["well"], q["curve"])]["cov"], q["cov"])
                for q in rows
                if (q["well"], q["curve"]) in ref and q["cov"] < ref[(q["well"], q["curve"])]["cov"] - 0.02]
        worse = [(q["well"], q["curve"], ref[(q["well"], q["curve"])]["med"], q["med"])
                 for q in rows
                 if (q["well"], q["curve"]) in ref and q["med"] > ref[(q["well"], q["curve"])]["med"] * 1.15 + 1]
        print(f"   cov упал у {len(drop)} кривых; med ухудшился у {len(worse)}")
        for w in worse[:6]:
            print(f"      ХУЖЕ {w[0][:9]:<10}{w[1]:<8} {w[2]:7.1f} -> {w[3]:7.1f}")
    res.append(r)

print("\n" + "=" * 86)
print(f"{'вариант':<38}{'med(med)':>10}{'med(cov)':>10}{'ЧЕСТНЫХ':>10}{'своя%':>8}{'латч%':>8}")
for r in res:
    sv = float(np.mean([q["своя"] for q in r["rows"]]))
    lt = float(np.mean([q["латч"] for q in r["rows"]]))
    print(f"{r['name']:<38}{r['med_med']:>10.1f}{r['med_cov']:>10.2f}"
          f"{r['honest']:>7}/{r['n']}{sv:>8.1f}{lt:>8.1f}")
print("\nЧитать так: если V1 (clipall в одиночку) даёт большую часть выигрыша — это НЕ")
print("«умное перезахватывание», а замена правила выбора точки на широком ране, и вести её")
print("в прод надо отдельно и своим замером (ср. §2 prefer_body: тело, а не вершина).")

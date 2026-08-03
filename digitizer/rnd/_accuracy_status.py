r"""_accuracy_status.py — ГДЕ МЫ ПО ТОЧНОСТИ И НА ЧТО УПИРАЕМСЯ. Разложение по ПРИЧИНАМ, не по кодам.

⚠ ЗАЧЕМ (вопрос Эдуарда 30.07: «давно не проверяли точность, в правильном ли направлении идём»).
Ветка §6.79-§6.90 три дня улучшала РАСКЛАДКУ (какая трасса какому слоту). Но раскладка может
исправить только те кривые, для которых В ПУЛЕ УЖЕ ЕСТЬ годная трасса. Здесь считается, сколько их,
и чем именно ограничена выдача — отбором или самой трассировкой.

РАЗЛОЖЕНИЕ КАЖДОЙ ЭКСПЕРТНОЙ КРИВОЙ ПО ПРИЧИНЕ (взаимно исключающие):
  1. НЕТ СЛОТА          — кривой нет среди слотов рамки (раскладка про неё ничего не знает);
  2. НЕТ КАНДИДАТОВ     — на треке слота нет ни одной линии (нечего назначать);
  3. НЕДОСТУПНА         — кандидаты есть, но ни один не покрывает ≥90% строк (трассировка не дошла);
  4. ДАЛЕКО             — покрытие есть, но лучший кандидат ошибается >3px (трассировка не туда);
  5. ★ ЕСТЬ, НЕ ВЫБРАНА — годный кандидат В ПУЛЕ ЕСТЬ, но прод записал другой ⇒ вина ОТБОРА;
  6. ★★ ЗАПИСАНА ЧЕСТНО — прод справился.
Сумма 5+6 — это потолок любой работы с раскладкой. 2+3+4 — то, что может исправить только
трассировка/распознавание. Именно это соотношение и отвечает на вопрос о направлении.

★ Печатается и РАСПРЕДЕЛЕНИЕ ошибки лучшего кандидата: если лучшая доступная трасса ошибается на
4-6px, узкое место — порог и уточнение (refine), а если на 50px — идентичность линий.

  <ComfyUI>\python_embeded\python.exe _accuracy_status.py
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div"])
a = ap.parse_args()

HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


WELL = {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WELL[q.name] = wlg.parent.name

CAUSE = ["1 нет слота", "2 нет кандидатов", "3 недоступна (cov<0.9)", "4 далеко (>3px)",
         "5 ★ есть в пуле, не выбрана", "6 ★★ записана честно"]
cnt = {c: 0 for c in CAUSE}
best_med, best_cov, sel_med = [], [], []
per_well = {}
# ⚠ КЛЮЧ ДЕДУПЛИКАЦИИ — ИМЯ ФАЙЛА, А НЕ ПОЛЕ `name`: `_pool_oracle.py` кладёт дамп как
# `{nlgx.stem[:60]}.pkl`, поэтому дубль виден ДО чтения и стоит ноль. ⚠ В отличие от
# `_start_probe`/`_param_sweep` здесь нет отдельного прохода «сначала список, потом счёт»: дамп
# нужен целиком, и 2.35 ГБ читаются по делу — экономится только повторное чтение дублей.
# Совпадение ключа с полем `name` проверяется при чтении, чтобы рассинхронизация не осталась немой.
seen, nsheet, npool = set(), 0, 0
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        if f.stem in seen:
            continue
        seen.add(f.stem)
        d = pickle.load(open(f, "rb"))
        if not d["name"].startswith(f.stem[:40]):
            print(f"  ⚠ имя дампа {f.stem[:40]!r} расходится с полем name {d['name'][:40]!r}")
        nsheet += 1; npool += len(d["lines"])
        slots = {s["name"]: s for s in d["slots"]}
        tl = {}
        for i, ln in enumerate(d["lines"]):
            tl.setdefault(ln["track"], []).append(i)
        wl = WELL.get(d["name"], "?")
        pw = per_well.setdefault(wl, dict(curves=0, hon=0, avail=0))
        for nm, gt in d["gts"].items():
            pw["curves"] += 1
            s = slots.get(nm)
            if s is None:
                cnt[CAUSE[0]] += 1; continue
            cand = tl.get(s["track"], [])
            if not cand:
                cnt[CAUSE[1]] += 1; continue
            ms = [err(d["lines"][i]["tr"], gt) for i in cand]
            cov_ok = [(m, c) for m, c in ms if c >= 0.9 and m is not None]
            w = d["written"].get(nm)
            wr = err(w, gt) if w is not None else (None, 0.0)
            if HON(*wr):
                cnt[CAUSE[5]] += 1; pw["hon"] += 1; pw["avail"] += 1
                sel_med.append(wr[0])
                bm = min(m for m, _ in cov_ok); best_med.append(bm)
                best_cov.append(max(c for _, c in ms))
                continue
            if not cov_ok:
                cnt[CAUSE[2]] += 1
                best_cov.append(max([c for _, c in ms] or [0.0]))
                continue
            bm = min(m for m, _ in cov_ok)
            best_med.append(bm); best_cov.append(max(c for _, c in ms))
            if bm <= 3.0:
                cnt[CAUSE[4]] += 1; pw["avail"] += 1
            else:
                cnt[CAUSE[3]] += 1

tot = sum(cnt.values())
W = 78
print(f"{'='*W}\nВЫБОРКА: листов {nsheet}, трасс в пулах {npool}, экспертных кривых {tot}, "
      f"скважин {len(per_well)}\n{'='*W}")
print(f"{'причина':<32}{'кривых':>8}{'доля':>8}   что это значит")
MEAN = {CAUSE[0]: "раскладка не при чём", CAUSE[1]: "линий на треке нет — распознавание",
        CAUSE[2]: "трассировка не прошла лист — распознавание",
        CAUSE[3]: "трасса не на той кривой — идентичность/ведение",
        CAUSE[4]: "★ ЭТО И ЧИНИТ РАСКЛАДКА", CAUSE[5]: "уже хорошо"}
for c in CAUSE:
    print(f"{c:<32}{cnt[c]:>8}{100*cnt[c]/max(1,tot):>7.1f}%   {MEAN[c]}")
avail = cnt[CAUSE[4]] + cnt[CAUSE[5]]
print(f"\n{'-'*W}")
print(f"★ ТОЧНОСТЬ ОТГРУЗКИ СЕЙЧАС:            {cnt[CAUSE[5]]:>5} из {tot} = "
      f"{100*cnt[CAUSE[5]]/max(1,tot):.1f}% экспертных кривых")
print(f"★ ПОТОЛОК ЛЮБОЙ РАБОТЫ С РАСКЛАДКОЙ:   {avail:>5} из {tot} = "
      f"{100*avail/max(1,tot):.1f}%  (годная трасса уже в пуле)")
print(f"⇒ раскладка может добавить максимум {cnt[CAUSE[4]]} кривых "
      f"({100*cnt[CAUSE[4]]/max(1,tot):.1f}% выборки), всё остальное — "
      f"{tot-avail} кривых ({100*(tot-avail)/max(1,tot):.1f}%) — за распознаванием")

print(f"\n{'='*W}\nОШИБКА ЛУЧШЕГО ДОСТУПНОГО КАНДИДАТА (там, где покрытие есть): {len(best_med)} кривых"
      f"\n{'='*W}")
bm = np.array(best_med)
for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 30), (30, 100), (100, 1e9)):
    n = int(((bm > lo) & (bm <= hi)).sum())
    lab = f">{lo}px" if hi > 1e8 else f"{lo}-{hi}px"
    print(f"  {lab:<12}{n:>6}{100*n/max(1,len(bm)):>7.1f}%  " + "█" * int(40 * n / max(1, len(bm))))
print(f"  медиана {np.median(bm):.2f}px, 90-й перцентиль {np.percentile(bm, 90):.1f}px")
print("⇒ " + ("узкое место — ПОРОГ И УТОЧНЕНИЕ: лучшая трасса рядом с 3px"
               if np.median(bm) < 6 else
               "узкое место — ИДЕНТИЧНОСТЬ/ВЕДЕНИЕ: лучшая трасса далеко от кривой"))

bc = np.array(best_cov)
print(f"\nПОКРЫТИЕ лучшего кандидата (порог честности 0.9): медиана {np.median(bc):.2f}, "
      f"доля кривых с покрытием <0.9: {100*np.mean(bc < 0.9):.1f}%")

print(f"\n{'='*W}\nПО СКВАЖИНАМ: точность отгрузки против потолка раскладки\n{'='*W}")
print(f"{'скважина':<26}{'кривых':>7}{'честных':>9}{'%':>7}{'потолок':>9}{'%':>7}")
for wl, v in sorted(per_well.items(), key=lambda q: -q[1]["curves"]):
    print(f"{wl[:25]:<26}{v['curves']:>7}{v['hon']:>9}{100*v['hon']/max(1,v['curves']):>6.0f}%"
          f"{v['avail']:>9}{100*v['avail']/max(1,v['curves']):>6.0f}%")

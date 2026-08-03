r"""_triage.py — СТОИТ ЛИ ОТКАЗЫВАТЬСЯ ОТ «ЧУЖИХ» ЛИСТОВ И ОТДАВАТЬ ИХ ЭКСПЕРТУ (вопрос Эдуарда 02.08).

ЗАЧЕМ. Разброс по семействам восемнадцатикратный (KREMEN 4.2% против VILHIV 75.0%), и одно
семейство BOGAT — это 36% всей выборки. Отсюда два вопроса, на которые нужен ЧИСЛОВОЙ ответ:
  (1) можно ли ЗАРАНЕЕ, не видя эталона, отличить лист, который мы сделаем плохо, от того, который
      сделаем хорошо, — по признакам, доступным проду в момент разбора;
  (2) сколько мы выигрываем в точности НА ОТДАННОМ, если худшие листы отправлять эксперту.

⚠⚠ ЧЕСТНОСТЬ ЗАМЕРА. Считаются ДВЕ кривые отказа:
  ОРАКУЛ  — сортировка листов по их ИСТИННОЙ доле честных (эталон известен). Это ПОТОЛОК триажа,
            он недостижим и нужен только чтобы понять, есть ли что делить вообще.
  ПРИЗНАК — сортировка по признаку, который прод знает БЕЗ эталона (число цветов на листе, кривых
            на трек, линий на лист, ширина полосы). Только это можно построить.
Разрыв между ними и есть цена «мы не знаем заранее».
⚠ Признаки берутся из дампов пула, то есть из того, что прод уже вычислил к моменту раскладки.
Никаких сведений от эксперта в них нет.

  <ComfyUI>\python_embeded\python.exe _triage.py [--pools …]
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
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
        WELL[q.stem[:60]] = wlg.parent.name

rows, seen = [], set()
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        if f.stem in seen:
            continue
        seen.add(f.stem)
        d = pickle.load(open(f, "rb"))
        gts, lines = d.get("gts") or {}, d["lines"]
        if not gts or not lines:
            continue
        # ЧТО ПОЛУЧИЛОСЬ (эталон) — только для оценки, в признаки не идёт
        hon = 0
        for nm, gd in gts.items():
            tr = d.get("written", {}).get(nm)
            if tr and HON(*err(tr, gd)):
                hon += 1
        # ПРИЗНАКИ, доступные проду без эталона
        cnt = Counter(l["track"] for l in lines)
        spans = [max(l["tr"].values()) - min(l["tr"].values()) for l in lines if l["tr"]]
        rows.append(dict(
            sheet=f.stem, well=WELL.get(f.stem, "?"), fam=(WELL.get(f.stem) or "?").split("_")[0],
            curves=len(gts), hon=hon, frac=hon / max(1, len(gts)),
            colors=len({l["color"] or "black" for l in lines}),
            nlines=len(lines), per_track=float(np.mean(list(cnt.values()))) if cnt else 0.0,
            span=float(np.median(spans)) if spans else 0.0,
            lines_per_curve=len(lines) / max(1, len(gts))))
if not rows:
    sys.exit("нет данных")
NC = sum(r["curves"] for r in rows); NH = sum(r["hon"] for r in rows)
print(f"листов {len(rows)}, кривых {NC}, честных {NH} ({100*NH/NC:.1f}%)\n")


def curve(key, rev, tag):
    """Доля честных СРЕДИ ОТДАННОГО при отказе от худших по ключу."""
    srt = sorted(rows, key=lambda r: r[key], reverse=rev)
    out = []
    for keep in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2):
        n = max(1, int(len(srt) * keep))
        sel = srt[:n]
        c = sum(r["curves"] for r in sel); h = sum(r["hon"] for r in sel)
        out.append((keep, c, h, 100 * h / max(1, c)))
    print(f"{tag:<34}" + "".join(f"{100*k:>5.0f}%" for k, _, _, _ in out))
    print(f"{'  точность на отданном':<34}" + "".join(f"{p:>5.1f}" for _, _, _, p in out))
    print(f"{'  кривых отдано':<34}" + "".join(f"{c:>6}" for _, c, _, _ in out))
    return out


print(f"{'доля листов, которые ОТДАЁМ':<34}" + "".join(f"{v:>5}%" for v in (100, 80, 60, 50, 40, 30, 20)))
print("-" * 76)
curve("frac", True, "★ ОРАКУЛ (потолок, недостижим)")
print()
for key, rev, tag in (("colors", True, "по числу цветов на листе"),
                      ("per_track", False, "по кривым на трек (реже — лучше)"),
                      ("nlines", False, "по числу линий на листе"),
                      ("lines_per_curve", False, "по линий/кривых")):
    curve(key, rev, tag)
    print()

print("⚠ Читать так: если признак даёт кривую, близкую к оракулу, — триаж строится; если он идёт")
print("  вровень со строкой 100% (то есть отказ ничего не меняет), — признак бесполезен.")

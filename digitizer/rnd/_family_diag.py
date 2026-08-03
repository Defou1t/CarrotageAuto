r"""_family_diag.py — ПОЧЕМУ У ОДНИХ СКВАЖИН ПОТОЛОК 80%, А У ДРУГИХ 7%. Разбор по семействам.

⚠ ЗАЧЕМ (§6.91). `_accuracy_status.py` показал разброс, который переворачивает приоритеты: три
крупнейшие скважины (BOGAT_011/014/015 — 652 кривые, 35% выборки) имеют ПОТОЛОК 7-13%, то есть в
пуле для них нет годных трасс вовсе, а у Semeguniv_020 и VILHIV_055 потолок 75-81% и узкое место —
раскладка. Значит «средняя точность по проекту» — бесполезное число: это две разные задачи.
Здесь считается, ЧЕМ семейства отличаются измеримо, чтобы следующий шаг выбирался по причине, а не
по среднему.

СЧИТАЕТСЯ НА ЛИСТ И НА КРИВУЮ: сколько трасс в пуле, сколько кривых, сколько кривых на трек,
покрытие лучшего кандидата, ошибка лучшего кандидата, доля кривых с полным промахом (>100px),
доля кривых, где лучший кандидат ЕСТЬ (≤3px). Плюс цвета: сколько разных цветов линий на листе
(монохромный бланк против цветного — §6.13).

  <ComfyUI>\python_embeded\python.exe _family_diag.py
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

# семейство = префикс до первого подчёркивания (BOGAT_011 → BOGAT): у одного заказчика/партии
# бланки, почерк и чернила общие, поэтому и ломается всё семейство целиком, а не скважина
fam_of = lambda w: (w or "?").split("_")[0]

F = {}
# ⚠ КЛЮЧ ДЕДУПЛИКАЦИИ — ИМЯ ФАЙЛА, А НЕ ПОЛЕ `name`: `_pool_oracle.py` кладёт дамп как
# `{nlgx.stem[:60]}.pkl`, поэтому дубль виден ДО чтения и стоит ноль. ⚠ В отличие от
# `_start_probe`/`_param_sweep` здесь нет отдельного прохода «сначала список, потом счёт»: дамп
# нужен целиком, и 2.35 ГБ читаются по делу — экономится только повторное чтение дублей.
# Совпадение ключа с полем `name` проверяется при чтении, чтобы рассинхронизация не осталась немой.
seen = set()
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        if f.stem in seen:
            continue
        seen.add(f.stem)
        d = pickle.load(open(f, "rb"))
        if not d["name"].startswith(f.stem[:40]):
            print(f"  ⚠ имя дампа {f.stem[:40]!r} расходится с полем name {d['name'][:40]!r}")
        fam = fam_of(WELL.get(d["name"]))
        s = F.setdefault(fam, dict(sheets=0, curves=0, lines=0, hon=0, avail=0, colors=[],
                                   best=[], cov=[], miss=0, ntr=[], cpt=[]))
        s["sheets"] += 1
        s["lines"] += len(d["lines"])
        s["colors"].append(len({ln["color"] for ln in d["lines"]}))
        slots = {q["name"]: q for q in d["slots"]}
        tl = {}
        for i, ln in enumerate(d["lines"]):
            tl.setdefault(ln["track"], []).append(i)
        s["ntr"].append(len(tl))
        by_track = {}
        for nm in d["gts"]:
            if nm in slots:
                by_track.setdefault(slots[nm]["track"], 0)
                by_track[slots[nm]["track"]] += 1
        s["cpt"].extend(by_track.values())
        for nm, gt in d["gts"].items():
            s["curves"] += 1
            q = slots.get(nm)
            cand = tl.get(q["track"], []) if q else []
            if not cand:
                s["miss"] += 1
                continue
            ms = [err(d["lines"][i]["tr"], gt) for i in cand]
            cov = max([c for _, c in ms] or [0.0])
            s["cov"].append(cov)
            ok = [m for m, c in ms if c >= 0.9 and m is not None]
            if not ok:
                s["miss"] += 1
                continue
            b = min(ok)
            s["best"].append(b)
            s["avail"] += 1 if b <= 3.0 else 0
            w = d["written"].get(nm)
            if w is not None:
                m2, c2 = err(w, gt)
                s["hon"] += 1 if (m2 is not None and m2 <= 3.0 and c2 >= 0.9) else 0

W = 118
print("=" * W)
print("СЕМЕЙСТВА БЛАНКОВ: почему потолок разный (сортировка по числу кривых)")
print("=" * W)
print(f"{'семейство':<12}{'листов':>7}{'кривых':>7}{'честно':>8}{'потолок':>9}"
      f"{'трасс/лист':>11}{'кривых/трек':>12}{'цветов':>8}{'медиана ошибки':>15}{'>100px':>8}")
rows = sorted(F.items(), key=lambda q: -q[1]["curves"])
for fam, s in rows:
    b = np.array(s["best"]) if s["best"] else np.array([np.nan])
    print(f"{fam[:11]:<12}{s['sheets']:>7}{s['curves']:>7}"
          f"{100*s['hon']/max(1,s['curves']):>7.0f}%{100*s['avail']/max(1,s['curves']):>8.0f}%"
          f"{s['lines']/max(1,s['sheets']):>11.1f}{np.mean(s['cpt'] or [0]):>12.1f}"
          f"{np.mean(s['colors']):>8.1f}{np.nanmedian(b):>14.0f}px"
          f"{100*np.mean(b > 100):>7.0f}%")

print("\n" + "=" * W)
print("ЧТО ЭТО ЗНАЧИТ: сравнение двух крайних групп")
print("=" * W)
big = [f for f, s in rows if s["curves"] >= 100 and s["avail"] / max(1, s["curves"]) < 0.15]
good = [f for f, s in rows if s["avail"] / max(1, s["curves"]) >= 0.35]
agg = lambda fams, k: sum(F[f][k] for f in fams)
for tag, fams in (("ПЛОХИЕ (потолок <15%)", big), ("ХОРОШИЕ (потолок ≥35%)", good)):
    if not fams:
        continue
    b = np.concatenate([np.array(F[f]["best"]) for f in fams if F[f]["best"]])
    print(f"\n{tag}: {', '.join(fams)}")
    print(f"  листов {agg(fams,'sheets')}, кривых {agg(fams,'curves')}, "
          f"честно {100*agg(fams,'hon')/max(1,agg(fams,'curves')):.0f}%, "
          f"потолок {100*agg(fams,'avail')/max(1,agg(fams,'curves')):.0f}%")
    print(f"  трасс на лист {agg(fams,'lines')/max(1,agg(fams,'sheets')):.1f}, "
          f"кривых на трек {np.mean([v for f in fams for v in F[f]['cpt']]):.1f}, "
          f"цветов на лист {np.mean([v for f in fams for v in F[f]['colors']]):.1f}")
    print(f"  ошибка лучшего кандидата: медиана {np.median(b):.0f}px, "
          f"доля >100px {100*np.mean(b > 100):.0f}%, доля ≤3px {100*np.mean(b <= 3):.0f}%")
    print(f"  кривых без доступного кандидата (нет линий / cov<0.9): "
          f"{100*agg(fams,'miss')/max(1,agg(fams,'curves')):.0f}%")

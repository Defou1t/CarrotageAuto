r"""_oracle_ladder_sum.py — СВОДКА ЛЕСТНИЦЫ ОРАКУЛОВ: детекция против назначения (§6.93).

Читает шарды `_oracle_ladder.py` и отвечает на один вопрос: из кривых, которые прод НЕ БЕРЁТ,
сколько не имеют туши в маске (⇒ детекция) и сколько имеют, но ран выбран не тот (⇒ назначение).

⚠ Порог «тушь есть» = blk3 ≥ 0.9: та же терпимость, что у метрики честности (покрытие ≥ 0.9).
⚠ L1 — оракульный выбор рана, поэтому корзина «назначение» — это ПОТОЛОК глобальной линковки
(laptrack/Stone Soup, §6.92), а не обещание. Контроль ширины рана печатается рядом: если ран широкий,
L1=0 достигается тривиально и обещание пустое.

  <ComfyUI>\python_embeded\python.exe _oracle_ladder_sum.py
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:\nds\output\taskS\oracle_ladder_v2")
ap.add_argument("--ink-thr", type=float, default=0.9)
ap.add_argument("--l1-thr", type=float, default=3.0)
# ⚠⚠ §6.94: корзины считаются по маске ЦВЕТА кривой (`col3`), как трассирует прод, а не по чёрной
# (`blk3`) — на чёрной все 139 красных SP/PS ложно попадали в «туши нет». Ключ переключается флагом,
# чтобы старые числа можно было воспроизвести и сравнить.
ap.add_argument("--key", default="col3", choices=["col3", "blk3", "ink3"])
a = ap.parse_args()

R = []
for f in sorted(Path(a.dir).glob("rows_*.pkl")):
    R.extend(pickle.load(open(f, "rb")))
print(f"кривых собрано {len(R)} из {len({r['sheet'] for r in R})} листов, "
      f"семейств {len({r['fam'] for r in R})}")
if not R:
    sys.exit(1)

W = 96
has = lambda r: r.get(a.key, r["blk3"]) >= a.ink_thr
near = lambda r: (r["l1med"] == r["l1med"]) and r["l1med"] <= a.l1_thr
print(f"критерий «тушь есть»: {a.key} ≥ {a.ink_thr}")
if all("col3" in r for r in R) and a.key == "col3":
    nb = sum(1 for r in R if r["col3"] >= a.ink_thr) - sum(1 for r in R if r["blk3"] >= a.ink_thr)
    print(f"★ переход с чёрной маски на маску цвета кривой: кривых «тушь есть» {nb:+d}")
    byc = {}
    for r in R:
        byc.setdefault(r.get("colname", "?"), []).append(r)
    print("  по словарному цвету: " + ", ".join(
        f"{c} {len(v)} (чёрная {100*np.mean([x['blk3'] >= a.ink_thr for x in v]):.0f}% → "
        f"своя {100*np.mean([x['col3'] >= a.ink_thr for x in v]):.0f}%)"
        for c, v in sorted(byc.items(), key=lambda q: -len(q[1]))))


def bucket(r):
    if r["prod"]:
        return "0 прод берёт честно"
    if not has(r):
        return "1 ТУШИ НЕТ В МАСКЕ ⇒ детекция"
    if near(r):
        return "2 тушь есть, ран рядом ⇒ НАЗНАЧЕНИЕ"
    return "3 тушь есть, ран не тот локально"


B = {}
for r in R:
    B.setdefault(bucket(r), []).append(r)
print(f"\n{'='*W}\nРАЗЛОЖЕНИЕ: почему кривая не записана честно\n{'='*W}")
print(f"{'корзина':<38}{'кривых':>8}{'доля':>7}{'L1 медиана':>12}{'ширина рана':>13}{'ранов/стр':>10}")
for k in sorted(B):
    v = B[k]
    l1 = np.array([r["l1med"] for r in v], float)
    wd = np.array([r["widmed"] for r in v], float)
    nr = np.array([r["nrunmed"] for r in v], float)
    print(f"{k:<38}{len(v):>8}{100*len(v)/len(R):>6.1f}%{np.nanmedian(l1):>11.1f}px"
          f"{np.nanmedian(wd):>11.0f}px{np.nanmedian(nr):>10.1f}")

miss = B.get("1 ТУШИ НЕТ В МАСКЕ ⇒ детекция", [])
asg = B.get("2 тушь есть, ран рядом ⇒ НАЗНАЧЕНИЕ", [])
loc = B.get("3 тушь есть, ран не тот локально", [])
ok = B.get("0 прод берёт честно", [])
print(f"\n{'-'*W}")
print(f"★★ ПОТОЛОК ПРАВИЛЬНОГО ВЫБОРА РАНА: {len(ok)+len(asg)} кривых из {len(R)} = "
      f"{100*(len(ok)+len(asg))/len(R):.1f}%  (сейчас {100*len(ok)/len(R):.1f}%)")
print(f"   ⇒ назначение может добавить до {len(asg)} кривых ({100*len(asg)/len(R):.1f} п.п.)")
print(f"★ ДЕТЕКЦИЯ (туши нет в маске): {len(miss)} кривых ({100*len(miss)/len(R):.1f}%) — "
      f"порог/гейт, никакая линковка их не берёт")
print(f"⚠ ран не тот даже локально: {len(loc)} ({100*len(loc)/len(R):.1f}%)")

print(f"\n{'='*W}\nКОНТРОЛЬ: не пуст ли потолок (широкий ран делает L1=0 даром)\n{'='*W}")
for tag, v in (("назначение", asg), ("прод берёт", ok), ("ран не тот", loc)):
    if not v:
        continue
    wd = np.array([r["widmed"] for r in v], float)
    nr = np.array([r["nrunmed"] for r in v], float)
    mg = np.array([r["merge"] for r in v], float)
    print(f"  {tag:<12} ширина рана медиана {np.nanmedian(wd):>5.0f}px, p90 "
          f"{np.nanpercentile(wd, 90):>5.0f}px; ранов на строку {np.nanmedian(nr):>4.1f}; "
          f"слияние с чужой кривой медиана {100*np.nanmedian(mg):>4.0f}%")

print(f"\n{'='*W}\nПО СЕМЕЙСТВАМ: где детекция, а где назначение\n{'='*W}")
print(f"{'семейство':<13}{'кривых':>7}{'честно':>8}{'детекция':>10}{'назначение':>12}"
      f"{'локально':>10}{'потолок выбора':>15}{'ink3':>7}{'blk3':>7}")
fams = {}
for r in R:
    fams.setdefault(r["fam"], []).append(r)
for fam, v in sorted(fams.items(), key=lambda q: -len(q[1])):
    b = [bucket(r) for r in v]
    c0 = b.count("0 прод берёт честно"); c1 = b.count("1 ТУШИ НЕТ В МАСКЕ ⇒ детекция")
    c2 = b.count("2 тушь есть, ран рядом ⇒ НАЗНАЧЕНИЕ"); c3 = b.count("3 тушь есть, ран не тот локально")
    print(f"{fam[:12]:<13}{len(v):>7}{100*c0/len(v):>7.0f}%{100*c1/len(v):>9.0f}%"
          f"{100*c2/len(v):>11.0f}%{100*c3/len(v):>9.0f}%{100*(c0+c2)/len(v):>14.0f}%"
          f"{100*np.mean([r['ink3'] for r in v]):>6.0f}%{100*np.mean([r['blk3'] for r in v]):>6.0f}%")

print(f"\n{'='*W}\nАСИММЕТРИЯ ГЕЙТОВ: детекция (ink) против трассировки (blk)\n{'='*W}")
ink = np.array([r["ink3"] for r in R]); blk = np.array([r["blk3"] for r in R])
print(f"кривых, где тушь ЕСТЬ у детекции, но НЕТ у трассировки (ink3≥0.9 > blk3<0.9): "
      f"{int(((ink >= 0.9) & (blk < 0.9)).sum())}")
print(f"обратное (blk3≥0.9, ink3<0.9): {int(((blk >= 0.9) & (ink < 0.9)).sum())}")
print(f"съедено structure_mask: медиана {100*np.median([r['strf'] for r in R]):.0f}% точек GT; "
      f"кривых с >30% съеденных: {sum(1 for r in R if r['strf'] > 0.3)}")
print(f"в вырожденном цветовом канале: кривых с >30% точек: "
      f"{sum(1 for r in R if r['degf'] > 0.3)}")

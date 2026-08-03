r"""_param_sweep_sum.py — сводка свипа `_param_sweep.py` с контролями, без которых число не читается.

Печатает на каждое сочетание:
  ЧЕСТНЫХ      — медиана |Δx| ≤3px И покрытие ≥90% (метрика приёмки);
  С ПОМЕТКОЙ   — медиана ≤10px И покрытие ≥90% (продуктовая корзина §6.96);
  ЛЕСТНИЦА     — ≤3 / 3-10 / 10-30 / >30px (§6.96: иначе прогресс 90px→20px выглядит нулём);
  УДЕРЖАНО     — сколько ПРОД-честных кривых сочетание сохраняет (контроль риска §6.99);
  ±БАЗА        — состав: сколько кривых сочетание отбирает у дефолта и сколько добавляет.
Плюс разрез по семействам для лучших сочетаний — деградация одного семейства при общем плюсе
уже ловилась (§6.99: BEZLUD −1) и обязана быть видна до внесения.

  <ComfyUI>\python_embeded\python.exe _param_sweep_sum.py [--tag oat] [--top 8]
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:\nds\output\taskS\param_sweep")
ap.add_argument("--tag", default="oat")
ap.add_argument("--top", type=int, default=8)
a = ap.parse_args()

names, rows, prog = None, [], []
for f in sorted(Path(a.dir).glob(f"{a.tag}_*of*.pkl")):
    d = pickle.load(open(f, "rb"))
    if names is None:
        names = d["names"]
    elif names != d["names"]:
        print(f"⚠ {f.name}: другой набор сочетаний — пропущен"); continue
    rows += d["rows"]
    prog.append((d.get("done", 0), d.get("total", 0)))
if not rows:
    sys.exit("нет данных")
dn, tl = sum(p[0] for p in prog), sum(p[1] for p in prog)
if dn < tl:
    print(f"⚠⚠ СВОДКА ПО НЕПОЛНОМУ ПРОГОНУ: {dn} из {tl} листов ({100*dn/max(1,tl):.0f}%). "
          f"Порядок сочетаний обычно устаканивается раньше конца, но ЧИСЛА цитировать нельзя.\n")

fam_of = lambda w: (w or "?").split("_")[0]
N = len(rows)
POOL = [i for i, r in enumerate(rows) if r["pool"]]
print(f"кривых {N}, листов {len({r['sheet'] for r in rows})}, "
      f"семейств {len({fam_of(r['well']) for r in rows})}, прод-честных {len(POOL)}")
print(f"шардов прочитано {len(list(Path(a.dir).glob(f'{a.tag}_*of*.pkl')))}\n")


def col(j):
    hon = np.array([r["res"][j][0] for r in rows], bool)
    med = np.array([r["res"][j][1] for r in rows], float)
    cov = np.array([r["res"][j][2] for r in rows], float)
    return hon, med, cov


BH, BM, BC = col(0)
base_set = set(np.flatnonzero(BH).tolist())
out = []
for j, nm in enumerate(names):
    h, m, c = col(j)
    ok = ~np.isnan(m)
    lad = [int(((m <= 3) & ok).sum()), int(((m > 3) & (m <= 10) & ok).sum()),
           int(((m > 10) & (m <= 30) & ok).sum()), int((((m > 30) & ok) | ~ok).sum())]
    s = set(np.flatnonzero(h).tolist())
    out.append(dict(j=j, name=nm, hon=int(h.sum()),
                    mark=int(((m <= 10) & (c >= 0.9) & ok).sum()), lad=lad,
                    keep=int(sum(h[i] for i in POOL)),
                    gain=len(s - base_set), lose=len(base_set - s),
                    med=float(np.nanmedian(m[ok])) if ok.any() else float("nan")))

b = out[0]
print(f"{'сочетание':<22}{'честных':>8}{'Δ':>5}{'помет':>7}{'удерж':>7}"
      f"{'+':>4}{'−':>4}{'мед':>7}   лестница ≤3 / 3-10 / 10-30 / >30")
for o in sorted(out, key=lambda q: (-q["hon"], q["j"])):
    star = " ★" if o["hon"] > b["hon"] else ("  " if o["j"] else " ·")
    print(f"{o['name']:<22}{o['hon']:>8}{o['hon']-b['hon']:>+5}{o['mark']:>7}"
          f"{o['keep']:>4}/{len(POOL):<3}{o['gain']:>4}{o['lose']:>4}{o['med']:>7.1f}   "
          f"{o['lad'][0]:>4} {o['lad'][1]:>5} {o['lad'][2]:>6} {o['lad'][3]:>5}{star}")

print(f"\nбаза = {names[0]} (band_pad=8 slmax=30 wide_run=14 jump_limit=None), "
      f"честных {b['hon']} ({100*b['hon']/N:.1f}%), прод-пул {len(POOL)} ({100*len(POOL)/N:.1f}%)")

# ── РАЗРЕЗ ПО СЕМЕЙСТВАМ для лучших сочетаний: общий плюс не должен прятать локальный обвал ──
best = [o for o in sorted(out, key=lambda q: -q["hon"])[:a.top] if o["j"] != 0]
fams = sorted({fam_of(r["well"]) for r in rows},
              key=lambda f: -sum(1 for r in rows if fam_of(r["well"]) == f))
print(f"\n{'='*100}\nПО СЕМЕЙСТВАМ (честных; в скобках Δ к базе). Показаны {len(fams)} семейств\n")
hdr = f"{'сочетание':<22}"
for f in fams:
    hdr += f"{f[:7]:>9}"
print(hdr + f"{'ИТОГО':>9}")
print(f"{'кривых':<22}" + "".join(f"{sum(1 for r in rows if fam_of(r['well'])==f):>9}"
                                  for f in fams) + f"{N:>9}")
bf = {f: sum(r["res"][0][0] for r in rows if fam_of(r["well"]) == f) for f in fams}
print(f"{'★ база':<22}" + "".join(f"{bf[f]:>9}" for f in fams) + f"{b['hon']:>9}")
for o in best:
    line = f"{o['name']:<22}"
    for f in fams:
        v = sum(r["res"][o["j"]][0] for r in rows if fam_of(r["well"]) == f)
        line += f"{v:>5}{v-bf[f]:>+4}" if v != bf[f] else f"{v:>5}    "
    line += f"{o['hon']:>5}{o['hon']-b['hon']:>+4}"
    print(line)

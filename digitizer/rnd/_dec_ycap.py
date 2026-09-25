r"""_dec_ycap.py — ОБРЕЗАЕТ ЛИ ДЕКОДЕР КРИВУЮ ПО ВЕРТИКАЛИ ВМЕСТЕ С U1 (§6.203 → B1b, кандидат 2).

ОТКУДА ВОПРОС. Декодер ведёт трек только в диапазоне `y0 = min(L.y0)`, `y1 = max(L.y1)` по линиям
U1 (`rowdec.py:370`, жёстко: `rowdec.py:222`). Если U1 нашёл на треке лишь короткие линии,
эталонная кривая за их пределами для декодера не существует, и cov ≥ 0.9 недостижим по построению.

⚠⚠ ЛОГИКА ГРАНИЦЫ — ИСПРАВЛЕНА РЕВИЗИЕЙ (12.09). Первая редакция считала объединение строк ТРАСС
U1 из пуловых дампов (`lines[*]['tr']`) лежащим ВНУТРИ штриховых границ линий и называла меру
ВЕРХНЕЙ границей. Это неверно: `trace2d._extend_ends` (`trace2d.py:96-114`) доводит трассу ЗА штрихи
по связному чернилу без предела длины (стоп — 25 пустых строк), и то же делает оконный селектор;
диапазон трасс ШИРЕ [y0, y1] декодера (у 803 треков из 1177 выше min y0, у 840 ниже max y1, медиана
выхода 69 px). ⇒ Доля строк эталона вне трасс U1 — НИЖНЯЯ граница обрезки. Точный пересчёт по
восстановленному [y0, y1] (глубины линий из `_understanding.json` + `frame.top_y`/`px_per_m` +
`top_depth` из nlgx): кривых с >10% строк вне — 39 из 2737 (стенд даёт 34), декодер взял 0.
Вывод «ограничение не работает» переживает пересчёт; сам стенд оставлен как есть — его число
занижено на 5 кривых и это записано здесь.

ЧТО ПЕЧАТАЕТ: по кривым честного поля — доля строк вне объединённого диапазона трасс U1 трека,
по корзинам; сколько кривых с долей вне > 10% (для них cov ≥ 0.9 у декодера невозможен даже по
нижней оценке) и сколько из них декодер всё же взял.

  <ComfyUI>\python_embeded\python.exe _dec_ycap.py
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict, Counter
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dump", default="u1_vs_dec.pkl")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
a = ap.parse_args()
TS = Path(a.ts)

D = pickle.load(open(TS / a.dump, "rb"))
flags = {(sh, t, nm): f for sh, t, nm, k, f in D}
bytr = defaultdict(list)
for sh, t, nm, k, f in D:
    bytr[(sh, t)].append(nm)
stem2f = {}
for root in ("pools", "pools_gate", "pools_wide", "pools_more", "pools_div",
             "pools_heldout", "pools_all"):
    for f in sorted((TS / root).glob("*.pkl")):
        stem2f.setdefault(f.stem, f)
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q

BINS = [(0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.30), (0.30, 1.01)]
tab = defaultdict(Counter)
sheets = sorted({sh for sh, _ in bytr})
done = miss = 0
for si, sh in enumerate(sheets, 1):
    f, q = stem2f.get(Path(sh).stem), SRC.get(sh)
    if not f or not q:
        miss += 1
        continue
    done += 1
    d = pickle.load(open(f, "rb"))
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    rng = {}
    for L in d["lines"]:
        t = int(L["track"]); ys = L["tr"]
        if not ys:
            continue
        lo, hi = min(ys), max(ys)
        if t in rng:
            rng[t] = (min(rng[t][0], lo), max(rng[t][1], hi))
        else:
            rng[t] = (lo, hi)
    for t, ns in ((t, ns) for (s, t), ns in bytr.items() if s == sh):
        for nm in ns:
            g = gts.get(nm)
            if not g:
                continue
            if t not in rng:
                out = 1.0
            else:
                lo, hi = rng[t]
                out = sum(1 for y in g if y < lo or y > hi) / max(1, len(g))
            fl = flags[(sh, t, nm)]
            for b0, b1 in BINS:
                if b0 <= out < b1:
                    k = (b0, b1)
                    tab[k]["кривых"] += 1
                    tab[k]["DEC"] += fl["DEC"]; tab[k]["CUR"] += fl["CUR"]
                    tab[k]["никем"] += not fl["P965∪DEC"]
                    break
    if si % 300 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

print(f"СВЕРКА СПИСКА: обработано {done} + пропущено {miss} = {done+miss} против {len(sheets)}"
      f"   {'★ СОШЛОСЬ' if done+miss == len(sheets) else '⛔ НЕ СОШЛОСЬ'}")
tot = sum(v["кривых"] for v in tab.values())
print(f"кривых {tot}")
print("\n★★ ДОЛЯ СТРОК ЭТАЛОНА ВНЕ ОБЪЕДИНЁННОГО ДИАПАЗОНА ТРАСС U1 ТРЕКА (НИЖНЯЯ граница обрезки декодера — см. шапку)")
print("| вне диапазона | кривых | декодер взял | нынешний прод | никем |")
print("|---|---|---|---|---|")
for k in BINS:
    v = tab[k]
    lab = f"{int(k[0]*100)}-{int(min(k[1],1.0)*100)}%" if k[1] <= 1.0 else f"≥{int(k[0]*100)}%"
    print(f"| {lab} | {v['кривых']} | {v['DEC']} ({100*v['DEC']/max(1,v['кривых']):.0f}%) | "
          f"{v['CUR']} | {v['никем']} |")
cap = sum(tab[k]["кривых"] for k in BINS if k[0] >= 0.10)
capd = sum(tab[k]["DEC"] for k in BINS if k[0] >= 0.10)
print(f"\n★ Кривых с >10% строк вне диапазона (нижняя оценка; точный пересчёт ревизии — 39): "
      f"{cap} из {tot} ({100*cap/max(1,tot):.1f}%); декодер всё же взял {capd} — "
      f"{'экран груб' if capd else 'экран согласован'}.")

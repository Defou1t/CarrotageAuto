r"""_err_kind.py — ЧТО ИМЕННО не так с выданной кривой: ветка масштаба, сдвиг или чужая кривая.

§6.76 показал: даже при ИДЕАЛЬНОЙ раскладке лучшая наша трасса отстоит от экспертной кривой на
~130px, то есть в сорок раз дальше порога честности (3px). Значит дело не в выборе рана и не в
назначении. Здесь остаток раскладывается по природе.

ИДЕЯ. Если трасса ЛЕЖИТ НА ЧЕРНИЛАХ (durable-замер 19.07: 99.9-100%), но промахивается на сотни
px, возможны ровно три случая, и они различимы аффинной подгонкой x:
  ВЕТКА МАСШТАБА — тот же зонд перерисован в другом масштабе ОТДЕЛЬНОЙ линией. Тогда существует
      a != 1 и b, при которых a*наш_x + b ложится на эксперта с малой невязкой. Чинить
      `decode_levels`, а не трассировку;
  СДВИГ         — a ~= 1, но b != 0: та же кривая, смещённая (сбой калибровки/рамки);
  ЧУЖАЯ КРИВАЯ  — невязка остаётся большой при ЛЮБЫХ a, b: трасса на посторонней линии. Чинить
      идентичность.
⚠ Подгонка ведётся по МЕДИАНЕ (не МНК): один выброс не должен решать классификацию.

  <ComfyUI>\python_embeded\python.exe _err_kind.py <dir с *_auto.nlgx>
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--tol", type=float, default=3.0, help="порог «легло» по медиане невязки, px")
ap.add_argument("--arch", default=r"F:\nds\projects\Archive")
a = ap.parse_args()

src_by_stem = {}
for wlg in Path(a.arch).glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        src_by_stem.setdefault(q.stem, q)
for extra in (r"F:\nds\projects\Semeguniv_001\wlg", r"F:\nds\projects\Semeguniv_020\wlg"):
    if Path(extra).is_dir():
        for q in Path(extra).glob("*.nlgx"):
            src_by_stem.setdefault(q.stem, q)


def series(c):
    return {c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL}


def affine_fit(ox, gx):
    """Медианная аффинная подгонка gx ~ a*ox + b перебором a по сетке масштабов.

    Сетка НЕ произвольная: это отношения реальных шкал планшета (×1/×5/×25 и обратные), плюс
    единица. Свободный перебор a дал бы «подгонку под шум» на кривой с малым размахом x."""
    best = None
    for aa in (1.0, 5.0, 0.2, 25.0, 0.04, 0.5, 2.0):
        b = float(np.median(gx - aa * ox))
        r = float(np.median(np.abs(aa * ox + b - gx)))
        if best is None or r < best[2]:
            best = (aa, b, r)
    return best


rows = []
for auto in sorted(Path(a.dir).glob("*_auto.nlgx")):
    src = src_by_stem.get(auto.stem[:-5])
    if not src:
        continue
    A = extract(str(auto)); G = extract(str(src))
    ours = {c["name"]: series(c) for c in A["curves"]
            if M.mnem_root(c["name"]) != "DA" and any(x != NULL for x in c["xs"])}
    gts = {c["name"]: dense(c) for c in G["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    for nm, od in ours.items():
        gd = gts.get(nm)
        if not gd:
            continue
        com = [y for y in od if y in gd]
        if len(com) < 30:
            continue
        ox = np.array([od[y] for y in com], float)
        gx = np.array([gd[y] for y in com], float)
        raw = float(np.median(np.abs(ox - gx)))
        aa, bb, res = affine_fit(ox, gx)
        if raw <= a.tol:
            kind = "уже верна"
        elif res <= a.tol and abs(aa - 1.0) < 1e-9:
            kind = "СДВИГ"
        elif res <= a.tol:
            kind = "ВЕТКА МАСШТАБА"
        elif res < raw * 0.5:
            kind = "частично (аффинно лучше)"
        else:
            kind = "ЧУЖАЯ КРИВАЯ"
        rows.append((kind, nm, raw, aa, bb, res, auto.stem[:34]))

print(f"\nвсего сопоставленных кривых: {len(rows)}  (порог «легло» {a.tol}px)\n")
order = ["уже верна", "СДВИГ", "ВЕТКА МАСШТАБА", "частично (аффинно лучше)", "ЧУЖАЯ КРИВАЯ"]
for k in order:
    sel = [r for r in rows if r[0] == k]
    if not sel:
        continue
    med = float(np.median([r[2] for r in sel]))
    print(f"{k:<26} {len(sel):>3} кривых ({100*len(sel)/len(rows):>4.0f}%)   медиана сырой ошибки {med:>7.1f}px")
    for r in sorted(sel, key=lambda q: -q[2])[:4]:
        print(f"      сырая {r[2]:>7.1f} → после (a={r[3]}, b={r[4]:+.0f}) {r[5]:>7.1f}   {r[1][:16]:<17}{r[6]}")

print("\nЧитать так: ВЕТКА МАСШТАБА → чинить decode_levels; СДВИГ → калибровку/рамку;")
print("ЧУЖАЯ КРИВАЯ → идентичность. Доля каждой группы и есть распределение работы.")

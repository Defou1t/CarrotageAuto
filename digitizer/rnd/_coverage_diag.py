r"""_coverage_diag.py — ПУЛ «ТОЧНА_НО_НЕПОЛНА»: восстановимое покрытие или ловушка добивки?

Прод-гейт (_prod_gate.py, Semeguniv) дал 19 кривых med<=3px но cov<0.9 — трасса идёт по СВОЕЙ
кривой идеально, но кроет 32-89% её высоты. Это НЕ проблема идентичности (§6.22) и НЕ латч —
это НЕДОБОР ПОКРЫТИЯ. Один шаг от честной. Но §6.20.3 durable: наращивать cov можно только
там, где под пропущенными строками РЕАЛЬНО есть тушь своей кривой; иначе это метрический
артефакт (нуль-контроль: сдвиг добитых точек на 1000px не менял «честных»).

Гейт-разделитель. Для каждой такой кривой берём строки GT, которые НАШ auto НЕ покрыл, и делим:
  • ВНЕ ОКНА  — строка за [frame.top_y, frame.bottom_y] (окно анализа обрезало кривую);
  • ЕСТЬ ТУШЬ — под точкой GT в ±tol есть чернила своей маски (трасса ОБОРВАЛАСЬ на живой туши
                → восстановимо: расширить окно / поднять max_gap / добить ПО ТУШИ);
  • НЕТ ТУШИ  — чернил нет (пунктир/выцветание → любая добивка = ловушка, не трогать).

  python _coverage_diag.py [--tol 4]
"""
import sys, io, json, argparse, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
import cv2
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image
from auto import frame as F, understand as U, confidence as C, trace2d as T, meta as M
from auto import imaging as im
from auto.config import DEFAULT

SEM = Path(r"F:\nds\projects\Semeguniv_001\wlg")
# кривые из пула ТОЧНА_НО_НЕПОЛНА (из prod_semeguniv_base.json), сгруппированы по листу
POOL = {
    "Semeguniv_1_STK_200_3390_500_D1": ["SP1", "GZ1", "PZ1"],
    "Semeguniv_1_STK+DS_10_200_500_D1": ["PZ1", "DS1", "GZ1", "SP1"],
    "Semeguniv_1_MK_190_1490_200_D1": ["MGZ1", "MPZ1"],
    "Semeguniv_1_BKZ_3690_3950_200_D1": ["GZ21", "GZ31"],
    "Semeguniv_1_RK_00_3400_200_D1": ["NGK1"],
}
ap = argparse.ArgumentParser()
ap.add_argument("--tol", type=int, default=4)
a = ap.parse_args()
p = DEFAULT.cv

TOT = {"вне_окна": 0, "есть_тушь": 0, "нет_туши": 0, "всего_пропущено": 0}
for stem, names in POOL.items():
    n = SEM / f"{stem}.nlgx"
    rgb = cv2.cvtColor(cv2.imread(str(find_image(n)), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m = M.parse_filename(n.name, r"F:\nds\Auto\mnemonics.json")
        fr = F.frame_from_nlgx(str(n), m, p, rgb=rgb)
        if getattr(fr, "row_shift", None) is not None:
            rgb = F.apply_row_shift(rgb, fr.row_shift)
        sheet = U.understand(rgb, fr, m, p); C.classify(sheet)
        traces = T.trace_auto(rgb, sheet, p)
    fg = im.ink_foreground(rgb, p) > 0
    H, W = fg.shape
    mo = extract(str(n))
    gts = {c["name"].split()[0]: c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])}
    # наш auto по слоту: сопоставляем трассу к GT по имени слота линии
    print(f"\n### {stem}  окно y[{fr.top_y}..{fr.bottom_y}]")
    for nm in names:
        g = gts.get(nm)
        if g is None:
            print(f"   {nm}: нет GT"); continue
        gxs = {g["top_y"] + i: x for i, x in enumerate(g["xs"]) if x != NULL}
        # наша трасса, ближайшая к GT по медиане (как гейт сопоставляет слот)
        gmed = np.median(list(gxs.values()))
        best, bd = None, 1e9
        for L, tr in traces:
            if not tr:
                continue
            cy = [y for y in tr if y in gxs]
            if len(cy) < 30:
                continue
            dd = np.median([abs(tr[y] - gxs[y]) for y in cy])
            if dd < bd:
                bd, best = dd, tr
        if best is None:
            print(f"   {nm}: наша трасса не сопоставлена"); continue
        missing = [y for y in gxs if y not in best]
        cls = {"вне_окна": 0, "есть_тушь": 0, "нет_туши": 0}
        for y in missing:
            if y < fr.top_y or y >= fr.bottom_y:
                cls["вне_окна"] += 1; continue
            x = int(round(gxs[y]))
            lo, hi = max(0, x - a.tol), min(W, x + a.tol + 1)
            cls["есть_тушь" if (0 <= y < H and fg[y, lo:hi].any()) else "нет_туши"] += 1
        s = sum(cls.values()) or 1
        for k in cls:
            TOT[k] += cls[k]
        TOT["всего_пропущено"] += len(missing)
        print(f"   {nm:<6} med {bd:4.1f}  покрыто {len(best & gxs.keys()) if False else len([y for y in best if y in gxs])}/{len(gxs)}"
              f"  пропущено {len(missing):>6}:  вне окна {100*cls['вне_окна']//s:>3}%  "
              f"есть тушь {100*cls['есть_тушь']//s:>3}%  нет туши {100*cls['нет_туши']//s:>3}%")

s = TOT["всего_пропущено"] or 1
print(f"\n{'='*70}\n=== ИТОГО пропущено {TOT['всего_пропущено']} строк GT ===")
print(f"   ВНЕ ОКНА  (окно обрезало): {100*TOT['вне_окна']//s:>3}%  ({TOT['вне_окна']})  -> расширить окно U0")
print(f"   ЕСТЬ ТУШЬ (трасса оборвалась на живой): {100*TOT['есть_тушь']//s:>3}%  ({TOT['есть_тушь']})  -> восстановимо")
print(f"   НЕТ ТУШИ  (пунктир/выцвет):  {100*TOT['нет_туши']//s:>3}%  ({TOT['нет_туши']})  -> ловушка, не добивать")
print("\nЧитать: 'есть тушь' высок -> покрытие восстановимо честно (расширить окно/поднять gap/")
print("добить ПО ТУШИ). 'нет туши' высок -> это добивка по пустому, метрический артефакт (§6.20.3).")

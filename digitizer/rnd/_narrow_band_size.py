"""СКОЛЬКО В АРХИВЕ «УЗКОПОЛОСНЫХ» КРИВЫХ — размер транша, на который есть смысл ставить.

Повод (замеры 19.07): на оракульной полосе честными (med<=3px И cov>=0.9) выходят ТОЛЬКО
узкополосные кривые (197/285/485px), все широкие (679-1545px) провалены; в latch-замере
чистые случаи — те же узкие (CALI1 1.5px, PS1 2.2px, SP1 1.3px), а худшие — широкие
(NNKM1 377px, NNKB1 410px, MCALI1 407px).

Меряем ПО ЭКСПЕРТНЫМ ТРАССАМ (без картинок, быстро): размах кривой по x = p98-p2, и какая доля
архива попадает в «узкую» зону. Плюс разрез по мнемоникам и по числу кривых на лист — чтобы
было видно, это отдельный класс планшетов или узкие кривые размазаны по всем.

  python _narrow_band_size.py [--narrow 500]
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from auto import meta as M

MN = r"F:\nds\Auto\mnemonics.json"
ARCHIVE = Path(r"F:\nds\projects\Archive")
ap = argparse.ArgumentParser()
ap.add_argument("--narrow", type=float, default=500.0, help="порог «узкой» полосы в px")
a = ap.parse_args()

spans, by_mnem, by_k, sheets_all_narrow, sheets = [], defaultdict(list), defaultdict(list), 0, 0
for wlg in sorted(ARCHIVE.glob("*/wlg")):
    for f in sorted(wlg.glob("*.nlgx")):
        if "_auto" in f.stem or not find_image(f):
            continue
        try:
            mo = extract(str(f))
        except Exception:
            continue
        cur = []
        for c in mo["curves"]:
            xs = [x for x in c["xs"] if x != NULL]
            if len(xs) < 100 or M.mnem_root(c["name"]) == "DA":
                continue
            q = np.percentile(xs, [2, 98])
            cur.append((M.curve_info(c["name"], MN)["root"], float(q[1] - q[0])))
        if not cur:
            continue
        sheets += 1
        for root, sp in cur:
            spans.append(sp); by_mnem[root].append(sp); by_k[len(cur)].append(sp)
        if all(sp <= a.narrow for _, sp in cur):
            sheets_all_narrow += 1

sp = np.array(spans)
print(f"планшетов {sheets}, кривых {len(sp)}")
print(f"\nРАЗМАХ КРИВОЙ ПО x (p98-p2): med {np.median(sp):.0f}px  p25 {np.percentile(sp,25):.0f}  "
      f"p75 {np.percentile(sp,75):.0f}  max {sp.max():.0f}")
print(f"\nДОЛЯ УЗКИХ (порог {a.narrow:.0f}px — там, где трассировщик уже даёт 1-2px):")
for t in (300, 400, 500, 600, 800):
    print(f"   <= {t:>4}px : {int((sp<=t).sum()):>5} кривых ({100*(sp<=t).mean():4.1f}%)")
print(f"\nЛИСТОВ, ГДЕ ВСЕ КРИВЫЕ УЗКИЕ (<= {a.narrow:.0f}px): {sheets_all_narrow} из {sheets} "
      f"({100*sheets_all_narrow/max(1,sheets):.0f}%) — это листы, которые можно брать ЦЕЛИКОМ")
print(f"\nПО ЧИСЛУ КРИВЫХ НА ЛИСТЕ (доля узких среди них):")
for k in sorted(by_k)[:8]:
    v = np.array(by_k[k])
    print(f"   K={k:<3} кривых {len(v):>5}  узких {100*(v<=a.narrow).mean():4.1f}%  med размах {np.median(v):.0f}px")
print(f"\nПО МНЕМОНИКАМ (топ-14 по числу кривых):")
for root, v in sorted(by_mnem.items(), key=lambda kv: -len(kv[1]))[:14]:
    v = np.array(v)
    print(f"   {root:<8} n={len(v):<5} med размах {np.median(v):>5.0f}px  узких {100*(v<=a.narrow).mean():4.1f}%")

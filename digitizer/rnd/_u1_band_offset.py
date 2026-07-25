r"""_u1_band_offset.py — МЕРЯЕМ ПОЛОСЫ U1 ЛИНЕЙКОЙ §6.29.

§6.29 дал числовое требование: полоса обязана быть центрирована на своей кривой точнее **100px**
(на 250px преимущество обученного селектора = 0), а ШИРИНУ можно брать с запасом (+400px терпимо).
Здесь тем же требованием мерится то, что U1 отдаёт НА САМОМ ДЕЛЕ — на тех же 5 листах стенда.

Для каждой экспертной кривой ищется ЛУЧШАЯ линия U1 (минимум |центр полосы − медиана GT|) и
считается:
  СМЕЩ   — |центр полосы U1 − медиана x кривой|, px  → попадает ли в допуск 100px;
  ВНУТРИ — доля точек эксперта, лежащих ВНУТРИ полосы (то, что вообще может быть оттрассировано);
  ШИР    — ширина полосы (для сверки с осью 1: большая ширина сама по себе не страшна).

Ответ на вопрос «чинится ли U1 доводкой»: если СМЕЩ у большинства кривых уже <100px, узкое место
не в позиционировании полосы, а в отборе линий/FLAG; если СМЕЩ сотни px — U1 не знает, где кривая,
и доводкой порогов это не лечится.

  python _u1_band_offset.py
"""
import sys, io, contextlib
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
from _multi_replica_probe import dense
from auto import frame as F, understand as U, confidence as C, meta as M
from auto.config import DEFAULT
from _relatch_bench import SH, ARCH, MN, BAND_PAD

TOL = 100          # допуск §6.29 на смещение центра полосы


def main():
    p = DEFAULT.cv
    rows = []
    print(f"{'скважина':<12}{'кривая':<8}{'линий U1':>9}{'СМЕЩ px':>9}{'ШИР px':>8}"
          f"{'ВНУТРИ%':>9}{'AUTO':>6}  вердикт")
    for rel in SH:
        n = ARCH / rel
        well = n.parent.parent.name
        img = find_image(n)
        rgb = cv2.cvtColor(cv2.imread(str(img), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m = M.parse_filename(n.name, MN)
            fr = F.frame_from_nlgx(str(n), m, p, rgb=rgb)
            if getattr(fr, "row_shift", None) is not None:
                rgb = F.apply_row_shift(rgb, fr.row_shift)
            sheet = U.understand(rgb, fr, m, p)
            C.classify(sheet)
        mo = extract(str(n))
        gts = [c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])]
        for g in gts:
            d = dense(g)
            if len(d) < 50:
                continue
            gx = np.array([d[y] for y in sorted(d)], float)
            gmed = float(np.median(gx))
            best = None
            for L in sheet.lines:
                lo = L.x_lo - BAND_PAD; hi = L.x_hi + BAND_PAD
                ctr = (lo + hi) / 2
                off = abs(ctr - gmed)
                if best is None or off < best[0]:
                    inside = float(((gx >= lo) & (gx <= hi)).mean())
                    best = (off, hi - lo, inside, L.confidence)
            if best is None:
                continue
            off, wid, inside, conf = best
            ok = "✓ в допуске" if off <= TOL else ("~ 100-250px" if off <= 250 else "✗ мимо")
            rows.append((off, wid, inside, conf == "AUTO"))
            print(f"{well:<12}{g['name'].split()[0]:<8}{len(sheet.lines):>9}{off:>9.0f}{wid:>8.0f}"
                  f"{100*inside:>9.1f}{('AUTO' if conf=='AUTO' else 'FLAG'):>6}  {ok}")
    a = np.array([r[0] for r in rows]); w = np.array([r[1] for r in rows])
    ins = np.array([r[2] for r in rows]); au = np.array([r[3] for r in rows])
    print(f"\nкривых: {len(rows)}")
    print(f"СМЕЩЕНИЕ центра полосы: med {np.median(a):.0f}px   "
          f"<=100px (допуск §6.29): {100*(a<=TOL).mean():.0f}%   "
          f"100-250px: {100*((a>TOL)&(a<=250)).mean():.0f}%   >250px: {100*(a>250).mean():.0f}%")
    print(f"ШИРИНА полосы U1: med {np.median(w):.0f}px (ось 1 §6.29: до +400px терпимо)")
    print(f"ТОЧЕК ЭКСПЕРТА ВНУТРИ полосы: med {100*np.median(ins):.0f}%")
    print(f"лучшая линия имеет AUTO: {100*au.mean():.0f}%  (остальные глушит FLAG-гейт, §6.15)")


if __name__ == "__main__":
    main()

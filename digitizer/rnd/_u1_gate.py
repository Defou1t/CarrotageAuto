r"""_u1_gate.py — ГЕЙТ U1 (§6.17): сколько линий нашли против того, сколько кривых в рамке.

Целевая метрика любой работы над U1 — НЕ px (это про трассировщик, его потолок замерен §6.20),
а два вопроса ПО КАЖДОМУ ЛИСТУ:
  1) счёт: n_lines vs n_curves → класс EXACT / ДРОБИТ / ТЕРЯЕТ / НОЛЬ;
  2) покривой: для каждой GT-кривой — есть ли линия U1, чья x-полоса накрывает медиану кривой
     И перекрывает её по строкам ≥30%. Это «ПОЛОСА НАЙДЕНА» — вход, без которого трассировщик
     слеп (§6.16: узкая полоса решает всё, но U1 обязан её найти).

Дёшево: только understand, без трассы и emit (~12 с/лист после ускорения 20.07).
Результат пишется в JSON покривой — A/B двух прогонов = diff двух JSON.

  python _u1_gate.py --set semeguniv          # 30 листов Semeguniv_001
  python _u1_gate.py --set multi              # 12 мульти-листов прежних гейтов
  python _u1_gate.py --set zero               # листы «0 линий» из §6.17/§6.21
  python _u1_gate.py --set all --tag baseline # всё + тег файла результата
"""
import sys, io, json, time, argparse, contextlib
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
from auto import frame as F, understand as U, confidence as C, meta as M
from auto.config import DEFAULT

MN = r"F:\nds\Auto\mnemonics.json"
ARCH = Path(r"F:\nds\projects\Archive")
SEM = Path(r"F:\nds\projects\Semeguniv_001\wlg")
OUT = Path(r"F:\nds\output\taskS\u1")

MULTI = [r"BOGAT_011\wlg\BOGAT_011_BKZ, DS_2800-3190_200_1984-03-02_D_1_B_1.nlgx",
         r"BOGAT_015\wlg\BOGAT_015_BKZ, DS_3300-3700_200_1990-01-08_D_1.nlgx",
         r"BOGAT_014\wlg\BOGAT_014_BKZ, DS_3690-3860_200_1990-11-16_D_1.nlgx",
         r"LEVEN_023\wlg\LEVEN_023_BKZ, DS_1010-1500_200_1996-12-02_D_1.nlgx",
         r"BEZLUD_051\wlg\BEZLUD_051_RK, AK, DS_2688-3142_500_1998-08-05_D_1.nlgx",
         r"YULIIV_055\wlg\YULIIV_055_MK, MBK, MDS_2084-3060_200_1998-07-18_D_1.nlgx"]
ZERO = [r"KREMEN_089\wlg\KREMEN_089_BKZ, SP, DS_0794-1410_200_1999-06-03_D_1.nlgx",
        r"RYBAL_168\wlg\1980.09.02_Rybal_168_BKZ1_(1280-2724)_GZ1_GZ2_GZ3_GZ4_PS.nlgx",
        r"RYBAL_168\wlg\1980.09.02_Rybal_168_BKZ2_(1280-2724)_GZ5_OGZ_KB.nlgx",
        r"RYBAL_017\wlg\1964.09.07_Rybal_017_KV_(0018-0975)_500.nlgx"]


def sheets_for(setname):
    out = []
    if setname in ("semeguniv", "all"):
        out += sorted(SEM.glob("*.nlgx"))
    if setname in ("multi", "all"):
        out += [ARCH / s for s in MULTI]
    if setname in ("zero", "all"):
        out += [ARCH / s for s in ZERO]
    return out


def gate_sheet(n):
    """Один лист: understand → счёт + покривой «полоса найдена»."""
    img = find_image(n)
    if not img:
        return {"sheet": n.stem, "err": "нет картинки"}
    rgb = cv2.cvtColor(cv2.imread(str(img), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    p = DEFAULT.cv
    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            m = M.parse_filename(n.name, MN)
            fr = F.frame_from_nlgx(str(n), m, p, rgb=rgb)
            if getattr(fr, "row_shift", None) is not None:
                rgb = F.apply_row_shift(rgb, fr.row_shift)
            sheet = U.understand(rgb, fr, m, p)
            C.classify(sheet)
    except Exception as e:
        return {"sheet": n.stem, "err": f"{type(e).__name__}: {e}"}
    mo = extract(str(n))
    gts = [c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])]
    curves = []
    for g in gts:
        pts = [(g["top_y"] + i, x) for i, x in enumerate(g["xs"]) if x != NULL]
        ys = [q[0] for q in pts]; xs = [q[1] for q in pts]
        gmed = float(np.median(xs)); gy0, gy1 = min(ys), max(ys)
        best = None
        for L in sheet.lines:
            if not (L.x_lo - 8 <= gmed <= L.x_hi + 8):
                continue
            ov = max(0, min(L.y1, gy1) - max(L.y0, gy0))
            r = ov / max(1, gy1 - gy0)
            if r >= 0.3 and (best is None or r > best[1]):
                best = (L, r)
        curves.append({"name": g["name"].split()[0], "x_med": round(gmed, 1),
                       "band": best is not None,
                       "row_cov": round(best[1], 2) if best else 0.0,
                       "conf": best[0].confidence if best else None})
    nl, nc = len(sheet.lines), len(gts)
    cls = ("НОЛЬ" if nl == 0 else "EXACT" if nl == nc else
           "ДРОБИТ" if nl > nc else "ТЕРЯЕТ")
    return {"sheet": n.stem, "n_lines": nl, "n_curves": nc, "class": cls,
            "sec": round(time.time() - t0, 1),
            "band_found": sum(1 for c in curves if c["band"]), "curves": curves}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    rows = []
    for n in sheets_for(a.set):
        r = gate_sheet(n)
        rows.append(r)
        if "err" in r:
            print(f"  !! {r['sheet'][:52]:<54} {r['err']}")
        else:
            print(f"  {r['sheet'][:52]:<54} {r['class']:<7} lines={r['n_lines']:>3} "
                  f"curves={r['n_curves']:>2}  полос найдено {r['band_found']}/{r['n_curves']}"
                  f"  ({r['sec']}s)")
    ok = [r for r in rows if "err" not in r]
    ncur = sum(r["n_curves"] for r in ok)
    nband = sum(r["band_found"] for r in ok)
    from collections import Counter
    cc = Counter(r["class"] for r in ok)
    print(f"\n=== ИТОГО {len(ok)} листов, {ncur} кривых ===")
    print("  классы:", dict(cc), f" EXACT {100.0*cc.get('EXACT',0)/max(1,len(ok)):.0f}%")
    print(f"  ★ ПОЛОСА НАЙДЕНА: {nband}/{ncur} = {100.0*nband/max(1,ncur):.1f}% кривых")
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"u1_{a.set}_{a.tag}.json"
    fp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  -> {fp}")

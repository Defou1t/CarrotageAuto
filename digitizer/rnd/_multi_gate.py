"""ШИРОКИЙ ГЕЙТ мульти-кривых планшетов (P0-2). Пайплайн in-process на N листах разных скважин
с мульти-токеном; ПОКРИВОЙ гейт vs экспертная трасса (cov/med/<=3px/<=10px) + проверка утечки
эталона (auto.xs == gt.xs). Имена слотов auto-nlgx совпадают с GT (emit пишет в слоты рамки).

  python multi_batch.py [N] [--tok "BK, IK|GK, NGK|..."]

Печать: строка на кривую + сводка по листам. Правило: НИКАКИХ выводов по одному листу.
"""
import sys, io, json, contextlib, argparse
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image
from auto.pipeline import run as pipe_run
from auto.config import Config
from auto import meta as M

MN = r"F:\nds\Auto\mnemonics.json"
ARCHIVE = Path(r"F:\nds\projects\Archive")
OUT = Path(r"F:\nds\output\p0_multicurve\gate")
OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", type=int, default=12)
ap.add_argument("--tok", default="BK, IK|GK, NGK|BKZ, DS|MK, MBK, MDS|MBK, MDS, MK|STK+DS|BK+IK")
ap.add_argument("--per-well", type=int, default=1)
ap.add_argument("--json", default="")
ap.add_argument("--prob", default="", help="папка prob-карт (_prob_batch.py) — ML НА ФОН")
a = ap.parse_args()
PROBDIR = Path(a.prob) if a.prob else None
TOKS = [t.strip().upper() for t in a.tok.split("|")]

cands, per_well = [], {}
for wlg in sorted(ARCHIVE.glob("*/wlg")):
    well = wlg.parent.name
    for n in sorted(wlg.glob("*.nlgx")):
        if "_auto" in n.stem or per_well.get(well, 0) >= a.per_well:
            continue
        m = M.parse_filename(n.name, MN)
        if m.curves_token.strip().upper() not in TOKS:
            continue
        img = find_image(n)
        if not img:
            continue
        try:
            mo = extract(str(n))
        except Exception:
            continue
        gts = [c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])]
        if len(gts) < 2:
            continue
        cands.append((well, n, img, m, gts))
        per_well[well] = per_well.get(well, 0) + 1

print(f"кандидатов: {len(cands)}; беру {min(a.n, len(cands))}")
rows, sheets = [], []
for well, n, img, m, gts in cands[:a.n]:
    cfg = Config(); cfg.out = OUT
    npy = None
    if PROBDIR is not None:
        cand = PROBDIR / f"{Path(img).stem}_prob.npy"
        npy = str(cand) if cand.is_file() else None
        if npy is None:
            print(f"  (нет prob-карты для {Path(img).stem[:40]} — по правилам)")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            sheet, traces, res = pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False,
                                          prob_npy=npy)
    except Exception as e:
        print(f"{well}/{m.curves_token:<14} ERR {type(e).__name__}: {e}")
        sheets.append({"well": well, "stem": n.stem, "err": f"{type(e).__name__}: {e}"})
        continue
    aup = Path(res.get("nlgx", ""))
    n_lines = len(sheet.lines)
    hdr = (f"{well}/{m.curves_token:<14} exp={len(m.expected_curves)} gt={len(gts)} "
           f"lines={n_lines} written={res.get('written')}")
    print(hdr)
    if not aup.is_file():
        sheets.append({"well": well, "stem": n.stem, "err": "нет nlgx"})
        continue
    A = extract(str(aup))
    abyname = {c["name"]: c for c in A["curves"]}
    srow = {"well": well, "stem": n.stem, "tok": m.curves_token, "n_gt": len(gts),
            "n_lines": n_lines, "curves": []}
    for g in gts:
        gxs = {g["top_y"] + i: x for i, x in enumerate(g["xs"]) if x != NULL}
        ac = abyname.get(g["name"])
        nm = g["name"].split()[0]
        if ac is None:
            print(f"    {nm:<8} — слот не записан"); srow["curves"].append({"name": nm, "state": "missing"}); continue
        if ac["xs"] == g["xs"]:
            print(f"    {nm:<8} !!! ЭТАЛОН НЕ ПЕРЕЗАПИСАН (leak)"); srow["curves"].append({"name": nm, "state": "leak"}); continue
        axs = {ac["top_y"] + i: x for i, x in enumerate(ac["xs"]) if x != NULL}
        common = [y for y in axs if y in gxs]
        if not common:
            print(f"    {nm:<8} cov=0"); srow["curves"].append({"name": nm, "state": "cov0"}); continue
        d = np.array([abs(axs[y] - gxs[y]) for y in common], float)
        cov = len(common) / max(1, len(gxs))
        r = {"name": nm, "state": "ok", "cov": round(cov, 3), "med": round(float(np.median(d)), 1),
             "p3": round(float((d <= 3).mean() * 100)), "p10": round(float((d <= 10).mean() * 100)),
             "p90": round(float(np.percentile(d, 90)), 1)}
        print(f"    {nm:<8} cov={cov:.2f} med={r['med']:6.1f} <=3px {r['p3']:3d}% <=10px {r['p10']:3d}% p90={r['p90']:7.1f}")
        srow["curves"].append(r); rows.append(r)
    sheets.append(srow)

print("\n=== СВОДКА ===")
ok = [r for r in rows if r["state"] == "ok"]
allc = sum(len(s.get("curves", [])) for s in sheets)
print(f"листов {len(sheets)}, кривых-слотов {allc}, измерено {len(ok)}, "
      f"missing {sum(1 for s in sheets for c in s.get('curves',[]) if c['state']=='missing')}, "
      f"leak {sum(1 for s in sheets for c in s.get('curves',[]) if c['state']=='leak')}, "
      f"cov0 {sum(1 for s in sheets for c in s.get('curves',[]) if c['state']=='cov0')}")
if ok:
    med = np.median([r["med"] for r in ok])
    print(f"по измеренным: med(med)={med:.1f}px  med(cov)={np.median([r['cov'] for r in ok]):.2f}  "
          f"<=3px avg {np.mean([r['p3'] for r in ok]):.0f}%  <=10px avg {np.mean([r['p10'] for r in ok]):.0f}%  "
          f"кривых med<=3px: {sum(1 for r in ok if r['med']<=3)}/{len(ok)}")
if a.json:
    Path(a.json).write_text(json.dumps(sheets, ensure_ascii=False, indent=1), encoding="utf-8")
    print("json →", a.json)

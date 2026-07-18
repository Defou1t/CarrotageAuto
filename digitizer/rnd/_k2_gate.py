"""ТРАНШ K=2 (направление «а»): гейт пайплайна на листах, где в файле РОВНО ДВЕ оцифрованные
кривые — 246 листов, 21% архива (§6.7). Это следующий по дешевизне объём после одиночных.

Отбор идёт по СОДЕРЖИМОМУ рамки (K=2), а не по токену имени: «мульти» в имени ничего не
обещает (§6.5). Берём по одному листу со скважины и предпочитаем МЕЛКИЕ сканы — пайплайн на
47000-строчных идёт минутами, а вывод от размера не зависит.

Печатает, ГДЕ рвётся: сколько линий нашёл U1, сколько AUTO, что записал emit, и px vs эксперт.

  python _k2_gate.py [N] [--per-well 1] [--prob DIR] [--json out.json]
"""
import sys, io, json, contextlib, argparse
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from PIL import Image
from extract_nlgx import extract, NULL
from dataset_build import find_image
from _multi_replica_probe import dense
from auto.pipeline import run as pipe_run
from auto.config import Config
from auto import meta as M

MN = r"F:\nds\Auto\mnemonics.json"
ARCHIVE = Path(r"F:\nds\projects\Archive")
OUT = Path(r"F:\nds\output\p0_multicurve\k2")
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", type=int, default=10)
ap.add_argument("--per-well", type=int, default=1)
ap.add_argument("--prob", default="")
ap.add_argument("--json", default="")
ap.add_argument("--max-mpx", type=float, default=45.0)
# ⚠ Нижняя граница ОБЯЗАТЕЛЬНА: сортировка «сначала мелкие» вытащила одну нетипичную скважину
# (RYBAL, сканы 2-7 Мпикс) и проба перестала быть широкой. Правило гейта — разные скважины И
# типичные листы, а не просто «что быстрее считается».
ap.add_argument("--min-mpx", type=float, default=8.0)
a = ap.parse_args()
PROBDIR = Path(a.prob) if a.prob else None

cands, per_well = [], {}
for wlg in sorted(ARCHIVE.glob("*/wlg")):
    well = wlg.parent.name
    for n in sorted(wlg.glob("*.nlgx")):
        if "_auto" in n.stem or per_well.get(well, 0) >= a.per_well:
            continue
        img = find_image(n)
        if not img:
            continue
        try:
            mo = extract(str(n))
        except Exception:
            continue
        gts = [c for c in mo["curves"] if sum(1 for x in c["xs"] if x != NULL) >= 100
               and M.mnem_root(c["name"]) != "DA"]
        if len(gts) != 2:
            continue
        w, h = Image.open(img).size
        if not (a.min_mpx * 1e6 <= w * h <= a.max_mpx * 1e6):
            continue
        cands.append((w * h, well, n, img, gts))
        per_well[well] = per_well.get(well, 0) + 1
cands.sort()
cands = cands[:a.n]
print(f"K=2 листов в пробе: {len(cands)} (по одному со скважины, до {a.max_mpx:.0f} Мпикс)")

rows = []
for px, well, n, img, gts in cands:
    m = M.parse_filename(n.name, MN)
    cfg = Config(); cfg.out = OUT
    npy = None
    if PROBDIR is not None:
        q = PROBDIR / f"{Path(img).stem}_prob.npy"
        npy = str(q) if q.is_file() else None
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            sheet, traces, res = pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False,
                                          prob_npy=npy)
    except Exception as e:
        print(f"{well:<14} {m.curves_token:<12} ERR {type(e).__name__}: {e}")
        continue
    conf = sheet.diag.get("confidence", {})
    names = ",".join(c["name"].split()[0] for c in gts)
    print(f"{well:<14} {m.curves_token:<12} {px/1e6:4.0f}Мпкс линий={len(sheet.lines):>3} "
          f"AUTO={conf.get('auto',0):>2} трасс={len(traces):>2} записано={res.get('written')} [{names}]")
    r = {"well": well, "tok": m.curves_token, "lines": len(sheet.lines),
         "auto": conf.get("auto", 0), "traces": len(traces),
         "written": res.get("written"), "curves": []}
    aup = Path(res.get("nlgx", ""))
    if aup.is_file():
        A = extract(str(aup))
        aby = {c["name"]: c for c in A["curves"]}
        for g in gts:
            nm = g["name"].split()[0]
            ac = aby.get(g["name"])
            if ac is None:
                r["curves"].append({"name": nm, "state": "missing"}); continue
            if ac["xs"] == g["xs"]:
                print(f"    {nm:<8} !!! ЭТАЛОН НЕ ПЕРЕЗАПИСАН (leak)")
                r["curves"].append({"name": nm, "state": "leak"}); continue
            d = dense(g)
            axs = {ac["top_y"] + i: x for i, x in enumerate(ac["xs"]) if x != NULL}
            common = [y for y in axs if y in d]
            if not common:
                r["curves"].append({"name": nm, "state": "cov0"}); continue
            e = np.array([abs(axs[y] - d[y]) for y in common], float)
            rec = {"name": nm, "state": "ok", "cov": round(len(common) / max(1, len(d)), 2),
                   "med": round(float(np.median(e)), 1),
                   "p3": int((e <= 3).mean() * 100), "p10": int((e <= 10).mean() * 100)}
            print(f"    {nm:<8} cov={rec['cov']:.2f} med={rec['med']:7.1f} ≤3px {rec['p3']:3d}% ≤10px {rec['p10']:3d}%")
            r["curves"].append(rec)
    rows.append(r)

ok = [c for r in rows for c in r["curves"] if c["state"] == "ok"]
allc = [c for r in rows for c in r["curves"]]
print("\n=== СВОДКА K=2 ===")
print(f"листов {len(rows)}, слотов {len(allc)}: измерено {len(ok)}, "
      f"leak {sum(1 for c in allc if c['state']=='leak')}, "
      f"missing {sum(1 for c in allc if c['state']=='missing')}")
if ok:
    med = np.array([c["med"] for c in ok])
    print(f"med(med)={np.median(med):.1f}px  кривых med≤3px: {(med<=3).sum()}/{len(med)}  "
          f"≤3px сред {np.mean([c['p3'] for c in ok]):.0f}%  покрытие сред {np.mean([c['cov'] for c in ok]):.2f}")
if a.json:
    Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("json →", a.json)

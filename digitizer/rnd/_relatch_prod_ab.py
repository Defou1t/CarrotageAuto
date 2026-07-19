r"""_relatch_prod_ab.py — ПЕРЕНОСИТСЯ ЛИ выигрыш «ширины» из оракульной полосы В ПРОД.

На оракульной полосе (_relatch_verify.py, V4) правка даёт med(med) 68.3 -> 12.9px, честных 3 -> 4,
«своя» 34.9 -> 41.4%. Но §6.16 durable: ОРАКУЛЬНЫЙ УСПЕХ В ПРОД НЕ ПЕРЕНОСИТСЯ — там полосу
обязан найти сам U1, и на трансе «узкополосных» это уже один раз всё обнулило (0 честных из 8).
Поэтому правка не имеет права идти в auto/ до этого замера.

ЧТО ПРОВЕРЯЕТСЯ (V4, две правки, обе внутри trace_line):
  1. ветка else («нет рана под предсказанием»): вместо «ближайший ЦЕНТР рана» берём ран с
     максимальной min(ширина, 20), зазор — тайбрейк. Обоснование замером: в момент ухода со
     своей кривой свой ран ШИРЕ (med 10px), а ложный кандидат — ТОНКИЙ обрывок (med 2px);
  2. точка на ШИРОКОМ ране (>=wide_run): clip(pred, a, b) вместо «дальнего края от базлайна».
     Ср. §2 prefer_body: эксперт ведёт по ТЕЛУ штриха (rel=0.50), вершинных лишь 13%.

Метод — ПРЯМОЙ A/B монкипатчем, как в §6.15: пакет auto/ НЕ трогаем, пока замер не разрешит.
Проверка утечки эталона (auto.xs == gt.xs) обязательна: без неё «идеальные 0.0px» на
незаписанном слоте (ловушка §4) читаются как успех.

  python _relatch_prod_ab.py [--n 5] [--files a.nlgx b.nlgx ...]
"""
import sys, io, json, argparse, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
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
from auto import imaging as im
from auto import trace2d as T

MN = r"F:\nds\Auto\mnemonics.json"
ARCH = Path(r"F:\nds\projects\Archive")
OUT = Path(r"F:\nds\output\taskS\prod_ab")

# те же 5 листов, на которых мерилась оракульная полоса — иначе перенос не с чем сравнивать
SH = [r"BOGAT_011\wlg\BOGAT_011_BKZ, DS_2800-3190_200_1984-03-02_D_1_B_1.nlgx",
      r"BOGAT_015\wlg\BOGAT_015_BKZ, DS_3300-3700_200_1990-01-08_D_1.nlgx",
      r"LEVEN_023\wlg\LEVEN_023_BKZ, DS_1010-1500_200_1996-12-02_D_1.nlgx",
      r"BEZLUD_051\wlg\BEZLUD_051_RK, AK, DS_2688-3142_500_1998-08-05_D_1.nlgx",
      r"YULIIV_055\wlg\YULIIV_055_MK, MBK, MDS_2084-3060_200_1998-07-18_D_1.nlgx"]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--files", nargs="*", default=None)
a = ap.parse_args()

ORIG = T.trace_line


def patched(fg, line, frame, p, band_pad=8, slmax=30.0, wide_run=14, x_range=None,
            jump_limit=None, wide_min=20.0):
    """trace_line с правкой V4. Тело скопировано с оригинала; отличия помечены ★."""
    H, W = fg.shape
    if x_range is not None:
        lo = max(0, int(x_range[0])); hi = min(W, int(x_range[1]) + 1)
    else:
        lo = max(0, int(line.x_lo) - band_pad); hi = min(W, int(line.x_hi) + band_pad + 1)
    base = line.x_center
    x = None; v = 0.0
    tr = {}
    for y in range(max(0, line.y0), min(H, line.y1 + 1)):
        runs = im.row_runs(fg[y, lo:hi])
        runs = [(q + lo, w + lo, e + lo) for q, w, e in runs]
        if not runs:
            if x is not None:
                x = x + float(np.clip(v, -slmax, slmax))
            continue
        if x is None:
            q, w, e = min(runs, key=lambda r: abs(r[2] - base))
            x = e; v = 0.0; tr[y] = x; continue
        pred = x + float(np.clip(v, -slmax, slmax))
        cont = [r for r in runs if r[0] - 2 <= pred <= r[1] + 2]
        if cont:
            q, w, e = min(cont, key=lambda r: abs(r[2] - pred))
        else:
            # ★ ПРАВКА 1: не «ближайший центр», а САМЫЙ ШИРОКИЙ (с потолком), зазор — тайбрейк.
            def _key(r):
                gap = max(r[0] - pred, pred - r[1], 0.0)
                return (-min(r[1] - r[0], wide_min), gap)
            q, w, e = min(runs, key=_key)
        if (w - q) >= wide_run:
            # ★ ПРАВКА 2: ближайшая к предсказанию точка ВНУТРИ рана вместо дальнего края.
            nx = min(max(pred, q), w)
        else:
            nx = e
        v = 0.6 * v + 0.4 * (nx - x); x = nx; tr[y] = float(nx)
    T._extend_ends(tr, fg, lo, hi, slmax)
    return tr


def measure(tag, sheets):
    """Прогон пайплайна на листах и ПОКРИВОЙ гейт против экспертной трассы."""
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    rows, per_sheet = [], []
    for well, n, img, m, gts in sheets:
        cfg = Config(); cfg.out = out
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                sheet, traces, res = pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
        except Exception as e:
            print(f"  {well:<12} ERR {type(e).__name__}: {e}")
            per_sheet.append({"well": well, "err": str(e)}); continue
        aup = Path(res.get("nlgx", ""))
        rec = {"well": well, "n_lines": len(sheet.lines), "n_gt": len(gts),
               "written": res.get("written"), "curves": []}
        if not aup.is_file():
            rec["err"] = "нет nlgx"; per_sheet.append(rec); continue
        A = extract(str(aup))
        byname = {c["name"]: c for c in A["curves"]}
        for g in gts:
            gxs = {g["top_y"] + i: xx for i, xx in enumerate(g["xs"]) if xx != NULL}
            nm = g["name"].split()[0]
            ac = byname.get(g["name"])
            if ac is None:
                rec["curves"].append({"name": nm, "state": "missing"}); continue
            if ac["xs"] == g["xs"]:          # ЛОВУШКА §4: эталон не перезаписан
                rec["curves"].append({"name": nm, "state": "leak"}); continue
            axs = {ac["top_y"] + i: xx for i, xx in enumerate(ac["xs"]) if xx != NULL}
            common = [yy for yy in axs if yy in gxs]
            if not common:
                rec["curves"].append({"name": nm, "state": "cov0"}); continue
            d = np.array([abs(axs[yy] - gxs[yy]) for yy in common], float)
            r = {"name": nm, "state": "ok", "cov": round(len(common) / max(1, len(gxs)), 3),
                 "med": round(float(np.median(d)), 1), "p3": round(float((d <= 3).mean() * 100))}
            rec["curves"].append(r); rows.append(r)
        per_sheet.append(rec)
        st = " ".join(f"{c['name']}:{c.get('med','-')}" for c in rec["curves"])
        print(f"  {well:<12} lines={rec['n_lines']:<4} written={rec['written']}  {st}")
    ok = [r for r in rows if r["state"] == "ok"]
    honest = [r for r in ok if r["med"] <= 3 and r["cov"] >= 0.9]
    allc = sum(len(s.get("curves", [])) for s in per_sheet)
    miss = sum(1 for s in per_sheet for c in s.get("curves", []) if c["state"] == "missing")
    leak = sum(1 for s in per_sheet for c in s.get("curves", []) if c["state"] == "leak")
    summ = {"tag": tag, "slots": allc, "measured": len(ok), "missing": miss, "leak": leak,
            "med_med": float(np.median([r["med"] for r in ok])) if ok else None,
            "med_cov": float(np.median([r["cov"] for r in ok])) if ok else None,
            "honest": len(honest), "sheets": per_sheet}
    print(f"  --- {tag}: слотов {allc}, измерено {len(ok)}, не записано {miss}, leak {leak}")
    if ok:
        print(f"      med(med) {summ['med_med']:.1f}px  med(cov) {summ['med_cov']:.2f}  "
              f"★ЧЕСТНЫХ (med<=3 И cov>=0.9) {len(honest)}/{len(ok)}")
    return summ


files = [Path(f) for f in a.files] if a.files else [ARCH / s for s in SH][:a.n]
sheets = []
for n in files:
    img = find_image(n)
    if not img:
        print(f"нет картинки: {n.name}"); continue
    m = M.parse_filename(n.name, MN)
    mo = extract(str(n))
    gts = [c for c in ds.real_curves(mo) if any(xx != NULL for xx in c["xs"])]
    sheets.append((n.parent.parent.name, n, img, m, gts))
print(f"листов: {len(sheets)}\n")

print("=== A) БАЗА (прод как есть) ===")
A_ = measure("A_base", sheets)
print("\n=== B) V4: ширина в else + clip на широком ране ===")
T.trace_line = patched
B_ = measure("B_v4", sheets)
T.trace_line = ORIG

print("\n" + "=" * 78)
print(f"{'':<22}{'слотов':>8}{'измер':>7}{'не зап':>8}{'leak':>6}{'med(med)':>10}{'med(cov)':>10}{'ЧЕСТНЫХ':>9}")
for s in (A_, B_):
    print(f"{s['tag']:<22}{s['slots']:>8}{s['measured']:>7}{s['missing']:>8}{s['leak']:>6}"
          f"{(s['med_med'] if s['med_med'] is not None else float('nan')):>10.1f}"
          f"{(s['med_cov'] if s['med_cov'] is not None else float('nan')):>10.2f}"
          f"{s['honest']:>9}")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "prod_ab.json").write_text(json.dumps({"A": A_, "B": B_}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
print("\nЧитать так: если ЧЕСТНЫХ и «не записано» в проде не сдвинулись — оракульный выигрыш")
print("НЕ ПЕРЕНОСИТСЯ (как в §6.16), и узкое место выше по конвейеру: U1 не находит полосу.")

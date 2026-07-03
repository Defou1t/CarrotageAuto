r"""ГЕЙТ СЕТКИ по эталонам (новый протокол 03.07): наш детект жирных горизонталей
(fit_bold_chain, свободная фаза от планшета) против ЭКСПЕРТНОЙ сетки из свежих nlgx
(Yatskivska_001 / Pn_Zavoda_001 в Archive). Оцифровка «с нуля»: из эталона берём только
рамку (ось) и зону кривых (для исключения из профилей) — сами линии не подсматриваем.

  python grid_gate.py [--wells Yatskivska_001,Pn_Zavoda_001] [--limit N]
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_nlgx import extract, NULL
from detect_calibration import fit_bold_ladder
from set_depth_grid import predict_lines
from dataset_build import find_image
import dataset as ds
Image.MAX_IMAGE_PIXELS = None
ARCHIVE = Path(r"F:\nds\projects\Archive")


def gate_one(nlgx):
    m = extract(str(nlgx))
    dg = m.get("depth_grid") or {}
    gt_ys = dg.get("ys") or []
    da = m.get("depth_axis")
    if len(gt_ys) < 12 or not da or not da.get("bottom_y"):
        return None
    img = find_image(nlgx)
    if not img:
        return None
    pl = predict_lines(m, step=4.0)
    if not pl:
        return None
    _, _, _, step_px = pl
    if not (30 <= step_px <= 400):                      # не 1:200-подобный масштаб — шаг из GT
        d = np.diff(sorted(gt_ys))
        step_px = float(np.median(d[d > 10]))
    g = np.asarray(Image.open(img).convert("L"))
    xs_tr = [x for c in ds.real_curves(m) for x in c["xs"] if x != NULL]
    exc = (max(0, int(np.percentile(xs_tr, 1)) - 30), int(np.percentile(xs_tr, 99)) + 60) if xs_tr else None
    y0, y1 = min(da["top_y"], da["bottom_y"]), max(da["top_y"], da["bottom_y"])
    chain = fit_bold_ladder(g, step_px, y0, y1, exclude_x=exc)
    if not chain:
        return {"stem": nlgx.stem[:44], "n_gt": len(gt_ys), "n_ours": 0}
    ax_k = ((da.get("x_bot") or 0) - (da.get("x_top") or 0)) / max(y1 - y0, 1)
    ours = []
    for y0l, sl, ok in chain:
        xa = (da.get("x_top") or 0) + (y0l - da["top_y"]) * ax_k
        ours.append(y0l + sl * xa)
    ge = np.array(sorted(gt_ys), float)
    go = np.array(sorted(ours), float)
    dmin = np.abs(go[:, None] - ge[None, :]).min(1)
    return {"stem": nlgx.stem[:44], "n_gt": len(ge), "n_ours": len(go),
            "med": round(float(np.median(dmin)), 1),
            "le2": round(float(np.mean(dmin <= 2)), 2), "le3": round(float(np.mean(dmin <= 3)), 2),
            "le5": round(float(np.mean(dmin <= 5)), 2), "max": round(float(dmin.max()), 0),
            "step_px": round(step_px, 1)}


def main():
    a = sys.argv[1:]
    wells = (a[a.index("--wells") + 1].split(",") if "--wells" in a
             else ["Yatskivska_001", "Pn_Zavoda_001"])
    limit = int(a[a.index("--limit") + 1]) if "--limit" in a else None
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    allr = []
    for well in wells:
        files = sorted((ARCHIVE / well / "wlg").glob("*.nlgx"))
        if limit:
            files = files[:limit]
        print(f"=== {well}: {len(files)} nlgx")
        for f in files:
            if "_auto" in f.stem:
                continue
            try:
                r = gate_one(f)
            except Exception as e:
                print(f"  {f.stem[:44]:<44} ОШИБКА: {str(e)[:60]}")
                continue
            if not r:
                continue
            allr.append(r)
            if r.get("med") is not None:
                print(f"  {r['stem']:<44} n {r['n_ours']:>3}/{r['n_gt']:<3} med={r['med']:>5}px "
                      f"≤3px {int(r['le3']*100):>3}% ≤5px {int(r['le5']*100):>3}% max={int(r['max'])}")
            else:
                print(f"  {r['stem']:<44} ДЕТЕКТ ПУСТ (gt {r['n_gt']})")
    good = [r for r in allr if r.get("med") is not None]
    if good:
        print(f"\nИТОГ: {len(good)} планшетов | med(med)={np.median([r['med'] for r in good]):.1f}px | "
              f"среднее ≤3px = {100*np.mean([r['le3'] for r in good]):.0f}% | "
              f"≤5px = {100*np.mean([r['le5'] for r in good]):.0f}% | "
              f"планшетов с med≤3px: {sum(1 for r in good if r['med'] <= 3)}/{len(good)}")


if __name__ == "__main__":
    main()

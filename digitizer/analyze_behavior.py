r"""
analyze_behavior.py — v2: различимы ли НАЛОЖЕННЫЕ кривые по ПОВЕДЕНИЮ (идея пользователя:
у линии есть характер — гладкая vs спайковая). Считаем из nlgx-трасс на кривую:
  hf  — высокочастотная энергия = std(x - сглаженное)  → спайковость
  tv  — total variation на строку = mean|dx|            → «дёрганость»
  d2  — mean|2-я разность|                               → кривизна/пики
Если кривые в файле различаются по hf/tv (разброс/группы) — поведение можно
использовать в трекере для развязки пересечений (спайковая → на спайковую ветку).

python analyze_behavior.py <nlgx> [<nlgx> ...]
"""
import sys
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds


def features(curve):
    pts = [(i, x) for i, x in enumerate(curve["xs"]) if x != NULL]
    if len(pts) < 30:
        return None
    rows = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts], float)
    grid = np.arange(rows[0], rows[-1] + 1)
    xi = np.interp(grid, rows, xs)
    w = 21; sm = np.convolve(xi, np.ones(w)/w, mode="same")
    hf = float(np.std(xi - sm))
    tv = float(np.sum(np.abs(np.diff(xi))) / len(xi))
    d2 = float(np.mean(np.abs(np.diff(xi, 2))))
    return {"hf": hf, "tv": tv, "d2": d2, "n": len(xi)}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    for nlgx in sys.argv[1:]:
        m = extract(nlgx)
        curves = ds.real_curves(m)
        print(f"\n=== {Path(nlgx).stem[:50]} ({len(curves)} кривых) ===")
        rows = []
        for c in curves:
            f = features(c)
            if f:
                rows.append((c["name"].split()[0], f))
                print(f"  {c['name'].split()[0]:<9} hf={f['hf']:6.1f}  tv={f['tv']:5.2f}  d2={f['d2']:5.2f}")
        if len(rows) >= 2:
            hfs = np.array([r[1]["hf"] for r in rows])
            tvs = np.array([r[1]["tv"] for r in rows])
            # разброс поведения: отношение max/min hf и tv
            hf_ratio = hfs.max()/max(hfs.min(), 0.1)
            tv_ratio = tvs.max()/max(tvs.min(), 0.01)
            verdict = ("РАЗЛИЧИМЫ по поведению" if (hf_ratio > 2 or tv_ratio > 2)
                       else "слабо различимы")
            order = [rows[i][0] for i in np.argsort(hfs)]
            print(f"  -> hf разброс ×{hf_ratio:.1f}, tv ×{tv_ratio:.1f}  ⇒ {verdict}")
            print(f"     по возрастанию спайковости (hf): {' < '.join(order)}")


if __name__ == "__main__":
    main()

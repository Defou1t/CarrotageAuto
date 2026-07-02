r"""Диагностика «ёмкость vs неоднозначность» (план анализа 02.07): на строках-ошибках смотрим,
есть ли в prob-карте пик у ИСТИННОЙ позиции (GT). Вердикт бинарный:
  пики ЕСТЬ → модель «знает, где тушь», проваливается экстракция/назначение → лечится
              инференсом/постом (constrained assignment) без переучивания;
  пиков НЕТ → модель слепа в этих местах → лечится только обучением (таргеты/данные).

Строки-ошибки двух видов (по декомпозиции eval_mk):
  collapse (|pm−pp| мал при большом GT-разделении) — проверяем пик у ПОКИНУТОЙ кривой;
  swap-выброс (точка канала мимо своей GT >12px)  — проверяем пик своего канала у GT.

ВНИМАНИЕ: гоняет GPU-инференс (venv) с --save-prob — не запускать во время обучения.
  python _mk_secondpeak.py [--ckpt <mk_sep.pt>] [--skip-infer]
"""
import subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from dataset_build import find_image
from extract_nlgx import extract
import dataset as ds
from eval_mk import gt_traces

AR = Path(r"F:\nds\projects\Archive"); RND = r"F:\nds\Auto\digitizer\rnd"
VENV = r"D:\ComfyUI\StabilityMatrix\Data\Packages\ComfyUI\venv\Scripts\python.exe"
OUT = Path(r"F:\nds\output\mk_data")
WELLS = ["BEZLUD_051", "BOGAT_011", "LEVEN_023"]


def local_peak(probc, y, x, win=8):
    """Максимум канала в окне ±win вокруг x (есть ли там «знание» модели)."""
    W = probc.shape[1]
    lo, hi = max(0, int(x) - win), min(W, int(x) + win + 1)
    return float(probc[y, lo:hi].max()) if hi > lo else 0.0


def analyze(prob, pm, pp, gm, gp):
    rows = sorted(set(gm) & set(gp) & set(pm) & set(pp))
    comb = np.maximum(prob[0], prob[1])
    col_pk, sw_own, sw_comb = [], [], []
    n_col = n_sw = 0
    for y in rows:
        a, b, g1, g2 = pm[y], pp[y], gm[y], gp[y]
        gsep = abs(g1 - g2)
        straight = abs(a - g1) + abs(b - g2); swapped = abs(a - g2) + abs(b - g1)
        sw = swapped < straight
        if abs(a - b) < max(3.0, 0.3 * gsep) and gsep > 10:      # COLLAPSE: пик у покинутой кривой?
            n_col += 1
            xc = (a + b) / 2
            far = g1 if abs(g1 - xc) > abs(g2 - xc) else g2
            col_pk.append(local_peak(comb, y, far))
            continue
        if not sw or gsep <= 3:
            continue
        n_sw += 1                                                # SWAP-ВЫБРОС: пик своего канала у GT?
        t1, t2 = (g2, g1) if sw else (g1, g2)                    # цели каналов в СВОПнутом назначении
        for ch, (px, tgt) in enumerate(((a, t1), (b, t2))):
            if abs(px - tgt) > 12:                               # этот канал промахнулся мимо цели
                sw_own.append(local_peak(prob[ch], y, tgt))
                sw_comb.append(local_peak(comb, y, tgt))
    def frac(v, t):
        return round(float(np.mean(np.array(v) >= t)), 2) if v else None
    return {"n_rows": len(rows), "n_collapse": n_col, "n_swap": n_sw,
            "collapse_peak>=0.15": frac(col_pk, 0.15), "collapse_peak>=0.30": frac(col_pk, 0.30),
            "swap_own_peak>=0.15": frac(sw_own, 0.15), "swap_own_peak>=0.30": frac(sw_own, 0.30),
            "swap_comb_peak>=0.15": frac(sw_comb, 0.15), "swap_comb_peak>=0.30": frac(sw_comb, 0.30),
            "col_med_peak": round(float(np.median(col_pk)), 3) if col_pk else None,
            "sw_med_own_peak": round(float(np.median(sw_own)), 3) if sw_own else None}


def main():
    a = sys.argv[1:]
    ckpt = str(OUT / "mk_sep.pt")
    if "--ckpt" in a:
        i = a.index("--ckpt"); ckpt = a[i + 1]; del a[i:i + 2]
    skip = "--skip-infer" in a
    extra = [x for x in a if x != "--skip-infer"]                # проброс (--ens список, --tta) в infer
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    for well in WELLS:
        nlgx = None
        for f in sorted((AR / well / "wlg").glob("*.nlgx")):
            if "_auto" in f.stem:
                continue
            try:
                m = extract(str(f))
            except Exception:
                continue
            if {"MGZ", "MPZ"} <= {ds.mnemonic(c["name"]).upper() for c in ds.real_curves(m)}:
                nlgx = f; break
        if not nlgx:
            continue
        scan = find_image(nlgx); stem = Path(scan).stem[:40]
        if not skip:                                             # прогон с prob-картой (перезаписывает mk_prob2.npy)
            r = subprocess.run([VENV, f"{RND}/infer_mk.py", ckpt, str(scan), "--nlgx", str(nlgx),
                                "--out", str(OUT), "--save-prob"] + extra, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[{well}] infer FAIL: {r.stderr[-200:]}"); continue
        prob = np.load(OUT / "mk_prob2.npy").astype(np.float32)
        d = np.load(OUT / f"{stem}_traces.npz")
        pm = {int(y): float(x) for y, x in zip(d["mgz_y"], d["mgz_x"])}
        pp = {int(y): float(x) for y, x in zip(d["mpz_y"], d["mpz_x"])}
        gm, gp = gt_traces(str(nlgx))
        print(f"[{well}] {analyze(prob, pm, pp, gm, gp)}")


if __name__ == "__main__":
    main()

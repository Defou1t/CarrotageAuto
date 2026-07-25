r"""_decoder_reg_data.py — ВЫБОРКА ДЛЯ ПОЛНОЙ РЕГРЕССИИ x(строка) (вторая, фоновая ветка §6.24).

Отличие от `_decoder_seq_data.py`: там модель ВЫБИРАЕТ ран (структура трассировщика сохранена),
здесь модель ПРЕДСКАЗЫВАЕТ координату — раны как сущность исчезают. Ставка рискованнее (ровно
тот путь, где U-Net-разделитель «блобил», §6.3/§6.8), поэтому идёт фоном и гейтится ТЕМ ЖЕ
стендом bench: пока честных не больше 3/24, ветка не претендует ни на что.

Здесь только СЫРЬЁ: для каждой кривой сохраняются растр её полосы (упакованные биты) и плотная
экспертная трасса. Нарезка на чанки — в обучении (`_decoder_reg.py`), чтобы менять окно без
переизвлечения (переизвлечение = минуты на лист).

  python _decoder_reg_data.py --sheets 25
"""
import sys, io, argparse, contextlib
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
from auto import trace2d as T
from auto.config import DEFAULT
from _decoder_data import train_sheets, HELD

OUT = Path(r"F:\nds\output\taskS\decoder\reg")


def extract_sheet(n, pad, p):
    rgb = cv2.cvtColor(cv2.imread(str(find_image(n)), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    mo = extract(str(n))
    gts = [c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])]
    if len(gts) < 2:
        return None
    colors = ["black", "red", "orange", "green", "blue"]
    fgs = {c: T._color_fg(rgb, c, p) for c in colors}
    out = []
    for g in gts:
        d = dense(g)
        if len(d) < 200:
            continue
        rows = np.array(sorted(d)); xsr = np.array([d[y] for y in rows], np.float32)
        lo = max(0, int(xsr.min()) - pad); hi = min(W, int(xsr.max()) + pad + 1)
        best_c, best_hit = "black", -1
        for c in colors:
            fg = fgs[c]
            hit = sum(1 for k in range(0, len(rows), 20)
                      if fg[rows[k], max(0, int(xsr[k]) - 3):int(xsr[k]) + 4].any())
            if hit > best_hit:
                best_hit, best_c = hit, c
        y0, y1 = int(rows.min()), int(rows.max())
        band = fgs[best_c][y0:y1 + 1, lo:hi] > 0
        allr = np.arange(y0, y1 + 1)
        allx = np.interp(allr, rows, xsr).astype(np.float32)
        out.append({"name": g["name"].split()[0], "lo": lo, "hi": hi, "y0": y0, "y1": y1,
                    # ПОСТРОЧНАЯ упаковка (axis=1): при обучении распаковывается ТОЛЬКО окно,
                    # иначе 25 листов держат в RAM ~10 ГБ развёрнутых масок.
                    "shape": band.shape, "band": np.packbits(band, axis=1), "gt": allx,
                    "color": best_c})
    return out or None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", type=int, default=25)
    ap.add_argument("--pad", type=int, default=28)
    a = ap.parse_args()
    p = DEFAULT.cv
    OUT.mkdir(parents=True, exist_ok=True)
    sheets = train_sheets(a.sheets)
    print(f"листов: {len(sheets)} (held-out {sorted(HELD)} исключены)")
    tot = 0
    for n in sheets:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                r = extract_sheet(n, a.pad, p)
        except Exception as e:
            print(f"  !! {n.name[:44]:<46} {type(e).__name__}: {e}"); continue
        if r is None:
            print(f"  -- {n.name[:44]:<46} нет данных"); continue
        f = OUT / f"{n.stem[:60]}.npz"
        np.savez(f, n=len(r), **{f"{k}_{i}": np.asarray(c[k]) for i, c in enumerate(r)
                                 for k in ("band", "gt", "shape")},
                 meta=np.array([[c["lo"], c["y0"], c["y1"]] for c in r]),
                 names=np.array([c["name"] for c in r]),
                 well=np.array(n.parent.parent.name))
        tot += len(r)
        print(f"  {n.name[:44]:<46} кривых {len(r)}, {f.stat().st_size/1e6:.1f} МБ")
    print(f"\nИТОГО кривых: {tot} -> {OUT}")

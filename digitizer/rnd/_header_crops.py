r"""_header_crops.py — §6.383 РАЗВЕДКА: шапки цветных листов — есть ли привязка «имя → цвет» без модели со зрением (16.11).

Разбор глазами 15.11: 15 из 35 ошибок выборки — цвет кривой по легенде шапки (листы 01 / 04 / 06 / 30); корпусный цвет имени листам
противоречит. Здесь — кадры шапки (строки над началом кривых эталона, вся ширина) для случайных листов с многоцветными треками
(`taskS/color_clash/rows.jsonl`, §6.378) → `output/header_2026-11-16/NN_<лист>.jpg` + index.tsv (имена кривых, цвета эталона по трекам).

  _header_crops.py [--n 24] [--seed 2]
"""
import sys, argparse, json, random, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")


def sheet_dir(n):
    return f"{n.stem[:40]}_{hashlib.md5(n.stem.encode('utf-8')).hexdigest()[:8]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default=r"F:/nds/output/header_2026-11-16")
    a = ap.parse_args()
    from PIL import Image
    from extract_nlgx import extract
    from auto import imaging as im
    im.EXIF_TRANSPOSE = False
    dist = lambda p, q: float(np.max(np.abs(np.array(p) - np.array(q))))
    pick = []
    for l in (TS / "color_clash" / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(l); by = {}
        for r in q["recs"]:
            if r["ct"] is not None:
                by.setdefault(r["track"], []).append(r)
        if any(len(R) >= 2 and max(dist(x["ct"], y["ct"]) for i, x in enumerate(R) for y in R[i + 1:]) >= 50 for R in by.values()):
            pick.append((q["sheet"], {tk: [(r["curve"], r["ct"]) for r in R] for tk, R in by.items()}))
    random.Random(a.seed).shuffle(pick); pick = pick[:a.n]
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (nm, cols) in enumerate(pick):
        n = SRC[nm]
        src = next((TS / d / (sheet_dir(n) + ".pkl") for d in ("tcache", "tcache_out") if (TS / d / (sheet_dir(n) + ".pkl")).exists()), None)
        if src is None:
            continue
        rgb = im.load_rgb(pickle.load(open(src, "rb"))["image"])
        m = extract(str(n))
        top = min(c["top_y"] for c in m["curves"] if c.get("top_y") is not None)
        y1 = int(min(rgb.shape[0], top + 150)); y0 = 0
        crop = Image.fromarray(rgb[y0:y1])
        if crop.width > 2400:
            crop = crop.resize((2400, int(crop.height * 2400 / crop.width)), Image.LANCZOS)
        fn = f"{i + 1:02d}_{nm[:50]}.jpg".replace(" ", "_").replace(",", "")
        crop.save(out / fn, quality=88)
        rows.append(f"{fn}\t{nm}\tначало кривых y={top} (кадр — строки 0…{y1})\t" +
                    " | ".join(f"{tk}: " + ", ".join(f"{c} {tuple(int(v) for v in rgb_)}" for c, rgb_ in R) for tk, R in cols.items()))
        del rgb
        print(f"  {fn}")
    (out / "index.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"★ кадров {len(rows)} → {out}")


if __name__ == "__main__":
    main()

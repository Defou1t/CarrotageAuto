r"""_segpath_switch_render.py — §6.373: картинки уходов пути с линии эксперта (15.11).

Случайные уходы из `taskS/segpath_switch/<V>.jsonl` → куски скана (±100 строк вокруг ухода, по ширине — обе линии + 60 px):
эксперт — зелёный, ушедший путь — красный, прочие пути варианта — синие (полупрозрачно). Подпись — класс (а/б/в), vis, ΔRGB, other.

  _segpath_switch_render.py [--var E] [--n 16] [--seed 7] [--out F:/nds/output/segpath_switch_2026-11-15]
"""
import sys, argparse, pickle, json, random
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", default="E")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dc", type=float, default=40.0)
    ap.add_argument("--out", default=r"F:/nds/output/segpath_switch_2026-11-15")
    a = ap.parse_args()
    from PIL import Image, ImageDraw
    from _segpath_var import sheet_dir, OUT as PV
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from _cand_ceiling import dense_tr
    from auto import imaging as im, frame as FR
    im.EXIF_TRANSPOSE = False
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    R = [dict(r, sheet=q["sheet"]) for l in (TS / "segpath_switch" / f"{a.var}.jsonl").read_text(encoding="utf-8").splitlines()
         for q in [json.loads(l)] for r in q["recs"]]
    random.Random(a.seed).shuffle(R)
    # не больше 2 уходов с одной кривой — иначе выборка — одна изрезанная кривая
    pick, cnt = [], {}
    for r in R:
        k = (r["sheet"], r["curve"])
        if cnt.get(k, 0) < 2:
            pick.append(r); cnt[k] = cnt.get(k, 0) + 1
        if len(pick) >= a.n:
            break
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    lines = []
    for i, r in enumerate(pick):
        n = SRC[r["sheet"]]
        src = next((TS / d / (sheet_dir(n) + ".pkl") for d in ("tcache_inert", "tcache_out_inert") if (TS / d / (sheet_dir(n) + ".pkl")).exists()))
        c = pickle.load(open(src, "rb"))
        rgb = im.load_rgb(c["image"])
        if getattr(c["frame"], "row_shift", None) is not None:
            rgb = FR.apply_row_shift(rgb, c["frame"].row_shift)
        g = dense([cc for cc in extract(str(n))["curves"] if cc["name"] == r["curve"]][0])
        P = [dense_tr(un(t)) for L, t in (pickle.load(open(PV / a.var / (sheet_dir(n) + ".pkl"), "rb"))["alt"] or [])]
        ys_ = r["y"]
        # ушедший путь: на линии в [y*-20, y*], далеко в [y*+25, y*+45]
        def is_dep(p):
            b = [abs(p[y] - g[y]) <= 4 for y in range(ys_ - 20, ys_ + 1) if y in p and y in g]
            f = [abs(p[y] - g[y]) > 8 for y in range(ys_ + 25, ys_ + 45) if y in p and y in g]
            return len(b) >= 15 and np.mean(b) > 0.8 and len(f) >= 10 and np.mean(f) > 0.8
        dep = [p for p in P if is_dep(p)]
        y0, y1 = max(0, ys_ - 100), min(rgb.shape[0], ys_ + 100)
        xs = [g[y] for y in range(y0, y1) if y in g] + [p[y] for p in dep for y in range(y0, y1) if y in p]
        x0, x1 = int(max(0, min(xs) - 60)), int(min(rgb.shape[1], max(xs) + 61))
        crop = Image.fromarray(rgb[y0:y1, x0:x1]).convert("RGB")
        S = 3 if (x1 - x0) < 400 else (2 if (x1 - x0) < 900 else 1)
        crop = crop.resize((crop.width * S, crop.height * S), Image.NEAREST)
        ov = Image.new("RGBA", crop.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
        for p in P:
            if any(p is q for q in dep):
                continue
            pts = [((p[y] - x0) * S, (y - y0) * S) for y in range(y0, y1) if y in p]
            if len(pts) > 1: dr.line(pts, fill=(40, 90, 255, 110), width=1)
        for p in dep:
            pts = [((p[y] - x0) * S, (y - y0) * S) for y in range(y0, y1) if y in p]
            if len(pts) > 1: dr.line(pts, fill=(255, 0, 0, 170), width=1)
        pts = [((g[y] - x0) * S + 2, (y - y0) * S) for y in range(y0, y1) if y in g]
        if len(pts) > 1: dr.line(pts, fill=(0, 200, 0, 170), width=1)
        dr.line([(0, (ys_ - y0) * S), (12, (ys_ - y0) * S)], fill=(255, 0, 255, 255), width=3)
        img = Image.alpha_composite(crop.convert("RGBA"), ov).convert("RGB")
        cl = "а" if r["vis"] < 0.5 else ("б" if (r["dc"] or 0) >= a.dc else "в")
        fn = f"{i + 1:02d}_{cl}.png"
        img.save(out / fn)
        lines.append(f"{fn}\t{r['sheet']}\t{r['curve']}\ty={ys_}\tvis={r['vis']}\tdc={r['dc']}\tother={r.get('other')}\tgap={r['gap']:.0f}\tdep_paths={len(dep)}")
        del rgb
    (out / "index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

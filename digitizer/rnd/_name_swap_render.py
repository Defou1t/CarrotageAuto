r"""_name_swap_render.py — §6.366: картинки чистых обменов имён для разбора глазами (14.11).

Из `taskS/name_swap/cases.json` (класс «обмен», выдача NSE26) — N случайных пар на разных листах: обзор и зумы (`_sample_review.render`)
плюс шапка листа над треком пары (там подписи / легенда). → `<out>/NN_*.jpg`, `<out>/diag.json`.

  _name_swap_render.py --n 24 --seed 20261114 --out F:/nds/output/name_swap_2026-11-14
"""
import sys, argparse, json, random
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--mode", default="NSE26")
    ap.add_argument("--out", default=r"F:/nds/output/name_swap_2026-11-14")
    a = ap.parse_args()
    import _sample_review as SR
    import _final_exif
    from dataset_build import find_image
    from PIL import Image
    import decode_levels as DL
    SR.MODE = a.mode; SR.OOF = TS / "rp_lvl" / a.mode
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    C = [c for c in json.loads((TS / "name_swap" / "cases.json").read_text(encoding="utf-8")) if c["cls"] == "обмен"]
    by = {}
    for c in C:
        by.setdefault(c["sheet"], c)
    sheets = sorted(by); random.Random(a.seed).shuffle(sheets)
    TRK = SR.load_tracks(); SRC = {q.name: q for q in SR.ARCH.glob("*/wlg/*.nlgx")}
    meta = []
    for nm in sheets:
        if len(meta) >= a.n:
            break
        src = SRC.get(nm); img = find_image(src) if src else None
        if img is None:
            continue
        with Image.open(img) as im:
            if im.size[0] * im.size[1] > 400e6:
                continue
        got = next(iter(sorted((SR.OOF / SR.sheet_dir(nm)).glob("*_auto.nlgx"))))
        recs, extra, cnt = SR.diagnose(src, got, TRK.get(nm, {}))
        mt, mw, gts, wr = SR.curves(src, got)
        rgb = _final_exif.load_rgb(str(img))
        tag = f"{len(meta) + 1:02d}"; c = by[nm]
        files = SR.render(rgb, mt, mw, gts, wr, recs, f"{tag}. {nm} — обмен имён {c['curve']} ↔ {c['partner']}", out, tag)
        # шапка над треком пары
        cc = {x["name"]: x for x in mt["curves"]}
        xs = []
        for n_ in (c["curve"], c["partner"]):
            if n_ in cc:
                fam = DL.build_family(mt, cc[n_])
                xs += [v for s in fam for v in (s["x_left"], s["x_right"])]
        top = min(x["top_y"] for x in mt["curves"] if x["name"] in (c["curve"], c["partner"]))
        H, W = rgb.shape[:2]
        x0 = int(max(0, (min(xs) if xs else 0) - 150)); x1 = int(min(W, (max(xs) if xs else W) + 150))
        y0 = 0; y1 = int(min(H, top + 400))
        hd = Image.fromarray(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
        s = min(1.0, 1800 / max(1, hd.size[0]), 2400 / max(1, hd.size[1]))
        hd = hd.resize((max(1, int(hd.size[0] * s)), max(1, int(hd.size[1] * s))))
        fh = f"{tag}_header.jpg"; hd.save(out / fh, "JPEG", quality=85)
        files["header"] = fh
        meta.append({"tag": tag, "sheet": nm, "part": c["part"], "pair": [c["curve"], c["partner"]], "oof": cnt, "curves": recs,
                     "files": files, "out_nlgx": str(got), "truth_nlgx": str(src), "scan": str(img)})
        print(f"  {tag} {nm}: {c['curve']} ↔ {c['partner']}")
        del rgb
    (out / "diag.json").write_text(json.dumps({"seed": a.seed, "sheets": meta}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"★ {len(meta)} листов → {out}")


if __name__ == "__main__":
    main()

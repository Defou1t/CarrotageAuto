r"""_spike_review.py — РАЗБОР ГЛАЗАМИ ЗАМЕН УКОЛОВ (§6.356): случайные уколы, которые `auto.spike` правило (под уколом туши сети
< ts, на хорде > tc) меняет на хорду, — скан до / после.

Берёт уколы из `_spike_ink_seg_150.json` (`_spike_ink.py --seg`), выдачу NSE19 и скан листа. Картинка — три панели одна под другой (×3, широкие мельче): скан, наша
трасса с уколом (красная, строки укола толще), после замены (синяя, хорда толще); жёлтая метка слева — строки укола. Эталона на картинке НЕТ (судья решает по
туши, как увидит заказчик); метка эталона — в `review_meta.json`.

  _spike_review.py [--n 30] [--seed 3] [--out F:/nds/output/spike_review_2026-11-03]
"""
import sys, argparse, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--ts", type=float, default=0.1)
    ap.add_argument("--tc", type=float, default=0.5)
    ap.add_argument("--src", default=r"F:/nds/output/taskS/_spike_ink_seg_150.json")
    ap.add_argument("--out", default=r"F:/nds/output/spike_review_2026-11-03")
    a = ap.parse_args()
    import _sample_review as S, _final_exif
    from dataset_build import find_image
    from PIL import Image, ImageDraw, ImageFont
    S.OOF = S.TS / "rp_lvl" / "NSE19"
    R = json.loads(Path(a.src).read_text(encoding="utf-8"))
    hits = [(nm, part, r) for nm, part, rows, _ in R for r in rows if len(r) > 10 and r[9] is not None and r[10] is not None
            and np.isfinite(r[9]) and r[9] < a.ts and r[10] > a.tc]
    rng = np.random.default_rng(a.seed)
    pick = [hits[i] for i in sorted(rng.permutation(len(hits))[:a.n])]
    print(f"★ замен по правилу {len(hits)}; в выборке {len(pick)}")
    SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta, cache = [], {}
    for k, (nm, part, r) in enumerate(pick, 1):
        g, y0, y1 = r[0], int(r[1]), int(r[2])
        if nm not in cache:
            got = next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))))
            mt, mw, gts, wr = S.curves(SRC[nm], got)
            cache = {nm: (wr, _final_exif.load_rgb(str(find_image(Path(SRC[nm])))))}
        wr, rgb = cache[nm]
        w = wr[g]; H, W = rgb.shape[:2]
        ya, yb = y0 - 1, y1 + 1
        ch = lambda y: w[ya] + (w[yb] - w[ya]) * (y - ya) / (yb - ya)
        xs = [w[y] for y in range(ya, yb + 1)]
        r0, r1 = max(0, y0 - 45), min(H, y1 + 46); c0, c1 = max(0, int(min(xs)) - 90), min(W, int(max(xs)) + 91)
        crop = Image.fromarray(np.ascontiguousarray(rgb[r0:r1, c0:c1])).convert("RGB")
        Z = min(3.0, 1400 / crop.width)                 # широкий укол — мельче, панели друг под другом
        crop = crop.resize((round(crop.width * Z), round(crop.height * Z)), Image.LANCZOS)
        P = lambda x, y: ((x - c0) * Z + Z / 2, (y - r0) * Z + Z / 2)
        panels = [crop.copy()]
        for after in (False, True):
            im = crop.copy(); d = ImageDraw.Draw(im)
            xv = {y: (ch(y) if after and y0 <= y <= y1 else w[y]) for y in range(r0, r1) if y in w}
            col = (0, 70, 255) if after else (230, 0, 0)
            ys_ = sorted(xv)
            for u, v in zip(ys_[:-1], ys_[1:]):
                hot = (y0 - 1 <= u <= y1) and (y0 <= v <= y1 + 1)
                d.line([P(xv[u], u), P(xv[v], v)], fill=col, width=4 if hot else 1)
            d.rectangle([0, (y0 - r0) * Z, 8, (y1 - r0 + 1) * Z], fill=(255, 200, 0))
            panels.append(im)
        ph = panels[0].height
        sheet = Image.new("RGB", (max(panels[0].width, 900), ph * 3 + 24 + 34), "white")
        for i_, im in enumerate(panels):
            sheet.paste(im, (0, 34 + i_ * (ph + 12)))
        try:
            fnt = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            fnt = None
        ImageDraw.Draw(sheet).text((4, 6), f"S{k:02d}: сверху скан, ниже до (красная, укол толще), внизу после (синяя, хорда толще); жёлтая метка — строки укола",
                                   fill=(0, 0, 0), font=fnt)
        fn = f"S{k:02d}.jpg"; sheet.save(out / fn, quality=90)
        meta.append({"file": fn, "sheet": nm, "part": part, "curve": g, "y0": y0, "y1": y1, "amp": round(float(r[3]), 1),
                     "label": r[6], "seg_spike": round(float(r[9]), 3), "seg_chord": round(float(r[10]), 3)})
        print(f"  {fn}: {nm} {g} {y0}–{y1}, амплитуда {r[3]:.0f}, эталон: {r[6]}")
    (out / "review_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"★ {out}")


if __name__ == "__main__":
    main()

r"""_lvl_review.py — ДЛИННЫЕ СЕРИИ НЕВЕРНОГО УРОВНЯ ГЛАЗАМИ (08.11, §6.354): по классам `_lvl_kind.py` — «метка» (та же тушь, наш 5× /
эксперт 1× и наоборот), «перо 1×→5×» (мы на 1×, эксперт на 5×), «перо 5×→1×» — случайные серии ≥ 300 строк, картинка участка скана.

Наша трасса цветом уровня (1× красный, 5× лиловый, 3-й синий, выше оранжевый), эксперт — толстая зелёная (1× плотная, 5× и выше
бледная), серые штрихи — края полосы шкалы уровня 0; в подписи — шкалы цепочки. Синяя метка слева — строки серии.

  _lvl_review.py [--n 10] [--seed 5] [--out F:/nds/output/lvl_review_2026-11-08]
"""
import sys, argparse, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
import numpy as np

LV = [(220, 0, 0), (200, 0, 200), (0, 60, 230), (230, 140, 0), (0, 0, 0)]
KINDS = ["метка 5×→1×", "метка 1×→5×", "перо 1×→5×", "перо 5×→1×"]


def kind(a_same, lo, lt):
    if a_same:
        return "метка 5×→1×" if lo > lt else "метка 1×→5×"
    return "перо 5×→1×" if lo > lt else "перо 1×→5×"


def scan_runs(args):
    """серии ≥ 300 строк одного класса (разрыв ≤ 3) → [(лист, кривая, y0, y1, класс)]"""
    nm, part, src, got = args
    import _sample_review as S, decode_levels as DL
    try:
        mt, mw, gts, wr = S.curves(src, got)
    except Exception:
        return []
    LG = {c["name"]: S.lv_rows(c) for c in mt["curves"] if c["name"] in gts}
    LW = {c["name"]: S.lv_rows(c) for c in mw["curves"] if c["name"] in wr}
    out = []
    for c in mt["curves"]:
        g = c["name"]
        if g not in gts or g not in wr:
            continue
        ch = DL.build_family(mt, c)
        if len(ch) < 2:
            continue
        w, gt, lw, lg = wr[g], gts[g], LW.get(g, {}), LG.get(g, {})
        S._TREES.clear()
        wl = S.relevel(w, lw, lg, ch)
        cur = None
        for y in sorted(w):
            k = None
            if y in gt and lw.get(y, 0) != lg.get(y, 0) and abs(wl[y] - gt[y]) > 10:
                k = kind(abs(w[y] - gt[y]) <= 10, lw.get(y, 0), lg.get(y, 0))
            if cur and (k != cur[2] or y - cur[1] > 3):
                if cur[1] - cur[0] >= 300:
                    out.append((nm, g, cur[0], cur[1], cur[2]))
                cur = None
            if k and not cur:
                cur = [y, y, k]
            elif k:
                cur[1] = y
        if cur and cur[1] - cur[0] >= 300:
            out.append((nm, g, cur[0], cur[1], cur[2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="серий на класс")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default=r"F:/nds/output/lvl_review_2026-11-08")
    a = ap.parse_args()
    import _sample_review as S, decode_levels as DL, _final_exif
    from dataset_build import find_image
    from PIL import Image, ImageDraw
    S.OOF = S.TS / "rp_lvl" / "NSE19"
    SETS = S.eval_sets(); SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    jobs = [(nm, part, SRC[nm], next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None))
            for part, SS in SETS.items() if part != "сорт A" for nm in sorted(SS) if nm in SRC]
    jobs = [j for j in jobs if j[3] is not None]
    rng = np.random.default_rng(11)
    jobs = [jobs[i] for i in sorted(rng.permutation(len(jobs))[:a.limit])]
    from multiprocessing import Pool
    with Pool(1) as pool:
        R = [x for r in pool.imap_unordered(scan_runs, jobs, chunksize=2) for x in r]
    print(f"★ серий ≥ 300 строк: " + ", ".join(f"{k} {sum(1 for x in R if x[4] == k)}" for k in KINDS))
    rng = np.random.default_rng(a.seed)
    pick = []
    for k in KINDS:
        L = [x for x in R if x[4] == k]
        pick += [L[i] for i in sorted(rng.permutation(len(L))[:a.n])]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = []
    JS = {j[0]: j for j in jobs}
    for i, (nm, g, y0, y1, k) in enumerate(pick, 1):
        _, part, src, got = JS[nm]
        mt, mw, gts, wr = S.curves(src, got)
        c = next(c for c in mt["curves"] if c["name"] == g)
        ch = DL.build_family(mt, c)
        lw = S.lv_rows(next(c_ for c_ in mw["curves"] if c_["name"] == g)); lg = S.lv_rows(c)
        w, gt = wr[g], gts[g]
        rgb = _final_exif.load_rgb(str(find_image(Path(src)))); H, W = rgb.shape[:2]
        za, zb = max(0, y0 - 200), min(H, y1 + 200)
        xs = [w[y] for y in range(za, zb) if y in w] + [gt[y] for y in range(za, zb) if y in gt] + [ch[0]["x_left"], ch[0]["x_right"]]
        xa, xb = max(0, int(min(xs)) - 60), min(W, int(max(xs)) + 60)
        zi = Image.fromarray(np.ascontiguousarray(rgb[za:zb, xa:xb])).convert("RGBA")
        zo = Image.new("RGBA", zi.size, (0, 0, 0, 0)); zd = ImageDraw.Draw(zo)
        for e in (ch[0]["x_left"], ch[0]["x_right"]):
            for yy in range(0, zb - za, 24):
                zd.line([(e - xa, yy), (e - xa, yy + 12)], fill=(90, 90, 90, 255), width=2)
        for tr, lv, wd in ((gt, lg, 9), (w, lw, 3)):
            pts = sorted(y for y in tr if za <= y < zb)
            for p1, p2 in zip(pts, pts[1:]):
                if p2 - p1 <= 3:
                    col = (0, 170, 0, 120 if lv.get(p1, 0) == 0 else 55) if tr is gt else LV[min(lv.get(p1, 0), 4)] + (255,)
                    zd.line([(tr[p1] - xa, p1 - za), (tr[p2] - xa, p2 - za)], fill=col, width=wd)
        zd.rectangle([(0, y0 - za), (10, y1 - za)], fill=(0, 90, 255, 255))
        im = Image.alpha_composite(zi, zo).convert("RGB")
        sc = min(1.0, 1600 / im.size[1], 1400 / im.size[0])
        if sc < 1:
            im = im.resize((max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))), Image.LANCZOS)
        sheet = Image.new("RGB", (max(im.size[0], 1100), im.size[1] + 52), "white"); sheet.paste(im, (0, 52))
        d = ImageDraw.Draw(sheet)
        scales = "; ".join(f"ур.{j}: [{s_['v_left']:g}, {s_['v_right']:g}]" for j, s_ in enumerate(ch))
        d.text((6, 4), f"L{i:02d} {k}: {nm[:60]} {g}, строки {y0}–{y1} ({y1 - y0 + 1})", fill=(0, 0, 0), font=S._font(15))
        d.text((6, 26), f"шкалы: {scales}; масштаб картинки {sc:.2f}", fill=(0, 0, 0), font=S._font(14))
        fn = f"L{i:02d}.jpg"; sheet.save(out / fn, quality=88)
        meta.append({"file": fn, "kind": k, "sheet": nm, "curve": g, "y0": y0, "y1": y1, "scales": scales})
        print(f"  {fn}: {k} {nm[:50]} {g} {y0}–{y1}")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"★ {out}")


if __name__ == "__main__":
    main()

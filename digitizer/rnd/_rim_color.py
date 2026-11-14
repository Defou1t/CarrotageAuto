r"""_rim_color.py — §6.369 РАЗВЕДКА: цвет линии по КАЙМЕ против нынешнего признака стиля (14.11).

Разбор обменов §6.366: красная ГК отличается от НГК каймой (R−G +39…+51 против +5…+9), а признак стиля прода (`name_style.style_along`:
c1 = R−B, c2 = G−(R+B)/2 у самого тёмного пикселя линии, медиана по строкам) берёт ядро. Здесь для пар ГК–НГК одного трека эталона (и пар
ГЗ разного цвета нет — их не знаем заранее) по скану вдоль трасс эталона: нынешний c1 и кайма — тёмные пиксели в ±7 px от трассы с
порогом от фона (D > 0.35 · max D окна), средний (R − G) и доля пикселей с R − G > 20. Доля пар, где ГК «краснее» НГК, по каждому признаку.

  _rim_color.py [--n 250]
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np


def rim(rgb, gray, paper, ys, xs, half=7):
    H, W = gray.shape
    ys = ys.astype(np.int64); xs = np.round(xs).astype(np.int64)
    ok = (ys >= 0) & (ys < H) & (xs >= half) & (xs < W - half)
    ys, xs = ys[ok], xs[ok]
    if len(ys) < 50:
        return np.nan, np.nan
    cols = xs[:, None] + np.arange(-half, half + 1)[None, :]
    D = paper[ys][:, None].astype(np.int16) - gray[ys[:, None], cols].astype(np.int16)
    thr = 0.35 * D.max(1, keepdims=True)
    m = (D > np.maximum(thr, 25))
    px = rgb[ys[:, None], cols].astype(np.int16)
    rg = (px[..., 0] - px[..., 1])
    v = rg[m]
    if v.size < 100:
        return np.nan, np.nan
    return float(v.mean()), float(np.mean(v > 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    a = ap.parse_args()
    import _sample_review as SR, _final_exif
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from dataset_build import find_image
    from auto import meta as M, name_style as NS, level_ink as LI
    TR = SR.load_tracks(); S = SR.eval_sets(); sheets = sorted(set().union(*S.values()))
    SRC = {q.name: q for q in SR.ARCH.glob("*/wlg/*.nlgx")}
    res = []; nsh = 0
    for nm in sheets:
        if nsh >= a.n or nm not in SRC:
            continue
        try:
            mt = extract(str(SRC[nm]))
        except Exception:
            continue
        tm = TR.get(nm, {})
        C = {c["name"]: c for c in mt["curves"] if M.mnem_root(c["name"]) in ("GK", "NGK")}
        gk = [n for n in C if M.mnem_root(n) == "GK"]; ngk = [n for n in C if M.mnem_root(n) == "NGK"]
        pairs = [(g, n) for g in gk for n in ngk if tm.get(g) is not None and tm.get(g) == tm.get(n) and g.split()[1] == n.split()[1]]
        if not pairs:
            continue
        img = find_image(SRC[nm])
        if img is None:
            continue
        rgb = _final_exif.load_rgb(str(img)); gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
        paper = LI.paper_of(gray)
        F = {}
        for n in set(x for p in pairs for x in p):
            d = dense(C[n])
            if len(d) < 300:
                continue
            ys = np.array(sorted(d))[::6]; xs = np.array([d[y] for y in ys])
            st = NS.style_along(rgb, gray, paper, ys, xs)
            r1, r2 = rim(rgb, gray, paper, ys, xs)
            F[n] = (st["c1"] if st else np.nan, r1, r2)
        for g, n in pairs:
            if g in F and n in F:
                res.append((nm, F[g], F[n]))
        nsh += 1
        del rgb, gray
        if nsh % 25 == 0:
            print(f"  {nsh} листов, пар {len(res)}")
    R = np.array([(g[0], n[0], g[1], n[1], g[2], n[2]) for _, g, n in res], float)
    ok = np.all(np.isfinite(R), 1); R = R[ok]
    print(f"★ пар ГК–НГК {len(R)} (листов {nsh})")
    print(f"  нынешний c1 (R−B ядра): ГК > НГК в {np.mean(R[:, 0] > R[:, 1]):.1%}; с разницей > 10: ГК {np.mean(R[:, 0] > R[:, 1] + 10):.1%}, "
          f"НГК {np.mean(R[:, 1] > R[:, 0] + 10):.1%}")
    print(f"  кайма R−G среднее: ГК > НГК в {np.mean(R[:, 2] > R[:, 3]):.1%}; с разницей > 10: ГК {np.mean(R[:, 2] > R[:, 3] + 10):.1%}, "
          f"НГК {np.mean(R[:, 3] > R[:, 2] + 10):.1%}")
    print(f"  кайма доля R−G > 20: ГК > НГК в {np.mean(R[:, 4] > R[:, 5]):.1%}; разница > 0.1: ГК {np.mean(R[:, 4] > R[:, 5] + 0.1):.1%}, "
          f"НГК {np.mean(R[:, 5] > R[:, 4] + 0.1):.1%}")


if __name__ == "__main__":
    main()

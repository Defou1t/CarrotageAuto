r"""_nocand_style.py — §6.370 РАЗВЕДКА: какие линии остаются без честного кандидата (15.11).

Кривые эталона поле+вне, у слота которых в прод-пуле (`line_choice_r` / `_out_r`) нет честного кандидата (§6.364: 552 неверных), против
кривых с честным кандидатом (контроль): стиль линии эталона по скану (`name_style.style_along` по трассе эталона: толщина, темнота, доля
разрывов gap — пунктир / прерывистость, цвет c1 / c2, доля годных строк used) + длина, извилистость, близость соседней кривой эталона (доля
строк ≤ 25 px от другой кривой). Какие свойства отличают «без кандидата» — туда и метить новое ведение.

  _nocand_style.py [--n 400]
"""
import sys, argparse, pickle, random
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    import _sample_review as SR, _final_exif
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from dataset_build import find_image
    from auto import meta as M, name_style as NS, level_ink as LI
    lab = {}
    for d in ("line_choice_r", "line_choice_out_r"):
        for f in (TS / d).glob("*.pkl"):
            r = pickle.load(open(f, "rb"))
            for s in r["slots"]:
                if s.get("has_truth"):
                    lab.setdefault(r["sheet"], {})[s["name"]] = any(x.get("lab") and x["lab"][0] for x in s["cands"])
    sheets = sorted(nm for nm, v in lab.items() if not all(v.values()))
    random.Random(7).shuffle(sheets)
    SRC = {q.name: q for q in SR.ARCH.glob("*/wlg/*.nlgx")}
    rows = []; ns = 0
    for nm in sheets:
        if ns >= a.n or nm not in SRC:
            continue
        img = find_image(SRC[nm])
        if img is None:
            continue
        try:
            mt = extract(str(SRC[nm]))
        except Exception:
            continue
        rgb = _final_exif.load_rgb(str(img)); gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
        paper = LI.paper_of(gray)
        D = {c["name"]: dense(c) for c in mt["curves"] if M.mnem_root(c["name"]) != "DA"}
        for n, has in lab[nm].items():
            d = D.get(n)
            if not d or len(d) < 200:
                continue
            ys = np.array(sorted(d))[::6]; xs = np.array([d[y] for y in ys])
            st = NS.style_along(rgb, gray, paper, ys, xs)
            near = 0.0
            for o, od in D.items():
                if o != n and od:
                    near = max(near, float(np.mean([abs(od[y] - d[y]) <= 25 for y in ys if y in od])) if any(y in od for y in ys) else 0.0)
            dx = np.abs(np.diff(xs))
            rows.append(dict(has=has, fam=M.mnem_root(n), n=len(d), near=near, rough=float(np.median(dx)) if len(dx) else np.nan,
                             **({k: st[k] for k in ("width", "dark", "gap", "c1", "c2", "used")} if st else {})))
        ns += 1
        del rgb, gray
        if ns % 50 == 0:
            print(f"  {ns} листов, кривых {len(rows)}")
    import collections
    A = [r for r in rows if not r["has"]]; B = [r for r in rows if r["has"]]
    print(f"★ кривых: без честного кандидата {len(A)}, с кандидатом {len(B)} (листов {ns})")
    for k in ("width", "dark", "gap", "used", "c1", "near", "rough", "n"):
        va = np.array([r.get(k, np.nan) for r in A], float); vb = np.array([r.get(k, np.nan) for r in B], float)
        print(f"   {k:6s}: без кандидата медиана {np.nanmedian(va):8.2f} (p25 {np.nanpercentile(va, 25):7.2f}, p75 {np.nanpercentile(va, 75):7.2f}) | "
              f"с кандидатом {np.nanmedian(vb):8.2f} (p25 {np.nanpercentile(vb, 25):7.2f}, p75 {np.nanpercentile(vb, 75):7.2f})")
    for lab_, cond in (("пунктир (gap > 0.2)", lambda r: r.get("gap", 0) > 0.2), ("бледная (dark < 60)", lambda r: r.get("dark", 999) < 60),
                       ("тонкая (width < 3)", lambda r: r.get("width", 99) < 3), ("рядом соседняя (near > 0.5)", lambda r: r["near"] > 0.5),
                       ("стиль не снят", lambda r: "width" not in r)):
        fa = np.mean([cond(r) for r in A]); fb = np.mean([cond(r) for r in B])
        print(f"   {lab_:28s}: без кандидата {fa:6.1%} | с кандидатом {fb:6.1%}")
    ca = collections.Counter(r["fam"] for r in A); cb = collections.Counter(r["fam"] for r in B)
    print("   семейства без кандидата (доля от семейства): " + ", ".join(f"{f} {ca[f]}/{ca[f] + cb[f]}" for f, _ in ca.most_common(12)))


if __name__ == "__main__":
    main()

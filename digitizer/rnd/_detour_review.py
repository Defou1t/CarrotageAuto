r"""_detour_review.py — ОБЪЕЗДЫ ГЛАЗАМИ: «ДО / ПОСЛЕ» НА СЛУЧАЙНЫХ КРИВЫХ (03.11, §6.355).

Случайные кривые повтора с объездами (`<лист>_detour.json` режима B) — две панели одного участка скана: выдача A (NSE19) и B (NSE22),
поверх — линия эксперта (толстая, полупрозрачная). Участок — самое длинное заменённое окно ± 350 строк. В подписи — доля лишних строк
кривой (> 10 px от эталона) до и после и вердикт приёмки.

  _detour_review.py [--a NSE19] [--b NSE22] [--n 24] [--seed 1] [--out F:/nds/output/detour_review_<дата>]
"""
import sys, argparse, json, random, html as H
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
import numpy as np


def off_share(S, w, gt, lw, lg, ch):
    S._TREES.clear()
    wl = S.relevel(w, lw, lg, ch)
    com = [y for y in sorted(wl) if y in gt]
    if len(com) < 30:
        return None, False
    S._TREES.clear()
    do = S._plane(gt, com, wl)
    return float(np.mean(do > 10.0)), bool(S.HON(*S.err(wl, gt)))


LV = [(220, 0, 0), (200, 0, 200), (0, 60, 230), (230, 140, 0), (0, 0, 0)]   # уровень масштаба нашей: 1× красный, 5× лиловый, …


def panel(rgb, za, zb, xa, xb, gt, w, wins, lab, lg, lw):
    from PIL import Image, ImageDraw
    import _sample_review as S
    zi = Image.fromarray(np.ascontiguousarray(rgb[za:zb, xa:xb])).convert("RGBA")
    zo = Image.new("RGBA", zi.size, (0, 0, 0, 0)); zd = ImageDraw.Draw(zo)
    for tr, lv, wd in ((gt, lg, 7), (w, lw, 2)):
        pts = sorted(y for y in tr if za <= y < zb)
        for y1, y2 in zip(pts, pts[1:]):
            if y2 - y1 <= 3:
                col = (0, 170, 0, 110 if lv.get(y1, 0) == 0 else 60) if tr is gt else LV[min(lv.get(y1, 0), 4)] + (255,)
                zd.line([(tr[y1] - xa, y1 - za), (tr[y2] - xa, y2 - za)], fill=col, width=wd)
    for a, b in wins:
        if b >= za and a < zb:
            zd.rectangle([(0, max(0, a - za)), (8, min(zb - za, b - za))], fill=(0, 90, 255, 255))
    im = Image.alpha_composite(zi, zo).convert("RGB")
    d = ImageDraw.Draw(im); d.rectangle([(0, 0), (im.size[0], 22)], fill="white"); d.text((12, 3), lab, fill="black", font=S._font(15))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="NSE19"); ap.add_argument("--b", default="NSE22")
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="F:/nds/output/detour_review_2026-11-03")
    a = ap.parse_args()
    import _sample_review as S, decode_levels as DL, _final_exif
    from dataset_build import find_image
    from PIL import Image, ImageDraw
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    SETS = S.eval_sets(); part = {nm: p for p, SS in SETS.items() for nm in SS}
    SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    RB = S.TS / "rp_lvl" / a.b; RA = S.TS / "rp_lvl" / a.a
    pool = []
    for d in sorted(RB.iterdir()):
        for j in d.glob("*_detour.json"):
            nm = j.name[:-len("_detour.json")] + ".nlgx"
            if nm in part and nm in SRC:
                for cv, L in _group(json.loads(j.read_text(encoding="utf-8"))).items():
                    pool.append((nm, d.name, cv, L))
    print(f"★ кривых с объездами в {a.b} (оценочные листы): {len(pool)}")
    random.seed(a.seed); pick = random.sample(pool, min(a.n, len(pool)))
    rows, stat = [], {"лучше": 0, "хуже": 0, "так же": 0, "нет эталона": 0}
    for i, (nm, sd, cv, L) in enumerate(sorted(pick)):
        src = SRC[nm]; ga = next((RA / sd).glob("*_auto.nlgx"), None); gb = next((RB / sd).glob("*_auto.nlgx"))
        if ga is None:
            continue
        mt, mwa, gts, wra = S.curves(src, ga); _, mwb, _, wrb = S.curves(src, gb)
        if cv not in gts or cv not in wra or cv not in wrb:
            stat["нет эталона"] += 1; continue
        c = next(x for x in mt["curves"] if x["name"] == cv); ch = DL.build_family(mt, c); lg = S.lv_rows(c)
        lwa = S.lv_rows(next(x for x in mwa["curves"] if x["name"] == cv)); lwb = S.lv_rows(next(x for x in mwb["curves"] if x["name"] == cv))
        oa, ha = off_share(S, wra[cv], gts[cv], lwa, lg, ch); ob, hb = off_share(S, wrb[cv], gts[cv], lwb, lg, ch)
        if oa is None or ob is None:
            stat["нет эталона"] += 1; continue
        k = "лучше" if ob < oa - 0.002 else "хуже" if ob > oa + 0.002 else "так же"; stat[k] += 1
        rgb = _final_exif.load_rgb(str(find_image(Path(src))))
        wins = [(r["a"], r["b"]) for r in L]; big = max(wins, key=lambda t: t[1] - t[0])
        H0, W0 = rgb.shape[:2]
        za, zb = max(0, big[0] - 350), min(H0, big[1] + 350)
        xs = [x for tr in (gts[cv], wra[cv], wrb[cv]) for y, x in tr.items() if za <= y < zb]
        if ch:
            xs += [min(min(s_["x_left"], s_["x_right"]) for s_ in ch), max(max(s_["x_left"], s_["x_right"]) for s_ in ch)]
        xa, xb = int(max(0, min(xs) - 60)), int(min(W0, max(xs) + 60))
        pa = panel(rgb, za, zb, xa, xb, gts[cv], wra[cv], wins, f"{a.a}: лишних {oa:.1%}, {'верна' if ha else 'НЕ верна'}", lg, lwa)
        pb = panel(rgb, za, zb, xa, xb, gts[cv], wrb[cv], wins, f"{a.b}: лишних {ob:.1%}, {'верна' if hb else 'НЕ верна'}", lg, lwb)
        s = min(1.0, 700 / pa.size[0]); sz = (max(1, round(pa.size[0] * s)), max(1, round(pa.size[1] * s)))
        if s < 1:
            pa, pb = pa.resize(sz, Image.LANCZOS), pb.resize(sz, Image.LANCZOS)
        can = Image.new("RGB", (pa.size[0] * 2 + 16, pa.size[1] + 48), "white")
        can.paste(pa, (0, 48)); can.paste(pb, (pa.size[0] + 16, 48))
        d = ImageDraw.Draw(can); f = S._font(15)
        d.text((6, 4), f"{i + 1:02d}. {nm} [{part[nm]}] — {cv}: окон заменено {len(L)}, строк {sum(b - a + 1 for a, b in wins)}; "
                       f"участок {za}–{zb}", fill="black", font=f)
        d.text((6, 24), "толстая зелёная — эксперт (бледнее — 5× и выше), тонкая — наша: красная 1×, лиловая 5×, синяя 3-й масштаб; синие метки слева — заменённые окна", fill=(90, 90, 90), font=f)
        fn = out / f"D{i + 1:02d}_{k}.jpg"; can.save(fn, quality=86)
        rows.append((fn.name, nm, part[nm], cv, len(L), oa, ob, ha, hb, k))
        print(f"  {fn.name}: {nm} {cv} — лишних {oa:.1%} → {ob:.1%}, верна {ha} → {hb}")
    print(f"★ итог выборки: {stat}")
    doc = ["<html><head><meta charset='utf-8'><title>Объезды до / после</title></head><body style='font-family:sans-serif'>",
           f"<h2>Объезды (§6.355): {a.a} → {a.b}, случайные {len(rows)} кривых (зерно {a.seed})</h2><p>{H.escape(str(stat))}</p>"]
    for fn, nm, p, cv, nw, oa, ob, ha, hb, k in rows:
        doc.append(f"<h3>{H.escape(fn)} — {H.escape(nm)} [{p}] {H.escape(cv)}: окон {nw}, лишних {oa:.1%} → {ob:.1%}, "
                   f"верна {'да' if ha else 'нет'} → {'да' if hb else 'нет'} ({k})</h3><img src='{fn}' style='max-width:100%'>")
    doc.append("</body></html>")
    (out / "ОБЪЕЗДЫ.html").write_text("\n".join(doc), encoding="utf-8")
    print(f"★ {out / 'ОБЪЕЗДЫ.html'}")


def _group(L):
    G = {}
    for r in L:
        G.setdefault(r["curve"], []).append(r)
    return G


if __name__ == "__main__":
    main()

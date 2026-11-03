r"""_switch_cases.py — МЕХАНИЗМ ОШИБОК УРОВНЯ МАСШТАБА (03.11, §6.353, после `_switch_audit.py`).

Кривые, потерянные ТОЛЬКО из-за уровня (с уровнями эталона при той же трассе — верна, сейчас нет): серии строк с чужим уровнем
(≥ 40 строк) → на каждой границе серии: чья смена (эталона / наша / обе), есть ли скачок трассы в пикселях (≥ 25% ширины шкалы
за ≤ 12 строк), положение трассы в полосе шкалы, уровень эталона против нашего. Сводка по классам + выборка картинок с уровнями
(толстая полупрозрачная — эталон, тонкая — наша; цвет — уровень: 0 синий, 1 оранжевый, 2 зелёный, 3 красный, далее фиолетовый).

  _switch_cases.py --workers 3                  → taskS/_switch_cases.json, сводка
  _switch_cases.py --render 40 --seed 3         → output/switch_cases_<дата>/NN_*.jpg + cases.json (по 1 картинке на кривую)
"""
import sys, argparse, json, time, math
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
from collections import Counter
import numpy as np

MINRUN, JUMP, JW = 40, 0.25, 12
LCOL = [(30, 90, 230), (245, 130, 20), (20, 160, 60), (220, 30, 30), (140, 60, 200)]


def runs(ys, mask):
    out, i, n = [], 0, len(ys)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1] and ys[j + 1] - ys[j] <= 5:
                j += 1
            out.append((i, j)); i = j + 1
        else:
            i += 1
    return out


def jump_at(tr, y, w):
    """скачок трассы у строки y: максимум |x(y2) − x(y1)| по парам строк в [y − JW, y + JW] с y1 < y ≤ y2, в долях ширины"""
    a = [tr[k] for k in range(y - JW, y) if k in tr]; b = [tr[k] for k in range(y, y + JW + 1) if k in tr]
    if not a or not b:
        return None
    return max(abs(max(b) - min(a)), abs(min(b) - max(a))) / w


def analyse(nm, part, src, got, want):
    import _sample_review as S
    import decode_levels as DL
    mt, mw, gts, wr = S.curves(src, got)
    out = []
    for c in mt["curves"]:
        g = c["name"]
        if g not in want:
            continue
        ch = DL.build_family(mt, c); cw = next(x for x in mw["curves"] if x["name"] == g)
        lg, lw = S.lv_rows(c), S.lv_rows(cw); gt, w = gts[g], wr[g]
        xl = min(min(s["x_left"], s["x_right"]) for s in ch); xr = max(max(s["x_left"], s["x_right"]) for s in ch)
        W0 = abs(ch[0]["x_right"] - ch[0]["x_left"]) or 1.0
        com = [y for y in sorted(w) if y in gt]; ys = np.array(com)
        mis = np.array([lw.get(y, 0) != lg.get(y, 0) for y in com], bool)
        R = []
        for i0, i1 in runs(ys, mis):
            a, b = int(ys[i0]), int(ys[i1])
            if b - a + 1 < MINRUN:
                continue
            ed = {}
            for side, y, yp in (("нач", a, a - 1), ("кон", b + 1, b)):
                tg = lg.get(y, 0) != lg.get(yp, 0); tw = lw.get(y, 0) != lw.get(yp, 0)
                ed[side] = {"чья": "обе" if tg and tw else "эталон" if tg else "наша" if tw else "нет",
                            "скачок_наш": jump_at(w, y, W0), "скачок_эт": jump_at(gt, y, W0)}
            u = np.array([(w[y] - xl) / max(1.0, xr - xl) for y in com[i0:i1 + 1]])
            lt = Counter(lg.get(y, 0) for y in com[i0:i1 + 1]).most_common(1)[0][0]
            lo = Counter(lw.get(y, 0) for y in com[i0:i1 + 1]).most_common(1)[0][0]
            R.append({"a": a, "b": b, "n": int(i1 - i0 + 1), "lt": int(lt), "lo": int(lo), "u_med": float(np.median(u)),
                      "u_max": float(u.max()), "ed": ed})
        out.append({"curve": g, "nlev": len(ch), "n_com": len(com), "mis": int(mis.sum()), "runs": R,
                    "chain": [[round(s["x_left"]), round(s["x_right"]), s["v_left"], s["v_right"]] for s in ch]})
    return out


def one(args):
    nm, part, src, got, want = args
    try:
        return nm, part, analyse(nm, part, src, got, want), ""
    except Exception as e:
        return nm, part, [], f"{type(e).__name__}: {e}"


def klass(r):
    """класс серии по её началу (конец — то же зеркально): чья смена и виден ли скачок трассы"""
    e = r["ed"]["нач"]; j = e["скачок_наш"]
    jt = "скачок" if j is not None and j >= JUMP else "без скачка" if j is not None else "край"
    return f"{e['чья']} / {jt}"


def report(res):
    for nmp, parts in (("поле+вне", ("поле", "вне")), ("сорт A", ("сорт A",))):
        R = [r for nm, part, cs, e in res if part in parts for c in cs for r in c["runs"]]
        tot = sum(r["n"] for r in R)
        print(f"\n## {nmp}: серий с чужим уровнем ≥ {MINRUN} строк {len(R)}, строк {tot}")
        K = Counter(); KN = Counter()
        for r in R:
            K[klass(r)] += 1; KN[klass(r)] += r["n"]
        for k, v in KN.most_common():
            print(f"  начало серии {k:28s}: серий {K[k]:5d}, строк {v / tot:6.1%}")
        D = Counter(); DN = Counter()
        for r in R:
            d = "наш ниже" if r["lo"] < r["lt"] else "наш выше"
            D[d] += 1; DN[d] += r["n"]
        print("  " + ", ".join(f"{k}: серий {D[k]}, строк {DN[k] / tot:.1%}" for k in D))
        E = Counter((r["ed"]["нач"]["чья"], r["ed"]["кон"]["чья"]) for r in R)
        print("  (начало, конец): " + ", ".join(f"{a}→{b} {v}" for (a, b), v in E.most_common(8)))
        U = np.array([r["u_med"] for r in R])
        print(f"  положение трассы в полосе шкал (медиана серии): квартили {np.percentile(U, [10, 25, 50, 75, 90]).round(2)}")


def render(rgb, mt, mw, gts, wr, c, title, fn):
    from PIL import Image, ImageDraw
    import _sample_review as S
    g = c["curve"]; gt, w = gts[g], wr[g]
    lg = S.lv_rows(next(x for x in mt["curves"] if x["name"] == g)); lw = S.lv_rows(next(x for x in mw["curves"] if x["name"] == g))
    r = max(c["runs"], key=lambda t: t["n"])
    H0, W0 = rgb.shape[:2]
    za, zb = max(0, r["a"] - 450), min(H0, r["a"] + 750)
    xl = min(min(s[0], s[1]) for s in c["chain"]); xr = max(max(s[0], s[1]) for s in c["chain"])
    xa, xb = int(max(0, xl - 80)), int(min(W0, xr + 80))
    s = min(1.0, 1100 / max(1, xb - xa))
    zi = Image.fromarray(np.ascontiguousarray(rgb[za:zb, xa:xb])).convert("RGBA")
    zo = Image.new("RGBA", zi.size, (0, 0, 0, 0)); zd = ImageDraw.Draw(zo)
    for k, (x0, x1, v0, v1) in enumerate(c["chain"]):
        for x_ in (x0, x1):
            zd.line([(x_ - xa, 0), (x_ - xa, zb - za)], fill=LCOL[min(k, 4)] + (90,), width=1)
    for tr, lv, wd, al in ((gt, lg, 7, 120), (w, lw, 2, 255)):
        pts = sorted(y for y in tr if za <= y < zb)
        for y1, y2 in zip(pts, pts[1:]):
            if y2 - y1 <= 3:
                zd.line([(tr[y1] - xa, y1 - za), (tr[y2] - xa, y2 - za)], fill=LCOL[min(lv.get(y1, 0), 4)] + (al,), width=wd)
    for y in (r["a"], r["b"]):
        if za <= y < zb:
            zd.line([(0, y - za), (12, y - za)], fill=(0, 0, 0, 255), width=3)
    im = Image.alpha_composite(zi, zo).convert("RGB")
    if s < 1:
        im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
    can = Image.new("RGB", (max(im.size[0], 900), im.size[1] + 70), "white"); can.paste(im, (0, 70))
    f = S._font(15); d = ImageDraw.Draw(can)
    d.text((6, 4), title, fill="black", font=f)
    d.text((6, 24), f"серия строк {r['a']}–{r['b']} ({r['n']}): эталон ур. {r['lt']}, наш {r['lo']}; начало: {r['ed']['нач']['чья']}, "
                    f"скачок нашей {r['ed']['нач']['скачок_наш'] if r['ed']['нач']['скачок_наш'] is None else round(r['ed']['нач']['скачок_наш'], 2)}",
           fill=(160, 0, 0), font=f)
    d.text((6, 44), "толстая — эталон, тонкая — наша; цвет — уровень (0 синий, 1 оранж., 2 зел., 3 красн.); вертикали — края шкал уровней; "
                    "чёрные метки слева — границы серии", fill=(90, 90, 90), font=f)
    can.save(fn, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--mode", default="NSE19")
    ap.add_argument("--render", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import _sample_review as S
    S.OOF = S.TS / "rp_lvl" / a.mode
    aud = json.loads((S.TS / f"_switch_audit_{a.mode}.json").read_text(encoding="utf-8"))
    want = {}
    for nm, part, rows, e in aud:
        for r in rows:
            if r["hp"] and not r["hv"]:
                want.setdefault((nm, part), set()).add(r["curve"])
    SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    jobs = []
    for (nm, part), cs in sorted(want.items()):
        got = next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
        if got is not None and nm in SRC:
            jobs.append((nm, part, SRC[nm], got, cs))
    if a.render:
        import _final_exif
        from dataset_build import find_image
        res = json.loads((S.TS / "_switch_cases.json").read_text(encoding="utf-8"))
        flat = [(nm, part, c) for nm, part, cs, e in res for c in cs if c["runs"]]
        order = np.random.default_rng(a.seed).permutation(len(flat))[:a.render]
        od = Path(a.out or (S.OUT / f"switch_cases_{time.strftime('%Y-%m-%d')}_s{a.seed}")); od.mkdir(parents=True, exist_ok=True)
        meta = []
        for k, i in enumerate(order, 1):
            nm, part, c = flat[i]
            img = find_image(SRC[nm])
            if img is None:
                continue
            got = next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))))
            mt, mw, gts, wr = S.curves(SRC[nm], got)
            rgb = _final_exif.load_rgb(str(img))
            fn = od / f"{k:02d}_{''.join(ch if ch.isalnum() else '_' for ch in c['curve'])}.jpg"
            render(rgb, mt, mw, gts, wr, c, f"{k:02d}. {nm} — {c['curve']} ({part}); уровней {c['nlev']}; чужой уровень на {c['mis']} "
                                              f"строках из {c['n_com']}", fn)
            meta.append({"k": k, "sheet": nm, "part": part, "file": fn.name, **c})
            del rgb
        (od / "cases.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"★ {len(meta)} картинок → {od}")
        return
    print(f"★ листов {len(jobs)}, кривых {sum(len(j[4]) for j in jobs)}")
    from multiprocessing import Pool
    res = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(one, jobs, chunksize=2):
            res.append(r)
    print(f"  ошибок {sum(1 for r in res if r[3])}")
    (S.TS / "_switch_cases.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    report(res)


if __name__ == "__main__":
    main()

r"""_segpath_switch.py — §6.373 РАЗВЕДКА: КУДА УХОДИТ ПУТЬ ПО КАРТЕ СЕТИ, КОГДА ПОКИДАЕТ ЛИНИЮ ЭКСПЕРТА (15.11).

§6.372: у слотов без честного кандидата пути по карте сети покрывают линию эксперта кусками (объединение ≤ 3 px — медиана 0.72), но ни
один путь не идёт по ней целиком. Уход пути = участок ≥ RUN строк на линии (плоскость ≤ 3 px) и сразу за ним ≥ RUN строк дальше OFF px.
В окне [y* + 5, y* + 5 + W] после ухода сравниваются: линия эксперта (её x) и место, куда ушёл путь. Классы ухода:
  (а) разрыв — сеть туши НЕ видит линию эксперта после ухода (доля строк с P ≥ 0.5 в ±2 px < 0.5): пунктир / пропуск туши;
  (б) другой цвет — линия видна, а под путём тушь другого цвета: ΔRGB медиан (по тёмным пикселям ±1 px) ≥ DC;
  (в) тот же цвет — линия видна, цвет близок: различить может только ведение (плавность / пересечения / обученный трекер).
Шум меры цвета — та же разность на участке, где путь ИДЁТ по линии (до ухода: путь против эксперта, одни и те же строки) — калибровка DC.
Пути — `taskS/segpath_var/<V>/` (§6.372), слоты без честного кандидата — прод-пул (`line_choice_r` / `_out_r`), скан — как в стенде
(кэш `tcache_inert` / `tcache_out_inert`, `row_shift`, без разворота EXIF), сеть — вне фолда (`ink_seg_f`).

  _segpath_switch.py --sheets taskS/segpath_sheets60.txt [--var E] [--dc 40]   → taskS/segpath_switch/<V>.jsonl (продолжаемо), сводка
"""
import sys, argparse, pickle, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
SEG = TS / "ink_seg_f"
RUN, OFF, W = 20, 8, 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", required=True)
    ap.add_argument("--var", default="E")
    ap.add_argument("--dc", type=float, default=40.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from _segpath_var import sheets_of, sheet_dir, OUT as PV
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from _cand_ceiling import _tree, dense_tr
    from auto import imaging as im, frame as FR, ink_seg as IS
    im.EXIF_TRANSPOSE = False
    folds = json.loads((SEG / "folds.json").read_text(encoding="utf-8"))
    lab = {}
    for d in ("line_choice_r", "line_choice_out_r"):
        for f in (TS / d).glob("*.pkl"):
            r = pickle.load(open(f, "rb"))
            lab[r["sheet"]] = {s["name"]: (any(x.get("lab") and x["lab"][0] for x in s["cands"]), s["band"]) for s in r["slots"] if s.get("has_truth")}
    od = TS / "segpath_switch"; od.mkdir(parents=True, exist_ok=True)
    jf = od / f"{a.var}.jsonl"
    done = {json.loads(l)["sheet"] for l in jf.read_text(encoding="utf-8").splitlines()} if jf.exists() else set()
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    T0 = time.time(); nsh = 0
    for n in sheets_of(a.sheets):
        if a.limit and nsh >= a.limit:
            break
        if n.name in done or n.name not in lab or all(h for h, _ in lab[n.name].values()):
            continue
        pf = PV / a.var / (sheet_dir(n) + ".pkl")
        src = next((TS / d / (sheet_dir(n) + ".pkl") for d in ("tcache_inert", "tcache_out_inert") if (TS / d / (sheet_dir(n) + ".pkl")).exists()), None)
        if not pf.exists() or src is None:
            continue
        P = [dense_tr(un(t)) for L, t in (pickle.load(open(pf, "rb"))["alt"] or [])]
        P = [p for p in P if len(p) >= 50]
        c = pickle.load(open(src, "rb"))
        rgb = im.load_rgb(c["image"])
        if getattr(c["frame"], "row_shift", None) is not None:
            rgb = FR.apply_row_shift(rgb, c["frame"].row_shift)
        H, Wd = rgb.shape[:2]
        fo = folds.get(n.parent.parent.name)
        net = str(SEG / (f"f{fo}.pt" if fo is not None else "all.pt"))
        G = {cc["name"]: dense(cc) for cc in extract(str(n))["curves"]}
        gray = rgb.mean(2)

        def ink_rgb(ys, xs):
            """медиана RGB тёмных пикселей (≤ 60-й перцентиль яркости окна ±1 px) по строкам ys у x = xs"""
            px = []
            for y, x in zip(ys, xs):
                xi = int(round(x))
                if 0 <= y < H and 1 <= xi < Wd - 1:
                    g = gray[y, xi - 1:xi + 2]; j = int(np.argmin(g))
                    px.append(rgb[y, xi - 1 + j])
            return np.median(np.array(px, float), 0) if len(px) >= 8 else None

        recs = []
        for nm, (had, band) in lab[n.name].items():
            g = G.get(nm)
            if had or not g or len(g) < 50:
                continue
            tg = _tree(g)
            for p in P:
                ys = np.array(sorted(y for y in p if y in g))
                if len(ys) < 2 * RUN:
                    continue
                d, _ = tg.query(np.array([[y, p[y]] for y in ys], float))
                on = d <= 3; off = d > OFF
                i = 0
                while i < len(ys) - 2 * RUN:
                    # конец участка на линии: RUN строк on подряд, затем RUN строк off подряд (строки подряд по y)
                    if on[i:i + RUN].all() and ys[i + RUN - 1] - ys[i] == RUN - 1:
                        j = i + RUN
                        while j < len(ys) and on[j]:
                            j += 1
                        k = j
                        while k < len(ys) and not off[k] and k - j < 15:   # переходная зона ≤ 15 строк
                            k += 1
                        if k + RUN <= len(ys) and off[k:k + RUN].all():
                            ystar = int(ys[j - 1])
                            ya = [y for y in range(ystar + 5, ystar + 5 + W) if y in g and y in p]
                            yb = [y for y in range(ystar - 5 - W, ystar - 5) if y in g and y in p]
                            if len(ya) >= 10 and len(yb) >= 10:
                                # сеть по окну, покрывающему обе линии
                                xa = [g[y] for y in ya] + [p[y] for y in ya]
                                x0 = int(max(0, min(xa) - 20)); x1 = int(min(Wd, max(xa) + 21))
                                y0 = int(ya[0]); y1 = int(ya[-1]) + 1
                                pm = IS.prob_map(rgb, y0, y1, x0, x1, net)
                                if pm.ndim == 3:
                                    pm = pm[0]
                                seen = []
                                for y in ya:
                                    cx = int(round(g[y])) - x0
                                    if 2 <= cx < pm.shape[1] - 2:
                                        seen.append(float(pm[y - y0, cx - 2:cx + 3].max()) >= 0.5)
                                vis = float(np.mean(seen)) if seen else 0.0
                                cg = ink_rgb(ya, [g[y] for y in ya]); cp = ink_rgb(ya, [p[y] for y in ya])
                                cg0 = ink_rgb(yb, [g[y] for y in yb]); cp0 = ink_rgb(yb, [p[y] for y in yb])
                                dc = float(np.abs(cg - cp).max()) if cg is not None and cp is not None else None
                                dc0 = float(np.abs(cg0 - cp0).max()) if cg0 is not None and cp0 is not None else None
                                dpath = float(np.abs(cp0 - cp).max()) if cp0 is not None and cp is not None else None
                                other = float(np.mean([any(q is not p and y in q and abs(q[y] - g[y]) <= 3 for q in P) for y in ya]))
                                recs.append(dict(curve=nm, y=ystar, vis=round(vis, 3), dc=dc, dc0=dc0, dpath=dpath, other=round(other, 3),
                                                 gap=float(np.median([abs(p[y] - g[y]) for y in ya]))))
                            i = k + RUN
                            continue
                    i += 1
        with open(jf, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(sheet=n.name, recs=recs), ensure_ascii=False) + "\n")
        nsh += 1
        del rgb, gray
        if nsh % 5 == 0:
            print(f"  листов {nsh}, {time.time() - T0:.0f} с")
    R = [dict(r, sheet=q["sheet"]) for l in jf.read_text(encoding="utf-8").splitlines() for q in [json.loads(l)] for r in q["recs"]]
    if not R:
        print("★ уходов нет"); return
    vis = np.array([r["vis"] for r in R])
    dcs = np.array([r["dc"] if r["dc"] is not None else np.nan for r in R])
    dc0 = np.array([r["dc0"] if r["dc0"] is not None else np.nan for r in R])
    A = vis < 0.5
    B = ~A & (dcs >= a.dc)
    C = ~A & ~B
    print(f"★ уходов пути с линии эксперта (вариант {a.var}): {len(R)} у {len({(r['sheet'], r['curve']) for r in R})} кривых")
    print(f"   (а) разрыв — сеть не видит линию после ухода: {A.sum()} = {A.mean():.0%}")
    print(f"   (б) видна, другой цвет (ΔRGB ≥ {a.dc:.0f}): {B.sum()} = {B.mean():.0%}")
    print(f"   (в) видна, цвет близок: {C.sum()} = {C.mean():.0%}")
    ok = ~np.isnan(dc0)
    print(f"   шум меры цвета (до ухода, путь на линии): ΔRGB ≥ {a.dc:.0f} у {np.mean(dc0[ok] >= a.dc):.0%}; медиана {np.nanmedian(dc0):.0f} "
          f"| после ухода медиана {np.nanmedian(dcs):.0f}")
    oth = np.array([r.get("other", np.nan) for r in R], float)
    print(f"   после ухода линию эксперта ведёт ДРУГОЙ путь (≥ 50% строк окна): {np.nanmean(oth >= 0.5):.0%} — обмен на пересечении; никто — "
          f"{np.nanmean(oth < 0.5):.0%}")
    from collections import Counter
    per = {}
    for r, ca in zip(R, np.where(A, "а", np.where(B, "б", "в"))):
        per.setdefault((r["sheet"], r["curve"]), []).append(ca)
    cc = Counter(Counter(v).most_common(1)[0][0] for v in per.values())
    print(f"   по кривым (класс большинства уходов): " + ", ".join(f"({k}) {cc[k]}" for k in "абв") + f" из {len(per)}")
    gp = np.array([r["gap"] for r in R])
    print(f"   отход пути после ухода: медиана {np.median(gp):.0f} px, ≤ 25 px у {np.mean(gp <= 25):.0%}")


if __name__ == "__main__":
    main()

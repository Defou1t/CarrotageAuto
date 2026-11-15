r"""_seg_sees_missing.py — §6.371 РАЗВЕДКА: видит ли сеть туши линии, которых нет в пуле кандидатов (15.11).

§6.370: без честного кандидата остаются изрезанные шумные кривые и пунктир (~320 линий нет в пуле, ~150 ведение рвётся). Если сеть туши
(`auto.ink_seg`, вне фолда — `taskS/ink_seg_f`, модель фолда скважины, иначе all.pt) их ВИДИТ, а пути по её карте (§6.349) их не дают — чинить
извлечение путей; если НЕ видит — доучивать сеть. Для слотов эталона без честного кандидата в прод-пуле (`line_choice_r` / `_out_r`) и
контрольных с кандидатом (на тех же листах): доля строк трассы эксперта (через 4), где вероятность туши ≥ 0.5 в ±2 px, и медиана вероятности.

  _seg_sees_missing.py [--n 120]   → taskS/seg_sees/rows.jsonl (продолжаемо), сводка
"""
import sys, argparse, pickle, json, random
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
SEG = TS / "ink_seg_f"
OUT = TS / "seg_sees"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()
    import _final_exif
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from dataset_build import find_image
    from auto import ink_seg as IS
    OUT.mkdir(parents=True, exist_ok=True)
    jf = OUT / "rows.jsonl"
    done = {json.loads(l)["sheet"] for l in jf.read_text(encoding="utf-8").splitlines()} if jf.exists() else set()
    lab = {}
    for d in ("line_choice_r", "line_choice_out_r"):
        for f in (TS / d).glob("*.pkl"):
            r = pickle.load(open(f, "rb"))
            for s in r["slots"]:
                if s.get("has_truth"):
                    lab.setdefault(r["sheet"], {})[s["name"]] = (any(x.get("lab") and x["lab"][0] for x in s["cands"]), s["band"])
    sheets = sorted(nm for nm, v in lab.items() if not all(h for h, _ in v.values()))
    random.Random(11).shuffle(sheets); sheets = sheets[:a.n]
    FO = json.loads((SEG / "folds.json").read_text(encoding="utf-8"))
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    for i, nm in enumerate(sheets):
        if nm in done or nm not in SRC:
            continue
        img = find_image(SRC[nm])
        if img is None:
            continue
        k = FO.get(SRC[nm].parent.parent.name)
        net = SEG / (f"f{k}.pt" if k is not None else "all.pt")
        mt = extract(str(SRC[nm]))
        rgb = _final_exif.load_rgb(str(img)); H, W = rgb.shape[:2]
        C = {c["name"]: c for c in mt["curves"]}
        recs = []
        for n, (has, band) in lab[nm].items():
            if n not in C:
                continue
            d = dense(C[n])
            if len(d) < 200:
                continue
            ys = np.array(sorted(d))[::4]; xs = np.array([d[y] for y in ys])
            x0 = int(max(0, min(band[0], xs.min()) - 40)); x1 = int(min(W, max(band[1], xs.max()) + 40))
            y0 = int(ys.min()); y1 = int(ys.max()) + 1
            P = IS.prob_map(rgb, y0, y1, x0, x1, net)
            if P.ndim == 3:
                P = P[0]
            pv = []
            for y, x in zip(ys, xs):
                c = int(round(x)) - x0
                if 2 <= c < P.shape[1] - 2:
                    pv.append(float(P[y - y0, c - 2:c + 3].max()))
            if len(pv) < 50:
                continue
            pv = np.array(pv)
            recs.append(dict(curve=n, has=bool(has), see=round(float(np.mean(pv >= 0.5)), 3), med=round(float(np.median(pv)), 3)))
        with open(jf, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(sheet=nm, recs=recs), ensure_ascii=False) + "\n")
        del rgb
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sheets)}")
    R = [r for l in jf.read_text(encoding="utf-8").splitlines() for r in json.loads(l)["recs"]]
    A = np.array([r["see"] for r in R if not r["has"]]); B = np.array([r["see"] for r in R if r["has"]])
    print(f"★ кривых: без честного кандидата {len(A)}, с кандидатом {len(B)}")
    for t in (0.5, 0.7, 0.9):
        print(f"   сеть видит ≥ {t:.0%} строк: без кандидата {np.mean(A >= t):6.1%} | с кандидатом {np.mean(B >= t):6.1%}")
    print(f"   медиана доли видимых строк: без кандидата {np.median(A):.2f} | с кандидатом {np.median(B):.2f}")


if __name__ == "__main__":
    main()

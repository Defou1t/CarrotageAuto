r"""_peak_ceiling.py — §6.382 РАЗВЕДКА: потолок «сшивателя пиков» карты сети туши у слотов без честного кандидата (16.11).

§6.372–6.375: жадный Витерби по карте сети и его настройки дают ≤ 10.9% честных путей у слотов без кандидата, а A/B §6.375 показал, что
и такая прибавка тонет в перемешивании переобученного выбора. Совместное назначение K путей (min-cost-flow с пропусками) — работа на
неделю; до неё — потолок: доля строк линии эксперта, где у карты сети (вне фолда) есть пик строки (локальный максимум ≥ thr · max строки)
в ±3 px от x эксперта. Если такая доля ≥ 0.9, идеальный сшиватель пиков мог бы провести честный путь (покрытие ≥ 0.9). Слоты без честного
кандидата (прод-пул `line_choice_r` / `_out_r`) на 59 листах стенда §6.372 и контроль — слоты с кандидатом на тех же листах.

  _peak_ceiling.py [--sheets taskS/segpath_sheets60.txt]   → taskS/peak_ceiling/rows.jsonl (продолжаемо), сводка
"""
import sys, argparse, pickle, json, hashlib, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
SEG = TS / "ink_seg_f"
OUT = TS / "peak_ceiling"
THR = (0.6, 0.3, 0.1)


def sheet_dir(n):
    return f"{n.stem[:40]}_{hashlib.md5(n.stem.encode('utf-8')).hexdigest()[:8]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", default=str(TS / "segpath_sheets60.txt"))
    a = ap.parse_args()
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from auto import imaging as im, ink_seg as IS
    im.EXIF_TRANSPOSE = False
    OUT.mkdir(parents=True, exist_ok=True)
    jf = OUT / "rows.jsonl"
    done = {json.loads(l)["sheet"] for l in jf.read_text(encoding="utf-8").splitlines()} if jf.exists() else set()
    lab = {}
    for d in ("line_choice_r", "line_choice_out_r"):
        for f in (TS / d).glob("*.pkl"):
            r = pickle.load(open(f, "rb"))
            lab[r["sheet"]] = {s["name"]: (any(x.get("lab") and x["lab"][0] for x in s["cands"]), s["band"]) for s in r["slots"] if s.get("has_truth")}
    folds = json.loads((SEG / "folds.json").read_text(encoding="utf-8"))
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    want = [l.strip() for l in Path(a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()]
    T0 = time.time(); k = 0
    for nm in want:
        if nm in done or nm not in SRC or nm not in lab:
            continue
        n = SRC[nm]
        src = next((TS / d / (sheet_dir(n) + ".pkl") for d in ("tcache", "tcache_out") if (TS / d / (sheet_dir(n) + ".pkl")).exists()), None)
        if src is None:
            continue
        rgb = im.load_rgb(pickle.load(open(src, "rb"))["image"])      # координаты nlgx — исходного скана
        H, W = rgb.shape[:2]
        fo = folds.get(n.parent.parent.name)
        net = str(SEG / (f"f{fo}.pt" if fo is not None else "all.pt"))
        C = {c["name"]: c for c in extract(str(n))["curves"]}
        recs = []
        for cn, (has, band) in lab[nm].items():
            if cn not in C:
                continue
            g = dense(C[cn])
            if len(g) < 200:
                continue
            ys = np.array(sorted(g)); xs = np.array([g[y] for y in ys])
            x0 = int(max(0, min(band[0], xs.min()) - 40)); x1 = int(min(W, max(band[1], xs.max()) + 40))
            y0 = int(ys.min()); y1 = int(ys.max()) + 1
            P = IS.prob_map(rgb, y0, y1, x0, x1, net)
            if P.ndim == 3:
                P = P[0]
            mx = P.max(1, keepdims=True)
            loc = (P >= np.roll(P, 1, 1)) & (P >= np.roll(P, -1, 1))
            share = {}
            for t in THR:
                pk = loc & (P >= t * mx) & (mx > 0.05)
                hit = 0
                for y, x in zip(ys, xs):
                    c = int(round(x)) - x0
                    if 3 <= c < P.shape[1] - 3 and pk[y - y0, c - 3:c + 4].any():
                        hit += 1
                share[str(t)] = round(hit / len(ys), 3)
            recs.append(dict(curve=cn, has=bool(has), share=share))
        with open(jf, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(sheet=nm, recs=recs), ensure_ascii=False) + "\n")
        k += 1
        del rgb
        if k % 10 == 0:
            print(f"  листов {k}, {time.time() - T0:.0f} с")
    R = [r for l in jf.read_text(encoding="utf-8").splitlines() for r in json.loads(l)["recs"]]
    for has, nmh in ((False, "без честного кандидата"), (True, "с кандидатом (контроль)")):
        S = [r for r in R if r["has"] == has]
        if not S:
            continue
        print(f"★ слотов {nmh}: {len(S)}")
        for t in THR:
            v = np.array([r["share"][str(t)] for r in S])
            print(f"   пики ≥ {t} · max строки: доля строк эксперта с пиком в ±3 px ≥ 0.9 — {np.mean(v >= 0.9):.1%}, ≥ 0.8 — {np.mean(v >= 0.8):.1%},"
                  f" медиана {np.median(v):.2f}")


if __name__ == "__main__":
    main()

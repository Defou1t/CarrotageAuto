r"""_swap_by_header.py — §6.386: обмен имён по цвету шапки (облачное чтение) — проверка и A/B против NSE26 (22.11).

Цвета имён по шапке — ответы агентов (`--items` JSON-списки {file, curve, color, sign}, `--index` — index.tsv кадров тех же партий);
цвет туши наших кривых — `taskS/color_clash/rows.jsonl` (co; семейство при разбросе каналов ≥ 20, иначе «чёрный»). В треке пара наших
кривых A, B: цвет A ≠ цвет имени A по шапке, цвет B ≠ цвет имени B, а цвет A = цвет имени B и цвет B = цвет имени A ⇒ обмен имён.
Мера — приёмка (`_sample_review.diagnose`, выдача NSE26 вне фолда): кривая эталона a верна после обмена, если на ней честна наша B
(`as` = B), и наоборот; верная до обмена — теряется. Безымянный счёт обменом не меняется по построению.

  _swap_by_header.py [--items a.json b.json] [--index a.tsv b.tsv]   → сводка по частям (поле / вне / сорт A), знаковый тест по листам
"""
import sys, json, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
from math import comb
import numpy as np

TS = Path(r"F:/nds/output/taskS")
W = {"red": 0, "orange": 1, "brown": 1, "yellow": 2, "green": 3, "teal": 4, "blue": 5, "purple": 6, "pink": 7, "black": -1, "grey": -1}


def sign_p(up, dn):
    n = up + dn
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(up, dn) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", default=[str(TS / "header_cloud" / "pilot_items.json")])
    ap.add_argument("--index", nargs="+", default=[r"F:/nds/output/header_cloud_pilot/index.tsv"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import _sample_review as SR
    from _header_colors import fam_of_hue, hue_deg
    SR.MODE = "NSE26"; SR.OOF = TS / "rp_lvl" / "NSE26"
    fam = lambda c: -1 if c is None or (max(c) - min(c)) < 20 else fam_of_hue(float(hue_deg(np.array([c]))[0]))
    H = {}
    for itf, ixf in zip(a.items, a.index):
        idx = {l.split("\t")[0]: l.split("\t")[1] for l in Path(ixf).read_text(encoding="utf-8").splitlines() if l.strip()}
        for it in json.load(open(itf, encoding="utf-8")):
            if it["color"] in W and it["file"] in idx:
                H[(idx[it["file"]], it["curve"])] = W[it["color"]]
    sheets = sorted({k[0] for k in H})
    CO = {}
    for l in (TS / "color_clash" / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(l)
        for r in q["recs"]:
            if r["co"] is not None:
                CO[(q["sheet"], r["curve"])] = fam(r["co"])
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    TRACK = SR.load_tracks(); S = SR.eval_sets()
    part_of = lambda nm: next((p for p, ss in S.items() if nm in ss), None)
    agg = {p: dict(base=0, gain=0, loss=0, up=0, dn=0, sw=0, sheets=0) for p in ("поле", "вне", "сорт A")}
    swaps_log = []
    for nm in sheets:
        p = part_of(nm)
        if p is None:
            continue
        got = next(iter(sorted((SR.OOF / SR.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
        if got is None or nm not in SRC:
            continue
        recs, extra, cnt = SR.diagnose(SRC[nm], got, TRACK.get(nm, {}))
        R = {r["curve"]: r for r in recs}
        A = agg[p]; A["sheets"] += 1; A["base"] += sum(r["cat"] == "верно" for r in recs)
        names = [c for c in R if (nm, c) in H and (nm, c) in CO]
        by = {}
        for c in names:
            by.setdefault((c.split() + [""])[1], []).append(c)
        used, g_, l_ = set(), 0, 0
        for tk, L in by.items():
            mis = [c for c in L if CO[(nm, c)] != H[(nm, c)]]
            for i, x0 in enumerate(mis):
                for y0 in mis[i + 1:]:
                    if x0 in used or y0 in used:
                        continue
                    if CO[(nm, x0)] == H[(nm, y0)] and CO[(nm, y0)] == H[(nm, x0)]:
                        used |= {x0, y0}; A["sw"] += 1
                        for x, y in ((x0, y0), (y0, x0)):
                            was = R[x]["cat"] == "верно"; now = R[x].get("as") == y
                            g_ += (not was) and now; l_ += was and not now
                        swaps_log.append(dict(sheet=nm, part=p, a=x0, b=y0))
        A["gain"] += g_; A["loss"] += l_; A["up"] += (g_ - l_) > 0; A["dn"] += (g_ - l_) < 0
    if a.out:
        Path(a.out).write_text(json.dumps(swaps_log, ensure_ascii=False, indent=0), encoding="utf-8")
    fo = {k: agg["поле"][k] + agg["вне"][k] for k in agg["поле"]}
    for nmp, A in (("поле+вне", fo), ("поле", agg["поле"]), ("вне", agg["вне"]), ("сорт A", agg["сорт A"])):
        print(f"★ {nmp}: листов с цветами шапки {A['sheets']}, именных верных NSE26 {A['base']}; обменов {A['sw']}; стало верными {A['gain']},"
              f" перестало {A['loss']}; Δ {A['gain'] - A['loss']:+d} (листов ↑{A['up']} / ↓{A['dn']}, p = {sign_p(A['up'], A['dn']):.4f})")


if __name__ == "__main__":
    main()

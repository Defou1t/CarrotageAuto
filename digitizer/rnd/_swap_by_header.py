r"""_swap_by_header.py — §6.386 проверка на выдаче: обмен имён по цвету шапки (облачное чтение) (22.11).

90 листов пилота (`output/header_cloud_pilot/index.tsv`), цвета имён по шапке — `taskS/header_cloud/pilot_items.json`; цвет туши наших
кривых — `taskS/color_clash/rows.jsonl` (co; семейство при разбросе каналов ≥ 20, иначе «чёрный»). В треке пара наших кривых A, B: цвет
A ≠ цвет имени A по шапке, цвет B ≠ цвет имени B, а цвет A = цвет имени B и цвет B = цвет имени A ⇒ обмен имён. Мера — приёмка
(`_sample_review.diagnose`, выдача NSE26 вне фолда): кривая эталона a верна после обмена, если на ней честна наша B (`as` = B), и наоборот;
верная до обмена — теряется.
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
W = {"red": 0, "orange": 1, "brown": 1, "yellow": 2, "green": 3, "teal": 4, "blue": 5, "purple": 6, "pink": 7, "black": -1, "grey": -1}


def main():
    import _sample_review as SR
    from _header_colors import fam_of_hue, hue_deg
    SR.MODE = "NSE26"; SR.OOF = TS / "rp_lvl" / "NSE26"
    fam = lambda c: -1 if c is None or (max(c) - min(c)) < 20 else fam_of_hue(float(hue_deg(np.array([c]))[0]))
    idx = {l.split("\t")[0]: l.split("\t")[1] for l in Path(r"F:/nds/output/header_cloud_pilot/index.tsv").read_text(encoding="utf-8").splitlines() if l.strip()}
    H = {}
    for it in json.load(open(TS / "header_cloud" / "pilot_items.json", encoding="utf-8")):
        if it["color"] in W:
            H[(idx[it["file"]], it["curve"])] = W[it["color"]]
    CO = {}
    for l in (TS / "color_clash" / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(l)
        for r in q["recs"]:
            if r["co"] is not None:
                CO[(q["sheet"], r["curve"])] = fam(r["co"])
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    TRACK = SR.load_tracks()
    gain = loss = swaps = sheets_up = sheets_dn = 0; base = 0; ncur = 0
    for nm in sorted(set(idx.values())):
        got = next(iter(sorted((SR.OOF / SR.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
        if got is None or nm not in SRC:
            continue
        recs, extra, cnt = SR.diagnose(SRC[nm], got, TRACK.get(nm, {}))
        R = {r["curve"]: r for r in recs}
        ncur += len(recs); base += sum(r["cat"] == "верно" for r in recs)
        names = [c for c in R if (nm, c) in H and (nm, c) in CO]
        by = {}
        for c in names:
            by.setdefault((c.split() + [""])[1], []).append(c)
        used, g_, l_ = set(), 0, 0
        for tk, L in by.items():
            mis = [c for c in L if CO[(nm, c)] != H[(nm, c)]]
            for i, a in enumerate(mis):
                for b in mis[i + 1:]:
                    if a in used or b in used:
                        continue
                    if CO[(nm, a)] == H[(nm, b)] and CO[(nm, b)] == H[(nm, a)]:
                        used |= {a, b}; swaps += 1
                        for x, y in ((a, b), (b, a)):
                            was = R[x]["cat"] == "верно"; now = R[x].get("as") == y
                            g_ += (not was) and now; l_ += was and not now
                        print(f"  {nm[:44]} {tk}: {a} ↔ {b}")
        gain += g_; loss += l_
        sheets_up += (g_ - l_) > 0; sheets_dn += (g_ - l_) < 0
    print(f"★ 90 листов пилота: кривых эталона {ncur}, именных верных NSE26 {base}; обменов {swaps}; стало верными {gain}, перестало {loss};"
          f" Δ {gain - loss:+d} (листов ↑{sheets_up} / ↓{sheets_dn})")


if __name__ == "__main__":
    main()

r"""_name_swap_stat.py — §6.366 РАЗВЕДКА: «верная линия под чужим именем» — устройство случаев (14.11).

По листам, где у кривых эталона причина «верно под другим именем» (`sample_2026-11-10/pop_curves.tsv`), заново — `_sample_review.diagnose`
на выдаче `--mode` (NSE26): для кривой эталона g — чья наша кривая w несёт её линию (поле as), и что стоит под именем g. Классы:
  обмен — w несёт линию g, а наша g — линию w (оба «под другим именем», партнёры друг друга);
  дубль — наша g совпадает с другой нашей кривой трека на ≥ 30% строк (≤ 3 px);
  цепочка / прочее.
Плюс семейства пар и стиль линий эталона пары (цвет c1/c2, толщина) — различимы ли они. → taskS/name_swap/cases.json

  _name_swap_stat.py [--mode NSE26]
"""
import sys, argparse, csv, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
from collections import Counter
import numpy as np
import _sample_review as SR
from auto import meta as M

TS = Path(r"F:/nds/output/taskS")


def ovl(a, b, tol=3.0):
    return sum(1 for y, x in a.items() if y in b and abs(b[y] - x) <= tol) / max(1, len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="NSE26")
    a = ap.parse_args()
    SR.MODE = a.mode; SR.OOF = TS / "rp_lvl" / a.mode
    rows = list(csv.reader(open(r"F:/nds/output/sample_2026-11-10/pop_curves.tsv", encoding="utf-8"), delimiter="\t"))[1:]
    sheets = sorted({(r[0], r[1]) for r in rows if r[4] == "верно под другим именем"})
    TRK = SR.load_tracks(); SRC = {q.name: q for q in SR.ARCH.glob("*/wlg/*.nlgx")}
    out = []; cls = Counter(); fam = Counter()
    for part, nm in sheets:
        got = next(iter(sorted((SR.OOF / SR.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
        if got is None or nm not in SRC:
            continue
        recs, extra, cnt = SR.diagnose(SRC[nm], got, TRK.get(nm, {}))
        _, mw, gts, wr = SR.curves(SRC[nm], got)
        R = {r["curve"]: r for r in recs}
        tm = TRK.get(nm, {})
        for g, r in R.items():
            if r["cat"] != "верно под другим именем":
                continue
            w = r.get("as")
            rw = R.get(w)
            dup = g in wr and any(o != g and tm.get(o) == tm.get(g) and ovl(wr[g], wr[o]) >= 0.3 for o in wr)
            if rw is not None and rw["cat"] == "верно под другим именем" and rw.get("as") == g:
                c = "обмен"
            elif dup:
                c = "дубль"
            else:
                c = "цепочка / прочее"
            cls[c] += 1
            fp = tuple(sorted([M.mnem_root(g), M.mnem_root(w or "?")]))
            fam[(c, fp)] += 1
            out.append(dict(part=part, sheet=nm, curve=g, partner=w, cls=c, cat_of_g_out=None if g not in R else R[g]["cat"],
                            partner_cat=rw["cat"] if rw else None))
    (TS / "name_swap").mkdir(exist_ok=True)
    (TS / "name_swap" / "cases.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(cls.values())
    print(f"★ {a.mode}: «верно под другим именем» {n} (листов {len(sheets)})")
    for k, v in cls.most_common():
        print(f"   {k:18s} {v:4d}  {v / max(n, 1):.0%}")
    print("★ частые пары семейств:")
    for (c, fp), v in fam.most_common(15):
        print(f"   {c:18s} {fp[0]}–{fp[1]}: {v}")


if __name__ == "__main__":
    main()

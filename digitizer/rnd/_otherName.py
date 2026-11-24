r"""_otherName.py — 24.11: разбор «верно под другим именем» (выдача NSE28 вне фолда): пары мнемоник, чистые обмены (A↔B) / циклы / вытеснение.
  _otherName.py → taskS/othername_nse28.json + сводка
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
from collections import Counter
import _sample_review as SR
TS = Path(r"F:/nds/output/taskS")
SR.MODE = "NSE28"; SR.OOF = TS / "rp_lvl" / "NSE28"
pop = [l.split("\t") for l in Path(r"F:/nds/output/sample_2026-11-24/pop_curves.tsv").read_text(encoding="utf-8").splitlines()[1:]]
want = sorted({r[1] for r in pop if r[4] == "верно под другим именем" and r[0] in ("поле", "вне")})
part = {r[1]: r[0] for r in pop}
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
TRACK = SR.load_tracks()
out = []
for i, nm in enumerate(want):
    got = next(iter(sorted((SR.OOF / SR.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
    if got is None: continue
    recs, extra, cnt = SR.diagnose(SRC[nm], got, TRACK.get(nm, {}))
    R = {r["curve"]: r for r in recs}
    for r in recs:
        if r["cat"] != "верно под другим именем": continue
        b = r.get("as"); rb = R.get(b)
        kind = "обмен" if rb is not None and rb.get("as") == r["curve"] else ("наше имя — не в эталоне" if rb is None else "цикл / вытеснение")
        out.append(dict(sheet=nm, part=part[nm], ref=r["curve"], ours=b, kind=kind, ours_cat=rb["cat"] if rb else None))
    if i % 50 == 0: print(i, len(want), len(out))
(TS / "othername_nse28.json").write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
mn = lambda c: "".join(ch for ch in c.split()[0] if not ch.isdigit())
print("★ случаев", len(out), Counter(o["kind"] for o in out).most_common())
print("★ пары мнемоник", Counter((mn(o["ref"]), mn(o["ours"] or "?")) for o in out).most_common(25))
print("★ та же мнемоника:", sum(mn(o["ref"]) == mn(o["ours"] or "?") for o in out))
print("★ кат. нашей кривой с именем эталона", Counter(o["ours_cat"] for o in out).most_common())

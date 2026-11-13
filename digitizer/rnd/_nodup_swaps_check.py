r"""_nodup_swaps_check.py — РАЗБОР ПЕРЕСТАНОВОК ИМЁН §6.361 (13.11): для каждой перестановки модели стиля на раздвинутой паре
(`<stem>_namestyle_nodup.json` в `rp_lvl/<mode>`) — верна ли пара по эталону ДО перестановки (режим без имён, `--base`) и ПОСЛЕ
(`--mode`). Мера — `_sample_review.diagnose` (приёмка, plane2 в значениях). Плюс: P модели против исхода (AUC), доля «до верно».

  _nodup_swaps_check.py --mode NNS3 --base NDS3
"""
import sys, argparse, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import _sample_review as SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    TRACK = SR.load_tracks()
    SRC = {q.name: q for q in SR.ARCH.glob("*/wlg/*.nlgx")}
    byd = {SR.sheet_dir(nm): nm for nm in SRC}
    M, B = SR.TS / "rp_lvl" / a.mode, SR.TS / "rp_lvl" / a.base
    rows, cnt = [], Counter()
    for js in sorted(M.glob("*/*_namestyle_nodup.json")):
        nm = byd.get(js.parent.name)
        gm = next(iter(sorted(js.parent.glob("*_auto.nlgx"))), None)
        gb = next(iter(sorted((B / js.parent.name).glob("*_auto.nlgx"))), None)
        if nm is None or gm is None or gb is None:
            cnt["нет листа"] += 1; continue
        rb, _, _ = SR.diagnose(SRC[nm], gb, TRACK.get(nm, {}))
        rm, _, _ = SR.diagnose(SRC[nm], gm, TRACK.get(nm, {}))
        cb = {r["curve"]: r["cat"] for r in rb}; cm = {r["curve"]: r["cat"] for r in rm}
        for s in json.loads(js.read_text(encoding="utf-8")):
            A, Bn, p = s["A"], s["B"], s["p_order_ok"]
            ok_b = sum(cb.get(x) == "верно" for x in (A, Bn)); ok_m = sum(cm.get(x) == "верно" for x in (A, Bn))
            other_b = sum(cb.get(x) == "верно под другим именем" for x in (A, Bn))
            k = f"до {ok_b} → после {ok_m}"
            cnt[k] += 1
            rows.append(dict(sheet=nm, A=A, B=Bn, p=p, before=ok_b, after=ok_m, other_before=other_b,
                             cat_b=[cb.get(A), cb.get(Bn)], cat_m=[cm.get(A), cm.get(Bn)]))
    print(f"★ {a.mode} против {a.base}: перестановок {len(rows)}")
    for k, v in sorted(cnt.items()):
        print(f"  {k:24s} {v}")
    d = sum(r["after"] - r["before"] for r in rows)
    print(f"★ именных Δ на переставленных парах: {d:+d}")
    pre = Counter((tuple(r["cat_b"])) for r in rows)
    print("★ категории ДО (A, B), частые:")
    for k, v in pre.most_common(8):
        print(f"  {v:4d}  {k}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

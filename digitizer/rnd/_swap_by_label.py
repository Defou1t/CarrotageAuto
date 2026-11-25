r"""_swap_by_label.py — §6.388: имена пары ГК / НГК и МГЗ / МПЗ по подписи у начала кривых (облачное чтение) — стенд против NSE28.

Ответы агентов (`--items` JSON {items: [{file, curve, side}]}, side ∈ LEFT / RIGHT / not_found) по кадрам `output/header_labels`
(`index.tsv`: файл, лист, «начало кривых y=…», треки). Наши две кривые трека с этими именами (выдача NSE28 вне фолда) — средний x трассы
в строках начала … +400; разница < 10 px — не трогать. Порядок по ответу облака противоположен нашему ⇒ обмен имён.
Правило ответа (предрегистрация 25.11, до ответов): ОСНОВНОЕ — оба имени названы и стороны разные; ВТОРИЧНОЕ (отчёт) — названо одно
имя, сторона второго — противоположная. Мера — приёмка (`_sample_review.diagnose`): кривая эталона a верна после обмена, если на ней
честна наша b (`as` = b); верная до обмена — теряется. Безымянные — только в A/B через emit (обмен меняет уровни масштаба).

  _swap_by_label.py --items a.json [b.json] [--rule both|one] [--out swaps.json]   → сводка поле / вне / сорт A, знаковый тест по листам
"""
import sys, json, re, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
from math import comb
import numpy as np

TS = Path(r"F:/nds/output/taskS")
LBL = Path(r"F:/nds/output/header_labels")
SPAN, MIN_DX = 400, 10.0


def sign_p(up, dn):
    n = up + dn
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(up, dn) + 1)) / 2 ** n)


def sides(ans, a, b, rule):
    """ответы {имя: LEFT/RIGHT} → {a: side, b: side} или None"""
    sa, sb = ans.get(a), ans.get(b)
    if sa in ("LEFT", "RIGHT") and sb in ("LEFT", "RIGHT"):
        return {a: sa, b: sb} if sa != sb else None
    if rule == "one":
        flip = {"LEFT": "RIGHT", "RIGHT": "LEFT"}
        if sa in flip:
            return {a: sa, b: flip[sa]}
        if sb in flip:
            return {a: flip[sb], b: sb}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--rule", default="both", choices=["both", "one"])
    ap.add_argument("--mode", default="NSE28")
    ap.add_argument("--out", default="")
    ap.add_argument("--win", default="crop", choices=["crop", "own"], help="окно x: от начала кривых кадра / от общего начала наших двух трасс")
    a = ap.parse_args()
    import _sample_review as SR
    SR.MODE = a.mode; SR.OOF = TS / "rp_lvl" / a.mode
    idx = {}
    for l in (LBL / "index.tsv").read_text(encoding="utf-8").splitlines():
        f = l.split("\t")
        if len(f) >= 3:
            m = re.search(r"y=(\d+)", f[2])
            idx[f[0]] = (f[1], int(m.group(1)) if m else None)
    ANS = {}
    for itf in a.items:
        js = json.load(open(itf, encoding="utf-8"))
        for it in (js["items"] if isinstance(js, dict) else js):
            if it["file"] in idx:
                ANS.setdefault(idx[it["file"]][0], {})[it["curve"]] = it["side"]
    top = {v[0]: v[1] for v in idx.values()}
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    TRACK = SR.load_tracks(); S = SR.eval_sets()
    part_of = lambda nm: next((p for p, ss in S.items() if nm in ss), None)
    agg = {p: dict(sheets=0, base=0, sw=0, gain=0, loss=0, up=0, dn=0, flat=0, noans=0) for p in ("поле", "вне", "сорт A")}
    log = []
    for nm, ans in sorted(ANS.items()):
        p = part_of(nm)
        got = next(iter(sorted((SR.OOF / SR.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
        if p is None or got is None or nm not in SRC or top.get(nm) is None:
            continue
        A = agg[p]; A["sheets"] += 1
        recs, extra, cnt = SR.diagnose(SRC[nm], got, TRACK.get(nm, {}))
        R = {r["curve"]: r for r in recs}
        A["base"] += sum(r["cat"] == "верно" for r in recs)
        _, _, _, wr = SR.curves(SRC[nm], got)
        names = [n for n in ans if n in wr and wr[n]]
        y0 = top[nm]
        mx = {}
        for n in names:
            o = [m for m in names if m != n and (m.split() + [""])[1] == (n.split() + [""])[1]]
            s0 = y0 if a.win == "crop" else max([min(wr[n])] + [min(wr[m]) for m in o])
            xs = [wr[n][y] for y in range(s0, s0 + SPAN) if y in wr[n]]
            if len(xs) >= 50:
                mx[n] = float(np.mean(xs))
        by = {}
        for n in mx:
            by.setdefault((n.split() + [""])[1], []).append(n)
        g_ = l_ = 0
        for tk, L in by.items():
            if len(L) != 2:
                continue
            x0, y0n = L
            sd = sides(ans, x0, y0n, a.rule)
            if sd is None:
                A["noans"] += 1; continue
            if abs(mx[x0] - mx[y0n]) < MIN_DX:
                A["flat"] += 1; continue
            ours_left = x0 if mx[x0] < mx[y0n] else y0n
            hdr_left = x0 if sd[x0] == "LEFT" else y0n
            if ours_left == hdr_left:
                continue
            A["sw"] += 1
            for x, y in ((x0, y0n), (y0n, x0)):
                if x not in R:
                    continue
                was = R[x]["cat"] == "верно"; now = R[x].get("as") == y
                g_ += (not was) and now; l_ += was and not now
            log.append(dict(sheet=nm, part=p, a=x0, b=y0n, mx=[round(mx[x0]), round(mx[y0n])], sides=sd))
        A["gain"] += g_; A["loss"] += l_; A["up"] += (g_ - l_) > 0; A["dn"] += (g_ - l_) < 0
    if a.out:
        Path(a.out).write_text(json.dumps(log, ensure_ascii=False, indent=0), encoding="utf-8")
    fo = {k: agg["поле"][k] + agg["вне"][k] for k in agg["поле"]}
    for nmp, A in (("поле+вне", fo), ("поле", agg["поле"]), ("вне", agg["вне"]), ("сорт A", agg["сорт A"])):
        print(f"★ [{a.rule}] {nmp}: листов {A['sheets']}, именных верных {a.mode} {A['base']}; обменов {A['sw']} (без ответа {A['noans']},"
              f" x ближе 10 px {A['flat']}); стало верными {A['gain']}, перестало {A['loss']}; Δ {A['gain'] - A['loss']:+d}"
              f" (листов ↑{A['up']} / ↓{A['dn']}, p = {sign_p(A['up'], A['dn']):.4f})")


if __name__ == "__main__":
    main()

r"""_value_order_stand.py — §6.367 СТЕНД: перестановка имён пары по ПОРЯДКУ ЗНАЧЕНИЙ после выбора трассы (14.11).

Порядок значений пары кривых одного трека по эталону (`_pair_value_order.py`): Т2 > Т1 в 97% пар (обратного 0%), ГК > НГК 78% / 7%,
ТАУ > ЗАТ 85% / 7%, BKZ_ГЗ меньший номер ниже 82% / 3%… Прод не знает порядка (имена по стилю корпуса — до выбора трассы, §6.299).
Стенд: для пары наших кривых одного трека (выдача `--mode`, NSE26) — доля строк f, где значение кривой под именем A выше кривой под именем
B (значения — наши уровни и шкалы рамки под нынешним именем). Статистика пар семейств (сколько пар эталона с f ≥ 0.9 / ≤ 0.1) — по ДРУГОЙ
части листов (поле ↔ вне; сорт A — по полю). Если f ≥ 0.9, правдоподобие «имена верны» ∝ n_hi, «переставлены» ∝ n_lo (и наоборот при
f ≤ 0.1); перестановка, если (n_lo + 0.5) / (n_hi + 0.5) > R. Исход по эталону — мера приёмки (`_sample_review`): именных до / после.

  _value_order_stand.py [--mode NSE26] [--R 10 20 40]
"""
import sys, argparse, json, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
from collections import defaultdict
import numpy as np
import _sample_review as SR
from extract_nlgx import extract
from _multi_replica_probe import dense
from auto import meta as M
import decode_levels as DL

TS = Path(r"F:/nds/output/taskS")


def vals(tr, lv, ch):
    out = {}
    for y, x in tr.items():
        s = ch[min(int(lv.get(y, 0)), len(ch) - 1)]
        w = (s["x_right"] - s["x_left"]) or 1.0
        out[y] = s["v_left"] + (x - s["x_left"]) * (s["v_right"] - s["v_left"]) / w
    return out


def frac(va, vb):
    com = [y for y in va if y in vb]
    return (float(np.mean([va[y] > vb[y] for y in com])), len(com)) if len(com) >= 200 else (None, len(com))


def pairs_of(names, tm):
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = sorted([names[i], names[j]])
            if tm.get(a) is not None and tm.get(a) == tm.get(b) and a.split()[1] == b.split()[1]:
                out.append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="NSE26")
    ap.add_argument("--R", nargs="+", type=float, default=[10, 20, 40])
    a = ap.parse_args()
    TR = SR.load_tracks(); S = SR.eval_sets(); SRC = {q.name: q for q in SR.ARCH.glob("*/wlg/*.nlgx")}
    OOF = TS / "rp_lvl" / a.mode
    stats = {p: defaultdict(lambda: [0, 0, 0]) for p in S}         # часть → (fa, fb) → [n_hi, n_lo, n]
    cand = []                                                      # (часть, лист, A, B, f_our, fa, fb)
    for part, SS in S.items():
        for nm in sorted(SS):
            if nm not in SRC:
                continue
            got = next(iter(sorted((OOF / SR.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
            try:
                mt = extract(str(SRC[nm]))
            except Exception:
                continue
            tm = TR.get(nm, {})
            TC = {c["name"]: c for c in mt["curves"] if M.mnem_root(c["name"]) != "DA"}
            CH = {n: DL.build_family(mt, c) for n, c in TC.items()}
            VT = {}
            for A, B in pairs_of(list(TC), tm):
                if not CH[A] or not CH[B]:
                    continue
                for n in (A, B):
                    if n not in VT:
                        d = dense(TC[n]); VT[n] = vals(d, SR.lv_rows(TC[n]), CH[n]) if len(d) >= 200 else None
                if VT[A] is None or VT[B] is None:
                    continue
                f, _ = frac(VT[A], VT[B])
                if f is None:
                    continue
                k = (M.mnem_root(A), M.mnem_root(B)); st = stats[part][k]
                st[0] += f >= 0.9; st[1] += f <= 0.1; st[2] += 1
            if got is None:
                continue
            mw = extract(str(got))
            WC = {c["name"]: c for c in mw["curves"] if M.mnem_root(c["name"]) != "DA"}
            for A, B in pairs_of([n for n in WC if n in TC], tm):
                if not CH.get(A) or not CH.get(B):
                    continue
                da, db = dense(WC[A]), dense(WC[B])
                if len(da) < 200 or len(db) < 200:
                    continue
                f, _ = frac(vals(da, SR.lv_rows(WC[A]), CH[A]), vals(db, SR.lv_rows(WC[B]), CH[B]))
                if f is not None and (f >= 0.9 or f <= 0.1):
                    cand.append((part, nm, A, B, f, M.mnem_root(A), M.mnem_root(B)))
    other = {"поле": "вне", "вне": "поле", "сорт A": "поле"}
    res = {}
    for R in a.R:
        sw = []
        for part, nm, A, B, f, fa, fb in cand:
            n_hi, n_lo, n = stats[other[part]].get((fa, fb), [0, 0, 0])
            lr = (n_lo + 0.5) / (n_hi + 0.5) if f >= 0.9 else (n_hi + 0.5) / (n_lo + 0.5)
            if lr > R:
                sw.append((part, nm, A, B, round(lr, 1)))
        res[R] = sw
    # исход по эталону для всех предложенных перестановок (объединение по R)
    need = sorted({(p, nm) for sw in res.values() for p, nm, *_ in sw})
    outc = {}
    for part, nm in need:
        got = next(iter(sorted((OOF / SR.sheet_dir(nm)).glob("*_auto.nlgx"))))
        SR._TREES.clear()
        mt, mw, gts, wr = SR.curves(SRC[nm], got)
        LG = {c["name"]: SR.lv_rows(c) for c in mt["curves"] if c["name"] in gts}
        CHg = {c["name"]: DL.build_family(mt, c) for c in mt["curves"] if c["name"] in gts}
        LW = {c["name"]: SR.lv_rows(c) for c in mw["curves"] if c["name"] in wr}
        hon = lambda w, g: int(g in gts and w in wr and SR.HON(*SR.err(SR.relevel(wr[w], LW.get(w, {}), LG.get(g, {}), CHg.get(g, [])), gts[g])))
        for p_, n_, A, B, _ in [x for sw in res.values() for x in sw if x[1] == nm]:
            if (nm, A, B) not in outc:
                outc[(nm, A, B)] = (hon(A, A) + hon(B, B), hon(A, B) + hon(B, A))
    print(f"★ {a.mode}: пар-кандидатов (|f − 0.5| ≥ 0.4) {len(cand)}")
    for R, sw in res.items():
        L = []
        for part in ("поле", "вне", "сорт A"):
            ss = [x for x in sw if x[0] == part]
            d = sum(outc[(nm, A, B)][1] - outc[(nm, A, B)][0] for _, nm, A, B, _ in ss)
            up = sum(outc[(nm, A, B)][1] > outc[(nm, A, B)][0] for _, nm, A, B, _ in ss)
            dn = sum(outc[(nm, A, B)][1] < outc[(nm, A, B)][0] for _, nm, A, B, _ in ss)
            L.append(f"{part}: перестановок {len(ss)}, Δ именных {d:+d} (↑{up}/↓{dn})")
        print(f"  R > {R:g}: " + "; ".join(L))
        fams = defaultdict(lambda: [0, 0])
        for _, nm, A, B, _ in sw:
            k = (M.mnem_root(A), M.mnem_root(B)); o = outc[(nm, A, B)]
            fams[k][0] += 1; fams[k][1] += o[1] - o[0]
        print("     по семействам: " + ", ".join(f"{k[0]}–{k[1]} {v[0]} ({v[1]:+d})" for k, v in sorted(fams.items(), key=lambda kv: -kv[1][0])[:10]))
    (TS / "value_order").mkdir(exist_ok=True)
    (TS / "value_order" / "stand.json").write_text(json.dumps({str(R): sw for R, sw in res.items()}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

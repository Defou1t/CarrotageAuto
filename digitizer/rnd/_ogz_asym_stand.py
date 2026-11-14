r"""_ogz_asym_stand.py — §6.368: ОГЗ ↔ ГЗ ПО АСИММЕТРИИ ФОРМЫ КРИВОЙ (14.11).

Разбор обменов §6.366 (скептик): обращённый градиент-зонд (ОГЗ) и ГЗ различаются асимметрией аномалий по глубине — отношение суммы крупных
(верхние 5% по модулю) шагов log ρ через k = 25 строк вниз к сумме крупных шагов вверх: у ОГЗ ≥ 1.6, у ГЗ ≤ 0.93 внутри пары (4 из 4).
Признак формы, не уровня значений (порядок значений §6.367 ⛔ — он ломался на неверных кривых).
  truth — по эталону (2373 листа): в треках с ОГЗ и ≥ 1 ГЗ — доля треков, где у ОГЗ асимметрия наибольшая (и с запасом m над лучшей ГЗ).
  stand — выдача `--mode` (NSE26): в треке с нашими ОГЗ и ГЗ, если у какой-то ГЗ асимметрия больше, чем у ОГЗ, на m — обмен трасс ОГЗ ↔ эта
          ГЗ; исход по эталону (мера приёмки): именных до / после, по частям.

  _ogz_asym_stand.py [--mode NSE26] [--m 0 0.3 0.6 1.0]
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


def asym(c, ch, k=25, q=95):
    """отношение крупных шагов log10 ρ вниз / вверх по глубине (строки — вниз), без стыков уровней; NaN, если мало строк"""
    tr = dense(c)
    if len(tr) < 1000 or not ch:
        return np.nan
    lv = SR.lv_rows(c)
    y0, y1 = min(tr), max(tr)
    V = np.full(y1 - y0 + 1, np.nan); K = np.zeros(y1 - y0 + 1, int)
    for y, x in tr.items():
        L = int(lv.get(y, 0)); s = ch[min(L, len(ch) - 1)]
        w = (s["x_right"] - s["x_left"]) or 1.0
        V[y - y0] = s["v_left"] + (x - s["x_left"]) * (s["v_right"] - s["v_left"]) / w; K[y - y0] = L
    v = np.log10(np.clip(V, 0.3, None))
    st = v[k:] - v[:-k]; same = K[k:] == K[:-k]
    idx = np.arange(len(st)); m = np.isfinite(st) & same & (idx > 300) & (idx < len(st) - 300)
    st = st[m]
    if len(st) < 500:
        return np.nan
    big = np.percentile(np.abs(st), q)
    pos = st[st > big].sum(); neg = -st[st < -big].sum()
    return float(pos / max(neg, 1e-9))


def is_ogz(n):
    return re.sub(r"^BKZ_", "", n.split()[0]).startswith("OGZ")


def is_gz(n):
    return re.sub(r"^BKZ_", "", n.split()[0]).startswith("GZ")


def tracks(names, tm):
    by = defaultdict(list)
    for n in names:
        if tm.get(n) is not None and (is_ogz(n) or is_gz(n)):
            by[(tm[n], n.split()[1])].append(n)
    return [v for v in by.values() if any(is_ogz(n) for n in v) and any(is_gz(n) for n in v)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="NSE26")
    ap.add_argument("--m", nargs="+", type=float, default=[0, 0.3, 0.6, 1.0])
    a = ap.parse_args()
    TRK = SR.load_tracks(); S = SR.eval_sets(); SRC = {q.name: q for q in SR.ARCH.glob("*/wlg/*.nlgx")}
    OOF = TS / "rp_lvl" / a.mode
    tru = []                      # (часть, асимметрия ОГЗ, лучшая ГЗ)
    props = []                    # (часть, лист, ОГЗ, ГЗ, a_ogz, a_gz)
    for part, SS in S.items():
        for nm in sorted(SS):
            if nm not in SRC:
                continue
            try:
                mt = extract(str(SRC[nm]))
            except Exception:
                continue
            tm = TRK.get(nm, {})
            TC = {c["name"]: c for c in mt["curves"] if M.mnem_root(c["name"]) != "DA"}
            for grp in tracks(list(TC), tm):
                A = {n: asym(TC[n], DL.build_family(mt, TC[n])) for n in grp}
                og = [n for n in grp if is_ogz(n) and np.isfinite(A[n])]; gz = [n for n in grp if is_gz(n) and np.isfinite(A[n])]
                if og and gz:
                    tru.append((part, A[og[0]], max(A[g] for g in gz)))
            got = next(iter(sorted((OOF / SR.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
            if got is None:
                continue
            mw = extract(str(got))
            WC = {c["name"]: c for c in mw["curves"] if M.mnem_root(c["name"]) != "DA" and c["name"] in TC}
            for grp in tracks(list(WC), tm):
                A = {n: asym(WC[n], DL.build_family(mt, TC[n])) for n in grp}
                og = [n for n in grp if is_ogz(n) and np.isfinite(A[n])]; gz = [n for n in grp if is_gz(n) and np.isfinite(A[n])]
                if og and gz:
                    g = max(gz, key=lambda n: A[n])
                    props.append((part, nm, og[0], g, A[og[0]], A[g]))
    T = np.array([(o, g) for _, o, g in tru], float)
    print(f"★ эталон: треков с ОГЗ и ГЗ {len(T)}; асимметрия ОГЗ медиана {np.median(T[:, 0]):.2f}, лучшей ГЗ {np.median(T[:, 1]):.2f}; "
          f"ОГЗ > лучшей ГЗ: {np.mean(T[:, 0] > T[:, 1]):.1%}; с запасом 0.3: {np.mean(T[:, 0] > T[:, 1] + 0.3):.1%}")
    for part in S:
        t = np.array([(o, g) for p, o, g in tru if p == part], float)
        if len(t):
            print(f"   {part}: {len(t)} треков, ОГЗ > лучшей ГЗ {np.mean(t[:, 0] > t[:, 1]):.1%}")
    # исход перестановок
    need = [p for p in props if p[5] > p[4] + min(a.m)]
    outc = {}
    for part, nm, og, g, ao, ag in need:
        got = next(iter(sorted((OOF / SR.sheet_dir(nm)).glob("*_auto.nlgx"))))
        SR._TREES.clear()
        mt, mw, gts, wr = SR.curves(SRC[nm], got)
        LG = {c["name"]: SR.lv_rows(c) for c in mt["curves"] if c["name"] in gts}
        CHg = {c["name"]: DL.build_family(mt, c) for c in mt["curves"] if c["name"] in gts}
        LW = {c["name"]: SR.lv_rows(c) for c in mw["curves"] if c["name"] in wr}
        hon = lambda w, gg: int(gg in gts and w in wr and SR.HON(*SR.err(SR.relevel(wr[w], LW.get(w, {}), LG.get(gg, {}), CHg.get(gg, [])), gts[gg])))
        outc[(nm, og, g)] = (hon(og, og) + hon(g, g), hon(og, g) + hon(g, og))
    print(f"★ стенд {a.mode}: треков с нашими ОГЗ и ГЗ {len(props)}")
    res = {}
    for m in a.m:
        L = []
        for part in S:
            ss = [p for p in props if p[0] == part and p[5] > p[4] + m]
            d = sum(outc[(p[1], p[2], p[3])][1] - outc[(p[1], p[2], p[3])][0] for p in ss)
            up = sum(outc[(p[1], p[2], p[3])][1] > outc[(p[1], p[2], p[3])][0] for p in ss)
            dn = sum(outc[(p[1], p[2], p[3])][1] < outc[(p[1], p[2], p[3])][0] for p in ss)
            L.append(f"{part}: обменов {len(ss)}, Δ {d:+d} (↑{up}/↓{dn})")
        print(f"  запас {m}: " + "; ".join(L))
        res[m] = [list(p) for p in props if p[5] > p[4] + m]
    (TS / "ogz_asym").mkdir(exist_ok=True)
    (TS / "ogz_asym" / "stand.json").write_text(json.dumps({"truth": tru, "props": props, "outc": {"|".join(k): v for k, v in outc.items()}},
                                                           ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

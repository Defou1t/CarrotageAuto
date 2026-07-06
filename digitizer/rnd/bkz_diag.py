r"""
bkz_diag.py — BASELINE-ДИАГНОСТИКА трека BKZ (06.07). Локализует, ГДЕ рвётся автономная
оцифровка слипшихся/перевынесённых чёрных градиент-зондов, ПО ЧИСЛАМ на эталоне
(Yatskivska_001 + Pn_Zavoda_001). Отвечает на step-1 вопрос постановки.

★ ВЕРДИКТ (durable — КОРРЕКТИРУЕТ премису «N-канальный сепаратор слипшихся чёрных GZ»):
BKZ — это НЕ same-color overlap (как MK), а ДЕКОД ПЕРЕВЫНОСОВ + ТРЕКИНГ в основном
РАЗДЕЛЬНЫХ нитей. Три подзадачи, замерены:

1. ТРЕКИНГ (auto/ pipeline) = ПЕРВИЧНЫЙ ПРОВАЛ (~0%). auto/ детектит 1 трек / 12 линий,
   но 5 «AUTO» = синие грид/ось (спурьёзные), ВСЕ чёрные кривые → FLAG bunched_crossing.
   Эмитит трассы, но vs GT медиана 60–915px (GZ21 219, GZ31 60, GZ11 66, SP1 375, MDS1 915),
   <15px в 0–3% строк. Оверлей: авто-трассы почти ВЕРТИКАЛЬНЫЕ (центры банд/грид-колонки),
   реальные кривые гуляют ±сотни px. (воспроизвести: python -m auto.pipeline <img> --frame
   <nlgx> --out DIR ; затем tracking_gate() ниже.)

2. ПЕРЕВЫНОС-ДЕКОД (decode_levels на ИДЕАЛЬНОЙ экспертной x) = ВТОРИЧНЫЙ, половинчатый.
   fam=2 (одиночный 5×): mean 0.84 / med 0.99 (55/73 чисто).
   fam≥3 (второй перевынос 25×): mean 0.48 / med 0.55 (26/33 ПРОВАЛ); λ-тюнинг НЕ лечит
   (выше λ — хуже) → структурная неоднозначность перекрытых шкал. fam≥3 = 31% overrun-кривых.
   ⇒ даже при идеальном трекинге 25×-зонды не декодятся текущим value-continuity DP.

3. ИДЕНТИЧНОСТЬ (GZ21 vs GZ31 и т.п.) = НЕ ПРОВАЛ. Пары расходятся на 100–150px (медиана),
   на одной шкале <15px лишь 1–13% строк → позиционно-разделимы (НЕ MK-стена свопов).
   Все каротажные кривые ЧЁРНЫЕ (medRGB 20–75, max<110); цвет-кода нет; синяя только ось DA1.

⇒ ПОРЯДОК АТАКИ: трекинг (первично) → 25× level-decode (вторично). НЕ идентичность.

  python bkz_diag.py [--census] [--levels] [--identity]   (по умолчанию всё)
"""
import sys, glob, os, re
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # digitizer на путь
from extract_nlgx import extract, NULL
import decode_levels as DL

ETALONS = ["F:/nds/projects/Archive/Yatskivska_001",
           "F:/nds/projects/Archive/Pn_Zavoda_001"]


def bkz_plates():
    files = []
    for w in ETALONS:
        files += sorted(glob.glob(w + "/wlg/*_BKZ_*.nlgx"))
    return [f for f in files if "_auto" not in f]


def curve_rows(c):
    ty = c["top_y"]
    return {ty + i: x for i, x in enumerate(c["xs"]) if x != NULL}


def curve_levels(c):
    """dict row->level из сегментов (тег 35498)."""
    out = {}
    for s, e, l in c["segments"]:
        for y in range(s, e + 1):
            out[y] = l
    return out


def census():
    """Перепись эталонных BKZ: кривые, x-диапазоны, число сегментов по уровням, цвет."""
    print("=== ПЕРЕПИСЬ УРОВНЕЙ (сегментов по level, все эталонные BKZ) ===")
    tot = {}
    for p in bkz_plates():
        m = extract(p)
        for c in m["curves"]:
            nm = c["name"].split()[0]
            if nm == "DA1":
                continue
            tot.setdefault(nm, [0, 0, 0, 0])
            tot[nm][0] += len(c["segments"])
            for _, _, l in c["segments"]:
                if l < 3:
                    tot[nm][1 + l] += 1
    for nm, (s, l0, l1, l2) in sorted(tot.items()):
        flag = "  <- перевынос-зонд" if (l1 + l2) > 0 else ""
        print(f"  {nm:<6} segs={s:<4} L0={l0:<4} L1={l1:<4} L2={l2:<3}{flag}")


def level_decode_gate(lam=0.7):
    """decode_levels на ЭКСПЕРТНОЙ x → level-acc vs GT (тег 35498), разбивка по размеру семейства.
    Изолирует КАЧЕСТВО ДЕКОДЕРА при идеальном трекинге."""
    print(f"\n=== ПЕРЕВЫНОС-ДЕКОД (decode_levels на экспертной x, lam={lam}) ===")
    by_fam = {2: [], 3: [], 4: []}
    for p in bkz_plates():
        m = extract(p)
        for c in m["curves"]:
            if c["name"].split()[0] == "DA1":
                continue
            fam = DL.build_family(m, c)
            gl = curve_levels(c)
            gmax = max(gl.values()) if gl else 0
            xs = curve_rows(c)
            if len(fam) < 2 or gmax == 0 or len(xs) < 100:
                continue
            dec = DL.decode(xs, fam, lam=lam)
            common = [y for y in xs if y in gl]
            acc = np.mean([dec[y] == gl[y] for y in common]) if common else np.nan
            by_fam.setdefault(len(fam), []).append(acc)
    alla = [a for v in by_fam.values() for a in v]
    for K in sorted(by_fam):
        a = np.array(by_fam[K])
        if len(a):
            print(f"  fam={K}: n={len(a):<3} med={np.nanmedian(a):.2f} mean={np.nanmean(a):.2f} "
                  f"провал(<.8)={int((a < .8).sum())}")
    if alla:
        a = np.array(alla)
        print(f"  ВСЕ: n={len(a)} med={np.nanmedian(a):.2f} mean={np.nanmean(a):.2f} "
              f"чисто(>=.95)={int((a >= .95).sum())} провал(<.8)={int((a < .8).sum())}")


def identity_closeness():
    """Расхождение слипающихся пар (D1: GZ21/GZ31; D2: GZ41/OGZ1) по всем уровням —
    проверка «стена идентичности или разделимо»."""
    print("\n=== ИДЕНТИЧНОСТЬ (расхождение пар, все уровни) ===")
    pairs = {"D1": ("GZ21", "GZ31"), "D2": ("GZ41", "OGZ1")}
    for p in bkz_plates():
        side = "D2" if "_D2" in p else "D1"
        a, b = pairs[side]
        m = extract(p)
        cv = {c["name"].split()[0]: c for c in m["curves"]}
        if a not in cv or b not in cv:
            continue
        A = curve_rows(cv[a]); B = curve_rows(cv[b])
        common = sorted(set(A) & set(B))
        if len(common) < 50:
            continue
        d = np.array([abs(A[y] - B[y]) for y in common])
        plate = os.path.basename(p).split("_1_")[1].replace("_200", "").replace(".nlgx", "")
        print(f"  {plate:<22} {a}-{b}: n={len(common):<5} |Δ|med={np.median(d):>4.0f}px "
              f"<15px={np.mean(d < 15) * 100:>3.0f}% <30px={np.mean(d < 30) * 100:>3.0f}%")


def tracking_gate(auto_nlgx, gt_nlgx):
    """px-ошибка авто-трасс vs GT по общим строкам (запусти auto/ pipeline заранее)."""
    g = extract(gt_nlgx); a = extract(auto_nlgx)

    def rows(m):
        return {c["name"].split()[0]: curve_rows(c) for c in m["curves"]}
    G, A = rows(g), rows(a)
    print(f"\n=== ТРЕКИНГ auto/ vs GT: {os.path.basename(gt_nlgx)} ===")
    for nm in G:
        if nm == "DA1" or nm not in A or not A[nm]:
            continue
        common = sorted(set(G[nm]) & set(A[nm]))
        if not common:
            continue
        err = np.array([abs(G[nm][y] - A[nm][y]) for y in common])
        print(f"  {nm:<6} common={len(common):<5} px_err med={np.median(err):>4.0f} "
              f"<15px={np.mean(err < 15) * 100:.0f}%")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    do_all = not any(a in args for a in ("--census", "--levels", "--identity"))
    if do_all or "--census" in args:
        census()
    if do_all or "--levels" in args:
        level_decode_gate()
    if do_all or "--identity" in args:
        identity_closeness()


if __name__ == "__main__":
    main()

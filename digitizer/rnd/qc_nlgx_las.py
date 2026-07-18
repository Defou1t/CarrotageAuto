r"""qc_nlgx_las.py — QC АРХИВА: где ЭКСПЕРТНЫЙ nlgx расходится с эталонным LAS.

Зачем (находка 18.07): corr эксперт↔LAS гуляет 0.58-0.97, а на BEZLUD_051 MBK отношение
медиан 6.28× ≈ РОВНО ОДИН УРОВЕНЬ ×5 — то есть в архиве встречаются кривые, смещённые на
целый перевынос (ошибка разметки уровня ЛИБО LAS от другого замера/обработки). Такие кривые
портят любую метрику value-фиделити и отравляют обучение.

Что делает: по каждой резистивной кривой с матчем в LAS считает
  corr0        — corr(log) реконструкции эксперта с LAS как есть;
  corr_best/shift — лучшая corr при сдвиге глубины ±5 м (ловит депт-оффсеты);
  ratio        — отношение медиан (эксперт/LAS);
  вердикт      — LEVEL x5^k (ratio кратно 5 → смещение на уровень), SHIFT (чинится сдвигом),
                 SCATTER (расходятся не кратно), OK.

python qc_nlgx_las.py [--out report.csv] [--mnem MBK,BK,...] [--limit N]
"""
import sys, csv, math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_nlgx import extract, NULL, depth_of
import dataset as ds
from dataset_build import find_las
from decode_levels import build_family, gt_levels, mnem, reconstruct, RESIST

ARCHIVE = Path(r"F:\nds\projects\Archive")


def corr_at_shift(dep, val, lasd, lasv, shift):
    d = dep + shift
    m = (lasd >= d.min()) & (lasd <= d.max()) & (lasv > 0)
    if m.sum() < 30:
        return None
    ri = np.interp(lasd[m], d, val)
    ok = ri > 0
    if ok.sum() < 30:
        return None
    a = np.log(ri[ok]); b = np.log(lasv[m][ok])
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def verdict(corr0, corr_best, shift, ratio):
    if ratio and ratio > 0:
        l5 = math.log(ratio) / math.log(5)
        if abs(l5 - round(l5)) < 0.22 and abs(round(l5)) >= 1:
            return f"LEVEL x5^{round(l5):+d}"
    if corr0 is not None and corr_best is not None and corr_best - corr0 > 0.06 and abs(shift) > 0.05:
        return f"SHIFT {shift:+.2f}m"
    if corr0 is not None and corr0 < 0.80:
        return "SCATTER"
    return "OK"


def scan(mnems, limit):
    rows = []
    done = 0
    for wlg in sorted(ARCHIVE.glob("*/wlg")):
        if done >= limit:
            break
        for f in sorted(wlg.glob("*.nlgx")):
            if "_auto" in f.stem or done >= limit:
                continue
            las = find_las(f)
            if not las:
                continue
            try:
                m = extract(str(f))
                cols, arr = ds.load_las(las)
            except Exception:
                continue
            if arr.size == 0:
                continue
            try:
                matches = ds.match_las(m, cols, arr)
            except Exception:
                continue
            for c in ds.real_curves(m):
                mn = mnem(c["name"])
                if mn not in mnems:
                    continue
                fam = build_family(m, c); gl = gt_levels(c)
                if len(fam) < 1 or not gl:
                    continue
                ci = matches.get(c["name"], {}).get("col_idx")
                if ci is None:
                    continue
                ty = c["top_y"]
                xs = {ty + i: x for i, x in enumerate(c["xs"]) if x != NULL}
                if len(xs) < 100:
                    continue
                rec = reconstruct(xs, gl, fam)
                ys = sorted(rec)
                dep = np.array([depth_of(m, y) for y in ys])
                val = np.array([abs(rec[y]) for y in ys])
                ok = np.isfinite(dep) & np.isfinite(val)
                dep, val = dep[ok], val[ok]
                if len(dep) < 50:
                    continue
                s = np.argsort(dep); dep, val = dep[s], val[s]
                lasd = arr[:, 0]; lasv = np.abs(arr[:, ci])
                c0 = corr_at_shift(dep, val, lasd, lasv, 0.0)
                if c0 is None:
                    continue
                best = (c0, 0.0)
                for sh in np.arange(-5, 5.01, 0.25):
                    cc = corr_at_shift(dep, val, lasd, lasv, float(sh))
                    if cc is not None and cc > best[0]:
                        best = (cc, float(sh))
                lm = (lasd >= dep.min()) & (lasd <= dep.max()) & (lasv > 0)
                if lm.sum() < 30:
                    continue
                ratio = float(np.median(val)) / max(1e-9, float(np.median(lasv[lm])))
                rows.append(dict(well=wlg.parent.name, file=f.stem, curve=c["name"].split()[0],
                                 mnem=mn, corr0=round(c0, 3), corr_best=round(best[0], 3),
                                 shift=best[1], ratio=round(ratio, 2),
                                 v_nlgx=round(float(np.median(val)), 2),
                                 v_las=round(float(np.median(lasv[lm])), 2),
                                 verdict=verdict(c0, best[0], best[1], ratio)))
                done += 1
    return rows


def main():
    a = sys.argv[1:]
    out = a[a.index("--out") + 1] if "--out" in a else r"F:\nds\output\track4_x1x5\qc_nlgx_las.csv"
    mn = (a[a.index("--mnem") + 1].split(",") if "--mnem" in a else sorted(RESIST))
    limit = int(a[a.index("--limit") + 1]) if "--limit" in a else 10 ** 6
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    rows = scan(set(mn), limit)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["well", "file", "curve", "mnem", "corr0", "corr_best", "shift",
                            "ratio", "v_nlgx", "v_las", "verdict"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    cnt = Counter(r["verdict"].split()[0] for r in rows)
    print(f"кривых проверено: {len(rows)}  -> {out}")
    print("вердикты:", dict(cnt))
    bad = [r for r in rows if not r["verdict"].startswith("OK")]
    bad.sort(key=lambda r: r["corr0"])
    print(f"\nПОДОЗРИТЕЛЬНЫЕ ({len(bad)}), худшие 25 по corr:")
    print(f"{'скважина':<15}{'кривая':<7}{'corr':>6}{'ratio':>7}  {'вердикт':<14} файл")
    for r in bad[:25]:
        print(f"{r['well'][:14]:<15}{r['curve'][:6]:<7}{r['corr0']:>6.2f}{r['ratio']:>7.2f}  "
              f"{r['verdict']:<14} {r['file'][:44]}")


if __name__ == "__main__":
    main()

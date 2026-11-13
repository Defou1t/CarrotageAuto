r"""_final_dups.py — СПИСОК ДУБЛЕЙ ДЛЯ ПРОВЕРКИ (13.11, §6.364): пары кривых выдачи одного листа, чьи трассы совпадают (≤ 3 px) на доле
строк ≥ 0.3 (любой из двух). По выборкам 10.11 / 13.11 это главная ошибка прода (12 из 18, 14 из 23): одна кривая пары легла на линию
другой; автоматически «кому какая» не решается (§6.359–6.363), поэтому — флаг для ручной проверки. Бывают и настоящие совпадения кривых.

Пишет `<out>/../ДУБЛИ_ДЛЯ_ПРОВЕРКИ.tsv` (лист, кривая A, кривая B, доля строк A, доля строк B, общих строк, глубины совпадения).

  _final_dups.py --out F:/nds/output/final_2026-10/FINAL [--sheets список]
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL, depth_of
from auto import meta as M


def arr(c):
    ys = np.array([c["top_y"] + i for i, x in enumerate(c["xs"]) if x != NULL], np.int64)
    xs = np.array([x for x in c["xs"] if x != NULL], float)
    return ys, xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheets", default="")
    ap.add_argument("--theta", type=float, default=0.3)
    ap.add_argument("--tsv", default="", help="куда писать (по умолчанию <out>/../ДУБЛИ_ДЛЯ_ПРОВЕРКИ.tsv)")
    a = ap.parse_args()
    out = Path(a.out)
    keep = None
    if a.sheets:
        keep = {Path(l.strip()).stem for l in Path(a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()}
    rows = []; ns = nd = 0
    for d in sorted(out.iterdir()):
        g = next(iter(sorted(d.glob("*_auto.nlgx"))), None) if d.is_dir() else None
        if g is None:
            continue
        stem = g.name[:-len("_auto.nlgx")]
        if keep is not None and stem not in keep:
            continue
        ns += 1
        try:
            m = extract(str(g))
        except Exception:
            continue
        C = [(c["name"], arr(c)) for c in m["curves"] if M.mnem_root(c["name"]) != "DA"]
        C = [(n, a_) for n, a_ in C if len(a_[0]) >= 50]
        hit = False
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                (na, (ya, xa)), (nb, (yb, xb)) = C[i], C[j]
                cm, ia, ib = np.intersect1d(ya, yb, assume_unique=True, return_indices=True)
                if not len(cm):
                    continue
                close = np.abs(xa[ia] - xb[ib]) <= 3.0
                k = int(close.sum())
                fa, fb = k / len(ya), k / len(yb)
                if max(fa, fb) < a.theta:
                    continue
                rr = cm[close]
                try:
                    d0, d1 = depth_of(m, int(rr.min())), depth_of(m, int(rr.max()))
                    dep = f"{d0:.1f}–{d1:.1f}" if np.isfinite(d0) and np.isfinite(d1) else f"строки {int(rr.min())}–{int(rr.max())}"
                except Exception:
                    dep = f"строки {int(rr.min())}–{int(rr.max())}"
                rows.append((stem + ".nlgx", na, nb, f"{fa:.2f}", f"{fb:.2f}", str(k), dep)); hit = True
        nd += hit
    f = Path(a.tsv) if a.tsv else out.parent / "ДУБЛИ_ДЛЯ_ПРОВЕРКИ.tsv"
    hdr = ("лист", "кривая A", "кривая B", "доля строк A", "доля строк B", "общих строк", "глубины совпадения, м")
    tmp = f.with_suffix(".tmp")
    tmp.write_text("\n".join("\t".join(r) for r in [hdr] + rows) + "\n", encoding="utf-8"); tmp.replace(f)
    print(f"★ дубли: листов {ns}, с дублем {nd}, пар {len(rows)} → {f}")


if __name__ == "__main__":
    main()

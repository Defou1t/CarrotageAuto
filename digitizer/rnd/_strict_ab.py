r"""_strict_ab.py — A/B СТРОГОЙ МЕРЫ ДВУХ ПОВТОРОВ (03.11, §6.355).

Берёт два итога `_strict_stand.py --mode <режим> --tag <тег>` (`taskS/_strict_<тег>.json`) и по общим листам считает чистых (верна
приёмкой И лишнее ≤ 1% / ≤ 5% строк выдачи И пропуск ≤ того же), лишние строки и пропуски; знаковый тест по листам (Δ чистых листа).

  _strict_ab.py <тег A> <тег B>
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from math import comb

TS = Path(r"F:/nds/output/taskS")


def sign_p(up, dn):
    n = up + dn
    if n == 0:
        return 1.0
    k = min(up, dn)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def per_sheet(path):
    out = {}
    for nm, part, rows, _ in json.loads(Path(path).read_text(encoding="utf-8")):
        v = [0, 0, 0, 0, 0, 0]                          # честных, чистых 1%, чистых 5%, лишних, строк выдачи, пропусков
        for rec in rows:
            s = rec["base"]
            if not s.get("n_o"):
                continue
            v[3] += s["off_o"]; v[4] += s["n_o"]; v[5] += s["off_t"]
            if s["hon"]:
                v[0] += 1
                v[1] += s["off_o"] <= 0.01 * s["n_o"] and s["off_t"] <= 0.01 * s["n_t"]
                v[2] += s["off_o"] <= 0.05 * s["n_o"] and s["off_t"] <= 0.05 * s["n_t"]
        out[nm] = (part, v)
    return out


def main():
    ta, tb = sys.argv[1], sys.argv[2]
    A, B = per_sheet(TS / f"_strict_{ta}.json"), per_sheet(TS / f"_strict_{tb}.json")
    com = sorted(set(A) & set(B))
    print(f"★ строгая мера {ta} → {tb}: общих листов {len(com)} (только в {ta} {len(set(A) - set(B))}, только в {tb} {len(set(B) - set(A))})")
    for pk in ("поле+вне", "сорт A"):
        nn = [nm for nm in com if (A[nm][0] == "сорт A") == (pk == "сорт A")]
        sa = [sum(A[nm][1][i] for nm in nn) for i in range(6)]; sb = [sum(B[nm][1][i] for nm in nn) for i in range(6)]
        print(f"## {pk} ({len(nn)} листов)")
        for i, lab in ((0, "верных приёмкой"), (1, "чистых ≤ 1%"), (2, "чистых ≤ 5%")):
            up = sum(B[nm][1][i] > A[nm][1][i] for nm in nn); dn = sum(B[nm][1][i] < A[nm][1][i] for nm in nn)
            print(f"  {lab:16s}: {sa[i]} → {sb[i]} ({sb[i] - sa[i]:+d}, листов ↑{up}/↓{dn}, p = {sign_p(up, dn):.4f})")
        print(f"  лишних строк    : {sa[3]} → {sb[3]} ({(sb[3] - sa[3]) / max(sa[3], 1):+.1%}); пропусков {sa[5]} → {sb[5]} "
              f"({(sb[5] - sa[5]) / max(sa[5], 1):+.1%})")


if __name__ == "__main__":
    main()

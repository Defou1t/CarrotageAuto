r"""_series_siblings.py — §6.366 РАЗВЕДКА: соседи по серии для раздвинутых пар (13.11).

Серия — листы одной скважины (папка архива) с одной датой записи в имени (YYYY.MM.DD / YYYY_MM_DD / YYYY-MM-DD). Для каждой пары скана
§6.362 (`taskS/nodup_selfstyle/pairs_NDS3.jsonl`, X — ушедший слот, Y — оставшийся) — есть ли в серии другой лист, в рамке которого есть оба
имени (корень + номер, напр. GZ41 / GZ51 — без DA/SA). Это верхняя граница того, сколько пар может получить «стиль имени» с соседнего листа.

  _series_siblings.py
"""
import sys, json, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
from collections import defaultdict
from extract_nlgx import extract
from auto import meta as M

TS = Path(r"F:/nds/output/taskS")
ARCH = Path(r"F:\nds\projects\Archive")
DATE = re.compile(r"(19|20)\d\d[._-]\d\d[._-]\d\d")


def series_key(q):
    m = DATE.search(q.stem)
    return (q.parent.parent.name, re.sub(r"[._-]", ".", m.group(0))) if m else None


def main():
    SRC = {q.name: q for q in ARCH.glob("*/wlg/*.nlgx")}
    ser = defaultdict(list)
    for nm, q in SRC.items():
        k = series_key(q)
        if k:
            ser[k].append(nm)
    names = {}

    def nm_of(sheet):
        if sheet not in names:
            try:
                names[sheet] = {c["name"].split()[0] for c in extract(str(SRC[sheet]))["curves"] if M.mnem_root(c["name"]) != "DA"}
            except Exception:
                names[sheet] = set()
        return names[sheet]
    P = []
    for l in (TS / "nodup_selfstyle" / "pairs_NDS3.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        for p in r["pairs"]:
            P.append((r["sheet"], p))
    n = has_ser = has_sib = 0; dec = dec_sib = 0
    for sh, p in P:
        n += 1
        k = series_key(SRC[sh]) if sh in SRC else None
        if not k or len(ser[k]) < 2:
            continue
        has_ser += 1
        a, b = p["X"].split()[0], p["Y"].split()[0]
        sib = [s for s in ser[k] if s != sh and a in nm_of(s) and b in nm_of(s)]
        d = p["cur"] != p["swp"]
        dec += d
        if sib:
            has_sib += 1; dec_sib += d
    print(f"★ пар {n}: лист в серии (≥ 2 листов с той же датой в скважине) — {has_ser}; есть сосед с обоими именами — {has_sib} "
          f"({has_sib / max(n, 1):.0%}); решающих (эталон различает раскладки) среди них {dec_sib} из {sum(p['cur'] != p['swp'] for _, p in P)}")


if __name__ == "__main__":
    main()

"""СЕЛЕКЦИЯ ТУШИ, шаг 3: вывести из архива правило «что цифруют в ФАЙЛЕ с таким токеном/частью».
Гипотеза 19.07: раз один бланк режется на несколько nlgx, набор кривых файла предсказуем по
имени (токен + часть D_11/D_12/...). Если так — expected_curves даёт ТОЧНЫЙ список слотов, и
задача сводится к сопоставлению K слотов с K из N физических кривых.

Считаем по архиву: токен(+часть) → какие наборы мнемоник реально лежат в nlgx, насколько
стабильно (доля самого частого набора) и что об этом думает mnemonics.json::filename_hints.

  python _token_curveset.py [--min 3] [--json out.json]
"""
import sys, argparse, json
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
from collections import Counter, defaultdict
from extract_nlgx import extract, NULL
from dataset_build import find_image
from auto import meta as M

MN = r"F:\nds\Auto\mnemonics.json"
ARCHIVE = Path(r"F:\nds\projects\Archive")
ap = argparse.ArgumentParser()
ap.add_argument("--min", type=int, default=3, help="печатать токены, встреченные не реже")
ap.add_argument("--by-part", action="store_true", help="различать части (D_11 vs D_12)")
ap.add_argument("--json", default="")
a = ap.parse_args()

hints = M.load_mnemonics(MN).get("filename_hints", {})
sets = defaultdict(Counter)
for wlg in sorted(ARCHIVE.glob("*/wlg")):
    for n in sorted(wlg.glob("*.nlgx")):
        if "_auto" in n.stem or not find_image(n):
            continue
        m = M.parse_filename(n.name, MN)
        if not m.curves_token:
            continue
        try:
            mo = extract(str(n))
        except Exception:
            continue
        # Корень берём ЧЕРЕЗ СЛОВАРЬ (curve_info), а не mnem_root: тот срезает ВЕСЬ хвост цифр
        # и «GZ41» (зонд 4, экземпляр 1) превращается в «GZ» — номер зонда теряется, а именно он
        # и есть содержание правила («BKZ, DS» = зонды 4 и 5, а не «какие-то GZ»).
        roots = Counter(M.curve_info(c["name"], MN)["root"] for c in mo["curves"]
                        if sum(1 for x in c["xs"] if x != NULL) >= 100
                        and M.mnem_root(c["name"]) != "DA")
        cs = [f"{r}×{k}" if k > 1 else r for r, k in sorted(roots.items())]
        if not cs:
            continue
        key = f"{m.curves_token}|{m.part}" if a.by_part else m.curves_token
        sets[key][",".join(cs)] += 1

rows = []
print(f"{'токен':<20} {'n':>4} {'стаб':>5}  самый частый набор   |  сейчас в filename_hints")
for key, c in sorted(sets.items(), key=lambda kv: -sum(kv[1].values())):
    tot = sum(c.values())
    if tot < a.min:
        continue
    top, cnt = c.most_common(1)[0]
    tok = key.split("|")[0]
    exp = M.expected_curves(tok, MN)
    same = sorted(set(top.split(","))) == sorted(set(exp))
    print(f"{key:<20} {tot:>4} {100*cnt//tot:>4}%  {top:<38} | {','.join(exp) or '—'}"
          + ("" if same else "   ⚠ РАСХОДИТСЯ"))
    rows.append({"token": key, "n": tot, "stability": cnt / tot, "top_set": top.split(","),
                 "variants": c.most_common(4), "hints_now": exp, "match": same})

if rows:
    st = [r["stability"] for r in rows]
    print(f"\nтокенов (n>={a.min}): {len(rows)}; медианная стабильность набора "
          f"{100*sorted(st)[len(st)//2]:.0f}%; совпадает с filename_hints: "
          f"{sum(1 for r in rows if r['match'])}/{len(rows)}")
if a.json:
    Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("json →", a.json)

"""РЕАЛЬНАЯ разделимость мульти-листов: не перекрытие x-диапазонов, а ПЕРЕСЕЧЕНИЯ по строкам.
Канон §6.6.11 «идентичность = СЧЁТ»: если пара кривых НИГДЕ не меняет порядок слева-направо,
разбор по счёту законен даже при полном перекрытии диапазонов.

Классы пары (на общих строках): band (диапазоны врозь) / ordered (порядок стабилен, <1% строк
меньшинства и <=2 смены знака) / crossing (реальные пересечения). Класс листа = худшая пара.
python crossings.py [--max-curves 6] [--json out.json]"""
import sys, argparse, json
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image
from auto import meta as M

MN = r"F:\nds\Auto\mnemonics.json"
ARCHIVE = Path(r"F:\nds\projects\Archive")
ap = argparse.ArgumentParser()
ap.add_argument("--max-curves", type=int, default=6)
ap.add_argument("--json", default="")
a = ap.parse_args()

MINORITY = 0.01     # доля строк, где порядок обратный
MAX_FLIPS = 2       # смен знака (устойчивых, после сглаживания)


def series(c):
    return {c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL}


def pair_class(sa, sb):
    ks = sorted(set(sa) & set(sb))
    if len(ks) < 100:
        return "sparse", 0.0, 0
    d = np.array([sa[k] - sb[k] for k in ks], float)
    # диапазоны врозь -> band
    if (d > 0).all() or (d < 0).all():
        pos = float((d > 0).mean())
        # стабильный порядок; band vs ordered решаем по перекрытию диапазонов ниже
        return "ordered", min(pos, 1 - pos), 0
    minority = float(min((d > 0).mean(), (d < 0).mean()))
    sgn = np.sign(d)
    sm = np.convolve(sgn, np.ones(51) / 51, "same")      # сгладить дрожь у нуля
    flips = int((np.diff(np.sign(sm)) != 0).sum())
    if minority < MINORITY and flips <= MAX_FLIPS:
        return "ordered", minority, flips
    return "crossing", minority, flips


cls_c, tok_cls = Counter(), defaultdict(Counter)
rows = []
for wlg in sorted(ARCHIVE.glob("*/wlg")):
    for n in sorted(wlg.glob("*.nlgx")):
        if "_auto" in n.stem:
            continue
        m = M.parse_filename(n.name, MN)
        if not find_image(n):
            continue
        try:
            mo = extract(str(n))
        except Exception:
            continue
        cs = [c for c in ds.real_curves(mo) if sum(1 for x in c["xs"] if x != NULL) >= 100]
        if not (2 <= len(cs) <= a.max_curves):
            continue
        ser = {c["name"]: series(c) for c in cs}
        col = {c["name"]: M.curve_info(c["name"], MN)["color"] for c in cs}
        worst, pairs = "ordered", []
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                ni, nj = cs[i]["name"], cs[j]["name"]
                k, mi, fl = pair_class(ser[ni], ser[nj])
                if k == "sparse":
                    continue
                pairs.append((ni.split()[0], nj.split()[0], k, round(mi, 4), fl))
                if k == "crossing":
                    ci, cj = col[ni], col[nj]
                    worst = "crossing_color" if (ci and cj and ci != cj) and worst != "crossing" else "crossing"
        cls_c[worst] += 1
        tok_cls[m.curves_token][worst] += 1
        rows.append({"well": wlg.parent.name, "stem": n.stem, "tok": m.curves_token,
                     "n": len(cs), "cls": worst,
                     "names": [c["name"].split()[0] for c in cs], "pairs": pairs})

tot = sum(cls_c.values())
print(f"ЛИСТОВ (2..{a.max_curves} кривых, с картинкой): {tot}")
print("КЛАССЫ:", dict(cls_c))
print(f"  ordered (порядок стабилен ⇒ разбор по СЧЁТУ законен): {cls_c['ordered']} = {100*cls_c['ordered']/max(1,tot):.0f}%")
print("\nпо токенам (топ 20):")
for tok, c in sorted(tok_cls.items(), key=lambda kv: -sum(kv[1].values()))[:20]:
    t = sum(c.values())
    print(f"  {tok or '(пусто)':<18} всего {t:4d}  ordered {c['ordered']:4d}  crossing {c['crossing']:4d}  cross-цвет {c['crossing_color']:4d}")
print("\nпримеры ordered-листов:")
for r in [r for r in rows if r["cls"] == "ordered"][:15]:
    print(f"  {r['well']:<14} {r['tok']:<14} n={r['n']} {','.join(r['names'])}   {r['stem'][:42]}")
if a.json:
    Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("json →", a.json)

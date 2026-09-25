r"""_dec_kcap.py — ОГРАНИЧИВАЕТ ЛИ ДЕКОДЕР ЧИСЛО ЛИНИЙ U1 (§6.203 → B1b, кандидат 1).

ОТКУДА ВОПРОС. Декодер берёт у U1 число кривых трека `K = len(lines)` (`rowdec.py:362-371`, все
линии трека при `trace_flagged=True`) и выдаёт РОВНО K траекторий. Если U1 нашёл на треке меньше
линий, чем там кривых у эксперта, недостающие декодер не выдаст физически — какая бы тушь под ними
ни была. §6.194 знает, что U1 выдаёт линий ВДВОЕ больше, чем нужно, — но это в среднем; вопрос
о хвосте распределения: сколько кривых честного поля стоит на треках, где K_U1 < K_эталона.

ЧТО СЧИТАЕТ (по замороженным данным, счёта не тратит). K_U1 — из `<лист>_understanding.json`
выдачи декодера (тот же объект `sheet.lines`, из которого `rowdec` берёт K); K_эталона — число
эталонных кривых трека по `slotmap.pkl`. Кривые раскладываются по корзинам ΔK = K_U1 − K_эталона,
в каждой — сколько взял чистый декодер, нынешний прод и потолок V2-подобный «взял хоть кто-то из
(A, B)». Считается из выгрузки `u1_vs_dec.pkl` (§6.200), поэтому счёт тот же, что там.

  <ComfyUI>\python_embeded\python.exe _dec_kcap.py
"""
import sys, argparse, pickle, hashlib, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import defaultdict, Counter

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dump", default="u1_vs_dec.pkl")
ap.add_argument("--dec", default="ab_rdhonest/B")
ap.add_argument("--ceil", default="rowceil_*of6.pkl", help="потолки V2 по кривым (§6.185, `_row_ceiling.py`)")
a = ap.parse_args()
TS = Path(a.ts)

D = pickle.load(open(TS / a.dump, "rb"))
# ★ ПОТОЛОК V2 ПО КРИВЫМ (тушь прод-маски + 1:1, §6.185/§6.197): r[9] = med, r[8] = cov
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9
V2 = {}
for f in sorted(TS.glob(a.ceil)):
    for r in pickle.load(open(f, "rb"))["rows"]:
        V2[(r[0], r[1])] = HON(r[9], r[8])
print(f"потолков V2 прочитано: {len(V2)} кривых")
bytr = defaultdict(list)
for sh, t, nm, k, f in D:
    bytr[(sh, t)].append((nm, k, f))


def kU1(sh):
    stem = Path(sh).stem
    pd = TS / a.dec / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    u = next(iter(sorted(pd.glob("*_understanding.json"))), None) if pd.is_dir() else None
    if not u:
        return None
    L = json.loads(u.read_text(encoding="utf-8")).get("lines") or []
    c = Counter(int(x["track"]) for x in L)
    return c


tab = defaultdict(Counter)
miss = 0
cache = {}
for (sh, t), rows in bytr.items():
    if sh not in cache:
        cache[sh] = kU1(sh)
    c = cache[sh]
    if c is None:
        miss += 1
        continue
    ku, kg = c.get(t, 0), len(rows)
    dk = ku - kg
    b = ("ΔK ≤ −2" if dk <= -2 else "ΔK = −1" if dk == -1 else "ΔK = 0" if dk == 0 else
         "ΔK = +1…+2" if dk <= 2 else "ΔK ≥ +3")
    tab[b]["треков"] += 1
    for nm, k, f in rows:
        tab[b]["кривых"] += 1
        tab[b]["DEC"] += f["DEC"]; tab[b]["CUR"] += f["CUR"]; tab[b]["P965"] += f["P965"]
        tab[b]["P965∪DEC"] += f["P965∪DEC"]
        v2 = V2.get((sh, nm))
        tab[b]["V2 есть"] += v2 is not None
        tab[b]["V2"] += bool(v2)
        tab[b]["V2, никем"] += bool(v2) and not f["P965∪DEC"]

print(f"треков {sum(v['треков'] for v in tab.values())}, кривых {sum(v['кривых'] for v in tab.values())}, "
      f"треков без understanding.json {miss}")
print("\n★★ КРИВЫЕ ПО ΔK = K_U1 − K_эталона (K_U1 — все линии трека, как берёт rowdec)")
print("| ΔK | треков | кривых | DEC | CUR | P965 | P965∪DEC | DEC/кривых | потолок V2 | ★ V2, а не взял никто |")
print("|---|---|---|---|---|---|---|---|---|---|")
for b in ("ΔK ≤ −2", "ΔK = −1", "ΔK = 0", "ΔK = +1…+2", "ΔK ≥ +3"):
    v = tab[b]
    if not v["кривых"]:
        continue
    print(f"| {b} | {v['треков']} | {v['кривых']} | {v['DEC']} | **{v['CUR']}** | {v['P965']} | "
          f"{v['P965∪DEC']} | {100*v['DEC']/v['кривых']:.1f}% | {v['V2']} | **{v['V2, никем']}** |")
neg = sum(tab[b]["кривых"] for b in ("ΔK ≤ −2", "ΔK = −1"))
negd = sum(tab[b]["DEC"] for b in ("ΔK ≤ −2", "ΔK = −1"))
negc = sum(tab[b]["CUR"] for b in ("ΔK ≤ −2", "ΔK = −1"))
tot = sum(v["кривых"] for v in tab.values())
print(f"\n★ ТАМ, ГДЕ U1 НЕДОСЧИТАЛ ЛИНИЙ (ΔK < 0): {neg} кривых из {tot} ({100*neg/max(1,tot):.1f}%), "
      f"декодер взял {negd}, нынешний прод {negc}.")
print("  ⇒ Верхняя оценка цены K-ограничения: кривые на таких треках, не взятые никем, "
      f"= {neg - sum(tab[b]['P965∪DEC'] for b in ('ΔK ≤ −2', 'ΔK = −1'))}; "
      f"из них ПРОВОДИМЫХ по чернилам (V2) = {sum(tab[b]['V2, никем'] for b in ('ΔK ≤ −2', 'ΔK = −1'))} "
      f"— это и есть ПРИЗ.")
nv = sum(v["V2 есть"] for v in tab.values())
print(f"  стыковка с потолками V2: {nv} из {tot} кривых ({100*nv/max(1,tot):.1f}%)")

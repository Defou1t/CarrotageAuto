r"""_name_band.py — ПОЛОСА ШКАЛЫ СЛОТА КАК ОГРАНИЧЕНИЕ РАСКЛАДКИ: сколько имён теряется ПОПЕРЁК полос (21.09, B4).

ОТКУДА ВОПРОС. BEZLUD_051 (§6.214): шаблон делит трек на две оси глубины — `DA1` x 326…1800 и `DA2`
x 1801…2974; слот `IK1 DA1 SA2` обязан лежать в левой половине, `IKA1/IKR1 DA2 …` — в правой. Раскладка
отдала IK1 кривую с медианой x 1896 (правая), IKA1 — кривую с 1242 (левая). Полосы шкал в шаблоне ЕСТЬ
(`scale_axes[x_left, x_right]`), а признаков раскладки про них нет (`slot_model.rows`: 14 признаков —
цвет, класс, ранг x среди линий трека, форма; полосы слота — нет).

ЧТО СЧИТАЕТ (замороженная выдача G, растры не нужны):
  1. эталон: доля экспертных кривых, чья медиана x лежит в полосе СВОЕЙ шкалы (санитарная проверка:
     если это не ~100%, полоса — не закон);
  2. выдача: для каждого слота с кривой — внутри ли полосы медиана x выданной кривой; кросс-таблица
     с исходом имени (верно / перестановка в семействе / чужой корень) из `_name_gap.py`;
  3. сколько треков имеют ≥ 2 разных полос (только там ограничение что-то меняет).
"""
import sys, argparse, pickle, hashlib, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--rows", default="pick_learn_honest2.pkl.dump")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--dir", default="ab_pregate/G")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--pad", type=int, default=12)
a = ap.parse_args()
TS = Path(a.ts)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def sa_suffix(nm):
    m = re.search(r"(DA\d+\s+SA\d+)\s*$", nm)
    return m.group(1) if m else None


def st(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    return float(np.median([abs(tr[y] - gt[y]) for y in com])), len(com) / max(1, len(gt))


def match(rows, cols, ok):
    pair = {}

    def try_(r, seen):
        for c in cols:
            if not ok.get((r, c)) or c in seen:
                continue
            seen.add(c)
            if c not in pair or try_(pair[c], seen):
                pair[c] = r
                return True
        return False
    for r in rows:
        try_(r, set())
    return {r: c for c, r in pair.items()}


def read_out(root, sh):
    stem = Path(sh).stem
    pd = TS / root / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    if not got:
        return None
    return {c["name"]: dense(c) for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"}


trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q
sheets = sorted({r[0] for r in trk})

gt_in = Counter(); out_tab = defaultdict(Counter); trk_bands = Counter(); n_slots_band = 0
fixable = Counter()      # среди неверных имён: кривая слота вне полосы (ограничение сработало бы)
done = 0
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    W = read_out(a.dir, sh)
    if W is None:
        continue
    G = extract(str(q))
    ax = {str(s.get("name", "")).strip(): (int(s["x_left"]), int(s["x_right"])) for s in G.get("scale_axes", [])
          if s.get("x_left") is not None and s.get("x_right") is not None}
    if not ax:
        continue
    done += 1
    gts = {c["name"]: dense(c) for c in G["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    band = {}
    for nm in set(gts) | set(W):
        suf = sa_suffix(nm)
        if suf and suf in ax:
            band[nm] = ax[suf]
    # 1. эталон в своей полосе?
    for nm, d in gts.items():
        if nm in band and d:
            med = float(np.median(list(d.values()))); lo, hi = band[nm]
            gt_in["в полосе" if lo - a.pad <= med <= hi + a.pad else "ВНЕ полосы"] += 1
    tm = smap.get(sh, {})
    bytr = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None:
            bytr[t].append(nm)
    for t, ns in bytr.items():
        if (sh, t) not in KOF:
            continue
        bands_t = {band[n] for n in ns if n in band}
        trk_bands["≥2 полос" if len(bands_t) >= 2 else "1 полоса"] += 1
        Wt = [k for k in W if tm.get(k) == t]
        ok = {(g, k): HON(*st(W[k], gts[g])) for g in ns for k in Wt}
        mt = match(ns, Wt, ok)
        for g, k in mt.items():
            own = g in W and ok.get((g, g), False)
            if own:
                cat = "имя верно"
            elif M.mnem_root(k) == M.mnem_root(g):
                cat = "перестановка в семействе"
            else:
                cat = "чужой корень"
            # кривая, лежащая под именем g (если есть) — в полосе слота g?
            if g in W and W[g] and g in band:
                med = float(np.median(list(W[g].values()))); lo, hi = band[g]
                inb = lo - a.pad <= med <= hi + a.pad
                out_tab[cat]["кривая под именем в полосе" if inb else "кривая под именем ВНЕ полосы"] += 1
                if not own and not inb:
                    fixable[cat] += 1
            else:
                out_tab[cat]["под именем пусто/нет полосы"] += 1
    if si % 200 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

print(f"\nлистов с полосами шкал {done}; треков: {dict(trk_bands)}")
print(f"★ ЭТАЛОН в полосе своей шкалы (±{a.pad} px): {dict(gt_in)}")
print("\n★★ ВЫДАЧА G: исход имени × положение кривой под этим именем относительно полосы слота")
cols = ["кривая под именем в полосе", "кривая под именем ВНЕ полосы", "под именем пусто/нет полосы"]
print("| исход имени | " + " | ".join(cols) + " |")
print("|---|---|---|---|")
for cat in ("имя верно", "перестановка в семействе", "чужой корень"):
    print(f"| {cat} | " + " | ".join(str(out_tab[cat][c]) for c in cols) + " |")
print(f"\n⇒ неверных имён, где кривая под именем лежит ВНЕ полосы слота (ограничение полосой их запретило бы): {dict(fixable)}, всего {sum(fixable.values())}")

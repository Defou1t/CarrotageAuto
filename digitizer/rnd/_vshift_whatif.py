r"""_vshift_whatif.py — ПОДСТРОЧНЫЙ СДВИГ ВЫДАЧИ ПО ВЕРТИКАЛИ ПРОТИВ ЭТАЛОНА (03.10, к §6.257/§6.258).

`_label_offset.py`: центр штриха − эталон ≈ d · (dx/dy) с d ≈ −0.25 строки — одинаково во всех корзинах крутизны и по листам
(квартили −0.36…−0.15). Эталон в строке y показывает тушь строки y + 0.25: систематический сдвиг на четверть строки,
по-видимому соглашение «глубина → строка». Здесь — сколько кривых станет честными при НЫНЕШНЕЙ мере (3 px по строке,
медиана, 1:1), если выдачу прода читать со сдвигом δ: x'(y) = x(y + δ) — линейная интерполяция соседних строк трассы.
Изображения не нужны. Треки — как `_tolerance_whatif.py` (`slotmap.pkl` и пуловые дампы).

  _vshift_whatif.py --dir F:/nds/output/taskS/rp_vc --mode N --sheets holdoutA_sheets.txt --deltas -0.5 -0.25 0 0.25 0.5
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_vc")
ap.add_argument("--mode", default="N")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--sheets", default="wellmap_sheets.txt")
ap.add_argument("--deltas", type=float, nargs="+", default=[-0.5, -0.25, 0.0, 0.25, 0.5])
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--offset", type=int, default=0)
a = ap.parse_args()
TS = Path(a.ts)
smap = pickle.load(open(TS / a.map, "rb"))
POOLS = ["pools", "pools_gate", "pools_wide", "pools_more", "pools_div", "pools_heldout", "pools_all"]
for root in POOLS:
    for f in sorted((TS / root).glob("*.pkl")):
        if f.stem in smap:
            continue
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        smap.setdefault(d["name"], {s["name"]: s["track"] for s in d["slots"]})
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
sheets = [l.strip() for l in (TS / a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()][a.offset::a.every]


def shifted(tr, dl):
    """x'(y) = x(y + dl) — линейная интерполяция по соседней строке трассы (нет соседа — без сдвига)"""
    if dl == 0:
        return tr
    out = {}
    for y, x in tr.items():
        if dl > 0:
            n = tr.get(y + 1)
            out[y] = x + dl * (n - x) if n is not None else x
        else:
            n = tr.get(y - 1)
            out[y] = x + (-dl) * (n - x) if n is not None else x
    return out


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


C = Counter(); PER = {dl: [] for dl in a.deltas}
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    stem = q.stem
    pd = Path(a.dir) / a.mode / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
    if not got:
        continue
    W = {c["name"]: dense(c) for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"}
    G = {c["name"]: dense(c) for c in extract(str(q))["curves"]
         if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    cnt = {dl: 0 for dl in a.deltas}
    for t in {tm.get(g) for g in G if tm.get(g) is not None}:
        gs = [g for g in G if tm.get(g) == t]; ws = [k for k in W if tm.get(k) == t and W[k]]
        if t is None:
            continue
        C["кривых"] += len(gs)
        for dl in a.deltas:
            Ws = {k: shifted(W[k], dl) for k in ws}
            ok = {}
            for g in gs:
                for k in ws:
                    m, c = st(Ws[k], G[g])
                    ok[(g, k)] = m is not None and m <= 3.0 and c >= 0.9
            n = len(match(gs, ws, ok))
            C[dl] += n; cnt[dl] += n
    for dl in a.deltas:
        PER[dl].append(cnt[dl])
    if si % 200 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)
base = C[0.0] if 0.0 in a.deltas else None
print(f"★ {a.sheets} (каждый {a.every}-й, сдвиг {a.offset}): кривых {C['кривых']}")
for dl in a.deltas:
    extra = ""
    if base is not None and dl != 0.0:
        dlt = np.array(PER[dl]) - np.array(PER[0.0])
        nz = dlt[dlt != 0]
        # парный знаковый перестановочный по листам
        rng = np.random.default_rng(0)
        obs = dlt.sum()
        if len(nz):
            sims = (rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)
            pv = float(np.mean(np.abs(sims) >= abs(obs)))
        else:
            pv = 1.0
        extra = f" (Δ {obs:+d}, листов ↑{int((dlt > 0).sum())}/↓{int((dlt < 0).sum())}, p = {pv:.4f})"
    print(f"   δ = {dl:+.2f} строки: честных 1:1 при 3 px {C[dl]}{extra}")

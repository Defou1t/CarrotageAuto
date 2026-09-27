r"""_fg_coverage.py — ВИДИТ ЛИ МАСКА ТУШИ ПОТЕРЯННЫЕ КРИВЫЕ (§6.234, 27.09).

§6.228: у 1662 кривых (45.6%) честной трассы нет ни у одного пути; §6.216: эталон GK лежит на туши прод-бинаря лишь в 39%
вершин. Если бледную/пунктирную кривую маска не видит, её не проведёт никакой селектор (§6.220–§6.233 упёрлись в ~85%).
Для каждой эталонной кривой (кэш — есть ли честный кандидат; выдача прода — взята ли): доля точек эталона (каждая 5-я строка),
у которых в ±3 px есть тушь — (а) в прод-маске (`trace2d._color_fg(цвет кривой)` ∪ чёрный), (б) в чувствительной маске
(адаптивный порог по яркости, окно 31 px, C = 8, минус структура). Разрез: взятые / с кандидатом / без кандидата.

  _fg_coverage.py --cache F:/nds/output/taskS/tcache --dir rp_fill/NF --every 3
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from dataset_build import find_image
from auto import meta as M, imaging as im, trace2d as T2
from auto.config import DEFAULT, Config

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--dir", default="rp_fill/NF")
ap.add_argument("--every", type=int, default=3)
a = ap.parse_args()
TS = Path(a.ts); MN = Config().mnemonics
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def unpack(t):
    rows, xs = t
    return dict(zip(rows.tolist(), xs.tolist()))


def hon_raw(tr, gt, bridge=30):
    com = [y for y in gt if y in tr]
    if len(com) < 30 or np.median([abs(tr[y] - gt[y]) for y in com]) > 3.0:
        return False
    rs = np.array(sorted(tr)); ok = 0
    for y in gt:
        if y in tr:
            ok += 1; continue
        i = np.searchsorted(rs, y)
        if 0 < i < len(rs) and rs[i] - rs[i - 1] <= bridge:
            ok += 1
    return ok / len(gt) >= 0.9


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


def dil(m, r=3):
    o = m.copy()
    for s in range(1, r + 1):
        o[:, s:] |= m[:, :-s]; o[:, :-s] |= m[:, s:]
    return o


SRC = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx"):
    SRC.setdefault(q.stem, q)
R = defaultdict(list)
files = sorted(Path(a.cache).glob("*.pkl"))[::a.every]
for i, f in enumerate(files, 1):
    v = pickle.load(open(f, "rb"))
    q = SRC.get(Path(v["frame_nlgx"]).stem)
    if not q:
        continue
    img = find_image(q)
    if not img:
        continue
    G = {c["name"]: dense(c) for c in extract(str(q))["curves"]
         if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    if not G:
        continue
    cands = [unpack(t) for _, t in v["traces"]] + ([unpack(t) for _, t in v["alt"]] if v.get("alt") else [])
    pd = TS / a.dir / f.stem
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    W = {c["name"]: dense(c) for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"} if got else {}
    ok = {(g, k): HON(*st(W[k], G[g])) for g in G for k in W if W[k]}
    taken = set(match(list(G), [k for k in W if W[k]], ok))
    rgb = im.load_rgb(str(img))
    sm = im.structure_mask(rgb, DEFAULT.cv)
    black = T2._color_fg(rgb, "black", DEFAULT.cv)
    fgc = {}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ada = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 8).astype(bool) & ~sm
    adaD = dil(ada); Hh, Ww = black.shape
    for g, gt in G.items():
        col = M.curve_info(g, MN).get("color") or "black"
        if col not in fgc:
            try:
                fgc[col] = dil(T2._color_fg(rgb, col, DEFAULT.cv) | black)
            except Exception:
                fgc[col] = dil(black)
        pm = fgc[col]
        ys = [y for y in sorted(gt)[::5] if 0 <= y < Hh and 0 <= int(round(gt[y])) < Ww]
        if len(ys) < 20:
            continue
        cp = float(np.mean([pm[y, int(round(gt[y]))] for y in ys]))
        ca = float(np.mean([adaD[y, int(round(gt[y]))] for y in ys]))
        cls = "взята" if g in taken else ("не взята, кандидат есть" if any(hon_raw(t, gt) for t in cands) else "не взята, кандидата нет")
        R[cls].append((cp, ca, M.mnem_root(g)))
    del rgb, sm, black, fgc, gray, ada, adaD
    if i % 50 == 0:
        print(f"  … {i}/{len(files)}", file=sys.stderr)
print("★ ДОЛЯ ТОЧЕК ЭТАЛОНА С ТУШЬЮ В ±3 px (медиана по кривым; доля кривых с покрытием < 0.5):")
for cls in ("взята", "не взята, кандидат есть", "не взята, кандидата нет"):
    x = R[cls]
    if not x:
        continue
    cp = np.array([t[0] for t in x]); ca = np.array([t[1] for t in x])
    print(f"   {cls}: кривых {len(x)}; прод-маска — медиана {np.median(cp):.2f}, < 0.5 у {100*np.mean(cp < 0.5):.0f}%; "
          f"чувствительная — медиана {np.median(ca):.2f}, < 0.5 у {100*np.mean(ca < 0.5):.0f}%; "
          f"прод < 0.5, а чувствительная ≥ 0.8 — {int(np.sum((cp < 0.5) & (ca >= 0.8)))}")
x = R["не взята, кандидата нет"]
fam = defaultdict(list)
for cp, ca, rt in x:
    fam[rt].append(cp)
print("   без кандидата по семействам (кривых, медиана прод-покрытия): " +
      ", ".join(f"{k} {len(v)} ({np.median(v):.2f})" for k, v in sorted(fam.items(), key=lambda kv: -len(kv[1]))[:12]))

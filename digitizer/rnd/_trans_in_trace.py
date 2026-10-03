r"""_trans_in_trace.py — ЕСТЬ ЛИ ПЕРЕХОД МАСШТАБА В ТРАССЕ ПРОДА (03.10, §6.262).

Для каждого перехода эталона (x до → x после, смена уровня) у той же кривой выдачи (`rp_vc/N`, то же имя): в окне ±R строк
есть ли в трассе прода скачок того же знака и не меньше половины скачка эксперта; и где трасса до и после — в 3 px от точек
эксперта до/после перехода. Классы:
  «скачок есть, по точкам» — трасса следует за пером через переход;
  «скачок есть, не туда» — скачок того же знака, но в чужую точку;
  «скачка нет» — трасса через переход не прыгает (осталась на одном пере / на упоре / ушла на другую линию).
И ложные скачки: скачки трассы (|dx| ≥ 30% ширины шкалы за ≤ 5 строк) вдали от переходов эталона — на 1000 строк.

  _trans_in_trace.py --every 3
"""
import sys, argparse, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from auto import meta as M
import decode_levels as DL

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_vc/N")
ap.add_argument("--every", type=int, default=3)
ap.add_argument("--r", type=int, default=25)
a = ap.parse_args()
TS = Path(a.ts)
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
C = Counter(); FAM = Counter(); FALSE = Counter()
for sh in (lst("wellmap_sheets.txt") + lst("holdoutA_sheets.txt"))[::a.every]:
    q = SRC.get(sh)
    if not q:
        continue
    stem = q.stem
    pd = Path(a.dir) / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
    if not got:
        continue
    mt = extract(str(q)); mw = {c["name"]: c for c in extract(str(got))["curves"]}
    for c in mt["curves"]:
        if M.mnem_root(c["name"]) == "DA" or c["name"] not in mw or len(c.get("segments") or []) < 2:
            continue
        fam = DL.build_family(mt, c)
        if len(fam) < 2:
            continue
        width = abs(fam[0]["x_right"] - fam[0]["x_left"]) or 1
        pts = sorted((c["top_y"] + i, float(x)) for i, x in enumerate(c["xs"]) if x != NULL)
        ys = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts])
        w = mw[c["name"]]
        wy = np.array([w["top_y"] + i for i, x in enumerate(w["xs"]) if x != NULL])
        wx = np.array([float(x) for x in w["xs"] if x != NULL])
        if len(wy) < 50:
            continue
        segs = sorted(c["segments"], key=lambda s: s[0])
        trows = []
        for s0, s1 in zip(segs, segs[1:]):
            if s0[2] == s1[2]:
                continue
            ia = np.searchsorted(ys, s0[1], side="right") - 1; ib = np.searchsorted(ys, s1[0])
            if ia < 0 or ib >= len(ys) or ys[ib] - ys[ia] > 40:
                continue
            ya, yb, xa, xb = int(ys[ia]), int(ys[ib]), xs[ia], xs[ib]
            trows.append(yb)
            dxe = xb - xa
            # трасса прода в окне: до перехода (ya − R … ya) и после (yb … yb + R)
            mb = (wy >= ya - a.r) & (wy <= ya); ma = (wy >= yb) & (wy <= yb + a.r)
            if not mb.any() or not ma.any():
                C["у трассы нет точек у перехода"] += 1; continue
            xb_t = wx[mb][-1]; xa_t = wx[ma][0]
            dxt = xa_t - xb_t
            if np.sign(dxt) == np.sign(dxe) and abs(dxt) >= 0.5 * abs(dxe):
                ok = abs(xb_t - xa) <= 6 and abs(xa_t - xb) <= 6
                k = "скачок есть, по точкам эксперта" if ok else "скачок есть, не туда"
            else:
                k = "скачка нет"
            C[k] += 1; FAM[(M.mnem_root(c["name"]), k)] += 1
        # ложные скачки трассы прода вдали от переходов эталона
        d = np.diff(wx); dy = np.diff(wy)
        big = np.flatnonzero((np.abs(d) >= 0.3 * width) & (dy <= 5))
        tr = np.array(trows) if trows else np.array([-10 ** 9])
        for i in big:
            if np.min(np.abs(tr - wy[i + 1])) > 60:
                FALSE["ложных скачков"] += 1
        FALSE["строк трассы"] += len(wy)
tot = sum(v for k, v in C.items() if k != "у трассы нет точек у перехода")
print(f"★ переходов эталона (каждый {a.every}-й лист): {tot} + у трассы нет точек рядом {C['у трассы нет точек у перехода']}")
for k in ("скачок есть, по точкам эксперта", "скачок есть, не туда", "скачка нет"):
    print(f"   {k}: {C[k]} ({100 * C[k] / max(1, tot):.0f}%)")
print(f"   ложных скачков трассы (≥ 30% ширины, вдали от переходов): {FALSE['ложных скачков']} на {FALSE['строк трассы']} строк "
      f"= {1000 * FALSE['ложных скачков'] / max(1, FALSE['строк трассы']):.2f} на 1000 строк")
roots = Counter(k[0] for k in FAM.elements())
for r, n in roots.most_common(10):
    print(f"   {r:8s} {n:5d}: по точкам {FAM[(r, 'скачок есть, по точкам эксперта')] / n:.0%}, не туда {FAM[(r, 'скачок есть, не туда')] / n:.0%}, "
          f"нет {FAM[(r, 'скачка нет')] / n:.0%}")

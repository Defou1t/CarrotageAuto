r"""_dec_see.py — ВИДИТ ЛИ КАРТА ДЕКОДЕРА КРИВЫЕ, КОТОРЫХ НЕ ПРОВЁЛ НИ ОДИН ПУТЬ (§6.237, 27.09).

§6.234/§6.236: у 1662 кривых нет честного кандидата; прод-маска видит 3/4 из них, объединение всех трасс держит лишь 62%
строк. Декодер (`rowdec.trace_track`) ведёт K линий ЖАДНО по очереди (Витерби по пикам, взятые пики исключаются, пропуска
строки нет). Если его карта вероятности ВИДИТ потерянную кривую (есть пик в 3 px от эталона на большинстве строк), предел —
в ВЫБОРЕ путей (совместное K-декодирование), а не в зрении. Для кривых без кандидата (и для взятых — опора): карта трека
пересчитывается тем же чекпойнтом, что в проде (скважина по `rowdec_wellmap.json` → фолд), и мерится доля строк эталона
(каждая 3-я), где пик строки в 3 px от эталона — при пороге прода 0.6·max строки, при 0.3 и 0.1.

  _dec_see.py --every 9
"""
import sys, argparse, pickle, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M, imaging as im, rowdec as RD
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--dir", default="rp_fill/NF")
ap.add_argument("--every", type=int, default=9)
ap.add_argument("--dump", default=r"F:/nds/output/taskS/dec_see.pkl")
a = ap.parse_args()
TS = Path(a.ts); P = DEFAULT.cv
WMAP = json.loads((TS / "rowdec_wellmap.json").read_text(encoding="utf-8"))
THRS = (0.6, 0.3, 0.1)
SRC = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx"):
    SRC.setdefault(q.stem, q)


def un(t):
    return dict(zip(t[0].tolist(), t[1].tolist()))


def hon(tr, gt, bridge=30):
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


def prob_map(rgb, track, ck, y0, y1):
    """Карта вероятности трека — ТОТ ЖЕ проход окнами, что `rowdec.trace_track` (копия, прод не тронут)."""
    import torch
    net, dev = RD._load(ck)
    x0, x1 = int(track.x_left), int(track.x_right) + 1
    band = RD._band(rgb, P, x0, x1)
    H, Wb = band.shape
    y0 = max(0, int(y0)); y1 = min(H, int(y1))
    prob = np.zeros((y1 - y0, Wb), np.float32)
    STEP, OV, WIN = 512, 64, 512
    xcuts = [(c, min(Wb, c + WIN)) for c in range(0, max(1, Wb - 1), WIN - 64)]
    for (cx0, cx1) in xcuts:
        for gy in range(y0, y1, STEP - 2 * OV):
            gy2 = min(y1, gy + STEP)
            if gy2 - gy < 32:
                continue
            sub = band[gy:gy2, cx0:cx1].astype(np.float32) / 255.0
            with torch.no_grad():
                pp, _ = net(torch.from_numpy(sub)[None, None].to(dev))
                pp = torch.sigmoid(pp)[0, 0].float().cpu().numpy()
            v0 = gy + (OV if gy > y0 else 0); v1 = gy2 - (OV if gy2 < y1 else 0)
            wx0 = cx0 + (32 if cx0 > 0 else 0); wx1 = cx1 - (32 if cx1 < Wb else 0)
            if wx1 <= wx0:
                wx0, wx1 = cx0, cx1
            prob[v0 - y0:v1 - y0, wx0:wx1] = pp[v0 - gy:v1 - gy, wx0 - cx0:wx1 - cx0]
    return prob, x0, y0


REC = []
files = sorted(Path(a.cache).glob("*.pkl"))[::a.every]
for fi, f in enumerate(files, 1):
    v = pickle.load(open(f, "rb"))
    stem = Path(v["frame_nlgx"]).stem
    q = SRC.get(stem)
    if not q:
        continue
    w = WMAP.get(Path(v["image"]).stem) or WMAP.get(v["stem"]) or WMAP.get(stem)
    if w is None:
        continue
    fold = RD.fold_of_well(w)
    ck = f"rowdec_of5_f{0 if fold is None else fold}_s0.pt"
    G = {c["name"]: dense(c) for c in extract(str(q))["curves"]
         if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    T = [un(t) for _, t in v["traces"]] + ([un(t) for _, t in v["alt"]] if v.get("alt") else [])
    tracks = v["frame"].tracks
    todo = defaultdict(list)
    for g, gt in G.items():
        cls = "с кандидатом" if any(hon(t, gt) for t in T) else "без кандидата"
        gx = float(np.median(list(gt.values())))
        ti = next((i for i, tk in enumerate(tracks) if tk.x_left <= gx <= tk.x_right), None)
        if ti is not None:
            todo[ti].append((g, gt, cls))
    if not todo:
        continue
    rgb = im.load_rgb(v["image"])
    for ti, lst in todo.items():
        ys_all = [y for _, gt, _ in lst for y in gt]
        prob, x0, py0 = prob_map(rgb, tracks[ti], ck, min(ys_all), max(ys_all) + 1)
        rmax = prob.max(1)
        for g, gt, cls in lst:
            hit = {t: 0 for t in THRS}; n = 0; pv = []
            for y in sorted(gt)[::3]:
                i = y - py0
                if not (0 <= i < prob.shape[0]):
                    continue
                row = prob[i]; xi = int(round(gt[y])) - x0
                lo, hi = max(1, xi - 3), min(len(row) - 1, xi + 4)
                if hi <= lo:
                    continue
                n += 1
                seg = row[lo:hi]
                loc = (seg >= row[lo - 1:hi - 1]) & (seg >= row[lo + 1:hi + 1])
                pv.append(float(seg.max()) / max(1e-6, float(rmax[i])))
                for t in THRS:
                    hit[t] += bool((loc & (seg >= t * rmax[i])).any())
            if n >= 20:
                REC.append(dict(sheet=f.stem, g=g, root=M.mnem_root(g), cls=cls, n=n,
                                vis={t: hit[t] / n for t in THRS}, rel=float(np.median(pv)), k=len(lst)))
    del rgb
    if fi % 20 == 0:
        print(f"  … {fi}/{len(files)}", file=sys.stderr)
pickle.dump(REC, open(a.dump, "wb"))
print(f"★ ВИДИМОСТЬ КРИВЫХ ДЛЯ КАРТЫ ДЕКОДЕРА (каждый {a.every}-й лист кэша; доля строк эталона с пиком в 3 px):")
for cls in ("с кандидатом", "без кандидата"):
    R = [r for r in REC if r["cls"] == cls]
    if not R:
        continue
    print(f"   {cls}: кривых {len(R)}; отн. высота пика у эталона (к max строки) — медиана {np.median([r['rel'] for r in R]):.2f}")
    for t in THRS:
        x = np.array([r["vis"][t] for r in R])
        print(f"      порог {t}·max: медиана {np.median(x):.2f}; ≥ 0.9 у {100*np.mean(x >= 0.9):.0f}%, ≥ 0.7 у {100*np.mean(x >= 0.7):.0f}%, "
              f"< 0.3 у {100*np.mean(x < 0.3):.0f}%")

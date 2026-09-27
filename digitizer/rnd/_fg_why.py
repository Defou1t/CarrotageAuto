r"""_fg_why.py — ПОЧЕМУ ПРОД-МАСКА НЕ ВИДИТ КРИВУЮ, И ОТДЕЛИМА ЛИ БЛЕДНАЯ ТУШЬ ОТ ФОНА (§6.234, 27.09).

`_fg_coverage.py`: у кривых без честного кандидата прод-маска (`_color_fg(цвет) ∪ чёрный`) покрывает эталон в ±3 px
с медианой 0.77, у 32% — < 0.5 (GK — 0.26). Здесь по каждой кривой (каждый 3-й лист кэша, точки эталона через 5 строк):
  • покрытие прод-маской, `dark_mask` (V < dark_v) без вычета цветов, цветовыми каналами, структурой;
  • КОНТРАСТ: min V в ±3 px от эталона против медианы V строки в ±40 px (фон) — насколько тушь темнее бумаги;
  • КОНТРОЛЬ: те же маски на эталоне, сдвинутом на ±15 px (там кривой нет) — плотность маски «по месту»;
  • маски по относительной темноте `V < фон − D` (D = 20, 30, 45) минус структура: покрытие на эталоне и на контроле.
Мерит, возможна ли маска, видящая бледные кривые, без затопления листа (§6.137: мягкий передний план целиком дал −55).

  _fg_why.py --every 3
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
ap.add_argument("--dump", default=r"F:/nds/output/taskS/fg_why.pkl")
a = ap.parse_args()
TS = Path(a.ts); MN = Config().mnemonics; P = DEFAULT.cv
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9
DS = (20, 30, 45)


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
REC = []
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
    V = im.value_channel(rgb).astype(np.int16)
    Hh, Ww = V.shape
    sm = im.structure_mask(rgb, P)
    CC = im.color_channels(rgb, P)
    black = T2._color_fg(rgb, "black", P)
    dark = dil(im.dark_mask(rgb, P))
    ccD = {k: dil(m) for k, m in CC.items()}
    smD = dil(sm, 1)
    # фон строки: медиана V по окну 81 px вдоль строки (медианный фильтр 1×81 на прореженной копии быстрее — берём
    # по точкам ниже вручную)
    rel = {}
    fgc = {}
    for g, gt in G.items():
        col = M.curve_info(g, MN).get("color") or "black"
        if col not in fgc:
            try:
                fgc[col] = dil(T2._color_fg(rgb, col, P) | black)
            except Exception:
                fgc[col] = dil(black)
        pm = fgc[col]
        ys = [y for y in sorted(gt)[::5] if 0 <= y < Hh and 20 <= int(round(gt[y])) < Ww - 20]
        if len(ys) < 20:
            continue
        xs = [int(round(gt[y])) for y in ys]

        def cov(m, sh=0):
            return float(np.mean([m[y, min(Ww - 1, max(0, x + sh))] for y, x in zip(ys, xs)]))

        cp = cov(pm)
        # контраст и относительная темнота в точке эталона и на контроле ±15
        con, relhit, relctl = [], {d: [] for d in DS}, {d: [] for d in DS}
        for y, x in zip(ys, xs):
            row = V[y]
            bg = float(np.median(row[max(0, x - 40):min(Ww, x + 41)]))
            mn = float(row[max(0, x - 3):x + 4].min())
            con.append(bg - mn)
            for d in DS:
                seg = (row[max(0, x - 3):x + 4] < bg - d) & ~sm[y, max(0, x - 3):x + 4]
                relhit[d].append(bool(seg.any()))
                c2 = 0
                for sh in (-15, 15):
                    xc = x + sh
                    seg2 = (row[max(0, xc - 3):xc + 4] < bg - d) & ~sm[y, max(0, xc - 3):xc + 4]
                    c2 += bool(seg2.any())
                relctl[d].append(c2 / 2)
        cls = "взята" if g in taken else ("кандидат есть" if any(hon_raw(t, gt) for t in cands) else "кандидата нет")
        REC.append(dict(sheet=f.stem, g=g, root=M.mnem_root(g), col=col, cls=cls, cp=cp,
                        cp_ctl=(cov(pm, -15) + cov(pm, 15)) / 2,
                        dark=cov(dark), dark_ctl=(cov(dark, -15) + cov(dark, 15)) / 2,
                        cc={k: cov(m) for k, m in ccD.items()}, st=cov(smD),
                        con=float(np.median(con)),
                        rel={d: float(np.mean(relhit[d])) for d in DS}, rel_ctl={d: float(np.mean(relctl[d])) for d in DS}))
    del rgb, V, sm, CC, black, dark, ccD, smD, fgc
    if i % 50 == 0:
        print(f"  … {i}/{len(files)}", file=sys.stderr)
pickle.dump(REC, open(a.dump, "wb"))


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


print(f"★ КРИВЫХ {len(REC)} (каждый {a.every}-й лист кэша)")
for cls in ("взята", "кандидат есть", "кандидата нет"):
    R = [r for r in REC if r["cls"] == cls]
    lo = [r for r in R if r["cp"] < 0.5]
    print(f"\n{cls}: {len(R)}; прод-маска на эталоне {med([r['cp'] for r in R]):.2f}, на контроле ±15 {med([r['cp_ctl'] for r in R]):.2f}; "
          f"контраст туши (фон − min V) {med([r['con'] for r in R]):.0f}")
    for d in DS:
        print(f"   относительная темнота D={d}: эталон {med([r['rel'][d] for r in R]):.2f}, контроль {med([r['rel_ctl'][d] for r in R]):.2f}")
    if lo:
        print(f"   ПРОД НЕ ВИДИТ (покрытие < 0.5): {len(lo)} — контраст {med([r['con'] for r in lo]):.0f}; dark {med([r['dark'] for r in lo]):.2f} "
              f"(контроль {med([r['dark_ctl'] for r in lo]):.2f}); структура {med([r['st'] for r in lo]):.2f}; "
              + "; ".join(f"{k} {med([r['cc'][k] for r in lo]):.2f}" for k in lo[0]["cc"]))
        for d in DS:
            hit = sum(1 for r in lo if r["rel"][d] >= 0.8 and r["rel_ctl"][d] <= 0.4)
            print(f"     D={d}: эталон {med([r['rel'][d] for r in lo]):.2f} / контроль {med([r['rel_ctl'][d] for r in lo]):.2f}; "
                  f"видна (≥ 0.8) при чистом контроле (≤ 0.4) — {hit}")
        by = defaultdict(list)
        for r in lo:
            by[(r["root"], r["col"])].append(r)
        print("     по семействам (кривых, контраст, dark, цвет в разметке): " + ", ".join(
            f"{k[0]}/{k[1]} {len(x)} ({med([r['con'] for r in x]):.0f}, {med([r['dark'] for r in x]):.2f})"
            for k, x in sorted(by.items(), key=lambda kv: -len(kv[1]))[:10]))

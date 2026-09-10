r"""_u1_why.py — ПОЧЕМУ ЛИНИЯ U1 УХОДИТ НЕ ТУДА (§6.194 → первый этаж).

ОТКУДА ВОПРОС. §6.194 переставил адрес работ: трасса линии U1 лежит лишь на 43.2% оцифрованных
кривых, а четыре пятых потерь происходят ВЫШЕ декодера. Разложение показало две беды сопоставимого
размера — «не найдена вовсе» (25.0%) и «линия рядом, но трасса ушла» (31.8%). Вторая крупнее и
понятнее для разбора: линия НАЙДЕНА, значит вопрос не в детекции, а в ведении.

ЧТО ДЕЛАЕТ. То же разложение, что `_k34_why.py` сделал для декодера, но этажом выше — над трассами
U1 из пуловых дампов (`lines[*]['tr']`). По каждой эталонной кривой берётся ЛУЧШАЯ по med трасса
линии того же трека и классифицируется:
  ★ накрыта                 — med ≤ 3px и cov ≥ 0.9, кривая взята;
  не покрыта                — трасса короткая: cov < 0.9 при med ≤ 3px;
  ведёт СОСЕДНЮЮ кривую     — та же трасса честна для ДРУГОЙ кривой трека (дефект тождества);
  ушла в ЧУЖОЙ ТРЕК         — честна для кривой другого трека листа;
  СКЛЕЙКА по глубине        — идёт по кривым (≥80% строк ближе 3px), но меняет их;
  вне линий                 — ни то, ни другое.

⚠ «Вне линий» здесь означает «не совпало с эталоном», а НЕ «не на туши»: без картинки различить
нельзя. Для декодера этот же вопрос закрыт §6.190 — там 85% таких трасс лежали на настоящей туши.

★ Счёта не тратит: только пуловые дампы и разметка.

  <ComfyUI>\python_embeded\python.exe _u1_why.py
"""
import sys, argparse, pickle
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
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--cap", type=int, default=0)
a = ap.parse_args()
TS = Path(a.ts)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def stats(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]): int(r[3]["n_prod"]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
stem2f = {}
for root in ("pools", "pools_gate", "pools_wide", "pools_more", "pools_div",
             "pools_heldout", "pools_all"):
    for f in sorted((TS / root).glob("*.pkl")):
        stem2f.setdefault(f.stem, f)
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q

sheets = sorted({r[0] for r in trk})
if a.cap:
    sheets = sheets[:a.cap]
cnt, nlines, OFF = Counter(), [], []
for si, sh in enumerate(sheets, 1):
    f, q = stem2f.get(Path(sh).stem), SRC.get(sh)
    if not f or not q:
        continue
    d = pickle.load(open(f, "rb"))
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    bytr = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None:
            bytr[t].append(nm)
    lines = defaultdict(list)
    for L in d["lines"]:
        lines[int(L["track"])].append(L["tr"])
    for t, ns in bytr.items():
        if (sh, t) not in KOF:
            continue
        LL = lines.get(t, [])
        nlines.append(len(LL))
        for nm in ns:
            g = gts[nm]
            cnt["всего"] += 1
            if not LL:
                cnt["линий на треке НЕТ"] += 1
                continue
            st = [stats(tr, g) for tr in LL]
            ok = [HON(m, c) for m, c in st]
            if any(ok):
                cnt["★ накрыта"] += 1
                continue
            # лучшая по med среди тех, где вообще есть пересечение
            idx = [k for k, (m, c) in enumerate(st) if m is not None]
            if not idx:
                cnt["трассы не пересекаются с кривой"] += 1
                continue
            b = min(idx, key=lambda k: st[k][0])
            m, c = st[b]
            if m <= 3.0 and c < 0.9:
                cnt["не покрыта (трасса короткая)"] += 1
                continue
            tr = LL[b]
            if any(HON(*stats(tr, gts[g2])) for g2 in ns if g2 != nm):
                cnt["ведёт СОСЕДНЮЮ кривую трека"] += 1
                continue
            if any(HON(*stats(tr, gts[g2])) for g2 in gts if g2 not in ns):
                cnt["ушла в ЧУЖОЙ ТРЕК"] += 1
                continue
            ys = [y for y in tr if any(y in gts[g2] for g2 in ns)]
            on, own = 0, Counter()
            for y in ys:
                cand = [(abs(tr[y] - gts[g2][y]), g2) for g2 in ns if y in gts[g2]]
                if not cand:
                    continue
                d0, g0 = min(cand)
                if d0 <= 3.0:
                    on += 1
                    own[g0] += 1
            share = on / max(1, len(ys))
            many = sum(1 for v in own.values() if v >= 0.2 * max(1, on)) >= 2
            if share >= 0.8 and many:
                cnt["СКЛЕЙКА: меняет кривую по глубине"] += 1
            elif share >= 0.8:
                cnt["на кривых, но смещена"] += 1
            else:
                cnt["вне линий (без картинки не судить)"] += 1
                # ★ выгрузка для разбора маской (`_offtrace_ink.py`) — формат тот же, что у
                #   `_k34_why.py --dump-off`, чтобы разбирались ТЕ ЖЕ трассы, а не отобранные заново
                OFF.append((sh, nm, f"u1line{b}", dict(tr)))
    if si % 300 == 0:
        print(f"  {si}/{len(sheets)}  кривых {cnt['всего']}")

pickle.dump(OFF, open(TS / "offtrace_U1.pkl", "wb"))
print(f"★ трасс «вне линий» выгружено {len(OFF)} → offtrace_U1.pkl")

n = cnt["всего"]
print(f"\n★★ ПОЧЕМУ ЛИНИЯ U1 НЕ ДАЁТ КРИВУЮ (эталонных кривых {n}, "
      f"линий на трек медиана {np.median(nlines):.0f})")
print("| исход | кривых | доля |")
for k, v in cnt.most_common():
    if k == "всего":
        continue
    print(f"| {k} | {v} | {100*v/max(1,n):.1f}% |")
bad = n - cnt["★ накрыта"]
print(f"\n★ НЕ ВЗЯТО {bad} ({100*bad/max(1,n):.1f}%). Крупнейшая корзина и есть адрес правки U1.")
print("⚠ «Вне линий» без картинки не разделяется на «настоящая тушь» и «шум»; у декодера тот же "
      "вопрос закрыт §6.190 — там 85% лежали на настоящей туши.")

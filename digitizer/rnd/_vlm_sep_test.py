r"""_vlm_sep_test.py — ТЕСТ «РАЗЛИЧИТ ЛИ ГЛАЗ (VLM) ЛИНИИ ТАМ, ГДЕ ПРОД-ТРАССА УШЛА НА ЧУЖУЮ» (§6.222, 25.09).

Набор: отрезки ухода из `_drift_anatomy.py --dump` (начало — «прыжок», длина ≥ `--min-len` строк, не больше 2 на лист). Для каждого
случая — картинка из двух половин: слева чистый скан участка [y0 − 250, y0 + 350], справа он же с метками:
  зелёные точки — кривая ДО развилки (эталон), но кончаются за `--gap-before` строк до неё; метки продолжений начинаются
  через `--gap-after` строк после — зону развилки смотрящий проводит по туши САМ (первая редакция давала подсказку:
  верное продолжение начиналось у конца зелёных точек, ушедшее — в стороне, после прыжка);
  красные кружки «A» и синие «B» — два продолжения ПОСЛЕ развилки: эталон и ушедшая трасса выдачи, буквы случайны.
Вопрос: какое продолжение идёт по ТОЙ ЖЕ нарисованной кривой, что зелёные точки. Ключ ответов — отдельный json (не показывать).
Каждый случай — провал прод-селектора по построению, поэтому доля верных ответов — прямая мера того, умеет ли зрение то, чего не
умеет селектор.

  _vlm_sep_test.py --dump <anat.pkl> --out <каталог> --n 40
"""
import sys, argparse, pickle, hashlib, json, random
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from dataset_build import find_image
from auto import meta as M, imaging as im

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dump", required=True)
ap.add_argument("--dir", default="ab_slot/RA")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--min-len", type=int, default=100)
ap.add_argument("--seed", type=int, default=20260925)
ap.add_argument("--gap-before", type=int, default=30, help="зелёные точки кончаются за столько строк ДО развилки")
ap.add_argument("--gap-after", type=int, default=40, help="метки продолжений начинаются через столько строк ПОСЛЕ")
a = ap.parse_args()
TS = Path(a.ts); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
REC = pickle.load(open(a.dump, "rb"))
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
rng = random.Random(a.seed)
cand = [r for r in REC if str(r[6]).startswith("прыжок") and r[4] >= a.min_len]
rng.shuffle(cand)
per = Counter(); pick = []
for r in cand:
    if per[r[0]] >= 2:
        continue
    per[r[0]] += 1; pick.append(r)
    if len(pick) >= a.n * 2:
        break
key = {}; made = 0
for (sh, g, src, y0, n, wt, kind, frm, *_rest) in pick:
    if made >= a.n:
        break
    q = SRC.get(sh)
    if not q:
        continue
    img = find_image(q)
    stem = q.stem
    pd = TS / a.dir / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(pd.glob("*_auto.nlgx"), None)
    if not img or not got:
        continue
    W = {c["name"]: dense(c) for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"}
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"] if M.mnem_root(c["name"]) != "DA"}
    gt = gts.get(g)
    if not gt:
        continue
    tm = smap.get(sh, {}); t = tm.get(g)
    ws = [k for k in W if tm.get(k) == t]
    if not ws:
        continue
    # выдача, которая шла по эталону перед развилкой и ушла после
    def before(k):
        ys = [y for y in range(y0 - 60, y0) if y in W[k] and y in gt]
        return np.median([abs(W[k][y] - gt[y]) for y in ys]) if len(ys) > 20 else 1e9
    k = min(ws, key=before)
    w = W[k]
    after = [y for y in range(y0, y0 + 350) if y in w and y in gt]
    if len(after) < 150 or np.median([abs(w[y] - gt[y]) for y in after]) < 8:
        continue                                   # продолжения должны реально расходиться
    rgb = im.load_rgb(str(img)); H, Wd = rgb.shape[:2]
    ya, yb = max(0, y0 - 250), min(H, y0 + 350)
    xs = [gt[y] for y in range(ya, yb) if y in gt] + [w[y] for y in range(y0, yb) if y in w]
    xa, xb = max(0, int(min(xs)) - 70), min(Wd, int(max(xs)) + 70)
    raw = np.ascontiguousarray(rgb[ya:yb, xa:xb, ::-1])
    ann = raw.copy()
    for y in range(ya, y0 - a.gap_before, 8):
        if y in gt:
            cv2.circle(ann, (int(gt[y] - xa), y - ya), 2, (0, 170, 0), -1)
    true_is_a = rng.random() < 0.5
    paths = {"A": gt if true_is_a else w, "B": w if true_is_a else gt}
    colr = {"A": (0, 0, 230), "B": (230, 0, 0)}
    for L, d in paths.items():
        for y in range(y0 + a.gap_after, yb, 14):
            if y in d:
                cv2.circle(ann, (int(d[y] - xa), y - ya), 4, colr[L], 1)
        ylab = min(yb - 12, y0 + a.gap_after + 20)
        if ylab in d:
            cv2.putText(ann, L, (int(d[ylab] - xa) + 7, ylab - ya), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colr[L], 2)
    sep = np.full((raw.shape[0], 12, 3), 255, np.uint8)
    both = np.hstack([raw, sep, ann])
    s = min(1.0, 1400 / both.shape[1], 1100 / both.shape[0])
    if s < 1:
        both = cv2.resize(both, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cid = f"case{made:02d}"
    cv2.imencode(".png", both)[1].tofile(str(OUT / f"{cid}.png"))
    key[cid] = dict(answer="A" if true_is_a else "B", sheet=sh, curve=g, src=src, y0=int(y0), run_len=int(n), frm=frm,
                    sep_px=float(np.median([abs(w[y] - gt[y]) for y in after])))
    made += 1
    print(f"  {cid}: {sh[:40]} {g} y0={y0} разнос {key[cid]['sep_px']:.0f} px, источник {src}, ушла {frm}")
json.dump(key, open(OUT / "_key.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"★ случаев {made}; ключ → {OUT / '_key.json'} (агентам не показывать)")

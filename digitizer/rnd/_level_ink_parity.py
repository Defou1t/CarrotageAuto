r"""_level_ink_parity.py — СВЕРКА ПРОД-МОДУЛЯ `auto/level_ink.py` С ОБУЧАЮЩИМ СТЕНДОМ `_level_ink.py` (§6.266, 03.10).

Признаки: на листах скана (`level_ink/*.pkl`) прод-модуль считает признаки заново — из выдачи повтора (`rp_vc/N`), скана из
кэша трасс (тот же файл, что получает emit) и цепочки каркаса — и сверяет с массивами скана побайтно.
Уровни (`--pred`): прод-предсказание на этих признаках против дампа стенда (та же модель: поле — фолд листа, сорт A — на всём
поле), доля совпавших позиций.

  _level_ink_parity.py --n 40 --pred F:/nds/output/taskS/level_ink_pred_v2r.pkl --models oof:F:/nds/output/taskS/level_ink_v2r
"""
import sys, argparse, pickle, hashlib, random
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from extract_nlgx import extract, NULL
from auto import level_ink as LI, meta as M
import decode_levels as DL

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--scan", default=r"F:/nds/output/taskS/level_ink")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_vc/N")
ap.add_argument("--tcache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pred", default="")
ap.add_argument("--models", default="")
a = ap.parse_args()

files = sorted(Path(a.scan).glob("*.pkl"))
random.Random(a.seed).shuffle(files)
PR = pickle.load(open(a.pred, "rb"))["pred"] if a.pred else {}
bad = Counter(); tot = Counter(); agree = []
for f in files[:a.n]:
    d = pickle.load(open(f, "rb"))
    sh = d["sheet"]; stem = Path(sh).stem
    key = f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next((Path(a.dir) / key).glob("*_auto.nlgx"), None)
    tc = Path(a.tcache) / (key + ".pkl")
    if not got or not tc.exists():
        tot["нет выдачи/кэша"] += 1; continue
    c = pickle.load(open(tc, "rb"))
    gray = np.asarray(Image.open(c["image"]).convert("L"))
    paper = LI.paper_of(gray)
    frame = extract(str(c["frame_nlgx"]))
    fc = {cc["name"]: cc for cc in frame["curves"]}
    out = extract(str(got))
    oc = {cc["name"]: cc for cc in out["curves"]}
    others = []
    for cc in out["curves"]:
        pts = [(cc["top_y"] + i, x) for i, x in enumerate(cc["xs"]) if x != NULL]
        if len(pts) >= 2:
            others.append((cc["name"], np.array([p[0] for p in pts], np.int64), np.array([p[1] for p in pts], np.float64)))
    for s in d["curves"]:
        if s is None:
            continue
        nm = s["name"]
        w = oc.get(nm); fr = fc.get(nm)
        if w is None or fr is None:
            tot["нет кривой"] += 1; continue
        fam = DL.build_family(frame, fr)
        ty, tx = LI.dense_rows(w["top_y"], w["xs"], NULL)
        g = LI.features(gray, paper, fam, nm, M.mnem_root(nm), ty, tx, list(w.get("segments") or []), others)
        tot["кривых"] += 1
        if g is None:
            bad["прод: None"] += 1; continue
        if len(g["P"]) != len(s["P"]) or not np.array_equal(g["P"], s["P"]):
            bad["позиции"] += 1; continue
        occ = np.unpackbits(s["occ"], axis=1)[:, :g["occ"].shape[1]]
        for k_, x_, y_ in (("ink", g["ink"], s["ink"]), ("occ", g["occ"], occ), ("tw", g["tw"], s["tw"]),
                           ("twin", g["twin"], s["twin"]), ("lw", g["lw"], s["lw"]), ("has", g["has"], s["has"])):
            if not np.array_equal(x_, y_):
                bad[k_] += 1
        if not np.array_equal(np.isnan(g["u"]), np.isnan(s["u"])) or not np.allclose(np.nan_to_num(g["u"]), np.nan_to_num(s["u"])):
            bad["u"] += 1
        if g["K"] != s["K"]:
            bad["K"] += 1
        if a.models and s["ci"] in PR:
            path = LI.resolve(a.models, sheet=sh)
            lv = LI.predict(g, path)
            agree.append(float(np.mean(lv == PR[s["ci"]][1])))
    del gray
print(f"★ сверка признаков: {dict(tot)}; расхождений {dict(bad) or 'нет'}")
if agree:
    ag = np.array(agree)
    print(f"★ уровни прода против дампа стенда: кривых {len(ag)}, совпадение позиций — медиана {np.median(ag):.4f}, "
          f"минимум {ag.min():.4f}, полностью совпали {np.mean(ag == 1.0):.0%}")

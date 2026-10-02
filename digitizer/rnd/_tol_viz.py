r"""_tol_viz.py — КАРТИНКИ КРИВЫХ, КОТОРЫЕ ЧЕСТНЫ ТОЛЬКО ПО НОРМАЛИ (02.10, к §6.257 и решению о допуске).

Берёт дамп `_tolerance_whatif.py --dump-gained` (лист, кривая эталона, кривая выдачи, медиана |Δ| px, медиана |Δ|/допуск) и
для каждой выбранной кривой рисует окно листа с наибольшей ДОЛЕЙ строк, где |Δ| > 3 px: эталон — зелёный, выдача — красная
(сдвиг +3 px, чтобы не закрывать), подпись — медианы. Глазами: идёт ли выдача по своей туши (тогда 3 px по строке занижают
счёт) или по чужой.

  _tol_viz.py --gained F:/nds/output/taskS/tol_normal_gained_A.pkl --n 10 --out <папка>
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from extract_nlgx import extract
from _multi_replica_probe import dense
from dataset_build import find_image

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--gained", required=True)
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_vc/N")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rows", type=int, default=700)
ap.add_argument("--out", required=True)
ap.add_argument("--window", default="worst", choices=["worst", "random"], help="окно: худшее по доле |Δ| > 3 px или случайное")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
G_ = pickle.load(open(a.gained, "rb"))
rng = np.random.default_rng(a.seed)
pick = [G_[i] for i in sorted(rng.choice(len(G_), min(a.n, len(G_)), replace=False))]
for j, (sh, g, k, mraw, mnorm) in enumerate(pick):
    q = SRC.get(sh)
    stem = q.stem
    pd = Path(a.dir) / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(pd.glob("*_auto.nlgx"))
    W = next(dense(c) for c in extract(str(got))["curves"] if c["name"] == k)
    gt = next(dense(c) for c in extract(str(q))["curves"] if c["name"] == g)
    com = sorted(y for y in W if y in gt)
    bad = np.array([abs(W[y] - gt[y]) > 3 for y in com], float)
    # окно с наибольшей долей плохих строк
    win = a.rows
    cs = np.concatenate([[0], np.cumsum(bad)])
    best = (int(np.argmax(cs[win:] - cs[:-win])) if a.window == "worst" else int(rng.integers(0, len(com) - win))) if len(com) > win else 0
    y0 = com[best]; y1 = y0 + win
    img = Image.open(find_image(q)).convert("RGB")
    xs = [gt[y] for y in range(y0, y1) if y in gt] + [W[y] for y in range(y0, y1) if y in W]
    x0 = int(max(0, min(xs) - 80)); x1 = int(min(img.width, max(xs) + 80))
    cr = img.crop((x0, y0, x1, min(img.height, y1))).copy()
    dr = ImageDraw.Draw(cr)
    pts = [(gt[y] - x0, y - y0) for y in range(y0, y1) if y in gt]
    if len(pts) > 1:
        dr.line(pts, fill=(0, 220, 0), width=1)
    for y in range(y0, y1):
        if y in W:
            dr.ellipse([(W[y] - x0 - 1, y - y0 - 1), (W[y] - x0 + 1, y - y0 + 1)], fill=(255, 0, 0))
    nm = f"{j:02d}_{stem[:24]}_{g}.png".replace(" ", "_")
    cr.save(OUT / nm)
    print(f"{nm}: выдача «{k}», медиана |Δ| {mraw:.1f} px, по нормали {mnorm:.2f}; окно строк {y0}–{y1}, "
          f"плохих строк в окне {bad[best:best + win].mean():.2f}")

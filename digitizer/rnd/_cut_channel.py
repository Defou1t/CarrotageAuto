r"""_cut_channel.py — КАНАЛ «ОБРУБ У КРАЯ» ДЛЯ МОДЕЛИ УРОВНЯ: ВЕРОЯТНОСТЬ СЕТИ У КРАЯ НА ПОЗИЦИЯХ КРИВЫХ СКАНА (23.10, §6.339).

Сеть `_level_cnn.py --data cpatch.npz --save-dir …` (кусок 96 × 96 у края шкалы уровня 0 со стороны v_right, `_cut_patch.py`) —
по каждой кривой сканов модели уровня (`level_ink_nslx`, `level_ink_out`) на её позициях P: модель фолда листа (карта
`folds.json`), лист вне карты — модель на поле (`all.pt`). Пишет `--out` = {(лист, имя кривой): float16 [len(P)]}.

  _cut_channel.py --model F:/nds/output/taskS/cut_cnn_w --out F:/nds/output/taskS/cut_channel.pkl
"""
import sys, argparse, pickle, hashlib, json, time
sys.path.insert(0, "F:/nds/Auto"); sys.path.insert(0, "F:/nds/Auto/digitizer"); sys.path.insert(0, "F:/nds/Auto/digitizer/rnd")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--scans", nargs="+", default=["F:/nds/output/taskS/level_ink_nslx", "F:/nds/output/taskS/level_ink_out"])
ap.add_argument("--model", default="F:/nds/output/taskS/cut_cnn_w")
ap.add_argument("--out", default="F:/nds/output/taskS/cut_channel.pkl")
ap.add_argument("--inn", type=int, default=64)
ap.add_argument("--height", type=int, default=96)
ap.add_argument("--outw", type=int, default=32)
a = ap.parse_args()
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from extract_nlgx import extract
from auto import level_ink as LI
from auto.trace_look import build_net
import decode_levels as DL
S = a.height
IN, OUT = a.inn, a.outw

TS = Path("F:/nds/output/taskS")
dev = "cuda"
FO = json.loads((Path(a.model) / "folds.json").read_text(encoding="utf-8"))
NETS = {}


def net_for(sh):
    k = FO.get(sh)
    f = Path(a.model) / (f"f{k}.pt" if k is not None else "all.pt")
    if f not in NETS:
        n = build_net(); n.load_state_dict(torch.load(f, map_location="cpu")); n.eval(); NETS[f] = n.to(dev)
    return NETS[f]


BY = defaultdict(list)
for d in a.scans:
    for f in sorted(Path(d).glob("*.pkl")):
        r = pickle.load(open(f, "rb"))
        for c in r["curves"]:
            if c is not None:
                BY[c["sheet"]].append((c["name"], c["P"]))
SRC = {q.name: q for q in Path("F:/nds/projects/Archive").glob("*/wlg/*.nlgx")}
OUTD = {}; t0 = time.time()
for n_, (sh, curves) in enumerate(sorted(BY.items())):
    q = SRC.get(sh)
    if q is None:
        continue
    key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
    tc = TS / "tcache" / (key + ".pkl")
    if not tc.exists():
        tc = TS / "tcache_out" / (key + ".pkl")
    if not tc.exists():
        continue
    try:
        gray = np.asarray(Image.open(pickle.load(open(tc, "rb"))["image"]).convert("L"))
    except Exception:
        continue
    paper = LI.paper_of(gray).astype(np.int16)
    H, W = gray.shape
    mt = extract(str(q)); CV = {c["name"]: c for c in mt["curves"]}
    net = net_for(sh)
    for name, P in curves:
        c = CV.get(name)
        if c is None:
            continue
        ch = DL.build_family(mt, c)
        if len(ch) < 2:
            continue
        s0 = ch[0]; xe = int(round(s0["x_right"])); sgn = 1 if s0["x_right"] >= s0["x_left"] else -1
        xs = xe + sgn * np.arange(-IN, OUT); okx = (xs >= 0) & (xs < W); xc = np.clip(xs, 0, W - 1)
        prob = np.full(len(P), np.nan, np.float32)
        Xb, ib = [], []
        for i, y in enumerate(np.asarray(P, np.int64)):
            ya, yb = int(y) - S // 2, int(y) + S // 2
            if ya < 0 or yb > H:
                continue
            X = np.zeros((1, 2, S, IN + OUT), np.uint8)
            g = gray[ya:yb][:, xc].astype(np.int16)
            X[0, 0] = np.where(okx[None, :], np.clip(paper[ya:yb, None] - g, 0, 255), 0).astype(np.uint8)
            X[0, 1, :, IN] = 255
            Xb.append(X); ib.append(i)
        with torch.no_grad():
            for j in range(0, len(Xb), 512):
                xb = torch.from_numpy(np.stack(Xb[j:j + 512])).to(dev).float() / 255.0
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prob[ib[j:j + 512]] = torch.sigmoid(net(xb).float()).cpu().numpy()
        OUTD[(sh, name)] = prob.astype(np.float16)
    if (n_ + 1) % 200 == 0:
        print(f"  {n_ + 1}/{len(BY)} листов, {time.time() - t0:.0f} с")
pickle.dump(OUTD, open(a.out, "wb"))
print(f"★ кривых {len(OUTD)} → {a.out}")

r"""_level_cnn.py — СЕТЬ «1× ИЛИ ВЫШЕ» ПО КУСОЧКУ СКАНА ВОКРУГ ЭТАЛОННОЙ ТРАССЫ (21.10, §6.329, потолок).

Кусочки `_level_patch.py`; метка — уровень эталона в строке ≥ 1. Фолды по скважинам — как у стенда выбора трассы `--eval-out` (поле ∪
вне поля без скважин сорта A, i % 5); сорт A — модель на всём поле. Печатает AUC вне фолда (по кускам и по кривым — среднее P
кусков участка). Сеть — `auto.trace_look.build_net` (одна точка).

  _level_cnn.py --data F:/nds/output/taskS/lpatch.npz
"""
import sys, argparse, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, "F:/nds/Auto")
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default="F:/nds/output/taskS")
ap.add_argument("--data", default="F:/nds/output/taskS/lpatch.npz")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--bs", type=int, default=512)
ap.add_argument("--out", default="F:/nds/output/taskS/level_cnn.pkl")
a = ap.parse_args()
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from auto.trace_look import build_net

TS = Path(a.ts)
lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
FIELD, HOLD = lst("wellmap_sheets.txt"), lst("holdoutA_sheets.txt")
SRC = {q.name: q for q in Path("F:/nds/projects/Archive").glob("*/wlg/*.nlgx")}
well = lambda sh: SRC[sh].parent.parent.name if sh in SRC else sh
z = np.load(a.data)
X, LV, SH = z["X"], z["lv"], z["sheet"]
Y = (LV >= 1).astype(np.int64)
HW = {well(x) for x in HOLD}
grp = []
FLD = sorted({s for s in set(SH.tolist()) if (s in FIELD) or (s not in HOLD and well(s) not in HW)})
wells = sorted({well(s) for s in FLD}); FOLD = {w: i % 5 for i, w in enumerate(wells)}
G = np.array([("A" if s in HOLD else (FOLD[well(s)] if s in set(FLD) else "out")) for s in SH], object)
print(f"кусков {len(Y)}, уровень ≥ 1: {Y.mean():.2f}; поле+вне {int(np.isin(G, list(range(5))).sum())}, сорт A {int((G == 'A').sum())}")
dev = "cuda"


def batches(idx, shuffle, rng):
    idx = np.array(idx)
    if shuffle:
        idx = idx[rng.permutation(len(idx))]
    for i in range(0, len(idx), a.bs):
        b = idx[i:i + a.bs]
        xb = torch.from_numpy(X[b]).to(dev).float() / 255.0
        if shuffle and rng.random() < 0.5:
            xb = xb.flip(4)
        yield b, xb


def train(tr):
    rng = np.random.default_rng(0); torch.manual_seed(0)
    net = build_net().to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps = a.epochs * (-(-len(tr) // a.bs)); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    for ep in range(a.epochs):
        net.train()
        for b, xb in batches(tr, True, rng):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = net(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(lo.float(), torch.from_numpy(Y[b]).to(dev).float())
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net


def score(net, te):
    net.eval(); out = np.zeros(len(te), np.float32); o = 0
    with torch.no_grad():
        for b, xb in batches(te, False, None):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[o:o + len(b)] = torch.sigmoid(net(xb).float()).cpu().numpy()
            o += len(b)
    return out


S = np.full(len(Y), np.nan, np.float32); t0 = time.time()
for k in range(5):
    tr = np.flatnonzero(np.isin(G, [f for f in range(5) if f != k])); te = np.flatnonzero(G == k)
    S[te] = score(train(tr), te)
    print(f"  фолд {k}: обучение {len(tr)}, AUC {roc_auc_score(Y[te], S[te]):.4f}, {time.time() - t0:.0f} с")
tr = np.flatnonzero(np.isin(G, list(range(5)))); te = np.flatnonzero(G == "A")
S[te] = score(train(tr), te)
for nm, sel in (("поле+вне", np.isin(G, list(range(5)))), ("сорт A", G == "A")):
    print(f"★ {nm}: AUC вне фолда по кускам {roc_auc_score(Y[sel], S[sel]):.4f} ({int(sel.sum())})")
import pickle
pickle.dump(dict(S=S, Y=Y, G=G, sheet=SH, curve=z["curve"]), open(a.out, "wb"))

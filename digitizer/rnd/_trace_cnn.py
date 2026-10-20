r"""_trace_cnn.py — СВЁРТОЧНАЯ ОЦЕНКА «НАСТОЯЩАЯ ЛИ ТРАССА» ПО КУСОЧКАМ СКАНА (20.10, §6.325).

Кусочки `_trace_patch.py` (`--patch`), метки — сканы `line_choice_i` / `line_choice_out_i`: кандидат честен хоть для одного слота
(кандидаты без эталона в полосе ни одного слота — без метки, только оценка). Фолды по скважинам — ТЕ ЖЕ, что у
`_line_choice_assign.py --eval-out` (скважины поля + вне поля без скважин сорта A, отсортированы, i % 5); сорт A — модель на всём поле.
Пишет `--out` = {(лист, k): оценка вне фолда} и печатает AUC вне фолда.

  _trace_cnn.py --epochs 6
"""
import sys, argparse, pickle, hashlib, time, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--data", nargs="+", default=[r"F:/nds/output/taskS/line_choice_i", r"F:/nds/output/taskS/line_choice_out_i"])
ap.add_argument("--patch", default=r"F:/nds/output/taskS/tpatch")
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--out", default=r"F:/nds/output/taskS/trace_cnn.pkl")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--save-dir", default="", help="веса: trace_cnn_f{k}.pt (фолды), trace_cnn_all.pt (на поле) — рядом с моделями выбора трассы")
ap.add_argument("--prod-out", default="", help="веса для прода: учится на ВСЁМ с меткой, включая сорт A")
a = ap.parse_args()
import torch, torch.nn as nn

TS = Path(a.ts)
lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
FIELD, HOLD = lst("wellmap_sheets.txt"), lst("holdoutA_sheets.txt")
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
well = lambda sh: SRC[sh].parent.parent.name if sh in SRC else sh

# метки и листы
LAB, SHEETS = {}, set()
for d in a.data:
    for f in sorted(Path(d).glob("*.pkl")):
        r = pickle.load(open(f, "rb")); sh = r["sheet"]
        if sh in SHEETS:
            continue
        SHEETS.add(sh)
        lab = defaultdict(lambda: None)
        for s in r["slots"]:
            for c in s["cands"]:
                if c["lab"] is not None:
                    lab[c["k"]] = max(lab[c["k"]] or 0, int(bool(c["lab"][0])))
        LAB[sh] = dict(lab)
# фолды как у стенда --eval-out: поле ∪ (вне поля без скважин сорта A)
HW = {well(x) for x in HOLD}
OUTS = {x for x in SHEETS if x not in FIELD and x not in HOLD}
EVO = {x for x in OUTS if well(x) not in HW}
FLD = (FIELD & SHEETS) | EVO
wells = sorted({well(x) for x in FLD}); FOLD = {w: i % a.folds for i, w in enumerate(wells)}

# кусочки
PF = [(sh, Path(a.patch) / (hashlib.md5(sh.encode("utf-8")).hexdigest()[:16] + ".npz")) for sh in sorted(SHEETS)]
PF = [(sh, pf) for sh, pf in PF if pf.exists()]
N = sum(len(np.load(pf)["k"]) for _, pf in PF)
X = None; Y, KEY = [], []; o = 0
for sh, pf in PF:
    z = np.load(pf); xs = z["X"]                         # ⚠ z["X"] распаковывает весь массив при КАЖДОМ обращении — читать один раз
    if X is None:
        X = np.zeros((N,) + xs.shape[1:], np.uint8)
    X[o:o + len(xs)] = xs; o += len(xs)
    for k in z["k"].tolist():
        Y.append(-1 if LAB[sh].get(k) is None else LAB[sh][k]); KEY.append((sh, k))
Y = np.array(Y)
GRP = np.array([("A" if sh in HOLD else (FOLD[well(sh)] if sh in FLD else "out")) for sh, _ in KEY], object)
print(f"кандидатов {len(Y)} (с меткой {int((Y >= 0).sum())}, честных {int((Y == 1).sum())}), листов {len(SHEETS)}")
dev = "cuda"


sys.path.insert(0, "F:/nds/Auto")
from auto.trace_look import build_net                    # сеть — та же, что в проде


def batches(idx, shuffle, rng):
    idx = np.array(idx)
    if shuffle:
        idx = idx[rng.permutation(len(idx))]
    for i in range(0, len(idx), a.bs):
        b = idx[i:i + a.bs]
        xb = torch.from_numpy(X[b]).to(dev).float() / 255.0
        if shuffle and rng.random() < 0.5:
            xb = xb.flip(4)                              # отражение по x
        yield b, xb


def train(tr):
    rng = np.random.default_rng(0); torch.manual_seed(0)
    net = build_net().to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps = a.epochs * (-(-len(tr) // a.bs)); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    pw = torch.tensor(float((Y[tr] == 0).sum()) / max(1, (Y[tr] == 1).sum()), device=dev)
    for ep in range(a.epochs):
        net.train()
        for b, xb in batches(tr, True, rng):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = net(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(lo.float(), torch.from_numpy(Y[b]).to(dev).float(), pos_weight=pw)
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


from sklearn.metrics import roc_auc_score
S = np.full(len(Y), np.nan, np.float32); t0 = time.time()
for k in range(a.folds):
    tr = np.flatnonzero((Y >= 0) & np.isin(GRP, [f_ for f_ in range(a.folds) if f_ != k]))   # листы скважин сорта A — никогда
    te = np.flatnonzero(GRP == k)
    net = train(tr); S[te] = score(net, te)
    if a.save_dir:
        Path(a.save_dir).mkdir(parents=True, exist_ok=True); torch.save(net.state_dict(), Path(a.save_dir) / f"trace_cnn_f{k}.pt")
    m = te[Y[te] >= 0]
    print(f"  фолд {k}: обучение {len(tr)}, AUC {roc_auc_score(Y[m], S[m]):.4f}, {time.time() - t0:.0f} с")
tr = np.flatnonzero((Y >= 0) & np.isin(GRP, list(range(a.folds)))); te = np.flatnonzero((GRP == "A") | (GRP == "out"))
net = train(tr); S[te] = score(net, te)
if a.save_dir:
    torch.save(net.state_dict(), Path(a.save_dir) / "trace_cnn_all.pt")
if a.prod_out:
    torch.save(train(np.flatnonzero(Y >= 0)).state_dict(), a.prod_out)
    print(f"★ веса для прода (всё с меткой, с сортом A): {a.prod_out}")
for nm, sel in (("поле+вне", np.isin(GRP, list(range(a.folds)))), ("сорт A", GRP == "A")):
    m = sel & (Y >= 0)
    print(f"★ {nm}: AUC вне фолда {roc_auc_score(Y[m], S[m]):.4f} ({int(m.sum())} кандидатов)")
pickle.dump({KEY[i]: float(S[i]) for i in range(len(KEY))}, open(a.out, "wb"))
print(f"★ записано {len(KEY)} оценок → {a.out}")

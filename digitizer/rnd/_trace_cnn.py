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
ap.add_argument("--seeds", type=int, default=1, help="§6.334: ансамбль из N сетей (зёрна 0…N−1), оценка — среднее; веса — список state_dict")
ap.add_argument("--fam", action="store_true", help="§6.327: вместо «честна ли» — семейство слота, где трасса честна (частые семейства, «прочее», 0 = мусор); "
                "пишет вектор вероятностей и список классов (ключ '__classes__')")
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
                    if a.fam:                            # семейство слота, где честна (первое), иначе 0
                        if c["lab"][0]:
                            lab[c["k"]] = lab[c["k"]] if isinstance(lab[c["k"]], str) else s["root"]
                        elif lab[c["k"]] is None:
                            lab[c["k"]] = 0
                    else:
                        lab[c["k"]] = max(lab[c["k"]] or 0, int(bool(c["lab"][0])))
        LAB[sh] = dict(lab)
if a.fam:                                                # классы: 0 мусор, частые семейства (≥ 300), «прочее»
    from collections import Counter as _C
    _fc = _C(v for L in LAB.values() for v in L.values() if isinstance(v, str))
    CLS = ["мусор"] + sorted(f for f, n in _fc.items() if n >= 300) + ["прочее"]
    _ci = {f: i for i, f in enumerate(CLS)}
    for L in LAB.values():
        for k in L:
            if isinstance(L[k], str):
                L[k] = _ci.get(L[k], len(CLS) - 1)
    print("классы:", CLS)
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


def train(tr, seed=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    net = build_net(len(CLS) if a.fam else 1).to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps = a.epochs * (-(-len(tr) // a.bs)); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    pw = torch.tensor(float((Y[tr] == 0).sum()) / max(1, (Y[tr] == 1).sum()), device=dev)
    for ep in range(a.epochs):
        net.train()
        for b, xb in batches(tr, True, rng):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = net(xb)
            if a.fam:
                loss = nn.functional.cross_entropy(lo.float(), torch.from_numpy(Y[b]).to(dev).long())
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(lo.float(), torch.from_numpy(Y[b]).to(dev).float(), pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net


def score(net, te):
    net.eval(); out = np.zeros((len(te), len(CLS)) if a.fam else len(te), np.float32); o = 0
    with torch.no_grad():
        for b, xb in batches(te, False, None):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = net(xb).float()
                out[o:o + len(b)] = (torch.softmax(lo, 1) if a.fam else torch.sigmoid(lo)).cpu().numpy()
            o += len(b)
    return out


from sklearn.metrics import roc_auc_score
class Ens:
    def __init__(s, nets):
        s.nets = nets

    def state(s):
        return s.nets[0].state_dict() if len(s.nets) == 1 else {"sds": [n.state_dict() for n in s.nets]}


_train1, _score1 = train, score
train = lambda tr: Ens([_train1(tr, sd) for sd in range(a.seeds)])
score = lambda ens, te: np.mean([_score1(n, te) for n in ens.nets], 0)
S = np.full((len(Y), len(CLS)) if a.fam else len(Y), np.nan, np.float32); t0 = time.time()
AUCY = (Y > 0).astype(int)                                # мера: «честна хоть для одного» (при --fam — 1 − P(мусор))
_auc = lambda m: roc_auc_score(AUCY[m], (1 - S[m, 0]) if a.fam else S[m])
for k in range(a.folds):
    tr = np.flatnonzero((Y >= 0) & np.isin(GRP, [f_ for f_ in range(a.folds) if f_ != k]))   # листы скважин сорта A — никогда
    te = np.flatnonzero(GRP == k)
    net = train(tr); S[te] = score(net, te)
    if a.save_dir:
        Path(a.save_dir).mkdir(parents=True, exist_ok=True); torch.save(net.state(), Path(a.save_dir) / f"trace_cnn_f{k}.pt")
    m = te[Y[te] >= 0]
    print(f"  фолд {k}: обучение {len(tr)}, AUC {_auc(m):.4f}, {time.time() - t0:.0f} с")
tr = np.flatnonzero((Y >= 0) & np.isin(GRP, list(range(a.folds)))); te = np.flatnonzero((GRP == "A") | (GRP == "out"))
net = train(tr); S[te] = score(net, te)
if a.save_dir:
    torch.save(net.state(), Path(a.save_dir) / "trace_cnn_all.pt")
if a.prod_out:
    torch.save(train(np.flatnonzero(Y >= 0)).state(), a.prod_out)
    print(f"★ веса для прода (всё с меткой, с сортом A): {a.prod_out}")
for nm, sel in (("поле+вне", np.isin(GRP, list(range(a.folds)))), ("сорт A", GRP == "A")):
    m = sel & (Y >= 0)
    print(f"★ {nm}: AUC вне фолда {_auc(m):.4f} ({int(m.sum())} кандидатов)")
if a.fam:
    _o = {KEY[i]: S[i].astype(np.float16) for i in range(len(KEY))}; _o["__classes__"] = CLS
    pickle.dump(_o, open(a.out, "wb"))
else:
    pickle.dump({KEY[i]: float(S[i]) for i in range(len(KEY))}, open(a.out, "wb"))
print(f"★ записано {len(KEY)} оценок → {a.out}")

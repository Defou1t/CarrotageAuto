r"""_seg_net.py — ОБУЧЕНИЕ СЕТИ «ТУШЬ ОЦИФРОВАННЫХ КРИВЫХ» (`auto/ink_seg.py`) ПО КУСКАМ `_seg_patch.py` (24.10, §6.345).

Проверочные листы — `--test-sheets` (их скважины целиком вне обучения); печатает AUC и долю верных пикселей по ним. Веса → `--save`.

  _seg_net.py --test-sheets F:/nds/output/taskS/orc_sheets.txt --save F:/nds/output/taskS/ink_seg_t.pt
"""
import sys, argparse, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"F:/nds/output/taskS/segpatch")
ap.add_argument("--test-sheets", default="")
ap.add_argument("--save", default="", help="одна сеть (с --test-sheets — проверка)")
ap.add_argument("--save-dir", default="", help="вне фолда: f{k}.pt — без скважин фолда k (скважины листов, по порядку, i % --folds), "
                "all.pt — на всех; folds.json — скважина → фолд")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--base", type=int, default=32)
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--bs", type=int, default=24)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--pos-weight", type=float, default=4.0)
ap.add_argument("--fam", action="store_true", help="§6.350: куски `_seg_patch.py --fam` — выход 1 + K каналов: тушь, затем семейства")
a = ap.parse_args()
import torch, torch.nn as nn
from auto.ink_seg import build_net

SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
TE = {l.strip() for l in Path(a.test_sheets).read_text(encoding="utf-8").splitlines() if l.strip()} if a.test_sheets else set()
TEW = {SRC[s].parent.parent.name for s in TE if s in SRC}
files = sorted(Path(a.data).glob("*.npz"))
meta = []
for f in files:
    z = np.load(f)
    meta.append((f, str(z["sheet"]), str(z["well"]), len(z["Yb"] if a.fam else z["Y"])))
FAMS = [str(x) for x in np.load(files[0])["fams"]] if a.fam else []
K = len(FAMS)
def train(TEW, TE, save):
    tr = [m for m in meta if m[2] not in TEW]; te = [m for m in meta if m[1] in TE]
    n_tr = sum(m[3] for m in tr); S = np.load(tr[0][0])["X"].shape[-1]
    print(f"кусков обучения {n_tr} ({len(tr)} листов), проверка {sum(m[3] for m in te)} ({len(te)} листов), S {S}")
    X = np.empty((n_tr, 3, S, S), np.uint8); Y = np.empty((n_tr, S, S), np.uint32 if a.fam else np.uint8); o = 0
    for f, _, _, n in tr:
        z = np.load(f); X[o:o + n] = z["X"]; Y[o:o + n] = z["Yb"] if a.fam else z["Y"]; o += n
    dev = torch.device("cuda")
    net = build_net(a.base, 1 + K if a.fam else 1).to(dev)
    bits = torch.arange(K, device=dev, dtype=torch.int64)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    pw = torch.tensor(a.pos_weight, device=dev)
    rs = np.random.RandomState(0); t0 = time.time(); run = 0.0
    for st in range(a.steps):
        b = rs.randint(0, n_tr, a.bs)
        xb = torch.from_numpy(X[b]).to(dev).float() / 255.0
        if a.fam:                                         # канал 0 — любая тушь, k + 1 — семейство FAMS[k]
            yi = torch.from_numpy(Y[b].astype(np.int64)).to(dev)
            yb = torch.cat([(yi > 0)[:, None], ((yi[:, None] >> bits[None, :, None, None]) & 1) > 0], 1).float()
        else:
            yb = torch.from_numpy(Y[b]).to(dev).float()
        if rs.rand() < 0.5:
            xb = torch.flip(xb, dims=[3]); yb = torch.flip(yb, dims=[yb.dim() - 1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = net(xb)
        loss = nn.functional.binary_cross_entropy_with_logits(lo.float(), yb, pos_weight=pw)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        run = 0.98 * run + 0.02 * float(loss.detach()) if st else float(loss.detach())
        if (st + 1) % 500 == 0:
            print(f"  шаг {st + 1}/{a.steps}: потеря {run:.4f}, {time.time() - t0:.0f} с")
    net.eval()
    torch.save(dict(sd=net.state_dict(), base=a.base, S=S, out=1 + K if a.fam else 1, fams=FAMS), save)
    if te:
        from sklearn.metrics import roc_auc_score
        P_, Y_, FA_ = [], [], []
        with torch.no_grad():
            for f, _, _, n in te:
                z = np.load(f)
                xb = torch.from_numpy(z["X"]).to(dev).float() / 255.0
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr = torch.sigmoid(net(xb).float()).cpu().numpy()
                if a.fam:                                 # §6.350: доля верного семейства (argmax) на пикселях одной кривой
                    yb_ = z["Yb"][:, ::4, ::4]; one_ = (yb_ > 0) & ((yb_ & (yb_ - 1)) == 0)
                    am_ = pr[:, 1:, ::4, ::4].argmax(1)
                    FA_.append((am_[one_], np.log2(yb_[one_]).round().astype(int)))
                    pr = pr[:, 0]; yy_ = (yb_ > 0).astype(np.uint8)
                else:
                    yy_ = z["Y"][:, ::4, ::4]
                P_.append(pr[:, ::4, ::4].ravel()); Y_.append(yy_.ravel())
        P_ = np.concatenate(P_); Y_ = np.concatenate(Y_)
        if a.fam:
            am_ = np.concatenate([x[0] for x in FA_]); tf_ = np.concatenate([x[1] for x in FA_])
            print(f"★ проверка семейств: верное семейство у {np.mean(am_ == tf_):.3f} пикселей одной кривой ({len(tf_)}); "
                  + ", ".join(f"{FAMS[k]} {np.mean(am_[tf_ == k] == k):.2f}" for k in range(K) if (tf_ == k).sum() > 500))
        print(f"★ проверка: AUC пикселей {roc_auc_score(Y_, P_):.4f}; при 0.5 — точность {((P_ >= 0.5) & (Y_ > 0)).sum() / max(1, (P_ >= 0.5).sum()):.3f}, "
              f"полнота {((P_ >= 0.5) & (Y_ > 0)).sum() / max(1, (Y_ > 0).sum()):.3f}")
    print(f"★ веса → {save}")


if a.save_dir:
    import json
    sd_ = Path(a.save_dir); sd_.mkdir(parents=True, exist_ok=True)
    ws = sorted({m[2] for m in meta}); FO = {w: i % a.folds for i, w in enumerate(ws)}
    (sd_ / "folds.json").write_text(json.dumps(FO, ensure_ascii=False), encoding="utf-8")
    for k in range(a.folds):
        print(f"=== фолд {k}")
        train({w for w, f in FO.items() if f == k}, {m[1] for m in meta if FO[m[2]] == k}, str(sd_ / f"f{k}.pt"))
    print("=== на всех")
    train(set(), set(), str(sd_ / "all.pt"))
else:
    train(TEW, TE, a.save)

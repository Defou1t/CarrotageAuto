r"""_seq_eval.py — ТОЧНОСТЬ РЕШЕНИЙ СЕЛЕКТОРА НА ОЦЕНОЧНОЙ ВЫБОРКЕ: СТАРЫЙ ПРОТИВ НОВОГО (§6.220, 25.09).

Выборка — `_seq_data_big.py --only-list wellmap_sheets.txt` (листы поля A/B: их скважины новый селектор НЕ видел).
Для каждого чекпойнта: доля решений, где выбран ран, накрывающий эталон; отдельно на ТРУДНЫХ решениях — где ближайший
к предсказанию ран чужой (именно там селектор бросает живую линию, §6.219). Разрез по скважинам — сколько за/против.

  _seq_eval.py --data seq_eval_field.npz --ckpt F:/nds/Auto/auto/models/seq_model_d45p.pt F:/nds/output/taskS/decoder/seq_model_big.pt
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from _decoder_seq import WindowSelector, OUT
from _decoder_seq_data import unpack

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", nargs="+", required=True)
a = ap.parse_args()
d = np.load(OUT / a.data if not Path(a.data).is_absolute() else a.data)
P, F, L, M, W = d["P"], d["F"], d["L"], d["M"], d["wells"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
good = L.sum(1) > 0
near = (F[:, :, 7] * M).argmax(1)
near_ok = L[np.arange(len(L)), near].astype(bool)
hard = good & ~near_ok
print(f"решений {len(P)} (с верным раном {int(good.sum())}), скважин {len(set(W.tolist()))}; "
      f"ближайший прав {100*near_ok[good].mean():.1f}% ⇒ трудных {int(hard.sum())}")
res = {}
for ck in a.ckpt:
    net = WindowSelector().to(dev)
    net.load_state_dict(torch.load(ck, map_location=dev)["sd"]); net.eval()
    pick = np.zeros(len(P), np.int64)
    with torch.no_grad():
        for s in range(0, len(P), 4096):
            ii = np.arange(s, min(s + 4096, len(P)))
            p = torch.from_numpy(np.stack([unpack(P[i]) for i in ii])).to(dev)
            sc = net(p, torch.from_numpy(F[ii]).to(dev), torch.from_numpy(M[ii].astype(np.float32)).to(dev))
            sc = sc[0] if isinstance(sc, tuple) else sc
            pick[ii] = sc.argmax(1).cpu().numpy()
    ok = L[np.arange(len(L)), pick].astype(bool)
    res[ck] = ok
    print(f"  {Path(ck).name:<28} верно {100*ok[good].mean():.2f}%  на трудных {100*ok[hard].mean():.2f}%  "
          f"(ошибок {int((~ok & good).sum())})")
if len(a.ckpt) == 2:
    o, n = (res[c] for c in a.ckpt)
    byw = defaultdict(lambda: [0, 0])
    for w, x, y, g in zip(W.tolist(), o, n, good):
        if g:
            byw[w][0] += int(x); byw[w][1] += int(y)
    up = sum(1 for v in byw.values() if v[1] > v[0]); dn = sum(1 for v in byw.values() if v[1] < v[0])
    fixed = int((~o & n & good).sum()); broke = int((o & ~n & good).sum())
    print(f"★ новый против старого: исправлено решений {fixed}, сломано {broke}; скважин лучше/хуже {up}/{dn}")

r"""_seqxl_parity.py — СВЕРКА ПРОД-СЕТИ (`auto/trace_seq.py`) С СЕТЬЮ СТЕНДА (`_decoder_seq.py`) ДЛЯ ВЕСОВ С ИСТОРИЕЙ И XL (§6.233).

Прод дублирует определение сети (не импортирует стенды). Для веса с полями `hist` / `arch` сверяются скоры обеих реализаций
на одних весах и одних случайных входах (P, F, M, H с пустыми и заполненными строками истории): max|Δ| < 1e-5 — иначе прод
считал бы не то, на чём учили.

  _seqxl_parity.py seqh_model_dag.pt seqxl_model_dag.pt
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
import numpy as np
import torch, torch.nn as nn
from auto import trace_seq as TS
from _decoder_seq import make_net, OUT
from _decoder_seq_data import NROW, NCOL, MAXC

rng = np.random.default_rng(0)
code = 0
for name in sys.argv[1:]:
    ck = torch.load(OUT / name, map_location="cpu", weights_only=False)
    arch, hist = ck.get("arch", "base"), bool(ck.get("hist", False))
    a = make_net(arch, hist); a.load_state_dict(ck["sd"]); a.eval()
    b = (TS._build_net_xl(torch, nn)(hist=hist) if arch == "xl" else TS._build_net(torch, nn, hist=hist)())
    b.load_state_dict(ck["sd"]); b.eval()
    B = 64
    P = torch.from_numpy((rng.random((B, 2, NROW, NCOL)) > 0.7).astype(np.float32))
    F = torch.from_numpy(rng.normal(0, 0.5, (B, MAXC, 10)).astype(np.float32))
    M = torch.from_numpy((rng.random((B, MAXC)) > 0.2).astype(np.float32)); M[:, 0] = 1
    H = torch.from_numpy(np.where(rng.random((B, NROW)) > 0.5, rng.integers(0, NCOL, (B, NROW)), -1).astype(np.int64))
    with torch.no_grad():
        sa = a(P, F, M, H=H if hist else None)
        sb = b(P, F, M, H) if hist else b(P, F, M)
    d = float((sa - sb).abs().max())
    ok = d < 1e-5
    code |= (not ok)
    print(f"  {name}: arch={arch}, hist={hist}: max|Δ| = {d:.2e}  {'★ совпадает' if ok else '⛔ РАСХОДИТСЯ'}")
sys.exit(2 if code else 0)

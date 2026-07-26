r"""gen_golden.py — сгенерировать эталоны для auto/trace_seq.selftest().

★ ГЛАВНОЕ: все эталоны считаются ОБУЧАЮЩЕЙ стороной (digitizer/rnd/_decoder_core.features,
_decoder_seq_data.patch, _decoder_seq.WindowSelector). Если бы их считала сама прод-копия, тест
сверял бы копию с собой и дрейф не ловил бы вовсе.

Входы подобраны, а не случайны (иначе тест ничего не различает):
  ПРИЗНАКИ  — есть ран, накрывающий pred (проверяет overlap и допуск ±2); есть ран шире 20px
              (проверяет width/20); argmax(width) НЕ совпадает с argmin|c-pred| (иначе колонки
              is_widest и is_nearest неразличимы — самая вероятная ошибка копирования);
              ранов больше 1, но меньше MAXC (проверяет паддинг F/M и признак len(A)/5);
              v ненулевая (проверяет vsmooth).
  ПАТЧ      — точка у края ПО ОБЕИМ осям сразу (проверяет valid и обнуление ink за краем);
              отдельный случай с pred=.5 (round() в Python банковское, а не «вверх»).
"""
import sys, json, hashlib
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")

import torch
from _decoder_core import features as ref_features
from _decoder_seq_data import patch as ref_patch, NROW, NCOL, MAXC, COL_STEP
from _decoder_seq import WindowSelector, NF

CKPT = r"F:\nds\Auto\auto\models\seq_model_d45p.pt"

# ── случай ПРИЗНАКОВ ────────────────────────────────────────────────────────────────────────
A = [40, 95, 150, 300]
B = [46, 100, 185, 306]          # ширины 6, 5, 35, 6 → самый широкий = индекс 2
C = [43.0, 97.5, 167.5, 303.0]   # ближайший к pred=98 → индекс 1 ≠ самый широкий
PRED, X0, V0, BASE = 98.0, 96.0, 3.5, 90.0
order, X = ref_features(np.array(A), np.array(B), np.array(C, float), PRED, X0, V0, BASE)
assert 1 < len(order) < MAXC, len(order)
assert int(np.argmax(np.array(B) - np.array(A))) != int(np.argmin(np.abs(np.array(C) - PRED)))

# ── случай ПАТЧА: маленькая полоса, точка у края по обеим осям ───────────────────────────────
H, WB = 40, 30
yy, xx = np.mgrid[0:H, 0:WB]
band = (np.abs(xx - (12 + 6 * np.sin(yy / 7.0))) < 2.0)      # аналитический штрих, без rng
ink, val = ref_patch(band, 0, 5, 3.0)
ink2, val2 = ref_patch(band, 0, 20, 10.5)                    # дробный pred → округление

# ── эталон СЕТИ ─────────────────────────────────────────────────────────────────────────────
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
net = WindowSelector().eval()
net.load_state_dict(ck["sd"])
P = torch.from_numpy(np.stack([ink, val]).astype(np.float32)[None])
F = torch.zeros(1, MAXC, NF); F[0, :len(order)] = torch.from_numpy(X.astype(np.float32))
M = torch.zeros(1, MAXC); M[0, :len(order)] = 1
with torch.no_grad():
    sc = net(P, F, M)[0].numpy()

golden = {
    "_": "Эталоны для auto/trace_seq.selftest(). СГЕНЕРИРОВАНЫ обучающей стороной "
         "(digitizer/rnd/_decoder_seq.py и _decoder_core.py) — см. scratchpad/gen_golden.py. "
         "Пересобирать ТОЛЬКО вместе со сменой чекпойнта.",
    "ckpt_sha256": hashlib.sha256(open(CKPT, "rb").read()).hexdigest(),
    "geom": ck["geom"],
    "n_params": int(sum(t.numel() for t in ck["sd"].values())),
    "sd_keys": sorted(ck["sd"].keys()),
    "features": {"A": A, "B": B, "C": C, "pred": PRED, "x": X0, "v": V0, "base": BASE,
                 "order": order.tolist(), "X": X.tolist()},
    "patch": {"H": H, "Wb": WB, "y": 5, "pred": 3.0,
              "shape": list(ink.shape), "ink_sum": int(ink.sum()), "val_sum": int(val.sum()),
              "ink_outside_val": int((ink & ~val).sum()),
              "ink_md5": hashlib.md5(np.packbits(ink).tobytes()).hexdigest()},
    "patch_round": {"y": 20, "pred": 10.5, "ink_sum": int(ink2.sum()),
                    "val_sum": int(val2.sum()),
                    "ink_md5": hashlib.md5(np.packbits(ink2).tobytes()).hexdigest()},
    "scores": [float(v) for v in sc],
}
out = r"F:\nds\Auto\auto\models\seq_model_d45p.golden.json"
open(out, "w", encoding="utf-8").write(json.dumps(golden, ensure_ascii=False, indent=1))
print("geom", ck["geom"], "| параметров", golden["n_params"])
print("order", golden["features"]["order"])
print("X[0]", [round(v, 4) for v in X[0]])
print("патч: shape", ink.shape, "ink", ink.sum(), "val", val.sum(),
      "ink вне val", golden["patch"]["ink_outside_val"])
print("скоры", [round(float(v), 6) for v in sc])
print("\n->", out)

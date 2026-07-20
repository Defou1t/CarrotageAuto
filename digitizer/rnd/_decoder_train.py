r"""_decoder_train.py — ОБУЧЕНИЕ селектора ранов + ЧЕСТНЫЙ ГЕЙТ на держанных скважинах (bench).

Пайплайн P2-прототипа: train.npz (teacher-forcing) → логистика → обучаемый трассировщик →
стенд bench (5 держанных скважин, 24 кривые) с ТОЙ ЖЕ метрикой, что база: честных (med<=3 И
cov>=0.9), med(med), своя% (идентичность). База: 3/24, med 68.3, своя 34.9%.

Санити ДО гейта: на ОТЛОЖЕННЫХ решениях (group-split) — берёт ли модель правильный ран чаще,
чем «ближайший к pred» (что делает база). Если нет — гейт можно не гонять.

  python _decoder_train.py
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
import _relatch_bench as BE
from _decoder_core import Logistic, make_tracer, features, FEAT_NAMES

OUT = Path(r"F:\nds\output\taskS\decoder")
d = np.load(OUT / "train.npz")
X, y, g = d["X"], d["y"], d["g"]
groups = np.unique(g)
# индекс «ближайший к pred» внутри решения = признак is_nearest (feat 7)
NEAR = FEAT_NAMES.index("is_nearest")

# multi-candidate решения = где есть выбор (иначе латч невозможен)
gsz = {gi: int((g == gi).sum()) for gi in groups}
multi = np.array([gi for gi in groups if gsz[gi] >= 2])
print(f"данные: {len(X)} кандидатов, {len(groups)} решений, из них с выбором (>=2 канд): {len(multi)}")
# на multi-решениях: как часто НАИБЛИЖАЙШИЙ (база) НЕ прав?
near_wrong = 0
for gi in multi:
    m = g == gi
    xi, yi = X[m], y[m]
    kn = int(np.argmax(xi[:, NEAR]))
    if yi[kn] == 0:
        near_wrong += 1
print(f"  на multi-решениях база (ближайший) БЕРЁТ ЧУЖОЙ ран: {near_wrong}/{len(multi)} "
      f"= {100*near_wrong/max(1,len(multi)):.1f}%  <- потолок выигрыша селектора")

# group-split 80/20 для честной оценки accuracy
rngkey = (g.astype(np.int64) * 2654435761) % 100          # детерминированный хэш группы
tr_mask = rngkey < 80
va_mask = ~tr_mask
model = Logistic().fit(X[tr_mask], y[tr_mask])
print(f"\nвеса модели: " + ", ".join(f"{n}={w:+.2f}" for n, w in zip(FEAT_NAMES, model.w)) +
      f"  b={model.b:+.2f}")

# accuracy выбора на ОТЛОЖЕННЫХ multi-решениях: модель vs ближайший
va_groups = [gi for gi in multi if (rngkey[g == gi][0] >= 80)]
mod_ok = near_ok = 0
for gi in va_groups:
    m = g == gi
    xi, yi = X[m], y[m]
    km = int(np.argmax(model.score(xi)))
    kn = int(np.argmax(xi[:, NEAR]))
    mod_ok += int(yi[km] == 1); near_ok += int(yi[kn] == 1)
n = max(1, len(va_groups))
print(f"\nОТЛОЖЕННЫЕ multi-решения ({len(va_groups)}):")
print(f"  берёт СВОЙ ран: модель {100*mod_ok/n:.1f}%   база(ближайший) {100*near_ok/n:.1f}%")

model.to_dict()
(OUT / "model.json").write_text(json.dumps(model.to_dict(), ensure_ascii=False), encoding="utf-8")

# ★ ЧЕСТНЫЙ ГЕЙТ на держанных 5 скважинах (bench) — та же метрика, что база
print(f"\n{'='*66}\n=== ГЕЙТ на держанных скважинах (bench, 24 кривые) ===")
BE.report("БАЗА (прод)", BE.run_strategy())
BE.report("ОБУЧАЕМЫЙ селектор", BE.run_strategy(tracer=make_tracer(model)))
print("\nЧитать: если у обучаемого честных > 3 И своя% > 34.9 без обвала med(cov) — селектор")
print("бьёт латч на держанных скважинах, и это оправдывает переход к тяжёлому декодеру (torch).")

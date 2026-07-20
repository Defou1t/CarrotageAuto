r"""_decoder_train.py — ОБУЧЕНИЕ селектора ранов + ЧЕСТНЫЙ ГЕЙТ на держанных скважинах (bench).

Пайплайн P2-прототипа: train.npz (teacher-forcing) → логистика → обучаемый трассировщик →
стенд bench (5 держанных скважин, 24 кривые) с ТОЙ ЖЕ метрикой, что база: честных (med<=3 И
cov>=0.9), med(med), своя% (идентичность). База: 3/24, med 68.3, своя 34.9%.

Все групповые операции ВЕКТОРНЫ (2M решений): argmax-в-группе через lexsort + границы
сегментов, НЕ маской `g==gi` в цикле (это было O(N*G) и вешало прогон).

  python _decoder_train.py
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
import _relatch_bench as BE
from _decoder_core import Logistic, make_tracer, FEAT_NAMES

OUT = Path(r"F:\nds\output\taskS\decoder")
d = np.load(OUT / "train.npz")
X, y, g = d["X"], d["y"], d["g"].astype(np.int64)
NEAR = FEAT_NAMES.index("is_nearest")


def group_bounds(gsorted):
    """Индексы ПОСЛЕДНЕГО элемента каждой группы в отсортированном по группе массиве."""
    return np.r_[np.nonzero(np.diff(gsorted))[0], len(gsorted) - 1]


def argmax_label_per_group(g, score, y):
    """Для каждой группы: метка кандидата с МАКС. score. Векторно (lexsort по (g, score))."""
    order = np.lexsort((score, g))
    gs = g[order]
    last = group_bounds(gs)
    return y[order][last], gs[last]


# размеры групп векторно
order0 = np.argsort(g, kind="stable")
gs0 = g[order0]
bnd = group_bounds(gs0)
starts = np.r_[0, bnd[:-1] + 1]
sizes = bnd - starts + 1
gid = gs0[bnd]
multi_gids = gid[sizes >= 2]
print(f"данные: {len(X)} кандидатов, {len(gid)} решений, с выбором (>=2 канд): {len(multi_gids)}")

# на multi-решениях: как часто НАИБЛИЖАЙШИЙ (база) берёт ЧУЖОЙ ран (потолок выигрыша)
lbl_near, gg = argmax_label_per_group(g, X[:, NEAR], y)
is_multi = np.isin(gg, multi_gids)
near_wrong = int((lbl_near[is_multi] == 0).sum())
print(f"  на multi-решениях база (ближайший) БЕРЁТ ЧУЖОЙ ран: {near_wrong}/{int(is_multi.sum())} "
      f"= {100*near_wrong/max(1,int(is_multi.sum())):.1f}%  <- потолок выигрыша селектора")

# group-split 80/20 детерминированным хэшем группы
rngkey = (g * 2654435761) % 100
tr_mask = rngkey < 80
model = Logistic().fit(X[tr_mask], y[tr_mask], iters=300)
print(f"\nвеса: " + ", ".join(f"{n}={w:+.2f}" for n, w in zip(FEAT_NAMES, model.w)) + f"  b={model.b:+.2f}")

# accuracy на ОТЛОЖЕННЫХ multi-решениях: модель vs ближайший
va = rngkey >= 80
sc_mod = model.score(X)
lm, gm = argmax_label_per_group(g[va], sc_mod[va], y[va])
ln, gn = argmax_label_per_group(g[va], X[va][:, NEAR], y[va])
mm = np.isin(gm, multi_gids)
print(f"\nОТЛОЖЕННЫЕ multi-решения ({int(mm.sum())}):")
print(f"  берёт СВОЙ ран: модель {100*lm[mm].mean():.1f}%   база(ближайший) {100*ln[np.isin(gn,multi_gids)].mean():.1f}%")

(OUT / "model.json").write_text(json.dumps(model.to_dict(), ensure_ascii=False), encoding="utf-8")

print(f"\n{'='*66}\n=== ГЕЙТ на держанных скважинах (bench, 24 кривые) ===")
BE.report("БАЗА (прод)", BE.run_strategy())
BE.report("ОБУЧАЕМЫЙ селектор", BE.run_strategy(tracer=make_tracer(model)))
print("\nЧитать: честных > 3 И своя% > 34.9 без обвала med(cov) -> селектор бьёт латч на")
print("держанных скважинах, что оправдывает переход к тяжёлому декодеру (torch, ComfyUI venv).")

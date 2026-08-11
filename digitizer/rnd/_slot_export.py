r"""_slot_export.py — ОБУЧИТЬ РАСКЛАДКУ И ВЫГРУЗИТЬ ЕЁ В NUMPY (без sklearn в проде).

§6.88/§6.89 померили механизм на пулах. Чтобы получить цифру ОТГРУЗКИ (память проекта: стенд и
отгрузка расходятся в разы), механизм надо перенести в прод. Здесь готовится вес.

⚠ ПОЧЕМУ НЕ pickle МОДЕЛИ. sklearn есть только в embedded-python (1.8.0), в проектном 3.14.5 его
нет, а pickle sklearn-модели ломается между версиями библиотеки. Поэтому выгружаются САМИ ДЕРЕВЬЯ
(feature/threshold/дети/значения) + база и learning_rate: предсказание — 20 строк на numpy, который
у прода и так есть. ⇒ у прода НЕ появляется новой зависимости, в отличие от torch в §6.68.

⚠ ЭКСПОРТ СЧИТАЕТСЯ ВЕРНЫМ, ТОЛЬКО ЕСЛИ СОВПАЛ С sklearn ПОБИТОВО (max|Δ| < 1e-9 на ВСЕХ парах).
Иначе прод будет считать не то, что мерил стенд, и это не всплывёт никогда.

★ АУДИТОРСКИЕ СКВАЖИНЫ ИСКЛЮЧАЮТСЯ ИЗ ОБУЧЕНИЯ: на них потом мерится отгружаемый путь. Список
объявлен ЗДЕСЬ и до прогона (§6.49) — это те скважины, на которых §6.89 (LOWO) показал движение в
обе стороны, включая ДВЕ деградировавшие, чтобы замер не был вишенкой.

  <ComfyUI>\python_embeded\python.exe _slot_export.py
"""
import sys, json, hashlib, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# ⚠ §6.109: кэш и имя веса — АРГУМЕНТАМИ. Корпус вырос с 702 листов до 2677, и переобучение на нём
# обязано класть НОВЫЙ файл: перезаписать `slot_model_g250.npz` значило бы, что все прежние замеры
# ветки задним числом относятся неизвестно к чему.
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--cache", default="F:/nds/output/taskS/_slot_abstain_cache_v2.pkl")
_ap.add_argument("--name", default="slot_model_g250.npz")
_a = _ap.parse_args()
CACHE = _a.cache
OUT = Path(r"F:\nds\Auto\auto\models")
NAME = _a.name

# ★ ОБЪЯВЛЕНО ДО ПРОГОНА: скважины, на которых будет мериться ОТГРУЗКА, в обучение не идут.
AUDIT = ["BOGAT_011", "BOGAT_014", "VILHIV_055", "RYBAL_058",
         "LOBACH_032", "KREMEN_057", "RYBAL_018", "RYBAL_168"]
FEATS = ["color_eq", "color_none", "class_ok", "class_sp", "rank", "x_center", "med",
         "wig", "rev", "rough", "span", "npts", "n_lines", "dy"]

C = pickle.load(open(CACHE, "rb"))
X, R, IDX, wells = C["X"], C["R"], C["IDX"], C["wells"]
aud = set(AUDIT)
tr_p = np.array([wells[si] not in aud for si, _, _ in IDX])
print(f"пар всего {len(R)}, на обучение {int(tr_p.sum())}, "
      f"аудиторских скважин {len(aud)} ({int((~tr_p).sum())} пар отложено)")
miss = aud - set(wells)
if miss:
    print(f"⚠ НЕТ В ДАННЫХ: {sorted(miss)}")

est = GradientBoostingRegressor(n_estimators=250, max_depth=3, random_state=0)
est.fit(X[tr_p], R[tr_p])
print(f"обучено: {est.n_estimators_} деревьев, learning_rate {est.learning_rate}")

# ── выгрузка деревьев ─────────────────────────────────────────────────────────────────────────
CL, CR, FE, TH, VA, OFF = [], [], [], [], [], [0]
for stage in est.estimators_:
    t = stage[0].tree_
    CL.append(t.children_left.astype(np.int32)); CR.append(t.children_right.astype(np.int32))
    FE.append(t.feature.astype(np.int32)); TH.append(t.threshold.astype(np.float64))
    VA.append(t.value.reshape(-1).astype(np.float64))
    OFF.append(OFF[-1] + t.node_count)
cl, cr, fe, th, va = (np.concatenate(v) for v in (CL, CR, FE, TH, VA))
off = np.array(OFF, np.int32)
lr = float(est.learning_rate)


def walk(Xq):
    """Предсказание БЕЗ sklearn: ровно то, что будет в проде (auto/slot_model.py).

    ⚠⚠ ПРИЗНАКИ ПРИВОДЯТСЯ К float32. sklearn внутри деревьев работает с float32 (DTYPE), поэтому
    сравнение с порогом идёт в ЕГО точности; в float64 строки, стоящие у самого порога, уходят в
    другую ветвь. Первая редакция экспорта на этом и разошлась: max|Δ| = 4e-2 при 22 тыс. пар."""
    Xq = np.asarray(Xq, np.float32).astype(np.float64)
    out = np.zeros(len(Xq))
    rows = np.arange(len(Xq))
    for k in range(len(off) - 1):
        b = off[k]
        node = np.zeros(len(Xq), np.int32)
        while True:
            f = fe[b + node]
            leaf = f < 0
            if leaf.all():
                break
            go_l = Xq[rows, np.where(leaf, 0, f)] <= th[b + node]
            nxt = np.where(go_l, cl[b + node], cr[b + node])
            node = np.where(leaf, node, nxt)
        out += va[b + node]
    return base + lr * out


# база: predict = base + lr*Σдеревья ⇒ base восстанавливается из одной строки, проверяется на всех
base = 0.0
base = float(est.predict(X[:1])[0] - (walk(X[:1])[0]))
ref = est.predict(X)
got = walk(X)
d = float(np.max(np.abs(ref - got)))
print(f"★ СВЕРКА С sklearn на всех {len(X)} парах: max|Δ| = {d:.3e}")
if d > 1e-9:
    print("⛔ ЭКСПОРТ НЕВЕРЕН — прод считал бы не то, что стенд. Ничего не записано.")
    sys.exit(1)

OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / NAME, children_left=cl, children_right=cr, feature=fe, threshold=th,
                    value=va, offsets=off, lr=np.array([lr]), base=np.array([base]),
                    n_features=np.array([X.shape[1]]))
# ★★ ВТОРАЯ СВЕРКА — С ОТГРУЖАЕМОЙ ФУНКЦИЕЙ, А НЕ ТОЛЬКО СО СТЕНДОВОЙ. Выше сверялся `walk` из
# этого файла; в прод же пойдёт `auto.slot_model.predict`, и до §6.90 его со sklearn не сверял никто
# (нашёл скептик wf_110d68ca-bbb). Совпадение двух РАЗНЫХ реализаций — то, что нужно проверять.
sys.path.insert(0, r"F:\nds\Auto")
from auto import slot_model as SM                                              # noqa: E402
d2 = float(np.max(np.abs(ref - (-SM.predict(SM.load(OUT / NAME), X)))))
print(f"★ СВЕРКА ПРОД-ФУНКЦИИ auto.slot_model.predict с sklearn: max|Δ| = {d2:.3e}")
if d2 > 1e-9:
    (OUT / NAME).unlink()
    print("⛔ ПРОД СЧИТАЛ БЫ НЕ ТО. Вес удалён, ничего не записано.")
    sys.exit(1)
np.savez_compressed(OUT / NAME, children_left=cl, children_right=cr, feature=fe, threshold=th,
                    value=va, offsets=off, lr=np.array([lr]), base=np.array([base]),
                    n_features=np.array([X.shape[1]]))
sha = hashlib.sha256((OUT / NAME).read_bytes()).hexdigest()[:16]
man = dict(name=NAME, sha256_16=sha, prod_predict_maxdiff=d2,
           trees=int(len(off) - 1), max_depth=3, lr=lr, base=base,
           n_features=int(X.shape[1]), features=FEATS,
           trained_pairs=int(tr_p.sum()), trained_wells=sorted(set(wells) - aud),
           audit_wells=sorted(aud), sklearn_maxdiff=d,
           recipe=dict(sheet_measure="frac", sheet_threshold=0.8, margin_for_frac=0.3,
                       single_candidate="pred_err", single_candidate_thr=2.0),
           roadmap="§6.88 (механизм), §6.89 (проверка), §6.90 (отгрузка)")
(OUT / (NAME.replace(".npz", ".json"))).write_text(json.dumps(man, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
print(f"записано: {OUT / NAME}  ({(OUT / NAME).stat().st_size // 1024} КБ, sha {sha})")
print(f"манифест: {OUT / NAME.replace('.npz', '.json')}")
print(f"⚠ обучено БЕЗ аудиторских скважин: {sorted(aud)}")

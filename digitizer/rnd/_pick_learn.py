r"""_pick_learn.py — МОЖНО ЛИ ВЫБИРАТЬ ПУТЬ ВЕДЕНИЯ ЛУЧШЕ, ЧЕМ ПОРОГОМ (§6.144 → §6.145)

ОТКУДА ВОПРОС. §6.144: прод и построчный декодер расходятся на 779 кривых из 2760, и выбор между
ними по треку даёт **+88 кривых на держанных скважинах** — но правило состоит из ОДНОГО сравнения
(«декодер, если он повёл ≤3 линии»), а оракульный потолок выбора вдвое выше. То есть три четверти
резерва лежат за более тонким признаком.

★ И ВПЕРВЫЕ ЕСТЬ НА ЧЁМ УЧИТЬСЯ. Раньше выбор пути был гипотезой; теперь есть 1185 треков, у
каждого известно, СКОЛЬКО честных кривых даёт каждый путь, — то есть готовая размеченная выборка,
собранная из уже посчитанных выдач, без единого нового прогона.

ЧТО ДЕЛАЕТ СТЕНД. По готовым выдачам двух режимов собирает признаки ТРЕКА, доступные на выдаче
(эталон в них не входит НИГДЕ), и учит логистическую регрессию «брать декодер или прод».
Проверка — перекрёстная ПО СКВАЖИНАМ (§6.87: по листам модель учит бланк, а не правило).
Сравнивается с четырьмя опорами сразу: всегда прод, всегда декодер, порог §6.144, оракул.

⚠⚠ ЧЕСТНОСТЬ ПРИЗНАКОВ. Ни один признак не смотрит в эталон. Метка — смотрит, на то она и метка.
Если признак случайно окажется функцией эталона, вся таблица станет враньём, поэтому список
признаков короткий и каждый прокомментирован тем, откуда он берётся на выдаче.

  python _pick_learn.py --dir F:/nds/output/taskS/ab_rowdec_pair --a A --b B
"""
import sys, argparse, pickle, hashlib, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:/nds/output/taskS/ab_rowdec_pair")
ap.add_argument("--a", default="A")
ap.add_argument("--b", default="B")
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--cache", default=r"F:/nds/output/taskS/pick_learn.pkl")
ap.add_argument("--rebuild", action="store_true")
# ⚠⚠ §6.146: признак, которого механизм НЕ ВИДИТ в момент решения, не признак, а ловушка.
# `n_lines` берётся из `_understanding.json` (все AUTO-линии трека), а `emit` в момент выбора
# знает только НАЗНАЧЕННЫЕ слоты — линия без слота туда не попадает. Поэтому предусмотрен
# отказ от признаков: модель обязана считаться ровно на том, что доступно на месте.
ap.add_argument("--drop", nargs="*", default=[], help="исключить признаки по имени")
ap.add_argument("--export", default="", help="куда выгрузить веса (.npz) для прода")
# ★★★ ДЕРЖАННАЯ ВЫГРУЗКА ДЛЯ ЗАМЕРА НА ОТГРУЗКЕ. `--export` пишет ПОЛНУЮ подгонку (по всем
# скважинам) — её нельзя мерить на тех же листах: это вес НА СВОЁМ ПОЛЕ, ровно ошибка §6.114,
# из-за которой три недели держался вывод, которого замер не поддерживал. Держанное число 1126
# считается ЗДЕСЬ перекрёстно, а на отгрузке лист обязан считаться весом, НЕ ВИДЕВШИМ ЕГО
# СКВАЖИНУ. ⇒ выгружаем ПО ВЕСУ НА ФОЛД плюс карту «скважина → фолд», и прогон идёт пятью
# кусками. Та же схема, что у `auto5` (§6.70): ИЗМЕРИТЕЛЬНАЯ, а не прод-схема.
ap.add_argument("--export-folds", default="", help="каталог: вес на каждый фолд + folds.json")
# ★★★ §6.147: ПРИЗНАКИ ИЗ САМОГО МЕХАНИЗМА. `emit` кладёт рядом с выдачей `<лист>_pick.json`
# с признаками РОВНО в том виде, в каком их видит решение. Учиться надо на них, а не на
# пересчитанных по выданным кривым: пересчёт живёт в другом пространстве, и модель, обученная
# на нём, в прогоне брала декодер на 17 треках из 17 вместо 9.
ap.add_argument("--from-dump", default="", help="каталог прогона с <лист>_pick.json")
a = ap.parse_args()
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


def match(rows, cols, ok):
    pair = {}

    def try_(r, seen):
        for c in cols:
            if not ok.get((r, c)) or c in seen:
                continue
            seen.add(c)
            if c not in pair or try_(pair[c], seen):
                pair[c] = r
                return True
        return False
    return sum(1 for r in rows if try_(r, set()))


FEATS = [
    "n_dec",        # сколько линий повёл ДЕКОДЕР на треке — единственный признак правила §6.144
    "n_prod",       # сколько линий повёл прод (может отличаться: короткие трассы отсеиваются)
    "n_lines",      # сколько AUTO-линий нашёл U1 на треке
    "agree_med",    # медиана расхождения путей по парам одного имени, px (лог)
    "agree_frac",   # доля пар, где пути сошлись ≤3px
    "cov_dec",      # средняя доля строк трека, покрытая трассой декодера
    "cov_prod",     # то же у прода
    "len_ratio",    # медиана длин трасс декодера / прода
    "spread",       # разброс медиан x трасс декодера, нормированный на ширину трека (лог)
]


def build():
    SRC, WELL = {}, {}
    for r in a.roots:
        for wlg in Path(r).glob("*/wlg"):
            for q in wlg.glob("*.nlgx"):
                SRC.setdefault(q.name, q); WELL.setdefault(q.name, wlg.parent.name)
    TRACK, seen = {}, set()
    for root in a.pools:
        for f in sorted(Path(root).glob("*.pkl")):
            if f.stem in seen:
                continue
            seen.add(f.stem)
            d = pickle.load(open(f, "rb"))
            TRACK[d["name"]] = {s["name"]: s["track"] for s in d["slots"]}
    BY = {f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}": nm
          for nm, q in SRC.items()}
    root = Path(a.dir)

    def curves(p):
        return {c["name"]: dense(c) for c in extract(str(p))["curves"]
                if M.mnem_root(c["name"]) != "DA"}

    rows, skip = [], Counter()
    for d in sorted((root / a.b).iterdir()):
        if not d.is_dir():
            continue
        nm = BY.get(d.name); dA = root / a.a / d.name
        if nm is None or not dA.is_dir():
            skip["нет базы"] += 1; continue
        fA = next(iter(sorted(dA.glob("*_auto.nlgx"))), None)
        fB = next(iter(sorted(d.glob("*_auto.nlgx"))), None)
        if not (fA and fB):
            skip["нет выдачи"] += 1; continue
        gts = {c["name"]: dense(c) for c in extract(str(SRC[nm]))["curves"]
               if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        if not gts:
            skip["нет эталона"] += 1; continue
        A_, B_ = curves(fA), curves(fB)
        lines = Counter()
        u = next(iter(sorted(d.glob("*_understanding.json"))), None)
        if u is not None:
            try:
                for L in json.loads(u.read_text(encoding="utf-8")).get("lines", []):
                    if L.get("confidence") == "AUTO":
                        lines[L.get("track")] += 1
            except Exception:
                pass
        tmap = TRACK.get(nm, {})
        for ti in {tmap.get(g) for g in gts}:
            if ti is None:
                continue
            G = [g for g in gts if tmap.get(g) == ti]
            # ⚠ ПУСТЫЕ ТРАССЫ ОТСЕИВАЮТСЯ ЗДЕСЬ, а не позже: в выдаче встречается кривая с нулём
            # точек (слот назначен, писать нечего), и она валила подсчёт разброса. Считать её
            # «повёл линию» тоже неверно — признак `n_dec` тогда врал бы в большую сторону.
            WA = [k for k in A_ if tmap.get(k) == ti and A_[k]]
            WB = [k for k in B_ if tmap.get(k) == ti and B_[k]]
            if not G or (not WA and not WB):
                continue
            hA = match(G, WA, {(g, w): HON(*err(A_[w], gts[g])) for g in G for w in WA}) if WA else 0
            hB = match(G, WB, {(g, w): HON(*err(B_[w], gts[g])) for g in G for w in WB}) if WB else 0
            # ── признаки; эталон НЕ используется ни в одном ───────────────────────────────
            comm = [k for k in WB if k in A_]
            dif = []
            for k in comm:
                cm = [y for y in B_[k] if y in A_[k]]
                if len(cm) >= 30:
                    dif.append(float(np.median([abs(B_[k][y] - A_[k][y]) for y in cm])))
            rowsB = [len(B_[k]) for k in WB] or [0]
            rowsA = [len(A_[k]) for k in WA] or [0]
            xs = [x for k in WB for x in B_[k].values()]
            span = max(1.0, (max(xs) - min(xs)) if xs else 1.0)
            med = [float(np.median(list(B_[k].values()))) for k in WB]
            f = dict(
                n_dec=len(WB), n_prod=len(WA), n_lines=lines.get(ti, 0),
                agree_med=float(np.log1p(np.median(dif))) if dif else float(np.log1p(1000.0)),
                agree_frac=float(np.mean([x <= 3 for x in dif])) if dif else 0.0,
                cov_dec=float(np.median(rowsB)), cov_prod=float(np.median(rowsA)),
                len_ratio=float(np.median(rowsB) / max(1.0, np.median(rowsA))),
                spread=float(np.log1p(np.std(med) / span)) if len(med) > 1 else 0.0)
            rows.append((nm, WELL.get(nm, "?"), ti, f, hA, hB))
    print(f"★ СВЕРКА: треков собрано {len(rows)}; пропущено {dict(skip)}")
    return rows


def build_from_dump():
    """Признаки — из `<лист>_pick.json` прогона; метки — из ЗАМОРОЖЕННЫХ выдач A и B.
    ⚠ Ключ — (лист, трек), и он один и тот же с обеих сторон: индекс трека берётся из рамки."""
    SRC, WELL = {}, {}
    for r in a.roots:
        for wlg in Path(r).glob("*/wlg"):
            for q in wlg.glob("*.nlgx"):
                SRC.setdefault(q.name, q); WELL.setdefault(q.name, wlg.parent.name)
    TRACK, seen = {}, set()
    for root in a.pools:
        for f in sorted(Path(root).glob("*.pkl")):
            if f.stem in seen:
                continue
            seen.add(f.stem)
            d = pickle.load(open(f, "rb"))
            TRACK[d["name"]] = {s2["name"]: s2["track"] for s2 in d["slots"]}
    BY = {f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}": nm
          for nm, q in SRC.items()}
    base = Path(a.dir)

    def curves(p2):
        return {c["name"]: dense(c) for c in extract(str(p2))["curves"]
                if M.mnem_root(c["name"]) != "DA"}

    rows, skip = [], Counter()
    for d in sorted(Path(a.from_dump).iterdir()):
        if not d.is_dir():
            continue
        nm = BY.get(d.name)
        pj = next(iter(sorted(d.glob("*_pick.json"))), None)
        dA, dB = base / a.a / d.name, base / a.b / d.name
        if nm is None or pj is None or not (dA.is_dir() and dB.is_dir()):
            skip["нет пары дамп/выдача"] += 1; continue
        fA = next(iter(sorted(dA.glob("*_auto.nlgx"))), None)
        fB = next(iter(sorted(dB.glob("*_auto.nlgx"))), None)
        if not (fA and fB):
            skip["нет выдачи"] += 1; continue
        gts = {c["name"]: dense(c) for c in extract(str(SRC[nm]))["curves"]
               if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        if not gts:
            skip["нет эталона"] += 1; continue
        A_, B_ = curves(fA), curves(fB)
        tmap = TRACK.get(nm, {})
        try:
            dump = json.loads(pj.read_text(encoding="utf-8"))
        except Exception:
            skip["дамп не читается"] += 1; continue
        for t in dump.get("tracks", []):
            ti = t["track"]
            G = [g for g in gts if tmap.get(g) == ti]
            WA = [k for k in A_ if tmap.get(k) == ti and A_[k]]
            WB = [k for k in B_ if tmap.get(k) == ti and B_[k]]
            if not G:
                continue
            hA = match(G, WA, {(g, w): HON(*err(A_[w], gts[g])) for g in G for w in WA}) if WA else 0
            hB = match(G, WB, {(g, w): HON(*err(B_[w], gts[g])) for g in G for w in WB}) if WB else 0
            f = {k: float(t[k]) for k in FEATS if k in t}
            if len(f) != len(FEATS):
                skip["в дампе нет признака"] += 1; continue
            rows.append((nm, WELL.get(nm, "?"), ti, f, hA, hB))
    print(f"★ СВЕРКА: треков из дампов {len(rows)}; пропущено {dict(skip)}")
    return rows


if a.from_dump:
    FEATS = [k for k in FEATS if k != "n_lines"]     # его в дампе нет и быть не может (§6.147)
    rows = build_from_dump()
    pickle.dump(rows, open(a.cache + ".dump", "wb"))
    print(f"★ выборка из МЕХАНИЗМА: {len(rows)} треков")
elif Path(a.cache).is_file() and not a.rebuild:
    rows = pickle.load(open(a.cache, "rb"))
    print(f"кэш признаков: {len(rows)} треков ({a.cache})")
else:
    rows = build()
    pickle.dump(rows, open(a.cache, "wb"))
    print(f"кэш признаков записан: {a.cache}")

wells = sorted({w for _, w, _, _, _, _ in rows})
fold = {w: i % a.folds for i, w in enumerate(wells)}
if a.drop:
    for k in a.drop:
        if k not in FEATS:
            sys.exit(f"⛔ признака {k!r} нет; есть: {FEATS}")
    FEATS = [k for k in FEATS if k not in a.drop]
    print(f"⚠ ИСКЛЮЧЕНЫ ПРИЗНАКИ {a.drop} ⇒ считается на {len(FEATS)}: {FEATS}")
X = np.array([[r[3][k] for k in FEATS] for r in rows], float)
hA = np.array([r[4] for r in rows], float)
hB = np.array([r[5] for r in rows], float)
F = np.array([fold[r[1]] for r in rows])
print(f"треков {len(rows)}, скважин {len(wells)}, признаков {len(FEATS)}")
print(f"опоры: всегда прод {int(hA.sum())}, всегда декодер {int(hB.sum())}, "
      f"★ оракул {int(np.maximum(hA, hB).sum())}")
thr = np.array([r[3]["n_dec"] for r in rows])
print(f"порог §6.144 (n_dec ≤ 3): {int(np.where(thr <= 3, hB, hA).sum())}")

# ── логистическая регрессия на numpy: метка «декодер лучше», вес = |разность честных| ─────────
# ⚠ ВЕС ОБЯЗАТЕЛЕН. Треков, где пути равны, три четверти, и без веса модель учится на них, а не
# на тех, где выбор что-то стоит. Цена ошибки на треке = сколько кривых она отнимает.
y = (hB > hA).astype(float)
w = np.abs(hB - hA)
mu, sd = X.mean(0), X.std(0) + 1e-9
Z = (X - mu) / sd


def fit(idx, iters=400, lr=0.3, l2=1e-3):
    b = np.zeros(Z.shape[1] + 1)
    Zi = np.c_[Z[idx], np.ones(len(idx))]
    yi, wi = y[idx], w[idx]
    if wi.sum() == 0:
        return b
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Zi @ b))
        g = Zi.T @ (wi * (p - yi)) / max(1e-9, wi.sum()) + l2 * b
        b -= lr * g
    return b


def cv_model(labels):
    """Перекрёстная проверка по скважинам; `labels` — метка «декодер лучше» (подменяется для
    нуль-контроля)."""
    global y
    keep, y = y, labels
    got = 0.0
    for f in range(a.folds):
        tr = np.where(F != f)[0]; te = np.where(F == f)[0]
        if not len(te):
            continue
        b = fit(tr)
        pr = 1 / (1 + np.exp(-(np.c_[Z[te], np.ones(len(te))] @ b)))
        got += float(np.where(pr >= 0.5, hB[te], hA[te]).sum())
    y = keep
    return got


def cv_thr():
    """Та же проверка для ПОРОГА: он подбирается на 4/5 скважин, зачёт на держанной пятой.
    ⚠⚠ БЕЗ ЭТОГО СРАВНЕНИЕ НЕЧЕСТНОЕ. «1080 у порога» снято на тех же данных, где порог и
    выбирался, а обученный выбор считается держанным — то есть сравнивались бы подгонка с
    проверкой. Ровно та подмена, из-за которой §6.114 три недели читался как довод."""
    got = 0.0
    for f in range(a.folds):
        tr = np.where(F != f)[0]; te = np.where(F == f)[0]
        if not len(te):
            continue
        t = max(range(0, 8), key=lambda q: float(np.where(thr[tr] <= q, hB[tr], hA[tr]).sum()))
        got += float(np.where(thr[te] <= t, hB[te], hA[te]).sum())
    return got


held = cv_model(y)
base = max(hA.sum(), hB.sum())
rule_cv = cv_thr()
orac = np.maximum(hA, hB).sum()
print(f"\n★★ ОБУЧЕННЫЙ ВЫБОР, перекрёстная проверка по скважинам ({a.folds} фолдов): {int(held)}")
print(f"   против {int(base)} у лучшего из путей ({held-base:+.0f})")
print(f"   против {int(rule_cv)} у ПОРОГА, проверенного ТАК ЖЕ ({held-rule_cv:+.0f})")
print(f"   потолок (оракул по треку) {int(orac)}: взято {100*(held-base)/max(1,orac-base):.0f}% "
      f"доступного, порогом {100*(rule_cv-base)/max(1,orac-base):.0f}%")

# ⚠⚠ НУЛЬ-КОНТРОЛЬ (§6.20.3): та же машина на ПЕРЕМЕШАННЫХ метках обязана дать примерно опору.
# Если прирост выходит и на шуме — его даёт процедура подсчёта, а не признак.
rng0 = np.random.default_rng(20260824)
nul = [cv_model(rng0.permutation(y)) for _ in range(5)]
ok_null = np.mean(nul) - base < 0.2 * (held - base)
print(f"⚠ нуль-контроль (метки перемешаны, 5 повторов): {int(np.mean(nul))} "
      f"(разброс {int(min(nul))}-{int(max(nul))}) против опоры {int(base)}   "
      f"{'★ шум прироста не даёт' if ok_null else '⛔ ПРОЦЕДУРА ЗАВЫШАЕТ — числа выше негодны'}")

b = fit(np.arange(len(rows)))
print(f"\nвес признака (нормированные единицы, знак «+» = в пользу ДЕКОДЕРА):")
for k, v in sorted(zip(FEATS, b[:-1]), key=lambda z: -abs(z[1])):
    print(f"   {k:<12} {v:+.3f}")
print(f"   {'смещение':<12} {b[-1]:+.3f}")

if a.export:
    # ⚠ Выгрузка в numpy, а не pickle модели: вес обязан читаться БЕЗ sklearn и без torch —
    # прод-путь считает скор numpy'ем (та же дисциплина, что `slot_model_g250.npz`, §6.90).
    np.savez(a.export, w=b[:-1], b0=b[-1], mu=mu, sd=sd,
             feats=np.array(FEATS), pick_thr=np.array([0.5]))
    print(f"\n★ вес выгружен: {a.export}  ({len(FEATS)} признаков)")
    print("   ⚠⚠ ЭТО ПОЛНАЯ ПОДГОНКА — мерить ею те же скважины НЕЛЬЗЯ (§6.114). Для замера на")
    print("      отгрузке брать `--export-folds`.")

if a.export_folds:
    # ⚠ `mu`/`sd` намеренно ОБЩИЕ, как и в `cv_model` выше: держанное число 1126 снято именно так,
    # и расхождение протоколов сделало бы прогон несравнимым с ним. Утечка есть (нормировка видит
    # все строки), она одна и та же с обеих сторон и на порядок меньше эффекта — но она ЕСТЬ.
    outd = Path(a.export_folds); outd.mkdir(parents=True, exist_ok=True)
    chk = 0.0
    for f in range(a.folds):
        tr, te = np.where(F != f)[0], np.where(F == f)[0]
        bf = fit(tr)
        np.savez(outd / f"pick_model_v2_f{f}.npz", w=bf[:-1], b0=bf[-1], mu=mu, sd=sd,
                 feats=np.array(FEATS), pick_thr=np.array([0.5]))
        if len(te):
            pr = 1 / (1 + np.exp(-(np.c_[Z[te], np.ones(len(te))] @ bf)))
            chk += float(np.where(pr >= 0.5, hB[te], hA[te]).sum())
        print(f"   фолд {f}: обучен на {len(tr)} треках, держано {len(te)}")
    json.dump({"folds": a.folds, "feats": FEATS,
               "fold_by_well": {w: int(fold[w]) for w in wells}},
              open(outd / "folds.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # ★★ СВЕРКА: пять выгруженных весов, применённые каждый к своему держанному фолду, обязаны
    # дать РОВНО то же число, что печатает `cv_model` выше. Иначе выгружено не то, что померено.
    ok = abs(chk - held) < 0.5
    print(f"\n★ выгружено {a.folds} весов: {outd}")
    print(f"   СВЕРКА держанного счёта: выгруженные веса дают {int(chk)}, `cv_model` — {int(held)}"
          f"   {'★ СОШЛОСЬ' if ok else '⛔ НЕ СОШЛОСЬ — весами мерить НЕЛЬЗЯ'}")
    if not ok:
        sys.exit(1)

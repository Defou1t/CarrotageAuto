r"""_slot_abstain.py — ОТКАЗ ОТ НАЗНАЧЕНИЯ ПРИ НИЗКОЙ УВЕРЕННОСТИ. Только замер, прод не трогается.

§6.87: обученная раскладка даёт +17% (147 → 172 честных кривых на 674 листах), но гейт §6.38 не
пройден — 63 листа вверх против 37 вниз. Здесь мерится ровно тот размен, который §6.63 принимал
явно: сколько прироста придётся отдать, чтобы деградировавших листов стало НОЛЬ.

⚠ ЧТО ЗНАЧИТ «ОТКАЗ». Пустой слот деградацию НЕ убирает: там, где прод дал честную кривую, а
модель промахнулась, пустой слот тоже хуже прода. Поэтому отказ = **слот остаётся за правилом**
(`emit._map_lines_to_slots`, порядок по `x_center`), а модель переназначает только уверенные слоты.
Обученная раскладка при внесении и есть переопределение действующего правила, так что откат к
правилу послотно реализуем — это замена ключа сортировки в том же цикле, без смены контракта.

ТРИ МЕРЫ УВЕРЕННОСТИ (все — из скоров самой модели, эталон не участвует):
  abs    — предсказанная величина ошибки пары ≤ t (регрессия обучена на log1p(px) + штраф покрытия,
           поэтому t интерпретируем: log1p(3) = 1.386 — «модель обещает попадание в 3px»);
  margin — на слоте лучший скор минус второй ≥ m (мера неоднозначности, а не качества);
  лист   — средний margin назначенных слотов ЛИСТА ≥ m, иначе лист целиком отдаётся правилу
           (прямое попадание в постановку «убрать 37 деградирующих листов»).

ТРИ СПОСОБА ЗАКРЫТЬ СЛОТ, КОТОРЫЙ МОДЕЛЬ НЕ ЗАНЯЛА (`fill`), и что с ними выяснилось:
  none  — как в `_slot_cv.py` §6.87: слот остаётся пустым;
  rule  — правило доигрывает на ОСТАТКЕ (свободные слоты × свободные линии) одним `used`;
  pick  — выбор ПРАВИЛА для слота, БЕЗ учёта занятых линий.
⚠⚠ ПРИ ПОРОГЕ 0 `rule` ТОЖДЕСТВЕННО `none`, И ЭТО НЕ ЗАМЕР, А СЛЕДСТВИЕ ПОСТРОЕНИЯ: кандидаты
слота — ВСЕ линии его трека, жадное 1:1 строит максимальное паросочетание, поэтому у незанятого
слота ни одной свободной линии на треке не остаётся. Проверено на пяти порядках скора (оракул,
антиоракул, три случайных): всегда 0 доигранных слотов. ⇒ `none` цену модели НЕ завышает:
реализуемого отката, который её возвращает, не существует. `rule` начинает работать только при
пороге > 0 и при отказе ЛИСТОМ ЦЕЛИКОМ, где обнуляется весь лист.
⚠ `pick` НЕРЕАЛИЗУЕМ: каждое его очко — линия, зачтённая в двух слотах (73 случая). Первая
редакция этого стенда цитировала его как базу (+34, ↓32) вместо честных +28, ↓38.

⚠⚠ P_ALL == P_VIS СТРУКТУРНО, А НЕ ПО ЗАМЕРУ. Подозрение было: в `_slot_cv.py` прод считался по
ВСЕМ слотам листа, а модель — только по слотам с кандидатами. Но прод (`emit.py:129`) фильтрует
линии по треку и пар вне трека не строит вовсе, поэтому слот без кандидатов не может быть заполнен
НИ У КОГО: разница ≡ 0 при любых данных. §6.87 этим не испорчен, но и «проверкой» это назвать
нельзя.

ПРОТОКОЛ (объявлен до прогона, §6.49): разбиение по СКВАЖИНАМ, как в §6.87 — скважина целиком
либо в обучение, либо в замер; порядок фолдов тот же.
⚠ Пороги в таблицах перебираются на тех же фолдах, где считается результат ⇒ это ВЕРХНЯЯ оценка.
★ Поэтому в конце — ВЛОЖЕННЫЙ ВЫБОР: порог берётся на остальных фолдах, применяется к невиданному.
Цитировать надо его, а не лучшую клетку таблицы.

  <ComfyUI>\python_embeded\python.exe _slot_abstain.py [--folds 4] [--refresh]
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div"])
ap.add_argument("--folds", type=int, default=4)
ap.add_argument("--cache", default="F:/nds/output/taskS/_slot_abstain_cache_v2.pkl")
ap.add_argument("--refresh", action="store_true")
ap.add_argument("--noise", type=int, default=0, help="перестановок скважин для оценки шума (×2 seed)")
ap.add_argument("--holdout", action="store_true", help="LOWO + аудит выбора механизма, БЕЗ таблиц")
ap.add_argument("--ab", type=int, default=4, help="разбиений скважин на конструкторскую/аудиторскую половины")
a = ap.parse_args()

HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


def tfeat(tr):
    ys = sorted(tr)
    x = np.array([tr[y] for y in ys], float)
    if len(x) < 5:
        return (0., 0., 1., 0., float(len(x)), 0., 0.)
    dx = np.diff(x)
    span = float(np.percentile(x, 90) - np.percentile(x, 10)) or 1.0
    # ⚠⚠ ОКНО НЕ ДЛИННЕЕ САМОЙ ТРАССЫ (§6.109). Было `len(x)//2*2+1`: при ЧЁТНОЙ длине это
    # len(x)+1, а `np.convolve(mode="same")` возвращает max(len(x), w) — вычитание `x - sm`
    # падало. Любая трасса чётной длины от 6 до 100 точек роняла раскладку, и после включения
    # `slot_model` по умолчанию (§6.108) это стало падением ПРОДА, а не только стенда.
    # ⚠ Правка НЕ меняет признаки там, где код работал: при нечётной длине w тот же, при
    # длине ≥101 окно и было 101. Значит обученный вес остаётся действительным.
    w = min(101, len(x) if len(x) % 2 else len(x) - 1)
    sm = np.convolve(x, np.ones(w) / w, mode="same") if w >= 3 else x
    return (float(np.median(np.abs(dx))),
            float(np.mean((dx[:-1] * dx[1:]) < 0)) if len(dx) > 2 else 0.,
            span, float(np.std(x - sm) / span), float(len(x)),
            float(np.median(x)), float(ys[-1] - ys[0]))


def build():
    """Пулы → пары (слот, трасса). Признаки и метки БАЙТ В БАЙТ как в _slot_cv.py (§6.87), плюс
    сырые входы ПРАВИЛА (цвет/класс/трек слота, цвет/поведение/x_center линии) и выбор прода
    индексом линии — чтобы правило доигрывалось на остатке без двойного использования."""
    well = {}
    for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            well[q.name] = wlg.parent.name
    for e in (r"F:\nds\projects\Semeguniv_001\wlg", r"F:\nds\projects\Semeguniv_020\wlg"):
        if Path(e).is_dir():
            for q in Path(e).glob("*.nlgx"):
                well[q.name] = Path(e).parent.name

    # ⚠ КЛЮЧ ДЕДУПЛИКАЦИИ — ИМЯ ФАЙЛА, А НЕ ПОЛЕ `name`: `_pool_oracle.py` кладёт дамп как
    # `{nlgx.stem[:60]}.pkl`, поэтому дубль виден ДО чтения и стоит ноль. ⚠ В отличие от
    # `_start_probe`/`_param_sweep` здесь нет прохода «сначала список, потом счёт»: дамп нужен
    # целиком, и 2.35 ГБ читаются по делу — экономится только повторное чтение дублей. Вдобавок
    # весь этот разбор кэшируется в `--cache`, так что пулы читаются один раз на много прогонов.
    sheets, seen = [], set()
    for root in a.pools:
        for f in sorted(Path(root).glob("*.pkl")):
            if f.stem in seen:
                continue
            seen.add(f.stem)
            d = pickle.load(open(f, "rb"))
            if not d["name"].startswith(f.stem[:40]):
                print(f"  ⚠ имя дампа {f.stem[:40]!r} расходится с полем name {d['name'][:40]!r}")
            sheets.append(d)

    X, Y, R, IDX, phon, pick = [], [], [], [], {}, {}
    p_all, SR_, LR_ = [0] * len(sheets), [], []
    for si, d in enumerate(sheets):
        slots = {s["name"]: s for s in d["slots"]}
        SR_.append([dict(name=s["name"], track=s["track"], color=s["color"], cls=s["class"])
                    for s in d["slots"]])
        LR_.append([dict(track=ln["track"], color=ln["color"], behavior=ln["behavior"],
                         x_center=float(ln["x_center"])) for ln in d["lines"]])
        # выбор ПРАВИЛА: written[nm] — это ровно объект трассы выбранной линии (один pickle.dump,
        # ссылки общие), поэтому индекс восстанавливается тождеством, а не эвристикой
        by_id = {id(ln["tr"]): i for i, ln in enumerate(d["lines"])}
        for nm, tr in d["written"].items():
            i = by_id.get(id(tr))
            if i is None:
                i = next((j for j, ln in enumerate(d["lines"]) if ln["tr"] == tr), None)
            if i is not None:
                pick[(si, nm)] = i
        tl = {}
        for i, ln in enumerate(d["lines"]):
            tl.setdefault(ln["track"], []).append(i)
        for nm, gt in d["gts"].items():
            s = slots.get(nm)
            if s is None:
                continue
            w = d["written"].get(nm)
            ph = 1 if (w is not None and HON(*err(w, gt))) else 0
            phon[(si, nm)] = ph
            p_all[si] += ph
            same = tl.get(s["track"], [])
            if not same:
                continue
            meds = sorted(float(np.median(list(d["lines"][i]["tr"].values()))) for i in same)
            for i in same:
                ln = d["lines"][i]
                wig, rev, span, rough, n, med, dy = tfeat(ln["tr"])
                cls = "SP" if ln["behavior"] == "smooth" else "RES"
                rank = meds.index(min(meds, key=lambda v: abs(v - med))) / max(1, len(meds) - 1)
                X.append([1.0 if (s["color"] and s["color"] == ln["color"]) else 0.0,
                          1.0 if s["color"] is None else 0.0,
                          1.0 if s["class"] in (cls, "OTHER", "CALI") else 0.0,
                          1.0 if s["class"] == "SP" else 0.0,
                          rank, ln["x_center"] / 1000.0, med / 1000.0,
                          wig, rev, rough, span / 1000.0, n / 10000.0,
                          len(same) / 10.0, dy / 10000.0])
                m, c = err(ln["tr"], gt)
                Y.append(1 if HON(m, c) else 0)
                R.append(np.log1p(m if m is not None else 5000.0) + 3.0 * max(0.0, 0.9 - c))
                IDX.append((si, nm, i))
    return dict(X=np.array(X, float), Y=np.array(Y, int), R=np.array(R, float), IDX=IDX,
                phon=phon, p_all=p_all, pick=pick, slots=SR_, lines=LR_,
                names=[d["name"] for d in sheets],
                wells=[well.get(d["name"], "?") for d in sheets],
                ngts=[len(d["gts"]) for d in sheets])


cp = Path(a.cache)
if cp.exists() and not a.refresh:
    C = pickle.load(open(cp, "rb")); print(f"кэш пар: {cp}")
else:
    C = build()
    cp.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(C, open(cp, "wb")); print(f"кэш пар записан: {cp}")

X, Y, R, IDX = C["X"], C["Y"], C["R"], C["IDX"]
PH, P_ALL, PICK, SLOTS, LINES = C["phon"], C["p_all"], C["pick"], C["slots"], C["lines"]
NS = len(C["names"])
wells = sorted(set(C["wells"]))
fold_of = {w: k % a.folds for k, w in enumerate(wells)}
sheet_fold = np.array([fold_of[w] for w in C["wells"]])
pair_fold = np.array([sheet_fold[si] for si, _, _ in IDX])

vis, pk, PKI, pk_all = {}, {}, {}, {}
for k, (si, nm, i) in enumerate(IDX):
    vis.setdefault(si, set()).add(nm)
    pk.setdefault((si, nm), []).append(k)
    pk_all.setdefault(si, []).append(k)
    PKI[(si, nm, i)] = k
P_VIS = [sum(PH[(si, nm)] for nm in vis.get(si, ())) for si in range(NS)]

print(f"листов {NS}, скважин {len(wells)}, пар {len(Y)}, честных пар {int(Y.sum())}")
print(f"прод: {sum(P_ALL)} честных кривых по ВСЕМ слотам (как в §6.87), "
      f"{sum(P_VIS)} по слотам с кандидатами ⇒ разница {sum(P_ALL) - sum(P_VIS)}")
print("⚠ этот ноль СТРУКТУРЕН, а не измерен: прод (emit.py:129) фильтрует линии по треку и пар вне\n"
      "  трека не строит, поэтому слот без кандидатов не заполняется ни у кого\n")


# ── ПРАВИЛО ПРОДА, воспроизведённое офлайн (emit._map_lines_to_slots, строки 127-154) ──────────
def rule(si, free_slots=None, free_lines=None):
    """{slot: line} по действующему правилу. free_* — доигрывание на остатке (None = всё доступно).
    ⚠ colors_on_track считается по ВСЕМ линиям трека, как в проде, а не по свободным: строгость
    цвета — свойство трека, а не остатка."""
    slots, lines = SLOTS[si], LINES[si]
    fs = {s["name"] for s in slots} if free_slots is None else free_slots
    fl = set(range(len(lines))) if free_lines is None else free_lines
    out, used, taken = {}, set(), set()
    for ti in {s["track"] for s in slots}:
        tsl = [s for s in slots if s["track"] == ti and s["name"] in fs]
        tli = [i for i, ln in enumerate(lines) if ln["track"] == ti]
        colors = {lines[i]["color"] for i in tli}
        pairs = []
        for s in tsl:
            strict = s["color"] is not None and s["color"] in colors
            for i in tli:
                if i not in fl or (strict and s["color"] != lines[i]["color"]):
                    continue
                cls = "SP" if lines[i]["behavior"] == "smooth" else "RES"
                pairs.append((0 if s["cls"] in (cls, "OTHER", "CALI") else 1,
                              lines[i]["x_center"], s["name"], i))
        pairs.sort(key=lambda q: (q[0], q[1]))
        for _, _, nm, i in pairs:
            if nm in taken or i in used:
                continue
            taken.add(nm); used.add(i); out[nm] = i
    return out


picked = {}
for (si, nm), i in PICK.items():
    picked.setdefault(si, {})[nm] = i
ok = bad = miss = 0
for si in range(NS):
    mine = rule(si)
    for nm, j in picked.get(si, {}).items():
        if nm not in mine:
            miss += 1
        elif mine[nm] == j:
            ok += 1
        else:
            bad += 1
tot_pick = ok + bad + miss
print(f"СВЕРКА ПРАВИЛА: совпало {ok}/{tot_pick} ({100*ok/max(1,tot_pick):.1f}%), "
      f"другая линия {bad}, слот не занят {miss}")
print("⚠ сверен ТОЛЬКО жадный цикл: трек слота, цвет/класс и фильтр DA* посчитаны прод-кодом ещё\n"
      "  при дампе пулов (_pool_oracle.py:149-155), разойтись там нечему. Путь rule() С ОСТАТКОМ\n"
      "  (free_slots/free_lines) сверкой НЕ проверяется.\n"
      "⚠ 100% содержательны: мутации правила её ломают — сортировка по -x_center 21%, без class_ok\n"
      "  87%, без строгого цвета 96% (проверено независимо, wf_e48ba56c-1a9)\n")
RULE_FULL = [rule(si) for si in range(NS)]     # выбор правила без отказа — для меры cap


def sheet_conf_of(si, got, kind):
    """Уверенность ЛИСТА по назначенным слотам. Ни одна мера не смотрит на эталон.
    cap — единственная, которой не нужны и скоры: «сколько слотов модель меняет против правила»."""
    ks = np.array([PKI[(si, nm, i)] for nm, i in got.items()])
    if kind == "mean":
        return float(np.mean(MGS[ks]))
    if kind == "min":
        return float(np.min(MGS[ks]))
    if kind == "minx":                         # то же, но слоты с одним кандидатом — по ошибке
        return float(np.min(MGX[ks]))
    if kind == "frac":
        return float(np.mean(MG[ks] >= 0.3))
    if kind == "maxerr":                       # худшая предсказанная ошибка на листе
        return float(np.min(SR[ks]))
    if kind == "cap":                          # -число расхождений с правилом
        return -float(sum(1 for nm, i in got.items() if RULE_FULL[si].get(nm) != i))
    raise ValueError(kind)


def assign(score, conf, thr, sheet=None, fill="rule", reserve=False):
    """Жадно 1:1 внутри листа по парам с conf >= thr; незанятые слоты закрываются по `fill`.
    sheet=(kind, thr) — второй ярус: лист ниже порога отдаётся правилу ЦЕЛИКОМ.

    ★ reserve=True — ОТКАЗ ДО ЖАДНОГО ШАГА, а не после. Слот, у которого ни одна пара не прошла
    порог, сначала получает выбор ПРАВИЛА, и эта линия РЕЗЕРВИРУЕТСЯ; только потом модель
    расставляет уверенные слоты по остатку. Замер §6.88 показал, что откат ПОСЛЕ модели невозможен
    по построению (жадность исчерпывает линии трека), поэтому порядок здесь и есть весь механизм."""
    per = {}
    for k, (si, nm, i) in enumerate(IDX):
        if conf[k] >= thr:
            per.setdefault(si, []).append(k)
    out = {}
    for si in range(NS):
        got, used, reserved = {}, set(), set()
        res_hon = 0
        if reserve:
            conf_s = {IDX[k][1] for k in per.get(si, ())}
            abst = {nm for nm in vis.get(si, ()) if nm not in conf_s}
            if abst:
                rl = rule(si, abst)                    # линии свободны все — правило выбирает первым
                used |= set(rl.values()); reserved = set(rl)
                res_hon = sum(int(Y[PKI[(si, nm, rl[nm])]]) for nm in rl
                              if nm in abst and (si, nm, rl[nm]) in PKI)
        for k in sorted(per.get(si, ()), key=lambda q: -score[q]):
            _, nm, i = IDX[k]
            if nm in got or i in used:
                continue
            got[nm] = i; used.add(i)
        if sheet is not None and got and sheet_conf_of(si, got, sheet[0]) < sheet[1]:
            got, used, res_hon, reserved = {}, set(), 0, set()   # лист целиком к правилу
        if not got and not reserved and fill == "rule":
            out[si] = (P_VIS[si], 0)      # правило на ВСЁМ листе == прод (сверено полистно)
            continue
        h = res_hon + sum(int(Y[PKI[(si, nm, i)]]) for nm, i in got.items())
        rest = [nm for nm in vis.get(si, ()) if nm not in got and nm not in reserved]
        if fill == "pick":
            h += sum(PH[(si, nm)] for nm in rest)
        elif fill == "rule":
            free_s = {s["name"] for s in SLOTS[si] if s["name"] not in got and s["name"] not in reserved}
            free_l = set(range(len(LINES[si]))) - used
            rl = rule(si, free_s, free_l)
            h += sum(int(Y[PKI[(si, nm, rl[nm])]]) for nm in rest
                     if nm in rl and (si, nm, rl[nm]) in PKI)
        out[si] = (h, len(got))
    return out


def row(tag, res, mask):
    """⚠ Знаменатель у ↑/↓ НЕ 702: лист может сдвинуться только если у него есть честные пары, а
    упасть — только если проду было что терять (P_VIS>0). Печатать надо против RISK, иначе гейт
    §6.38 читается в 5 раз мягче, чем он есть."""
    idxs = [si for si in range(NS) if mask[si]]
    return dict(tag=tag, hon=sum(res[si][0] for si in idxs),
                asg=sum(res[si][1] for si in idxs),
                up=sum(1 for si in idxs if res[si][0] > P_VIS[si]),
                dn=sum(1 for si in idxs if res[si][0] < P_VIS[si]))


HONP = {}                                  # честных пар на лист — «лист вообще способен сдвинуться»
for k, (si, nm, i) in enumerate(IDX):
    HONP[si] = HONP.get(si, 0) + int(Y[k])
NCAND = np.array([len(pk[(IDX[k][0], IDX[k][1])]) for k in range(len(Y))])


SC = np.zeros(len(Y)); SR = np.zeros(len(Y)); MG = np.zeros(len(Y)); MGC = np.zeros(len(Y))
done = np.zeros(NS, bool)
for k in range(a.folds):
    te_s = sheet_fold == k
    tr_p = pair_fold != k
    if Y[tr_p].sum() < 10 or not te_s.any():
        continue
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)
    clf.fit(X[tr_p], Y[tr_p])
    reg = GradientBoostingRegressor(n_estimators=250, max_depth=3, random_state=0)
    reg.fit(X[tr_p], R[tr_p])
    te_p = pair_fold == k
    SC[te_p] = clf.predict_proba(X[te_p])[:, 1]
    SR[te_p] = -reg.predict(X[te_p])
    done |= te_s
    print(f"фолд {k}: обучен на {int(tr_p.sum())} парах, замер на {int(te_s.sum())} листах")

for (si, nm), ks in pk.items():
    for src, dst in ((SR, MG), (SC, MGC)):
        v = sorted((src[q] for q in ks), reverse=True)
        for q in ks:
            dst[q] = 1e9 if len(v) == 1 else src[q] - (v[1] if src[q] >= v[0] else v[0])
MGS = np.clip(MG, -5.0, 5.0)      # для СРЕДНЕГО по листу: ±∞ иначе съедает среднее
# ⚠ MG — margin ПАРЫ: у не-лучших пар слота он отрицателен, поэтому порог m≥0 выбрасывает их из
# кандидатов и лишает жадность возможности взять вторую трассу слота. Это уже другой алгоритм, а не
# отказ. MGA — margin СЛОТА (одно число на слот, у всех его пар): порог m=-∞ здесь честный no-op.
MGA = np.zeros(len(Y))
for (si, nm), ks in pk.items():
    v = sorted((SR[q] for q in ks), reverse=True)
    for q in ks:
        MGA[q] = 1e9 if len(v) == 1 else v[0] - v[1]
# ⚠ У слота с ЕДИНСТВЕННЫМ кандидатом неоднозначности нет, поэтому margin = +∞ — и такой слот
# проходит ЛЮБОЙ порог. Их 323 из 1839 (17.6%): у любой margin-меры из-за них неустранимый пол по
# ↓. MGX закрывает дыру: для них берётся не margin, а предсказанная ошибка (≥0 ⟺ pred ≤ 2.0, ~6px).
# ⚠⚠ БРАТЬ ЗДЕСЬ MG (margin ПАРЫ), А НЕ MGA: сигнал, на котором работает мера `min`, — это
# «слот получил ИМЕННО свою лучшую трассу», то есть отрицательный margin ПАРЫ при конкуренции.
# MGA (margin слота) всегда ≥0, и первая редакция minx из-за этого почти не отказывала (+41, ↓33).
MGX = np.where(NCAND > 1, np.clip(MG, -5.0, 5.0), np.clip(SR + 2.0, -5.0, 5.0))

NONE = np.full(len(Y), -1e18)
ZERO = np.zeros(len(Y))
W = 78

# ══ ПРОВЕРКА БЕЗ НОВЫХ ДАННЫХ (§6.89) ═════════════════════════════════════════════════════════
# §6.88 назвала следующим шагом чистую проверку на НОВЫХ скважинах. Её выполнить НЕЛЬЗЯ: в архиве
# 41 скважина и все они уже в пулах (Archive_incomplete — те же 7 скважин, 34 кривые). Поэтому
# здесь два сильнейших доступных заменителя:
#   1) LOWO — скважина за скважиной наружу, 42 фолда, рецепт ЗАМОРОЖЕН. Отвечает на «держится ли
#      эффект на всех скважинах или на трёх»;
#   2) АУДИТ ВЫБОРА МЕХАНИЗМА — скважины делятся на конструкторскую и аудиторскую половины,
#      семейство И порог выбираются ТОЛЬКО на конструкторской (своей внутренней 2-фолдовой
#      проверкой), применяются к аудиторской ОДИН раз. Это проверяет не мои знания задним числом,
#      а устойчивость самой ПРОЦЕДУРЫ выбора: садится ли она на тот же механизм на другой половине.
# ⚠ Оба заменителя НЕ снимают того, что семейства мер придуманы после просмотра данных. Они
# показывают лишь, что процедура не рассыпается; чистую проверку по-прежнему может дать только
# новый набор скважин.
if a.holdout:
    FROZEN = ("minx", 0.0)          # ★ рецепт §6.88, зафиксирован ДО этого прогона
    FAMS = [("minx", (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0)),
            ("min", (-1.0, -0.5, -0.2, 0.0, 0.05, 0.2, 0.5, 1.0)),
            ("frac", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
            ("maxerr", (-6.0, -5.0, -4.0, -3.0, -2.5, -2.0)),
            ("mean", (0.0, 0.1, 0.3, 0.5, 1.0, 2.0)),
            ("cap", (-1e9, -5.0, -3.0, -2.0, -1.0, 0.0))]
    WELLS = np.array(C["wells"])

    def fit_score(tr_pairs, sc_pairs, seed=0):
        rg = GradientBoostingRegressor(n_estimators=250, max_depth=3, random_state=seed)
        rg.fit(X[tr_pairs], R[tr_pairs])
        SR[sc_pairs] = -rg.predict(X[sc_pairs])

    def margins():
        for (si, nm), ks in pk.items():
            v = sorted((SR[q] for q in ks), reverse=True)
            for q in ks:
                MGA[q] = 1e9 if len(v) == 1 else v[0] - v[1]
                MG[q] = 1e9 if len(v) == 1 else SR[q] - (v[1] if SR[q] >= v[0] else v[0])
        MGS[:] = np.clip(MG, -5.0, 5.0)
        MGX[:] = np.where(NCAND > 1, np.clip(MG, -5.0, 5.0), np.clip(SR + 2.0, -5.0, 5.0))

    prodm = lambda m: sum(P_VIS[si] for si in range(NS) if m[si])

    # ── 1. LEAVE-ONE-WELL-OUT, рецепт заморожен ───────────────────────────────────────────────
    print(f"\n{'='*W}\n★ LOWO: скважина за скважиной наружу, {len(wells)} фолдов, "
          f"рецепт заморожен {FROZEN}\n{'='*W}")
    CELLS = [("без отказа", lambda: assign(SR, ZERO, 0.0)),
             ("★ заморожен minx≥0", lambda: assign(SR, ZERO, 0.0, FROZEN)),
             ("frac≥0.8", lambda: assign(SR, ZERO, 0.0, ("frac", 0.8))),
             ("maxerr≥-3.0", lambda: assign(SR, ZERO, 0.0, ("maxerr", -3.0)))]
    agg = {c[0]: dict(hon=0, up=0, dn=0, neg=0) for c in CELLS}
    prod_tot = 0
    per_well = []
    for wi, w in enumerate(wells):
        te = np.array([ww == w for ww in WELLS])
        te_p = np.array([te[si] for si, _, _ in IDX])
        if not te.any() or Y[~te_p].sum() < 10:
            continue
        fit_score(~te_p, te_p)
        margins()
        p = prodm(te)
        prod_tot += p
        line = []
        for nm_c, fn in CELLS:
            r = row("", fn(), te)
            agg[nm_c]["hon"] += r["hon"]; agg[nm_c]["up"] += r["up"]; agg[nm_c]["dn"] += r["dn"]
            agg[nm_c]["neg"] += 1 if r["hon"] < p else 0
            line.append(r["hon"] - p)
        per_well.append((w, int(te.sum()), p, line))
    print(f"{'скважина':<24}{'листов':>7}{'прод':>6}" + "".join(f"{c[0][:15]:>17}" for c in CELLS))
    for w, n, p, line in sorted(per_well, key=lambda q: q[3][1]):
        print(f"{w[:23]:<24}{n:>7}{p:>6}" + "".join(f"{d:>+17}" for d in line))
    print(f"\n{'ИТОГО по ' + str(len(per_well)) + ' скважинам':<31}{prod_tot:>6}"
          + "".join(f"{agg[c[0]]['hon'] - prod_tot:>+17}" for c in CELLS))
    print(f"{'листов ↑/↓':<31}{'':>6}"
          + "".join(f"{f'{agg[c[0]][chr(117)+chr(112)]}/{agg[c[0]][chr(100)+chr(110)]}':>17}" for c in CELLS))
    print(f"{'скважин в минусе':<31}{'':>6}"
          + "".join(f"{agg[c[0]]['neg']:>17}" for c in CELLS))

    # ── 2. АУДИТ ПРОЦЕДУРЫ ВЫБОРА: механизм выбирается на одной половине скважин, проверяется на другой
    print(f"\n{'='*W}\n★★ АУДИТ ВЫБОРА МЕХАНИЗМА: {a.ab} разбиений × 2 направления\n{'='*W}")
    print("процедура: на КОНСТРУКТОРСКОЙ половине (своя внутренняя 2-фолдовая проверка) выбирается")
    print("семейство И порог по критерию размена max Δ−2·↓; затем модель учится на ВСЕЙ этой")
    print("половине и выбранное правило применяется к АУДИТОРСКОЙ половине ОДИН раз.")
    print(f"\n{'разбиение':<12}{'выбрано процедурой':<26}{'Δ аудит':>9}{'↑/↓':>9}"
          f"{'Δ заморож.':>12}{'↑/↓':>9}")
    picks, dproc, dfroz = {}, [], []
    for s in range(a.ab):
        order = list(np.random.default_rng(1000 + s).permutation(wells))
        half = len(order) // 2
        for direction, (des, aud) in enumerate(((order[:half], order[half:]),
                                                (order[half:], order[:half]))):
            des_s = np.array([w in set(des) for w in WELLS])
            aud_s = np.array([w in set(aud) for w in WELLS])
            des_p = np.array([des_s[si] for si, _, _ in IDX])
            aud_p = np.array([aud_s[si] for si, _, _ in IDX])
            # внутренняя проверка ВНУТРИ конструкторской половины
            in1 = np.array([w in set(des[:len(des) // 2]) for w in WELLS])
            in1_p = np.array([in1[si] for si, _, _ in IDX])
            fit_score(des_p & ~in1_p, des_p & in1_p)
            fit_score(des_p & in1_p, des_p & ~in1_p)
            margins()
            best, bkey = None, None
            for kind, grid in FAMS:
                for t in grid:
                    r = row("", assign(SR, ZERO, 0.0, (kind, t)), des_s)
                    sc = r["hon"] - 2 * r["dn"]
                    if best is None or sc > best:
                        best, bkey = sc, (kind, t)
            # аудит: модель на ВСЕЙ конструкторской половине, правило применяется один раз
            fit_score(des_p, aud_p)
            margins()
            pa = prodm(aud_s)
            rp = row("", assign(SR, ZERO, 0.0, bkey), aud_s)
            rf = row("", assign(SR, ZERO, 0.0, FROZEN), aud_s)
            picks[bkey[0]] = picks.get(bkey[0], 0) + 1
            dproc.append(rp["hon"] - pa); dfroz.append(rf["hon"] - pa)
            lab = f"{bkey[0]}≥{-1e9 if bkey[1] < -1e8 else round(bkey[1], 2)}"
            print(f"{f'р{s}·{direction}':<12}{lab:<26}{rp['hon']-pa:>+9}"
                  f"{f'{rp[chr(117)+chr(112)]}/{rp[chr(100)+chr(110)]}':>9}"
                  f"{rf['hon']-pa:>+12}{f'{rf[chr(117)+chr(112)]}/{rf[chr(100)+chr(110)]}':>9}")
    med = lambda v: sorted(v)[len(v) // 2]
    print(f"\nпроцедура выбрала: " + ", ".join(f"{k} × {v}" for k, v in
                                               sorted(picks.items(), key=lambda q: -q[1])))
    print(f"Δ на аудиторской половине: процедура медиана {med(dproc):+d} "
          f"({min(dproc):+d}..{max(dproc):+d}); замороженный рецепт медиана {med(dfroz):+d} "
          f"({min(dfroz):+d}..{max(dfroz):+d})")
    # §6.106: числа — ИЗ СЧЁТЧИКА, а не текстом. Половины режутся по скважинам, и их число
    # зависит от выборки: строковые «21 вместо ~31» пережили бы смену набора незаметно.
    print(f"⚠ обе цифры ПЕССИМИСТИЧНЫ против §6.88: модель учится на конструкторской половине — "
          f"{len(wells) // 2} скважин из {len(wells)}, а не на всех")
    sys.exit(0)


def prt(r, ref=None):
    print(f"{r['tag']:<30}{r['hon']:>9}{r['asg']:>9}{r['up']:>5}{r['dn']:>5}"
          f"{r['hon'] - (P0 if ref is None else ref):>+7}")


HEAD = f"{'вариант':<30}{'★честных':>9}{'назнач.':>9}{'↑':>5}{'↓':>5}{'Δ':>7}"
P0 = sum(P_VIS[si] for si in range(NS) if done[si])       # Δ считается к ПРОДУ на тех же листах

RISK = sum(1 for si in range(NS) if done[si] and P_VIS[si] > 0)
MOV = sum(1 for si in range(NS) if done[si] and HONP.get(si, 0) > 0)
WITHP = sum(1 for si in range(NS) if done[si] and si in vis)
print(f"\n{'='*W}\nЗНАМЕНАТЕЛЬ (иначе гейт §6.38 читается мягче, чем есть)\n{'='*W}")
print(f"листов всего {int(done.sum())}, с кандидатами {WITHP} (это число сравнимо с 674 в §6.87)")
print(f"★ способны сдвинуться вверх (есть честные пары) {MOV}; способны УПАСТЬ (P_VIS>0) {RISK}")
print(f"⇒ ↓ надо читать как долю от {RISK}, а не от {int(done.sum())}")
print(f"слотов с кандидатами {sum(len(v) for v in vis.values())}, из них с ЕДИНСТВЕННЫМ кандидатом "
      f"{sum(1 for key in pk if len(pk[key]) == 1)} — они проходят любой порог margin")

print(f"\n{'='*W}\nБАЗА: чем закрывать слот, который модель не заняла (листов {int(done.sum())})\n{'='*W}")
print(HEAD)
prt(row("прод / правило целиком", assign(SR, NONE, 0.0, fill="rule"), done))
for f, note in (("none", ""), ("rule", " (≡none при пороге 0)"),
                ("pick", " ⚠НЕРЕАЛИЗУЕМО: двойное исп.")):
    prt(row(f"регрессия, fill={f}{note}", assign(SR, ZERO, 0.0, fill=f), done))
prt(row("классификатор, fill=none", assign(SC, ZERO, 0.0, fill="none"), done))
print("⚠ fill=pick закрывает слот выбором прода, НЕ глядя на линии, израсходованные моделью: у "
      "пустого слота свободных линий на треке нет по построению ⇒ каждое его очко — линия, "
      "зачтённая дважды. Цитировать нельзя.")

print(f"\n{'='*W}\nОТКАЗ ПОСЛОТНО: предсказанная ошибка пары ≤ t (регрессия, fill=rule)\n{'='*W}")
print(HEAD)
for t in (4.0, 3.0, 2.5, 2.0, 1.6, 1.386, 1.0, 0.8):
    prt(row(f"t={t:<5} (~{np.expm1(t):.0f}px)", assign(SR, SR, -t), done))

MS = (-1e18, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
mlab = lambda m: "m=-∞ (без отказа)" if m < -1 else f"m={m}"
RES_M = {m: assign(SR, MG, m) for m in MS}
print(f"\n{'='*W}\nОТКАЗ ПОСЛОТНО: margin по слоту ≥ m (регрессия, fill=rule)\n{'='*W}")
print(HEAD)
for m in MS:
    prt(row(mlab(m), RES_M[m], done))

print(f"\n{'='*W}\nОТКАЗ ПОСЛОТНО: вероятность классификатора ≥ p (fill=rule)\n{'='*W}")
print(HEAD)
for p in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
    prt(row(f"p={p}", assign(SC, SC, p), done))

# ── ПОЧЕМУ ПОСЛОТНЫЙ ОТКАЗ НЕ РАБОТАЕТ: цена отказа не локальна ───────────────────────────────
res0 = assign(SR, ZERO, 0.0, fill="rule")
nvis = sum(len(v) for v in vis.values())
lost = stolen = filled = filled_hon = 0
for si in range(NS):
    if not done[si]:
        continue
    got, used = {}, set()
    for k in sorted(pk_all.get(si, ()), key=lambda q: -SR[q]):
        _, nm, i = IDX[k]
        if nm in got or i in used:
            continue
        got[nm] = i; used.add(i)
    rest = [nm for nm in vis.get(si, ()) if nm not in got]
    free_l = set(range(len(LINES[si]))) - used
    rl = rule(si, {s["name"] for s in SLOTS[si] if s["name"] not in got}, free_l)
    for nm in rest:
        if PH[(si, nm)]:                       # прод брал этот слот честно, а модель слот не заняла
            lost += 1
            if PICK.get((si, nm)) in used:      # ...и линию прода забрала модель на другой слот
                stolen += 1
        if nm in rl:
            filled += 1
            filled_hon += int(Y[PKI[(si, nm, rl[nm])]]) if (si, nm, rl[nm]) in PKI else 0
print(f"\n{'='*W}\nПОЧЕМУ ОТКАЗ НЕ ЛОКАЛЕН (регрессия без отказа, листов {int(done.sum())})\n{'='*W}")
print(f"слотов с кандидатами {nvis}, модель заняла {sum(r[1] for r in res0.values())}, "
      f"осталось незанятых {nvis - sum(r[1] for r in res0.values())}")
print(f"из незанятых прод брал честно {lost}; линию прода модель забрала на другой слот {stolen}"
      + ("  ⚠ тождественно: линия прода лежит на треке слота ⇒ всегда израсходована"
         if stolen == lost else ""))
print(f"правило доиграло на остатке {filled} слотов, из них честных {filled_hon}")
print(f"⇒ потолок ЛЮБОГО послотного отката — {lost} кривых на {int(done.sum())} листов, "
      f"против ↓{row('', res0, done)['dn']} листов деградации. Механизм не тот.")

# ── ОТКАЗ ЛИСТОМ ЦЕЛИКОМ: пять мер уверенности листа ──────────────────────────────────────────
SHEET = {
    "mean": ("средний margin", (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)),
    "min": ("худший margin", (-5.0, -1.0, -0.5, -0.2, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0)),
    "minx": ("★ худший margin, одиночки по ошибке", (-5.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0)),
    "frac": ("доля слотов margin≥0.3", (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0)),
    "maxerr": ("худшая предсказ. ошибка", (-6.0, -5.0, -4.0, -3.0, -2.5, -2.0, -1.6, -1.386)),
    "cap": ("★ расхождений с правилом ≤ D", (-1e9, -20.0, -10.0, -5.0, -3.0, -2.0, -1.0, 0.0)),
}
RES_S = {}
for kind, (lab, grid) in SHEET.items():
    print(f"\n{'='*W}\nОТКАЗ ЛИСТОМ ЦЕЛИКОМ: {lab} (fill=rule)\n{'='*W}")
    print(HEAD)
    for t in grid:
        RES_S[(kind, t)] = assign(SR, ZERO, 0.0, (kind, t))
        tag = (f"D={-int(t)}" if t > -1e8 else "D=∞") if kind == "cap" else f"{kind}≥{t}"
        prt(row(tag, RES_S[(kind, t)], done))

# ── ПОСЛОТНЫЙ ОТКАЗ, СДЕЛАННЫЙ ПРАВИЛЬНО: margin СЛОТА + резерв линии ДО жадного шага ──────────
TS = (4.0, 3.0, 2.5, 2.0, 1.6, 1.386, 1.0)
RES_A = {m: assign(SR, MGA, m) for m in MS}
RES_AR = {m: assign(SR, MGA, m, reserve=True) for m in MS}
RES_MR = {m: assign(SR, MG, m, reserve=True) for m in MS}
RES_TR = {t: assign(SR, SR, -t, reserve=True) for t in TS}
for lab, GR, RES, f in (("margin СЛОТА, откат после модели", MS, RES_A, mlab),
                        ("★ margin СЛОТА + РЕЗЕРВ линии до модели", MS, RES_AR, mlab),
                        ("★ margin ПАРЫ + РЕЗЕРВ линии до модели", MS, RES_MR, mlab),
                        ("★ предсказ. ошибка ≤ t + РЕЗЕРВ", TS, RES_TR,
                         lambda t: f"t={t} (~{np.expm1(t):.0f}px)")):
    print(f"\n{'='*W}\nОТКАЗ ПОСЛОТНО: {lab}\n{'='*W}")
    print(HEAD)
    for g in GR:
        prt(row(f(g), RES[g], done))

# ── ВЛОЖЕННЫЙ ВЫБОР: порог (и сам МЕХАНИЗМ) берутся на ОСТАЛЬНЫХ фолдах, применяются к невиданному
# ⚠ Это единственные числа, которые можно цитировать: в таблицах выше порог подобран там же, где
# и посчитан результат. Мягкая оговорка остаётся: модели остальных фолдов видели скважины этого.
FAM = [("послотно margin пары", [(("pair", m), RES_M[m]) for m in MS]),
       ("послотно margin слота", [(("slotA", m), RES_A[m]) for m in MS]),
       ("★ margin слота + резерв", [(("slotA+рез", m), RES_AR[m]) for m in MS]),
       ("★ margin пары + резерв", [(("pair+рез", m), RES_MR[m]) for m in MS]),
       ("★ предсказ. ошибка + резерв", [(("err+рез", t), RES_TR[t]) for t in TS])]
for kind, (lab, grid) in SHEET.items():
    FAM.append((f"лист: {lab}", [((kind, t), RES_S[(kind, t)]) for t in grid]))
FAM.append(("★ ВСЁ ВМЕСТЕ (механизм тоже выбирается)",
            [x for _, lst in FAM for x in lst]))

print(f"\n{'='*W}\n★ ВЛОЖЕННЫЙ ВЫБОР (это и есть цитируемые числа)\n{'='*W}")
print("критерии: [строгий] ↓=0, потом max Δ (гейт §6.38); [размен] max Δ−2·↓ (цена §6.63)")
CRIT = (("строгий", lambda r: (r["dn"] == 0, -r["dn"], r["hon"])),
        ("размен", lambda r: (True, 0, r["hon"] - 2 * r["dn"])))
PRODF = {k: sum(P_VIS[si] for si in range(NS) if done[si] and sheet_fold[si] == k)
         for k in range(a.folds)}
for nm_c, crit in CRIT:
    print(f"\n--- критерий [{nm_c}] " + "-" * (W - 18 - len(nm_c)))
    print(f"{'семейство':<38}{'★честных':>9}{'↑':>5}{'↓':>5}{'Δ':>7}   пороги по фолдам")
    for nm_g, cand in FAM:
        agg = dict(hon=0, up=0, dn=0, prod=0)
        chosen = []
        for k in range(a.folds):
            te = done & (sheet_fold == k)
            tr = done & (sheet_fold != k)
            if not te.any() or not tr.any():
                continue
            key, res = max(cand, key=lambda c: crit(row("", c[1], tr)))
            r = row("", res, te)
            gate = "" if row("", res, tr)["dn"] == 0 else "!"   # ! = и на обучающих фолдах ↓>0
            lab = f"{key[0]}={'-∞' if key[1] < -1e8 else round(key[1], 3)}"
            chosen.append(f"ф{k}:{lab}{gate}→{r['hon'] - PRODF[k]:+d}")
            agg["hon"] += r["hon"]; agg["up"] += r["up"]; agg["dn"] += r["dn"]; agg["prod"] += PRODF[k]
        d = agg["hon"] - agg["prod"]
        print(f"{nm_g:<38}{agg['hon']:>9}{agg['up']:>5}{agg['dn']:>5}{d:>+7}   " + " ".join(chosen))
print(f"\nпрод на тех же листах: {P0}. читать так: цель — ↓=0 при Δ>0; «!» = порог не давал ↓=0 "
      f"и на обучающих фолдах, то есть критерий выродился")

# ── СЕТКА ШУМА: та же клетка при ДРУГОМ разбиении скважин и ДРУГОМ seed модели ─────────────────
# ⚠ Зачем. 9 листов, дописанных в пулы после §6.87, не несут НИ ОДНОЙ честной пары (взять там
# нечего никому), но сдвинули результат по фолдам на 30/36/47/59 → 33/36/43/63, то есть ±4 кривые
# на фолд от возмущения обучающей выборки на 1.9%. Значит различия ±5 в таблицах выше могут быть
# дрожью. Без этой сетки ни одну клетку цитировать нельзя.
if a.noise:
    CELLS = [("без отказа", lambda: assign(SR, ZERO, 0.0)),
             ("лист: min margin ≥ 0", lambda: assign(SR, ZERO, 0.0, ("min", 0.0))),
             ("лист: minx ≥ 0 (одиночки по ошибке)", lambda: assign(SR, ZERO, 0.0, ("minx", 0.0))),
             ("лист: доля margin≥0.3 ≥ 0.8", lambda: assign(SR, ZERO, 0.0, ("frac", 0.8))),
             ("лист: худшая ошибка ≥ -3.0", lambda: assign(SR, ZERO, 0.0, ("maxerr", -3.0))),
             ("лист: расхождений ≤ 1", lambda: assign(SR, ZERO, 0.0, ("cap", -1.0))),
             ("послотно margin слота ≥0.3 + резерв",
              lambda: assign(SR, MGA, 0.3, reserve=True))]
    acc = {c[0]: [] for c in CELLS}
    print(f"\n{'='*W}\n★★ СЕТКА ШУМА: {a.noise} перестановок скважин × 2 seed модели\n{'='*W}")
    print(f"{'конфиг':<10}" + "".join(f"{c[0][:17]:>18}" for c in CELLS))
    for pi in range(a.noise):
        for ms in (0, 1):
            if pi == 0:
                order = wells                                  # исходное разбиение §6.87
            else:
                order = list(np.random.default_rng(pi).permutation(wells))
            fo = {w: k % a.folds for k, w in enumerate(order)}
            sheet_fold[:] = np.array([fo[w] for w in C["wells"]])
            pair_fold[:] = np.array([sheet_fold[si] for si, _, _ in IDX])
            done[:] = False
            for k in range(a.folds):
                te_s = sheet_fold == k
                tr_p = pair_fold != k
                if Y[tr_p].sum() < 10 or not te_s.any():
                    continue
                rg = GradientBoostingRegressor(n_estimators=250, max_depth=3, random_state=ms)
                rg.fit(X[tr_p], R[tr_p])
                SR[pair_fold == k] = -rg.predict(X[pair_fold == k])
                done |= te_s
            for (si, nm), ks in pk.items():                    # margin пересчитывается под новый SR
                v = sorted((SR[q] for q in ks), reverse=True)
                for q in ks:
                    MGA[q] = 1e9 if len(v) == 1 else v[0] - v[1]
                    MG[q] = 1e9 if len(v) == 1 else SR[q] - (v[1] if SR[q] >= v[0] else v[0])
            MGS[:] = np.clip(MG, -5.0, 5.0)
            MGX[:] = np.where(NCAND > 1, np.clip(MG, -5.0, 5.0), np.clip(SR + 2.0, -5.0, 5.0))
            p0 = sum(P_VIS[si] for si in range(NS) if done[si])
            cells = []
            for nm_c, fn in CELLS:
                r = row("", fn(), done)
                acc[nm_c].append((r["hon"] - p0, r["up"], r["dn"]))
                cells.append(f"{r['hon']-p0:+d} {r['up']}/{r['dn']}")
            print(f"{f'п{pi}·s{ms}':<10}" + "".join(f"{c:>18}" for c in cells))
    print(f"\n{'клетка':<38}{'Δ медиана':>10}{'Δ мин..макс':>14}{'↓ медиана':>11}{'↓ макс':>8}")
    for nm_c, _ in CELLS:
        d = sorted(v[0] for v in acc[nm_c]); dn = sorted(v[2] for v in acc[nm_c])
        med = d[len(d) // 2]; mdn = dn[len(dn) // 2]
        print(f"{nm_c:<38}{med:>+10}{f'{d[0]:+d}..{d[-1]:+d}':>14}{mdn:>11}{dn[-1]:>8}")
    print("★ цитировать можно только клетку, у которой Δ медиана заметно больше размаха, "
          "а ↓ макс мал: остальное — дрожь разбиения")

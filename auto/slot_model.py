r"""slot_model.py — ОБУЧЕННАЯ РАСКЛАДКА (слот ← трасса) С ОТКАЗОМ ЛИСТОМ ЦЕЛИКОМ.

⚠⚠ ВЫКЛЮЧЕНА ПО УМОЛЧАНИЮ. Включается только `CVParams.slot_model`. **ОТКАТ = поставить ""**,
больше ничего менять не надо.

ЗАЧЕМ. Действующее правило (`emit._map_lines_to_slots`) назначает трассы слотам по порядку
`x_center` внутри трека. §6.80-§6.84: ~20 рукописных правил перебраны, ни одно не проходит гейт.
§6.85-§6.87: обученный ранжировщик пар (слот × трасса) даёт двузначный процент, но 37 листов из
674 деградируют. §6.88: деградацию убирает ОТКАЗ, причём единица отказа — **ЛИСТ, а не слот**
(отказ от слота невозможен по построению: жадное 1:1 израсходовало линии трека, и правилу нечем
доигрывать). §6.89: три независимые схемы проверки сошлись, а процедура выбора механизма садится
на один и тот же вариант в 8 случаях из 8.

ЧТО ДЕЛАЕТ. Для каждой пары (слот, трасса того же трека) считает 14 признаков — РОВНО те, что были
у прода в момент раскладки (совпадение цвета и класса, ранг по медиане x, форма, длина, число линий
на треке); ошибки против эталона среди них нет. Назначение жадное 1:1 по убыванию скора — ТОТ ЖЕ
контракт, что у правила. Затем **лист целиком** проверяется мерой уверенности, и если он её не
проходит, возвращается None ⇒ вызывающий берёт назначение ПРАВИЛА, то есть прежнее поведение.

⚠ SKLEARN В ПРОД НЕ ПРИХОДИТ. Обученный бустинг выгружен в деревья (`_slot_export.py`), скор
считается на numpy. В отличие от torch в §6.68, новой зависимости у развёртывания НЕТ.
⚠⚠ ПРИЗНАКИ СРАВНИВАЮТСЯ С ПОРОГАМИ В float32 — так делает sklearn внутри деревьев. В float64
строки у самого порога уходят в другую ветвь; на этом первая редакция экспорта разошлась с
обучением на 4e-2. Сверка экспорта: max|Δ| = 1.6e-14 на 22118 парах.

МЕРЫ УВЕРЕННОСТИ ЛИСТА (`CVParams.slot_gate`), из §6.88-§6.89:
  `frac0.8` — ★ по умолчанию: доля назначенных слотов с margin ≥ 0.3 не ниже 0.8.
              LOWO: +12 кривых, 14 листов вверх / 2 вниз, **ни одной скважины в минусе**;
  `minx0.0` — худший margin листа ≥ 0 («не было конкуренции за линии»), а слоты с ЕДИНСТВЕННЫМ
              кандидатом судятся не по margin (у них его нет), а по предсказанной ошибке ≤ ~6px.
              LOWO: +16 кривых, 20/4, две скважины в минусе по одной кривой;
  `off`     — модель считается, но лист всегда отдаётся правилу (для абляции).

⚠ ЧТО ЕЩЁ НЕ ПРОВЕРЕНО (важно для чтения любых цифр отсюда). Всё выше померено на ПУЛАХ, где за
линии соперничали только слоты с экспертной кривой. В проде за них соперничают ВСЕ слоты рамки,
поэтому раскладка здесь работает в более тесной обстановке, чем в замере. Цифру отгрузки даёт
только A/B по выданным файлам — §6.90.
"""
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL = "slot_model_g250.npz"     # лежит В РЕПОЗИТОРИИ (§6.68), 37 КБ
DEFAULT_GATE = "frac0.8"
NF = 14

_W = {}                                   # имя → выгруженные деревья (грузятся один раз)
_SAID = set()


def _announce(msg):
    """Сказать РОВНО ОДИН РАЗ на процесс: режим раскладки одинаков для всех листов."""
    if msg not in _SAID:
        _SAID.add(msg)
        print(f"  {msg}")


def resolve(model_path):
    """Имя файла → `auto/models/<имя>`; путь с каталогом — как есть. Пусто = выключено.
    ⚠ Та же причина, что в `trace_seq.resolve`: путь в чужой `output/...` есть на одной машине."""
    if not model_path:
        return None
    p = Path(model_path)
    return MODELS_DIR / p.name if p.name == str(model_path) else p


def available(model_path):
    """Есть ли вес. numpy у прода уже есть, поэтому других условий нет."""
    p = resolve(model_path)
    return bool(p and p.is_file())


def load(model_path):
    """Выгруженные деревья. Кэш на процесс."""
    p = resolve(model_path)
    key = str(p)
    if key not in _W:
        z = np.load(p)
        w = {k: z[k] for k in z.files}
        if int(w["n_features"][0]) != NF:
            raise ValueError(f"вес {p.name}: признаков {int(w['n_features'][0])}, код ждёт {NF}")
        _W[key] = w
    return _W[key]


def predict(w, X):
    """Скор пар = ПРЕДСКАЗАННАЯ ВЕЛИЧИНА ОШИБКИ (log1p(px) + штраф покрытия), чем МЕНЬШЕ, тем лучше.
    Возвращается уже со знаком «больше = лучше», как в стенде (`SR = -reg.predict`)."""
    X = np.asarray(X, np.float32).astype(np.float64)
    if X.ndim != 2 or X.shape[1] != NF:
        raise ValueError(f"ожидалась матрица (n, {NF}), пришло {X.shape}")
    cl, cr, fe, th = w["children_left"], w["children_right"], w["feature"], w["threshold"]
    va, off = w["value"], w["offsets"]
    rows = np.arange(len(X))
    acc = np.zeros(len(X))
    for k in range(len(off) - 1):
        b = int(off[k])
        node = np.zeros(len(X), np.int32)
        while True:
            f = fe[b + node]
            leaf = f < 0
            if leaf.all():
                break
            go_l = X[rows, np.where(leaf, 0, f)] <= th[b + node]
            node = np.where(leaf, node, np.where(go_l, cl[b + node], cr[b + node]))
        acc += va[b + node]
    return -(float(w["base"][0]) + float(w["lr"][0]) * acc)


def _tfeat(tr):
    """Форма трассы. ИДЕНТИЧНО обучению (`_slot_abstain.tfeat`, `_slot_learn.tfeat`)."""
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


def _parse_gate(spec):
    """'frac0.8' → ('frac', 0.8). Пусто → умолчание. 'off' → лист всегда за правилом."""
    s = (spec or DEFAULT_GATE).strip()
    if s == "off":
        return ("off", 0.0)
    for kind in ("frac", "minx", "mean"):
        if s.startswith(kind):
            try:
                return (kind, float(s[len(kind):]))
            except ValueError:
                break
    raise ValueError(f"slot_gate={spec!r}: ожидалось frac<порог>, minx<порог>, mean<порог> или off")


def map_lines(traces, model, frame, mnemonics_path, cv):
    """{slot_name: (Line, trace)} по ОБУЧЕННОЙ раскладке, либо None = «отказ, берите правило».

    ⚠ Кандидаты слота — ВСЕ линии его трека, без фильтра цвета: так строилось обучение. Фильтр
    цвета есть у правила, и он остаётся его свойством.
    """
    from . import emit as emit_mod                      # ленивый импорт: иначе цикл emit↔slot_model
    from . import meta as meta_mod
    name = getattr(cv, "slot_model", "") or ""
    if not name:
        return None
    if not available(name):
        _announce(f"⚠ slot_model={name!r} ЗАПРОШЕНА, НО НЕ НАЙДЕНА ({resolve(name)}) — "
                  f"раскладка идёт ПРАВИЛОМ, выдача прежняя")
        return None
    kind, thr = _parse_gate(getattr(cv, "slot_gate", "") or DEFAULT_GATE)
    w = load(name)
    _announce(f"раскладка: ОБУЧЕННАЯ МОДЕЛЬ {resolve(name).name}, отказ листом при {kind}<{thr}")

    slots = []
    for c in model.get("curves", []):
        nm = c.get("name", "")
        if meta_mod.mnem_root(nm) != "DA":
            info = meta_mod.curve_info(nm, mnemonics_path)
            slots.append(dict(name=nm, track=emit_mod._slot_track(model, c, frame),
                              color=info["color"], cls=info["class"]))
    if not slots:
        return None

    by_track = {}
    for L, tr in traces:
        by_track.setdefault(L.track_index, []).append((L, tr))
    from . import slot_geom
    got = assign(slots, by_track, w, kind, thr, sib=float(getattr(cv, "slot_sib", 0.0) or 0.0),
                 forbid=slot_geom.make_forbid(model, cv))
    if got is None:
        _announce("  (лист(ы) не прошли меру уверенности — там раскладка ПРАВИЛОМ)")
        return None
    return {nm: (L, tr) for nm, (L, tr, _) in got.items()}


def rows(slots, by_track, forbid=None):
    """Признаки всех пар (слот × линия ТОГО ЖЕ трека) → (матрица, [(имя слота, Line, трасса)]).

    ⚠ ИДЕНТИЧНО ОБУЧЕНИЮ (`_slot_abstain.build`), сверено побитово в `_slot_parity.py`. Кандидаты
    слота — ВСЕ линии его трека, без фильтра цвета: фильтр цвета есть у правила и остаётся его
    свойством, а модель про цвет узнаёт из признаков 1-2."""
    out, pairs = [], []
    for s in slots:
        same = by_track.get(s["track"], [])
        if not same:
            continue
        cache = [(L, tr, _tfeat(tr)) for L, tr in same]
        meds = sorted(f[5] for _, _, f in cache)
        for L, tr, f in cache:
            # ★ §6.215: пара, нарушающая геометрию шкалы слота, не существует (как цветовой фильтр у
            #   правила). `forbid=None` (умолчание) — поведение прежнее бит-в-бит.
            if forbid is not None and forbid(s["name"], tr):
                continue
            wig, rev, span, rough, n, med, dy = f
            cls = "SP" if L.behavior == "smooth" else "RES"
            rank = meds.index(min(meds, key=lambda v: abs(v - med))) / max(1, len(meds) - 1)
            out.append([1.0 if (s["color"] and s["color"] == L.color) else 0.0,
                        1.0 if s["color"] is None else 0.0,
                        1.0 if s["cls"] in (cls, "OTHER", "CALI") else 0.0,
                        1.0 if s["cls"] == "SP" else 0.0,
                        rank, float(L.x_center) / 1000.0, med / 1000.0,
                        wig, rev, rough, span / 1000.0, n / 10000.0,
                        len(same) / 10.0, dy / 10000.0])
            pairs.append((s["name"], L, tr))
    return (np.array(out, float) if out else np.zeros((0, NF))), pairs


def _sib_term(slots, pairs):
    """§6.130: согласие ранга слота по индексу зонда с рангом кандидата по x, [0..1] на пару."""
    import re as _re
    fam = {}
    for s in slots:
        tok = s["name"].split()[0]
        m = _re.match(r"^([A-Za-z_]+)(\d+)$", tok)
        if m:
            fam.setdefault((s["track"], m.group(1)), []).append((int(m.group(2)), s["name"]))
    rank = {}
    for key, items in fam.items():
        if len(items) < 2:
            continue
        for r, (_, nm) in enumerate(sorted(items)):
            rank[nm] = (r, len(items), key)
    out = np.zeros(len(pairs))
    if not rank:
        return out
    for key in {v[2] for v in rank.values()}:
        ks = [k for k in range(len(pairs)) if rank.get(pairs[k][0], (0, 0, None))[2] == key]
        if not ks:
            continue
        uniq = sorted({pairs[k][1].x_center for k in ks})
        for k in ks:
            r, n, _ = rank[pairs[k][0]]
            fx = uniq.index(pairs[k][1].x_center) / max(1, len(uniq) - 1)
            fs = r / max(1, n - 1)
            out[k] = 1.0 - abs(fx - fs)
    return out


def assign(slots, by_track, w, kind, thr, sib=0.0, forbid=None):
    """Ядро: признаки → скор → жадное 1:1 → отказ листом. None = отказ.

    Вынесено из `map_lines`, чтобы стенд мог сверить прод-путь с обучением ПО ПУЛАМ, не запуская
    пайплайн (`_slot_parity.py`). slots — список dict(name, track, color, cls);
    by_track — {трек: [(Line, {row: x})]}, у Line нужны .color, .behavior, .x_center.

    ★ §6.130: `sib` — вес добавки «порядок зондов в семействе». 0.0 = ПРОД, поведение бит-в-бит
    прежнее. Раскладка путает братские зонды одного семейства (GZ1…GZ5): из 14 признаков ни один
    не кодирует, который из зондов перед нами. Добавка сближает ранг слота по индексу зонда с
    рангом кандидата по x. Эталон не нужен — оба ранга известны на месте.
    ⚠ Направление конвенции ОДНО и прямое: по листу его выбрать нельзя (неоткуда узнать без
    эталона), берётся большинство — 53% против 20% (§6.129)."""
    X, pairs = rows(slots, by_track, forbid)
    if not len(X):
        return None
    sc = predict(w, X)
    if sib:
        sc = sc + sib * _sib_term(slots, pairs)

    # ── margin ПАРЫ внутри слота: её скор минус лучший из ОСТАЛЬНЫХ кандидатов этого слота ─────
    per_slot = {}
    for k, (nm, _, _) in enumerate(pairs):
        per_slot.setdefault(nm, []).append(k)
    marg, ncand = np.zeros(len(sc)), np.zeros(len(sc), int)
    for nm, ks in per_slot.items():
        v = sorted((sc[q] for q in ks), reverse=True)
        for q in ks:
            ncand[q] = len(ks)
            marg[q] = np.inf if len(v) == 1 else sc[q] - (v[1] if sc[q] >= v[0] else v[0])

    # ── жадное 1:1 по убыванию скора: тот же контракт, что у правила ───────────────────────────
    got, used = {}, set()
    for k in sorted(range(len(sc)), key=lambda q: -sc[q]):
        nm, L, tr = pairs[k]
        if nm in got or id(L) in used:
            continue
        got[nm] = (L, tr, k); used.add(id(L))
    if not got:
        return None

    # ── ОТКАЗ ЛИСТОМ ЦЕЛИКОМ ──────────────────────────────────────────────────────────────────
    ks = [v[2] for v in got.values()]
    if kind == "off":
        ok = False
    elif kind == "frac":
        ok = float(np.mean([marg[k] >= 0.3 for k in ks])) >= thr
    elif kind == "mean":
        # §6.109: СРЕДНИЙ margin назначенных слотов. На корпусе 2677 листов (202 скважины,
        # вложенный выбор порога) даёт 832 → 947 против 832 → 894 у `frac` — почти вдвое.
        # ⚠⚠ КЛИП ±5 ОБЯЗАТЕЛЕН И ПОВТОРЯЕТ СТЕНД (`_slot_abstain.MGS`): у слота с ЕДИНСТВЕННЫМ
        # кандидатом margin = +∞, и без клипа один такой слот делает среднее бесконечным, то есть
        # гейт пропускал бы лист всегда. Разойдись клип со стендом — прод считал бы не то, что
        # померено, и это не всплыло бы: обе стороны «работают».
        ok = float(np.mean(np.clip([marg[k] for k in ks], -5.0, 5.0))) >= thr
    else:                                              # minx: одиночки — по предсказанной ошибке
        ok = min((marg[k] if ncand[k] > 1 else sc[k] + 2.0) for k in ks) >= thr
    return got if ok else None


def selftest(model_path=DEFAULT_MODEL):
    """Вес грузится, форма признаков совпадает, скор конечен, гейты разбираются. Без данных листа."""
    if not available(model_path):
        return f"нет веса: {resolve(model_path)}"
    w = load(model_path)
    X = np.zeros((3, NF)); X[1] = 0.5; X[2] = 1.0
    s = predict(w, X)
    if not np.all(np.isfinite(s)):
        return "скор не конечен"
    if len(s) != 3:
        return f"скоров {len(s)}, ждали 3"
    for g in ("frac0.8", "minx0.0", "mean0.05", "off"):
        _parse_gate(g)
    return ""


if __name__ == "__main__":
    e = selftest()
    print(f"slot_model selftest: {'ЧИСТО' if not e else 'ОШИБКА: ' + e}")
    w = load(DEFAULT_MODEL)
    print(f"деревьев {len(w['offsets']) - 1}, узлов {len(w['feature'])}, "
          f"lr {float(w['lr'][0])}, база {float(w['base'][0]):.4f}")

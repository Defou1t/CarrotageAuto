r"""_slot_rules.py — ПЕРЕБОР ПРАВИЛ РАСКЛАДКИ трасс по слотам, ОФЛАЙН по сохранённым пулам.

§6.79: в пуле 15-17 честных кривых, в файл уходит 5. §6.80 (разбор `_slot_why`): 60% потерь даёт
ПОРЯДОК по `x_center`, 30% — класс (`behavior` SP/RES), 10% — фильтр цвета.

⚠ ПОЧЕМУ ОФЛАЙН. Каждый вариант правила требовал бы полного прогона пайплайна по всем листам
(~20 минут на вариант). Но правило — ЧИСТАЯ ФУНКЦИЯ над пулом трасс и списком слотов, поэтому
пул сохраняется один раз (`_pool_oracle.py --dump`), а варианты сравниваются за секунды. Это та же
идея, на которой стоит `_pick_gate` («пайплайн гоняется ОДИН РАЗ на лист»).

ПРАВИЛА (каждое — как выбрать, какая линия идёт в какой слот внутри трека):
  prod        — как сейчас: фильтр цвета, затем сортировка по (класс, x_center);
  no_color    — то же без фильтра цвета;
  no_class    — то же без учёта класса;
  bare        — только порядок по x_center, без цвета и класса;
  med_x       — вместо x_center линии берётся МЕДИАНА x самой трассы (x_center — центр полосы от
                U1, он не обязан совпадать с тем, где реально идёт трасса);
  hungarian   — глобальное назначение (венгерский алгоритм) по стоимости |ранг слота − ранг линии|
                вместо жадной сортировки: слоты рамки упорядочены, линии упорядочены по медиане x,
                и минимизируется суммарное расхождение рангов, а не назначается по одному.
★ ОРАКУЛ — верхняя граница: назначение по РЕАЛЬНОЙ ошибке против эталона (в проде недоступно,
нужен только как потолок).

  <ComfyUI>\python_embeded\python.exe _slot_rules.py [--pools <dir>]
"""
import sys, argparse, pickle, itertools
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pools", default=r"F:\nds\output\taskS\pools")
a = ap.parse_args()

NULL_OK = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


_FEAT = {}


def feats(tr):
    """Признаки ФОРМЫ трассы (§6.83): та же тройка, что меряет `behavior_priors` по эталонам.

    ⚠ ЗАЧЕМ ФОРМА. §6.82 закрыл ВСЕ способы прочитать ПОЛОЖЕНИЕ. Форма — другая ось, и у неё есть
    физическое основание: зонды БКЗ различаются ДЛИНОЙ, а длинный зонд имеет меньшее вертикальное
    разрешение, то есть даёт более ГЛАДКУЮ кривую. Значит гладкость должна УПОРЯДОЧИВАТЬ зонды —
    это не бит «SP/RES» (он уже используется и ошибается на 30% потерь, §6.80), а ключ сортировки.
    ⚠ Приоры `corpus/priors.json` тут не годятся: они классовые (RES против SP) и построены на
    7 кривых — тот же один бит. Поэтому признак берётся ОТНОСИТЕЛЬНЫЙ, без внешних порогов."""
    k = id(tr)
    if k in _FEAT:
        return _FEAT[k]
    ys = sorted(tr)
    x = np.array([tr[y] for y in ys], float)
    if len(x) < 5:
        f = {"wig": 0.0, "rev": 0.0, "span": 0.0, "rough_n": 0.0}
    else:
        dx = np.diff(x)
        span = float(np.percentile(x, 90) - np.percentile(x, 10)) or 1.0
        w = min(101, len(x) // 2 * 2 + 1)
        sm = np.convolve(x, np.ones(w) / w, mode="same") if w >= 3 else x
        f = {"wig": float(np.median(np.abs(dx))),
             "rev": float(np.mean((dx[:-1] * dx[1:]) < 0)) if len(dx) > 2 else 0.0,
             "span": span,
             "rough_n": float(np.std(x - sm) / span)}
    _FEAT[k] = f
    return f


def key_x(ln, how):
    if how == "med":
        v = list(ln["tr"].values())
        return float(np.median(v)) if v else ln["x_center"]
    if how.startswith("-"):                       # обратный порядок того же признака
        return -feats(ln["tr"])[how[1:]]
    if how in ("wig", "rev", "span", "rough_n"):
        return feats(ln["tr"])[how]
    return ln["x_center"]


def copeland_order(lines, sep=0.0):
    """Порядок линий по МАЖОРИТАРНОМУ голосованию ПО СТРОКАМ, а не по сводной координате.

    ⚠ ЗАЧЕМ. Сводная координата (`x_center` или медиана трассы) ломается ровно там, где ломается
    вся задача: на ПЕРЕСЕЧЕНИЯХ. Две кривые могут иметь почти одну медиану и при этом на 80% строк
    идти в определённом порядке. Здесь для каждой пары считается доля общих строк, где A левее B;
    побеждает тот, кто левее НА БОЛЬШИНСТВЕ строк, а итоговый ранг — число побед (Copeland).

    `sep` — учитывать только строки, где линии разнесены дальше этого: у самого пересечения знак
    разности случаен и добавляет шум в голосование."""
    idx = [i for i, _ in lines]
    wins = {i: 0 for i in idx}
    for (ia, la), (ib, lb) in itertools.combinations(lines, 2):
        ta, tb = la["tr"], lb["tr"]
        com = [y for y in ta if y in tb]
        if not com:
            continue
        dif = np.array([ta[y] - tb[y] for y in com], float)
        if sep > 0:
            dif = dif[np.abs(dif) > sep]
        if not len(dif):
            continue
        if (dif < 0).mean() > 0.5:
            wins[ia] += 1
        elif (dif < 0).mean() < 0.5:
            wins[ib] += 1
    return sorted(lines, key=lambda q: (-wins[q[0]], key_x(q[1], "med")))


def assign(d, use_color=True, use_class=True, xhow="center", hungarian=False,
           majority=False, sep=0.0):
    """Вернуть {имя слота: трасса}. Повторяет прод-правило, кроме явно отключённого."""
    out = {}
    used = set()
    tracks = {s["track"] for s in d["slots"]}
    for ti in tracks:
        tslots = [s for s in d["slots"] if s["track"] == ti]
        tlines = [(i, ln) for i, ln in enumerate(d["lines"]) if ln["track"] == ti]
        if not tslots or not tlines:
            continue
        if majority or hungarian:
            order = (copeland_order(tlines, sep) if majority
                     else sorted(tlines, key=lambda q: key_x(q[1], xhow)))
            n = min(len(tslots), len(order))
            for si in range(n):                       # ранг слота ↔ ранг линии, один к одному
                i, ln = order[si]
                if i in used:
                    continue
                used.add(i); out[tslots[si]["name"]] = ln["tr"]
            continue
        colors = {ln["color"] for _, ln in tlines}
        # ⚠ Мажоритарный порядок как КЛЮЧ СОРТИРОВКИ, а не как отдельная ветка назначения: иначе
        # тест смешал бы два изменения сразу (новый порядок И отказ от цвета/класса) и стал бы
        # неинтерпретируемым — ровно эта ошибка была в первой редакции §6.82.
        rank = ({i: r for r, (i, _) in enumerate(copeland_order(tlines, sep))}
                if xhow == "major" else None)
        pairs = []
        for s in tslots:
            strict = use_color and s["color"] is not None and s["color"] in colors
            for i, ln in tlines:
                if strict and s["color"] != ln["color"]:
                    continue
                cls = "SP" if ln["behavior"] == "smooth" else "RES"
                ok = (not use_class) or (s["class"] in (cls, "OTHER", "CALI"))
                pairs.append((0 if ok else 1, s, i, ln))
        pairs.sort(key=lambda q: (q[0], rank[q[2]] if rank is not None else key_x(q[3], xhow)))
        taken = set()
        for _, s, i, ln in pairs:
            if s["name"] in taken or i in used:
                continue
            taken.add(s["name"]); used.add(i); out[s["name"]] = ln["tr"]
    return out


def oracle(d):
    """ПОТОЛОК: назначение по реальной ошибке (в проде невозможно)."""
    pairs = []
    for nm, gt in d["gts"].items():
        for i, ln in enumerate(d["lines"]):
            m, c = err(ln["tr"], gt)
            if m is not None:
                pairs.append((c < 0.9, m, nm, i))
    pairs.sort()
    out, used = {}, set()
    for _, _, nm, i in pairs:
        if nm in out or i in used:
            continue
        out[nm] = d["lines"][i]["tr"]; used.add(i)
    return out


RULES = {
    "prod (как сейчас)": dict(use_color=True, use_class=True, xhow="center"),
    "без фильтра цвета": dict(use_color=False, use_class=True, xhow="center"),
    "без класса": dict(use_color=True, use_class=False, xhow="center"),
    "только порядок": dict(use_color=False, use_class=False, xhow="center"),
    "медиана x трассы": dict(use_color=True, use_class=True, xhow="med"),
    "медиана x, без цвета": dict(use_color=False, use_class=True, xhow="med"),
    "медиана x, без цвета и класса": dict(use_color=False, use_class=False, xhow="med"),
    "венгерский по рангу (мед. x)": dict(hungarian=True, xhow="med"),
    # ★ НОВЫЙ ПРИЗНАК (§6.82): порядок ПО СТРОКАМ вместо сводной координаты.
    # Ниже он подставлен КЛЮЧОМ СОРТИРОВКИ в прод-правило (цвет и класс на месте) — иначе
    # сравнение мерило бы сумму двух изменений, а не сам признак.
    "мажоритарный порядок": dict(xhow="major"),
    "мажоритарный, разнос >20px": dict(xhow="major", sep=20.0),
    "мажоритарный, разнос >50px": dict(xhow="major", sep=50.0),
    "мажоритарный, без цвета": dict(xhow="major", use_color=False),
    "мажоритарный ранг-в-ранг (без цвета/класса)": dict(majority=True),
    # ★ ДРУГАЯ ОСЬ (§6.83): форма трассы вместо положения. Обе стороны каждого признака —
    # направление связи «длина зонда ↔ гладкость» заранее не известно, а гейт трёх наборов
    # не даст принять случайную.
    "форма: гладкость (wig)": dict(xhow="wig"),
    "форма: гладкость, обратно": dict(xhow="-wig"),
    "форма: развороты (rev)": dict(xhow="rev"),
    "форма: развороты, обратно": dict(xhow="-rev"),
    "форма: ВЧ-дрожь (rough_n)": dict(xhow="rough_n"),
    "форма: ВЧ-дрожь, обратно": dict(xhow="-rough_n"),
    "форма: размах (span)": dict(xhow="span"),
    "форма: размах, обратно": dict(xhow="-span"),
}

files = sorted(Path(a.pools).glob("*.pkl"))
data = [pickle.load(open(f, "rb")) for f in files]
ncur = sum(len(d["gts"]) for d in data)
print(f"листов {len(data)}, экспертных кривых {ncur}, трасс в пулах "
      f"{sum(len(d['lines']) for d in data)}\n")

def per_sheet(kw):
    """Честных НА ЛИСТ — иначе не проверить монотонность (§6.38/§6.49: сумма мало что значит,
    если она собрана из плюсов на одних листах и минусов на других)."""
    out = []
    for d in data:
        got = assign(d, **kw)
        out.append(sum(1 for nm, gt in d["gts"].items()
                       if got.get(nm) and NULL_OK(*err(got[nm], gt))))
    return out


base = None
base_ps = None
print(f"{'правило':<32}{'честных':>9}{'против прода':>14}{'листов ↑ / ↓':>16}")
for name, kw in RULES.items():
    ps = per_sheet(kw)
    tot = sum(ps)
    if base is None:
        base, base_ps = tot, ps
    up = sum(1 for a, b in zip(base_ps, ps) if b > a)
    dn = sum(1 for a, b in zip(base_ps, ps) if b < a)
    print(f"{name:<32}{tot:>9}{tot - base:>+14d}{f'{up} / {dn}':>16}")

orc = sum(1 for d in data for nm, gt in d["gts"].items()
          if oracle(d).get(nm) and NULL_OK(*err(oracle(d)[nm], gt)))
print(f"{'★ ОРАКУЛ (потолок)':<32}{orc:>9}{orc - base:>+14d}")

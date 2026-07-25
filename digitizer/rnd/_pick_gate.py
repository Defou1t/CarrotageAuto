r"""_pick_gate.py — ШИРОКИЙ ГЕЙТ ОТБОРА K ТРАСС ИЗ N (открытая подзадача §6.45).

ПОЧЕМУ ОТДЕЛЬНЫЙ СТЕНД. `_emit_traces.py` кладёт в слоты K трасс из N найденных, и на контрольном
листе Semeguniv цикл «чиню одну кривую — ломаю другую» повторился ТРИЖДЫ (охват глубины,
разнесённость медиан, гладкость). Durable-вывод §6.45: подбор критерия на ОДНОМ листе — подгонка,
решать надо широким гейтом. Здесь тот же набор листов, что у `_esc_limit`/`_esc_gate` (26 листов),
и та же метрика с контролями §6.36.

★ ГЛАВНОЕ ПО ЦЕНЕ: пайплайн гоняется ОДИН РАЗ НА ЛИСТ, его трассы и плотный GT кладутся в кэш
(идея `_relatch_bench`). Критерий отбора — чистая функция над кэшем, поэтому перебор семи
критериев стоит секунды, а не семь прогонов архива. Добавить критерий = дописать функцию в PICKERS
и прогнать без --build.

МЕТРИКА (§6.9/§6.36, не менять — иначе цифры несравнимы с историей):
  * назначение трасс кривым ОДИН-К-ОДНОМУ среди ОТОБРАННЫХ (иначе одна трасса «закрывает» две);
  * ЧЕСТНАЯ кривая = med<=3px И cov>=0.9; med без cov не цитируется;
  * контроль копирования эксперта (сетка+значения, доля точных совпадений x) — утёкшие не в зачёт;
  * имена НЕ проверяются (постановка §6.33): вопрос только в том, ОТЛИЧЕНЫ ли K линий.

★ ПОТОЛОК (`oracle`) — назначение среди ВСЕХ N трасс без отбора. Это то, что §6.45 показал на
контрольном листе: верная трасса в пуле ЕСТЬ, её не находит отбор. Разрыв `oracle − критерий` и
есть цена подзадачи; смысл гейта — двигать именно его, а не абсолютное число честных.

⚠⚠ ГЛАВНОЕ, ЧТО ПОКАЗАЛ ЭТОТ СТЕНД (§6.48-§6.49): ПОРЯДОК КРИТЕРИЕВ ЗАВИСИТ ОТ ПОПУЛЯЦИИ ЛИСТОВ.
Лучшие на 60 новых листах (`nl+xvar`, `npts*xvar`) были ХУДШИМИ на гейте; разброс всех разумных
критериев внутри одного набора — +1..+4 к проду, то есть они там статистически неразличимы.
★ ПРАВИЛО: критерий выбирается ТОЛЬКО режимом `--cv` по ТРЁМ независимым наборам (106 листов /
417 кривых), по сумме И по числу деградировавших листов. Цифры в списке ниже — замер на ГЕЙТЕ,
это НЕ рейтинг «что лучше вообще»; итоговый рейтинг — в §6.49.

ИТОГ (106 листов, потолок 75): nl+npts 55 ★ПРИНЯТ, nl+loc 53, npts*xvar 52, npts 51,
nl+xvar 49, ОХВАТ ГЛУБИНЫ (прод до 24.07) 46, smooth 37.

КРИТЕРИИ (цифры — честных из 90 НА ГЕЙТЕ, потолок 19; §6.46-§6.49):
  oracle    19  потолок, отбора нет (в прод не годится: K трасс выбрать всё равно надо);
  depth      9  охват глубины — ПРОД ДО 24.07. ⚠ Латч ДЛИННЕЕ настоящей ПО ПОСТРОЕНИЮ (его охват —
                сумма двух кривых), поэтому «самая длинная» систематически берёт латч;
  xspread    9  жадная разнесённость медиан (КОНТРОЛЬ: ломает плотно стоящие тройки);
  smooth    10  гладкость, медиана |dx| (КОНТРОЛЬ: топит кривые с НАСТОЯЩИМИ оборотами пера);
  nl+xvar   11  фильтр латча + размах пера как порядок;
  dens      12  плотность len/span при отсеве коротышей;
  loc       12  ВТОРАЯ подпись латча: ведущий двойник МЕНЯЕТСЯ ПО ГЛУБИНЕ;
  npts      13  по числу точек;
  nl&loc    13  обе подписи латча как И;
  npts*xvar 13  точки × размах;
  nl+loc    14  обе подписи латча как ИЛИ;
  nl+dens   15  фильтр латча + плотность;
★ nl+npts   16  ПРИНЯТ В ПРОД (§6.49) при --dedup-tol 50: на 106 листах трёх наборов 55 против
                46 у охвата, улучшено 10 листов, деградировало 2. ⚠ При --dedup-tol 40 давал на
                валидационном наборе −1 (§6.48) — параметры подбирать ТОЛЬКО на всех трёх.

★ ПОДПИСЬ ЛАТЧА (`split_score`): у трассы, идущей ЧАСТЬЮ по одной кривой и частью по другой, ДВА
разных ЧАСТИЧНЫХ согласия с другими кандидатами. У настоящей — либо один почти полный двойник
(норма: линий U1 больше, чем кривых), либо ничего. Берётся ВТОРОЕ по величине согласие среди
НЕ-дубликатов. Признак ВНУТРЕННИЙ, GT не нужен.
⚠ Сам фильтр в одиночку не помогает (`nolatch` = 9, как прод) — работает только связка
«фильтр латча + порядок по массе точек».
⚠ Обе подписи латча ложно срабатывают на ШИРОКИХ кривых: кто ходит по всему треку, тот частично
согласен со всеми (GZ11 контрольного листа: размах 636px, split 0.43). Отсюда и потолок метода.

⚠ ДВА РАЗНЫХ ПОРОГА, НЕ СКЛЕИВАТЬ (§6.47): `--dup-tol` 20 = «идут вместе» (для согласия),
`--dedup-tol` 50 = «это одна и та же кривая» (для слияния). Плато НА 106 ЛИСТАХ:
слияние 20/30/40/50/70 → 53/54/54/55/54 (охват на всём диапазоне 44..46);
разделённость 0.10/0.15/0.20/0.25/0.35 → 52/54/55/51/51. Выигрыш держится всюду.

  python _pick_gate.py --build [--n 20]        # собрать кэш (дорого, один раз)
  python _pick_gate.py                         # оценить критерии по кэшу (минута)
  python _pick_gate.py --diag PDGRAK           # ПОЧЕМУ отбор ошибся: признаки всех кандидатов
  python _pick_gate.py --build --skip 20 --n 20 --cache <dir>   # ещё один набор листов
  python _pick_gate.py --cv "<dir1>,<dir2>,<dir3>" --dedup-tol 50   # ★ТАК И ТОЛЬКО ТАК выбирать
"""
import sys, io, json, pickle, argparse, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image
from _multi_replica_probe import dense
from _emit_traces import curve_ifds
from write_nlgx import read_full

OUT = Path(r"F:\nds\output\taskS\pick_gate")
CACHE = OUT / "cache"

ap = argparse.ArgumentParser()
ap.add_argument("--build", action="store_true", help="прогнать пайплайн и собрать кэш")
ap.add_argument("--n", type=int, default=20, help="сколько листов добрать из train_sheets")
ap.add_argument("--picks", default="oracle,depth,npts,dens,nolatch,nl+npts,nl+dens")
ap.add_argument("--dup-tol", type=float, default=20.0, help="порог «та же кривая» для дубликатов")
ap.add_argument("--split-thr", type=float, default=0.20, help="порог разделённого согласия (nolatch)")
ap.add_argument("--switch-thr", type=float, default=0.20, help="порог смены партнёра по глубине")
ap.add_argument("--bins", type=int, default=5, help="бинов по глубине для switch_score")
ap.add_argument("--skip", type=int, default=0,
                help="ВАЛИДАЦИЯ: пропустить первые N листов train_sheets (они уже в гейте)")
ap.add_argument("--cache", default="", help="иной каталог кэша (для валидационного набора)")
ap.add_argument("--dedup-tol", type=float, default=50.0,
                help="порог СЛИЯНИЯ трасс в одну кривую — ПРОД-значение (§6.49); НЕ путать с "
                     "--dup-tol 20 (согласие). Были склеены до §6.47")
ap.add_argument("--diag", default="", help="подстрока листа: печать признаков всех кандидатов")
ap.add_argument("--diag-pick", default="nl+npts", help="какой критерий показывать в --diag")
ap.add_argument("--k2", type=int, default=0,
                help="СБОРКА ТРАНША K=2 по содержимому рамки (§6.7), а не по train_sheets")
ap.add_argument("--dark-from", default="",
                help="ПЕРЕСБОРКА: взять из этих кэшей листы с медианой V < --dark-v и собрать их "
                     "заново (для замера правки тона, §6.52)")
ap.add_argument("--dark-v", type=int, default=200, help="порог медианы V «тёмная бумага»")
ap.add_argument("--p90-band", default="", help="ПЕРЕСБОРКА по полосе p90(V): 'lo,hi'")
ap.add_argument("--bridge", type=int, default=0,
                help="МОСТИК: линейно заполнить разрывы трассы длиной <= N строк (§6.52). "
                     "0 = как сейчас. Экспертная полилиния имеет вершины через 6-23 строки, "
                     "поэтому разрыв короче этого в ЕЁ представлении вообще не существует")
ap.add_argument("--jump-limit", type=float, default=0,
                help="СБОРКА С ПРЕДОХРАНИТЕЛЕМ ПРЫЖКА (§6.56): не садиться на ран дальше N px от "
                     "предсказания, а коастить по инерции и НЕ писать строку. 0 = как в проде "
                     "(предохранитель есть в trace2d, но НИКЕМ не передаётся)")
ap.add_argument("--dp", action="store_true",
                help="СБОРКА ДЕКОДЕРОМ ПУТИ (§6.61): глобальный ДП по колонке вместо жадного "
                     "выбора рана. Дешёвая проверка постановки P2 §6.8 без обучения")
ap.add_argument("--dp-lam", type=float, default=0.02, help="приор полосы для ДП")
ap.add_argument("--dp-skip", type=float, default=6.0, help="штраф пропуска строки для ДП")
ap.add_argument("--dp-wid", type=float, default=0.0,
                help="ДП + ПРИЗНАК ИДЕНТИЧНОСТИ: вес штрафа за несоответствие ширины штриха "
                     "(§6.64). 0 = чистая гладкость, как в §6.61")
ap.add_argument("--seq", default="",
                help="СБОРКА ОБУЧЕННЫМ ОКОННЫМ СЕЛЕКТОРОМ (§6.66): имя чекпойнта в "
                     "output/taskS/decoder, напр. seq_model_d45p.pt. Требует torch")
ap.add_argument("--cv", default="",
                help="каталоги кэшей через запятую: перекрёстный замер по НЕЗАВИСИМЫМ наборам")
ap.add_argument("--json", default=str(OUT / "res.json"))
a = ap.parse_args()

if a.cache:
    CACHE = Path(a.cache)
MINPTS = 30                      # как в _emit_traces: короче — не кандидат


def bridge(t, maxgap):
    """Заполнить ЛИНЕЙНО разрывы <= maxgap строк. Это не выдумывание данных: эксперт хранит
    кривую ПОЛИЛИНИЕЙ с шагом вершин 6-23 строки (докстринг `dense`), и разрыв короче шага в его
    представлении не существует вовсе. Длинные разрывы НЕ мостим — там кривая действительно
    потеряна, и её надо оставить видимой в метрике."""
    if maxgap <= 0 or len(t) < 2:
        return t
    ys = sorted(t); out = dict(t)
    for y0, y1 in zip(ys, ys[1:]):
        g = y1 - y0 - 1
        if 0 < g <= maxgap:
            x0, x1 = t[y0], t[y1]
            for k in range(1, g + 1):
                out[y0 + k] = x0 + (x1 - x0) * k / (g + 1)
    return out


# ─────────────────────────── общая геометрия трасс (без GT) ───────────────────────────
def span(t):
    return (max(t) - min(t)) if t else 0


def agree(t1, t2, tol):
    """Доля ОБЩИХ строк, где трассы совпадают в пределах tol. Это ЧАСТИЧНОЕ согласие: у латча
    оно велико сразу с двумя разными кандидатами, у дубликата — почти 1 с одним."""
    com = [y for y in t1 if y in t2]
    if len(com) < 50:
        return 0.0
    return float(np.mean([abs(t1[y] - t2[y]) < tol for y in com]))


def same_curve(t1, t2, tol):
    """«Та же кривая» — как в проде (_emit_traces): медиана |dx| на общих строках < tol.
    ⚠ Не заменять на расстояние медиан: правая тройка Semeguniv стоит плотно (1980/2046/2089)."""
    com = [y for y in t1 if y in t2]
    if len(com) < 50:
        return False
    return float(np.median([abs(t1[y] - t2[y]) for y in com])) < tol


def dedup_take(order, K, tol):
    """Взять K различных по порядку `order`; если различных не хватило — добрать остальными.
    Это ровно прод-логика §6.45 (две трассы могут идти по одной кривой, и тогда настоящая
    кривая остаётся без слота)."""
    picked = []
    for t in order:
        if len(picked) >= K:
            break
        if not any(same_curve(t, q, tol) for q in picked):
            picked.append(t)
    for t in order:
        if len(picked) >= K:
            break
        if all(t is not q for q in picked):
            picked.append(t)
    return picked


def split_score(t, cand, tol):
    """ВТОРОЕ по величине частичное согласие с кандидатами, которые НЕ являются дубликатами.
    Дубликат (согласие >= .85) исключён специально: две трассы одной кривой — норма, а вот
    согласие с ДВУМЯ РАЗНЫМИ кривыми по кускам — подпись латча."""
    ps = []
    for q in cand:
        if q is t:
            continue
        p = agree(t, q, tol)
        if p < 0.85:
            ps.append(p)
    ps.sort(reverse=True)
    return ps[1] if len(ps) > 1 else 0.0


# ─────────────────────────────── критерии отбора ───────────────────────────────
def pick_oracle(cand, K, o):
    return list(cand)                                   # потолок: отбора нет


def pick_depth(cand, K, o):
    return dedup_take(sorted(cand, key=span, reverse=True), K, o.dedup_tol)


def pick_npts(cand, K, o):
    return dedup_take(sorted(cand, key=len, reverse=True), K, o.dedup_tol)


def pick_dens(cand, K, o):
    if not cand:
        return []
    smax = max(span(t) for t in cand) or 1
    keep = [t for t in cand if span(t) >= 0.6 * smax] or list(cand)
    return dedup_take(sorted(keep, key=lambda t: len(t) / max(1, span(t)), reverse=True), K, o.dedup_tol)


def _wig(t):
    ys = sorted(t)
    d = np.abs(np.diff([t[y] for y in ys]))
    return float(np.median(d)) if len(d) else 0.0


def xvar(t):
    """РАЗМАХ ПЕРА ПО X (p95-p5). Кривая каротажа обязана ходить; рельс, рамка и линия сетки —
    нет. Признак внутренний, GT не нужен. Диагностика Pn_Zavoda: неверные трассы шли с
    плотностью 1.00 и медианой |dx| = 0.0 на 28000 строк — это прямая, а не кривая."""
    v = np.fromiter(t.values(), float)
    return float(np.percentile(v, 95) - np.percentile(v, 5)) if len(v) else 0.0


def pick_smooth(cand, K, o):
    return dedup_take(sorted(cand, key=_wig), K, o.dedup_tol)


def pick_xspread(cand, K, o):
    """Жадно: первая — самая длинная, дальше каждый раз самая ДАЛЬНЯЯ по медиане x от взятых."""
    if not cand:
        return []
    med = {id(t): float(np.median(list(t.values()))) for t in cand}
    rest = sorted(cand, key=span, reverse=True)
    picked = [rest.pop(0)]
    while rest and len(picked) < K:
        t = max(rest, key=lambda q: min(abs(med[id(q)] - med[id(p)]) for p in picked))
        picked.append(t); rest.remove(t)
    return picked


def switch_score(t, cand, tol, nb=5):
    """★ ЛОКАЛИЗАЦИЯ СОГЛАСИЯ ПО ГЛУБИНЕ (кандидат §6.46). `split_score` спрашивает, СКОЛЬКО у
    трассы партнёров; здесь — МЕНЯЕТСЯ ли ведущий партнёр ПО ГЛУБИНЕ. Латч по построению идёт по
    одной кривой сверху и по другой снизу, значит в верхних бинах его почти-двойник один, а в
    нижних — другой. У дубликата партнёр один на всю глубину, у одинокой трассы сильного
    партнёра нет вовсе. Возвращает долю сильных бинов, где ведёт НЕ доминирующий партнёр."""
    ys = sorted(t)
    if len(ys) < 30 * nb:
        return 0.0
    edges = np.linspace(ys[0], ys[-1], nb + 1)
    lead = []
    for b in range(nb):
        rows = [y for y in ys if edges[b] <= y <= edges[b + 1]]
        if len(rows) < 30:
            continue
        bm, bq = 0.0, None
        for qi, q in enumerate(cand):
            if q is t:
                continue
            com = [y for y in rows if y in q]
            if len(com) < 30:
                continue
            ag = float(np.mean([abs(t[y] - q[y]) < tol for y in com]))
            if ag > bm:
                bm, bq = ag, qi
        if bq is not None and bm >= 0.7:
            lead.append(bq)
    if len(lead) < 2:
        return 0.0
    dom = max(set(lead), key=lead.count)
    return (len(lead) - lead.count(dom)) / len(lead)


def pick_loc(cand, K, o):
    sw = {id(t): switch_score(t, cand, o.dup_tol, o.bins) for t in cand}
    order = sorted(cand, key=lambda t: (sw[id(t)] >= o.switch_thr, -len(t)))
    return dedup_take(order, K, o.dedup_tol)


def pick_nl_loc(cand, K, o):
    """Обе подписи латча как ОДИН фильтр: разделённость ИЛИ смена партнёра по глубине."""
    sp = {id(t): split_score(t, cand, o.dup_tol) for t in cand}
    sw = {id(t): switch_score(t, cand, o.dup_tol, o.bins) for t in cand}
    order = sorted(cand, key=lambda t: (sp[id(t)] >= o.split_thr or sw[id(t)] >= o.switch_thr,
                                        -len(t)))
    return dedup_take(order, K, o.dedup_tol)


def pick_nl_and_loc(cand, K, o):
    """Строгий вариант: в конец очереди только те, у кого ОБЕ подписи (union над-фильтровал)."""
    sp = {id(t): split_score(t, cand, o.dup_tol) for t in cand}
    sw = {id(t): switch_score(t, cand, o.dup_tol, o.bins) for t in cand}
    order = sorted(cand, key=lambda t: (sp[id(t)] >= o.split_thr and sw[id(t)] >= o.switch_thr,
                                        -len(t)))
    return dedup_take(order, K, o.dedup_tol)


def pick_nl_xvar(cand, K, o):
    """Фильтр латча + порядок по РАЗМАХУ ПЕРА: кривая обязана ходить, рельс и рамка — нет."""
    sp = {id(t): split_score(t, cand, o.dup_tol) for t in cand}
    order = sorted(cand, key=lambda t: (sp[id(t)] >= o.split_thr, -xvar(t)))
    return dedup_take(order, K, o.dedup_tol)


def pick_npts_xvar(cand, K, o):
    """БЕЗ фильтра латча: только «полная и ходячая» — точки × размах."""
    order = sorted(cand, key=lambda t: -len(t) * min(xvar(t), 400.0))
    return dedup_take(order, K, o.dedup_tol)


def pick_nolatch(cand, K, o):
    sp = {id(t): split_score(t, cand, o.dup_tol) for t in cand}
    order = sorted(cand, key=lambda t: (sp[id(t)] >= o.split_thr, -span(t)))
    return dedup_take(order, K, o.dedup_tol)


def pick_nl_dens(cand, K, o):
    sp = {id(t): split_score(t, cand, o.dup_tol) for t in cand}
    order = sorted(cand, key=lambda t: (sp[id(t)] >= o.split_thr, -len(t) / max(1, span(t))))
    return dedup_take(order, K, o.dedup_tol)


def pick_nl_npts(cand, K, o):
    sp = {id(t): split_score(t, cand, o.dup_tol) for t in cand}
    order = sorted(cand, key=lambda t: (sp[id(t)] >= o.split_thr, -len(t)))
    return dedup_take(order, K, o.dedup_tol)


PICKERS = {"oracle": pick_oracle, "depth": pick_depth, "npts": pick_npts, "dens": pick_dens,
           "smooth": pick_smooth, "xspread": pick_xspread, "nolatch": pick_nolatch,
           "nl+dens": pick_nl_dens, "nl+npts": pick_nl_npts,
           "loc": pick_loc, "nl+loc": pick_nl_loc, "nl&loc": pick_nl_and_loc,
           "nl+xvar": pick_nl_xvar, "npts*xvar": pick_npts_xvar}


# ─────────────────────────────── метрика (§6.36) ───────────────────────────────
def leaked(tr, g):
    """Контроль копирования эксперта: совпала сетка строк И значения, либо >50% точных x."""
    er = np.array([g["top_y"] + i for i, x in enumerate(g["xs"]) if x != NULL])
    ex = np.array([x for x in g["xs"] if x != NULL], float)
    oy = np.array(sorted(tr)); ox = np.array([tr[y] for y in oy], float)
    if len(oy) == len(er) and np.array_equal(oy, er) and np.allclose(ox, ex):
        return True
    com = np.intersect1d(oy, er)
    if len(com) >= 30:
        om = dict(zip(oy.tolist(), ox.tolist())); em = dict(zip(er.tolist(), ex.tolist()))
        if np.mean([abs(om[y] - em[y]) < 1e-9 for y in com]) > 0.5:
            return True
    return False


def score(picked, GM, raw):
    """ЧЕСТНЫЕ среди отобранных при назначении 1-к-1. Возвращает (честных, кривых, med(med))."""
    if not GM:
        return (0, 0, float("nan"))
    pairs = []
    for nm, gm in GM.items():
        for ti, tr in enumerate(picked):
            com = [y for y in tr if y in gm]
            if len(com) < 30:
                continue
            dd = np.array([abs(tr[y] - gm[y]) for y in com])
            pairs.append((float(np.median(dd)), len(com) / len(gm), nm, ti))
    pairs.sort(key=lambda q: (q[1] < 0.9, q[0]))
    got, used = {}, set()
    for med, cov, nm, ti in pairs:
        if nm in got or ti in used:
            continue
        got[nm] = (med, cov, ti); used.add(ti)
    honest, meds = 0, []
    for nm, (med, cov, ti) in got.items():
        if leaked(picked[ti], raw[nm]):
            continue
        meds.append(med)
        if med <= 3 and cov >= 0.9:
            honest += 1
    return (honest, len(GM), float(np.median(meds)) if meds else float("nan"))


# ─────────────────────────────── сбор кэша ───────────────────────────────
def n_slots(nlgx):
    """K = сколько слотов кривых в рамке (без осей глубины DA*) — ровно как в _emit_traces."""
    ifds = read_full(open(nlgx, "rb").read())
    k = 0
    for _, nm in curve_ifds(ifds):
        head = nm.split()[0].upper() if nm.split() else ""
        if not head.startswith("DA"):
            k += 1
    return k


def build(sheets):
    from auto.pipeline import run as pipe_run
    from auto.config import Config
    if a.seq:
        # ★ ОБУЧЕННЫЙ СЕЛЕКТОР РАНОВ (§6.22-§6.24), решение на строке опирается на ПАТЧ ±64 строки.
        # Это ровно «идентичность из контекста», к которой свёлся вывод §6.64. Код инференса —
        # тот же, что в `_emit_traces.py --seq`, вынесен сюда без изменений.
        import torch
        from auto import trace2d as _T
        from auto import imaging as _im
        from _decoder_core import features
        from _decoder_seq import WindowSelector, OUT as MODELS
        from _decoder_seq_data import MAXC, patch
        DEV = "cuda" if torch.cuda.is_available() else "cpu"
        NET = WindowSelector().to(DEV)
        NET.load_state_dict(torch.load(MODELS / a.seq, map_location=DEV)["sd"]); NET.eval()

        def _seq(fg, line, frame, p, band_pad=8, slmax=30.0, wide_run=14, x_range=None,
                 jump_limit=None):
            H, W = fg.shape
            lo = max(0, int(x_range[0])) if x_range is not None else max(0, int(line.x_lo) - band_pad)
            hi = min(W, int(x_range[1]) + 1) if x_range is not None else min(W, int(line.x_hi) + band_pad + 1)
            base = line.x_center
            band = np.ascontiguousarray(fg[:, lo:hi] > 0)
            x = None; v = 0.0; tr = {}
            with torch.no_grad():
                for y in range(max(0, line.y0), min(H, line.y1 + 1)):
                    runs = _im.row_runs(fg[y, lo:hi])
                    if not runs:
                        if x is not None:
                            x = x + float(np.clip(v, -slmax, slmax))
                        continue
                    A = np.array([r[0] + lo for r in runs]); Bb = np.array([r[1] + lo for r in runs])
                    C = np.array([r[2] + lo for r in runs], float)
                    if x is None:
                        k = int(np.argmin(np.abs(C - base))); x = float(C[k]); v = 0.0; tr[y] = x; continue
                    pred = x + float(np.clip(v, -slmax, slmax))
                    idx, X = features(A, Bb, C, pred, x, v, base, MAXC)
                    if len(idx) == 1:
                        k = int(idx[0])
                    else:
                        ink, val = patch(band, lo, y, pred)
                        pt = torch.from_numpy(np.stack([ink, val]).astype(np.float32))[None].to(DEV)
                        f = np.zeros((1, MAXC, 10), np.float32); f[0, :len(idx)] = X
                        mm = np.zeros((1, MAXC), np.float32); mm[0, :len(idx)] = 1
                        sc = NET(pt, torch.from_numpy(f).to(DEV), torch.from_numpy(mm).to(DEV))
                        k = int(idx[int(sc[0].argmax().item())])
                    nx = float(C[k]); v = 0.6 * v + 0.4 * (nx - x); x = nx; tr[y] = float(nx)
            _T._extend_ends(tr, fg, lo, hi, slmax)
            return tr
        _T.trace_line = _seq
    if a.dp:
        # ★ Подменяем ТОЛЬКО выбор пути; полоса, цвет, refine и всё остальное — прод.
        from auto import trace2d as _T
        from _dp_trace import dp_trace_line

        def _dp(*ar, **kw):
            kw.pop("jump_limit", None)
            kw.setdefault("lam", a.dp_lam); kw.setdefault("skip_cost", a.dp_skip)
            kw.setdefault("wid", a.dp_wid)
            return dp_trace_line(*ar, **kw)
        _T.trace_line = _dp
    if a.jump_limit:
        # ★ Предохранитель написан в auto/trace2d.py:69 и снабжён разбором дефекта (19.07), но
        # `jump_limit` не передаёт НИКТО — ни refine, ни конфиг. Здесь он включается обёрткой,
        # чтобы померить цену его отключённости, ничего не меняя в проде.
        from auto import trace2d as _T
        _tl = _T.trace_line

        def _tl_capped(*ar, **kw):
            kw.setdefault("jump_limit", a.jump_limit)
            return _tl(*ar, **kw)
        _T.trace_line = _tl_capped
    from auto import confidence as confidence_mod
    from auto import emit as emit_mod
    _classify = confidence_mod.classify

    def classify_all_auto(sheet, *args, **kw):
        r = _classify(sheet, *args, **kw)
        for L in sheet.lines:
            L.confidence = "AUTO"        # иначе трассируется 1-2 линии и отбирать нечего
        return r
    confidence_mod.classify = classify_all_auto
    TR = {}
    _map = emit_mod._map_lines_to_slots

    def map_capture(traces, model, frame, mnemonics_path):
        TR["all"] = list(traces)
        return _map(traces, model, frame, mnemonics_path)
    emit_mod._map_lines_to_slots = map_capture

    CACHE.mkdir(parents=True, exist_ok=True)
    for j, n in enumerate(sheets, 1):
        p = CACHE / f"{n.stem[:60]}.pkl"
        if p.exists():
            print(f"[{j}/{len(sheets)}] {n.name[:50]:<52} кэш есть")
            continue
        img = find_image(n)
        if not img:
            print(f"[{j}/{len(sheets)}] {n.name[:50]:<52} нет картинки"); continue
        TR.clear()
        cfg = Config(); cfg.out = OUT / "pipe" / n.stem[:40]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
        except Exception as e:
            print(f"[{j}/{len(sheets)}] {n.name[:50]:<52} ПАДЕНИЕ {type(e).__name__}: {e}")
            continue
        traces = [{int(y): float(x) for y, x in tr.items()} for _, tr in TR.get("all", [])]
        m = extract(str(n))
        raw = {c["name"]: c for c in ds.real_curves(m) if any(x != NULL for x in c["xs"])}
        GM = {}
        for nm, g in raw.items():
            d = dense(g)
            if len(d) >= 50:
                GM[nm] = {int(y): float(d[y]) for y in sorted(d)}
        try:
            K = n_slots(n)
        except Exception:
            K = len(GM)
        pickle.dump({"traces": traces, "GM": GM, "raw": raw, "K": K, "name": n.name}, open(p, "wb"))
        print(f"[{j}/{len(sheets)}] {n.name[:50]:<52} трасс {len(traces):>3}  кривых {len(GM):>2}  K={K}")


# ─────────────────────────────── оценка по кэшу ───────────────────────────────
def evaluate(picks):
    files = sorted(CACHE.glob("*.pkl"))
    if not files:
        print("кэша нет — сначала `--build`"); return
    res = {mo: [] for mo in picks}
    print(f"листов в кэше: {len(files)}   критерии: {picks}   "
          f"dup_tol={a.dup_tol} split_thr={a.split_thr}\n")
    print(f"{'лист':<44}{'N':>3}{'K':>3}{'GT':>4}  " + "".join(f"{mo:>9}" for mo in picks))
    for f in files:
        d = pickle.load(open(f, "rb"))
        cand = [bridge(t, a.bridge) for t in d["traces"] if len(t) >= MINPTS]
        K = max(1, d["K"] or len(d["GM"]))
        row = f"{d['name'][:42]:<44}{len(d['traces']):>3}{K:>3}{len(d['GM']):>4}  "
        for mo in picks:
            r = score(PICKERS[mo](cand, K, a), d["GM"], d["raw"])
            res[mo].append(r)
            row += f"{r[0]:>5}/{r[1]:<3}"
        print(row)

    print(f"\n{'критерий':<12}{'ЧЕСТНЫХ':>9}{'кривых':>8}{'med(med)':>10}{'доля потолка':>14}")
    tot = {}
    for mo in picks:
        rr = res[mo]
        h = sum(r[0] for r in rr); k = sum(r[1] for r in rr)
        md = float(np.nanmedian([r[2] for r in rr])) if rr else float("nan")
        tot[mo] = h
        cap = tot.get("oracle")
        frac = f"{100.0*h/cap:>13.0f}%" if cap else " " * 14
        mark = ""
        if "depth" in tot and mo not in ("oracle", "depth"):
            d = h - tot["depth"]
            mark = f"   {'★ +' if d > 0 else ('✗ -' if d < 0 else '= ')}{abs(d)} к проду"
        print(f"{mo:<12}{h:>9}{k:>8}{md:>10.1f}{frac}{mark}")
    if "oracle" in tot and "depth" in tot:
        print(f"\nЦена подзадачи §6.45: потолок {tot['oracle']} − прод {tot['depth']} = "
              f"{tot['oracle'] - tot['depth']} честных кривых теряется НА ОТБОРЕ, "
              f"а не на трассировке.")
    OUT.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(
        {"picks": {mo: res[mo] for mo in picks}, "dup_tol": a.dup_tol, "split_thr": a.split_thr},
        ensure_ascii=False), encoding="utf-8")
    print(f"\n{a.json}")


def diag(sub, picker):
    """Для листа, где отбор теряет кривую: печать всех кандидатов с их ВНУТРЕННИМИ признаками
    и с ЛУЧШИМ совпадением по GT. Смысл — увидеть, каким признаком верная трасса отличается от
    выбранной, а не угадывать критерий. GT здесь ТОЛЬКО для печати, в отбор не входит."""
    fs = [f for f in sorted(CACHE.glob("*.pkl")) if sub.lower() in f.stem.lower()]
    if not fs:
        print(f"нет листа по подстроке {sub!r}"); return
    for f in fs:
        d = pickle.load(open(f, "rb"))
        cand = [t for t in d["traces"] if len(t) >= MINPTS]
        K = max(1, d["K"] or len(d["GM"]))
        picked = PICKERS[picker](cand, K, a)
        pid = {id(t) for t in picked}
        sp = {id(t): split_score(t, cand, a.dup_tol) for t in cand}
        sw = {id(t): switch_score(t, cand, a.dup_tol, a.bins) for t in cand}
        print(f"\n=== {d['name']}   N={len(cand)} K={K} GT={len(d['GM'])}   отбор `{picker}` ===")
        print(f"{'#':>3}{'взята':>7}{'точек':>8}{'охват':>8}{'плотн':>7}{'split':>7}{'switch':>8}"
              f"{'|dx|':>7}{'размах':>8}   лучшая GT (med/cov)")
        for i, t in enumerate(cand):
            best = ("-", 9e9, 0.0)
            for nm, gm in d["GM"].items():
                com = [y for y in t if y in gm]
                if len(com) < 30:
                    continue
                md = float(np.median([abs(t[y] - gm[y]) for y in com]))
                if md < best[1]:
                    best = (nm.split()[0], md, len(com) / len(gm))
            hon = "★" if best[1] <= 3 and best[2] >= 0.9 else " "
            print(f"{i:>3}{('ДА' if id(t) in pid else ''):>7}{len(t):>8}{span(t):>8}"
                  f"{len(t)/max(1,span(t)):>7.2f}{sp[id(t)]:>7.2f}{sw[id(t)]:>8.2f}{_wig(t):>7.1f}{xvar(t):>8.0f}"
                  f"   {hon}{best[0]:<8}{best[1]:>7.1f}{best[2]:>6.2f}")


def cross(dirs, picks):
    """ПЕРЕКРЁСТНЫЙ ЗАМЕР ПО НЕЗАВИСИМЫМ НАБОРАМ ЛИСТОВ (§6.48: один гейт — это популяция).
    Протокол объявлен ДО прогона: критерий выбирается на самом большом и НИ РАЗУ не
    использованном наборе, остальные — проверочные. Печатается и сумма, и число ДЕГРАДИРОВАВШИХ
    листов против прода — правило §6.38 требует монотонности, а не только суммы."""
    sets = []
    for d in dirs:
        rows = []
        for f in sorted(Path(d).glob("*.pkl")):
            dd = pickle.load(open(f, "rb"))
            cand = [bridge(t, a.bridge) for t in dd["traces"] if len(t) >= MINPTS]
            rows.append((cand, max(1, dd["K"] or len(dd["GM"])), dd["GM"], dd["raw"]))
        sets.append((Path(d).name, rows))
        print(f"набор {Path(d).name:<10} листов {len(rows):>3}  кривых "
              f"{sum(len(r[2]) for r in rows):>4}")
    res = {mo: [[score(PICKERS[mo](c, K, a), GM, raw) for c, K, GM, raw in rows]
                for _, rows in sets] for mo in picks}
    print("\n" + f"{'критерий':<12}" + "".join(f"{nm[:9]:>11}" for nm, _ in sets) +
          f"{'ВСЕГО':>9}{'дегр.':>7}")
    for mo in picks:
        line = f"{mo:<12}"
        for si in range(len(sets)):
            h = sum(r[0] for r in res[mo][si]); k = sum(r[1] for r in res[mo][si])
            line += f"{h:>6}/{k:<4}"
        tot = sum(r[0] for st in res[mo] for r in st)
        deg = sum(1 for si in range(len(sets))
                  for x, y in zip(res["depth"][si], res[mo][si]) if y[0] < x[0])
        line += f"{tot:>9}{deg:>7}"
        print(line + ("   ← ПРОД" if mo == "depth" else ""))
    cap = sum(r[0] for st in res.get("oracle", []) for r in st)
    if cap:
        pd = sum(r[0] for st in res["depth"] for r in st)
        print(f"\nпотолок {cap}; прод берёт {100 * pd / cap:.0f}% от него")
    Path(a.json).write_text(json.dumps(
        {"sets": [nm for nm, _ in sets], "picks": {mo: res[mo] for mo in picks}},
        ensure_ascii=False), encoding="utf-8")
    print("\n" + str(a.json))


if __name__ == "__main__":
    picks = [p for p in a.picks.split(",") if p in PICKERS]
    if a.diag:
        diag(a.diag, a.diag_pick); sys.exit(0)
    if a.cv:
        cross([d for d in a.cv.split(",") if d], picks); sys.exit(0)
    if a.build:
        from _decoder_data import train_sheets
        from _relatch_bench import SH, ARCH
        if a.dark_from:
            # ★ ПЕРЕСБОРКА ТЁМНЫХ ЛИСТОВ. Берём из готовых кэшей те листы, у которых бумага
            # тёмная, и гоняем их заново — так замер правки §6.52 идёт на ТЕХ ЖЕ листах,
            # что и замер до неё, без подмешивания новых.
            import cv2, pickle as _pk
            idx = {}
            for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
                for q in wlg.glob("*.nlgx"):
                    idx.setdefault(q.name, q)
            for extra in (r"F:\nds\projects\Semeguniv_001\wlg",
                          r"F:\nds\projects\Semeguniv_020\wlg"):
                if Path(extra).is_dir():
                    for q in Path(extra).glob("*.nlgx"):
                        idx.setdefault(q.name, q)
            sheets = []
            for d in a.dark_from.split(","):
                for fp in sorted(Path(d).glob("*.pkl")):
                    src = idx.get(_pk.load(open(fp, "rb"))["name"])
                    if src is None or src in sheets:
                        continue
                    img = find_image(src)
                    if not img:
                        continue
                    imz = cv2.imread(str(img), cv2.IMREAD_COLOR)
                    if imz is None:
                        continue
                    vv = cv2.cvtColor(imz[::8, ::8], cv2.COLOR_BGR2HSV)[:, :, 2]
                    if a.p90_band:
                        lo, hi = (int(z) for z in a.p90_band.split(","))
                        if lo <= int(np.percentile(vv, 90)) < hi:
                            sheets.append(src)
                    elif float(np.median(vv)) < a.dark_v:
                        sheets.append(src)
        elif a.k2:
            # ★ ТРАНШ K=2 ПО СОДЕРЖИМОМУ РАМКИ (отбор как в `_k2_gate.py`, §6.7), а НЕ по
            # порядку train_sheets. Нужен, чтобы проверить утверждение §6.50: не легче ли
            # двухкривые листы, чем выборка с приоритетом кроссинг-классов. По одному листу
            # со скважины, размер скана ограничен снизу и сверху — иначе проба перестаёт быть
            # широкой (durable `_k2_gate`).
            from auto import meta as M
            from PIL import Image
            ARCHIVE = Path(r"F:\nds\projects\Archive")
            cands, per_well = [], {}
            for wlg in sorted(ARCHIVE.glob("*/wlg")):
                well = wlg.parent.name
                for n in sorted(wlg.glob("*.nlgx")):
                    if "_auto" in n.stem or per_well.get(well, 0) >= 1:
                        continue
                    img = find_image(n)
                    if not img:
                        continue
                    try:
                        mo = extract(str(n))
                    except Exception:
                        continue
                    gts = [c for c in mo["curves"]
                           if sum(1 for x in c["xs"] if x != NULL) >= 100
                           and M.mnem_root(c["name"]) != "DA"]
                    if len(gts) != 2:
                        continue
                    w, h = Image.open(img).size
                    if not (8e6 <= w * h <= 45e6):
                        continue
                    cands.append((w * h, n))
                    per_well[well] = per_well.get(well, 0) + 1
            cands.sort()
            sheets = [n for _, n in cands[:a.k2]]
        elif a.skip:
            # ★ ВАЛИДАЦИЯ НА СВЕЖИХ ЛИСТАХ: ни контрольного, ни SH, ни первых --skip листов
            # гейта. Критерий на них НЕ подбирался — только на них и видно, не подгонка ли это.
            gate = set(train_sheets(a.skip))
            sheets = [s for s in train_sheets(a.skip + a.n) if s not in gate][:a.n]
        else:
            sheets = [Path(r"F:\nds\projects\Semeguniv_001\wlg\Semeguniv_1_BKZ_3400_3640_200_D1.nlgx")]
            sheets += [ARCH / s for s in SH]
            sheets += [s for s in train_sheets(a.n) if s not in sheets]
        print(f"сбор кэша: {len(sheets)} листов"
              f"{' — ВАЛИДАЦИОННЫХ, вне гейта' if a.skip else ' (набор _esc_limit/_esc_gate)'}\n")
        build(sheets)
        print()
    evaluate(picks)

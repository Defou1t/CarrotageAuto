r"""
trace2d.py — 2D-обход штриха для AUTO-бакета (замена per-row x(y), durable §6.6.8/§6.6.9).

ПОЧЕМУ не per-row: x(y) однозначен на ВЕРТИКАЛЬНЫХ участках, но на горизонтальном спайке
(резкий пик резистива / уход микрозонда) один ряд = длинный прогон → центроид «срезает угол,
садится на базлайн». Решение — вести по СВЯЗНОСТИ РАНОВ (run-adjacency граф, PLAN §6.6.11 ст.0):
ряд→ряд переходим в перекрывающийся по x ран (стрих физически непрерывен), а X на строке берём
как ВЕРШИНУ выноса (дальний край рана от базлайна), а не его центр — так пик доводится до конца.

Только для линий, помеченных confidence='AUTO' (одиночные/разнесённые/цвето-уникальные). Сбитый
пучок (FLAG) сюда НЕ идёт. Чистый numpy/cv2, без scipy/skimage.
"""
import numpy as np
from . import imaging as im


def _color_fg(rgb, color, p):
    """Бинарь переднего плана одного цвета (для трассировки конкретной линии)."""
    if color == "black":
        m = im.dark_mask(rgb, p)
        # ★★ §6.152: канал, покрывающий десятки процентов листа, — БУМАГА, а не тушь, и вычитать
        # его из чёрного значит стирать чёрный штрих. Та же проверка и тот же порог, что в
        # `imaging.ink_foreground` (`imaging.py:186`). ⚠ Умолчание `False` = прежнее поведение.
        degen = bool(getattr(p, "color_fg_degen", False))
        for cm in im.color_channels(rgb, p).values():
            if degen and float(cm.mean()) > p.color_max_frac:
                continue
            m = m & ~cm
        return m & ~im.structure_mask(rgb, p)
    cm = im.color_channels(rgb, p).get(color)
    if cm is None:
        return np.zeros(rgb.shape[:2], bool)
    return cm & ~im.structure_mask(rgb, p)


def trace_line(fg, line, frame, p, band_pad=8, slmax=30.0, wide_run=None, x_range=None,
               jump_limit=None):
    """Трасса одной AUTO-линии как x(row). fg — bool-маска цвета линии.
    band_pad — допуск вокруг x-полосы линии (анти-перескок на соседа). slmax — кламп скорости.
    wide_run — ран шире этого = горизонтальный спайк → берём ВЕРШИНУ (дальний край), не центр.
      None = взять из конфига (`p.trace_wide_run`, §6.102: свип показал, что правило вершины
      подкармливает дрейф и порог выгоднее поднять). Явно переданное значение сильнее конфига —
      на этом держится `refine`, который ставит 10**6 для свипующего пера (`prefer_body`).
    x_range=(lo,hi) — ЯВНЫЙ band (для refine: расширение до трека, чтобы догнать выносы к упору;
    у цвето-уникальной AUTO соседа того же цвета нет → расширение безопасно, связность держит нить)."""
    if wide_run is None:
        wide_run = getattr(p, "trace_wide_run", 14)
    H, W = fg.shape
    if x_range is not None:
        lo = max(0, int(x_range[0])); hi = min(W, int(x_range[1]) + 1)
    else:
        lo = max(0, int(line.x_lo) - band_pad); hi = min(W, int(line.x_hi) + band_pad + 1)
    base = line.x_center            # ориентир базлайна (вершину пика тянем ОТ него)
    x = None; v = 0.0
    tr = {}
    for y in range(max(0, line.y0), min(H, line.y1 + 1)):
        runs = im.row_runs(fg[y, lo:hi])
        runs = [(a + lo, b + lo, c + lo) for a, b, c in runs]
        if not runs:
            if x is not None:                          # коаст по инерции через короткий разрыв
                x = x + float(np.clip(v, -slmax, slmax))
            continue
        if x is None:
            a, b, c = min(runs, key=lambda r: abs(r[2] - base))
            x = c; v = 0.0; tr[y] = x; continue
        pred = x + float(np.clip(v, -slmax, slmax))
        # выбрать РАН, перекрывающий предсказание (связность штриха), иначе ближайший по центру
        cont = [r for r in runs if r[0] - 2 <= pred <= r[1] + 2]
        if cont:
            a, b, c = min(cont, key=lambda r: abs(r[2] - pred))
        else:
            a, b, c = min(runs, key=lambda r: abs(r[2] - pred))
            # ПРЫЖОК НА СОСЕДА (19.07). Ветка «нет рана под предсказанием» НИЧЕМ не ограничена по
            # расстоянию: если чернил своей кривой на строке нет (пересечение, разрыв, бледное
            # место), трасса садится на ближайший ран хоть за сотни px и ТАМ ОСТАЁТСЯ — отсюда
            # «странные скачки» в QC. Замер: трасса на своей кривой лишь 32% строк, латч 32%.
            # В соседней _extend_ends такой предохранитель ЕСТЬ, в основном цикле его не было.
            # jump_limit=None — прежнее поведение (неограниченный прыжок).
            # ⚠ РАНЬШЕ ЗДЕСЬ БЫЛО `x = pred; continue` — КОАСТ ПО ИНЕРЦИИ, и это делало
            # предохранитель НЕПРИГОДНЫМ (§6.57): строка не пишется, x уезжает по v, на следующей
            # строке предсказание ещё дальше от чернил, предохранитель срабатывает снова —
            # восстановления нет, ОДНО срабатывание убивало трассу до конца листа. Замер: медиана
            # длины трассы 26781 → 728 точек, покрытие 0.86 → 0.03, честных 19 → 1 на гейте.
            # ★ ПОДХВАТ: держим x НА МЕСТЕ и гасим скорость. Тогда `pred` следующей строки
            # остаётся у последнего достоверного положения, и как только своя кривая снова даст
            # ран рядом, трасса подхватится сама. Строка при этом честно не пишется — «здесь не
            # знаю» вместо выдуманного значения; короткие пропуски закрывает мостик §6.53.
            if jump_limit is not None and abs(c - pred) > jump_limit + (b - a):
                v *= 0.5
                continue
        if (b - a) >= wide_run:                        # горизонтальный спайк → вершина выноса
            nx = b if abs(b - base) >= abs(a - base) else a
        else:
            nx = c
        v = 0.6 * v + 0.4 * (nx - x); x = nx; tr[y] = float(nx)
    _extend_ends(tr, fg, lo, hi, slmax)
    return tr


def _extend_ends(tr, fg, lo, hi, slmax, max_gap=25):
    """Доводка трассы за [y0,y1] по СВЯЗНОМУ чернилу цвета (G4/BK_4020: верхний заход кривой
    4017-4020 терялся при группировке — линия стартовала на 4020). Идём от края вверх/вниз,
    садимся на ближайший к предсказанию ран, стоп после max_gap пустых строк подряд."""
    H = fg.shape[0]
    if not tr:
        return
    for direction in (-1, +1):
        y0 = min(tr) if direction < 0 else max(tr)
        x = tr[y0]; gap = 0; y = y0 + direction
        while 0 <= y < H and gap <= max_gap:
            runs = im.row_runs(fg[y, lo:hi])
            if not runs:
                gap += 1; y += direction; continue
            runs = [(a + lo, b + lo, c + lo) for a, b, c in runs]
            a, b, c = min(runs, key=lambda r: abs(r[2] - x))
            if abs(c - x) > slmax + (b - a):        # разрыв идентичности — не тянем на соседа
                break
            x = c; tr[y] = float(c); gap = 0; y += direction


_SEQ = {}                    # чекпойнт → готовый трассировщик (веса грузятся один раз)
_SAID = set()                # что уже сказано в этом процессе (сообщение на лист — это шум)


def _announce(msg):
    """Сказать РОВНО ОДИН РАЗ на процесс. Режим трассировки одинаков для всех листов, поэтому
    повтор на каждый лист — шум, а молчание — хуже: см. предупреждение ниже."""
    if msg not in _SAID:
        _SAID.add(msg)
        print(f"  {msg}")


def _pregate_ok(sheet, p):
    """→ гнать ли ВТОРОЙ ПУТЬ на этом листе (§6.161, §6.178). `cv.rowdec_pregate` = 0.0 — гейта нет.

    ⚠⚠ ПРИЗНАК СЧИТАЕТСЯ ПО ВСЕМ ЛИНИЯМ ЛИСТА, БЕЗ ФИЛЬТРОВ, и это не придирка. Ровно так он
    считался в замере: `_pregate.py` берёт `lines` из `<лист>_understanding.json`, а тот —
    `sheet.to_dict()` (`emit.py:31`). Отфильтруй здесь по `confidence == "AUTO"` или по цвету — и
    прод будет решать по ДРУГОЙ величине, чем измерено. Это класс §6.144: там выбор считался по
    «сколько линий декодер ПОВЁЛ», а мерился по «сколько кривых НАПИСАЛ», и цена подмены признака
    вышла 14 кривых из ожидавшихся 20.
    ⚠ Гейт стоит ДО `rowdec.trace_auto` намеренно: вся экономия в том, что на отсеянном листе
    декодер НЕ СЧИТАЕТСЯ (§6.160 — порогом эта цена не регулируется). Гейт после вызова дал бы те
    же кривые и НОЛЬ экономии.
    """
    thr = float(getattr(p, "rowdec_pregate", 0.0) or 0.0)
    if thr <= 0.0:
        return True
    L = getattr(sheet, "lines", None) or []
    if not L:
        return True                    # не по чему решать — ведём себя как без гейта
    from collections import Counter
    c = Counter(str(getattr(x, "color", None)) for x in L)
    return max(c.values()) / len(L) >= thr


def trace_auto(rgb, sheet, p=None):
    """Трассировать все AUTO-линии листа. Возвращает list[(Line, {row:x})].
    Каждая трасса ДЕСПАЙКается (refine.despike): изолированные выбросы-спайки (перескок на рамку/
    сосед на 1-2 строки) заменяются локальной медианой; устойчивый пик кривой сохраняется."""
    from .config import DEFAULT
    from . import refine
    p = p or DEFAULT.cv
    # ★ ПОСТРОЧНЫЙ ДЕКОДЕР (Задача 9, §6.140): заменяет ведение целиком, а не выбор рана.
    # Выключен, пока не задан `row_decoder`; при недоступности говорит вслух и откатывается сюда.
    alt, pick, want_both = None, 0, False
    # ★ §6.213: выбор по слоту (`rowdec_slot_len`) может просить декодер и ЗА предгейтом
    #   (`rowdec_slot_all`); гейт при этом остаётся условием выбора ПО ТРЕКУ — `emit` узнаёт его
    #   через `alt_gated`. При выключенной ручке условие тождественно прежнему.
    slot_len = float(getattr(p, "rowdec_slot_len", 0.0) or 0.0)
    gated = _pregate_ok(sheet, p)
    if getattr(p, "row_decoder", "") and (gated or (slot_len > 0 and getattr(p, "rowdec_slot_all", False))):
        from . import rowdec
        alt = rowdec.trace_auto(rgb, sheet, p)
        pick = int(getattr(p, "rowdec_pick", 0) or 0)
        # ★★ §6.153: ОБУЧЕННЫЙ выбор (`rowdec_pick_model`) тоже требует ОБОИХ путей, иначе он
        # НЕДОСТИЖИМ. `config.py:294` обещает «задан — перебивает `rowdec_pick`», но при
        # `rowdec_pick = 0` ведение возвращало декодер ЦЕЛИКОМ, `emit` не видел `traces.alt`,
        # выбор не считался ни разу и `<лист>_pick.json` не писался.
        # ⚠ Как поймано: прогон 221 листа с `rdmodel=` дал 0 файлов `_pick.json` против 1128 у
        # порогового прогона, а числа читались как «обученный выбор хуже порога на 14 кривых» —
        # хотя мерилось «ВСЕГДА ДЕКОДЕР». Ровно класс «умолчание кода ≠ то, что решено замером».
        # ⚠⚠ ФЛАГ ОДИН НА ОБА МЕСТА. Ворот два — ранний возврат здесь и прицеп `.alt` ниже, — и
        # первая правка починила ТОЛЬКО первый: `alt` переставал возвращаться целиком, но и не
        # прицеплялся, так что `emit` по-прежнему не видел второго пути. Держать условие в
        # переменной, а не повторять его дважды.
        want_both = pick > 0 or bool(getattr(p, "rowdec_pick_model", "") or "") or slot_len > 0
        if alt is not None and not want_both:
            return alt
        if alt is None:
            # декодер не включился — выбирать не из чего, идём прод-путём
            pick, want_both = 0, False
    out = []
    fg_cache = {}
    # ★ ОБУЧЕННЫЙ СЕЛЕКТОР (§6.66): подменяет ТОЛЬКО выбор рана, всё остальное (полоса, цвет,
    # refine с границей Вороного, деспайк) остаётся прежним. Выключен, пока не задан seq_model.
    tl = trace_line
    if getattr(p, "seq_model", ""):
        from . import trace_seq
        if trace_seq.available(p.seq_model):
            tl = _SEQ.get(p.seq_model) or _SEQ.setdefault(
                p.seq_model, trace_seq.make_tracer(p.seq_model))
            _announce(f"трассировка: ОКОННЫЙ СЕЛЕКТОР {trace_seq.resolve(p.seq_model).name}")
        else:
            # ⚠ ГОВОРИМ ГРОМКО, А НЕ МОЛЧА. Селектор ЗАПРОСИЛИ явно, но включить нечем — нет torch
            # либо нет файла. Тихий откат на жадный выбор недопустим: выдача заметно хуже (§6.66),
            # а внешне НЕОТЛИЧИМА — ровно так «те же цифры» и расходятся между машинами. Сам откат
            # при этом сохранён намеренно (§6.68: torch не становится жёсткой зависимостью).
            try:
                import torch                                    # noqa: F401
                why = "нет файла чекпойнта"
            except Exception:
                why = "torch не установлен"
            _announce(f"⚠ seq_model={p.seq_model!r} ЗАПРОШЕН, НО НЕ ВКЛЮЧЁН ({why}: "
                      f"{trace_seq.resolve(p.seq_model)}) — идёт ЖАДНЫЙ выбор, выдача хуже")
    # ★ §6.236 ИСКЛЮЧЕНИЕ ЗАНЯТЫХ РАНОВ (`cv.trace_exclusive`, умолчание False = прод бит-в-бит). §6.228: в 476 парах две
    #   трассы трека идут по ОДНОЙ линии и занимают место невзятой кривой, а версия второго пути лежит там же. Здесь линии
    #   ведутся по убыванию `row_cov` (сильная линия занимает первой), и трасса не может взять ран, который уже занят
    #   трассой того же трека и цвета: такой ран стирается из маски этой линии. На пересечении это разрыв — селектор идёт
    #   по инерции (коаст). Порядок ВЫДАЧИ прежний.
    excl = bool(getattr(p, "trace_exclusive", False))
    order = list(range(len(sheet.lines)))
    if excl:
        order.sort(key=lambda i: -float(getattr(sheet.lines[i], "row_cov", 0.0) or 0.0))
    occ, got = {}, {}
    for i in order:
        L = sheet.lines[i]
        # p.trace_flagged=True — вести и FLAG-линии тоже (§6.54-§6.55, смена контракта выдачи).
        if L.confidence != "AUTO" and not getattr(p, "trace_flagged", False):
            continue
        if L.color not in fg_cache:
            fg_cache[L.color] = _color_fg(rgb, L.color, p)
        track = sheet.frame.tracks[L.track_index]
        fg = fg_cache[L.color]
        prev = occ.get((L.track_index, L.color)) if excl else None
        if prev:
            fg = _exclude_runs(fg, prev)
        # refine-петля: тесная→широкая трасса при недотяге до упора + деспайк (refine.refine_trace)
        # siblings — все линии листа: refine ограничивает расширение полосы границами Вороного
        # до соседей ТОГО ЖЕ ЦВЕТА на том же треке (см. refine._neighbor_bounds).
        tr = refine.refine_trace(fg, L, sheet.frame, p, tl, track,
                                 siblings=sheet.lines)
        if len(tr) >= 30:
            got[i] = tr
            if excl:
                occ.setdefault((L.track_index, L.color), []).append(tr)
    out = [(sheet.lines[i], got[i]) for i in sorted(got)]
    if want_both:
        # ⚠⚠ РЕШАЕТ НЕ ЗДЕСЬ, А `emit`. Первая редакция выбирала путь прямо тут, по числу линий,
        # которые декодер ПОВЁЛ, — и прогон на 137 листах намерил, что это не тот признак: выигрыш
        # считался по числу кривых, которые он НАПИСАЛ, а это величина ПОСЛЕ раскладки. Выбор
        # разошёлся на 70 треках из 153 и стоил 14 кривых из 20 (см. §6.146). Поэтому оба пути
        # уезжают дальше, а решение принимается там, где известно назначение слотов.
        out = _WithAlt(out); out.alt = alt
        out.alt_conf = getattr(alt, "conf", None)     # §6.240: уверенность трасс декодера для вето в `emit`
        out.alt_gated = gated          # §6.213: выбор по треку — только за предгейтом
    return out


def _exclude_runs(fg, traces):
    """§6.236: копия маски без ранов, занятых трассами `traces` (ран строки, накрывающий точку трассы ±1 px)."""
    fg = fg.copy()
    H, W = fg.shape
    for tr in traces:
        for y, x in tr.items():
            if not (0 <= y < H):
                continue
            row = fg[y]
            xi = int(round(x))
            c = next((c for c in (xi, xi - 1, xi + 1) if 0 <= c < W and row[c]), None)
            if c is None:
                continue
            l = r = c
            while l > 0 and row[l - 1]:
                l -= 1
            while r < W - 1 and row[r + 1]:
                r += 1
            row[l:r + 1] = False
    return fg


class _WithAlt(list):
    """Список пар (Line, трасса) с прицепленным ВТОРЫМ путём (`.alt`).
    ⚠ Подкласс списка, а не кортеж: `traces` уходит в десяток мест (`emit_understanding`,
    `_overlay`, стенды), и все они ждут именно список. Прицеп невидим для тех, кто про него
    не знает, и это ровно то, что нужно — включённым он оказывается только в `emit`."""
    alt = None
    alt_gated = True


def _pick_by_track(prod, dec, pick):
    """⛔ ОТОЗВАНА §6.146. Оставлена для воспроизводимости прогона `ab_rdpick`.
    Решала по числу линий, которые декодер ПОВЁЛ; правильный признак — сколько он НАПИСАЛ, то есть
    после раскладки. Разница стоила 14 кривых из 20 на 137 листах.

    §6.144: ПО КАЖДОМУ ТРЕКУ взять декодер, если он повёл не больше `pick` линий, иначе прод.

    ⚠⚠ ЗАЧЕМ ЭТО ВООБЩЕ ЕСТЬ. §6.143 намерил, что на отгрузке декодер и прод дают одно и то же
    (976 против 978 честных без имени), и это верно как СУММА. Но §6.144 разобрал по кривым:
    **расходятся 779 из 2760**, 377 берёт только прод, 402 только декодер — пути ДОПОЛНЯЮЩИЕ,
    и суммы совпали случайно. Потолок выбора между ними — 1364 кривые (49.4%) против 962/987.
    ★ Признак, по которому выбирать, наблюдаем на месте: СКОЛЬКО ЛИНИЙ ДЕКОДЕР ПОВЁЛ НА ТРЕКЕ.
    Среди расходящихся кривых он прав в 85% при одной линии и в 28% при шести и более —
    на одиночных кривых он почти всегда лучше, на плотных треках почти всегда хуже.
    ⇒ Порог выбран перекрёстной проверкой ПО СКВАЖИНАМ (4/5 подбор, зачёт на держанной пятой):
    **+88 кривых (+9.0%)** к лучшему из двух путей; в лоб 976 → 1085, SD 23.0, p < 0.00001.
    Оптимум ВНУТРЕННИЙ (при pick=4 уже 1027) ⇒ это действительно выбор, а не «кто-то лучше».

    ⚠ ВЫБОР СТРОГО ПОТРЕКОВЫЙ: смешивать трассы двух путей внутри одного трека нельзя. Раскладка
    (`emit._map_lines_to_slots`) назначает слоты 1:1 внутри трека и берёт линию по `id(L)`; две
    трассы одной линии из разных путей дали бы ей два кандидата и сломали бы счёт — ровно то, на
    чём отпал пул кандидатов (§6.141, −189 кривых)."""
    from collections import defaultdict
    bd, bp = defaultdict(list), defaultdict(list)
    for L, tr in dec:
        bd[L.track_index].append((L, tr))
    for L, tr in prod:
        bp[L.track_index].append((L, tr))
    out, took = [], {"декодер": 0, "прод": 0}
    for ti in sorted(set(bd) | set(bp)):
        d, q = bd.get(ti, []), bp.get(ti, [])
        use_d = bool(d) and len(d) <= pick
        out += d if use_d else q
        took["декодер" if use_d else "прод"] += 1
    # ⚠ Печатать ОБЯЗАТЕЛЬНО: без этого «режим включился» и «режим молча не сработал» выглядят
    # снаружи одинаково (§6.68). Здесь это особенно легко: оба пути выдают правдоподобный файл.
    _announce(f"выбор пути по треку (порог {pick}): декодер на {took['декодер']} треках, "
              f"прод на {took['прод']}")
    return out

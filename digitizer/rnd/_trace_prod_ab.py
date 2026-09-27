r"""_trace_prod_ab.py — A/B ПАРАМЕТРОВ ТРАССИРОВЩИКА НА ОТГРУЖАЕМОМ ПУТИ (§6.101 шаг 1, часть 2).

⚠⚠ ЗАЧЕМ ОТДЕЛЬНЫЙ СТЕНД. `_param_sweep.py` меряет ГОЛУЮ `trace_line` на восстановленной линии.
В файл же уходит кривая после `refine_trace` (эскалация полосы до трека, деспайк, границы Вороного),
раскладки по слотам и пост-обработки уровней. Durable-память ветки: **стенд и отгрузка расходятся
в разы, цитировать надо отгрузку** (§6.69, §6.75, §6.48 — правка, принятая по стенду, была откачена).

КАК. Пайплайн гоняется по разу на каждый режим на ОДНИХ И ТЕХ ЖЕ листах. Режим = набор умолчаний
`trace_line`, наложенный обёрткой:
  `kw.setdefault(...)` — то есть ЯВНО переданные вызывающим значения не трогаются. Это важно:
  `refine_trace` сам передаёт `x_range` при эскалации полосы и `wide_run=10**6` для `prefer_body`
  (свипующее перо MBK), и оба этих решения обязаны выжить.
Честность считается в выданном `<stem>_auto.nlgx`, а не в пуле; контроль утечки эксперта — как
в `_pool_oracle.py`.

★ ОТПЕЧАТОК ВЫДАЧИ обязателен: равные счётчики честности сами по себе НЕ доказывают, что режим
включился (§6.71). Если файлы совпали бит-в-бит — замер недействителен, и стенд это говорит вслух.

⚠⚠ ПУТЬ ТРАССИРОВКИ ЗАДАЁТСЯ РЕЖИМОМ (`seq=`), И ЭТО НЕ ФОРМАЛЬНОСТЬ. По умолчанию
`CVParams.seq_model = "seq_model_d45p.pt"`, и при доступном torch линии ведёт ОБУЧЕННЫЙ СЕЛЕКТОР
`trace_seq`, у которого правила вершины выноса НЕТ ВООБЩЕ (`nx = C[k]`, всегда центр). Первый заход
этого стенда наследовал умолчание и намерил «правка ничего не меняет» — файлы вышли бит-в-бит
одинаковыми на 19 листах, потому что патч `trace2d.trace_line` не вызывался ни разу.

⚠⚠ РАСКЛАДКА (`slot=`) ПИННИТСЯ ТАК ЖЕ, И ЭТО НЕ ИЗБЫТОЧНО. `_slot_prod_ab.py` (§6.90, число
42 → 50) задавал `cfg.cv.slot_model`, но `seq_model` НЕ пиннил — значит наследовал умолчание
`seq_model_d45p.pt` и под ComfyUI-питоном мерил СЕЛЕКТОРНЫЙ путь, нигде этого не объявив. Хуже
того, он пропускал листы, на которых модель отказывается, определяя отказ ОФЛАЙН ПО ПУЛАМ, — а пулы
собраны ЖАДНЫМ прогоном (§6.104, побитово 4/4 против 0/4). Условие из его же шапки («верно ровно
пока пулы собраны ТЕМ ЖЕ кодом трассировки») при этом нарушено. Здесь оба режима гоняются на ВСЕХ
листах выборки, без офлайн-отсева.

  <ComfyUI>\python_embeded\python.exe _trace_prod_ab.py \
      --mode "A жадный:seq=,slot=" --mode "B центр:seq=,slot=,wide_run=1000000" \
      --mode "C селектор:seq=seq_model_d45p.pt,slot=" \
      --mode "D сел+раскладка:seq=seq_model_d45p.pt,slot=slot_model_g250.npz" \
      --mode "E сел+медиана:seq=seq_model_d45p.pt,slot=,order=med_x" [--cap 3] [--shard 0/12]

⚠ Имена режимов БЕЗ ПРОБЕЛОВ, если стенд запускается через `Start-Process -ArgumentList`: массив
склеивается пробелом БЕЗ квотирования, и «A сел.база:…» доезжает до argparse двумя аргументами.

⚠⚠ ЕСЛИ ВСЕ РЕЖИМЫ СЕЛЕКТОРНЫЕ — СЧИТАТЬ НА CPU, А НЕ НА ВИДЕОКАРТЕ (§6.106). Замер 07.08: 12 шардов
селектора на занятой видеокарте (VRAM 15.9/16.3 ГБ отданы LM Studio, игре и браузерам) дали
2 лист-режима за 28 минут — счёт практически встал, потому что процессы делили остаток VRAM.
Те же 12 шардов на CPU (32 ядра, `OMP_NUM_THREADS=3`) дают ≈7 лист-режимов в минуту, то есть
118 листов × 4 режима ≈ час. Одиночный лист при этом: GPU 31 с, CPU 50 с — видеокарта быстрее
ПРОЦЕССОМ, но она одна, а ядер 32.
★ КОНТРОЛЬ СОПОСТАВИМОСТИ ПРОЙДЕН: выдача CPU и GPU на одном листе (5 кривых) совпала БИТ-В-БИТ,
то есть числа CPU-прогона сравнимы с §6.102, посчитанным на видеокарте.
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 <ComfyUI>\python_embeded\python.exe _trace_prod_ab.py …
⚠ `CUDA_VISIBLE_DEVICES=""` на torch 2.10+cu130 оставляет `is_available()=True` при
`device_count()=0`; до правки §6.106 в `auto/trace_seq.py` это роняло ЗАГРУЗКУ чекпойнта.
"""
import sys, io, argparse, contextlib, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from _multi_replica_probe import dense
from auto import meta as M, trace2d as T, imaging as im
from auto.pipeline import run as pipe_run
from auto.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div"])
ap.add_argument("--mode", action="append", required=True,
                help='"имя:seq=<чекпойнт|пусто>,slot=<вес|пусто>,k=v,…"; k — параметры trace_line')
ap.add_argument("--slot-gate", default="frac0.8", help="мера уверенности обученной раскладки")
ap.add_argument("--flush", type=int, default=20,
                help="через сколько листов сбрасывать ПРОМЕЖУТОЧНЫЙ дамп (0 = только в конце)")
ap.add_argument("--cap", type=int, default=3, help="листов на скважину (0 = все)")
ap.add_argument("--shard", default="0/1")
# ★ §6.118: стоимость листа определяет ОБЪЁМ ДАМПА ПУЛА (r=0.84 со временем листа против 0.34 у
# площади в пикселях и 0.28 у размера растра), и он известен ДО запуска — дампы лежат на диске.
ap.add_argument("--shard-by-pool", action="store_true",
                help="раскладка по шардам LPT по размеру дампа пула, а не блоком по алфавиту")
ap.add_argument("--out", default=r"F:\nds\output\taskS\prod_ab_trace")
# ⚠ §6.107: отбор по СПИСКУ ИМЁН — для правок, эффект которых ДОКАЗУЕМО локализован (правка разбора
# глубины меняет разбор ровно 36 листов из 702, на остальных вход пайплайна побитово тот же).
# Мерить такую правку на всей выборке — размывать её собственный сигнал. Файл: имя nlgx на строку.
ap.add_argument("--only-from", default="", help="файл со списком имён nlgx (по одному на строку)")
# ★ 18.09: ПАМЯТЬ ШАРДА РАСТЁТ С ЧИСЛОМ ЛИСТОВ (шард 0 прогона §6.213: 17.6 ГБ после 72 листов; в
#   §6.209 так пропали 4 листа по MemoryError). Причина не найдена (4 листа подряд в одном процессе —
#   RSS ровный), поэтому лечение механическое: выше порога — сохранить промежуточный дамп и выйти
#   кодом 75; драйвер перезапускает шард, и тот продолжает с дампа. 0 = выкл.
ap.add_argument("--max-rss-gb", type=float, default=0.0, help="порог RSS, ГБ: выше — дамп и выход кодом 75 (перезапуск)")
# ★ 19.09 (правило заказчика): партия не дольше N часов — после листа, на котором время вышло, дамп и выход кодом 75.
ap.add_argument("--max-hours", type=float, default=0.0, help="партия: дамп и выход кодом 75, если шард работает дольше часов")
ap.add_argument("--wlg-roots", nargs="+", default=[r"F:\nds\projects\Archive"],
                help=r"корни, где лежат <скважина>/wlg/*.nlgx (держанный набор — intake\sorted)")
ap.add_argument("--overlay", action="store_true", help="писать overlay.png на каждый лист (по умолчанию выкл., §6.222)")
a = ap.parse_args()
# ★ 26.09: оверлеи A/B никто не читает, а весили 73 ГБ — по умолчанию не пишем (выдачу .nlgx это не меняет)
import os as _os
if not a.overlay:
    _os.environ["CARROTAGE_NO_OVERLAY"] = "1"
SH_I, SH_N = (int(v) for v in a.shard.split("/"))
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def parse_mode(s):
    """`имя:seq=<чекпойнт|пусто>,slot=<вес|пусто>,wide_run=…` → (имя, seq_model, slot_model, kwargs).
    ⚠⚠ `seq` И `slot` ОБЯЗАТЕЛЬНЫ И ПИННЯТСЯ ЯВНО. В `CVParams` по умолчанию стоит
    `seq_model_d45p.pt`, и при доступном torch `trace_auto` ведёт линии ОБУЧЕННЫМ СЕЛЕКТОРОМ
    `trace_seq`, а не `trace2d.trace_line`, — тогда любая правка констант `trace_line` до кода
    НЕ ДОХОДИТ (проверено: выдача выходит бит-в-бит одинаковой на 19 листах). Наследовать режим из
    умолчаний нельзя: он зависит от интерпретатора (torch есть только в embedded-python ComfyUI),
    то есть один и тот же стенд давал бы РАЗНЫЕ пути на разных машинах — ровно запрет §6.71.
    ⚠ `slot` от пути трассировки НЕ зависит (скор считает numpy) и действует на обоих."""
    nm, _, tail = s.partition(":")
    # ⚠⚠ УМОЛЧАНИЕ `depth0` = ПРОД (True). Первая редакция §6.108 перевернула смысл ручки и
    # оставила умолчание False — и режимы БЕЗ ключа `depth0=` молча пошли ПРЕЖНИМ разбором,
    # то есть два соседних прогона мерили разные конфигурации прода. Умолчание стенда обязано
    # совпадать с продом; отклонение от прода объявляется ЯВНО.
    kw, seq, slot, order, depth0, gate, prob = {}, None, None, "x_center", True, None, ""
    softfg = 0.0
    rowdec = ""
    rddir = ""
    for part in filter(None, tail.split(",")):
        k, _, v = part.partition("=")
        if k == "seq":
            seq = v
        elif k == "slot":
            slot = v
        elif k == "order":
            order = v or "x_center"
        elif k == "prob":
            # §6.110: recall-модель переднего плана (`config.prob_provider`). §6.3 закрыл её
            # выводом «на чернилах работает, на выдачу не переносится», но это замер ИЮЛЯ — до
            # селектора и до починки отбора. Тогда выигрыш было чем съесть; теперь проверяем заново.
            prob = v
        elif k == "rdpool":
            # ★ Способ отдачи траекторий раскладке: 1 = ПУЛ кандидатов, 0 = готовая пара 1:1.
            # ⚠ Заведено после того, как редакция 3 (пул) дала −120 кривых против +71 у пары, НО
            # одновременно сменился набор чекпойнтов — два изменения разом, вклад неразделим
            # (§6.74: без флага абляции вклад неизмерим). Этот ключ и есть флаг абляции.
            kw["__rdpool__"] = int(v or 0)
        elif k == "rdmodel":
            # ★★ §6.145: ОБУЧЕННЫЙ выбор пути вместо порога (`cv.rowdec_pick_model`).
            # Перебивает `rdpick`. ⚠ Требует `rowdec=`, как и порог.
            kw["__rdmodel__"] = v
        elif k == "rdpick":
            # ★★ §6.144: выбор пути ПО ТРЕКУ — декодер там, где он повёл ≤ N линий, иначе прод.
            # 0 = выкл (декодер заменяет ведение целиком). ⚠ Требует `rowdec=`: без декодера
            # выбирать не из чего, и ключ молча ничего не делает — поэтому ниже проверка.
            kw["__rdpick__"] = int(v or 0)
        elif k == "rdcolor":
            # ★ Вес штрафа «цвет под траекторией ≠ цвет линии» при привязке 1:1 (`cv.rowdec_color`),
            # в пикселях. 0 = ВЫКЛ = поведение §6.141. ⚠ Действует только при `rdpool=0`:
            # пул кандидатов отвергнут замером (−120 кривых, §6.141), и штрафу там негде сработать.
            kw["__rdcolor__"] = float(v or 0.0)
        elif k == "rddir":
            # ★ КАТАЛОГ ВАРИАНТА ОБУЧЕНИЯ ДЕКОДЕРА (`cv.rowdec_dir`). Пусто = замороженный набор
            # `frozen_nobg`, на котором сняты числа §6.141. ⚠⚠ Вариант обучения пиннится ИМЕННО
            # ЗДЕСЬ, а не подменой файлов поверх: именно подмена и стоила пересчёта §6.141.
            rddir = v
        elif k == "rowdec":
            # ★ Задача 9 (§6.140): построчный декодер вместо ведения. "auto5" = чекпойнт по
            # скважине (каждый лист декодируется моделью, его скважину НЕ видевшей, §6.70).
            rowdec = v
        elif k == "softfg":
            # ★ §6.133: МЯГКИЙ ПЕРЕДНИЙ ПЛАН ДЛЯ ТРАССИРОВКИ. Потолок построчного решения держит
            # ПОКРЫТИЕ: под пропущенными строками тушь есть, но бледная (V≈123 при бумаге 243 и
            # пороге dark_v=110), и чтение пикселей вместо прод-бинаря даёт +688 кривых потолка.
            # ⚠ Это НЕ то же, что `prob=`: recall-модель подключается через `prob_provider`, а он
            # кормит `ink_foreground` (U1, ПОИСК линий). Трассировка же идёт по `trace2d._color_fg`,
            # куда prob не попадает вовсе ⇒ §6.110 («−6 кривых») мерил другой механизм.
            softfg = float(v or 0.0)
        elif k == "sib":
            # ★ §6.130: вес добавки «порядок зондов в семействе». По умолчанию 0 = прод.
            kw["__sib__"] = float(v or 0.0)
        elif k == "pregate":
            # ★★ §6.178: ПРЕДГЕЙТ ПО ЛИСТУ — гнать второй путь только там, где доля самого частого
            # цвета среди линий листа не ниже порога. 0 = гейта нет (второй путь на каждом листе).
            # ⚠ Требует `rowdec=`: без второго пути гейту нечего пропускать, и ключ молча ничего бы
            # не значил — а молчаливо пустая ручка уже стоила ветке замера (§6.153).
            kw["__pregate__"] = float(v or 0.0)
        elif k == "kslots":
            # ★ §6.204: K декодера не ниже числа слотов трека (`cv.rowdec_k_slots`). 0 = прод.
            # ⚠ Требует `rowdec=`: без декодера ручка не значит ничего.
            kw["__kslots__"] = bool(int(v or 0))
        elif k == "rdpeak":
            # ★ §6.207: порог пиков декодера (доля максимума строки), прод 0.6.
            kw["__rdpeak__"] = float(v or 0.6)
        elif k == "kspick":
            # ★ §6.206: как считать n_dec для порога rowdec_pick при kslots: count (все слоты) | u1.
            kw["__kspick__"] = v or "count"
        elif k == "slotlen":
            # ★ §6.213: выбор по слоту по длине (`cv.rowdec_slot_len`), порог log1p-разности
            #   непустых строк; 0 = выкл. ⚠ Требует `rowdec=`.
            kw["__slotlen__"] = float(v or 0.0)
        elif k == "slotgeom":
            # ★ §6.215: геометрия шкалы слота из шаблона как ограничение раскладки (`cv.slot_template_geom`).
            kw["__slotgeom__"] = bool(int(v or 0))
        elif k == "slotfill":
            # ★ §6.223: правило слота видит и слоты, заполненные только вторым путём (`cv.rowdec_slot_fill`).
            kw["__slotfill__"] = bool(int(v or 0))
        elif k == "slotall":
            # ★ §6.213: декодер и за предгейтом — ради выбора по слоту (`cv.rowdec_slot_all`).
            #   ⚠ Требует `slotlen=`: без правила по слоту декодер за гейтом ничего не решает.
            kw["__slotall__"] = bool(int(v or 0))
        elif k == "wellmap":
            # ★★ §6.153: карта «лист → скважина манифеста». Делает держанность `auto5`
            # настоящей; без неё 51% листов декодирует модель, видевшая скважину.
            kw["__wellmap__"] = v or ""
        elif k == "degen":
            # ★★ §6.152: не вычитать ВЫРОЖДЕННЫЙ цветовой канал из чёрного бинаря трассировки
            # (`cv.color_fg_degen`). Тот же порог `color_max_frac`, что у `ink_foreground`.
            # ⚠ На листе БЕЗ вырожденного канала режим обязан быть бит-в-бит продом — это и есть
            # его контроль: расхождение там означало бы, что правка трогает не то, что обещала.
            kw["__degen__"] = bool(int(v or 0))
        elif k == "gate":
            # §6.108: мера уверенности обученной раскладки — ПО РЕЖИМУ, а не одна на прогон:
            # сравнивать порога гейта иначе нечем, а именно они и решают, работает ли механизм.
            gate = v or None
        elif k == "depth0":
            # §6.108: ПРАВКА УЖЕ В ПРОДЕ, поэтому ручка теперь ВЫКЛЮЧАЕТ её: `depth0=0` = прежнее
            # поведение (`0 < a`, интервал от нуля не разбирается). Смысл перевёрнут осознанно:
            # §6.107 померил правку ТОЛЬКО на старом архиве (36 листов, 6 → 6 кривых) и записал
            # «нейтральна», а на новых скважинах разница между стендами оказалась +34 кривые.
            depth0 = v not in ("", "0")
        else:
            kw[k] = None if v in ("None", "") else (float(v) if k == "slmax" else int(v))
    if seq is None:
        sys.exit(f"режим {nm!r}: не задан seq= (пусто = жадный путь, иначе имя чекпойнта)")
    if slot is None:
        sys.exit(f"режим {nm!r}: не задан slot= (пусто = раскладка ПРАВИЛОМ, иначе имя веса)")
    # ⚠ `order=` НЕ обязателен, в отличие от `seq`/`slot`, и это осознанно. Пиннить требуется то,
    # что молча разъезжается МЕЖДУ МАШИНАМИ: `seq` зависит от наличия torch, `slot` — от наличия
    # веса. `slot_order` же берётся из конфига одинаково везде, поэтому умолчание тут честное.
    if kw.get("__pregate__") and not rowdec:
        sys.exit(f"⛔ режим {nm!r}: `pregate=` без `rowdec=` — гейту нечего пропускать, "
                 f"ручка не значила бы ничего")
    if kw.get("__kslots__") and not rowdec:
        sys.exit(f"⛔ режим {nm!r}: `kslots=` без `rowdec=` — ручка декодера без декодера")
    if ("__rdpeak__" in kw or "__kspick__" in kw) and not rowdec:
        sys.exit(f"⛔ режим {nm!r}: `rdpeak=`/`kspick=` без `rowdec=` — ручка декодера без декодера")
    if "__kspick__" in kw and not kw.get("__kslots__"):
        sys.exit(f"⛔ режим {nm!r}: `kspick=` без `kslots=1` — считать нечего, режим молча совпал бы с K")
    if kw.get("__slotlen__") and not rowdec:
        sys.exit(f"⛔ режим {nm!r}: `slotlen=` без `rowdec=` — выбирать по слоту не из чего")
    if kw.get("__slotall__") and not kw.get("__slotlen__"):
        sys.exit(f"⛔ режим {nm!r}: `slotall=1` без `slotlen=` — декодер за гейтом ничего не решал бы, режим молча совпал бы с продом")
    if (kw.get("__rdpick__") or kw.get("__rdmodel__")) and not rowdec:
        sys.exit(f"режим {nm!r}: rdpick= задан, но rowdec= пуст — выбирать не из чего, режим молча совпал бы с продом")
    if order not in ("x_center", "med_x", "rough_n"):
        sys.exit(f"режим {nm!r}: order={order!r} — ждали x_center | med_x | rough_n")
    # ⚠ `__sib__` — ручка РАСКЛАДКИ (§6.130), а не параметр `trace_line`: под селектором она
    # действует, поэтому из этой проверки исключена.
    _kwv = [k for k in kw if k not in ("__sib__", "__rdcolor__", "__rdpool__", "__rdpick__",
                                      "__rdmodel__", "__degen__", "__wellmap__", "__pregate__",
                                      "__kslots__", "__kspick__", "__rdpeak__", "__slotlen__", "__slotall__", "__slotgeom__", "__slotfill__")]
    if seq and _kwv:
        print(f"⚠ режим {nm!r}: при включённом селекторе параметры {_kwv} НЕ действуют — "
              f"`trace_seq` строит свой трассировщик и правила вершины у него нет вовсе")
    return nm, seq, slot, order, depth0, gate, prob, softfg, rowdec, rddir, kw


# ── §6.107: вариант `_depth_marker`, принимающий интервал от НУЛЯ ─────────────────────────────
# Собирается из исходника прод-функции подменой одного сравнения: так вариант не разъедется с
# продом при любой другой правке разбора имени (та же логика, что у паритета §6.105).
import inspect, textwrap
_dm_orig = M._depth_marker
_ns = dict(M.__dict__)
exec(compile(textwrap.dedent(inspect.getsource(_dm_orig)).replace("0 <= a < 12000", "0 < a < 12000"),
             "<depth0>", "exec"), _ns)
_dm_zero = _ns["_depth_marker"]          # ⚠ теперь это ПРЕЖНЕЕ поведение, а прод — вариант с нулём
if _dm_zero.__code__.co_code == _dm_orig.__code__.co_code:
    sys.exit("⛔ вариант depth0 совпал с продом — подмена сравнения не сработала, замер бессмыслен")

MODES = [parse_mode(s) for s in a.mode]
_PROV = {}
for _m in MODES:
    _ck = _m[6]
    if _ck and _ck not in _PROV:
        from auto.prob import make_prob_provider
        _PROV[_ck] = make_prob_provider(_ck)
        print(f"★ recall-модель загружена: {_ck}")
_orig_trace = T.trace_line
_orig_color_fg = T._color_fg


def make_softfg(delta):
    """Маска трассировки, дополненная БЛЕДНОЙ ТУШЬЮ (§6.133, шаг 1 Задачи 9).

    ⚠⚠ ТОЛЬКО ДЛЯ ЧЁРНОГО, И ЭТО НЕ ПЕДАНТИЗМ. Мягкий критерий «темнее бумаги своей строки на
    delta» БЕСЦВЕТЕН: в потолке §6.133 (вариант V3) он и мерился бесцветным, потому что построчный
    декодер цветом не пользуется. Влить его во все цветовые маски нельзя — трассировка нынешнего
    прода СТОИТ на цвето-разделении (`fg_cache[L.color]`), и красная линия получила бы в свою маску
    чужую чёрную тушь. Поэтому добавка идёт в маску `black` и только там, где пиксель НЕ цветной;
    цветные маски остаются прод-масками бит-в-бит. Чёрных кривых в корпусе 5365 из 6862.
    ⚠ Порог ОТНОСИТЕЛЬНЫЙ (бумага своей строки), иначе на пожелтевших сканах накроет лист (§6.52).
    Структура вычитается — иначе добавка потащит светлую сетку (§6.6.13, V≈166)."""
    def wrapped(rgb, color, p):
        base = _orig_color_fg(rgb, color, p)
        if color != "black":
            return base
        res = im._mres(rgb)                    # мемо на скан: маска считается раз на лист
        key = ("softfg", float(delta))
        if key not in res:
            V = im.value_channel(rgb)
            H = V.shape[0]
            paper = np.empty(H, np.float32)
            for y0 in range(0, H, 4096):       # ⚠ память: percentile по всей матрице поднимает float64
                y1 = min(H, y0 + 4096)
                paper[y0:y1] = np.percentile(V[y0:y1, ::4], 90, axis=1)
            soft = (V <= (paper - float(delta))[:, None]) & ~im.structure_mask(rgb, p)
            for cm in im.color_channels(rgb, p).values():
                soft = soft & ~cm              # добавка — только НЕцветное, цвет ведёт своя маска
            res[key] = soft
        return base | res[key]
    return wrapped


def make(kw):
    def wrapped(fg, line, frame, p, **rest):
        for k, v in kw.items():
            rest.setdefault(k, v)          # ⚠ setdefault: явные x_range/wide_run refine'а живут
        return _orig_trace(fg, line, frame, p, **rest)
    return wrapped


def err(ours, gt):
    com = [y for y in ours if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(ours[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


def leaked(tr, raw):
    """Контроль копирования эксперта (§6.36)."""
    er = np.array([raw["top_y"] + i for i, x in enumerate(raw["xs"]) if x != NULL])
    ex = np.array([x for x in raw["xs"] if x != NULL], float)
    oy = np.array(sorted(tr)); ox = np.array([tr[y] for y in oy], float)
    if len(oy) == len(er) and np.array_equal(oy, er) and np.allclose(ox, ex):
        return True
    com = np.intersect1d(oy, er)
    if len(com) >= 30:
        om = dict(zip(oy.tolist(), ox.tolist())); em = dict(zip(er.tolist(), ex.tolist()))
        if np.mean([abs(om[y] - em[y]) < 1e-9 for y in com]) > 0.5:
            return True
    return False


# ⚠ §6.108: КОРНИ РАЗМЕТКИ — СПИСКОМ. Держанные скважины лежат в `intake\sorted`, а не в
# `projects\Archive`, и стенд, знающий только архив, молча взял бы НОЛЬ листов и отработал.
WELL, WLG = {}, {}
for root in a.wlg_roots:
    for wlg in Path(root).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            WELL.setdefault(q.name, wlg.parent.name); WLG.setdefault(q.name, q)
for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
    WELL.setdefault(q.name, "Semeguniv"); WLG.setdefault(q.name, q)
print(f"разметки найдено: {len(WLG)} листов в {len(a.wlg_roots)} корнях")

# ── выборка: до --cap листов на СКВАЖИНУ, чтобы плотные скважины не решали за всех ────────────
# ⚠ Список строится ПО ИМЕНАМ ФАЙЛОВ. Унаследованный цикл `pickle.load` по всем пулам читал 2.35 ГБ
# ради одного поля `name` — на шардах это десятки гигабайт до начала работы (та же ловушка, что
# в `_param_sweep.py`). `_pool_oracle.py` кладёт дамп как `{nlgx.stem[:60]}.pkl`.
BY_STEM = {q.stem[:60]: q for q in WLG.values()}
seen, per_well, sheets = set(), {}, []
POOLF = {}                    # имя nlgx → тот самый дамп, которым лист и будет считаться (вес шарда)
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        q = BY_STEM.get(f.stem)
        if f.stem in seen or q is None:
            continue
        seen.add(f.stem)
        wl = WELL.get(q.name, "?")
        if a.cap and len(per_well.get(wl, [])) >= a.cap:
            continue
        per_well.setdefault(wl, []).append(q.name); sheets.append(q); POOLF[q.name] = f
if a.only_from:
    want = {ln.strip() for ln in Path(a.only_from).read_text(encoding="utf-8").splitlines() if ln.strip()}
    sheets = [q for q in sheets if q.name in want]
    missing = sorted(want - {q.name for q in sheets})
    print(f"★ ОТБОР ПО СПИСКУ {a.only_from}: просили {len(want)}, нашлось {len(sheets)}"
          f"   {'★ СОШЛОСЬ' if not missing else '⛔ НЕ СОШЛОСЬ'}")
    for m in missing[:5]:
        print(f"    нет в пулах: {m}")
print(f"листов в пулах {len(seen)}, взято {len(sheets)} со {len(per_well)} скважин "
      f"(cap {a.cap or '—'}), режимов {len(MODES)}")
Path(a.out).mkdir(parents=True, exist_ok=True)
if SH_N > 1:
    sheets = sorted(sheets, key=lambda p: p.name)
    if a.shard_by_pool:
        # ⚠ Блочное разбиение (ветка else) отдало ОДНОМУ шарду все дорогие пулы: перекос 2.9×,
        # хвост 49 ч при 7-8 ч у остальных, машина при этом занята на треть (§6.118).
        # LPT: самые дорогие листы раскладываем первыми, каждый — в самый лёгкий на тот момент шард.
        # ★ Раскладка ДЕТЕРМИНИРОВАНА (тай-брейк по имени и по номеру шарда) — все шарды считают её
        # независимо и обязаны получить одно и то же разбиение, никакой связи между процессами нет.
        wt = {q.name: (POOLF[q.name].stat().st_size if q.name in POOLF else 0) for q in sheets}
        bins, load = [[] for _ in range(SH_N)], [0] * SH_N
        for q in sorted(sheets, key=lambda p: (-wt[p.name], p.name)):
            z = min(range(SH_N), key=lambda t: (load[t], t))
            bins[z].append(q); load[z] += wt[q.name]
        sheets = sorted(bins[SH_I], key=lambda p: p.name)
        print(f"★ ШАРД {SH_I}/{SH_N} ПО ПУЛУ: {len(sheets)} листов × {len(MODES)} режима, "
              f"пул {load[SH_I]/2**20:.0f} МБ; по шардам {min(load)/2**20:.0f}-"
              f"{max(load)/2**20:.0f} МБ (перекос {max(load)/max(1, min(load)):.2f}×)")
    else:
        lo = len(sheets) * SH_I // SH_N; hi = len(sheets) * (SH_I + 1) // SH_N
        sheets = sheets[lo:hi]      # блоком, а не через шаг: сканы скважины лежат рядом
        print(f"★ ШАРД {SH_I}/{SH_N}: {len(sheets)} листов × {len(MODES)} режима")

res, FP, LK = {}, {}, {}
# ★ A5 (27.09): отпечаток режима — ДО цикла, пока `kw` не опустошён `pop`-ами; дамп другой конфигурации не подхватывается
_SPEC = {m[0]: repr(m) for m in MODES}
import time as _time_mod
_T_RUN = _time_mod.time()
# ★★★ ПРОМЕЖУТОЧНЫЙ ДАМП: ОСТАНОВКА ДОЛЖНА СТОИТЬ ЛИСТОВ, А НЕ ЧАСОВ (§6.186).
# Прежде дамп писался ОДИН раз, в самом конце шарда. Значит снятый шард терял ВСЁ: 05.09 так ушли
# 249 посчитанных листов, потому что дампов было ноль. Теперь раз в `--flush` листов пишется
# `ab_<i>of<N>.part.pkl`, и перезапуск эти листы ПРОПУСКАЕТ.
# ⚠⚠ ИМЯ ДРУГОЕ, И ЭТО ГЛАВНОЕ. Готовый дамп `ab_<i>of<N>.pkl` означает «шард ДОСЧИТАН» — по нему
# тик и `_fold_missing.py` судят о полноте. Положи туда недосчитанное — и контролёры объявят готовым
# неполное, ровно дефект §6.166. Поэтому промежуточный лежит под `.part.pkl` и удаляется в конце.
PART = Path(a.out) / f"ab_{SH_I}of{SH_N}.part.pkl"
if PART.is_file():
    try:
        _part = pickle.load(open(PART, "rb"))
        _prev = _part.get("res", {})
        _spec = _part.get("spec")
        if _spec is None:
            print("⚠ промежуточный дамп без отпечатка режимов (до A5) — подхватываю без сверки конфигурации")
        else:
            for _nm in [k for k in _prev if _spec.get(k) != _SPEC.get(k)]:
                print(f"⛔ режим {_nm!r}: в дампе другая конфигурация — его листы считаются заново")
                _prev.pop(_nm)
        res.update(_prev)
        FP.update({k: v for k, v in _part.get("fp", {}).items() if k in _prev})   # ★ 19.09 (Д5 ревизии): отпечатки тоже
        LK.update({k: v for k, v in _part.get("lk", {}).items() if k in _prev})
        print(f"★ НАЙДЕН ПРОМЕЖУТОЧНЫЙ ДАМП: режимов {len(_prev)}, листов "
              f"{sum(len(v[1]) for v in _prev.values())} — они будут ПРОПУЩЕНЫ")
    except Exception as e:
        print(f"⚠ промежуточный дамп не читается ({type(e).__name__}) — считаю с нуля")


def _flush(why, force=False):
    """Атомарно: пишем во временный и переименовываем, иначе снятый в момент записи шард оставил
    бы обрезанный файл, и следующий запуск считал бы мусор за прогресс.
    `force` — выход партии/по памяти: пишем и при `--flush 0`, иначе перезапуск начинал бы с нуля бесконечно (A5)."""
    if not a.flush and not force:
        return
    tmp = PART.with_suffix(".tmp")
    pickle.dump({"res": res, "fp": FP, "lk": LK, "spec": _SPEC, "modes": [tuple(m) for m in MODES]}, open(tmp, "wb"))
    tmp.replace(PART)
    print(f"  ★ промежуточный дамп сохранён ({why})")


for nm_mode, seq, slot, order, depth0, gate, prob, softfg, rowdec, rddir, kw in MODES:
    # ★ §6.130: `sib` — не параметр `trace_line`, а ручка РАСКЛАДКИ. Вынимаем ДО построения
    # обёртки: иначе он уедет в kwargs трассировщика, который его не ждёт, и заодно включит
    # обёртку там, где режим её не просил.
    sib_w = float(kw.pop("__sib__", 0.0) or 0.0)
    # ⚠ ВЫНИМАТЬ ДО `make(kw)`: иначе ключ уедет в kwargs `trace_line`, который его не ждёт.
    rd_pool = bool(kw.pop("__rdpool__", 1))
    rd_col = float(kw.pop("__rdcolor__", 0.0) or 0.0)
    rd_pick = int(kw.pop("__rdpick__", 0) or 0)
    pregate = float(kw.pop("__pregate__", 0.0) or 0.0)
    rd_model = kw.pop("__rdmodel__", "") or ""
    degen = bool(kw.pop("__degen__", False))
    wellmap = kw.pop("__wellmap__", "") or ""
    kslots = bool(kw.pop("__kslots__", False))
    kspick = kw.pop("__kspick__", "count") or "count"
    rdpeak = float(kw.pop("__rdpeak__", 0.6) or 0.6)
    slotlen = float(kw.pop("__slotlen__", 0.0) or 0.0)
    slotall = bool(kw.pop("__slotall__", False))
    slotgeom = bool(kw.pop("__slotgeom__", False))
    # ⚠ 26.09: `rowdec_slot_fill` в проде True — режим задаёт его ЯВНО (умолчание режима — выкл), иначе прежние режимы
    #   P/K/RA/N молча получили бы новую правку и перестали бы значить то, что значили в журналах
    slotfill = bool(kw.pop("__slotfill__", False))
    M._depth_marker = _dm_orig if depth0 else _dm_zero   # depth0=1 → прод; 0 → прежнее
    T.trace_line = make(kw) if kw else _orig_trace
    # ★ §6.133: мягкий передний план — режим, а не умолчание (§6.71: путь задаёт стенд).
    T._color_fg = make_softfg(softfg) if softfg else _orig_color_fg
    # ★ ПОДХВАТ ПРОМЕЖУТОЧНОГО: если этот режим уже частично посчитан прошлым заходом, берём его
    #   счёт и словарь листов — тогда цикл ниже пропустит их по `if n.name in per`.
    _since, _done_skip, _redo = 0, 0, 0
    tot, per = res.get(nm_mode, (dict(hon=0, curves=0, sheets=0, leak=0), {}))
    tot, per = dict(tot), dict(per)
    FP[nm_mode] = FP.get(nm_mode, {})
    LK[nm_mode] = LK.get(nm_mode, {})
    SKIP = {}                    # §6.106: сверка списка — обязательная печать, а не отладка
    print(f"\n{'='*78}\n{nm_mode}: seq={seq or '— (жадный trace2d)'}, "
          f"slot={slot or '— (раскладка правилом)'}, order={order}, "
          f"depth0={'ВКЛ' if depth0 else 'выкл'}, гейт={gate or a.slot_gate}, "
          f"softfg={softfg or '— (прод-бинарь)'}, "
          f"rowdec={rowdec or '— (выкл, прод-ведение)'}, "
          f"rddir={rddir or '— (замороженный frozen_nobg)'}, "
          f"rdcolor={rd_col or '— (выкл)'}, "
          f"rdpick={rd_pick or '— (выкл, путь один)'}, "
          f"pregate={pregate or '— (выкл, второй путь на каждом листе)'}, "
          f"rdmodel={rd_model or '— (выкл)'}, "
          f"degen={'ВКЛ' if degen else '— (выкл, прод)'}, "
          f"wellmap={'ЕСТЬ (честная держанность)' if wellmap else '— (скважина из имени)'}, "
          f"kslots={'ВКЛ (K ≥ слотов трека, §6.204)' if kslots else '— (выкл, K от U1)'}, "
          f"kspick={kspick}, rdpeak={rdpeak}, "
          f"slotlen={slotlen or '— (выкл)'}, slotall={'ВКЛ (декодер и за предгейтом)' if slotall else '— (выкл)'}, "
          f"slotgeom={'ВКЛ (полоса+ноль шкалы, §6.215)' if slotgeom else '— (выкл)'}, "
          f"slotfill={'ВКЛ (пустые слоты, §6.223)' if slotfill else '— (выкл)'}, "
          f"{kw or 'константы trace_line по умолчанию'}\n{'='*78}")
    for n in sheets:
        img = find_image(n)
        if not img:
            SKIP["нет картинки"] = SKIP.get("нет картинки", 0) + 1
            continue
        cfg = Config()
        cfg.cv.seq_model = seq            # §6.71: путь трассировки задаёт стенд, а не умолчания
        cfg.cv.slot_model = slot          # §6.71: и раскладку тоже — см. шапку про `_slot_prod_ab`
        cfg.cv.slot_gate = gate or a.slot_gate
        # ⚠ Провайдер строится ОДИН раз на процесс: загрузка чекпойнта и сборка сети на каждый
        # лист съели бы больше, чем сам инференс.
        cfg.prob_provider = _PROV.get(prob) if prob else None
        cfg.cv.slot_order = order         # §6.105: ключ порядка в раскладке ПРАВИЛОМ
        cfg.cv.slot_sib = sib_w
        cfg.cv.row_decoder = rowdec
        cfg.cv.rowdec_pool = rd_pool
        cfg.cv.rowdec_dir = rddir
        cfg.cv.rowdec_color = rd_col
        cfg.cv.rowdec_pick = rd_pick
        cfg.cv.rowdec_pregate = pregate
        cfg.cv.rowdec_pick_model = rd_model
        cfg.cv.color_fg_degen = degen
        cfg.cv.rowdec_wellmap = wellmap
        cfg.cv.rowdec_k_slots = kslots
        cfg.cv.rowdec_k_slots_pick = kspick
        cfg.cv.rowdec_peak_thr = rdpeak
        cfg.cv.rowdec_slot_len = slotlen
        cfg.cv.rowdec_slot_all = slotall
        cfg.cv.slot_template_geom = slotgeom
        cfg.cv.rowdec_slot_fill = slotfill
        # ⚠⚠⚠ ИМЯ КАТАЛОГА — ПОЛНОЕ, А НЕ ОБРЕЗАННОЕ (§6.117). Здесь стояло `n.stem[:40]`, а варианты
        # одного бланка различаются ПОСЛЕ 40-го символа (`..._200_D_1`, `_D_2`, `_D_3`) — они писали
        # выдачу в ОДИН каталог, и чтение `sorted(glob("*_auto.nlgx"))[0]` возвращало файл ЧУЖОГО
        # листа (первый по алфавиту). Коллизия межшардовая: в каталог режима пишут все 12 процессов,
        # поэтому результат зависел от разбиения и порядка — два прогона одной конфигурации давали
        # 248 и 245. Задето 52 листа из 450 в сорте A и 20 из 337 в аудиторском наборе.
        # Хвост-хэш держит длину пути в узде и при этом уникален.
        cfg.out = (Path(a.out) / nm_mode.split()[0] /
                   f"{n.stem[:40]}_{hashlib.md5(n.stem.encode('utf-8')).hexdigest()[:8]}")
        # ★ лист уже посчитан прошлым заходом — пропускаем, счёт для него уже в `per`.
        # ⛔⛔ 20.09: ПОРЯДОК БЫЛ ОБРАТНЫМ — чистка выдачи стояла ДО этой проверки, и каждый перезапуск
        #   шарда (партия 6 ч, память, гибель) СТИРАЛ `_auto.nlgx` всех уже посчитанных листов, а потом
        #   пропускал их. A/B §6.213 дошёл до конца с полными дампами и БЕЗ выдач G и R (0 из 1123) —
        #   ведущий счёт (`_name_cost_prod.py` читает файлы) остался без данных. Теперь пропуск требует
        #   и записи в `per`, и файла выдачи; чистка — только перед пересчётом.
        _have_out = cfg.out.is_dir() and any(cfg.out.glob("*_auto.nlgx"))
        if n.name in per and _have_out:
            _done_skip += 1
            continue
        if n.name in per and not _have_out:
            _redo += 1                          # запись есть, файла нет — считаем заново
            # ⛔ A5 (27.09): прежний счёт листа вычитается, иначе пересчёт удваивал его в `tot` (утечки прежних
            #   дампов без `lk` не известны — считаются 0)
            _ng = sum(1 for c in extract(str(n))["curves"]
                      if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50)
            tot["hon"] -= per.pop(n.name); tot["curves"] -= _ng; tot["sheets"] -= 1
            tot["leak"] -= LK[nm_mode].pop(n.name, 0)
        # ⚠ Чистим выдачу прошлого прогона: иначе при повторе с другой конфигурацией прочтётся старый
        # файл, и правка окажется «нейтральной», хотя она просто не доехала.
        if cfg.out.is_dir():
            for old in cfg.out.glob("*_auto.nlgx"):
                old.unlink()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
        except Exception as e:
            SKIP["падение пайплайна"] = SKIP.get("падение пайплайна", 0) + 1
            print(f"  {n.stem[:44]:<46} ПАДЕНИЕ {type(e).__name__}: {e}"); continue
        G = extract(str(n))
        raws = {c["name"]: c for c in G["curves"]
                if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        gts = {nm: dense(c) for nm, c in raws.items()}
        got = next(iter(sorted(cfg.out.glob("*_auto.nlgx"))), None)
        if got is None:
            SKIP["файл не выдан"] = SKIP.get("файл не выдан", 0) + 1
            print(f"  {n.stem[:44]:<46} файл не выдан"); continue
        # ★★ КОНТРОЛЬ ПРИНАДЛЕЖНОСТИ (§6.117): прочитанный файл обязан быть выдачей ИМЕННО этого
        # листа. Имя выдачи строит `pipeline.run` из имени КАРТИНКИ, поэтому сверка точная. Без неё
        # путаница листов молчалива: числа выглядят правдоподобно и расходятся только между прогонами.
        if got.stem != Path(img).stem + "_auto":
            SKIP["выдача чужого листа"] = SKIP.get("выдача чужого листа", 0) + 1
            print(f"  {n.stem[:44]:<46} ⛔ ВЫДАЧА ЧУЖОГО ЛИСТА: {got.stem[:44]}"); continue
        W = {c["name"]: dense(c) for c in extract(str(got))["curves"]
             if M.mnem_root(c["name"]) != "DA"}
        FP[nm_mode][n.name] = tuple(sorted(
            (k, len(v), round(float(np.median(list(v.values()))), 3)) for k, v in W.items() if v))
        h = lk = 0
        for k, gd in gts.items():
            if k not in W or not W[k]:
                continue
            if leaked(W[k], raws[k]):
                lk += 1; continue
            h += HON(*err(W[k], gd))
        per[n.name] = h; LK[nm_mode][n.name] = lk
        tot["hon"] += h; tot["curves"] += len(gts); tot["sheets"] += 1; tot["leak"] += lk
        print(f"  {n.stem[:42]:<44} кривых {len(gts):>2}  ЧЕСТНЫХ {h}")
        _since += 1
        if a.flush and _since >= a.flush:
            res[nm_mode] = (tot, per)
            _flush(f"{tot['sheets']} листов режима {nm_mode}")
            _since = 0
        if a.max_hours > 0 and (_time_mod.time() - _T_RUN) / 3600 > a.max_hours:
            res[nm_mode] = (tot, per)
            _flush(f"партия окончена: {(_time_mod.time() - _T_RUN) / 3600:.1f} ч > {a.max_hours} ч", force=True)
            print(f"★ ПАРТИЯ ОКОНЧЕНА: {(_time_mod.time() - _T_RUN) / 3600:.1f} ч — выхожу кодом 75, драйвер запустит шард заново с дампа")
            sys.exit(75)
        if a.max_rss_gb > 0:
            import psutil
            _rss = psutil.Process().memory_info().rss / 2**30
            if _rss > a.max_rss_gb:
                res[nm_mode] = (tot, per)
                _flush(f"перезапуск по памяти: RSS {_rss:.1f} ГБ > {a.max_rss_gb} ГБ", force=True)
                print(f"★ ПЕРЕЗАПУСК ПО ПАМЯТИ: RSS {_rss:.1f} ГБ после {tot['sheets']} листов режима {nm_mode} — "
                      f"выхожу кодом 75, драйвер запустит шард заново с дампа")
                sys.exit(75)
    res[nm_mode] = (tot, per)
    if _done_skip:
        print(f"★ пропущено как уже посчитанное: {_done_skip} листов")
    if _redo:
        print(f"★ пересчитано заново (запись была, выдачи не было): {_redo} листов")
    _flush(f"режим {nm_mode} завершён")
    print(f"ИТОГО {nm_mode}: листов {tot['sheets']}, кривых {tot['curves']}, "
          f"★честных {tot['hon']}, утечек {tot['leak']}")
    # ⚠⚠ СВЕРКА СПИСКА (§6.106, образец `_pool_oracle.py`). Без неё прогон по НЕПОЛНОЙ выборке
    # завершается успешно и печатает правдоподобные числа — так ветка уже дважды получала
    # цифру не про тот объём. Здесь она к тому же ловит РАЗЪЕЗД РЕЖИМОВ: если один режим
    # уронил больше листов, чем другой, сравнивать их дельты нельзя.
    _sk = sum(SKIP.values())
    print(f"  СВЕРКА СПИСКА: обработано {tot['sheets']} + пропущено {_sk} = "
          f"{tot['sheets'] + _sk} против длины списка {len(sheets)}"
          f"   {'★ СОШЛОСЬ' if tot['sheets'] + _sk == len(sheets) else '⛔ НЕ СОШЛОСЬ'}")
    for k, v in sorted(SKIP.items(), key=lambda q: -q[1]):
        print(f"    пропущено «{k}»: {v}")
    # ★★★ ВЫБОР ПУТИ ЗАПРОШЕН — ЗНАЧИТ ОН ОБЯЗАН БЫЛ ХОТЬ РАЗ СРАБОТАТЬ (§6.153).
    # ⚠ Как это стоило прогона: `rdmodel=` был НЕДОСТИЖИМ (`trace2d` возвращал декодер целиком и
    # не прицеплял `.alt`), выбор не считался ни разу, а прогон 221 листа завершился успешно и
    # напечатал правдоподобные числа — читались они как «обученный выбор хуже порога на 14
    # кривых», хотя мерили «ВСЕГДА ДЕКОДЕР». Единственный видимый след — отсутствие `_pick.json`.
    # ⇒ Проверять здесь, ГРОМКО: стенд, который не может отличить свой режим от чужого, врёт.
    if rd_pick or rd_model:
        _np = len(list((Path(a.out) / nm_mode).glob("*/*_pick.json")))
        print(f"  ★ ВЫБОР ПУТИ: листов с выгруженными признаками {_np} из {tot['sheets']}"
              f"{'' if _np else '   ⛔⛔ НИ ОДНОГО — ВЫБОР НЕ РАБОТАЛ, ЧИСЛА РЕЖИМА НЕГОДНЫ'}")
        if not _np:
            sys.exit(f"⛔ режим {nm_mode!r}: запрошен выбор пути "
                     f"({'модель ' + rd_model if rd_model else 'порог ' + str(rd_pick)}), "
                     f"но ни один лист не выгрузил признаки — измерялся НЕ ТОТ режим")
T.trace_line = _orig_trace

pickle.dump({"res": res, "fp": FP, "modes": [tuple(m) for m in MODES]},
            open(Path(a.out) / f"ab_{SH_I}of{SH_N}.pkl", "wb"))
# ★ ШАРД ДОСЧИТАН — промежуточный больше не нужен и не должен путаться под ногами
if PART.is_file():
    PART.unlink()
base_nm = MODES[0][0]
(ta, pa) = res[base_nm]
print(f"\n{'='*78}\n★★ ОТГРУЖАЕМЫЙ ПУТЬ (база = {base_nm}: {ta['hon']} честных / "
      f"{ta['curves']} кривых / {ta['sheets']} листов)\n{'='*78}")
for nm_mode, _seq, _slot, _order, _d0, _g, _p, _sf, _rd, _rdd, _kw in MODES[1:]:
    tb, pb = res[nm_mode]
    up = sum(1 for k in pa if pb.get(k, 0) > pa[k]); dn = sum(1 for k in pa if pb.get(k, 0) < pa[k])
    same = sum(1 for k in FP[base_nm] if FP[nm_mode].get(k) == FP[base_nm][k])
    flag = "" if same < len(FP[base_nm]) else "  ⛔ ВЫДАЧА НЕ ИЗМЕНИЛАСЬ — режим не включился"
    print(f"{nm_mode:<24} {ta['hon']} → {tb['hon']} ({tb['hon']-ta['hon']:+d}), "
          f"листов ↑{up}/↓{dn} из {len(pa)}; выдача различается на "
          f"{len(FP[base_nm])-same}/{len(FP[base_nm])} листах{flag}")

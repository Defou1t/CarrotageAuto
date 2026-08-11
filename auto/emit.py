r"""
emit.py — СДАЧА: понимание (U1/U2) + автономные трассы (trace2d) → артефакты.

DURABLE-ограничение (PLAN §6.6.10): фабрикация рамки С НУЛЯ КРАШИТ NeuraLOG (кэш-теги bbox
35478-86 должны родиться в NeuraLOG). Поэтому nlgx-выдача = ИНЪЕКЦИЯ наших трасс в ЛЁГКУЮ рамку,
сделанную экспертом в NeuraLOG (Depth/Scale Axis + слоты кривых, БЕЗ трассы). Это НЕ «сглаживание
по эксперту»: трасса 100% наша из анализа снимка; рамка — лишь лёгкий каркас (§6.6.9 «фрейминг
лёгкий», эксперт только проверяет). Маппинг линия→слот — автономный (трек + цвет/класс + порядок),
а не по экспертной трассе.

Режимы:
  emit_understanding  — understanding.json + overlay.png (всегда; первичный артефакт «понимания»);
  emit_into_frame     — инъекция AUTO-трасс в лёгкую рамку → _auto.nlgx(+bck) [+ las через export_las];
  emit                — диспетчер: понимание всегда, nlgx — если дана рамка.
"""
import json
from pathlib import Path
import numpy as np

from . import meta as meta_mod

NULL = 0xFFFFFFFF
COLOR_RGB = {"black": (255, 0, 255), "red": (255, 0, 0), "green": (0, 200, 0),
             "blue": (60, 120, 255), "orange": (255, 140, 0)}


# ---------- понимание: JSON + overlay ----------

def emit_understanding(sheet, traces, out, stem, rgb=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    js = out / f"{stem}_understanding.json"
    js.write_text(json.dumps(sheet.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    res = {"understanding": str(js)}
    if rgb is not None:
        res["overlay"] = _overlay(rgb, sheet, traces, out, stem)
    return res


def _overlay(rgb, sheet, traces, out, stem):
    from PIL import Image
    H, W = rgb.shape[:2]
    ov = rgb.copy()
    # рамка (U0) — циан: грани треков (вертикали) + верх/низ (горизонтали). Видно «рамку» анализа.
    fr = sheet.frame
    FRAME_RGB = (0, 200, 220)
    if fr.tracks and fr.bottom_y > fr.top_y:
        ty, by = max(0, fr.top_y), min(H, fr.bottom_y)
        for t in fr.tracks:
            for x in (t.x_left, t.x_right):
                if 0 <= x < W:
                    ov[ty:by, max(0, x - 1):x + 2] = FRAME_RGB
        x0 = max(0, fr.tracks[0].x_left); x1 = min(W, fr.tracks[-1].x_right)
        for y in (fr.top_y, fr.bottom_y):
            if 0 <= y < H:
                ov[max(0, y - 1):y + 2, x0:x1] = FRAME_RGB
    tr_by_id = {id(L): tr for L, tr in traces}
    for L in sheet.lines:
        # FLAG — жёлтым контуром x-полосы; AUTO с трассой — цветом линии
        col = COLOR_RGB.get(L.color, (255, 140, 0))
        tr = tr_by_id.get(id(L))
        if L.confidence == "FLAG":
            for y in range(max(0, L.y0), min(H, L.y1), 4):
                for x in (int(L.x_lo), int(L.x_hi)):
                    if 0 <= x < W:
                        ov[y, max(0, x - 1):x + 2] = (255, 230, 0)
        elif tr:
            for y, x in tr.items():
                x = int(x)
                if 0 <= y < H and 0 <= x < W:
                    ov[y, max(0, x - 1):x + 2] = col
    p = out / f"{stem}_overlay.png"
    Image.fromarray(ov).resize((max(1, W // 5), max(1, H // 5)), Image.LANCZOS).save(p)
    return str(p)


# ---------- nlgx: инъекция в лёгкую рамку ----------

def _slot_track(model, curve, frame):
    """Какому треку (индекс) принадлежит слот кривой — по x его scale-семейства vs колонки U0.

    КЛЮЧ = ПОЛНЫЙ суффикс имени («GZ11 DA1 SA1» → «DA1 SA1»), а не последний токен (19.07).
    Хвостовой токен «SA1» одинаков у РАЗНЫХ DA-семейств («DA1 SA1» и «DA2 SA1»), поэтому
    median(x_left) усреднялась по осям разных колонок и слот уезжал в чужой трек.
    Замер по архиву (многотрековые листы, 323 кривые): трек не содержал даже медиану самой
    кривой у 144 (44.6%); с полным ключом — 57 (17.6%). Тот же ключ уже применяет
    decode_levels.build_family. Фолбэк на старое поведение — если точного имени оси нет."""
    # ⚠ НЕ переписывать на model["curve_desc"]["axis_idx"] (запись kind=6): пробовали 19.07 —
    # индекс из файла даёт ТОТ ЖЕ ответ, что и суффикс имени (суффикс совпадает с именем оси у
    # 3074/3074 кривых, 100%), и та же доля промахов 17.6%. Выигрыша нет, ветвление лишнее.
    # Остаток промахов — НЕ ошибка поиска оси: у этих кривых ось эксперта реально стоит не там,
    # где нарисована кривая (x_left оси = калибровочный пролёт, а не область рисования, §6.11).
    full = " ".join(curve["name"].split()[1:])
    xs = [s["x_left"] for s in model["scale_axes"]
          if s.get("name") == full and s.get("x_left") is not None]
    if not xs:
        key = curve["name"].split()[-1]
        xs = [s["x_left"] for s in model["scale_axes"]
              if s["name"].split()[-1] == key and s.get("x_left") is not None]
    if not xs or not frame.tracks:
        return 0
    sx = float(np.median(xs))
    for t in frame.tracks:
        if t.x_left - 20 <= sx <= t.x_right + 20:
            return t.index
    return min(range(len(frame.tracks)),
               key=lambda i: abs((frame.tracks[i].x_left + frame.tracks[i].x_right) / 2 - sx))


def _trace_rough_n(tr):
    """Относительная ВЧ-дрожь трассы: std остатка от скользящего среднего, нормированная на размах.

    ⚠ ФОРМУЛА ПОВТОРЯЕТ `_slot_rules.feats` ОДИН В ОДИН (окно 101 или len//2*2+1, перцентили 10/90,
    `mode="same"`). Любое расхождение сделало бы A/B на отгрузке замером ДРУГОГО правила, а не того,
    что намерено на стенде, — паритет проверяется `_slot_order_parity.py`.
    """
    if not tr:
        return 0.0
    x = np.array([tr[y] for y in sorted(tr)], float)
    if len(x) < 5:
        return 0.0
    span = float(np.percentile(x, 90) - np.percentile(x, 10)) or 1.0
    # ⚠⚠ ОКНО НЕ ДЛИННЕЕ САМОЙ ТРАССЫ (§6.109). Было `len(x)//2*2+1`: при ЧЁТНОЙ длине это
    # len(x)+1, а `np.convolve(mode="same")` возвращает max(len(x), w) — вычитание `x - sm`
    # падало. Любая трасса чётной длины от 6 до 100 точек роняла раскладку, и после включения
    # `slot_model` по умолчанию (§6.108) это стало падением ПРОДА, а не только стенда.
    # ⚠ Правка НЕ меняет признаки там, где код работал: при нечётной длине w тот же, при
    # длине ≥101 окно и было 101. Значит обученный вес остаётся действительным.
    w = min(101, len(x) if len(x) % 2 else len(x) - 1)
    sm = np.convolve(x, np.ones(w) / w, mode="same") if w >= 3 else x
    return float(np.std(x - sm) / span)


def _slot_order_key(L, tr, how):
    """Ключ порядка линий внутри трека для раскладки (`CVParams.slot_order`).

    `x_center` — прежнее поведение: центр полосы, поставленный ещё на U1.
    `med_x`    — медиана x САМОЙ трассы: полоса и трасса не обязаны совпадать.
    `rough_n`  — порядок по ВЧ-дрожи вместо положения (§6.83: длинный зонд БКЗ даёт более гладкую
                 кривую, то есть форма упорядочивает зонды там, где положение уже не различает).
    ⚠ Неизвестное значение НЕ падает, а откатывается к прежнему поведению: раскладка не то место,
    где опечатка в конфиге должна ронять сдачу листа.
    """
    if how == "med_x":
        v = list(tr.values()) if tr else None
        return float(np.median(v)) if v else float(L.x_center)
    if how == "rough_n":
        return _trace_rough_n(tr)
    return float(L.x_center)


def _map_lines_to_slots(traces, model, frame, mnemonics_path, cv=None):
    """Автономный маппинг AUTO-линий → слоты кривых рамки: тот же трек + совместимый цвет/класс +
    порядок слева-направо (идентичность = СЧЁТ, §6.6.11). Возвращает {slot_name: (Line, trace)}.
    Слоты берём ПО ИМЕНИ (все кривые кроме оси DA*): в ЛЁГКОМ каркасе слоты пусты, а
    dataset.real_curves требует ≥50 точек и выкидывал их ВСЕ (G4-баг: written=[] на BK_4020)."""
    slots = [c for c in model.get("curves", [])
             if meta_mod.mnem_root(c.get("name", "")) != "DA"]
    # слоты по треку, с приором цвета/класса из мнемоники
    slot_info = []
    for c in slots:
        info = meta_mod.curve_info(c["name"], mnemonics_path)
        slot_info.append({"curve": c, "track": _slot_track(model, c, frame),
                          "color": info["color"], "class": info["class"],
                          "root": info["root"]})
    how = (getattr(cv, "slot_order", "") or "x_center") if cv is not None else "x_center"
    used = set(); mapping = {}
    # внутри трека: назначение по ГЛОБАЛЬНОМУ score (не жадно по слотам — иначе первый слот
    # забирает единственную линию при полном несовпадении: G4-баг STK_4020, оранжевая ушла в
    # красный SP1 вместо SP21). score: цвет 0/2 + класс 0/1; 3 = ничего не совпало → НЕ назначать.
    for ti in {s["track"] for s in slot_info}:
        tslots = [s for s in slot_info if s["track"] == ti]
        tlines = [(L, tr) for L, tr in traces if L.track_index == ti]
        # ЦВЕТ — ФИЛЬТР ТОЛЬКО ЕСЛИ ТАКАЯ ЛИНИЯ НА ТРЕКЕ ЕСТЬ (19.07). Строгий цвет ввели против
        # реального бага (чёрная PZ влезала в красный SP-слот, STK_4020), но цвет в mnemonics.json
        # — НЕ доменный факт: в словаре заказчика цветов нет вообще, их задаёт эксперт под каждый
        # бланк. Замер по 60 листам (чернила под экспертной трассой): SP нарисована ЧЁРНОЙ на
        # 17 листах из 20, а словарь объявляет её red ⇒ слот SP на монохромном бланке нельзя было
        # заполнить В ПРИНЦИПЕ. Сходится с гейтом: на всех листах «BKZ, DS» SP1 не записывался ни разу.
        # Условие ниже сохраняет защиту там, где цветная линия действительно есть, и снимает
        # запрет там, где её нет.
        colors_on_track = {L.color for L, _ in tlines}
        # ★ КЛЮЧ ПОРЯДКА (`CVParams.slot_order`) СЧИТАЕТСЯ ОДИН РАЗ НА ЛИНИЮ, до сборки пар: одна
        # линия входит в пары со ВСЕМИ слотами трека, а `rough_n` — свёртка по всей трассе.
        xkey = {id(L): _slot_order_key(L, tr, how) for L, tr in tlines}
        pairs = []
        for s in tslots:
            strict_color = s["color"] is not None and s["color"] in colors_on_track
            for L, tr in tlines:
                if strict_color and s["color"] != L.color:
                    continue
                cls = "SP" if L.behavior == "smooth" else "RES"
                class_ok = (s["class"] in (cls, "OTHER", "CALI"))
                pairs.append((0 if class_ok else 1, s, L, tr))
        pairs.sort(key=lambda q: (q[0], xkey[id(q[2])]))
        taken_slots = set()
        for score, s, L, tr in pairs:
            nm = s["curve"]["name"]
            if nm in taken_slots or id(L) in used:
                continue
            taken_slots.add(nm); used.add(id(L)); mapping[nm] = (L, tr)
    return mapping


def _is_mbk_name(curve_name):
    """MBK-мнемоника (первый токен без хвостовых цифр). Для ленивой загрузки скана в emit."""
    import re
    return re.sub(r"\d+$", "", curve_name.split()[0]).upper() == "MBK"


def _level_segments(tr, model, curve, top_y, n, gray=None):
    """Уровни-перевыносы (×5/×25) из НАШЕЙ трассы через decode_levels, если каркас несёт цепочку
    масштабов (base→next). Возвращает (tops, bottoms, levels) для тегов 35494/35496/35498.
    Одношкальный каркас / не-резистивная кривая → один сегмент level 0 (текущее поведение).
    5х-переход декодируется, ТОЛЬКО когда эксперт задал 5х-шкалу в каркасе (как в Archive BKZ).

    MBK (скоуп СТРОГО MBK): уровень = не value/wrap, а траектория одной свингующей линии.
    При наличии картинки (gray) используем decode_img_mbk (DP + image off-scale rail-гейт):
    MBK 0.538→0.595 на Archive (сессия 3, 17.07). Без картинки — обычный DP (безопасный fallback).
    GZ/OGZ/PZ/BK НЕ трогаем (у них чистый wrap; image-гейт их регрессирует)."""
    single = ([int(top_y)], [int(top_y + n - 1)], [0])
    try:
        import decode_levels as DL
        from . import refine
        from .config import DEFAULT
        if not DL.is_resistive(curve["name"]):
            return single
        fam = DL.build_family(model, curve)
        if len(fam) < 2:
            return single
        xs_by_row = {y: float(x) for y, x in tr.items()}
        # DP decode_levels (непрерывность лог-значения) + min-run сглаживание. Калибровка на
        # Archive GT (12.07): на ЧИСТОЙ трассе DP даёт level-acc мед 0.99 (lam=0.4) против 0.48 у
        # правила-рельса — DP ловит обороты гораздо точнее; мельтешение, что видел Эдуард, было от
        # ШУМА трассы (снят despike в trace2d), не от DP. min_run гасит остаточные короткие сегменты.
        try:
            import decode_img_mbk as MBK
            use_mbk = MBK.is_mbk(curve["name"]) and gray is not None
        except Exception:
            use_mbk = False
        if use_mbk:
            raw = MBK.mbk_levels(xs_by_row, fam, gray, lam=DEFAULT.cv.level_lam)
        else:
            raw = DL.decode(xs_by_row, fam, lam=DEFAULT.cv.level_lam)
        lv = refine.enforce_min_run(raw, DEFAULT.cv.level_min_run)
        if not lv:
            return single
        # сегменты по СМЕНЕ УРОВНЯ, НЕ по разрывам строк (пропуски внутри уровня = NULL в xs).
        # Незаданные строки наследуют последний уровень — сегмент непрерывен по глубине.
        tops, bots, levs = [], [], []
        last = lv.get(top_y, 0); s = top_y
        for y in range(top_y + 1, top_y + n):
            cur = lv.get(y, last)
            if cur != last:
                tops.append(s); bots.append(y - 1); levs.append(int(last))
                s = y; last = cur
        tops.append(s); bots.append(top_y + n - 1); levs.append(int(last))
        return tops, bots, levs
    except Exception:
        return single


def emit_into_frame(traces, frame_nlgx, frame, out, stem, mnemonics_path,
                    image=None, las=False, cv=None):
    """Инъекция AUTO-трасс в лёгкую рамку NeuraLOG → _auto.nlgx(+bck). Рамка НЕ фабрикуется."""
    import struct
    from extract_nlgx import extract
    from write_nlgx import read_full, write_full, set_tag, find_ifd
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model = extract(str(frame_nlgx))
    # ★ ОБУЧЕННАЯ РАСКЛАДКА (§6.88), выключена по умолчанию (`CVParams.slot_model` = ""). Вернёт
    # None, если её не просили, нет веса ИЛИ лист не прошёл меру уверенности ⇒ работает ПРАВИЛО.
    # ⚠ Правило вызывается тем же именем, что и раньше: стенды, которые его перехватывают
    # (`_pool_oracle`, `_pick_gate`, …), продолжают перехватывать именно правило.
    mapping = None
    if cv is not None and getattr(cv, "slot_model", ""):
        from . import slot_model as slot_mod
        mapping = slot_mod.map_lines(traces, model, frame, mnemonics_path, cv)
    if mapping is None:
        mapping = _map_lines_to_slots(traces, model, frame, mnemonics_path, cv)
    ifds = read_full(open(frame_nlgx, "rb").read())

    def curve_pred(short):
        def is_this(tags):
            if 34768 not in tags or struct.unpack("<I", tags[34768][2][:4])[0] != 7:
                return False
            if 35470 not in tags:
                return False
            nm = tags[35470][2].split(b"\x00")[0].decode("latin1")
            return nm == short or nm.startswith(short.split()[0] + " ")
        return is_this

    # gray скана грузим ЛЕНИВО и ОДИН раз — только если есть MBK-кривая (для image off-scale
    # декода уровней). Большие сканы (до 62k px) не грузим зря.
    _gray = {"arr": None, "loaded": False}

    def _get_gray():
        if not _gray["loaded"]:
            _gray["loaded"] = True
            try:
                if image:
                    from PIL import Image as _Im
                    _Im.MAX_IMAGE_PIXELS = None
                    import numpy as _np
                    _gray["arr"] = _np.asarray(_Im.open(image).convert("L"))
            except Exception:
                _gray["arr"] = None
        return _gray["arr"]

    # ПУТЬ К СКАНУ (тег 34878) — ПАТЧИМ ВСЕГДА, ДО цикла по кривым (19.07, отчёт Эдуарда
    # «не находит изображение»). Раньше патч стоял ВНУТРИ цикла и выполнялся только если хоть
    # одна кривая записалась: на листе с пустой выдачей в файле оставался путь из ИСХОДНОЙ рамки,
    # а он в архиве бывает мёртвый (RYBAL_179: `C:\nds\projects\RYBAL_179_MBK_1096\img\...`,
    # такого диска/раскладки нет) ⇒ NeuraLOG не открывал скан и QC был невозможен ИМЕННО ТАМ,
    # где он нужнее всего — на листах, где мы ничего не выдали.
    if image:
        for i in find_ifd(ifds, lambda tags: 34878 in tags):
            set_tag(ifds, i, 34878, 2, str(image))
    written, wrote_ifd = [], set()
    for c in model.get("curves", []):
        name = c["name"]
        if name not in mapping:
            continue
        L, tr = mapping[name]
        top_y = c["top_y"]; n = c["n_rows"]
        new_xs = [int(round(tr[top_y + i])) if (top_y + i) in tr else NULL for i in range(n)]
        if sum(1 for x in new_xs if x != NULL) < 30:
            continue
        idxs = find_ifd(ifds, curve_pred(name))
        if not idxs:
            continue
        k = idxs[0]
        set_tag(ifds, k, 35490, 4, new_xs)
        # сегменты масштаба: decode_levels раскладывает 5х-перевыносы, ЕСЛИ каркас несёт цепочку
        # масштабов (base→next); иначе один сегмент level 0. Без 35492>0 NeuraLOG НЕ рисует кривую.
        _g = _get_gray() if _is_mbk_name(name) else None
        tops, bots, levs = _level_segments(tr, model, c, top_y, n, gray=_g)
        set_tag(ifds, k, 35492, 4, [len(tops)]); set_tag(ifds, k, 35494, 4, tops)
        set_tag(ifds, k, 35496, 4, bots); set_tag(ifds, k, 35498, 4, levs)
        wrote_ifd.add(k)
        vx = [(i, x) for i, x in enumerate(new_xs) if x != NULL]
        rws = [top_y + i for i, _ in vx]; xsv = [x for _, x in vx]
        set_tag(ifds, k, 35478, 4, [min(xsv)]); set_tag(ifds, k, 35480, 4, [min(rws)])
        set_tag(ifds, k, 35482, 4, [max(xsv)]); set_tag(ifds, k, 35484, 4, [max(rws)])
        written.append(name.split()[0])

    # ⚠ ЧУЖИЕ ТРАССЫ ИЗ РАМКИ — ВЫЧИСТИТЬ (19.07). Инцидент QC: Эдуард разбирал MBK в нашем
    # KREMEN_089 и находил её «странной» — а файл побайтово повторял ЕГО ЖЕ рамку, мы там не
    # записали НИЧЕГО (U1 нашёл 0 линий). Смешанный файл неотличим на глаз: эксперт рецензирует
    # собственную трассу, считая её нашей, и тратит время впустую. Наша выдача обязана содержать
    # ТОЛЬКО нашу работу. Слот, который мы не заполнили, отдаём ПУСТЫМ: xs=NULL и 35492=0
    # (без сегментов NeuraLOG кривую не рисует). В продакшне рамка ЛЁГКАЯ (трасс нет) — там это
    #но-оп; эффект только на тестах по архивным рамкам, где трассы эксперта присутствуют.
    cleared = []
    for c in model.get("curves", []):
        if meta_mod.mnem_root(c.get("name", "")) == "DA":       # ось глубин — не трасса, не трогаем
            continue
        idxs = find_ifd(ifds, curve_pred(c["name"]))
        if not idxs or idxs[0] in wrote_ifd:
            continue
        k = idxs[0]
        if not any(x != NULL for x in c["xs"]):                 # и так пустой
            continue
        set_tag(ifds, k, 35490, 4, [NULL] * c["n_rows"])
        set_tag(ifds, k, 35492, 4, [0])
        cleared.append(c["name"].split()[0])

    data = write_full(ifds)
    dst = out / f"{stem}_auto.nlgx"
    open(dst, "wb").write(data)
    # .bck = ТОЧНАЯ копия .nlgx (предыдущее сохранение NeuraLOG). Старый write_bck флипал байт
    # на off=5489 — артефакт одного файла, в других попадает ВНУТРЬ тега 35490 и молча портит
    # трассу (write_nlgx docstring, депрекейт).
    open(out / f"{stem}_auto.bck", "wb").write(data)
    res = {"nlgx": str(dst), "written": written, "cleared": cleared}
    if las:
        try:
            import export_las
            export_las.export(str(dst), verbose=False)
            res["las"] = str(dst).replace("_auto.nlgx", "_auto.las")
        except Exception as e:
            res["las_error"] = repr(e)[:120]
    return res


def emit(sheet, traces, out, stem, rgb=None, frame_nlgx=None, mnemonics_path=None,
         image=None, las=False, cv=None):
    """Диспетчер: понимание всегда; nlgx — если дана лёгкая рамка.
    ⚠ `cv` нужен только обученной раскладке (§6.88); None = прежнее поведение, правило."""
    res = emit_understanding(sheet, traces, out, stem, rgb=rgb)
    if frame_nlgx:
        res.update(emit_into_frame(traces, frame_nlgx, sheet.frame, out, stem,
                                   mnemonics_path, image=image, las=las, cv=cv))
    return res

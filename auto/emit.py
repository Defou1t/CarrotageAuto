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
import json, math, os
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
    # ★ 26.09: `CARROTAGE_NO_OVERLAY=1` — не писать overlay.png (выдачу `.nlgx` не меняет). Стенды A/B пишут по оверлею на
    #   КАЖДЫЙ лист каждого режима: к 26.09 их накопилось 18.9 тыс. файлов на 73 ГБ, и не читает их никто (§6.106). Прод —
    #   по умолчанию как прежде.
    if rgb is not None and os.environ.get("CARROTAGE_NO_OVERLAY", "") not in ("1", "true", "yes"):
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
    # ★ §6.215: геометрия шкалы слота из шаблона (полоса, ноль) — запрет пар-нарушителей; None = прод
    from . import slot_geom
    _forbid = slot_geom.make_forbid(model, cv) if cv is not None else None
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
                if _forbid is not None and _forbid(s["curve"]["name"], tr):
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


_PICK = {}


def _pick_feats(e):
    """Восемь признаков ТРЕКА для выбора пути (§6.145). Считаются РОВНО из двух раскладок, то есть
    из того, что известно `emit` в момент решения.

    ⚠⚠ ЭТО НЕ ПРИДИРКА, А ЦЕНА В 14 КРИВЫХ. §6.146: девятый признак модели (`n_lines` — все
    AUTO-линии трека) берётся из `_understanding.json`, а здесь видны только НАЗНАЧЕННЫЕ слоты.
    Модель переобучена без него: держанный счёт упал с 1129 до 1127, то есть признак ничего не
    стоил, — но пока он был в списке, реализовать выбор «как померено» было НЕЛЬЗЯ."""
    import numpy as np
    D, P = e["dec"], e["prod"]
    dif = []
    pd_ = dict(D)
    for nm, t in P:
        u = pd_.get(nm)
        if u:
            com = [y for y in u if y in t]
            if len(com) >= 30:
                dif.append(float(np.median([abs(u[y] - t[y]) for y in com])))
    lb = [len(t) for _, t in D] or [0]
    lp = [len(t) for _, t in P] or [0]
    xs = [x for _, t in D for x in t.values()]
    span = max(1.0, (max(xs) - min(xs)) if xs else 1.0)
    med = [float(np.median(list(t.values()))) for _, t in D]
    return dict(
        n_dec=float(len(D)), n_prod=float(len(P)),
        agree_med=float(np.log1p(np.median(dif))) if dif else float(np.log1p(1000.0)),
        agree_frac=float(np.mean([x <= 3 for x in dif])) if dif else 0.0,
        cov_dec=float(np.median(lb)), cov_prod=float(np.median(lp)),
        len_ratio=float(np.median(lb) / max(1.0, np.median(lp))),
        spread=float(np.log1p(np.std(med) / span)) if len(med) > 1 else 0.0)


def _pick_score(name, e):
    """→ вероятность «брать декодер» по весу `auto/models/<name>` (numpy, без torch и sklearn).
    ⚠ Отсутствие веса — ГРОМКАЯ ошибка, а не тихий откат к порогу: молчаливая подмена режима уже
    стоила ветке полугода споров (§6.68)."""
    import numpy as np
    if name not in _PICK:
        p = Path(__file__).resolve().parent / "models" / name
        if not p.is_file():
            raise FileNotFoundError(f"rowdec_pick_model={name!r} — веса нет: {p}")
        d = np.load(str(p), allow_pickle=False)
        _PICK[name] = (d["w"], float(d["b0"]), d["mu"], d["sd"], [str(x) for x in d["feats"]])
    w, b0, mu, sd, feats = _PICK[name]
    f = _pick_feats(e)
    z = (np.array([f[k] for k in feats], float) - mu) / sd
    return float(1 / (1 + np.exp(-(z @ w + b0))))


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


def _name_style_pass(spec, model, frame, mapping, get_gray, get_rgb, sheet_name, out, stem):
    """§6.273: стиль каждой кривой (вдоль трассы, как её запишет цикл ниже) → парная модель «какой стиль какому имени» в
    треке → перестановки трасс между слотами (меняется `mapping` на месте). Перестановки — в `<stem>_namestyle.json`."""
    from . import name_style as NS, level_ink as LI
    if not NS.available(spec):
        NS._announce(f"⚠ имена по стилю линии ({spec}) НЕ включены: нет sklearn или модели — раскладка как есть")
        return
    gray, rgb = get_gray(), get_rgb()
    if gray is None or rgb is None:
        return
    paper = LI.paper_of(gray)
    entries = []
    for c in model.get("curves", []):
        name = c["name"]
        if name not in mapping or meta_mod.mnem_root(name) == "DA":
            continue
        tr = mapping[name][1]
        top_y = c["top_y"]; n = c["n_rows"]
        new_xs = [int(round(tr[top_y + i])) if (top_y + i) in tr else NULL for i in range(n)]
        st = None
        if sum(1 for x in new_xs if x != NULL) >= 30:
            ty, tx = LI.dense_rows(top_y, new_xs, NULL)
            if len(ty):
                st = NS.style_along(rgb, gray, paper, ty[::8], tx[::8])
        entries.append((name, _slot_track(model, c, frame), st))
    path = NS.resolve(spec, sheet=sheet_name)
    if path is None:
        return
    sw = NS.swaps(entries, sheet_name, path, meta_mod.mnem_root)
    for A, B, p in sw:
        mapping[A], mapping[B] = mapping[B], mapping[A]
    if sw:
        try:
            (Path(out) / f"{stem}_namestyle.json").write_text(json.dumps(
                [{"A": A, "B": B, "p_order_ok": round(p, 4)} for A, B, p in sw], ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


def _line_choice_pass(spec, model, mapping, traces, get_gray, get_rgb, sheet_name, out, stem):
    """§6.280: модель выбора трассы для слота (auto/line_choice.py) — замена трассы слота на свободного кандидата ведения
    (меняется `mapping` на месте). Замены — в `<stem>_linechoice.json`."""
    from . import line_choice as LC, name_style as NS, level_ink as LI
    if not LC.available(spec):
        LC._announce(f"⚠ выбор трассы для слота ({spec}) НЕ включён: нет sklearn или модели — раскладка как есть")
        return
    try:
        import decode_levels as DL
    except Exception:
        return
    gray, rgb = get_gray(), get_rgb()
    if gray is None or rgb is None:
        return
    paper = LI.paper_of(gray)
    raw = [("prod", L, tr) for L, tr in list(traces or [])] + [("dec", L, tr) for L, tr in list(getattr(traces, "alt", None) or [])]
    cands = []
    for src, L, tr in raw:
        d = LC.dense_tr(tr)
        if len(d) < 50:
            continue
        ys = np.array(sorted(d)); xs = np.array([d[y] for y in ys])
        cands.append(dict(src=src, L=L, tr=tr, dense=d, style=NS.style_along(rgb, gray, paper, ys[::8], xs[::8]),
                          rough=LC.rough_robust(xs)))
    if not cands:
        return
    slots, written = [], []
    for c in model.get("curves", []):
        name = c["name"]
        if meta_mod.mnem_root(name) == "DA":
            continue
        if name in mapping:
            tr = mapping[name][1]; top_y = c["top_y"]
            w = {top_y + i: int(round(tr[top_y + i])) for i in range(c["n_rows"]) if (top_y + i) in tr}
            written.append((name, LC.dense_tr(w)))
        fam = DL.build_family(model, c)
        if fam:
            xl = min(min(s["x_left"], s["x_right"]) for s in fam); xr = max(max(s["x_left"], s["x_right"]) for s in fam)
            slots.append((name, c["top_y"], c["n_rows"], (xl, xr)))
    path = LC.resolve(spec, sheet=sheet_name)
    if path is None:
        return
    rep_ = LC.choose(sheet_name, slots, cands, written, path, meta_mod.mnem_root)
    for name, (k, p) in rep_.items():
        if k is None:                                   # §6.283: трассу слота забрал другой слот, своей нет — слот пуст
            mapping.pop(name, None)
        else:
            mapping[name] = (cands[k]["L"], cands[k]["tr"])
    if rep_:
        try:
            (Path(out) / f"{stem}_linechoice.json").write_text(json.dumps(
                [{"slot": n, "k": k, "src": None if k is None else cands[k]["src"], "p": round(p, 4)} for n, (k, p) in rep_.items()],
                ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


def _level_ink_pass(spec, model, ifds, ink_jobs, get_gray, set_tag, sheet_name, shift=False):
    """§6.266: переразложить уровни масштаба кривых с цепочкой моделью по скану (auto/level_ink.py). Чужие трассы — как в
    готовом файле: записанные нами + DA (прочие слоты рамки вычищены).
    §6.267 (`shift`): у СДВИГОВЫХ цепочек (термометрия) — уровень по гладким участкам трассы, без скана и без torch;
    участков нет — как у прочих."""
    from . import level_ink as LI
    try:
        import decode_levels as DL
    except Exception:
        return
    jobs = []
    for c, k, new_xs, lw_segs in ink_jobs:
        fam = DL.build_family(model, c)
        if len(fam) < 2:
            continue
        if shift and LI.is_shift(fam):
            ty, tx = LI.dense_rows(c["top_y"], new_xs, NULL)
            lv = LI.decode_shift(ty, tx, fam) if len(ty) >= 2 else None
            if lv is not None:
                tops, bots, levs = LI.segments_from_changes(lv, c["top_y"], c["n_rows"])
                set_tag(ifds, k, 35492, 4, [len(tops)]); set_tag(ifds, k, 35494, 4, tops)
                set_tag(ifds, k, 35496, 4, bots); set_tag(ifds, k, 35498, 4, levs)
                continue
        jobs.append((c, k, new_xs, lw_segs, fam))
    if not jobs or not spec:
        return
    if not LI.available(spec):
        LI._announce(f"⚠ уровни масштаба по скану ({spec}) НЕ включены: нет torch или весов — остаётся декодер оборотов")
        return
    gray = get_gray()
    if gray is None:
        LI._announce("⚠ уровни масштаба по скану: скан не загрузился — остаётся декодер оборотов")
        return
    paper = LI.paper_of(gray)
    others = []
    for c, k, new_xs, _ in ink_jobs:
        rows = [c["top_y"] + i for i, x in enumerate(new_xs) if x != NULL]
        if len(rows) >= 2:
            others.append((c["name"], np.array(rows, np.int64), np.array([x for x in new_xs if x != NULL], np.float64)))
    for c in model.get("curves", []):
        if meta_mod.mnem_root(c.get("name", "")) == "DA":
            pts = [(c["top_y"] + i, x) for i, x in enumerate(c["xs"]) if x != NULL]
            if len(pts) >= 2:
                others.append((c["name"], np.array([p_[0] for p_ in pts], np.int64), np.array([p_[1] for p_ in pts], np.float64)))
    path = LI.resolve(spec, sheet=sheet_name)
    for c, k, new_xs, lw_segs, fam in jobs:
        ty, tx = LI.dense_rows(c["top_y"], new_xs, NULL)
        f = LI.features(gray, paper, fam, c["name"], meta_mod.mnem_root(c["name"]), ty, tx, lw_segs, others)
        if f is None:
            continue
        lv = LI.predict(f, path)
        tops, bots, levs = LI.segments(f, lv, c["top_y"], c["n_rows"])
        set_tag(ifds, k, 35492, 4, [len(tops)]); set_tag(ifds, k, 35494, 4, tops)
        set_tag(ifds, k, 35496, 4, bots); set_tag(ifds, k, 35498, 4, levs)


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
    def _map(tr, path="prod"):
        m = None
        if cv is not None and getattr(cv, "slot_model", ""):
            from . import slot_model as slot_mod
            m = slot_mod.map_lines(tr, model, frame, mnemonics_path, cv, path=path)
        return m if m is not None else _map_lines_to_slots(tr, model, frame, mnemonics_path, cv)

    mapping = _map(traces)
    # ★★ ВЫБОР ПУТИ ВЕДЕНИЯ ПО ТРЕКУ (§6.144-§6.145), `cv.rowdec_pick`. Считается ЗДЕСЬ, а не в
    # `trace2d`, и это не вопрос вкуса — это правка ошибки, которую поймал прогон.
    # ⚠⚠ ЧТО БЫЛО. Первая редакция решала в `trace2d._pick_by_track` по числу линий, которые
    # декодер ПОВЁЛ. Признак же, на котором мерился выигрыш, — сколько кривых он НАПИСАЛ, а это
    # величина ПОСЛЕ раскладки: линия, не получившая слота, ведётся, но не пишется. На 137 листах
    # прогона выбор разошёлся с посчитанным на 70 треках из 153, и цена расхождения — ровно
    # 14 кривых из ожидавшихся 20 (+6 вместо +20). Числа сошлись до единицы, то есть дело было
    # именно в подмене признака, а не в «стенд врёт».
    # ⇒ Раскладка считается для ОБОИХ путей (она дешёвая, дорого ведение), и трек берёт тот путь,
    # у которого назначено не больше `rowdec_pick` слотов у декодера.
    alt = getattr(traces, "alt", None)
    pick = int(getattr(cv, "rowdec_pick", 0) or 0) if cv is not None else 0
    model_f = (getattr(cv, "rowdec_pick_model", "") or "") if cv is not None else ""
    # ★ §6.213: выбор ПО СЛОТУ по длине (`rowdec_slot_len`); за предгейтом (`alt_gated=False`)
    #   выбор по треку не действует — опора там прод, как и без ручки.
    slot_len = float(getattr(cv, "rowdec_slot_len", 0.0) or 0.0) if cv is not None else 0.0
    slot_fill = bool(getattr(cv, "rowdec_slot_fill", False)) if cv is not None else False
    alt_gated = bool(getattr(traces, "alt_gated", True))
    # ★ §6.237: разрывы трасс ДЕКОДЕРА ≤ `rowdec_gapfill` строк заполняются линейно (0 = выкл = прежнее бит-в-бит). С
    #   удержанием (`rowdec_hold`) строки на пересечениях в трассу не пишутся — трасса короче, и правило слота по длине
    #   (ниже) отдаёт слот проду чаще, чем надо. Чистая функция трассы ⇒ проверяется повтором с кэша.
    gapfill = int(getattr(cv, "rowdec_gapfill", 0) or 0) if cv is not None else 0
    if alt is not None and gapfill > 0:
        alt = [(L, _fill_gaps(tr, gapfill)) for L, tr in alt]
    # ★ §6.240 ВЕТО НЕУВЕРЕННОЙ ВЕРСИИ ДЕКОДЕРА (`cv.rowdec_conf_veto`, 0 = выкл = прежнее бит-в-бит). Уверенность трассы
    #   декодера — медиана вероятности его карты вдоль трассы (`traces.alt_conf`, по порядку `alt`). Стенд §6.240: ни один
    #   честный путь не имел медианы < 0.8 (AUC 0.95 против 0.69 у длины, по которой решает правило слота). Слот, получивший
    #   версию декодера с уверенностью ниже порога, берёт прод-версию, если она есть.
    veto = float(getattr(cv, "rowdec_conf_veto", 0.0) or 0.0) if cv is not None else 0.0
    # ★ §6.241: и обратное — УВЕРЕННАЯ версия декодера (медиана p ≥ `rowdec_conf_take`) берёт слот у прод-версии
    #   независимо от длины (0 = выкл). Точность уверенных путей на стенде — 65% при 0.9, поэтому только повтором.
    take = float(getattr(cv, "rowdec_conf_take", 0.0) or 0.0) if cv is not None else 0.0
    alt_conf = getattr(traces, "alt_conf", None)
    cmap = {}
    if (veto > 0 or take > 0) and alt is not None and alt_conf is not None and len(alt_conf) == len(alt):
        cmap = {id(tr): (c[0] if c else None) for (L, tr), c in zip(alt, alt_conf)}
    if alt is not None and (pick > 0 or model_f or slot_len > 0):
        m_alt = _map(alt, "dec")                    # §6.231: путь — признак расширенной раскладки (вес на 14 его не видит)
        # ⚠⚠ ДВА СЧЁТА НАЗНАЧЕННЫХ СЛОТОВ, И ЭТО НЕ ИЗБЫТОЧНОСТЬ.
        #   `n_all`  — ВСЕ назначенные слоты трека. Именно на нём проверено тождество механизма
        #              офлайновому расчёту (§6.146: 170 = 170 на 137 листах, 45 = 45 на 12), и
        #              менять его без нового доказательства нельзя.
        #   списки   — только слоты с НЕПУСТОЙ трассой: на них считались признаки модели (§6.145),
        #              а пустых в выдаче 6.2%, так что разница не косметическая.
        # Порог берёт первое, модель — второе. Каждый работает ровно на том, на чём померен.
        by_t = {}
        # ★ §6.206: при `rowdec_k_slots` декодер выдаёт траектории сверх линий U1, и им даются
        #   синтетические линии (`rowdec.SYNTH` в `flag_reason`). Порог `n_dec ≤ pick` калибровался,
        #   когда n_dec был ограничен K_U1 (§6.157); с синтетическими он завышается ровно на тех
        #   треках, где ручка добавила кривые, и на 24 треках из 125 (§6.205) ВЫКЛЮЧАЛ декодер.
        #   `rowdec_k_slots_pick = "u1"` — синтетические в порог не считать. Умолчание "count".
        _skip_synth = (getattr(cv, "rowdec_k_slots_pick", "count") if cv is not None else "count") == "u1"
        for src, mp in (("dec", m_alt), ("prod", mapping)):
            for nm, (L, t) in mp.items():
                e = by_t.setdefault(L.track_index,
                                    {"dec": [], "prod": [], "n_all": {"dec": 0, "prod": 0}})
                if not (_skip_synth and src == "dec" and getattr(L, "flag_reason", None) == "kslots-synth"):
                    e["n_all"][src] += 1
                if t:
                    e[src].append((nm, t))
        use = {}
        for ti, e in by_t.items():
            if not alt_gated:
                use[ti] = False               # §6.213: лист за предгейтом — по треку всегда прод
            elif model_f:
                use[ti] = _pick_score(model_f, e) >= 0.5
            else:
                use[ti] = 0 < e["n_all"]["dec"] <= pick
                # ★ §6.206 (только при "u1"): трек, где у прода НЕТ ни одного слота (линий U1 не
                #   было, декодер вёл по всей рамке), берёт декодер — иначе слоты трека уходят в
                #   пустой прод-путь и лист теряет кривые, которых прод дать не может.
                if _skip_synth and e["n_all"]["prod"] == 0 and any(
                        getattr(L, "flag_reason", None) == "kslots-synth" for L, _ in
                        (m_alt[nm] for nm in m_alt if m_alt[nm][0].track_index == ti)):
                    use[ti] = True
        merged, took = {}, {"декодер": 0, "прод": 0}
        # ★ §6.213: окно слота из каркаса — та же величина, что пишется в файл (`xs[i] = tr[top_y+i]`
        #   либо NULL, см. запись ниже): непустых строк = |строки трассы ∩ [top_y, top_y+n_rows)|.
        _win = {c["name"]: (int(c["top_y"]), int(c["n_rows"])) for c in model.get("curves", [])
                if "top_y" in c and "n_rows" in c}
        _slots = []
        _veto_nm, _take_nm, _dd_nm = set(), set(), set()

        def _nn(nm, t):
            w = _win.get(nm)
            if w is None:
                return len(t)
            return sum(1 for y in t if w[0] <= y < w[0] + w[1])
        for nm in set(mapping) | set(m_alt):
            ti = (m_alt.get(nm) or mapping.get(nm))[0].track_index
            src = m_alt if use.get(ti) else mapping
            if nm in src:
                merged[nm] = src[nm]
            oth = mapping if use.get(ti) else m_alt
            # ★ 26.09 (аудит, §6.223): `rowdec_slot_fill` — слот, который базовый путь оставил ПУСТЫМ, а второй путь заполнил,
            #   проходит то же правило с nb = 0 (прежде такой слот уходил в выдачу пустым — правило его не видело). Выкл = прежнее.
            if slot_len > 0 and (nm in src or (slot_fill and nm in oth)):
                nb = _nn(nm, src[nm][1]) if nm in src else 0
                na = _nn(nm, oth[nm][1]) if nm in oth else 0
                # ⚠ кандидат короче 30 строк не берётся: `emit` такую кривую не пишет, и в стенде
                #   его нет (`have` = написанные кривые)
                flip = na >= 30 and (math.log1p(na) - math.log1p(nb)) >= slot_len
                if flip:
                    merged[nm] = oth[nm]
                _slots.append({"name": nm, "track": int(ti), "base": "dec" if use.get(ti) else "prod",
                               "nn_base": int(nb), "nn_alt": int(na), "flip": bool(flip)})
            if veto > 0 and cmap and nm in m_alt and nm in mapping and merged.get(nm) is m_alt[nm]:
                dc = cmap.get(id(m_alt[nm][1]))
                if dc is not None and dc < veto:
                    merged[nm] = mapping[nm]          # §6.240: неуверенная версия декодера уступает прод-версии
                    _veto_nm.add(nm)
            if take > 0 and cmap and nm in m_alt and merged.get(nm) is not m_alt[nm] and len(m_alt[nm][1]) >= 30:
                dc = cmap.get(id(m_alt[nm][1]))
                if dc is not None and dc >= take:
                    merged[nm] = m_alt[nm]            # §6.241: уверенная версия декодера берёт слот
                    _take_nm.add(nm)
        for ti in by_t:
            took["декодер" if use.get(ti) else "прод"] += 1
        print(f"  выбор пути по треку ({'модель ' + model_f if model_f else 'порог ' + str(pick)}"
              f"{'' if alt_gated else ', лист за предгейтом — прод'}): "
              f"декодер на {took['декодер']} треках, прод на {took['прод']}"
              + (f"; по слоту (длина ≥ {slot_len}): перевёрнуто {sum(1 for q in _slots if q['flip'])} "
                 f"из {len(_slots)}" if slot_len > 0 else ""))
        # ⚠⚠ ВЫГРУЗКА ПРИЗНАКОВ РЯДОМ С ВЫДАЧЕЙ — НЕ ОТЛАДКА, А ЕДИНСТВЕННЫЙ СПОСОБ УЧИТЬ МОДЕЛЬ
        # НА ТОМ, ЧТО ОНА УВИДИТ. §6.146 повторился второй раз за день: признаки §6.145 считались
        # по ВЫДАННЫМ кривым (после нарезки уровней), а здесь доступны только трассы раскладки, и
        # решения разошлись — стенд взял бы декодер на 9 треках из 17, механизм взял на 17.
        # Пока признаки не выгружены отсюда, «обучить и внедрить» неразрешимо в принципе:
        # обучающая выборка живёт в одном пространстве, а механизм — в другом.
        # ★ Пишется ВСЕГДА, когда выбор работает: файл крошечный, а прогон становится проверяемым.
        # ★ §6.228: ДВЕ КРИВЫЕ ТРЕКА НА ОДНОЙ ЛИНИИ (`rowdec_dedup` — доля общих строк в 3 px, выше которой пара — дубль;
        #   0 = выкл, прод бит-в-бит). Замер 26.09 (`_dup_traces.py`): на поле 447 треков с дублями, в 476 парах одна кривая
        #   занимает место, а эталон трека не взят. Слот пары берёт версию ДРУГОГО пути, если она сама ни с кем в треке не
        #   дублируется и не короче 30 строк в окне слота.
        dedup = float(getattr(cv, "rowdec_dedup", 0.0) or 0.0) if cv is not None else 0.0
        refill = bool(getattr(cv, "rowdec_refill", False)) if cv is not None else False
        if dedup > 0:
            def _close(t1, t2):
                com = [y for y in t1 if y in t2]
                if len(com) < 200:
                    return 0.0
                return sum(1 for y in com if abs(t1[y] - t2[y]) <= 3) / max(1, min(len(t1), len(t2)))
            _bt = {}
            # ⚠ 27.09 (разбор, 3-й круг): порядок — по имени, а не по `set` (он зависит от хэш-сида процесса, и пара
            #   дублей решалась по-разному от прогона к прогону); прод не задет — блок выключен (`rowdec_dedup = 0`)
            for nm in sorted(merged):
                _bt.setdefault(merged[nm][0].track_index, []).append(nm)
            n_sw = n_rf = 0
            for ti, names in _bt.items():
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        s1, s2 = names[i], names[j]
                        if _close(merged[s1][1], merged[s2][1]) < dedup:
                            continue
                        # ★ §6.229 (`rowdec_refill`): сначала — СВОБОДНАЯ линия трека (не взятая ни одним слотом), самая
                        #   длинная в окне слота и ни с кем не дублирующая; §6.229: у 182 потерянных кривых честная трасса
                        #   есть, но линия не получила слота, а слоты трека заняты другими (часто дублями).
                        if refill:
                            used = {id(merged[o][0]) for o in merged}
                            pool = [(L, t) for L, t in list(traces) + list(alt or [])
                                    if L.track_index == ti and id(L) not in used]
                            done_pair = False
                            for s in (s2, s1):
                                best = None
                                for L, t in pool:
                                    nn = _nn(s, t)
                                    if nn < 30 or any(_close(t, merged[o][1]) >= 0.2 for o in names):
                                        continue
                                    if best is None or nn > best[0]:
                                        best = (nn, (L, t))
                                if best:
                                    merged[s] = best[1]; n_sw += 1; n_rf += 1; done_pair = True
                                    _dd_nm.add(s)
                                    break
                            if done_pair:
                                continue
                        for s in (s2, s1):
                            alt_v = m_alt.get(s) if merged[s] is mapping.get(s) else mapping.get(s)
                            if alt_v is None or alt_v is merged[s] or _nn(s, alt_v[1]) < 30:
                                continue
                            if any(_close(alt_v[1], merged[o][1]) >= 0.2 for o in names if o != s):
                                continue
                            merged[s] = alt_v; n_sw += 1
                            _dd_nm.add(s)
                            break
            print(f"  дубли трека (≥ {dedup:.0%} строк в 3 px): заменено слотов {n_sw}, из них свободной линией {n_rf}")
        # ⛔ 27.09 (разбор, 2-й круг): итог источника — ПОСЛЕ вето/взятия и блока дублей (§6.228 меняет слоты позже);
        #   `src` — что реально записано, `veto`/`take`/`dedup` — что сработало; вето/взятие вне правила слота
        #   (`slot_len == 0`, записей слотов нет) — в `override` {слот: источник}.
        _alt_ids = {id(t) for _, t in (alt or [])}

        def _src_of(nm_):
            v = merged.get(nm_)
            if v is None:
                return "none"
            if nm_ in m_alt and v is m_alt[nm_]:
                return "dec"
            if nm_ in mapping and v is mapping[nm_]:
                return "prod"
            return "dec" if id(v[1]) in _alt_ids else "prod"          # свободная линия добора (§6.229)
        # §6.245: индексы ОБЕИХ версий слота (прод — в `traces`, декодер — в `alt`) — для разбора выбора версии по кэшу
        _pidx = {id(t): i for i, (_, t) in enumerate(traces)}
        _didx = {id(t): i for i, (_, t) in enumerate(alt or [])}
        for q in _slots:
            nm_ = q["name"]
            q["pi"] = _pidx.get(id(mapping[nm_][1])) if nm_ in mapping else None
            q["di"] = _didx.get(id(m_alt[nm_][1])) if nm_ in m_alt else None
            q["src"] = _src_of(nm_)
            if nm_ in _veto_nm:
                q["veto"] = True
            if nm_ in _take_nm:
                q["take"] = True
            if nm_ in _dd_nm:
                q["dedup"] = True
        _in_slots = {q["name"] for q in _slots}
        _ovr = {nm_: _src_of(nm_) for nm_ in (_veto_nm | _take_nm | _dd_nm) if nm_ not in _in_slots}
        try:
            import json as _json
            (out / f"{stem}_pick.json").write_text(_json.dumps(
                {"pick": pick, "model": model_f, "gated": alt_gated, "slot_len": slot_len,
                 "tracks": [{"track": int(ti), "dec": bool(use.get(ti)),
                             "n_all": e["n_all"], **_pick_feats(e)} for ti, e in by_t.items()],
                 **({"slots": _slots} if slot_len > 0 else {}),
                 **({"override": _ovr} if _ovr else {})},
                ensure_ascii=False), encoding="utf-8")
        except Exception as _e:                    # выгрузка не должна ронять выдачу
            print(f"  ⚠ признаки выбора не выгружены: {_e}")
        mapping = merged
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

    _rgb = {"arr": None, "loaded": False}

    def _get_rgb():
        if not _rgb["loaded"]:
            _rgb["loaded"] = True
            try:
                if image:
                    from PIL import Image as _Im
                    _Im.MAX_IMAGE_PIXELS = None
                    import numpy as _np
                    _rgb["arr"] = _np.asarray(_Im.open(image).convert("RGB"))
            except Exception:
                _rgb["arr"] = None
        return _rgb["arr"]

    # ПУТЬ К СКАНУ (тег 34878) — ПАТЧИМ ВСЕГДА, ДО цикла по кривым (19.07, отчёт Эдуарда
    # «не находит изображение»). Раньше патч стоял ВНУТРИ цикла и выполнялся только если хоть
    # одна кривая записалась: на листе с пустой выдачей в файле оставался путь из ИСХОДНОЙ рамки,
    # а он в архиве бывает мёртвый (RYBAL_179: `C:\nds\projects\RYBAL_179_MBK_1096\img\...`,
    # такого диска/раскладки нет) ⇒ NeuraLOG не открывал скан и QC был невозможен ИМЕННО ТАМ,
    # где он нужнее всего — на листах, где мы ничего не выдали.
    if image:
        for i in find_ifd(ifds, lambda tags: 34878 in tags):
            set_tag(ifds, i, 34878, 2, str(image))
    # ★★ 04.10 (§6.273): ИМЕНА ТРЕКА ПО СТИЛЮ ЛИНИИ (цвет, толщина, штрих) — перестановка трасс между слотами ДО записи, чтобы
    #   уровни масштаба считались уже для нового слота. Стенд: поле +25 именных (p = 0.0048), сорт A +2.
    _cvp0 = cv
    if _cvp0 is None:
        from .config import DEFAULT as _D0
        _cvp0 = _D0.cv
    _nsp = getattr(_cvp0, "name_style", "") or ""
    if _nsp:
        _name_style_pass(_nsp, model, frame, mapping, _get_gray, _get_rgb, Path(frame_nlgx).name, out, stem)
    # ★★ 04.10 (§6.280): ВЫБОР ТРАССЫ ДЛЯ СЛОТА среди кандидатов ведения (прод-путь + декодер) — после перестановки имён, как
    #   на стенде. Стенд: поле +24 (25 / 1), сорт A +8.
    _lcp = getattr(_cvp0, "line_choice", "") or ""
    if _lcp:
        _line_choice_pass(_lcp, model, mapping, traces, _get_gray, _get_rgb, Path(frame_nlgx).name, out, stem)
    written, wrote_ifd = [], set()
    ink_jobs = []                       # §6.266: (кривая, ifd, xs, сегменты прежнего декодера) — для второго прохода
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
        ink_jobs.append((c, k, new_xs, list(zip(tops, bots, levs))))
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

    # ★★ 03.10 (§6.266): УРОВНИ МАСШТАБА ПО СКАНУ ВОКРУГ ЛИНИИ — второй проход, когда известны ВСЕ трассы листа (модель видит
    #   занятость полосы чужими трассами). Кривые с цепочкой масштабов в каркасе; остальные и короткие — прежний декодер.
    #   Замер §6.266 v2 (в значениях): поле 683 → 712 (+29, p = 0.0054, вне фолда), сорт A 172 → 181. ⚠ ОТКАТ: level_ink = "".
    _cvp = cv
    if _cvp is None:
        from .config import DEFAULT as _D
        _cvp = _D.cv
    _li = getattr(_cvp, "level_ink", "") or ""
    _ls = bool(getattr(_cvp, "level_shift", False))
    if (_li or _ls) and ink_jobs:
        _level_ink_pass(_li, model, ifds, ink_jobs, _get_gray, set_tag, Path(frame_nlgx).name, shift=_ls)

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


def _fill_gaps(tr, gmax):
    """§6.237: {row: x} с заполненными линейно разрывами ≤ gmax строк (края не продлеваются). Порядок ключей — по строкам."""
    if len(tr) < 2:
        return tr
    ys = sorted(tr)
    out = {}
    for y0, y1 in zip(ys, ys[1:]):
        out[y0] = tr[y0]
        g = y1 - y0
        if 1 < g <= gmax + 1:
            x0, x1 = tr[y0], tr[y1]
            for k in range(1, g):
                out[y0 + k] = x0 + (x1 - x0) * k / g
    out[ys[-1]] = tr[ys[-1]]
    return out

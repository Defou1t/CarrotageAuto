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
             "blue": (60, 120, 255)}


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
    """Какому треку (индекс) принадлежит слот кривой — по x его scale-семейства vs колонки U0."""
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


def _map_lines_to_slots(traces, model, frame, mnemonics_path):
    """Автономный маппинг AUTO-линий → слоты кривых рамки: тот же трек + совместимый цвет/класс +
    порядок слева-направо (идентичность = СЧЁТ, §6.6.11). Возвращает {slot_name: (Line, trace)}."""
    import dataset as ds
    slots = ds.real_curves(model) if model.get("curves") else []
    # слоты по треку, с приором цвета/класса из мнемоники
    slot_info = []
    for c in slots:
        info = meta_mod.curve_info(c["name"], mnemonics_path)
        slot_info.append({"curve": c, "track": _slot_track(model, c, frame),
                          "color": info["color"], "class": info["class"],
                          "root": info["root"]})
    used = set(); mapping = {}
    # внутри трека: сортируем линии и слоты по x, паруем с предпочтением цвет/класс
    for ti in {s["track"] for s in slot_info}:
        tslots = [s for s in slot_info if s["track"] == ti]
        tlines = [(L, tr) for L, tr in traces if L.track_index == ti]
        tlines.sort(key=lambda lt: lt[0].x_center)
        for s in tslots:
            best, bd = None, None
            for L, tr in tlines:
                if id(L) in used:
                    continue
                # совместимость: цвет совпал (если задан) ИЛИ класс по поведению совпал
                color_ok = (s["color"] is None) or (s["color"] == L.color)
                cls = "SP" if L.behavior == "smooth" else "RES"
                class_ok = (s["class"] in (cls, "OTHER", "CALI"))
                score = (0 if color_ok else 2) + (0 if class_ok else 1)
                if best is None or score < bd:
                    best, bd = (L, tr), score
            if best is not None:
                used.add(id(best[0])); mapping[s["curve"]["name"]] = best
    return mapping


def emit_into_frame(traces, frame_nlgx, frame, out, stem, mnemonics_path,
                    image=None, las=False):
    """Инъекция AUTO-трасс в лёгкую рамку NeuraLOG → _auto.nlgx(+bck). Рамка НЕ фабрикуется."""
    import struct
    from extract_nlgx import extract
    from write_nlgx import read_full, write_full, write_bck, set_tag, find_ifd
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model = extract(str(frame_nlgx))
    mapping = _map_lines_to_slots(traces, model, frame, mnemonics_path)
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

    written = []
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
        # одношкальный сегмент level 0 (значения/уровни — эксперт в NeuraLOG, §6.6.9);
        # без 35492>0 NeuraLOG НЕ рисует кривую.
        set_tag(ifds, k, 35492, 4, [1]); set_tag(ifds, k, 35494, 4, [int(top_y)])
        set_tag(ifds, k, 35496, 4, [int(top_y + n - 1)]); set_tag(ifds, k, 35498, 4, [0])
        vx = [(i, x) for i, x in enumerate(new_xs) if x != NULL]
        rws = [top_y + i for i, _ in vx]; xsv = [x for _, x in vx]
        set_tag(ifds, k, 35478, 4, [min(xsv)]); set_tag(ifds, k, 35480, 4, [min(rws)])
        set_tag(ifds, k, 35482, 4, [max(xsv)]); set_tag(ifds, k, 35484, 4, [max(rws)])
        if image:
            for i in find_ifd(ifds, lambda tags: 34878 in tags):
                set_tag(ifds, i, 34878, 2, str(image))
        written.append(name.split()[0])

    data = write_full(ifds)
    dst = out / f"{stem}_auto.nlgx"
    open(dst, "wb").write(data)
    open(out / f"{stem}_auto.bck", "wb").write(write_bck(data))
    res = {"nlgx": str(dst), "written": written}
    if las:
        try:
            import export_las
            export_las.export(str(dst), verbose=False)
            res["las"] = str(dst).replace("_auto.nlgx", "_auto.las")
        except Exception as e:
            res["las_error"] = repr(e)[:120]
    return res


def emit(sheet, traces, out, stem, rgb=None, frame_nlgx=None, mnemonics_path=None,
         image=None, las=False):
    """Диспетчер: понимание всегда; nlgx — если дана лёгкая рамка."""
    res = emit_understanding(sheet, traces, out, stem, rgb=rgb)
    if frame_nlgx:
        res.update(emit_into_frame(traces, frame_nlgx, sheet.frame, out, stem,
                                   mnemonics_path, image=image, las=las))
    return res

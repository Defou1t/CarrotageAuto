"""
Экстрактор калибровки и пиксель-трассы из .nlgx (NeuraLOG).
Структура: multi-IFD TIFF (LE), запись типизирована тегом 34768:
  1=документ/изображение, 2=well-info, 3=параметры, 4=Depth Axis,
  5=Scale Axis (по одной на масштаб/перевынос), 6=описание кривой,
  7=трасса кривой (пиксели!), 8=Depth Grid, 11=EOF.

Ключевые теги:
  Depth Axis (тип 4):
    35168 name, 35170 units, 35184 top_y(px), 35188 bottom_y(px),
    35190 top_depth(dbl), 35192 bottom_depth(dbl), 35194 span_depth,
    35196 span_px, 35182/35186 = x верх/низ оси.
  Scale Axis (тип 5):
    35268 name ("DA1 SA1", " 5X  1_DA1 SA1", ...), 35270 units,
    35292 x_left, 35296 x_right, 35294/35298 y, 35308 px_width(=x_right-x_left),
    35300 v_left, 35302 v_right, 35304/35306 grid v_left/v_right (×5-прогрессия),
    35272 idx семейства, 35273 next-idx (связный список перевыносов).
  Depth Grid (тип 8):
    35590 N линий, 35596 Y-координаты (px) горизонталей, 35578 шаг(м, dbl).
  Трасса (тип 7):
    35470 name ("BK1 DA1 SA1"), 35474 top_y, 35476 bottom_y, 35488 N rows,
    35490 X-по-строкам (LONG, 0xFFFFFFFF=нет данных),
    35492 N сегментов, 35494 seg_start_y, 35496 seg_end_y, 35498 seg_level.
"""
import sys, struct

FMT = {1:("B",1),2:("s",1),3:("H",2),4:("I",4),5:("II",8),6:("b",1),7:("B",1),
       8:("h",2),9:("i",4),10:("ii",8),11:("f",4),12:("d",8),13:("I",4),
       16:("Q",8),17:("q",8),18:("Q",8)}
NULL = 0xFFFFFFFF


def read_ifds(data):
    assert data[:2] == b"II", "expected little-endian TIFF"
    e = "<"
    ifds = []
    off = struct.unpack(e+"I", data[4:8])[0]
    seen = set()
    while off and off not in seen:
        seen.add(off)
        n = struct.unpack(e+"H", data[off:off+2])[0]
        tags = {}
        p = off+2
        for _ in range(n):
            tag, typ, count = struct.unpack(e+"HHI", data[p:p+8])
            fmt, sz = FMT.get(typ, ("B",1))
            total = sz*count
            if typ in (5,10):  # rational: 2 ints per item
                total = sz*count
            if total <= 4:
                raw = data[p+8:p+12][:total]
            else:
                ptr = struct.unpack(e+"I", data[p+8:p+12])[0]
                raw = data[ptr:ptr+total]
            tags[tag] = (typ, count, raw)
            p += 12
        nxt = struct.unpack(e+"I", data[p:p+4])[0]
        ifds.append(tags)
        off = nxt
    return ifds


def val(tags, tag, default=None):
    if tag not in tags:
        return default
    typ, count, raw = tags[tag]
    if typ == 2:
        return raw.split(b"\x00")[0].decode("latin1")
    fmt, sz = FMT[typ]
    vals = struct.unpack("<"+fmt*count, raw[:sz*count])
    return vals[0] if count == 1 else list(vals)


def arr(tags, tag):
    typ, count, raw = tags[tag]
    fmt, sz = FMT[typ]
    return list(struct.unpack("<"+fmt*count, raw[:sz*count]))


def extract(path):
    data = open(path, "rb").read()
    ifds = read_ifds(data)
    model = {"path": path, "scale_axes": [], "curves": []}
    for t in ifds:
        kind = val(t, 34768)
        if kind == 1:
            model["img_path"] = val(t, 34878)
            model["step_m"] = val(t, 34886)
            model["null"] = val(t, 34884)
            model["units"] = val(t, 34888)
        elif kind == 2:
            model["well"] = val(t, 34970)
            model["field"] = val(t, 34974)
            model["top_depth"] = val(t, 35006)
            model["bottom_depth"] = val(t, 35008)
        elif kind == 4:  # Depth Axis
            # ⚠ ЛИСТ МОЖЕТ НЕСТИ НЕСКОЛЬКО ОСЕЙ ГЛУБИН (DA1, DA2, …): замер 19.07 — 100 из 1180
            # планшетов, и у 99 из них оси РАЗЛИЧАЮТСЯ по y-диапазону/глубинам. Присваивание ниже
            # оставляет только ПОСЛЕДНЮЮ, т.е. на этих листах берётся чужая система координат.
            # Здесь копим ВСЕ в depth_axes (аддитивно, поведение не меняется); какую выбирать —
            # отдельный вопрос с гейтом, потому что смена ломает координаты 99 листов молча.
            model.setdefault("depth_axes", []).append({
                "name": val(t, 35168), "units": val(t, 35170),
                "top_y": val(t, 35184), "bottom_y": val(t, 35188),
                "top_depth": val(t, 35190), "bottom_depth": val(t, 35192),
                "span_depth": val(t, 35194), "span_px": val(t, 35196),
                "x_top": val(t, 35182), "x_bot": val(t, 35186),
            })
            model["depth_axis"] = {
                "name": val(t, 35168), "units": val(t, 35170),
                "top_y": val(t, 35184), "bottom_y": val(t, 35188),
                "top_depth": val(t, 35190), "bottom_depth": val(t, 35192),
                "span_depth": val(t, 35194), "span_px": val(t, 35196),
                "x_top": val(t, 35182), "x_bot": val(t, 35186),
            }
        elif kind == 5:  # Scale Axis
            model["scale_axes"].append({
                "name": val(t, 35268), "units": val(t, 35270),
                "idx": val(t, 35272), "next": val(t, 35273),
                "x_left": val(t, 35292), "x_right": val(t, 35296),
                "y": val(t, 35294),
                "v_left": val(t, 35300), "v_right": val(t, 35302),
                "grid_v_left": val(t, 35304), "grid_v_right": val(t, 35306),
                "px_width": val(t, 35308),
            })
        elif kind == 8:  # Depth Grid
            model["depth_grid"] = {
                "n": val(t, 35590), "step_m": val(t, 35578),
                "ys": arr(t, 35596) if 35596 in t else [],
            }
        elif kind == 7:  # Curve trace
            xs = arr(t, 35490) if 35490 in t else []
            segs = []
            if 35492 in t and val(t, 35492) > 0:
                ss = arr(t, 35494); ee = arr(t, 35496); ll = arr(t, 35498)
                segs = list(zip(ss, ee, ll))
            model["curves"].append({
                "name": val(t, 35470),
                "top_y": val(t, 35474), "bottom_y": val(t, 35476),
                "n_rows": val(t, 35488),
                "xs": xs, "segments": segs,
            })
    return model


def depth_of(model, y):
    da = model["depth_axis"]
    sp = da.get("span_px") or 0
    if not sp:  # вырожденная/неполная ось — глубину не восстановить
        return float("nan")
    return da["top_depth"] + (y - da["top_y"]) * da["span_depth"] / sp


def depth_axis_ok(model):
    """Калибровка глубины пригодна (ось присутствует и невырождена)."""
    da = model.get("depth_axis")
    return bool(da) and bool(da.get("span_px")) and da.get("span_depth") not in (None, 0)


def main():
    path = sys.argv[1]
    m = extract(path)
    print(f"== {m.get('well')} / {m.get('field')} ==")
    print(f"img: {m.get('img_path')}")
    print(f"depth {m.get('top_depth')}..{m.get('bottom_depth')} step {m.get('step_m')} units {m.get('units')}")
    da = m["depth_axis"]
    print(f"\nDEPTH AXIS '{da['name']}': y {da['top_y']}..{da['bottom_y']} -> depth {da['top_depth']}..{da['bottom_depth']}")
    print(f"  span {da['span_depth']} m / {da['span_px']} px = {da['span_depth']/da['span_px']:.5f} m/px ; x={da['x_top']},{da['x_bot']}")
    dg = m.get("depth_grid", {})
    if dg:
        ys = dg["ys"]
        print(f"\nDEPTH GRID: {dg['n']} lines, step {dg['step_m']} m; y[0..3]={ys[:4]} ... y[-1]={ys[-1] if ys else '-'}")
        if len(ys) >= 2:
            d0, d1 = depth_of(m, ys[0]), depth_of(m, ys[-1])
            print(f"  grid depth span: {d0:.2f}..{d1:.2f}  (px/4m ~ {(ys[-1]-ys[0])/(len(ys)-1):.2f})")
    print(f"\nSCALE AXES ({len(m['scale_axes'])}):")
    for s in m["scale_axes"]:
        print(f"  idx{s['idx']}->{s['next'] if s['next']!=NULL else 'end':>3}  '{s['name']:<16}' "
              f"x[{s['x_left']}..{s['x_right']}] v[{s['v_left']}..{s['v_right']}] "
              f"grid[{s['grid_v_left']}..{s['grid_v_right']}] {s['units']}")
    print(f"\nCURVES ({len(m['curves'])}):")
    for c in m["curves"]:
        xs = c["xs"]
        valid = [x for x in xs if x != NULL]
        print(f"  '{c['name']}' rows={c['n_rows']} valid_px={len(valid)} "
              f"x_range[{min(valid) if valid else '-'}..{max(valid) if valid else '-'}] segs={len(c['segments'])}")
        for s,e,l in c["segments"][:6]:
            print(f"     seg y{s}..{e} (d {depth_of(m,s):.1f}..{depth_of(m,e):.1f}) level={l}")
        if len(c["segments"]) > 6:
            print(f"     ... {len(c['segments'])} segments, levels={[l for _,_,l in c['segments']]}")


if __name__ == "__main__":
    main()

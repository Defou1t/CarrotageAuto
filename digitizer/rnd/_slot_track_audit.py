"""Аудит ДВУХ дефектов, найденных 19.07 (проверка независимая, не на слово):

[1] emit._slot_track берёт ключ = ПОСЛЕДНИЙ токен имени кривой («GZ11 DA1 SA1» → «SA1»).
    У осей разных DA-семейств хвост одинаков («DA1 SA1» и «DA2 SA1» → обе «SA1»), поэтому
    median(x_left) усредняется по осям РАЗНЫХ колонок и слот уезжает в чужой трек.
    Правильный ключ уже используется в decode_levels.build_family: " ".join(name.split()[1:]).
    Меряем: у скольких кривых трек НЕ СОДЕРЖИТ даже медиану самой кривой — сейчас и с фиксом.

[2] extract_nlgx: `model["depth_axis"] = {...}` в цикле по IFD — на листах с DA1+DA2 выживает
    только ПОСЛЕДНЯЯ. Меряем: сколько листов с >1 Depth Axis и РАСХОДЯТСЯ ли они по
    y-диапазону/глубинам (если совпадают — дефект безвреден на практике, и это надо знать).

  python _slot_track_audit.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL, val, read_ifds
from dataset_build import find_image
from auto import meta as M, frame as F, emit as E, imaging as im
from auto.config import Config

MN = r"F:\nds\Auto\mnemonics.json"
ARCHIVE = Path(r"F:\nds\projects\Archive")
p = Config().cv


def slot_track_fixed(model, curve, frame):
    """Тот же _slot_track, но ключ = ПОЛНЫЙ суффикс имени («DA1 SA1»), как в build_family."""
    full = " ".join(curve["name"].split()[1:])
    xs = [s["x_left"] for s in model["scale_axes"]
          if s.get("name") == full and s.get("x_left") is not None]
    if not xs:                                  # фолбэк на старое поведение
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


bad_now = bad_fix = tot = 0
multi_da = 0
da_differ = Counter()
examples = []
sheets = 0
for wlg in sorted(ARCHIVE.glob("*/wlg")):
    for n in sorted(wlg.glob("*.nlgx")):
        if "_auto" in n.stem:
            continue
        img = find_image(n)
        if not img:
            continue
        try:
            mo = extract(str(n))
        except Exception:
            continue
        sheets += 1
        # [2] сколько Depth Axis в файле и различаются ли они
        try:
            das = [t for t in read_ifds(open(str(n), "rb").read()) if val(t, 34768) == 4]
        except Exception:
            das = []
        if len(das) > 1:
            multi_da += 1
            ys = {(val(t, 35184), val(t, 35188), val(t, 35190), val(t, 35192)) for t in das}
            da_differ["различаются" if len(ys) > 1 else "совпадают"] += 1
        # [1] трек слота: сейчас vs с полным ключом
        try:
            m = M.parse_filename(n.name, MN)
            # rgb НЕ грузим: он нужен frame_from_nlgx только для grid_period, а границы треков
            # берутся из nlgx. Загрузка 1180 сканов превращала аудит в часы.
            fr = F.frame_from_nlgx(str(n), m, p, rgb=None)
        except Exception:
            continue
        if len(fr.tracks) < 2:                  # на 1-трековом листе ошибиться треком нельзя
            continue
        for c in mo["curves"]:
            xs = [x for x in c["xs"] if x != NULL]
            if len(xs) < 50 or M.mnem_root(c["name"]) == "DA":
                continue
            tot += 1
            med = float(np.median(xs))
            for fn, cnt in ((E._slot_track, "now"), (slot_track_fixed, "fix")):
                ti = fn(mo, c, fr)
                t = fr.tracks[ti if ti is not None and ti < len(fr.tracks) else 0]
                inside = t.x_left - 20 <= med <= t.x_right + 20
                if not inside:
                    if cnt == "now":
                        bad_now += 1
                        if len(examples) < 8:
                            examples.append(f"{wlg.parent.name}/{c['name']:<16} медиана x={med:.0f} "
                                            f"трек[{t.x_left},{t.x_right}]")
                    else:
                        bad_fix += 1

print(f"листов прочитано: {sheets}")
print(f"\n[1] ТРЕК СЛОТА (только многотрековые листы), кривых: {tot}")
print(f"    сейчас  — трек НЕ содержит медиану кривой: {bad_now} ({100*bad_now/max(1,tot):.1f}%)")
print(f"    с полным ключом «DA1 SA1»:                 {bad_fix} ({100*bad_fix/max(1,tot):.1f}%)")
for e in examples:
    print("      ", e)
print(f"\n[2] DEPTH AXIS: листов с >1 DA: {multi_da} из {sheets}")
print(f"    из них {dict(da_differ)}")
print("    (если 'совпадают' — потеря лишних DA на практике безвредна)")

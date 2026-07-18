"""СЕЛЕКЦИЯ ТУШИ, шаг 1: ЧТО ИМЕННО лишнее в треке мульти-листа.
Гипотеза: «лишние» раны — это НЕ мусор, а ПЕРЕВЫНОСЫ (×5/×25) тех же кривых. Связь между
шкалами ТОЧНО задана рамкой: у каждой scale-axis есть (x_left,x_right,v_left,v_right), значит
для точки кривой со значением v позиция её двойника на шкале k считается АНАЛИТИЧЕСКИ.

Меряем на каждой строке: сколько ранов объяснено (а) самой GT-точкой, (б) её предсказанным
двойником на другом уровне семейства, (в) не объяснено ничем. Это отвечает, есть ли у селекции
дешёвый геометрический признак или придётся искать другой.

  python _multi_replica_probe.py <nlgx> [<nlgx> ...] [--tol 8]
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from decode_levels import build_family, gt_levels
from auto import imaging as im, meta as M, frame as F, emit as E
from auto.config import Config

MN = r"F:\nds\Auto\mnemonics.json"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
TOL = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 8.0
p = Config().cv


def dense(curve, max_gap=200):
    """⚠ ЭКСПЕРТНАЯ ТРАССА — ЭТО ВЕРШИНЫ ПОЛИЛИНИИ, НЕ ОТСЧЁТ НА СТРОКУ (замер 18.07: точек
    4-14% строк, медианный шаг 6-23 строки). Любое сравнение «ран vs GT» ПО СТРОКАМ обязано
    сначала интерполировать полилинию, иначе на большинстве строк GT просто нет и метрика врёт.
    Разрывы >max_gap строк не мостим (там кривой действительно нет)."""
    pts = [(curve["top_y"] + i, x) for i, x in enumerate(curve["xs"]) if x != NULL]
    out = {}
    for (y0v, x0v), (y1v, x1v) in zip(pts, pts[1:]):
        out[y0v] = float(x0v)
        if 0 < y1v - y0v <= max_gap:
            for yy in range(y0v + 1, y1v):
                out[yy] = x0v + (x1v - x0v) * (yy - y0v) / (y1v - y0v)
    if pts:
        out[pts[-1][0]] = float(pts[-1][1])
    return out


def axis_map(s):
    """пиксель→значение и значение→пиксель для одной scale-axis (линейная шкала)."""
    xl, xr = float(s["x_left"]), float(s["x_right"])
    vl, vr = float(s["v_left"]), float(s["v_right"])
    if xr == xl or vr == vl:
        return None, None
    return (lambda x: vl + (x - xl) * (vr - vl) / (xr - xl),
            lambda v: xl + (v - vl) * (xr - xl) / (vr - vl))


for arg in (args if __name__ == "__main__" else []):   # dense() импортируют другие пробы
    n = Path(arg)
    img = find_image(n)
    mo = extract(str(n))
    m = M.parse_filename(n.name, MN)
    rgb = im.load_rgb(str(img))
    fr = F.frame_from_nlgx(str(n), m, p, rgb=rgb)
    fg = im.ink_foreground(rgb, p)
    gts = [c for c in mo["curves"] if sum(1 for x in c["xs"] if x != NULL) >= 100
           and M.mnem_root(c["name"]) != "DA"]
    tid = Counter(E._slot_track(mo, c, fr) for c in gts)
    best = tid.most_common(1)[0][0]
    gts = [c for c in gts if E._slot_track(mo, c, fr) == best]
    t = fr.tracks[best if best is not None else 0]
    lo, hi = int(t.x_left) + 3, int(t.x_right) - 3
    fam = {c["name"]: build_family(mo, c) for c in gts}
    lev = {c["name"]: gt_levels(c) for c in gts}
    ser = {c["name"]: dense(c) for c in gts}
    print(f"\n{n.stem[:52]}  K={len(gts)} трек[{lo}..{hi}]")
    for c in gts:
        print(f"   {c['name']:<20} уровней в семействе: {len(fam[c['name']])}")
    n_gt = n_rep = n_un = 0
    step = max(1, (int(fr.bottom_y) - int(fr.top_y)) // 2500)
    for y in range(int(fr.top_y), int(fr.bottom_y), step):
        runs = im.row_runs(fg[y, lo:hi], gap=4)
        if not runs:
            continue
        cents = [lo + r[2] for r in runs]
        used = set()
        pred = []                                    # предсказанные позиции двойников
        for c in gts:
            nm = c["name"]
            if y not in ser[nm]:
                continue
            x = ser[nm][y]
            for k, ci in enumerate(cents):           # сама GT-точка
                if abs(ci - x) <= TOL:
                    used.add(k)
            ch = fam[nm]
            if len(ch) < 2:
                continue
            k0 = lev[nm].get(y, 0)
            if k0 >= len(ch):
                continue
            f_v, _ = axis_map(ch[k0])
            if f_v is None:
                continue
            v = f_v(x)                               # значение точки на её текущем уровне
            for kk, s in enumerate(ch):
                if kk == k0:
                    continue
                _, f_x = axis_map(s)
                if f_x:
                    pred.append(f_x(v))              # где ЭТО ЖЕ значение лежит на шкале kk
        for k, ci in enumerate(cents):
            if k in used:
                n_gt += 1
            elif any(abs(ci - px) <= TOL for px in pred):
                n_rep += 1; used.add(k)
            else:
                n_un += 1
    tot = max(1, n_gt + n_rep + n_un)
    print(f"   раны: GT-кривые {100*n_gt/tot:.0f}% | ПЕРЕВЫНОСЫ (предсказаны рамкой) {100*n_rep/tot:.0f}%"
          f" | не объяснено {100*n_un/tot:.0f}%   (всего {tot}, tol={TOL:.0f}px)")

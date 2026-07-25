r"""_levels_vsweep.py — ПОДБОР ДЕКОДЕРА УРОВНЕЙ ПО ЗНАЧЕНИЯМ (широкая выборка, честный сплит).

Почему не по переходам. §6.35 подбирал полноту/точность переходов и поднял её 32%→76%. Но §6.38
показал, что это НЕ ТА ЦЕЛЬ: у OGZ1 ложных переходов единицы, а corr с LAS падает 0.66→0.16,
потому что шкала backup впятеро шире и ОДНА ложная строка даёт выброс ±55 при кривой в ±11.
⇒ **цена ошибки, а не её количество**. Здесь целевая функция — медианная corr значений с LAS.

Дорогая часть (чтение nlgx + LAS + плотная GT) считается ОДИН раз и кэшируется; перебор
параметров потом дешёвый. Сплит ПО СКВАЖИНАМ (не по кривым): соседние кривые одного листа
делят рамку и потому не независимы.

Сравнение всегда против двух опорных точек:
  база0  — «всегда level 0» (не понимать переходы вовсе);
  оракул — уровни эксперта на НАШЕЙ трассе (⚠ §6.38: это НЕ потолок, когда трасса своя).

  python _levels_vsweep.py --sheets 150 [--rebuild]
"""
import sys, io, pickle, argparse, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds
import decode_levels as DL
from dataset_build import find_las
from _multi_replica_probe import dense
from _levels_decode2 import decode2, is_additive
from _decoder_data import train_sheets

CACHE = Path(r"F:\nds\output\taskS\decoder\vsweep_items.pkl")

ap = argparse.ArgumentParser()
ap.add_argument("--sheets", type=int, default=150)
ap.add_argument("--rebuild", action="store_true")
a = ap.parse_args()


def build(sheets):
    items = []
    for n in sheets:
        las = find_las(n)
        if not las:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m = extract(str(n))
                cols, arr = ds.load_las(str(las))
                matches = ds.match_las(m, cols, arr)
        except Exception:
            continue
        da = m.get("depth_axis") or {}
        if not da.get("bottom_y") or da["bottom_y"] == da["top_y"]:
            continue
        ty, by = da["top_y"], da["bottom_y"]; td, bd = da["top_depth"], da["bottom_depth"]
        depths = arr[:, 0]
        for c in ds.real_curves(m):
            fam = DL.build_family(m, c); gl = DL.gt_levels(c)
            ci = matches.get(c["name"], {}).get("col_idx")
            if len(fam) < 2 or not gl or ci is None:
                continue
            d = dense(c)
            exp = {y: d[y] for y in sorted(d) if y in gl}
            if len(exp) < 200:
                continue
            lasv = arr[:, ci]
            rows, xs, lv = [], [], []
            for y, x in exp.items():
                dep = td + (y - ty) * (bd - td) / (by - ty)
                li = int(np.clip(np.searchsorted(depths, dep), 0, len(depths) - 1))
                if np.isfinite(lasv[li]):
                    rows.append(y); xs.append(x); lv.append(float(lasv[li]))
            if len(rows) < 100:
                continue
            sp = [abs(s["v_right"] - s["v_left"]) or 1e-9 for s in fam]
            items.append({"well": n.parent.parent.name, "short": c["name"].split()[0],
                          "fam": fam, "gl": {y: gl[y] for y in rows}, "K": len(fam),
                          "rows": np.array(rows), "xs": np.array(xs, float),
                          "las": np.array(lv, float), "add": is_additive(fam),
                          "wratio": float(np.median([sp[i + 1] / sp[i] for i in range(len(sp) - 1)]))})
        print(f"  {n.name[:44]:<46} всего кривых с LAS: {len(items)}")
    return items


if CACHE.exists() and not a.rebuild:
    items = pickle.loads(CACHE.read_bytes())
    print(f"кэш: {len(items)} кривых")
else:
    items = build(train_sheets(a.sheets))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(items))
    print(f"построено: {len(items)} кривых -> {CACHE}")


def corr_of(it, levels):
    maps = [DL.scale_map(s) for s in it["fam"]]; K = it["K"]
    V = np.array([maps[min(int(levels.get(y, 0)), K - 1)](x)
                  for y, x in zip(it["rows"].tolist(), it["xs"].tolist())])
    if V.std() == 0:
        return float("nan")
    return float(np.corrcoef(V, it["las"])[0, 1])


def score(items_, lam, dxf, gw, bias, rail, auto=True):
    cc = []
    for it in items_:
        exp = dict(zip(it["rows"].tolist(), it["xs"].tolist()))
        dec = decode2(exp, it["fam"], lam, dxf, gw, level_bias=bias, rail_gate=rail, auto=auto)
        c = corr_of(it, dec)
        if c == c:
            cc.append(c)
    return float(np.median(cc)) if cc else float("nan")


wells = sorted({it["well"] for it in items})
TRs = [it for it in items if it["well"] in set(wells[::2])]
VAs = [it for it in items if it["well"] in set(wells[1::2])]
print(f"кривых {len(items)} ({len(wells)} скважин): подбор {len(TRs)} / проверка {len(VAs)}")
print(f"аддитивных {sum(1 for it in items if it['add'])}, "
      f"медиана отношения ширин шкал {np.median([it['wratio'] for it in items]):.2f}")

# nanmedian, а не median: у части кривых corr не определена (LAS-константа на интервале или
# вырожденная трасса). np.median по списку с nan даёт nan и молча обнуляет опорную точку.
b_tr = float(np.nanmedian([corr_of(it, {}) for it in TRs]))
b_va = float(np.nanmedian([corr_of(it, {}) for it in VAs]))
o_tr = float(np.nanmedian([corr_of(it, it["gl"]) for it in TRs]))
o_va = float(np.nanmedian([corr_of(it, it["gl"]) for it in VAs]))
print(f"\nбаза0 (всегда 1х): подбор {b_tr:.3f}  проверка {b_va:.3f}")
print(f"оракул (уровни эксперта): подбор {o_tr:.3f}  проверка {o_va:.3f}")
cur_tr = score(TRs, 0.7, 0.12, 4.0, 0.0, 0.0, auto=False)
cur_va = score(VAs, 0.7, 0.12, 4.0, 0.0, 0.0, auto=False)
print(f"прод (0.7,0.12,4.0): подбор {cur_tr:.3f}  проверка {cur_va:.3f}")
p35_tr = score(TRs, 0.05, 0.01, 8.0, 0.0, 0.0)
p35_va = score(VAs, 0.05, 0.01, 8.0, 0.0, 0.0)
print(f"§6.35 (0.05,0.01,8.0,auto): подбор {p35_tr:.3f}  проверка {p35_va:.3f}")

print("\nперебор (lam, dxfrac, gate_w, level_bias, rail_gate) по МЕДИАННОЙ corr значений:")
best = (p35_tr, 0.05, 0.01, 8.0, 0.0, 0.0)
grid = []
for lam in (0.02, 0.05, 0.15, 0.4):
    for dxf in (0.01, 0.03, 0.08):
        for gw in (2.0, 4.0, 8.0):
            for bias in (0.0, 0.01, 0.03, 0.08):
                for rail in (0.0, 0.5):
                    s = score(TRs, lam, dxf, gw, bias, rail)
                    grid.append((s, lam, dxf, gw, bias, rail))
                    if s > best[0]:
                        best = (s, lam, dxf, gw, bias, rail)
s, lam, dxf, gw, bias, rail = best
v = score(VAs, lam, dxf, gw, bias, rail)
print(f"ЛУЧШИЕ на подборе: lam={lam} dxfrac={dxf} gate_w={gw} bias={bias} rail={rail} → {s:.3f}")
print(f"★ НА ПРОВЕРОЧНЫХ СКВАЖИНАХ: {v:.3f}   (база0 {b_va:.3f}, прод {cur_va:.3f}, "
      f"§6.35 {p35_va:.3f}, оракул {o_va:.3f})")
grid.sort(reverse=True)
print("\nтоп-6 (подбор) — плато или одиночный пик:")
for g in grid[:6]:
    print(f"   corr {g[0]:.3f}  lam={g[1]} dxfrac={g[2]} gate_w={g[3]} bias={g[4]} rail={g[5]}")

r"""_rail_gate2.py — ГЕЙТ ЖЁСТКОГО УПОРА (§6.41) НА ШИРОКОЙ ВЫБОРКЕ, ПО СЫРЫМ ВЕРШИНАМ.

§6.41: перед переходом ВВЕРХ перо стоит в 5% от СВОЕГО упора у 47% переходов против 4% в
случайных строках. Мягкий штраф (`rail_gate`) при lam=0.05/gate_w=8 стоит всего 0.4 и не спасает
кривые, у которых упор совпадает с краем оси. Здесь проверяется ЗАПРЕТ.

★ Всё считается по СЫРЫМ ВЕРШИНАМ (durable §6.41: интерполяция убивает прыжок), сплит по
скважинам, метрика — медианная corr значений с LAS + доля кривых «хуже, чем ничего не делать».

  python _rail_gate2.py [--sheets 80]
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
from _levels_decode2 import decode2
from _decoder_data import train_sheets

CACHE = Path(r"F:\nds\output\taskS\decoder\raw_items.pkl")
ap = argparse.ArgumentParser()
ap.add_argument("--sheets", type=int, default=80)
ap.add_argument("--rebuild", action="store_true")
a = ap.parse_args()


def build():
    out = []
    for n in train_sheets(a.sheets):
        las = find_las(n)
        if not las:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m = extract(str(n)); cols, arr = ds.load_las(str(las))
                mt = ds.match_las(m, cols, arr)
        except Exception:
            continue
        da = m.get("depth_axis") or {}
        if not da.get("bottom_y") or da["bottom_y"] == da["top_y"]:
            continue
        ty, by, td, bd = da["top_y"], da["bottom_y"], da["top_depth"], da["bottom_depth"]
        dep = arr[:, 0]
        for c in ds.real_curves(m):
            fam = DL.build_family(m, c); gl = DL.gt_levels(c)
            ci = mt.get(c["name"], {}).get("col_idx")
            if len(fam) < 2 or not gl or ci is None:
                continue
            lasv = arr[:, ci]
            # ★ СЫРЫЕ ВЕРШИНЫ, без dense
            vy = [c["top_y"] + i for i, x in enumerate(c["xs"]) if x != NULL]
            vx = [float(x) for x in c["xs"] if x != NULL]
            R, X, L = [], [], []
            for y, x in zip(vy, vx):
                dd = td + (y - ty) * (bd - td) / (by - ty)
                li = int(np.clip(np.searchsorted(dep, dd), 0, len(dep) - 1))
                if np.isfinite(lasv[li]):
                    R.append(y); X.append(x); L.append(float(lasv[li]))
            if len(R) >= 100:
                out.append({"well": n.parent.parent.name, "short": c["name"].split()[0],
                            "fam": fam, "gl": gl, "R": R, "X": np.array(X), "L": np.array(L)})
        print(f"  {n.name[:44]:<46} кривых накоплено {len(out)}")
    return out


if CACHE.exists() and not a.rebuild:
    items = pickle.loads(CACHE.read_bytes())
else:
    items = build()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(items))
print(f"кривых: {len(items)} (сырые вершины)")


def corr(it, lv):
    mp = [DL.scale_map(s) for s in it["fam"]]; K = len(it["fam"])
    V = np.array([mp[min(int(lv.get(y, 0)), K - 1)](x) for y, x in zip(it["R"], it["X"].tolist())])
    return float(np.corrcoef(V, it["L"])[0, 1]) if V.std() > 0 else float("nan")


wells = sorted({it["well"] for it in items})
TR = [it for it in items if it["well"] in set(wells[::2])]
VA = [it for it in items if it["well"] in set(wells[1::2])]
print(f"подбор {len(TR)} / проверка {len(VA)} (скважин {len(wells)})\n")


def ev(S, **kw):
    cc, bad, n = [], 0, 0
    for it in S:
        d = corr(it, decode2(dict(zip(it["R"], it["X"].tolist())), it["fam"],
                             0.05, 0.01, 8.0, auto=True, **kw))
        b = corr(it, {})
        if d == d:
            cc.append(d); n += 1
            if b == b and d < b - 0.02:
                bad += 1
    return float(np.median(cc)), 100 * bad / max(1, n)


base = float(np.nanmedian([corr(it, {}) for it in VA]))
orac = float(np.nanmedian([corr(it, it["gl"]) for it in VA]))
print(f"{'вариант':<38}{'подбор':>9}{'ПРОВЕРКА':>11}{'хуже чем ничего':>18}")
VARIANTS = [
    ("как есть (§6.35)", {}),
    ("упор из трассы, МЯГКИЙ", dict(rail_from_trace=True, rail_span=0.05)),
    ("упор из трассы, ЖЁСТКИЙ 0.02", dict(rail_from_trace=True, rail_span=0.02, rail_hard=True)),
    ("упор из трассы, ЖЁСТКИЙ 0.05", dict(rail_from_trace=True, rail_span=0.05, rail_hard=True)),
    ("упор из трассы, ЖЁСТКИЙ 0.10", dict(rail_from_trace=True, rail_span=0.10, rail_hard=True)),
    ("упор из трассы, ЖЁСТКИЙ 0.20", dict(rail_from_trace=True, rail_span=0.20, rail_hard=True)),
]
for tag, kw in VARIANTS:
    t, _ = ev(TR, **kw); v, bad = ev(VA, **kw)
    print(f"{tag:<38}{t:>9.3f}{v:>11.3f}{bad:>17.0f}%")
print(f"\nопоры на проверочных: база0 {base:.3f}, оракул {orac:.3f}")
print("Читать: жёсткий гейт засчитывается, если растёт ПРОВЕРКА и падает «хуже чем ничего».")

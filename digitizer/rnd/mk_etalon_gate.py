r"""Гейт трасс vs ЭТАЛОН (протокол 03.07): наша _auto трасса против кликов эксперта.
Эталон хранит КЛИКИ (точка каждые ~10 строк, между — NULL) — сравнение только на
строках кликов. Выход: CDF |dx| (≤2/3/5/15px), дыры, худшие интервалы глубин
(кластеры строк с |dx|>15px) — это рабочий список этапа B.

  python mk_etalon_gate.py <auto.nlgx> [--etalon <e.nlgx>] [--top N]

Эталон по умолчанию ищется в F:\nds\projects\Archive\<well>\wlg по стему без _auto*.
"""
import sys, os, re, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from extract_nlgx import extract, NULL

ARCHIVE = r"F:\nds\projects\Archive"


def find_etalon(auto_path):
    stem = os.path.splitext(os.path.basename(auto_path))[0]
    base = re.sub(r"_auto.*$", "", stem)
    hits = glob.glob(os.path.join(ARCHIVE, "*", "wlg", base + ".nlgx"))
    return hits[0] if hits else None


def trace_dict(curve):
    ty = curve["top_y"]
    return {ty + i: x for i, x in enumerate(curve["xs"]) if x != NULL}


def depth_fn(model):
    da = model["depth_axis"]
    return lambda y: da["top_depth"] + (y - da["top_y"]) * da["span_depth"] / da["span_px"]


def cluster_bad(ys, max_gap=60):
    """Кластеры соседних плохих строк (разрыв > max_gap px -> новый кластер)."""
    if not ys:
        return []
    ys = sorted(ys)
    out, start, prev = [], ys[0], ys[0]
    for y in ys[1:]:
        if y - prev > max_gap:
            out.append((start, prev))
            start = y
        prev = y
    out.append((start, prev))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    a = sys.argv[1:]
    auto_path = a[0]
    et_path = a[a.index("--etalon") + 1] if "--etalon" in a else find_etalon(auto_path)
    top = int(a[a.index("--top") + 1]) if "--top" in a else 10
    if not et_path or not os.path.exists(et_path):
        print(f"эталон не найден для {auto_path}"); return
    print(f"auto:   {auto_path}\nэталон: {et_path}")

    ma, me = extract(auto_path), extract(et_path)
    dof = depth_fn(me)
    ours = {c["name"].split()[0]: trace_dict(c) for c in ma["curves"]}
    all_bad = []
    for ce in me["curves"]:
        mn = ce["name"].split()[0]
        if mn.rstrip("0123456789") in ("DA",):
            continue
        gt = trace_dict(ce)
        if len(gt) < 30:
            continue
        our = ours.get(mn) or ours.get(mn.rstrip("0123456789") + "1")
        if not our:
            print(f"[{mn}] нет нашей кривой — пропуск"); continue
        dx, miss, bad_rows = [], 0, []
        for y, xg in gt.items():
            xo = our.get(y)
            if xo is None:
                miss += 1
                bad_rows.append(y)
                continue
            d = abs(xo - xg)
            dx.append(d)
            if d > 15:
                bad_rows.append(y)
        dx = np.array(dx)
        n = len(gt)
        print(f"\n[{mn}] GT-кликов {n}, дыр {miss} ({miss/n*100:.1f}%)")
        if len(dx):
            print(f"  median {np.median(dx):.1f}px | ≤2px {(dx<=2).mean()*100:.0f}% | ≤3px {(dx<=3).mean()*100:.0f}% "
                  f"| ≤5px {(dx<=5).mean()*100:.0f}% | ≤15px {(dx<=15).mean()*100:.0f}% | >15px {(dx>15).mean()*100:.1f}%")
        clusters = cluster_bad(bad_rows)
        clusters.sort(key=lambda ab: -(ab[1] - ab[0]))
        for (y0, y1) in clusters[:top]:
            rows = [y for y in bad_rows if y0 <= y <= y1]
            print(f"    {dof(y0):7.1f}..{dof(y1):7.1f}m  ({len(rows)} плохих кликов, y {y0}..{y1})")
        all_bad += [(mn, y0, y1, dof(y0), dof(y1)) for (y0, y1) in clusters]

    print(f"\nвсего плохих интервалов: {len(all_bad)}")


if __name__ == "__main__":
    main()

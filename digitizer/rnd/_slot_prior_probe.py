r"""_slot_prior_probe.py — ЕСТЬ ЛИ У СЛОТА РАМКИ РАЗЛИЧАЮЩИЙ ПРОСТРАНСТВЕННЫЙ ПРИОР? (§6.20.7-п.2)

§6.20.6 (лучевой поиск) показал: идентичность задаётся ПРИОРОМ, не геометрией. §6.20.7-п.2
предположил, что «настоящие приоры слота (класс/порядок/шкала) в трассировщик не подаются».
Но прежде чем ПОДАВАТЬ приор — надо проверить, СУЩЕСТВУЕТ ли он. Первый же замер (BOGAT_011):
все 4 кривые имеют scale-axis с ОДНИМ x_left/x_right (74..1981 = весь трек), различаются лишь
value-калибровкой (омы). То есть x scale-оси идентичности НЕ несёт (это §6.11).

Здесь — систематически по всем мульти-листам. Для каждого листа:
  • x-медиана и размах каждой GT-кривой (где кривая ЖИВЁТ);
  • x_left/x_right её scale-оси (через curve_desc.axis_idx, связь 3074/3074, §6.14);
  • РАЗДЕЛИМЫ ли кривые: (а) по x scale-оси, (б) по value-калибровке, (в) по факт. x-медиане.

Вывод, который нужен: для каких КЛАССОВ листов существует пространственный приор слота
(тогда «подать его в трассировщик» осмысленно), а для каких кривые физически неразличимы
ничем, кроме собственного хода (тогда приор слота — тупик, и это надо закрыть замером).

  python _slot_prior_probe.py
"""
import sys, io, json, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds

ARCH = Path(r"F:\nds\projects\Archive")
SEM = Path(r"F:\nds\projects\Semeguniv_001\wlg")

SHEETS = [
    ("BOGAT_011", ARCH / r"BOGAT_011\wlg\BOGAT_011_BKZ, DS_2800-3190_200_1984-03-02_D_1_B_1.nlgx"),
    ("BOGAT_015", ARCH / r"BOGAT_015\wlg\BOGAT_015_BKZ, DS_3300-3700_200_1990-01-08_D_1.nlgx"),
    ("LEVEN_023", ARCH / r"LEVEN_023\wlg\LEVEN_023_BKZ, DS_1010-1500_200_1996-12-02_D_1.nlgx"),
    ("BEZLUD_051", ARCH / r"BEZLUD_051\wlg\BEZLUD_051_RK, AK, DS_2688-3142_500_1998-08-05_D_1.nlgx"),
    ("YULIIV_055", ARCH / r"YULIIV_055\wlg\YULIIV_055_MK, MBK, MDS_2084-3060_200_1998-07-18_D_1.nlgx"),
    ("KREMEN_089", ARCH / r"KREMEN_089\wlg\KREMEN_089_BKZ, SP, DS_0794-1410_200_1999-06-03_D_1.nlgx"),
    ("RYBAL_168_B1", ARCH / r"RYBAL_168\wlg\1980.09.02_Rybal_168_BKZ1_(1280-2724)_GZ1_GZ2_GZ3_GZ4_PS.nlgx"),
    ("SEM_STKDS_3400", SEM / "Semeguniv_1_STK_DS_3400_3650_200_D1.nlgx"),
    ("SEM_BKZ_3400", SEM / "Semeguniv_1_BKZ_3400_3640_200_D1.nlgx"),
    ("SEM_MK_1380_D2", SEM / "Semeguniv_1_MK_1380_2340_200_D2.nlgx"),
]


def sa_for(mo):
    """idx scale-оси -> (x_left, x_right, v_left, v_right) по имени/индексу."""
    return {i: s for i, s in enumerate(mo.get("scale_axes", []))}


def main():
    for tag, n in SHEETS:
        if not n.is_file():
            print(f"\n### {tag}: НЕТ ФАЙЛА {n.name}"); continue
        mo = extract(str(n))
        sas = sa_for(mo)
        cd = {c["name"]: c for c in mo.get("curve_desc", [])}
        gts = [c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])]
        print(f"\n### {tag}  ({len(gts)} кривых)")
        rows = []
        for g in gts:
            xs = [x for x in g["xs"] if x != NULL]
            xmed = float(np.median(xs)); xlo = float(np.percentile(xs, 2)); xhi = float(np.percentile(xs, 98))
            d = cd.get(g["name"])
            sa = sas.get(d["axis_idx"]) if d and d.get("axis_idx") is not None else None
            sx = (sa["x_left"], sa["x_right"]) if sa else None
            sv = (sa["v_left"], sa["v_right"]) if sa else None
            rows.append({"name": g["name"].split()[0], "xmed": xmed, "xlo": xlo, "xhi": xhi,
                         "sa_x": sx, "sa_v": sv})
        # разделимость по x scale-оси
        sax = [tuple(r["sa_x"]) for r in rows if r["sa_x"]]
        uniq_sax = len(set(sax))
        sav = [tuple(r["sa_v"]) for r in rows if r["sa_v"]]
        uniq_sav = len(set(sav))
        for r in rows:
            print(f"   {r['name']:<7} x-медиана {r['xmed']:>6.0f}  размах [{r['xlo']:>5.0f}..{r['xhi']:>5.0f}]"
                  f"   SA_x {r['sa_x']}  SA_v {r['sa_v']}")
        # факт. x-медианы: попарно разделимы?
        meds = sorted(r["xmed"] for r in rows)
        gaps = np.diff(meds) if len(meds) > 1 else np.array([0])
        overlaps = 0
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                lo = max(a["xlo"], b["xlo"]); hi = min(a["xhi"], b["xhi"])
                if hi > lo:                      # диапазоны пересекаются
                    overlaps += 1
        npair = len(rows) * (len(rows) - 1) // 2
        print(f"   -> уникальных SA_x: {uniq_sax}/{len(sax)}   уникальных SA_v: {uniq_sav}/{len(sav)}"
              f"   пар с ПЕРЕСЕЧЕНИЕМ x-размаха: {overlaps}/{npair}"
              f"   мин.зазор x-медиан: {gaps.min() if len(gaps) else 0:.0f}px")


if __name__ == "__main__":
    main()

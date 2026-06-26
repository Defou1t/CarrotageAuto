r"""
run_pure.py — ЧИСТЫЕ проверки логики (без numpy/cv2): meta (разбор имени/мнемоники) и
confidence (карта AUTO/FLAG). Запускаются где угодно, в т.ч. в CI без тяжёлых зависимостей.

  python auto/tests/run_pure.py        (или python -m auto.tests.run_pure)

Печатает PASS/FAIL по каждому кейсу и итог; код возврата 0/1. CV-стадии (U0/U1/trace) проверяются
на боевых сканах по auto/VALIDATION.md (нужны numpy/cv2 и данные).
"""
import sys
from pathlib import Path

_FAILS = []


def check(name, cond, got=None):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"  got={got!r}"))
    if not ok:
        _FAILS.append(name)


def test_meta():
    print("meta:")
    # путь к mnemonics.json — корень репо (на 3 уровня выше: auto/tests/ → repo)
    mn = str(Path(__file__).resolve().parent.parent.parent / "mnemonics.json")
    import importlib
    M = importlib.import_module("auto.meta")

    m = M.parse_filename("Semeguniv_20_BK+MBK_3080_3520_200_D1.jpg", mn)
    check("delivery: well", m.well == "Semeguniv_20", m.well)
    check("delivery: depths", (m.top_depth, m.bottom_depth) == (3080.0, 3520.0), (m.top_depth, m.bottom_depth))
    check("delivery: scale", m.scale == 200, m.scale)
    check("delivery: curves BK+MBK", m.expected_curves == ["BK", "MBK"], m.expected_curves)

    a = M.parse_filename("BOGAT_011_BKZ_3630-3896_200_1984-04-17_D_1_B_1.jpg", mn)
    check("archive(dash): well", a.well == "BOGAT_011", a.well)
    check("archive(dash): depths", (a.top_depth, a.bottom_depth) == (3630.0, 3896.0), (a.top_depth, a.bottom_depth))
    check("archive(dash): BKZ→zonds", a.expected_curves == ["GZ1", "GZ2", "GZ3", "GZ4", "GZ5", "OGZ"], a.expected_curves)

    check("alias МБК→MBK", M.curve_info("МБК", mn)["root"] == "MBK")
    check("SP expected red", M.curve_info("SP", mn)["color"] == "red")
    check("class GZ31=RES", M.curve_class("GZ31") == "RES", M.curve_class("GZ31"))
    check("class SP=SP", M.curve_class("SP") == "SP")


def test_confidence():
    print("confidence (AUTO/FLAG):")
    import importlib
    CF = importlib.import_module("auto.confidence")

    class L:
        def __init__(s, **k):
            s.__dict__.update(k); s.confidence = None; s.flag_reason = None
        @property
        def x_band(s):
            return s.x_hi - s.x_lo

    class Sheet:
        def __init__(s, lines):
            s.lines = lines; s.diag = {}

    lines = [
        L(track_index=0, color="red", behavior="smooth", density=50, thickness=2, x_lo=100, x_hi=120, n_strokes=1, n_levels_est=1),
        L(track_index=0, color="black", behavior="peaky", density=40, thickness=2, x_lo=200, x_hi=240, n_strokes=4, n_levels_est=1),
        L(track_index=0, color="black", behavior="peaky", density=38, thickness=2, x_lo=235, x_hi=275, n_strokes=4, n_levels_est=1),
        L(track_index=0, color="green", behavior="peaky", density=3, thickness=1, x_lo=300, x_hi=320, n_strokes=2, n_levels_est=1),
    ]
    CF.classify(Sheet(lines))
    check("SP smooth → AUTO", lines[0].confidence == "AUTO", lines[0].confidence)
    check("пучок A → FLAG bunched", lines[1].confidence == "FLAG" and lines[1].flag_reason == "bunched_crossing", (lines[1].confidence, lines[1].flag_reason))
    check("пучок B → FLAG bunched", lines[2].confidence == "FLAG" and lines[2].flag_reason == "bunched_crossing", (lines[2].confidence, lines[2].flag_reason))
    check("выцветшая → FLAG faint", lines[3].confidence == "FLAG" and lines[3].flag_reason == "faint", (lines[3].confidence, lines[3].flag_reason))


def main():
    if __package__ in (None, ""):                     # запуск как скрипт: репо на путь
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    test_meta()
    test_confidence()
    print(f"\nИТОГ: {'ВСЕ PASS' if not _FAILS else str(len(_FAILS)) + ' FAIL: ' + ', '.join(_FAILS)}")
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

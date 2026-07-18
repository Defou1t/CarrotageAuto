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
    # ЧАСТЬ бланка уточняет набор зондов (замер 19.07): D_1_B_1 → зонды 1-3 + OGZ, а зонды 4-5
    # уходят в файл-брат. Раньше часть архивных имён не парсилась (стоит ПОСЛЕ даты) и приор
    # выдавал все шесть зондов сразу.
    check("archive(dash): part=D1B1", a.part == "D1B1", a.part)
    check("archive(dash): BKZ|D1B1→зонды 1-3+OGZ",
          a.expected_curves == ["GZ1", "GZ2", "GZ3", "OGZ"], a.expected_curves)
    b = M.parse_filename("YULIIV_107_BKZ_2090-3550_200_1996-09-02_D_12.jpg", mn)
    check("archive: BKZ|D12→зонды 4-5", b.expected_curves == ["GZ4", "GZ5"], b.expected_curves)
    check("BKZ без части → весь набор зондов",
          M.expected_curves("BKZ", mn) == ["GZ1", "GZ2", "GZ3", "GZ4", "GZ5", "OGZ"],
          M.expected_curves("BKZ", mn))

    # МУЛЬТИ-ЛИСТ (P0-2, 18.07): в имени кривые перечислены ЧЕРЕЗ ЗАПЯТУЮ и приходят ОДНИМ
    # токеном (split по «_» запятую не делит) — без расщепления лист выглядел однокривым.
    c = M.parse_filename("BEZLUD_051_BK, IK_2680-3404_200_1998-09-14_D_1.jpg", mn)
    check("мульти: well", c.well == "BEZLUD_051", c.well)
    check("мульти: токен цел", c.curves_token == "BK, IK", c.curves_token)
    # ЦЕЛЫЙ токен важнее расщепления (19.07): набор замерен по архиву ПО ТОКЕНУ ЦЕЛИКОМ.
    # «BK, IK» несёт BK + ИКА/ИКР (не «IK»); «BKZ, DS» — зонды 4-5 + CALI + SP, а зонды 1-3
    # уходят в файл-брат «BKZ» (один бланк цифруется несколькими nlgx).
    check("мульти: «BK, IK»→BK,IKA,IKR", c.expected_curves == ["BK", "IKA", "IKR"], c.expected_curves)
    check("мульти: «MBK, MDS, MK»→4",
          M.expected_curves("MBK, MDS, MK", mn) == ["MBK", "MDS", "MGZ", "MPZ"],
          M.expected_curves("MBK, MDS, MK", mn))
    check("мульти: «GK, NGK»→2", M.expected_curves("GK, NGK", mn) == ["GK", "NGK"],
          M.expected_curves("GK, NGK", mn))
    check("мульти: «BKZ, DS»→зонды 4-5+CALI+SP",
          M.expected_curves("BKZ, DS", mn) == ["CALI", "GZ4", "GZ5", "SP"],
          M.expected_curves("BKZ, DS", mn))
    check("расщепление живо для токена вне словаря",
          M.expected_curves("BK+MGZ", mn) == ["BK", "MGZ"], M.expected_curves("BK+MGZ", mn))
    check("одиночный токен не тронут", M.expected_curves("MBK", mn) == ["MBK"],
          M.expected_curves("MBK", mn))

    # ВТОРАЯ КОНВЕНЦИЯ ИМЕНИ (скважины RYBAL, 19% архива): дата первой, интервал в скобках И СВОЙ
    # у каждой группы, хвост после интервала = явные имена кривых. Раньше из такого имени не
    # бралось НИЧЕГО (well='1966.01.12', токен пуст) — приор был пуст у пятой части архива.
    r = M.parse_filename("1966.01.12_Rybal_058_GK_(0012-1395)_NGK_(1300-1395)_500.nlgx", mn)
    check("скобки: ветка разбора", r.naming == "paren", r.naming)
    check("скобки: well", r.well == "Rybal_058", r.well)
    check("скобки: общий интервал", (r.top_depth, r.bottom_depth) == (12.0, 1395.0),
          (r.top_depth, r.bottom_depth))
    check("скобки: масштаб", r.scale == 500, r.scale)
    check("скобки: кривые", r.expected_curves == ["GK", "NGK"], r.expected_curves)
    # ★ ГЛАВНОЕ, чего нет в основной конвенции: интервал У КАЖДОЙ кривой свой.
    check("скобки: интервал ПОКРИВОЙ", r.curve_spans.get("NGK") == (1300.0, 1395.0),
          r.curve_spans)
    # хвостовые имена проходят через словарь: в имени пишут PS, слот в nlgx — SP (37 файлов)
    q = M.parse_filename("1963.06.02_Rybal_017_BKZ3_(1264-1468)_GZ3_PS.nlgx", mn)
    check("скобки: PS→SP через словарь", q.expected_curves == ["GZ3", "SP"], q.expected_curves)
    # основная конвенция НЕ должна уходить в скобочную ветку
    check("основная конвенция цела", M.parse_filename(
        "Semeguniv_20_BK+MBK_3080_3520_200_D1.jpg", mn).naming == "main")

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

    # Правило: ЦВЕТО-УНИКАЛЬНАЯ линия (единственная своего цвета в треке) → всегда AUTO (цвет
    # задаёт идентичность). faint/bunched-по-ранам срабатывают ТОЛЬКО при одноцветном соседе.
    lines = [
        # track0: red одиночная smooth (SP) + пара black-пучок + пара green (одна faint)
        L(track_index=0, color="red", behavior="smooth", density=50, thickness=2, x_lo=100, x_hi=120, n_strokes=1, n_levels_est=1),
        L(track_index=0, color="black", behavior="peaky", density=40, thickness=2, x_lo=200, x_hi=240, n_strokes=4, n_levels_est=1),
        L(track_index=0, color="black", behavior="peaky", density=38, thickness=2, x_lo=235, x_hi=275, n_strokes=4, n_levels_est=1),
        L(track_index=0, color="green", behavior="peaky", density=3, thickness=1, x_lo=300, x_hi=320, n_strokes=2, n_levels_est=1),
        L(track_index=0, color="green", behavior="peaky", density=50, thickness=2, x_lo=360, x_hi=385, n_strokes=4, n_levels_est=1),
        # одиночная ШИРОКАЯ пиковая (DT): band велик, но цвето-уникальная → AUTO
        L(track_index=1, color="black", behavior="peaky", density=40, thickness=5, x_lo=100, x_hi=340, n_strokes=20, n_levels_est=1, n_runs_med=1.0),
        # цвето-уникальная зигзаг n_runs~2 (STK PZ/SP) → AUTO
        L(track_index=2, color="orange", behavior="peaky", density=40, thickness=5, x_lo=100, x_hi=340, n_strokes=20, n_levels_est=1, n_runs_med=2.0),
        # цвето-уникальная даже n_runs≥3 (резкая SP) → AUTO
        L(track_index=2, color="red", behavior="peaky", density=40, thickness=5, x_lo=400, x_hi=600, n_strokes=20, n_levels_est=1, n_runs_med=3.0),
        # ДВЕ одноцветные n_runs~2 в одном треке = реальный пучок → обе FLAG
        L(track_index=3, color="black", behavior="peaky", density=40, thickness=5, x_lo=100, x_hi=340, n_strokes=20, n_levels_est=1, n_runs_med=2.0),
        L(track_index=3, color="black", behavior="peaky", density=40, thickness=5, x_lo=110, x_hi=350, n_strokes=20, n_levels_est=1, n_runs_med=2.0),
    ]
    CF.classify(Sheet(lines))
    check("SP smooth цвето-уник → AUTO", lines[0].confidence == "AUTO", lines[0].confidence)
    check("пучок A → FLAG bunched", lines[1].confidence == "FLAG" and lines[1].flag_reason == "bunched_crossing", (lines[1].confidence, lines[1].flag_reason))
    check("пучок B → FLAG bunched", lines[2].confidence == "FLAG" and lines[2].flag_reason == "bunched_crossing", (lines[2].confidence, lines[2].flag_reason))
    check("выцветшая с одноцв.соседом → FLAG faint", lines[3].confidence == "FLAG" and lines[3].flag_reason == "faint", (lines[3].confidence, lines[3].flag_reason))
    check("яркая зелёная сосед → AUTO", lines[4].confidence == "AUTO", (lines[4].confidence, lines[4].flag_reason))
    check("одиночная широкая (1 ран/стр) → AUTO", lines[5].confidence == "AUTO", (lines[5].confidence, lines[5].flag_reason))
    check("цвето-уник зигзаг (≥2/стр) → AUTO", lines[6].confidence == "AUTO", (lines[6].confidence, lines[6].flag_reason))
    check("цвето-уник (≥3/стр) → AUTO", lines[7].confidence == "AUTO", (lines[7].confidence, lines[7].flag_reason))
    check("две одноцветные (≥2/стр) → FLAG", lines[8].confidence == "FLAG" and lines[9].confidence == "FLAG", (lines[8].confidence, lines[9].confidence))


def main():
    if __package__ in (None, ""):                     # запуск как скрипт: репо на путь
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    test_meta()
    test_confidence()
    print(f"\nИТОГ: {'ВСЕ PASS' if not _FAILS else str(len(_FAILS)) + ' FAIL: ' + ', '.join(_FAILS)}")
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

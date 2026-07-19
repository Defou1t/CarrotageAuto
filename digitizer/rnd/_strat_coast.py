r"""_strat_coast.py — стратегия «КОАСТ С ЗАТУХАНИЕМ + ПЕРЕЗАХВАТ + ДОБИВКА».

ИДЕЯ (чиним то, что сломала правка §6.19 «jump_limit»).
Базовый trace_line в ветке else («нет рана, перекрывающего предсказание») садится на
БЛИЖАЙШИЙ ран без ограничения расстояния — и остаётся на чужой кривой. Прошлая правка
просто запрещала дальний прыжок (x = pred; continue) и роняла cov до 1-3%, потому что:
  1) коаст шёл с НЕИЗМЕННОЙ скоростью — x улетал по инерции и уже не возвращался;
  2) пройденные в коасте строки НЕ ПИСАЛИСЬ в трассу вообще.
Здесь оба дефекта закрыты:
  • v *= decay на каждой строке коаста (x замирает вместо разгона);
  • максимальная длина коаста: после неё разрешён принудительный перезахват;
  • ПЕРЕЗАХВАТ по ветке cont (ран накрыл pred) либо по близкому рану (<= jump_limit);
  • ★ ДОБИВКА: строки коаста заполняются линейной интерполяцией между последней
    надёжной точкой и точкой перезахвата — cov не рушится.

⛔ метрика: ЧЕСТНЫЕ = med<=3px И cov>=0.9. med без cov не цитируется.
"""
import sys
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
import _relatch_bench as BE

SLMAX = BE.SLMAX
WIDE = BE.WIDE_RUN

# BE.load() каждый раз читает 236 МБ пикла с диска; при переборе десятков конфигураций это
# основная цена прогона. Кэшируем В СВОЁМ процессе (файл стенда не трогаем).
_SHEETS = None
_BE_LOAD = BE.load


def _cached_load():
    global _SHEETS
    if _SHEETS is None:
        _SHEETS = _BE_LOAD()
    return _SHEETS


BE.load = _cached_load


def make_tracer(jump_limit=30.0, decay=0.7, max_coast=200, fill=True,
                reset_v=True, fill_empty=True):
    """Собрать tracer(rec, csr, H) -> {row: x}.

    jump_limit  — дальше этого (плюс ширина рана) прыжок в ветке else запрещён -> коаст;
    decay       — затухание скорости на каждой строке коаста;
    max_coast   — сколько строк коастим максимум, потом принудительный перезахват;
    fill        — добивать строки коаста линейной интерполяцией (главное отличие);
    reset_v     — обнулять скорость в момент перезахвата (после долгого коаста v мусорная);
    fill_empty  — добивать также строки БЕЗ чернил (пустые), попавшие внутрь коаста.
    """
    def tracer(rec, csr, H):
        base = rec["base"]
        x = None
        v = 0.0
        tr = {}
        anchor = None            # (y, x) последней НАДЁЖНОЙ (не интерполированной) точки
        coast_rows = []          # строки, пройденные в текущем коасте
        coasting = False

        def close_coast(y_b, x_b):
            """Перезахват состоялся: добить строки коаста интерполяцией."""
            nonlocal coast_rows, coasting
            if fill and anchor is not None and coast_rows:
                y_a, x_a = anchor
                dy = y_b - y_a
                if dy > 0:
                    for r in coast_rows:
                        tr[r] = x_a + (x_b - x_a) * (r - y_a) / dy
            coast_rows = []
            coasting = False

        for y in range(max(0, rec["y0"]), min(H, rec["y1"] + 1)):
            A, B, Cc = BE._runs_at(csr, y)

            if not len(A):
                if x is not None:
                    x = x + float(np.clip(v, -SLMAX, SLMAX))
                    if coasting:
                        v *= decay
                        if fill_empty:
                            coast_rows.append(y)
                        if len(coast_rows) > max_coast:
                            # коаст исчерпан, но выбирать не из чего — сбрасываем якорь,
                            # интерполировать через такой провал уже нечестно
                            coast_rows = []
                            coasting = False
                            anchor = None
                continue

            if x is None:
                k = int(np.argmin(np.abs(Cc - base)))
                x = float(Cc[k]); v = 0.0; tr[y] = x
                anchor = (y, x)
                continue

            pred = x + float(np.clip(v, -SLMAX, SLMAX))
            cont = np.nonzero((A - 2 <= pred) & (pred <= B + 2))[0]
            if len(cont):
                k = int(cont[np.argmin(np.abs(Cc[cont] - pred))])
            else:
                k = int(np.argmin(np.abs(Cc - pred)))
                a0, b0, c0 = int(A[k]), int(B[k]), float(Cc[k])
                if abs(c0 - pred) > jump_limit + (b0 - a0):
                    # ── дальний прыжок запрещён ──────────────────────────────
                    if not coasting:
                        coasting = True
                        coast_rows = []
                    coast_rows.append(y)
                    if len(coast_rows) <= max_coast:
                        x = pred
                        v *= decay
                        continue
                    # коаст исчерпан -> принудительный перезахват на этот ран,
                    # но добивку НЕ делаем (провал слишком длинный, интерполяция
                    # через него — выдумка); просто начинаем заново.
                    coast_rows = []
                    coasting = False
                    anchor = None
                    v = 0.0

            a, b, c = int(A[k]), int(B[k]), float(Cc[k])
            nx = (b if abs(b - base) >= abs(a - base) else a) if (b - a) >= WIDE else c
            if coasting:
                close_coast(y, float(nx))
                if reset_v:
                    v = 0.0
                else:
                    v = 0.6 * v + 0.4 * (nx - x)
            else:
                v = 0.6 * v + 0.4 * (nx - x)
            x = float(nx)
            tr[y] = float(nx)
            anchor = (y, x)

        BE._extend_ends(tr, csr, H, SLMAX)
        return tr

    return tracer


# ─────────────────── ВЕРСИЯ 2: коаст С ПОДТВЕРЖДЕНИЕМ ВПЕРЁД ───────────────────
# Слепой коаст проигрывает (см. замеры): далёкий прыжок в ветке else ЧАСТО правильный —
# кривая реально уходит быстро, а pred отстаёт из-за клампа скорости SLMAX=30. Блокировать
# прыжок «просто по расстоянию» нельзя (это же говорит диагностика: ошибочные прыжки БЛИЗКИЕ).
# Значит коастить надо ТОЛЬКО когда есть основание считать это ОБРЫВОМ ТУШИ, а не уходом
# кривой: в ближайших N строках чернила ВОЗОБНОВЛЯЮТСЯ рядом с текущим x. Проверяем это
# заглядыванием вперёд ОДИН РАЗ в момент решения (векторно по CSR).

def _lookahead(csr, y, tgt, n, tol, H):
    """Первая строка в (y, y+n], где есть ран, накрывающий tgt±tol. None — нет такой."""
    A, B, Cc, ptr = csr
    y2 = min(H - 1, y + n)
    i0, i1 = int(ptr[y + 1]), int(ptr[y2 + 1])
    if i1 <= i0:
        return None
    a = A[i0:i1]; b = B[i0:i1]
    hit = np.nonzero((a - tol <= tgt) & (tgt <= b + tol))[0]
    if not len(hit):
        return None
    j = i0 + int(hit[0])
    return int(np.searchsorted(ptr, j, side="right")) - 1


def make_tracer2(jump_limit=30.0, decay=0.7, max_coast=200, fill=True,
                 look_tol=6.0, reset_v=True, stats=None):
    """Коаст только с подтверждением: чернила должны возобновиться рядом с x.
    Иначе — штатный прыжок (как в базе), т.е. стратегия НИКОГДА не хуже базы по механике."""
    def tracer(rec, csr, H):
        base = rec["base"]
        x = None; v = 0.0; tr = {}
        anchor = None
        coast_until = -1          # строка, на которой запланирован перезахват
        coast_rows = []

        for y in range(max(0, rec["y0"]), min(H, rec["y1"] + 1)):
            A, B, Cc = BE._runs_at(csr, y)

            if not len(A):
                if x is not None:
                    x = x + float(np.clip(v, -SLMAX, SLMAX))
                    if y <= coast_until:
                        v *= decay
                        coast_rows.append(y)
                continue

            if x is None:
                k = int(np.argmin(np.abs(Cc - base)))
                x = float(Cc[k]); v = 0.0; tr[y] = x; anchor = (y, x)
                continue

            pred = x + float(np.clip(v, -SLMAX, SLMAX))

            if y < coast_until:                      # ещё в коасте — молчим
                x = pred; v *= decay; coast_rows.append(y)
                continue

            cont = np.nonzero((A - 2 <= pred) & (pred <= B + 2))[0]
            if len(cont):
                k = int(cont[np.argmin(np.abs(Cc[cont] - pred))])
            else:
                k = int(np.argmin(np.abs(Cc - pred)))
                a0, b0, c0 = int(A[k]), int(B[k]), float(Cc[k])
                if abs(c0 - pred) > jump_limit + (b0 - a0) and y > coast_until:
                    y2 = _lookahead(csr, y, pred, max_coast, look_tol, H)
                    if stats is not None:
                        stats["решений"] = stats.get("решений", 0) + 1
                    if y2 is not None:
                        if stats is not None:
                            stats["коастов"] = stats.get("коастов", 0) + 1
                            stats["строк_коаста"] = stats.get("строк_коаста", 0) + (y2 - y)
                        coast_until = y2
                        coast_rows = [y]
                        x = pred; v *= decay
                        continue
                    # чернила рядом не возобновляются -> это НЕ обрыв, кривая ушла: прыгаем

            a, b, c = int(A[k]), int(B[k]), float(Cc[k])
            nx = (b if abs(b - base) >= abs(a - base) else a) if (b - a) >= WIDE else c
            if coast_rows:                            # перезахват: добить пройденное
                if fill and anchor is not None:
                    y_a, x_a = anchor
                    dy = y - y_a
                    if dy > 0:
                        for r in coast_rows:
                            tr[r] = x_a + (float(nx) - x_a) * (r - y_a) / dy
                coast_rows = []
                v = 0.0 if reset_v else 0.6 * v + 0.4 * (nx - x)
            else:
                v = 0.6 * v + 0.4 * (nx - x)
            x = float(nx); tr[y] = float(nx); anchor = (y, x)

        BE._extend_ends(tr, csr, H, SLMAX)
        return tr

    return tracer


# ─────────────── ВЕРСИЯ 3: ТОЛЬКО ДОБИВКА (без запрета прыжка) ───────────────
# Раз запрет далёкого прыжка вреден (замер: далёкий прыжок попадает в СВОЙ ран в 48.4%
# случаев — чаще, чем близкий, 43.3%), от стратегии остаётся её вторая половина: ДОБИВКА.
# Базовая трасса не пишет строки, где в полосе вообще нет чернил (обрыв туши). Такие
# внутренние дыры заполняются линейной интерполяцией — cov растёт, а med не портится,
# если дыра короткая и концы дыры близки друг к другу (значит кривая никуда не ушла).

def make_tracer3(max_gap=40, max_dx=20.0, inner=None):
    def tracer(rec, csr, H):
        tr = (inner(rec, csr, H) if inner else BE.trace(rec, csr, H))
        if len(tr) < 2:
            return tr
        ys = sorted(tr)
        for i in range(len(ys) - 1):
            ya, yb = ys[i], ys[i + 1]
            g = yb - ya - 1
            if g <= 0 or g > max_gap:
                continue
            xa, xb = tr[ya], tr[yb]
            if abs(xb - xa) > max_dx:
                continue
            for r in range(ya + 1, yb):
                tr[r] = xa + (xb - xa) * (r - ya) / (yb - ya)
        return tr
    return tracer


def main():
    """ИТОГОВЫЙ ПРОТОКОЛ. Каждая строка — реальный прогон 24 кривых."""
    BE.report("0. baseline (как в проде)", BE.run_strategy())
    # ГЕЙТ: с выключенным коастом мой tracer обязан быть бит-в-бит = baseline
    BE.report("0'. ГЕЙТ: мой tracer, коаст выключен (jump_limit=1e9)",
              BE.run_strategy(tracer=make_tracer(1e9, 0.7, 200, True)))
    BE.report("1. слепой коаст jl=30 d=0.7 mc=200 БЕЗ добивки",
              BE.run_strategy(tracer=make_tracer(30.0, 0.7, 200, fill=False)))
    BE.report("2. слепой коаст jl=30 d=0.7 mc=200 С добивкой",
              BE.run_strategy(tracer=make_tracer(30.0, 0.7, 200, fill=True)))
    BE.report("3. коаст с подтверждением вперёд jl=30 mc=60 tol=6 С добивкой",
              BE.run_strategy(tracer=make_tracer2(30.0, 0.7, 60, True, 6.0)))
    BE.report("4. ★ ТОЛЬКО ДОБИВКА (max_gap=60, max_dx=30), прыжок НЕ трогаем",
              BE.run_strategy(tracer=make_tracer3(60, 30.0)))
    BE.report("5. лучший коаст (jl=600 mc=20 d=0.5) + добивка",
              BE.run_strategy(tracer=make_tracer3(60, 30.0,
                                                  inner=make_tracer(600.0, 0.5, 20, True))))


if __name__ == "__main__":
    main()

r"""_dp_trace.py — ГЛОБАЛЬНЫЙ ДЕКОДЕР ПУТИ ВМЕСТО ЖАДНОГО ВЫБОРА (§6.61, дешёвая проверка P2).

§6.60 закрыл класс «ограничить прыжок и пропустить строку»: экскурсию нельзя ВЫРЕЗАТЬ, её надо
ИСПРАВИТЬ — на той же строке выдать правильный x. Это постановка декодера пути (P2 §6.8), и там
подразумевался ML. ⚠ Прежде чем открывать ML, проверяется ДЕШЁВАЯ форма той же идеи: заменить
ЖАДНЫЙ выбор рана (auto/trace2d.py: «ближайший к предсказанию, решение принято и забыто») на
ГЛОБАЛЬНЫЙ ДП по всей колонке — выбрать последовательность ранов, минимизирующую суммарную
негладкость.

ПОЧЕМУ ЭТО ЧЕСТНАЯ ПРОВЕРКА ГИПОТЕЗЫ. Жадный трассировщик принимает решение по ОДНОЙ строке и
назад не смотрит: §6.58 показал, что отказ — это экскурсии, а экскурсия видна только в контексте
(«ушёл и вернулся»). ДП видит весь столбец сразу. Если глобальная непрерывность САМА ПО СЕБЕ
ничего не даёт, то и обучаемый декодер на тех же входах вряд ли спасёт — значит дело не в
постановке, а в самих чернилах. Если даёт — направление подтверждено ДО вложения в ML.

МОДЕЛЬ (без обучения, два параметра):
  стоимость перехода = |x_t − x_{t−1}|                    — негладкость;
  штраф пропуска строки = skip_cost (строка без рана или отвергнутая);
  приор полосы = lam * |x_t − base| / полуширина полосы    — не уходить от своей линии.
Решается за O(строк × ранов²) вперёд-назад по столбцу.

  python _dp_trace.py --self-test        # сверка ДП с жадным на синтетике
Встраивается в гейт: `_pick_gate.py --build --dp` (см. там опцию).
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
import numpy as np
from auto import imaging as im

MAXRUNS = 24            # больше — это уже не линия, а сплошная тушь; берём ближайшие к base


def dp_trace_line(fg, line, frame, p, band_pad=8, slmax=30.0, wide_run=14, x_range=None,
                  jump_limit=None, lam=0.02, skip_cost=6.0, maxjump=None, wid=0.0):
    """Та же сигнатура, что `auto.trace2d.trace_line`, но выбор ранов — ГЛОБАЛЬНЫЙ ДП.
    Возвращает dict{row: x}. Строки, где выгоднее промолчать, НЕ пишутся (как и у жадного)."""
    H, W = fg.shape
    if x_range is not None:
        lo = max(0, int(x_range[0])); hi = min(W, int(x_range[1]) + 1)
    else:
        lo = max(0, int(line.x_lo) - band_pad); hi = min(W, int(line.x_hi) + band_pad + 1)
    base = float(line.x_center)
    half = max(1.0, (hi - lo) / 2.0)
    y0, y1 = max(0, line.y0), min(H, line.y1 + 1)
    if y1 <= y0 or hi <= lo:
        return {}

    rows, cands, widths = [], [], []
    for y in range(y0, y1):
        rr = im.row_runs(fg[y, lo:hi])
        if not rr:
            continue
        pts, wds = [], []
        for a, b, c in rr:
            a += lo; b += lo; c += lo
            # широкий ран = горизонтальный спайк: вершина, а не центр (правило жадного)
            x = (b if abs(b - base) >= abs(a - base) else a) if (b - a) >= wide_run else c
            pts.append(float(x)); wds.append(float(b - a + 1))
        order = np.argsort(pts)
        pts = [pts[i] for i in order]; wds = [wds[i] for i in order]
        if len(pts) > MAXRUNS:
            keep = sorted(range(len(pts)), key=lambda i: abs(pts[i] - base))[:MAXRUNS]
            keep.sort()
            pts = [pts[i] for i in keep]; wds = [wds[i] for i in keep]
        rows.append(y); cands.append(np.array(pts, float)); widths.append(np.array(wds, float))
    # ★ ПРИЗНАК ИДЕНТИЧНОСТИ БЕЗ ОБУЧЕНИЯ (§6.64): ШИРИНА ШТРИХА. Перо одной кривой пишет своей
    # толщиной; соседняя кривая, рамка или сетка обычно другой. §6.61 показал, что ГЛАДКОСТИ ОДНОЙ
    # НЕ ХВАТАЕТ — глобально самый гладкий путь может целиком уйти на соседа. Ширина — самый
    # дешёвый доступный признак «это тот же штрих», проверяемый БЕЗ обучения.
    # Целевая ширина берётся робастно: медиана ширин ранов, ближайших к базлайну по каждой строке.
    if wid > 0 and widths:
        near = [w[int(np.argmin(np.abs(c - base)))] for c, w in zip(cands, widths) if len(c)]
        w_target = float(np.median(near)) if near else 1.0
    else:
        w_target = None
    if not rows:
        return {}

    # ── ПРЯМОЙ ХОД. Состояние = выбранный ран на строке; плюс состояние «пропуск» (индекс -1),
    # чтобы ДП мог честно промолчать там, где любая привязка дороже штрафа.
    INF = 1e18
    prev_x = None
    prev_cost = None
    back = []                                   # для каждой строки: откуда пришли
    for i, xs in enumerate(cands):
        n = len(xs)
        prior = lam * np.abs(xs - base) / half * 100.0
        if w_target:
            # штраф за несоответствие ширины: относительный, чтобы не зависеть от масштаба скана
            prior = prior + wid * np.abs(widths[i] - w_target) / max(1.0, w_target)
        if i == 0:
            cost = prior + np.abs(xs - base) * 0.0
            bk = np.full(n + 1, -2, int)
            cost = np.append(cost, skip_cost)   # последний элемент — состояние «пропуск»
            back.append(bk)
        else:
            m = len(prev_x)
            # переход ран→ран: |dx|, но не дороже maxjump (иначе запрет)
            d = np.abs(xs[None, :] - prev_x[:, None])
            trans = d.copy()
            if maxjump is not None:
                trans[d > maxjump] = INF
            tot = prev_cost[:m, None] + trans                     # из ранов
            skip_from = prev_cost[m] + np.abs(xs - prev_hold)      # из пропуска: держим x
            best_prev = np.argmin(tot, axis=0)
            best_val = tot[best_prev, np.arange(n)]
            take_skip = skip_from < best_val
            src = np.where(take_skip, m, best_prev)
            cost = np.where(take_skip, skip_from, best_val) + prior
            # состояние «пропуск» на этой строке
            skip_val = min(float(prev_cost[:m].min()) if m else INF, float(prev_cost[m])) + skip_cost
            skip_src = int(np.argmin(prev_cost[:m])) if m and prev_cost[:m].min() <= prev_cost[m] else m
            bk = np.append(src, skip_src)
            cost = np.append(cost, skip_val)
            back.append(bk)
        prev_x = xs
        prev_cost = cost
        prev_hold = float(xs[int(np.argmin(cost[:len(xs)]))]) if len(xs) else base

    # ── ОБРАТНЫЙ ХОД
    k = int(np.argmin(prev_cost))
    out = {}
    for i in range(len(rows) - 1, -1, -1):
        xs = cands[i]
        if k < len(xs):
            out[rows[i]] = float(xs[k])
        k = int(back[i][k]) if i else -2
        if k == -2:
            break
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        # синтетика: своя кривая (синус) + прямая-приманка рядом; жадный должен на неё сесть
        H, W = 800, 300
        fg = np.zeros((H, W), bool)
        true_x = 120 + 40 * np.sin(np.arange(H) / 60.0)
        for y in range(H):
            fg[y, int(true_x[y]) - 1:int(true_x[y]) + 2] = True
            fg[y, 200:203] = True                       # приманка: прямая
        for y in range(300, 340):                        # пропуск на своей кривой
            fg[y, int(true_x[y]) - 1:int(true_x[y]) + 2] = False

        class L:
            x_lo, x_hi, x_center, y0, y1 = 80, 240, 120.0, 0, H - 1
        from auto.trace2d import trace_line
        from auto.config import Config
        p = Config().cv
        g = trace_line(fg, L(), None, p)
        d = dp_trace_line(fg, L(), None, p)
        for nm, tr in (("жадный", g), ("ДП", d)):
            e = [abs(tr[y] - true_x[y]) for y in tr]
            print(f"{nm:<8} точек {len(tr):>4}  медиана ошибки {np.median(e):>6.1f}px  "
                  f"p90 {np.percentile(e, 90):>7.1f}px  доля точных(<=3px) {np.mean(np.array(e) <= 3):.2f}")

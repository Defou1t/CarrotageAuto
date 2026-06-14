"""
b4_trace.py — АВТОНОМНАЯ оцифровка кривой в NeuraLOG (Digitize Curve, CMD 32879).

Два примитива (выбор пользователя — сравнить):
  • RMB-авто  (digitize_rmb)        — ПКМ авто-трасса NeuraLOG по Color/BW Mode.
                                      Несколько кликов по линии + проскролл вниз.
  • LMB-реплей (digitize_lmb_replay) — проигрывает точки трассы Блока 2 как
                                      ЛКМ-вершины ломаной.

Наведение/клик — реальный OS-ввод из auto_click_value (seek по статус-бару +
mouse_event), БЕЗ Claude/computer-use. Навигация по глубине — ползунок
slider_depth_position (= глубина напрямую, форсит перерисовку).

ПРЕДУСЛОВИЕ: открыт nlgx с Depth Axis + Scale Axis + выбранной кривой.

CLI:
  python b4_trace.py <image.jpg> rmb   [--top 3180 --bottom 3580 --bw]
  python b4_trace.py <image.jpg> lmb   [--top 3180 --bottom 3580 --step-m 0.5]
"""
from __future__ import annotations
import sys, time, json, argparse, bisect
from pathlib import Path

import win32api, win32gui, win32con

sys.path.insert(0, r"F:\nds\Auto")
from neuralog_win32 import NeuraLog, WM_COMMAND
from auto_click_value import seek, force_foreground, read_cursor

CMD_DIGITIZE_CURVE = 32879
MOUSEEVENTF_LEFTDOWN  = 0x0002
MOUSEEVENTF_LEFTUP    = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP   = 0x0010
TBM_SETPOS = 0x0405
TBM_GETPOS = 0x0400


# ── загрузка трассы Блока 2 ───────────────────────────────────────

def load_trace(image_path: str):
    ana = Path(image_path).with_name(Path(image_path).stem + "_analysis.json")
    r = json.loads(ana.read_text(encoding="utf-8"))
    cal = r["pixel_calibration"]
    mpp, b = cal["meters_per_pixel"], cal["depth_at_y0"]
    pts = r["curves"][0]["trace"]["points_px"]   # [image_y, image_x]
    pts = sorted(pts, key=lambda p: p[0])
    ys = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    return {"ys": ys, "xs": xs, "mpp": mpp, "b": b,
            "img_h": r["image"]["height"]}

def depth_to_iy(tr, depth):  return (depth - tr["b"]) / tr["mpp"]
def iy_to_depth(tr, iy):     return tr["mpp"] * iy + tr["b"]

def x_at_iy(tr, iy):
    """Линейная интерполяция image_x трассы Блока 2 на image_y."""
    ys, xs = tr["ys"], tr["xs"]
    i = bisect.bisect_left(ys, iy)
    if i <= 0:      return xs[0]
    if i >= len(ys): return xs[-1]
    y0, y1 = ys[i-1], ys[i]
    x0, x1 = xs[i-1], xs[i]
    if y1 == y0:    return x0
    return x0 + (x1 - x0) * (iy - y0) / (y1 - y0)


# ── управление NeuraLOG ──────────────────────────────────────────

def arm_digitize(nl: NeuraLog):
    armed = any(b["command_id"] == CMD_DIGITIZE_CURVE and b["checked"]
                for b in nl.get_toolbar_buttons(nl.cm.toolbar_curve_edit))
    if not armed:
        win32api.SendMessage(nl.cm.main_window, WM_COMMAND, CMD_DIGITIZE_CURVE, 0)
        time.sleep(0.3)
    return any(b["command_id"] == CMD_DIGITIZE_CURVE and b["checked"]
               for b in nl.get_toolbar_buttons(nl.cm.toolbar_curve_edit))

def set_color_mode(nl: NeuraLog, want_bw: bool):
    """Выставить ComboBox режима цвета: 'BW Mode' / 'Color Mode'."""
    combo = nl.cm.combo_color_mode
    if not combo:
        return False
    CB_FINDSTRING = 0x014C; CB_SETCURSEL = 0x014E; CB_GETCOUNT = 0x0146
    target = "BW" if want_bw else "Color"
    n = win32gui.SendMessage(combo, CB_GETCOUNT, 0, 0)
    import ctypes
    for i in range(max(n, 0)):
        buf = ctypes.create_unicode_buffer(64)
        win32gui.SendMessage(combo, 0x0148, i, buf)  # CB_GETLBTEXT
        if target.lower() in buf.value.lower():
            win32gui.SendMessage(combo, CB_SETCURSEL, i, 0)
            # уведомить родителя (CBN_SELCHANGE)
            parent = win32gui.GetParent(combo)
            cid = win32gui.GetDlgCtrlID(combo)
            win32gui.SendMessage(parent, WM_COMMAND, (1 << 16) | cid, combo)
            time.sleep(0.3)
            return True
    return False

def goto_depth(nl: NeuraLog, depth: float):
    ds = nl.cm.slider_depth_position
    win32gui.SendMessage(ds, TBM_SETPOS, 1, int(round(depth)))
    parent = win32gui.GetParent(ds)
    SB_THUMBPOSITION = 4
    win32gui.SendMessage(parent, win32con.WM_HSCROLL,
                         (int(round(depth)) << 16) | SB_THUMBPOSITION, ds)
    time.sleep(0.35)
    return nl.get_current_depth()

def _click_image(nl, ix, iy, button="left"):
    """seek к пикселю изображения и реальный OS-клик нужной кнопкой."""
    res = seek(nl, int(round(ix)), int(round(iy)))
    if not res:
        return False, None
    sx, sy, rix, riy = res
    force_foreground(nl.cm.main_window)
    win32api.SetCursorPos((sx, sy)); time.sleep(0.05)
    if button == "right":
        win32api.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        time.sleep(0.03)
        win32api.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
    else:
        win32api.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.03)
        win32api.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    return True, (rix, riy)


# ── RMB авто-трасса ──────────────────────────────────────────────

def digitize_rmb(nl, tr, top=3180.0, bottom=3580.0, bw=True,
                 step_m=60.0, settle=2.0):
    """
    Автотрасса ПКМ: на каждом шаге наводимся на линию (x из Блока 2) в текущем
    верху видимой зоны и жмём ПКМ — NeuraLOG ведёт трассу вниз по линии.
    Затем сдвигаем глубину на step_m и повторяем, пока не дойдём до bottom.
    """
    arm_digitize(nl)
    set_color_mode(nl, bw)
    d = top
    log = []
    while d < bottom - 1:
        goto_depth(nl, d)
        iy = depth_to_iy(tr, d + 3)        # чуть ниже текущего верха
        ix = x_at_iy(tr, iy)
        ok, got = _click_image(nl, ix, iy, button="right")
        time.sleep(settle)
        cur = nl.get_current_depth()
        log.append((round(d, 1), ok, round(cur, 1) if cur else None))
        print(f"RMB @~{d:.0f}m  click={ok}  -> depth now {cur}")
        # продвинуться: если трасса ушла дальше — продолжить оттуда
        nxt = (cur + 1) if (cur and cur > d + 2) else (d + step_m)
        d = nxt
    return log


# ── LMB реплей точек Блока 2 ──────────────────────────────────────

def _simplify(points, eps=2.0):
    """Дугласа-Пекера упрощение полилинии [(iy,ix)] для меньшего числа кликов."""
    if len(points) < 3:
        return points
    def rdp(pts):
        if len(pts) < 3:
            return pts
        (y0, x0), (y1, x1) = pts[0], pts[-1]
        dmax, idx = 0.0, 0
        for i in range(1, len(pts) - 1):
            y, x = pts[i]
            num = abs((x1 - x0) * (y0 - y) - (x0 - x) * (y1 - y0))
            den = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 or 1
            dd = num / den
            if dd > dmax:
                dmax, idx = dd, i
        if dmax > eps:
            left = rdp(pts[:idx + 1]); right = rdp(pts[idx:])
            return left[:-1] + right
        return [pts[0], pts[-1]]
    return rdp(points)

# ── НАВИГАЦИЯ ВИДА (проверено 13.06): closed-loop колесо + калибровка ─────────
# Все программные пути зума/скролла/арма НЕ применяются — нужны реальные клики/
# колесо. Зум: реальный клик кнопки zoom-out. Арм: клик пункта кривой в CURVES
# (активировать!) + клик иконки Digitize, с ретраем (тоггл). См. PLAN §6.5.
MOUSEEVENTF_WHEEL = 0x0800

def _canvas_rect(nl):
    v = nl._canvas_view()
    l, t, r, b = win32gui.GetWindowRect(v)
    return v, l, t, r, b, (l + r) // 2, (t + b) // 2

def _jiggle_read(nl, x, y):
    """Сдвиг +2/-2 форсит свежий WM_MOUSEMOVE → НеураLOG обновляет статус-бар."""
    win32api.SetCursorPos((x, y)); time.sleep(0.04)
    win32api.SetCursorPos((x + 2, y + 2)); time.sleep(0.04)
    win32api.SetCursorPos((x, y)); time.sleep(0.16)
    return read_cursor(nl)

def _wheel(nl, cx, cy, notches):
    win32api.SetCursorPos((cx, cy)); time.sleep(0.04)
    win32api.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(round(notches)) * 120, 0)
    time.sleep(0.18)

def scroll_to_iy(nl, target_iy, tol=120, iters=50):
    """Closed-loop скролл вида к image_y реальным колесом (−delta=вниз=↑iy).
    Проверка по статус-бару (center_iy). Иммунно к багу залипания."""
    v, l, t, r, b, cx, cy = _canvas_rect(nl)
    force_foreground(nl.cm.main_window); time.sleep(0.1)
    cur = _jiggle_read(nl, cx, cy)[1]
    _wheel(nl, cx, cy, -2); c2 = _jiggle_read(nl, cx, cy)[1]
    per = (c2 - cur) / 2.0 or 30.0
    cur = c2
    for _ in range(iters):
        if abs(cur - target_iy) <= tol:
            break
        n = max(-15, min(15, -(target_iy - cur) / per))
        if -1 < n < 0: n = -1
        if 0 < n < 1: n = 1
        _wheel(nl, cx, cy, n); cur = _jiggle_read(nl, cx, cy)[1]
    return cur

def calibrate_viewport(nl):
    """Линейный маппинг image↔screen текущего вида (оси расцеплены).
    Возвращает (ixA,iyA,ax,cys,iy_top,iy_bot) и center (cx,cy)."""
    v, l, t, r, b, cx, cy = _canvas_rect(nl)
    ixA, iyA = _jiggle_read(nl, cx, cy)
    ixB, _ = _jiggle_read(nl, cx + 200, cy)
    _, iyC = _jiggle_read(nl, cx, cy + 200)
    ax = (ixB - ixA) / 200.0 or 0.6
    cys = (iyC - iyA) / 200.0 or 0.6
    iy_top = iyA - (cy - t) * cys
    iy_bot = iyA + (b - cy) * cys
    return dict(ixA=ixA, iyA=iyA, ax=ax, cys=cys, iy_top=iy_top, iy_bot=iy_bot,
                cx=cx, cy=cy, l=l, t=t, r=r, b=b)

def digitize_lmb_fast(nl, tr, top=3180.0, bottom=3580.0, eps=2.5, margin=60):
    """
    Быстрый ЛКМ-реплей точек Блока 2 БЕЗ per-point seek: на каждый видовой экран
    калибруем маппинг и кликаем все точки диапазона по прямому пересчёту.
    Предусловие: Digitize Curve армирован, зум выставлен (реальные клики).
    Проверено 13.06: 266/267 точек, ломаная идёт по линии.
    """
    LDOWN, LUP = 0x02, 0x04
    iy_lo = depth_to_iy(tr, top); iy_hi = depth_to_iy(tr, bottom)
    pts = [(tr["ys"][i], tr["xs"][i]) for i in range(len(tr["ys"]))
           if iy_lo <= tr["ys"][i] <= iy_hi]
    pts = _simplify(pts, eps=eps)
    log.info(f"digitize_lmb_fast: {len(pts)} точек (RDP eps={eps})")
    i = placed = 0
    while i < len(pts):
        v, l, t, r, b, cx, cy = _canvas_rect(nl)
        scroll_to_iy(nl, pts[i][0] + (b - t) * 0.30)
        cal = calibrate_viewport(nl)
        lo, hi = min(cal["iy_top"], cal["iy_bot"]), max(cal["iy_top"], cal["iy_bot"])
        force_foreground(nl.cm.main_window); time.sleep(0.1)
        progressed = False
        while i < len(pts) and pts[i][0] <= hi - margin:
            iy, ix = pts[i]
            if iy < lo + margin:
                i += 1; continue
            sx = max(cal["l"] + 5, min(cal["r"] - 5,
                     int(cal["cx"] + (ix - cal["ixA"]) / cal["ax"])))
            sy = max(cal["t"] + 5, min(cal["b"] - 5,
                     int(cal["cy"] + (iy - cal["iyA"]) / cal["cys"])))
            win32api.SetCursorPos((sx, sy)); time.sleep(0.02)
            win32api.mouse_event(LDOWN, 0, 0, 0, 0); time.sleep(0.02)
            win32api.mouse_event(LUP, 0, 0, 0, 0); time.sleep(0.03)
            placed += 1; i += 1; progressed = True
        if not progressed:
            i += 1
    log.info(f"digitize_lmb_fast: placed {placed}/{len(pts)}")
    return placed


def digitize_lmb_replay(nl, tr, top=3180.0, bottom=3580.0, step_m=0.5,
                        view_window_m=120.0, eps=2.0):
    """
    Проигрывает точки трассы Блока 2 как ЛКМ-вершины. Точки в диапазоне глубин,
    упрощаются RDP, кликаются по порядку; вид подкручивается ползунком, чтобы
    целевой пиксель был виден.
    """
    arm_digitize(nl)
    iy_top = depth_to_iy(tr, top)
    iy_bot = depth_to_iy(tr, bottom)
    # отобрать точки в диапазоне
    pts = [(tr["ys"][i], tr["xs"][i]) for i in range(len(tr["ys"]))
           if iy_top <= tr["ys"][i] <= iy_bot]
    pts = _simplify(pts, eps=eps)
    print(f"LMB replay: точек после RDP = {len(pts)}")
    placed = 0
    last_depth_set = None
    for iy, ix in pts:
        d = iy_to_depth(tr, iy)
        # держим вид так, чтобы точка была видна: ползунок ~ на глубину точки
        if last_depth_set is None or abs(d - last_depth_set) > view_window_m * 0.4:
            goto_depth(nl, d)
            last_depth_set = d
        ok, got = _click_image(nl, ix, iy, button="left")
        if ok:
            placed += 1
        if placed % 25 == 0:
            print(f"  LMB placed {placed}/{len(pts)} (depth ~{d:.0f})")
    print(f"LMB replay done: {placed}/{len(pts)} точек")
    return placed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("method", choices=["rmb", "lmb"])
    ap.add_argument("--top", type=float, default=3180.0)
    ap.add_argument("--bottom", type=float, default=3580.0)
    ap.add_argument("--bw", action="store_true")
    ap.add_argument("--step-m", type=float, default=None)
    args = ap.parse_args()

    tr = load_trace(args.image)
    nl = NeuraLog()
    if not nl._canvas_view():
        print("Нет открытого проекта/изображения"); return 1
    st = nl.panel_state()
    print("DA:", st["depth_axes"], "SA:", st["scale_axes"], "curves:", st["curves"])

    if args.method == "rmb":
        log = digitize_rmb(nl, tr, args.top, args.bottom, bw=args.bw,
                           step_m=args.step_m or 60.0)
        print("RMB log:", log)
    else:
        n = digitize_lmb_replay(nl, tr, args.top, args.bottom,
                                step_m=args.step_m or 0.5)
        print("LMB placed:", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())

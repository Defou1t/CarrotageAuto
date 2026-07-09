"""
auto_click_value.py — АВТОНОМНЫЙ канва-клик NeuraLOG (без эксперта, без Claude).

Наводится на пиксель ИЗОБРАЖЕНИЯ по закрытому контуру (SetCursorPos →
читать статус-бар 'Cursor Location: ix iy' → корректировать), делает РЕАЛЬНЫЙ
OS-клик (mouse_event — НЕ absolute, поэтому 2 монитора не ломают), затем ловит
модальный value-диалог и вписывает значение.

Доказано 11.06.2026: открывает 'Set Depth Axis Value' и 'Set Scale Axis Value'.

CLI:  python auto_click_value.py <img_x> <img_y> <value> [--combo1 N] [--combo2 N]
"""
import sys, time, re, argparse, threading
import win32api, win32gui, win32con, win32process
sys.path.insert(0, r"F:\nds\Auto")
from neuralog_win32 import NeuraLog, BM_CLICK


def read_cursor(nl):
    s = nl._read_status_part(0)
    m = re.search(r"Cursor Location:\s*(-?\d+)\s+(-?\d+)", s)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def seek(nl, target_ix, target_iy, tol=2, iters=14):
    """SetCursorPos-итерации к пикселю изображения. Возвращает (sx,sy,ix,iy)."""
    view = nl._canvas_view()
    l, t, r, b = win32gui.GetWindowRect(view)
    sx, sy = (l + r) // 2, (t + b) // 2
    win32api.SetCursorPos((sx, sy)); time.sleep(0.12)
    ix0, iy0 = read_cursor(nl)
    win32api.SetCursorPos((sx + 40, sy + 40)); time.sleep(0.12)
    ix1, iy1 = read_cursor(nl)
    if None in (ix0, ix1):
        return None
    kx = (ix1 - ix0) / 40.0 or 1.5
    ky = (iy1 - iy0) / 40.0 or 1.5
    # клампинг в прямоугольник канвы (с полем) — НИКОГДА не выводить курсор за
    # пределы вида (иначе реальный клик попадёт в таскбар/«Пуск» → Win+X меню).
    m = 6
    cx0, cy0, cx1, cy1 = l + m, t + m, r - m, b - m
    def clamp(x, y):
        return max(cx0, min(x, cx1)), max(cy0, min(y, cy1))
    ix = iy = None
    for _ in range(iters):
        sx, sy = clamp(sx, sy)
        # джиггл: статус-бар NeuraLOG обновляет 'Cursor Location' только на свежий
        # WM_MOUSEMOVE — без сдвига значение залипает. Двинуть +2/-2 и осесть.
        win32api.SetCursorPos((sx, sy)); time.sleep(0.04)
        win32api.SetCursorPos((sx + 2, sy + 2)); time.sleep(0.04)
        win32api.SetCursorPos((sx, sy)); time.sleep(0.22)
        ix, iy = read_cursor(nl)
        if ix is None:
            continue
        if abs(ix - target_ix) <= tol and abs(iy - target_iy) <= tol:
            return sx, sy, ix, iy
        sx += int(round((target_ix - ix) / kx))
        sy += int(round((target_iy - iy) / ky))
    # не сошлись к цели — НЕ возвращаем приблизительную точку (не кликаем мимо).
    # Цель, скорее всего, вне видимой зоны (нужен скролл ползунком к глубине).
    if ix is not None and abs(ix - target_ix) <= tol and abs(iy - target_iy) <= tol:
        return clamp(sx, sy) + (ix, iy)
    return None


def force_foreground(hwnd):
    """
    Жёстко вывести окно вперёд (для seek нужен WM_MOUSEMOVE → live статус-бар).
    Без этого SetCursorPos не обновляет 'Cursor Location' (окно не активно).
    Комбинация: ShowWindow + AttachThreadInput + SetForegroundWindow +
    BringWindowToTop + topmost-toggle (обходит блокировку SetForegroundWindow).
    """
    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        cur = win32api.GetCurrentThreadId()
        fg = win32gui.GetForegroundWindow()
        fgt = win32process.GetWindowThreadProcessId(fg)[0] if fg else 0
        if fgt:
            win32process.AttachThreadInput(cur, fgt, True)
        try:
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass
        if fgt:
            win32process.AttachThreadInput(cur, fgt, False)
        # topmost-toggle — поднимает поверх даже когда SetForegroundWindow молчит
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                              win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                              win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        time.sleep(0.15)
    except Exception as e:
        print("force_foreground:", e)


def fill_dialog(nl, value, combo1=None, combo2=None, timeout=20.0):
    before = set(nl._process_toplevel())
    dl = time.time() + timeout
    while time.time() < dl:
        new = [h for h in nl._process_toplevel() if h not in before]
        if new:
            dlg = new[0]
            time.sleep(0.2)
            title = win32gui.GetWindowText(dlg)
            ctrls = nl._dialog_controls_flat(dlg)
            edit = next((h for h, c, _, _ in ctrls if c == "Edit"), 0)
            combos = [h for h, c, _, _ in ctrls if c == "ComboBox"]
            okb = next((h for h, c, cid, t in ctrls if c == "Button"
                        and (t.replace('&', '').strip().lower() == "ok" or cid == 1)), 0)
            if edit:
                win32gui.SendMessage(edit, win32con.WM_SETTEXT, 0, str(value))
            if combo1 is not None and len(combos) >= 1:
                win32gui.SendMessage(combos[0], 0x014E, combo1, 0)  # CB_SETCURSEL
            if combo2 is not None and len(combos) >= 2:
                win32gui.SendMessage(combos[1], 0x014E, combo2, 0)
            time.sleep(0.12)
            if okb:
                win32gui.SendMessage(okb, BM_CLICK, 0, 0)
                return f"FILLED '{title}' val={value}"
            return f"NO-OK '{title}'"
        time.sleep(0.15)
    return "FILLER TIMEOUT"


def auto_click(nl, img_x, img_y, value, combo1=None, combo2=None):
    res = seek(nl, img_x, img_y)
    if not res:
        return False, "seek failed (no cursor readout)"
    sx, sy, ix, iy = res
    force_foreground(nl.cm.main_window)
    time.sleep(0.2)
    win32api.SetCursorPos((sx, sy)); time.sleep(0.15)
    out = {}
    th = threading.Thread(target=lambda: out.update(
        msg=fill_dialog(nl, value, combo1, combo2)))
    th.start()
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    th.join()
    msg = out.get("msg", "?")
    ok = msg.startswith("FILLED")
    return ok, f"cursor img=({ix},{iy}) target=({img_x},{img_y}) | {msg}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("img_x", type=int)
    ap.add_argument("img_y", type=int)
    ap.add_argument("value")
    ap.add_argument("--combo1", type=int, default=None)
    ap.add_argument("--combo2", type=int, default=None)
    args = ap.parse_args()
    nl = NeuraLog()
    ok, info = auto_click(nl, args.img_x, args.img_y, args.value,
                          args.combo1, args.combo2)
    print(("OK  " if ok else "FAIL ") + info)
    print("prompt after:", nl.get_prompt_text())
    st = nl.panel_state()
    print("scale_axes:", st["scale_axes"], "| depth:", st["depth_axes"])

"""
probe_panel.py — пошаговая разведка диалогов панели CURVES/SCALE AXIS.

Команды:
  python probe_panel.py state          — состояние панели (листбоксы, edits)
  python probe_panel.py addnew         — BM_CLICK 'Add New' (3019), дамп диалога
  python probe_panel.py dumpdlg        — дамп всех видимых top-level диалогов процесса
  python probe_panel.py cancel         — закрыть открытый диалог (кнопка Cancel/Escape)
  python probe_panel.py lbdump <hwnd>  — прочитать элементы ListBox (hex hwnd)
"""
import sys
import time
import ctypes
import win32gui
import win32con
import win32process

sys.path.insert(0, r"F:\nds\Auto")
from neuralog_win32 import NeuraLog, BM_CLICK, LB_GETCOUNT, LB_GETTEXT, LB_GETTEXTLEN

PANEL_IDS = {
    3011: "lb_depth_axes", 3012: "lb_scale_axes", 3013: "lb_curves",
    3019: "btn_add_new", 3028: "btn_add_backup", 3029: "btn_set_type",
    3056: "edit_depth_top", 3059: "edit_depth_bottom",
    3066: "edit_scale_left", 3068: "edit_scale_right", 3024: "combo_scale_units",
    3022: "radio_feet", 3023: "radio_meters",
}


def panel_controls(nl: NeuraLog) -> dict:
    """Найти контролы панели Curve/Track Dialog по dlg-ctrl-id."""
    res = {}
    root = nl.cm.main_window

    def visit(h, _):
        cid = win32gui.GetDlgCtrlID(h)
        if cid in PANEL_IDS:
            # отсечь дубликаты id (3368 в тулбаре и панели) — берём первый
            res.setdefault(PANEL_IDS[cid], h)
        return True

    win32gui.EnumChildWindows(root, visit, None)
    return res


def lb_items(nl: NeuraLog, hwnd: int) -> list:
    """Элементы ListBox: сперва LB_GETTEXT, при owner-draw — item-data."""
    n = win32gui.SendMessage(hwnd, LB_GETCOUNT, 0, 0)
    items = []
    for i in range(n):
        ln = win32gui.SendMessage(hwnd, LB_GETTEXTLEN, i, 0)
        text = None
        if 0 < ln < 256:
            buf = ctypes.create_unicode_buffer(ln + 1)
            ctypes.windll.user32.SendMessageW(hwnd, LB_GETTEXT, i, buf)
            text = buf.value
        if not text or not text.strip():
            text = nl._read_curve_item_name(hwnd, i) or f"<item {i}>"
        items.append(text)
    return items


def process_toplevel(nl: NeuraLog) -> list:
    _, pid = win32process.GetWindowThreadProcessId(nl.cm.main_window)
    out = []

    def cb(h, _):
        _, p = win32process.GetWindowThreadProcessId(h)
        if p == pid and win32gui.IsWindowVisible(h) and h != nl.cm.main_window:
            out.append(h)
        return True

    win32gui.EnumWindows(cb, None)
    return out


def dump_dialog(h: int):
    print(f"DIALOG 0x{h:08X} '{win32gui.GetWindowText(h)}' "
          f"class={win32gui.GetClassName(h)}")
    def visit(c, depth):
        cid = win32gui.GetDlgCtrlID(c)
        print(f"{'  '*depth}  0x{c:08X} id={cid:<6d} "
              f"{win32gui.GetClassName(c):<24s} '{win32gui.GetWindowText(c)[:50]}'")
        kids = []
        try:
            win32gui.EnumChildWindows(c, lambda x, _: kids.append(x) or True, None)
        except Exception:
            pass
        for k in [k for k in kids if win32gui.GetParent(k) == c]:
            visit(k, depth + 1)
    kids = []
    try:
        win32gui.EnumChildWindows(h, lambda x, _: kids.append(x) or True, None)
    except Exception:
        pass
    for k in [k for k in kids if win32gui.GetParent(k) == h]:
        visit(k, 1)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "state"
    nl = NeuraLog()
    if not nl.cm.main_window:
        print("NeuraLOG не найден")
        return 1
    pc = panel_controls(nl)

    if cmd == "state":
        for name, h in sorted(pc.items()):
            cls = win32gui.GetClassName(h)
            txt = win32gui.GetWindowText(h)
            print(f"{name:<20s} 0x{h:08X} {cls:<12s} '{txt}'")
            if cls == "ListBox":
                for i, it in enumerate(lb_items(nl, h)):
                    print(f"    [{i}] {it}")

    elif cmd == "addnew":
        before = set(process_toplevel(nl))
        win32gui.SendMessage(pc["btn_add_new"], BM_CLICK, 0, 0)
        time.sleep(1.0)
        new = [h for h in process_toplevel(nl) if h not in before]
        if not new:
            print("Новых окон нет. Текущие top-level:")
            new = process_toplevel(nl)
        for h in new:
            dump_dialog(h)
            # листбоксы диалога — содержимое
            lbs = []
            win32gui.EnumChildWindows(
                h, lambda x, _: lbs.append(x) or True, None)
            for lb in lbs:
                if win32gui.GetClassName(lb) == "ListBox":
                    print(f"  ListBox 0x{lb:08X} items:")
                    for i, it in enumerate(lb_items(nl, lb)):
                        print(f"    [{i}] {it}")

    elif cmd == "dumpdlg":
        for h in process_toplevel(nl):
            dump_dialog(h)

    elif cmd == "cancel":
        for h in process_toplevel(nl):
            btns = []
            win32gui.EnumChildWindows(
                h, lambda x, _: btns.append(x) or True, None)
            for b in btns:
                if (win32gui.GetClassName(b) == "Button"
                        and win32gui.GetWindowText(b).strip().lower()
                        in ("cancel", "отмена")):
                    print(f"Cancel в 0x{h:08X}")
                    win32gui.SendMessage(b, BM_CLICK, 0, 0)
                    break

    elif cmd == "lbdump":
        h = int(sys.argv[2], 16)
        for i, it in enumerate(lb_items(nl, h)):
            print(f"  [{i}] {it}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

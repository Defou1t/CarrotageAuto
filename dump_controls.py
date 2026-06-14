"""
dump_controls.py — полный дамп дерева контролов NeuraLOG.

Для каждого дочернего окна главного фрейма: HWND, класс, заголовок,
dlg-ctrl-id, видимость, rect, родитель. Цель — найти панели CURVES /
SCALE AXIS: кнопки Add New / Add Backup, Edit'ы Left/Right, combo единиц.

Запуск:  python dump_controls.py [фильтр-подстрока]
"""
import sys
import win32gui

sys.path.insert(0, r"F:\nds\Auto")
from neuralog_win32 import NeuraLog


def dump_tree(root: int, filt: str = ""):
    rows = []

    def visit(hwnd, depth):
        cls = win32gui.GetClassName(hwnd)
        title = win32gui.GetWindowText(hwnd)
        ctrl_id = win32gui.GetDlgCtrlID(hwnd)
        vis = win32gui.IsWindowVisible(hwnd)
        r = win32gui.GetWindowRect(hwnd)
        rows.append((depth, hwnd, cls, title, ctrl_id, vis, r))
        # дети
        kids = []
        try:
            win32gui.EnumChildWindows(hwnd, lambda h, _: kids.append(h) or True, None)
        except Exception:
            pass
        # EnumChildWindows рекурсивен — берём только прямых детей
        direct = [k for k in kids if win32gui.GetParent(k) == hwnd]
        for k in direct:
            visit(k, depth + 1)

    visit(root, 0)

    for depth, hwnd, cls, title, ctrl_id, vis, r in rows:
        line = (f"{'  '*depth}0x{hwnd:08X} id={ctrl_id:<6d} {cls:<28s} "
                f"{'vis' if vis else 'HID'} ({r[0]},{r[1]},{r[2]-r[0]}x{r[3]-r[1]}) "
                f"'{title[:60]}'")
        if not filt or filt.lower() in line.lower():
            print(line)


if __name__ == "__main__":
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    nl = NeuraLog()
    if not nl.cm.main_window:
        print("NeuraLOG не найден")
        sys.exit(1)
    print(f"=== дерево контролов 0x{nl.cm.main_window:08X} "
          f"'{win32gui.GetWindowText(nl.cm.main_window)}' ===")
    dump_tree(nl.cm.main_window, filt)

"""
Опознать command_id калибровочного инструмента, который СЕЙЧАС нажат (armed).

Использование:
  1. В NeuraLOG кликните нужную кнопку тулбара (например «Create Depth and
     Scale Axes»). Она станет нажатой.
  2. python detect_armed_tool.py
  Выведет command_id нажатой кнопки.
"""
import sys
sys.path.insert(0, r"F:\nds\Auto")
from neuralog_win32 import NeuraLog

nl = NeuraLog()
btns = nl.get_toolbar_buttons(nl.cm.toolbar_calibrate)
real = [b for b in btns if not b["separator"]]
print("Все кнопки калибр-тулбара (cmd | state | checked | pressed):")
hot = []
for b in real:
    st = b["state"]
    checked = bool(st & 0x01)
    pressed = bool(st & 0x02)
    if checked or pressed:
        hot.append((b["command_id"], checked, pressed))
    mark = "  <== АКТИВНА" if (checked or pressed) else ""
    print(f"  cmd {b['command_id']:>6} | state=0x{st:02X} | checked={checked} | pressed={pressed}{mark}")
print()
if hot:
    print("АКТИВНАЯ(ые) КОМАНДА(ы):", hot)
else:
    print("Ни одна кнопка не активна — кликните инструмент в NeuraLOG и повторите.")

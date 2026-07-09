"""
apply_patches.py — патчит neuralog_win32.py
Запуск: python apply_patches.py
"""
import sys
import shutil
from pathlib import Path

TARGET = Path(__file__).parent / "neuralog_win32.py"

if not TARGET.exists():
    print(f"ОШИБКА: файл не найден: {TARGET}")
    sys.exit(1)

backup = TARGET.with_suffix(".py.bak")
shutil.copy2(TARGET, backup)
print(f"Резервная копия: {backup}")

src = TARGET.read_text(encoding="utf-8")
original = src
applied = 0

PATCHES = [
    (
        "OpenProcess flags (get_toolbar_buttons)",
        """\
            hprocess = win32api.OpenProcess(
                0x0010 | 0x0020,  # PROCESS_VM_READ | PROCESS_VM_OPERATION
                False, pid
            )""",
        """\
            hprocess = win32api.OpenProcess(
                PROCESS_VM_OPERATION | PROCESS_VM_READ | PROCESS_VM_WRITE,
                False, pid
            )""",
    ),
    (
        "slider_tracer_speed — второй trackbar",
        """\
            elif cls == "msctls_trackbar32":
                if not self.cm.slider_tracer_speed:
                    self.cm.slider_tracer_speed = hwnd""",
        """\
            elif cls == "msctls_trackbar32":
                _trackbars.append(hwnd)""",
    ),
    (
        "get_status — GetWindowText",
        """\
    def get_status(self) -> str:
        \"\"\"Текст статус-бара (Ready / Tracing / ...).\"\"\"
        if self.cm.status_bar:
            return win32gui.GetWindowText(self.cm.status_bar)
        return \"\"""",
        None,  # replaced by full rewrite — skip if already has SB_GETTEXTW
    ),
]

for name, old, new in PATCHES:
    if new is None:
        if "SB_GETTEXTW" in src:
            print(f"⚠ {name}: уже применён")
        continue
    if old in src:
        src = src.replace(old, new, 1)
        print(f"✓ {name}")
        applied += 1
    else:
        print(f"⚠ {name}: пропущен (уже применён или не найден)")

if src != original:
    TARGET.write_text(src, encoding="utf-8")
    print(f"\n✓ Файл обновлён: {TARGET} ({applied} патч(ей))")
else:
    print("\n— Файл не изменён (все патчи уже применены)")

print("\nГотово. Запустите: python test_buttons.py")

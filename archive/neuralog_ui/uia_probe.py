"""
uia_probe.py — Диагностика UIAutomation / MSAA для NeuraLOG v2015
Запускать ПОСЛЕ того как NeuraLOG открыт.

Установка:  pip install pywinauto comtypes
Запуск:     python uia_probe.py
Вывод:      uia_probe_report.txt  (в той же папке)
"""

import sys
import os
import time
import json
from datetime import datetime
from io import StringIO

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uia_probe_report.txt")

def section(title, f):
    f.write("\n" + "=" * 60 + "\n")
    f.write(f"  {title}\n")
    f.write("=" * 60 + "\n")


def try_connect_backends(f):
    """Пробуем оба бэкенда pywinauto — uia и win32"""
    from pywinauto import Application, findwindows

    results = {}

    section("1. ПОИСК ОКОН NeuraLOG", f)

    # Найти все окна с NeuraLOG в заголовке
    try:
        handles = findwindows.find_windows(title_re=".*[Nn]eura.*")
        f.write(f"Найдено окон по 'Neura*': {len(handles)}\n")
        for h in handles:
            from pywinauto import win32functions
            import win32gui
            title = win32gui.GetWindowText(h)
            cls   = win32gui.GetClassName(h)
            f.write(f"  HWND=0x{h:08X}  class='{cls}'  title='{title}'\n")
        results["windows_found"] = len(handles)
    except Exception as e:
        f.write(f"  Ошибка поиска: {e}\n")
        results["windows_found"] = 0

    # ── UIA backend ──────────────────────────────────────────────
    section("2. UIA BACKEND (современный, рекомендуется)", f)
    uia_app = None
    try:
        uia_app = Application(backend="uia").connect(title_re=".*[Nn]eura.*", timeout=5)
        f.write("✓ UIA backend подключился успешно\n")
        results["uia"] = True
    except Exception as e:
        f.write(f"✗ UIA backend: {e}\n")
        results["uia"] = False

    if uia_app:
        try:
            dlg = uia_app.top_window()
            f.write(f"  Главное окно: '{dlg.window_text()}'\n")
            f.write(f"  Класс: '{dlg.class_name()}'\n")
            f.write(f"  Rect: {dlg.rectangle()}\n\n")
            f.write("  --- Дерево контролов (UIA) ---\n")
            buf = StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            dlg.print_control_identifiers(depth=4)
            sys.stdout = old_stdout
            tree_text = buf.getvalue()
            f.write(tree_text[:8000])  # первые 8000 символов
            if len(tree_text) > 8000:
                f.write("\n  ... (обрезано, полный вывод в консоли)\n")
            results["uia_tree_len"] = len(tree_text)
        except Exception as e:
            f.write(f"  Ошибка дерева UIA: {e}\n")

    # ── win32 backend ─────────────────────────────────────────────
    section("3. WIN32 BACKEND (классический MFC/WinForms)", f)
    w32_app = None
    try:
        w32_app = Application(backend="win32").connect(title_re=".*[Nn]eura.*", timeout=5)
        f.write("✓ win32 backend подключился успешно\n")
        results["win32"] = True
    except Exception as e:
        f.write(f"✗ win32 backend: {e}\n")
        results["win32"] = False

    if w32_app:
        try:
            dlg = w32_app.top_window()
            f.write(f"  Главное окно: '{dlg.window_text()}'\n")
            f.write(f"  Класс: '{dlg.class_name()}'\n\n")
            f.write("  --- Дерево контролов (win32) ---\n")
            buf = StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            dlg.print_control_identifiers(depth=4)
            sys.stdout = old_stdout
            tree_text = buf.getvalue()
            f.write(tree_text[:8000])
            results["win32_tree_len"] = len(tree_text)
        except Exception as e:
            f.write(f"  Ошибка дерева win32: {e}\n")

    return results, uia_app, w32_app


def probe_controls(app, backend_name, f):
    """Ищем кнопки, меню, тулбары, диалоги"""
    if app is None:
        return

    section(f"4. ДЕТАЛЬНЫЙ ПОИСК КОНТРОЛОВ [{backend_name}]", f)

    from pywinauto import findwindows
    import win32gui

    try:
        dlg = app.top_window()

        # Кнопки
        try:
            buttons = dlg.children(control_type="Button")
            f.write(f"Кнопок найдено: {len(buttons)}\n")
            for b in buttons[:20]:
                f.write(f"  Button: '{b.window_text()}'\n")
        except:
            pass

        # Меню
        try:
            menus = dlg.children(control_type="MenuBar")
            f.write(f"\nMenuBar: {len(menus)}\n")
            for m in menus:
                items = m.children()
                for item in items:
                    f.write(f"  MenuItem: '{item.window_text()}'\n")
        except:
            pass

        # ToolBar
        try:
            toolbars = dlg.descendants(control_type="ToolBar")
            f.write(f"\nToolBar: {len(toolbars)}\n")
            for tb in toolbars:
                f.write(f"  ToolBar items: {[b.window_text() for b in tb.children()]}\n")
        except:
            pass

        # TreeView (дерево файлов/проектов)
        try:
            trees = dlg.descendants(control_type="Tree")
            f.write(f"\nTreeView: {len(trees)}\n")
        except:
            pass

        # ListView (список файлов)
        try:
            lists = dlg.descendants(control_type="List")
            f.write(f"ListView: {len(lists)}\n")
        except:
            pass

        # Edit поля
        try:
            edits = dlg.descendants(control_type="Edit")
            f.write(f"Edit fields: {len(edits)}\n")
            for e in edits[:5]:
                f.write(f"  Edit: '{e.window_text()}'\n")
        except:
            pass

    except Exception as e:
        f.write(f"Ошибка probe_controls: {e}\n")


def probe_win32_classes(f):
    """Прямой обход Win32 окон — работает всегда, без pywinauto"""
    section("5. WIN32 ПРЯМОЙ ОБХОД (win32gui)", f)

    import win32gui
    import win32con

    neura_windows = []

    def enum_callback(hwnd, results):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        cls   = win32gui.GetClassName(hwnd)
        if "neura" in title.lower() or "neura" in cls.lower():
            rect = win32gui.GetWindowRect(hwnd)
            results.append({
                "hwnd": f"0x{hwnd:08X}",
                "title": title,
                "class": cls,
                "rect": rect
            })
        return True

    win32gui.EnumWindows(enum_callback, neura_windows)
    f.write(f"Окна NeuraLOG через EnumWindows: {len(neura_windows)}\n")
    for w in neura_windows:
        f.write(f"  {w}\n")

    # Дочерние окна
    section("5b. ДОЧЕРНИЕ ОКНА (child controls)", f)

    all_children = []

    def enum_children(hwnd, param):
        title = win32gui.GetWindowText(hwnd)
        cls   = win32gui.GetClassName(hwnd)
        rect  = win32gui.GetWindowRect(hwnd)
        all_children.append({
            "hwnd": f"0x{hwnd:08X}",
            "class": cls,
            "title": title[:60],
            "rect": rect
        })
        return True

    if neura_windows:
        # берём первое найденное окно NeuraLOG
        root_hwnd = int(neura_windows[0]["hwnd"], 16)
        win32gui.EnumChildWindows(root_hwnd, enum_children, None)
        f.write(f"Дочерних контролов: {len(all_children)}\n\n")

        # Группируем по классу
        from collections import Counter
        class_counter = Counter(c["class"] for c in all_children)
        f.write("Классы контролов:\n")
        for cls, cnt in class_counter.most_common():
            f.write(f"  {cnt:3d}x  {cls}\n")

        f.write("\nПервые 40 контролов:\n")
        for c in all_children[:40]:
            f.write(f"  hwnd={c['hwnd']}  class={c['class']:<30s}  title='{c['title']}'\n")

    return neura_windows


def probe_keyboard_shortcuts(app, f):
    """Ищем доступные горячие клавиши через меню"""
    section("6. АНАЛИЗ МЕНЮ И ГОРЯЧИХ КЛАВИШ", f)
    if app is None:
        f.write("  App не подключен\n")
        return

    try:
        dlg = app.top_window()
        menu = dlg.menu()
        if menu:
            def recurse_menu(m, depth=0):
                indent = "  " * depth
                for i in range(m.item_count()):
                    try:
                        item = m.item(i)
                        text = item.text()
                        f.write(f"{indent}[{i}] {text}\n")
                        if item.sub_menu():
                            recurse_menu(item.sub_menu(), depth + 1)
                    except:
                        pass
            recurse_menu(menu)
        else:
            f.write("  Меню недоступно через UIA\n")
    except Exception as e:
        f.write(f"  Ошибка: {e}\n")


def main():
    print(f"=== UIAutomation Probe для NeuraLOG ===")
    print(f"Вывод: {OUTPUT_FILE}\n")

    # Проверка зависимостей
    missing = []
    for pkg in ["pywinauto", "win32gui"]:
        try:
            __import__(pkg if pkg != "win32gui" else "win32gui")
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Не установлено: {missing}")
        print("pip install pywinauto pywin32")
        sys.exit(1)

    import win32gui

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"UIAutomation Probe Report\n")
        f.write(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Python: {sys.version}\n")

        # 1. Поиск + бэкенды
        results, uia_app, w32_app = try_connect_backends(f)

        # 2. Детальные контролы
        active_app = uia_app or w32_app
        backend_name = "uia" if uia_app else "win32"
        probe_controls(active_app, backend_name, f)

        # 3. Win32 прямой обход (всегда работает)
        neura_windows = probe_win32_classes(f)

        # 4. Меню
        probe_keyboard_shortcuts(active_app, f)

        # Итог
        section("ИТОГ", f)
        f.write(f"UIA backend:   {'✓ работает' if results.get('uia') else '✗ недоступен'}\n")
        f.write(f"win32 backend: {'✓ работает' if results.get('win32') else '✗ недоступен'}\n")
        f.write(f"Окон найдено:  {results.get('windows_found', 0)}\n")
        f.write(f"Win32 окна:    {len(neura_windows)}\n")

        if results.get("uia"):
            f.write("\n→ РЕКОМЕНДАЦИЯ: Используйте UIA backend (pywinauto, backend='uia')\n")
            f.write("  Контролы доступны по имени/automationId без координат\n")
        elif results.get("win32"):
            f.write("\n→ РЕКОМЕНДАЦИЯ: Используйте win32 backend\n")
            f.write("  MFC-контролы доступны по классу/тексту\n")
        elif neura_windows:
            f.write("\n→ РЕКОМЕНДАЦИЯ: UIAutomation недоступен\n")
            f.write("  Используйте win32gui.SendMessage/PostMessage по HWND\n")
            f.write("  или продолжайте pyautogui + OpenCV\n")
        else:
            f.write("\n→ NeuraLOG не найден. Убедитесь что программа запущена.\n")

    print(f"✓ Готово. Откройте: {OUTPUT_FILE}")
    print(f"\nКраткий итог:")
    print(f"  UIA backend:   {'✓' if results.get('uia') else '✗'}")
    print(f"  win32 backend: {'✓' if results.get('win32') else '✗'}")
    print(f"  Окон найдено:  {results.get('windows_found', 0)}")


if __name__ == "__main__":
    main()

"""Скрипт для глубокого сканирования открытых диалоговых окон NeuraLOG."""
import win32gui
import win32process

def main():
    print("=== СКАНИРОВАНИЕ ОТКРЫТЫХ ДИАЛОГОВ NEURALOG ===")
    
    # 1. Найти главное окно NeuraLOG
    main_hwnd = 0
    def find_main(hwnd, _):
        nonlocal main_hwnd
        title = win32gui.GetWindowText(hwnd)
        cls = win32gui.GetClassName(hwnd)
        if "NeuraLog" in title and cls.startswith("Afx:"):
            main_hwnd = hwnd
        return True
    win32gui.EnumWindows(find_main, None)
    
    if not main_hwnd:
        print("❌ NeuraLOG не найден! Запустите NDSLOG.EXE.")
        return
        
    _, pid = win32process.GetWindowThreadProcessId(main_hwnd)
    
    # 2. Найти все видимые диалоги, принадлежащие этому процессу
    dialogs = []
    def find_dialogs(hwnd, _):
        _, wpid = win32process.GetWindowThreadProcessId(hwnd)
        if wpid == pid and win32gui.IsWindowVisible(hwnd):
            cls = win32gui.GetClassName(hwnd)
            # Ищем стандартные диалоги Windows или окна мастера
            if cls == "#32770" or "Getting Started" in win32gui.GetWindowText(hwnd):
                dialogs.append(hwnd)
        return True
    win32gui.EnumWindows(find_dialogs, None)
    
    if not dialogs:
        print("❌ Открытых диалоговых окон не найдено.")
        print("👉 ИНСТРУКЦИЯ: Откройте окно 'File -> New' (или любой другой нужный диалог) вручную в NeuraLOG и запустите скрипт снова.")
        return
        
    # 3. Сканируем внутренности найденных диалогов
    for dlg in dialogs:
        title = win32gui.GetWindowText(dlg)
        print(f"\n🎯 НАЙДЕН ДИАЛОГ: '{title}' (HWND: 0x{dlg:08X})")
        print("-" * 70)
        print(f"  {'ID':<6} | {'КЛАСС':<18} | {'ТЕКСТ ЭЛЕМЕНТА'}")
        print("-" * 70)
        
        def enum_child(hwnd, _):
            cls = win32gui.GetClassName(hwnd)
            txt = win32gui.GetWindowText(hwnd)
            cid = win32gui.GetDlgCtrlID(hwnd)
            
            # Очистка текста от переносов строк для красивого вывода
            txt_clean = txt.replace('\n', ' ').replace('\r', '')
            print(f"  {cid:<6} | {cls:<18} | '{txt_clean[:60]}'")
            return True
            
        win32gui.EnumChildWindows(dlg, enum_child, None)
        print("-" * 70)

if __name__ == "__main__":
    main()
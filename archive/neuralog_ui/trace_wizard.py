"""Скрипт сквозного прохода мастера создания проекта NeuraLOG (Шаг 1 -> Шаг 2 -> Шаг 3)."""
import win32gui
import win32con
import win32process
import time

TEST_IMAGE_PATH = r"F:\nds\projects\Yatskivska_001\img\Yatskivska_1_AK_3180_3520.tif"

def find_dialog_by_title(substring):
    found_hwnd = 0
    def enum_cb(hwnd, _):
        nonlocal found_hwnd
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if substring.lower() in title.lower():
                found_hwnd = hwnd
                return False
        return True
    win32gui.EnumWindows(enum_cb, None)
    return found_hwnd

def find_child_by_id(parent_hwnd, target_id):
    found_hwnd = 0
    def enum_child(hwnd, _):
        nonlocal found_hwnd
        if win32gui.GetDlgCtrlID(hwnd) == target_id:
            found_hwnd = hwnd
            return False
        return True
    try:
        win32gui.EnumChildWindows(parent_hwnd, enum_child, None)
    except Exception:
        pass
    return found_hwnd

def scan_window(hwnd):
    title = win32gui.GetWindowText(hwnd)
    print(f"\n🎯 СТРУКТУРА ТЕКУЩЕГО ОКНА: '{title}' (HWND: 0x{hwnd:08X})")
    print("-" * 75)
    print(f"  {'ID':<6} | {'КЛАСС':<20} | {'ТЕКСТ ЭЛЕМЕНТА'}")
    print("-" * 75)
    
    def enum_child(child_hwnd, _):
        cls = win32gui.GetClassName(child_hwnd)
        txt = win32gui.GetWindowText(child_hwnd)
        cid = win32gui.GetDlgCtrlID(child_hwnd)
        txt_clean = txt.replace('\n', ' ').replace('\r', '')
        print(f"  {cid:<6} | {cls:<20} | '{txt_clean[:55]}'")
        return True
        
    win32gui.EnumChildWindows(hwnd, enum_child, None)
    print("-" * 75)

def handle_open_file_dialog(parent_pid, file_path):
    print("[Робот]: Ожидаю появление системного окна 'Открыть'...")
    for _ in range(40):
        time.sleep(0.1)
        hwnd = win32gui.GetForegroundWindow()
        if win32gui.IsWindowVisible(hwnd):
            cls = win32gui.GetClassName(hwnd)
            title = win32gui.GetWindowText(hwnd)
            # Убрали "select", оставили только точные совпадения для окна открытия
            if cls == "#32770" and title.lower() in ["open", "открыть", "open document"]:
                _, wpid = win32process.GetWindowThreadProcessId(hwnd)
                if wpid == parent_pid:
                    print(f"[Робот]: Окно выбора файла обнаружено: '{title}'")
                    
                    edit_hwnd = win32gui.FindWindowEx(hwnd, 0, "Edit", None)
                    if not edit_hwnd:
                        combobox_hwnd = win32gui.FindWindowEx(hwnd, 0, "ComboBoxEx32", None)
                        if combobox_hwnd:
                            edit_hwnd = win32gui.FindWindowEx(combobox_hwnd, 0, "Edit", None)
                            
                    if edit_hwnd:
                        print(f"[Робот]: Вставляю абсолютный путь (авто-переход в нужную папку)...")
                        win32gui.SendMessage(edit_hwnd, win32con.WM_SETTEXT, 0, file_path)
                        time.sleep(0.5)
                        
                        win32gui.SendMessage(edit_hwnd, win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)
                        win32gui.SendMessage(edit_hwnd, win32con.WM_KEYUP, win32con.VK_RETURN, 0)
                        print("[Робот]: Путь подтвержден.")
                        
                        # --- ОБРАБОТКА ВСПЛЫВАЮЩЕГО ОКНА BROWSE FILE ACTIONS ---
                        print("[Робот]: Проверяю наличие окна 'Browse File Actions'...")
                        time.sleep(1.0) # Даем время на появление окна
                        popup_hwnd = win32gui.GetForegroundWindow()
                        popup_title = win32gui.GetWindowText(popup_hwnd)
                        if "Browse File" in popup_title or "Action" in popup_title:
                            print(f"[Робот]: Обнаружено блокирующее окно: '{popup_title}'. Нажимаю OK...")
                            # ID 1 - это стандартный ID для кнопки OK в Windows
                            btn_ok = win32gui.GetDlgItem(popup_hwnd, 1)
                            if btn_ok:
                                win32gui.SendMessage(btn_ok, win32con.BM_CLICK, 0, 0)
                                time.sleep(0.5)
                            else:
                                # Если ID 1 не сработал, шлем просто Enter окну
                                win32gui.SendMessage(popup_hwnd, win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)
                                win32gui.SendMessage(popup_hwnd, win32con.WM_KEYUP, win32con.VK_RETURN, 0)
                        return True
    return False

def main():
    print("=== УМНЫЙ СКВОЗНОЙ РОБОТ-РАЗВЕДЧИК МАСТЕРА ===")
    wizard_hwnd = find_dialog_by_title("New Well Log")
    if not wizard_hwnd:
        print("❌ Окно мастера 'New Well Log...' не найдено на экране.")
        return
        
    title = win32gui.GetWindowText(wizard_hwnd)
    print(f"✓ Обнаружен мастер на этапе: '{title}'")
    _, pid = win32process.GetWindowThreadProcessId(wizard_hwnd)
    
    if "Select Project" in title:
        print("[Робот]: Обнаружен Шаг 1. Перехожу на Шаг 2...")
        btn_next = find_child_by_id(wizard_hwnd, 12324)
        if btn_next:
            win32gui.SendMessage(btn_next, win32con.BM_CLICK, 0, 0)
            time.sleep(1.5)
            wizard_hwnd = find_dialog_by_title("New Well Log")
            title = win32gui.GetWindowText(wizard_hwnd)
            print(f"✓ Мастер переключился на этап: '{title}'")
            
    if "Image Select" in title:
        print("[Робот]: Начинаю автоматизацию выбора файла изображения...")
        btn_browse = find_child_by_id(wizard_hwnd, 3034)
        if btn_browse:
            win32gui.SendMessage(btn_browse, win32con.BM_CLICK, 0, 0)
            if handle_open_file_dialog(pid, TEST_IMAGE_PATH):
                time.sleep(1.0)
            
        btn_next = find_child_by_id(wizard_hwnd, 12324)
        if btn_next:
            print("[Робот]: Нажимаю кнопку 'Далее >' для выхода на ШАГ 3...")
            win32gui.SendMessage(btn_next, win32con.BM_CLICK, 0, 0)
            time.sleep(2.0)
            
    wizard_hwnd = find_dialog_by_title("New Well Log")
    if wizard_hwnd:
        scan_window(wizard_hwnd)
    else:
        print("❌ Окно мастера исчезло.")

if __name__ == "__main__":
    main()
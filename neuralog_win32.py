"""
neuralog_win32.py — Контроллер NeuraLOG v2015 через win32 API
=============================================================
Основан на данных uia_probe_report.txt:
  - Приложение: 32-bit MFC (Afx:400000:... классы)
  - UIA: подключается но дерево недоступно → НЕ используем
  - win32 backend: 64 контрола видны, всё работает
  - Ключевые HWND: BTN_FORWARD/STOP/REVERSE, Tracer Speed, Current Depth

Использование:
    from neuralog_win32 import NeuraLog
    nl = NeuraLog()
    nl.ensure_visible()
    nl.start_trace()
    depth = nl.get_current_depth()
    nl.stop_trace()
    nl.save()
    cmd = nl.probe_export_command()   # найти ID экспорта один раз
    nl.export_las(r"F:\\nds\\output\\well.las", cmd_id=cmd)

Установка: pip install pywinauto pywin32

── Лог изменений ────────────────────────────────────────────
v2  (08.06.2026):
  FIX  _read_status_part — убран WriteProcessMemory, исправлен
       offset текста (было raw[2:], нужен raw[0:]).
  FIX  open_project — typewrite заменён на clipboard + Ctrl+V:
       typewrite не умеет вводить '\\' и ':' в путях Windows.
  FIX  _read_curve_item_name — защита от item-data < 0x10000
       (целочисленные значения, не указатели → None).
  NEW  _clipboard_set — вставка через win32clipboard / pyperclip.
  NEW  get_depth_slider_position — читает первый trackbar (позиция).
  NEW  _find_save_dialog — ищет открытый стандартный Save As диалог.
  NEW  export_las(output_path, cmd_id) — полный экспорт с диалогом.
  NEW  probe_export_command — авто-определение command_id экспорта.
"""

import time
import os
import re
import ctypes
import ctypes.wintypes
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import win32gui
import win32con
import win32api
import win32process
from pywinauto import Application
from pywinauto.controls.win32_controls import ButtonWrapper

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# WIN32 КОНСТАНТЫ
# ─────────────────────────────────────────────────────────────────
TB_BUTTONCOUNT   = 0x0418
TB_GETBUTTON     = 0x0417
TB_PRESSBUTTON   = 0x0408
TB_GETBUTTONTEXT = 0x042D
WM_COMMAND       = 0x0111
BM_CLICK         = 0x00F5
TBM_SETPOS       = 0x0405
TBM_GETPOS       = 0x0400
TBM_SETRANGE     = 0x0406
LB_GETCOUNT      = 0x018B
LB_GETTEXT       = 0x0189
LB_GETTEXTLEN    = 0x018A
LB_GETITEMDATA   = 0x0199
LB_SETCURSEL     = 0x0186
LB_GETCURSEL     = 0x0188
SB_GETTEXTW      = 0x040D
SB_GETPARTS      = 0x0406

PROCESS_VM_OPERATION = 0x0008
PROCESS_VM_READ      = 0x0010
PROCESS_VM_WRITE     = 0x0020
MEM_COMMIT           = 0x1000
MEM_RESERVE          = 0x2000
MEM_RELEASE          = 0x8000
PAGE_READWRITE       = 0x04


@dataclass
class ControlMap:
    """HWND всех ключевых контролов NeuraLOG (из probe-отчёта)."""
    main_window:         int = 0
    mdi_client:          int = 0
    status_bar:          int = 0
    # Toolbar panels
    toolbar_main:        int = 0
    toolbar_calibrate:   int = 0
    toolbar_curve_edit:  int = 0
    toolbar_curve_tools: int = 0
    toolbar_zoom:        int = 0
    toolbar_color:       int = 0
    toolbar_qc:          int = 0
    # Tracer controls
    btn_forward:         int = 0
    btn_stop:            int = 0
    btn_reverse:         int = 0
    slider_depth_position: int = 0  # ползунок позиции по глубине
    slider_tracer_speed: int = 0    # ползунок скорости трейсера
    lbl_current_depth:   int = 0   # Static "Current Depth:"
    val_current_depth:   int = 0   # Static "3013"  ← значение
    # Color / mode
    combo_color_mode:    int = 0
    edit_color_value:    int = 0
    # Curve/Track dialog
    panel_curve_dialog:  int = 0
    listbox_curves:      int = 0
    btn_hide_grid:       int = 0
    btn_remove_curve:    int = 0
    btn_units:           int = 0
    edit_top:            int = 0
    edit_bottom:         int = 0
    # User prompt / log
    user_prompt:         int = 0
    richedit_log:        int = 0


class NeuraLog:
    """Высокоуровневый контроллер NeuraLOG v2015."""

    WINDOW_TITLE_RE = r".*[Nn]eura[Ll][Oo][Gg].*"

    def __init__(self, auto_connect: bool = True):
        self.app: Optional[Application] = None
        self.dlg = None
        self.cm  = ControlMap()
        if auto_connect:
            self.connect()

    # ─────────────────────────────────────────────────────────────
    # ПОДКЛЮЧЕНИЕ И ПОИСК КОНТРОЛОВ
    # ─────────────────────────────────────────────────────────────

    def connect(self, timeout: float = 10.0) -> bool:
        """
        Подключиться к запущенному NeuraLOG.
        Стратегия: сначала ищем HWND через win32gui (обходит проблему
        'multiple elements' и не требует видимости окна), затем цепляемся
        по handle — это надёжнее чем title_re при наличии нескольких окон.
        """
        main_hwnd = self._find_main_hwnd()
        if not main_hwnd:
            log.error("NeuraLog главное окно не найдено через win32gui")
            return False

        try:
            # Подключаемся напрямую по handle — без поиска по заголовку
            self.app = Application(backend="win32").connect(handle=main_hwnd)
            self.dlg = self.app.top_window()
            self.cm.main_window = main_hwnd

            # Восстановить окно если оно за экраном / свёрнуто
            self._restore_window(main_hwnd)

            self._build_control_map()
            log.info(f"NeuraLog подключён: HWND=0x{main_hwnd:08X} "
                     f"\'{win32gui.GetWindowText(main_hwnd)}\'")
            return True
        except Exception as e:
            log.error(f"Ошибка подключения по handle: {e}")
            return False

    def _find_main_hwnd(self) -> int:
        """
        Найти главное окно NeuraLOG через EnumWindows.
        Критерии главного окна:
          - заголовок содержит NeuraLog
          - класс начинается с Afx: (MFC main frame)
          - имеет дочерний MDIClient (признак главного MDI-окна)
        Возвращает HWND или 0.
        """
        candidates = []

        def callback(hwnd, _):
            title = win32gui.GetWindowText(hwnd)
            cls   = win32gui.GetClassName(hwnd)
            if not re.search(r"[Nn]eura[Ll]og", title):
                return True
            # Только MFC frame-окна, не диалоги
            if not cls.startswith("Afx:"):
                return True
            # Проверить наличие MDIClient среди дочерних — признак главного окна
            has_mdi = []
            def check_mdi(h, _):
                if win32gui.GetClassName(h) == "MDIClient":
                    has_mdi.append(h)
                return True
            try:
                win32gui.EnumChildWindows(hwnd, check_mdi, None)
            except Exception:
                pass
            if has_mdi:
                candidates.append(hwnd)
            return True

        win32gui.EnumWindows(callback, None)

        if not candidates:
            log.warning("_find_main_hwnd: кандидаты не найдены")
            return 0
        if len(candidates) == 1:
            log.debug(f"Найдено главное окно: 0x{candidates[0]:08X}")
            return candidates[0]

        # Несколько кандидатов — выбираем с самым длинным заголовком
        # (главное окно содержит путь к проекту, остальные — нет)
        best = max(candidates, key=lambda h: len(win32gui.GetWindowText(h)))
        log.debug(f"Несколько кандидатов ({len(candidates)}), выбрано: "
                  f"0x{best:08X} \'{win32gui.GetWindowText(best)}\'")
        return best

    def _restore_window(self, hwnd: int):
        """
        Вернуть окно на экран если оно свёрнуто или за пределами монитора.
        Координаты (-32000, -32000) — стандартный Windows признак свёрнутого окна.
        """
        rect = win32gui.GetWindowRect(hwnd)
        x, y = rect[0], rect[1]
        off_screen = (x <= -31000 or y <= -31000)
        minimized  = win32gui.IsIconic(hwnd)

        if minimized or off_screen:
            log.info(f"Окно свёрнуто/за экраном (rect={rect}), восстанавливаем...")
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.4)
            # Проверить повторно — если всё ещё за экраном, переместить
            rect2 = win32gui.GetWindowRect(hwnd)
            if rect2[0] <= -31000 or rect2[1] <= -31000:
                win32gui.SetWindowPos(
                    hwnd, win32con.HWND_TOP,
                    100, 100, 1400, 900,
                    win32con.SWP_SHOWWINDOW
                )
                log.info("Окно перемещено в (100, 100, 1400x900)")
            time.sleep(0.2)

    def _build_control_map(self):
        """Обойти дочерние окна и заполнить ControlMap по классу и заголовку."""
        hwnd_root = self.cm.main_window
        _trackbars = []
        _listboxes = []

        def visit(hwnd, _):
            title = win32gui.GetWindowText(hwnd)
            cls   = win32gui.GetClassName(hwnd)
            key = f"{cls}|{title}"

            # Главные панели
            if cls == "MDIClient":
                self.cm.mdi_client = hwnd
            elif cls == "msctls_statusbar32":
                self.cm.status_bar = hwnd

            # Тулбары (ToolbarWindow32) — по заголовку
            elif cls == "ToolbarWindow32":
                t = title.strip()
                if t == "Main":
                    self.cm.toolbar_main = hwnd
                elif t == "Calibrate & Trace":
                    self.cm.toolbar_calibrate = hwnd
                elif t == "Curve Edit":
                    self.cm.toolbar_curve_edit = hwnd
                elif t == "Curve/Track Tools":
                    self.cm.toolbar_curve_tools = hwnd
                elif t == "Zoom":
                    self.cm.toolbar_zoom = hwnd
                elif t == "Color Selection":
                    self.cm.toolbar_color = hwnd
                elif t == "Quality Control":
                    self.cm.toolbar_qc = hwnd

            # Кнопки трейсера
            elif cls == "Button":
                if title == "BTN_FORWARD":
                    self.cm.btn_forward = hwnd
                elif title == "BTN_STOP":
                    self.cm.btn_stop = hwnd
                elif title == "BTN_REVERSE":
                    self.cm.btn_reverse = hwnd
                elif title == "Hide Grid":
                    self.cm.btn_hide_grid = hwnd
                elif title == "Remove":
                    self.cm.btn_remove_curve = hwnd
                elif title == "Units":
                    self.cm.btn_units = hwnd

            # Trackbar (скорость трейсера)
            elif cls == "msctls_trackbar32":
                _trackbars.append(hwnd) 

            # Static метки
            elif cls == "Static":
                if title == "Current Depth:":
                    self.cm.lbl_current_depth = hwnd
                elif self.cm.lbl_current_depth and not self.cm.val_current_depth:
                    # Static сразу после "Current Depth:" — это значение
                    self.cm.val_current_depth = hwnd

            # ComboBox Color Mode
            elif cls == "ComboBox":
                if not self.cm.combo_color_mode:
                    self.cm.combo_color_mode = hwnd

            # ListBox — собираем все, выберем лучший после обхода
            elif cls == "ListBox":
                _listboxes.append(hwnd)

            # Edit поля (Top/Bottom глубин)
            elif cls == "Edit":
                # Не перезаписываем color edit
                pass

            # Кастомные MFC панели
            elif cls.startswith("Afx:"):
                if title == "Curve/Track Dialog":
                    self.cm.panel_curve_dialog = hwnd
                elif "100063" in cls:   # User prompt
                    self.cm.user_prompt = hwnd

            # RICHEDIT — лог
            elif cls == "RICHEDIT":
                self.cm.richedit_log = hwnd

            return True
        



        win32gui.EnumChildWindows(hwnd_root, visit, None)

        if len(_trackbars) >= 2:
            # первый — ползунок позиции глубины, второй — скорость трейсера
            self.cm.slider_depth_position = _trackbars[0]
            self.cm.slider_tracer_speed = _trackbars[1]
        elif len(_trackbars) == 1:
            self.cm.slider_tracer_speed = _trackbars[0]

        self._assign_listbox_curves(_listboxes)

        # Вытащить Edit'ы внутри Curve/Track Dialog
        if self.cm.panel_curve_dialog:
            edits = []
            def collect_edits(h, _):
                if win32gui.GetClassName(h) == "Edit":
                    edits.append(h)
                return True
            win32gui.EnumChildWindows(self.cm.panel_curve_dialog, collect_edits, None)
            if len(edits) >= 2:
                self.cm.edit_top    = edits[0]
                self.cm.edit_bottom = edits[1]

        self._log_control_map()

    def _log_control_map(self):
        cm = self.cm
        log.debug("=== ControlMap ===")
        for attr, val in vars(cm).items():
            if val:
                log.debug(f"  {attr:<25s}: 0x{val:08X}")

    def _assign_listbox_curves(self, listboxes: List[int]):
        """Выбрать ListBox с наибольшим числом кривых (owner-draw, в Curve/Track Dialog)."""
        if not listboxes:
            return

        candidates = listboxes
        if self.cm.panel_curve_dialog:
            panel_lbs = []
            def collect_lb(h, _):
                if win32gui.GetClassName(h) == "ListBox":
                    panel_lbs.append(h)
                return True
            win32gui.EnumChildWindows(self.cm.panel_curve_dialog, collect_lb, None)
            if panel_lbs:
                candidates = panel_lbs

        self.cm.listbox_curves = max(
            candidates,
            key=lambda h: win32gui.SendMessage(h, LB_GETCOUNT, 0, 0),
        )

    def _open_process_for_hwnd(self, hwnd: int, write: bool = False):
        """Открыть handle процесса окна для cross-process чтения/записи."""
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        flags = PROCESS_VM_OPERATION | PROCESS_VM_READ
        if write:
            flags |= PROCESS_VM_WRITE
        return win32api.OpenProcess(flags, False, pid)

    def _with_remote_buffer(self, hwnd: int, size: int, write: bool = False):
        """
        Контекстный менеджер: VirtualAllocEx в процессе окна.
        Возвращает (hprocess, remote_addr). Вызывающий закрывает handle и освобождает буфер.
        """
        class _RemoteBuf:
            def __init__(self, outer, hwnd, size, write):
                self.outer = outer
                self.hwnd = hwnd
                self.size = size
                self.write = write
                self.hproc = None
                self.remote = None

            def __enter__(self):
                self.hproc = self.outer._open_process_for_hwnd(self.hwnd, self.write)
                self.remote = ctypes.windll.kernel32.VirtualAllocEx(
                    int(self.hproc), None, self.size,
                    MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE,
                )
                if not self.remote:
                    win32api.CloseHandle(self.hproc)
                    raise RuntimeError("VirtualAllocEx failed")
                return self.hproc, self.remote

            def __exit__(self, *args):
                if self.remote:
                    ctypes.windll.kernel32.VirtualFreeEx(
                        int(self.hproc), self.remote, 0, MEM_RELEASE,
                    )
                if self.hproc:
                    win32api.CloseHandle(self.hproc)

        return _RemoteBuf(self, hwnd, size, write)

    def _read_process_memory(self, hwnd: int, address: int, size: int) -> bytes:
        """Прочитать size байт из памяти процесса окна."""
        hproc = self._open_process_for_hwnd(hwnd)
        try:
            buf = ctypes.create_string_buffer(size)
            br = ctypes.c_size_t(0)
            ok = ctypes.windll.kernel32.ReadProcessMemory(
                int(hproc), address, buf, size, ctypes.byref(br),
            )
            return bytes(buf[:br.value]) if ok else b""
        finally:
            win32api.CloseHandle(hproc)

    def _read_status_part(self, part_index: int) -> str:
        """
        Прочитать часть статус-бара через SB_GETTEXTW (cross-process).

        SB_GETTEXTW: wParam = part index, lParam = WCHAR* buffer (без префикса размера).
        Возвращает: LOWORD = длина текста в символах (без нуля), HIWORD = тип.
        """
        hwnd = self.cm.status_bar
        if not hwnd:
            return ""
        BUF_SIZE = 512  # 256 WCHAR
        try:
            with self._with_remote_buffer(hwnd, BUF_SIZE, write=False) as (hproc, remote):
                ret = win32gui.SendMessage(hwnd, SB_GETTEXTW, part_index, remote)
                if not ret:
                    return ""
                length = ret & 0xFFFF   # LOWORD = кол-во символов без нуля
                if length <= 0 or length > 255:
                    return ""
                # SB_GETTEXTW пишет текст начиная с offset 0 буфера (не 2!)
                raw = self._read_process_memory(hwnd, remote, length * 2 + 2)
                if not raw:
                    return ""
                return raw[:length * 2].decode("utf-16-le", errors="replace").strip()
        except Exception as e:
            log.debug(f"_read_status_part({part_index}): {e}")
            return ""

    def _read_curve_item_name(self, hwnd: int, index: int) -> Optional[str]:
        """
        Прочитать имя кривой из owner-draw ListBox через LB_GETITEMDATA.
        LB_GETTEXT не работает — имена лежат в структуре по item-data-указателю.
        """
        data_ptr = win32gui.SendMessage(hwnd, LB_GETITEMDATA, index, 0)
        if not data_ptr or data_ptr == 0xFFFFFFFF:
            return None
        # В 32-bit процессе валидные heap-указатели > 0x10000.
        # Значения меньше — целочисленные item-data, не указатели.
        if data_ptr < 0x10000:
            log.debug(f"_read_curve_item_name[{index}]: item data {data_ptr:#x} — integer, not pointer")
            return None
        raw = self._read_process_memory(hwnd, data_ptr, 256)
        if not raw:
            return None
        strings = [
            s.decode("ascii", errors="replace")
            for s in re.findall(rb"[\x20-\x7e]{3,}", raw)  # минимум 3 символа
        ]
        units = {"Ohmm", "MBK", "M", "GAPI", "API", "MV", "MM", "in", "ft", "ohm"}
        names = [s.strip() for s in strings if s.strip() not in units and len(s.strip()) > 1]
        if not names:
            return strings[-1].strip() if strings else None
        return max(names, key=len)

    def _clipboard_set(self, text: str):
        """
        Скопировать текст в буфер обмена.
        Первичный способ: win32clipboard (pywin32).
        Fallback: pyperclip (если установлен).
        Используется для вставки путей в file-диалоги, минуя ограничения typewrite.
        """
        try:
            import win32clipboard
            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
            finally:
                win32clipboard.CloseClipboard()
            return
        except Exception:
            pass
        try:
            import pyperclip
            pyperclip.copy(text)
        except Exception as e:
            log.warning(f"_clipboard_set: оба метода не сработали: {e}")

    # ─────────────────────────────────────────────────────────────
    # СОСТОЯНИЕ ОКНА
    # ─────────────────────────────────────────────────────────────

    def ensure_visible(self) -> bool:
        """Восстановить и вывести окно NeuraLOG на передний план."""
        hwnd = self.cm.main_window
        if not hwnd:
            return False
        try:
            # Если свёрнуто — развернуть
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                time.sleep(0.3)
            win32gui.SetForegroundWindow(hwnd)
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            time.sleep(0.2)
            return True
        except Exception as e:
            log.warning(f"ensure_visible: {e}")
            return False

    def get_title(self) -> str:
        """Текущий заголовок окна (содержит путь к проекту)."""
        return win32gui.GetWindowText(self.cm.main_window)

    def get_current_project(self) -> Optional[str]:
        """Извлечь путь к текущему проекту из заголовка окна."""
        title = self.get_title()
        # Формат: "NeuraLog  F:\nds\projects\Stanulska_002"
        match = re.search(r"NeuraLog\s+(.+)$", title)
        if match:
            return match.group(1).strip()
        return None

    def get_status(self) -> str:
        """Текст статус-бара (Ready / Tracing / координаты)."""
        if not self.cm.status_bar:
            return ""
        parts = win32gui.SendMessage(self.cm.status_bar, SB_GETPARTS, 0, 0)
        texts = []
        for i in range(max(parts, 1)):
            text = self._read_status_part(i)
            if text:
                texts.append(text)
        return " | ".join(texts)

    def get_user_prompt(self) -> str:
        """Текст из панели User Prompt (подсказки программы)."""
        if self.cm.user_prompt:
            return win32gui.GetWindowText(self.cm.user_prompt)
        return ""
    

    def probe_wizard_refined(self, start_id=1120, end_id=1170):
        import win32gui, win32con, time
        print(f"Запуск точечного медленного сканирования в диапазоне {start_id} - {end_id}...")
        
        main_hwnd = self.cm.main_window
        initial_windows = self._get_all_process_windows()
        
        for cmd_id in range(start_id, end_id + 1):
            # Печатаем в одну строку, чтобы не засорять терминал
            print(f"Проверяем ID: {cmd_id}...", end="\r")
            
            # Отправляем команду
            win32gui.SendMessage(main_hwnd, win32con.WM_COMMAND, cmd_id, 0)
            
            # Даем МНОГО времени (0.5 сек), чтобы исключить лаг отрисовки
            time.sleep(0.5)  
            
            current_windows = self._get_all_process_windows()
            new_windows = [w for w in current_windows if w not in initial_windows]
            
            if new_windows:
                new_hwnd = new_windows[0]
                title = win32gui.GetWindowText(new_hwnd)
                cls = win32gui.GetClassName(new_hwnd)
                
                print(f"\n\n🎯 ТОЧНО НАЙДЕНО! Настоящий CMD ID: {cmd_id}")
                print(f"  HWND: 0x{new_hwnd:08X}")
                print(f"  Заголовок: '{title}'")
                print(f"  Класс окна: '{cls}'")
                
                # Теперь мы НЕ закрываем окно автоматически! 
                # Пусть оно останется на экране, чтобы мы убедились глазами.
                print("\n💡 Окно оставлено открытым на экране компьютера!")
                return cmd_id
                
        print(f"\nВ диапазоне {start_id}-{end_id} ничего не появилось.")
        return None


    def _get_all_process_windows(self):
        """Вспомогательная функция: получает все видимые окна нашего процесса."""
        import win32gui, win32process
        hwnds = []
        
        # Извлекаем PID процесса из нашего главного окна NeuraLOG
        if not self.cm.main_window:
            return hwnds
        _, target_pid = win32process.GetWindowThreadProcessId(self.cm.main_window)
        
        def callback(hwnd, _):
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid == target_pid and win32gui.IsWindowVisible(hwnd):
                hwnds.append(hwnd)
                
        win32gui.EnumWindows(callback, None)
        return hwnds
    # ─────────────────────────────────────────────────────────────
    # ТРЕЙСЕР — ВЕКТОРИЗАЦИЯ
    # ─────────────────────────────────────────────────────────────

    def get_current_depth(self) -> Optional[float]:
        """Прочитать текущую глубину (из Static контрола рядом с 'Current Depth:')."""
        if not self.cm.val_current_depth:
            return None
        try:
            text = win32gui.GetWindowText(self.cm.val_current_depth)
            return float(text.strip())
        except (ValueError, Exception):
            return None

    def start_trace(self) -> bool:
        """Нажать BTN_FORWARD — начать трейсинг вперёд."""
        return self._click_button(self.cm.btn_forward, "BTN_FORWARD")

    def stop_trace(self) -> bool:
        """Нажать BTN_STOP — остановить трейсинг."""
        return self._click_button(self.cm.btn_stop, "BTN_STOP")

    def reverse_trace(self) -> bool:
        """Нажать BTN_REVERSE — трейсинг назад."""
        return self._click_button(self.cm.btn_reverse, "BTN_REVERSE")

    def _click_button(self, hwnd: int, name: str) -> bool:
        """Отправить BM_CLICK напрямую кнопке по HWND."""
        if not hwnd:
            log.warning(f"_click_button: HWND для '{name}' не найден")
            return False
        try:
            win32gui.SendMessage(hwnd, BM_CLICK, 0, 0)
            time.sleep(0.15)
            log.info(f"Clicked: {name} (0x{hwnd:08X})")
            return True
        except Exception as e:
            log.error(f"_click_button({name}): {e}")
            return False

    def set_tracer_speed(self, value: int) -> bool:
        """
        Установить скорость трейсера (Slider).
        value: 0 (медленно) .. 100 (быстро) — нормализованное значение.
        Реальный диапазон уточнить через TBM_GETRANGEMIN/MAX.
        """
        hwnd = self.cm.slider_tracer_speed
        if not hwnd:
            return False
        try:
            win32gui.SendMessage(hwnd, TBM_SETPOS, True, value)
            log.info(f"Tracer speed set to {value}")
            return True
        except Exception as e:
            log.error(f"set_tracer_speed: {e}")
            return False

    def get_tracer_speed(self) -> Optional[int]:
        """Прочитать текущее положение слайдера скорости."""
        hwnd = self.cm.slider_tracer_speed
        if not hwnd:
            return None
        return win32gui.SendMessage(hwnd, TBM_GETPOS, 0, 0)

    def get_depth_slider_position(self) -> Optional[int]:
        """Прочитать позицию ползунка глубины (первый trackbar — позиция по глубине)."""
        hwnd = self.cm.slider_depth_position
        if not hwnd:
            return None
        return win32gui.SendMessage(hwnd, TBM_GETPOS, 0, 0)

    def trace_to_depth(self, target_depth: float, timeout: float = 300.0,
                       check_interval: float = 1.0) -> bool:
        """
        Запустить трейсер и ждать пока глубина достигнет target_depth.
        Останавливает трейсер по достижении или таймауту.
        """
        current = self.get_current_depth()
        if current is None:
            log.error("trace_to_depth: не удалось прочитать текущую глубину")
            return False

        direction = "forward" if target_depth > current else "reverse"
        log.info(f"Трейсинг {direction}: {current} → {target_depth}")

        if direction == "forward":
            self.start_trace()
        else:
            self.reverse_trace()

        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(check_interval)
            depth = self.get_current_depth()
            if depth is None:
                continue
            log.debug(f"  Depth: {depth}")

            reached = (direction == "forward" and depth >= target_depth) or \
                      (direction == "reverse" and depth <= target_depth)
            if reached:
                self.stop_trace()
                log.info(f"Достигнута глубина {depth}")
                return True

        self.stop_trace()
        log.warning(f"trace_to_depth: таймаут {timeout}с на глубине {self.get_current_depth()}")
        return False

    def wait_for_trace_complete(self, stable_seconds: float = 3.0,
                                timeout: float = 600.0,
                                check_interval: float = 1.0) -> bool:
        """
        Ждать окончания трейсинга по стабилизации глубины.
        Считается завершённым когда глубина не меняется stable_seconds подряд.
        """
        prev_depth = self.get_current_depth()
        stable_since = time.time()
        start = time.time()

        while time.time() - start < timeout:
            time.sleep(check_interval)
            depth = self.get_current_depth()
            status = self.get_status()
            log.debug(f"  depth={depth}  status='{status}'")

            if depth != prev_depth:
                prev_depth = depth
                stable_since = time.time()
            elif time.time() - stable_since >= stable_seconds:
                log.info(f"Трейсинг завершён на глубине {depth}")
                return True

        log.warning("wait_for_trace_complete: таймаут")
        return False

    # ─────────────────────────────────────────────────────────────
    # КРИВЫЕ И ТРЕКИ
    # ─────────────────────────────────────────────────────────────

    def get_curves_list(self) -> List[str]:
        """Получить список кривых из owner-draw ListBox (LB_GETITEMDATA)."""
        hwnd = self.cm.listbox_curves
        if not hwnd:
            return []
        try:
            count = win32gui.SendMessage(hwnd, LB_GETCOUNT, 0, 0)
            if count <= 0:
                return []
            curves = []
            for i in range(count):
                name = self._read_curve_item_name(hwnd, i)
                if name:
                    curves.append(name)
            return curves
        except Exception as e:
            log.error(f"get_curves_list: {e}")
            return []

    def select_curve(self, index: int) -> bool:
        """Выбрать кривую в списке по индексу."""
        hwnd = self.cm.listbox_curves
        if not hwnd:
            return False
        win32gui.SendMessage(hwnd, LB_SETCURSEL, index, 0)
        time.sleep(0.1)
        return True

    def get_selected_curve_index(self) -> int:
        """Индекс выбранной кривой (-1 если ничего не выбрано)."""
        hwnd = self.cm.listbox_curves
        if not hwnd:
            return -1
        return win32gui.SendMessage(hwnd, LB_GETCURSEL, 0, 0)

    # ─────────────────────────────────────────────────────────────
    # МЕНЮ (через win32gui напрямую)
    # ─────────────────────────────────────────────────────────────

    def get_menu_structure(self) -> Dict:
        """Прочитать структуру главного меню через GetMenuString (надёжнее для MFC)."""
        hwnd  = self.cm.main_window
        hmenu = win32gui.GetMenu(hwnd)
        if not hmenu:
            log.warning("get_menu_structure: GetMenu вернул 0")
            return {}

        def read_menu(hm):
            result = {}
            count = win32gui.GetMenuItemCount(hm)
            for i in range(count):
                try:
                    text = win32gui.GetMenuString(hm, i, win32con.MF_BYPOSITION)
                    text = text.replace("&", "").split("\t")[0].strip()
                except Exception:
                    text = f"[item_{i}]"
                if not text:
                    continue  # разделитель
                submenu = win32gui.GetSubMenu(hm, i)
                if submenu and submenu != 0:
                    result[text] = read_menu(submenu)
                else:
                    item_id = win32gui.GetMenuItemID(hm, i)
                    result[text] = item_id
            return result

        return read_menu(hmenu)

    def send_menu_command(self, command_id: int,
                           target: str = "main") -> bool:
        """
        Выполнить команду меню по ID через WM_COMMAND.
        target: "main"  — главное окно (маршрутизация MFC)
                "mdi"   — активный MDI-child (для команд вида/документа)
        Использует SendMessage (синхронно), не PostMessage.
        """
        hwnd = self.cm.main_window
        if target == "mdi":
            mdi_child = self._get_active_mdi_child()
            if mdi_child:
                hwnd = mdi_child
        try:
            # SendMessage синхронен — команда обработана до возврата
            win32api.SendMessage(hwnd, WM_COMMAND, command_id, 0)
            time.sleep(0.3)
            return True
        except Exception as e:
            log.error(f"send_menu_command({command_id}): {e}")
            return False

    def _get_active_mdi_child(self) -> int:
        """
        Получить HWND активного MDI-child окна.
        В MDI-приложении (NeuraLOG) каждый открытый проект — отдельный child.
        """
        mdi = self.cm.mdi_client
        if not mdi:
            return 0
        # WM_MDIGETACTIVE = 0x0229; возвращает HWND активного child
        return win32gui.SendMessage(mdi, 0x0229, 0, 0)

    def get_open_projects(self) -> list:
        """
        Получить список всех открытых проектов (MDI-children).
        Возвращает: [{"hwnd": int, "title": str}, ...]
        """
        mdi = self.cm.mdi_client
        if not mdi:
            return []
        children = []
        def cb(hwnd, _):
            title = win32gui.GetWindowText(hwnd)
            if title:
                children.append({"hwnd": hwnd, "title": title})
            return True
        try:
            win32gui.EnumChildWindows(mdi, cb, None)
        except Exception:
            pass
        return children

    def switch_to_project(self, path: str) -> bool:
        """
        Переключиться на уже открытый проект в MDI по части пути/имени.
        Возвращает True если найден и переключён.
        """
        name = os.path.splitext(os.path.basename(path))[0]
        children = self.get_open_projects()
        for ch in children:
            if name.lower() in ch["title"].lower():
                # WM_MDIACTIVATE = 0x0222
                win32gui.SendMessage(self.cm.mdi_client, 0x0222,
                                     ch["hwnd"], 0)
                time.sleep(0.4)
                log.info(f"switch_to_project: переключились на '{ch['title']}'")
                return True
        return False

    # Кандидаты ID экспорта (из дампа меню File, owner-draw — текст недоступен)
    EXPORT_CMD_CANDIDATES = (3006, 32845, 32786, 32836, 32837)

    def export_result(self, command_id: int = 3006) -> bool:
        """
        Экспорт результата (LAS). command_id уточняется через dump_menu_ids()
        или parse_menu_resources.py + ResourceHacker.
        """
        log.info(f"export_result: отправка WM_COMMAND {command_id}")
        return self.send_menu_command(command_id)

    def _find_save_dialog(self) -> int:
        """
        Найти открытый диалог Save As / Open — стандартный #32770 без родителя
        или с родителем = главное окно NeuraLOG.
        Возвращает HWND диалога или 0.
        """
        candidates = []

        def cb(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return True
            cls    = win32gui.GetClassName(hwnd)
            parent = win32gui.GetParent(hwnd)
            if cls == "#32770" and (parent == 0 or parent == self.cm.main_window):
                candidates.append(hwnd)
            return True

        win32gui.EnumWindows(cb, None)
        return candidates[0] if candidates else 0

    def export_las(self, output_path: str, cmd_id: int = 3006,
                   dialog_timeout: float = 8.0) -> bool:
        """
        Экспортировать текущий проект как LAS через диалог "Digital Curve Output".

        Алгоритм:
          1. SendMessage WM_COMMAND (основной) + MDI-child fallback + hotkey fallback
          2. Найти диалог по заголовку "Digital Curve Output"
          3. WM_SETTEXT в Edit[0] (поле File) → снять "View After Generation"
          4. Нажать OK (id=1)
          5. Обработать overwrite-confirm
          6. Дождаться файла; если не по нашему пути — найти в /las/ и скопировать
        """
        import pyautogui, ctypes, shutil
        WM_SETTEXT = 0x000C
        BM_CLICK   = 0x00F5

        log.info(f"export_las → {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.ensure_visible()
        time.sleep(0.8)  # окно должно получить фокус

        # ── Шаг 1: открыть диалог ────────────────────────────────
        # Попытка 1: SendMessage к main_window (синхронно)
        win32api.SendMessage(self.cm.main_window, WM_COMMAND, cmd_id, 0)
        dlg = self._find_export_dialog(timeout=4.0)

        # Попытка 2: SendMessage к активному MDI-child
        if not dlg:
            mdi_child = self._get_active_mdi_child()
            if mdi_child:
                log.info("  Fallback: SendMessage к MDI-child...")
                win32api.SendMessage(mdi_child, WM_COMMAND, cmd_id, 0)
                dlg = self._find_export_dialog(timeout=4.0)

        # Попытка 3: pyautogui Ctrl+Shift+F
        if not dlg:
            log.info("  Fallback: pyautogui Ctrl+Shift+F...")
            pyautogui.hotkey("ctrl", "shift", "f")
            dlg = self._find_export_dialog(timeout=dialog_timeout)

        if not dlg:
            log.error("export_las: диалог 'Digital Curve Output' не найден ни одним методом")
            return False

        log.info(f"export_las: диалог 0x{dlg:08X} '{win32gui.GetWindowText(dlg)}'")
        win32gui.SetForegroundWindow(dlg)
        time.sleep(0.3)

        # ── Шаг 2: контролы диалога ──────────────────────────────
        ctrls   = self._get_dialog_controls(dlg)
        edits   = ctrls.get("Edit", [])
        buttons = ctrls.get("Button", [])

        # ── Шаг 3: установить путь в Edit id=3369 (поле File) ────
        # File Edit — первый Edit с id=3369 по данным probe
        file_edit = next(
            (h for h in edits if win32gui.GetDlgCtrlID(h) == 3369),
            edits[0] if edits else None
        )
        default_path = ""
        if file_edit:
            default_path = win32gui.GetWindowText(file_edit)
            log.info(f"  Default path: '{default_path}'")
            ctypes.windll.user32.SendMessageW(file_edit, WM_SETTEXT, 0, output_path)
            time.sleep(0.15)
            log.info(f"  Set path:     '{win32gui.GetWindowText(file_edit)[:80]}'")
        else:
            log.warning("  Поле File не найдено — экспорт будет в дефолтный путь")

        # ── Шаг 3b: снять "View After Generation" (id=3387) ─────
        view_btn = next(
            (h for h in buttons if win32gui.GetDlgCtrlID(h) == 3387),
            None
        )
        if view_btn and win32gui.SendMessage(view_btn, 0x00F0, 0, 0):  # BM_GETCHECK
            win32gui.SendMessage(view_btn, BM_CLICK, 0, 0)
            log.info("  Unchecked 'View After Generation'")

        # ── Шаг 4: нажать OK (id=1) ──────────────────────────────
        ok_btn = next(
            (h for h in buttons if win32gui.GetDlgCtrlID(h) == 1),
            None
        )
        if ok_btn:
            log.info(f"  Нажимаем OK (0x{ok_btn:08X})")
            win32gui.SendMessage(ok_btn, BM_CLICK, 0, 0)
        else:
            log.warning("  OK не найден, нажимаем Enter")
            pyautogui.press("enter")
        time.sleep(2.5)

        # ── Шаг 5: overwrite confirm ─────────────────────────────
        for _ in range(2):
            confirm = None
            def _cb(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    t = win32gui.GetWindowText(hwnd)
                    if any(w in t for w in ("Confirm","Overwrite","Replace","существует")):
                        confirm_box.append(hwnd)
                return True
            confirm_box = []
            win32gui.EnumWindows(_cb, None)
            confirm = confirm_box[0] if confirm_box else self._find_save_dialog()
            if not confirm:
                break
            log.info(f"  Confirm 0x{confirm:08X} — нажимаем Yes")
            win32gui.SetForegroundWindow(confirm)
            cc = self._get_dialog_controls(confirm).get("Button", [])
            yes = next((h for h in cc if
                        win32gui.GetWindowText(h) in ("Yes","Да","&Yes") or
                        win32gui.GetDlgCtrlID(h) == 6), None)
            if yes:
                win32gui.SendMessage(yes, BM_CLICK, 0, 0)
            else:
                pyautogui.press("enter")
            time.sleep(1.0)

        # ── Шаг 6: дождаться файла ───────────────────────────────
        deadline = time.time() + 15.0
        while time.time() < deadline:
            if os.path.isfile(output_path):
                size = os.path.getsize(output_path) / 1024
                log.info(f"export_las: ✓ {output_path}  ({size:.1f} KB)")
                return True
            time.sleep(0.5)

        # ── Шаг 7: fallback — NeuraLOG сохранил в /las/ проекта ─
        log.warning(f"  Файл не найден по нашему пути. Ищем в папке проекта...")
        found = self._find_recent_las(
            hint_paths=[default_path, self.get_current_project()],
            max_age_sec=30.0
        )
        if found:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(found, output_path)
            size = os.path.getsize(output_path) / 1024
            log.info(f"export_las: ✓ скопирован из {found}  ({size:.1f} KB)")
            return True

        log.error(f"export_las: ✗ LAS не создан")
        return False

    def _find_recent_las(self, hint_paths: list = None,
                          max_age_sec: float = 30.0) -> str:
        """Найти свежесозданный LAS (возраст ≤ max_age_sec) в hint_paths и папках проекта."""
        from pathlib import Path as _P
        dirs = set()
        for hint in (hint_paths or []):
            if not hint: continue
            p = _P(hint)
            if p.suffix.lower() == ".las":
                dirs.add(str(p.parent))
            elif p.is_dir():
                dirs.add(str(p))
                dirs.add(str(p / "las"))
            else:
                dirs.add(str(p.parent))
                dirs.add(str(p.parent / "las"))
        proj = self.get_current_project()
        if proj:
            pp = _P(proj)
            if pp.suffix.lower() in (".nlgx", ".las"):
                pp = pp.parent
            for d in (pp, pp / "las", pp.parent / "las"):
                dirs.add(str(d))
        now = time.time()
        best, best_t = "", 0.0
        for d in dirs:
            dp = _P(d)
            if not dp.exists(): continue
            for las in dp.glob("*.las"):
                try:
                    mt = las.stat().st_mtime
                    if (now - mt) <= max_age_sec and mt > best_t:
                        best, best_t = str(las), mt
                except Exception:
                    pass
        if best:
            log.info(f"_find_recent_las: ✓ {best}  (age={(now-best_t):.1f}s)")
        return best

    def probe_export_command(self, wait_seconds: float = 2.0) -> Optional[int]:
        """
        Автоматически определить command_id для экспорта LAS.

        Перебирает EXPORT_CMD_CANDIDATES по одному, после каждого проверяет:
          а) появился ли Save As диалог — значит команда правильная
          б) изменился ли статус — косвенный признак
        При появлении диалога — закрывает его и возвращает найденный ID.

        Запускать один раз для нового экземпляра NeuraLOG, потом зафиксировать ID в коде.
        """
        log.info("probe_export_command: перебираем кандидатов...")
        for cmd_id in self.EXPORT_CMD_CANDIDATES:
            status_before = self.get_status()
            log.info(f"  → CMD {cmd_id}...")
            self.send_menu_command(cmd_id)
            time.sleep(wait_seconds)

            dialog_hwnd = self._find_save_dialog()
            if dialog_hwnd:
                log.info(f"probe_export_command: ✓ диалог появился на CMD={cmd_id}")
                # Закрыть диалог Escape
                import pyautogui
                win32gui.SetForegroundWindow(dialog_hwnd)
                pyautogui.press("escape")
                time.sleep(0.5)
                return cmd_id

            status_after = self.get_status()
            if status_before != status_after:
                log.info(f"probe_export_command: статус изменился на CMD={cmd_id}: '{status_after}'")
                return cmd_id

        log.warning("probe_export_command: ни один кандидат не открыл диалог — "
                    "используй parse_menu_resources.py или ResourceHacker")
        return None

    def dump_menu_ids(self) -> List[Dict]:
        """Дамп всех пунктов меню с ID (без текста — owner-draw)."""
        hwnd = self.cm.main_window
        hmenu = win32gui.GetMenu(hwnd)
        if not hmenu:
            return []

        items = []

        def walk(hm, path: str = ""):
            count = win32gui.GetMenuItemCount(hm)
            for i in range(count):
                item_id = win32gui.GetMenuItemID(hm, i)
                submenu = win32gui.GetSubMenu(hm, i)
                loc = f"{path}/{i}" if path else str(i)
                if submenu and submenu != 0:
                    walk(submenu, loc)
                elif item_id not in (-1, 0):
                    items.append({"path": loc, "command_id": item_id})

        walk(hmenu)
        return items

    # ─────────────────────────────────────────────────────────────
    # ТУЛБАР КНОПКИ (enumerate ToolbarWindow32)
    # ─────────────────────────────────────────────────────────────

    def get_toolbar_buttons(self, toolbar_hwnd: int) -> List[Dict]:
        """
        Получить список кнопок тулбара через TB_BUTTONCOUNT / TB_GETBUTTON.
        Возвращает: [{index, command_id, state, style, text}, ...]
        """
        if not toolbar_hwnd:
            return []

        count = win32gui.SendMessage(toolbar_hwnd, TB_BUTTONCOUNT, 0, 0)
        if count <= 0:
            return []

        # Нужно читать память процесса — для 32-bit процесса из 64-bit Python
        # используем ReadProcessMemory
        buttons = []
        try:
            _, pid = win32process.GetWindowThreadProcessId(toolbar_hwnd)
            hprocess = win32api.OpenProcess(
                PROCESS_VM_OPERATION | PROCESS_VM_READ | PROCESS_VM_WRITE,
                False, pid
            )

            TBBUTTON_SIZE = 20
            VirtualAllocEx = ctypes.windll.kernel32.VirtualAllocEx
            VirtualFreeEx  = ctypes.windll.kernel32.VirtualFreeEx
            ReadProcessMemory = ctypes.windll.kernel32.ReadProcessMemory

            buf_addr = VirtualAllocEx(
                int(hprocess), None, TBBUTTON_SIZE,
                MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE,
            )
            if not buf_addr:
                log.error("get_toolbar_buttons: VirtualAllocEx failed")
                return []

            for i in range(count):
                win32gui.SendMessage(toolbar_hwnd, TB_GETBUTTON, i, buf_addr)

                local_buf = ctypes.create_string_buffer(TBBUTTON_SIZE)
                bytes_read = ctypes.c_size_t(0)
                ReadProcessMemory(int(hprocess), buf_addr, local_buf, TBBUTTON_SIZE,
                                  ctypes.byref(bytes_read))

                # Парсим структуру
                import struct
                iBitmap, idCommand, fsState, fsStyle = struct.unpack_from("<iibb", local_buf)
                buttons.append({
                    "index":      i,
                    "command_id": idCommand,
                    "state":      fsState,
                    "style":      fsStyle,
                    "enabled":    bool(fsState & 0x04),
                    "checked":    bool(fsState & 0x01),
                    "separator":  bool(fsStyle & 0x08),
                })

            VirtualFreeEx(int(hprocess), buf_addr, 0, MEM_RELEASE)  # MEM_RELEASE
            win32api.CloseHandle(hprocess)

        except Exception as e:
            log.error(f"get_toolbar_buttons: {e}")

        return buttons

    def click_toolbar_button(self, toolbar_hwnd: int, command_id: int) -> bool:
        """Нажать кнопку тулбара по command_id через WM_COMMAND."""
        if not toolbar_hwnd:
            return False
        try:
            # Отправляем в главное окно
            win32api.PostMessage(self.cm.main_window, WM_COMMAND, command_id, 0)
            time.sleep(0.3)
            return True
        except Exception as e:
            log.error(f"click_toolbar_button: {e}")
            return False

    # ─────────────────────────────────────────────────────────────
    # ФАЙЛОВЫЕ ОПЕРАЦИИ (через горячие клавиши)
    # ─────────────────────────────────────────────────────────────

    def _hotkey(self, *keys):
        """Отправить горячую клавишу в главное окно."""
        import pyautogui
        self.ensure_visible()
        time.sleep(0.1)
        pyautogui.hotkey(*keys)
        time.sleep(0.5)

    def save(self) -> bool:
        """Сохранить проект (Ctrl+S)."""
        try:
            self._hotkey("ctrl", "s")
            log.info("Сохранено (Ctrl+S)")
            return True
        except Exception as e:
            log.error(f"save: {e}")
            return False

    def open_project(self, path: str) -> bool:
        """
        Открыть .nlgx проект в NeuraLOG.

        Алгоритм:
          1. Проверить — не открыт ли проект уже в MDI (переключить фокус)
          2. Если нет — Ctrl+O → вставить путь → Enter
          3. Дождаться загрузки (заголовок MDI-child обновится)
        """
        import pyautogui
        name = os.path.splitext(os.path.basename(path))[0]

        # Шаг 1: проект уже открыт в MDI?
        if self.switch_to_project(path):
            log.info(f"open_project: проект уже открыт, переключились")
            return True

        # Шаг 2: открыть через диалог
        try:
            self.ensure_visible()
            time.sleep(0.5)          # дать окну время получить фокус
            pyautogui.hotkey("ctrl", "o")
            time.sleep(1.5)          # ждать диалог открытия

            self._clipboard_set(path)
            pyautogui.hotkey("ctrl", "a")
            time.sleep(0.1)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.2)
            pyautogui.press("enter")
            time.sleep(3.0)          # NeuraLOG загружает изображение

            # Шаг 3: подождать пока MDI-child появится с нужным именем
            deadline = time.time() + 15.0
            while time.time() < deadline:
                if self.switch_to_project(path):
                    log.info(f"open_project: ✓ проект загружен '{name}'")
                    return True
                time.sleep(0.5)

            # Проект не появился в MDI — но мог открыться всё равно
            # Проверяем заголовок главного окна
            title = win32gui.GetWindowText(self.cm.main_window)
            if name[:12].lower() in title.lower():
                log.info(f"open_project: ✓ заголовок обновился: {title}")
                return True

            log.warning(f"open_project: проект не появился в MDI за 15с, продолжаем")
            return True   # не падаем — возможно проект открылся нестандартно

        except Exception as e:
            log.error(f"open_project: {e}")
            return False

    def new_project(self, image_path: str, project_path: str) -> bool:
        """
        Создать новый проект: File > New, указать изображение и сохранить.
        Требует уточнения под реальный UI диалога New Project.
        """
        import pyautogui
        try:
            self.ensure_visible()
            pyautogui.hotkey("ctrl", "n")
            time.sleep(1.5)
            # Дальнейшие шаги зависят от конкретного диалога New Project
            # → записать скриншот и уточнить
            log.warning("new_project: требует настройки под диалог NeuraLOG")
            return False
        except Exception as e:
            log.error(f"new_project: {e}")
            return False

    # ─────────────────────────────────────────────────────────────
    # ДИАГНОСТИКА И ОТЛАДКА
    # ─────────────────────────────────────────────────────────────

    def print_state(self):
        """Вывести текущее состояние в лог."""
        print("=" * 50)
        print(f"Проект:          {self.get_current_project()}")
        print(f"Статус:          {self.get_status()}")
        print(f"Текущая глубина: {self.get_current_depth()}")
        print(f"Ползунок глубины:{self.get_depth_slider_position()}")
        print(f"Скорость:        {self.get_tracer_speed()}")
        print(f"User prompt:     {self.get_user_prompt()}")
        curves = self.get_curves_list()
        print(f"Кривых:          {len(curves)}: {curves}")
        print("=" * 50)

    def dump_toolbar_info(self, toolbar_name: str, toolbar_hwnd: int):
        """Вывести кнопки тулбара — для первичного исследования."""
        if not toolbar_hwnd:
            print(f"[{toolbar_name}] HWND не найден")
            return
        buttons = self.get_toolbar_buttons(toolbar_hwnd)
        print(f"\n[{toolbar_name}] HWND=0x{toolbar_hwnd:08X}, кнопок={len(buttons)}:")
        for b in buttons:
            marker = "SEP" if b["separator"] else f"CMD={b['command_id']:5d}"
            enabled = "ON " if b["enabled"] else "off"
            print(f"  [{b['index']:2d}] {marker}  {enabled}  checked={b['checked']}")

    def explore_all_toolbars(self):
        """Исследовать все тулбары — запустить один раз для маппинга команд."""
        cm = self.cm
        toolbars = {
            "Main":           cm.toolbar_main,
            "Calibrate&Trace": cm.toolbar_calibrate,
            "Curve Edit":     cm.toolbar_curve_edit,
            "Curve/Track":    cm.toolbar_curve_tools,
            "Zoom":           cm.toolbar_zoom,
            "Color Selection": cm.toolbar_color,
            "Quality Control": cm.toolbar_qc,
        }
        for name, hwnd in toolbars.items():
            self.dump_toolbar_info(name, hwnd)

    def explore_menu(self):
        """Распечатать структуру меню."""
        menu = self.get_menu_structure()
        def _print(d, indent=0):
            for k, v in d.items():
                if isinstance(v, dict):
                    print("  " * indent + f"[{k}]")
                    _print(v, indent + 1)
                else:
                    print("  " * indent + f"  {k}: {v}")
        print("\n=== Меню NeuraLOG ===")
        _print(menu)


# ─────────────────────────────────────────────────────────────────
# СКРИПТ ИССЛЕДОВАНИЯ (запустить один раз после подключения)
# ─────────────────────────────────────────────────────────────────

def explore(nl: NeuraLog):
    """Полная диагностика — запустить один раз для маппинга UI."""
    print("\n" + "=" * 60)
    print("  ИССЛЕДОВАНИЕ UI NeuraLOG")
    print("=" * 60)

    # Текущее состояние
    nl.print_state()

    # Меню
    nl.explore_menu()

    # Тулбары → command_id кнопок
    nl.explore_all_toolbars()

    print("\n✓ Исследование завершено")
    print("  Запишите command_id нужных кнопок для дальнейшей автоматизации")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    nl = NeuraLog()
    if not nl.cm.main_window:
        print("NeuraLOG не найден. Убедитесь что программа запущена.")
        exit(1)

    nl.ensure_visible()

    # Режим исследования — запустить первым делом
    explore(nl)

    # Пример: прочитать текущее состояние
    print(f"\nТекущая глубина: {nl.get_current_depth()}")
    print(f"Кривые: {nl.get_curves_list()}")
    print(f"Статус: {nl.get_status()}")

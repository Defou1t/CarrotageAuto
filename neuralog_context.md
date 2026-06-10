# Контекст: Автоматизация NeuraLOG v2015 — состояние на 07.06.2026

## Цель проекта

Автоматизировать векторизацию каротажных изображений (well logs) в программе **NeuraLOG v2015** (Windows 32-bit MFC desktop app) с помощью Python-агента на базе Claude API.

Рабочие директории:
- `F:\nds\images` — исходные изображения (.png/.tif)
- `F:\nds\projects` — проекты NeuraLOG (.nlgx)
- `F:\nds\output` — экспорт результатов (.las)
- `F:\nds\ScreensAutoGUI` — скриншоты агента
- `F:\nds\Auto\` — скрипты автоматизации

---

## Файлы проекта

| Файл | Назначение | Статус |
|------|-----------|--------|
| `neuralog_agent.py` | Главный агент (Claude tool_use + pyautogui + win32) | Рабочий, не тестировался вживую |
| `neuralog_win32 (1).py` | Низкоуровневый контроллер NeuraLOG через win32 API | **Активно разрабатывается** |
| `uia_probe.py` | Диагностика UIAutomation/MSAA | Выполнен, результат в отчёте |
| `uia_probe_report.txt` | Отчёт о структуре UI NeuraLOG | Ключевой источник истины |
| `apply_patches.py` | Скрипт автоматического применения патчей | Создан, применён |

Среда: Python 3.14.5 **64-bit** на Windows. NeuraLOG — **32-bit** MFC.
Путь к Python: `C:\Users\Defou1t\AppData\Local\Python\pythoncore-3.14-64\python.exe`

---

## Архитектурное решение

**UIA backend — непригоден.** Несмотря на то что подключается, дерево контролов недоступно:
```
✗ Ошибка дерева UIA: No windows for that process could be found
```

**win32 backend — работает.** 64 дочерних контрола видны через `EnumChildWindows`.
**Меню — owner-draw.** `GetMenuString` возвращает пустые строки для всех пунктов меню NeuraLOG. Текст меню отрисовывается программой, не хранится в стандартном поле.

Итоговый стек:
- `win32gui.EnumChildWindows` — маппинг HWND контролов
- `win32gui.SendMessage(BM_CLICK)` — клики по Button
- `win32api.PostMessage(WM_COMMAND, cmd_id)` — команды меню/тулбара
- `VirtualAllocEx + ReadProcessMemory` — чтение ListBox/Toolbar из 32-bit процесса
- `pyautogui` — файловые диалоги (Ctrl+O, Ctrl+S)

---

## Применённые патчи (apply_patches.py)

### Патч 1 — OpenProcess: неверные флаги (КРИТИЧЕСКИЙ)
```python
# БЫЛО (в get_toolbar_buttons):
hprocess = win32api.OpenProcess(0x0010 | 0x0020, False, pid)
# 0x0020 = VM_WRITE, а не VM_OPERATION! VirtualAllocEx требует 0x0008.

# СТАЛО:
PROCESS_VM_OPERATION = 0x0008
PROCESS_VM_READ      = 0x0010
PROCESS_VM_WRITE     = 0x0020
hprocess = win32api.OpenProcess(
    PROCESS_VM_OPERATION | PROCESS_VM_READ | PROCESS_VM_WRITE, False, pid
)
```

### Патч 2 — get_curves_list: cross-process чтение ListBox
Оригинальный код передавал 64-битный адрес Python-буфера в 32-битный процесс → мусор (`'\x01\x01\x01яяяDA1'`). Заменён на `VirtualAllocEx` + `ReadProcessMemory`, аналогично toolbar buttons.

### Патч 3 — get_menu_structure: GetMenuString вместо GetMenuItemInfo
`GetMenuItemInfo` в pywin32 тихо падал для всех пунктов MFC-меню. Заменён на `GetMenuString` с `MF_BYPOSITION`. Результат: ID читаются, текст по-прежнему пустой (owner-draw меню).

---

## Текущее состояние контролов (ControlMap) — 07.06.2026 19:06

```
main_window:          0x000E0A44  (меняется при перезапуске!)
mdi_client:           0x000E0386
status_bar:           0x000C0A54  (WM_GETTEXT → ""; нужен SB_GETTEXT 0x0401)
toolbar_main:         0x000D093E
toolbar_calibrate:    0x00020908
toolbar_curve_edit:   0x00020906
toolbar_curve_tools:  0x000B0164
toolbar_zoom:         0x00020910
toolbar_color:        0x000F02DE
toolbar_qc:           0x000208F0
btn_forward:          0x000208F4  (BTN_FORWARD — старт трейса вперёд)
btn_stop:             0x000208F2  (BTN_STOP)
btn_reverse:          0x000208E4  (BTN_REVERSE)
slider_tracer_speed:  0x000208F8  ⚠️ НЕВЕРНО — это ползунок глубины!
val_current_depth:    0x000208FA  (Static, GetWindowText → "3080" — работает)
lbl_current_depth:    0x00030954  (Static "Current Depth:")
combo_color_mode:     0x00100136
panel_curve_dialog:   0x000B00CA
listbox_curves:       0x00030996  (LB_GETCOUNT → 0, пусто в текущем проекте)
btn_hide_grid:        0x00030998
btn_remove_curve:     0x00020978
btn_units:            0x000308CC
edit_top:             0x00030904
edit_bottom:          0x0003090E
user_prompt:          0x000208E8  (GetWindowText → "User prompt" — заглушка)
richedit_log:         0x000208E6
```

**Примечание:** HWND'ы меняются при каждом перезапуске NeuraLOG. `_build_control_map()` вызывается при `connect()` и заполняет актуальные значения автоматически.

Открытый проект при последнем запуске:
`NeuraLog - [Semeguniv_020\wlg\Semeguniv_20_BK+MBK_3080_3520_200_D1.nlgx ( MBK )]`

---

## Известная ошибка: slider_tracer_speed

`EnumChildWindows` находит два `msctls_trackbar32`. `_build_control_map` берёт первый — но первый это ползунок позиции глубины (TBM_GETPOS возвращает 3080 = текущая глубина), а не скорость трейсера.

**Нужный fix в `_build_control_map`:**
```python
# Вместо текущего блока в callback visit:
elif cls == "msctls_trackbar32":
    if not self.cm.slider_tracer_speed:
        self.cm.slider_tracer_speed = hwnd  # ← берёт первый (неверно)

# Заменить на сбор обоих, после EnumChildWindows:
_trackbars = []
# ... в callback:
elif cls == "msctls_trackbar32":
    _trackbars.append(hwnd)
# ... после win32gui.EnumChildWindows():
if len(_trackbars) >= 2:
    self.cm.slider_tracer_speed = _trackbars[1]   # второй = реальная скорость
elif len(_trackbars) == 1:
    self.cm.slider_tracer_speed = _trackbars[0]
```

---

## Карта команд (Command IDs)

### Стандартные MFC (постоянные, не зависят от версии):
| ID | Назначение |
|----|-----------|
| 57601 | File > New |
| 57602 | File > Open |
| 57603 | File > Save |
| 57604 | File > Save As |
| 57607 | File > Print |
| 57616 | File > Print Preview |
| 57643 | Недавний файл 1 |
| 57644 | Недавний файл 2 |
| 57664 | Window > New |
| 57665 | Window > Arrange |

### Toolbar: Main (18 кнопок)
```
[0] CMD=57664  ON    [1] CMD= 3104  ON    [2] CMD=57603  OFF (Save)
[3] CMD=32835  ON    [4] CMD=32786  ON    [5] CMD=57607  ON
[6] CMD=    0  ---   [7] CMD=32848  ON    [8] CMD=    0  ---
[9] CMD=32849  ON   [10] CMD=32850  ON   [11] CMD=    0  ---
[12] CMD=57643 OFF  [13] CMD=57644  OFF  [14] CMD=32774  ON
[15] CMD= 3167 ON   [16] CMD=32794  ON   [17] CMD=32793  ON
```

### Toolbar: Calibrate & Trace (7 кнопок) — ключевые для автоматизации
```
[0] CMD= 3194  ON              — Calibrate/Setup
[1] CMD=32840  ON  (toggle)    — режим трейсинга (checked в запуске 1)
[2] CMD=32841  ON  (toggle)
[3] CMD=32842  ON  (toggle)
[4] CMD=32843  ON  (toggle)
[5] CMD=32844  ON  (toggle)    — (checked в запуске 2)
[6] CMD=32846  ON  (toggle)
```
Checked-состояние меняется между запусками → взаимоисключающие режимы трейсинга.

### Toolbar: Curve Edit (7 кнопок)
```
[0] CMD=32879  [1] CMD=32791  [2] CMD=32926
[3] CMD=32880  [4] CMD=32881  [5] CMD=32882  [6] CMD=32790
```

### Toolbar: Curve/Track Tools (12 кнопок)
```
[0] CMD= 3147  [1] CMD= 3145  [2] CMD= 3146  [3] CMD= 3144
[4] CMD= 3142  (checked)      [5] CMD= 3143
[6] SEP        [7] CMD=32909  [8] CMD=32863  [9] CMD= 3132
[10] CMD=32868 [11] CMD=32853
```

### Toolbar: Zoom (6 кнопок)
```
[0] CMD=  468  [1] SEP        [2] CMD=30778  [3] CMD=30779
[4] CMD=30780  [5] CMD=30775
```

### Toolbar: Color Selection (7 кнопок)
```
[0] SEP  [1] CMD=414  [2] SEP  [3] CMD=468
[4] SEP  [5] CMD=415  [6] CMD=30791
```

### Toolbar: Quality Control (8 кнопок) — все disabled
```
[0] CMD=3109  [1] CMD=3213  [2] CMD=3121  [3] CMD=3110
[4] CMD=3111  [5] CMD=3129  [6] CMD=3149  [7] CMD=3214
```

---

## Что работает / не работает

| Функция | Статус | Примечание |
|---------|--------|-----------|
| `connect()` | ✅ | Надёжно находит главное окно через MDIClient-детект |
| `get_current_depth()` | ✅ | Возвращает 3080.0 — верно |
| `start_trace()` | ✅ | BM_CLICK на btn_forward, работает без VirtualAllocEx |
| `stop_trace()` | ✅ | BM_CLICK на btn_stop |
| `reverse_trace()` | ✅ | BM_CLICK на btn_reverse |
| `save()` | ✅ | Ctrl+S через pyautogui |
| `get_toolbar_buttons()` | ✅ | После патча флагов — все тулбары читаются |
| `get_tracer_speed()` | ❌ | Возвращает глубину (3080) — неверный HWND |
| `get_curves_list()` | ❌ | Listbox пуст в текущем проекте |
| `get_status()` | ❌ | Статусбар пустой (нужен SB_GETTEXT 0x0401) |
| `get_menu_structure()` | ⚠️ | ID читаются, текст — нет (owner-draw меню) |
| `open_project()` | ⚠️ | Через Ctrl+O + pyautogui, не тестировалось |
| `export_result()` | ❌ | command_id для экспорта LAS не найден |

---

## Нераскрытые вопросы

1. **Command ID для экспорта LAS** — не найден. Вероятно среди кастомных ID: `3006`, `32845`, `32786`, `32836` или в подменю. Рекомендуется открыть `NeuraLOG.exe` в **ResourceHacker** (free) → раздел Menu → все пункты с ID будут подписаны.

2. **Как загрузить изображение в проект** — неизвестна последовательность диалогов. Нужна запись скриншотов ручного процесса создания нового проекта.

3. **Curves listbox** — почему пуст? Возможно Curve/Track Dialog показывает кривые только когда активна конкретная track-панель. Нужно открыть панель вручную и проверить.

4. **SB_GETTEXT для статусбара** — аналогичная cross-process проблема, нужен тот же паттерн VirtualAllocEx.

5. **Что делает команда 3194** (первая кнопка Calibrate&Trace) — нужно нажать и наблюдать.

---

## Следующие шаги (приоритет)

1. **Открыть NeuraLOG.exe в ResourceHacker** → найти ID команды экспорта LAS
2. **Применить fix для slider_tracer_speed** (см. секцию выше)
3. **Протестировать BTN_FORWARD/STOP** вживую:
   ```python
   nl.start_trace(); time.sleep(5); print(nl.get_current_depth()); nl.stop_trace()
   ```
4. **Разведать Calibrate&Trace команды** — поочерёдно вызывать и наблюдать
5. **Записать ручной процесс** создания нового проекта (скриншоты каждого шага) для `tool_open_project`

---

## Зависимости и установка

```bash
pip install anthropic pywinauto pywin32 pillow pyautogui opencv-python
```

Переменная окружения: `ANTHROPIC_API_KEY`
Модель агента: `claude-opus-4-5` (в neuralog_agent.py)

---

## Предупреждение pywinauto

При каждом запуске появляется:
```
UserWarning: 32-bit application should be automated using 32-bit Python (you use 64-bit Python)
```
Это **не ошибка**, pywinauto работает. VirtualAllocEx/ReadProcessMemory для cross-process чтения работают корректно после патча флагов.

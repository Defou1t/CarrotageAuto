# NeuraLOG Automation — Полный план проекта
## Документ для продолжения разработки

---

## 1. Суть задачи

**Цель:** Автоматически оцифровать каротажные кривые из растровых изображений (.tif/.jpg/.png) в формат LAS 2.0 с помощью программы NeuraLOG v2015.

**Что такое оцифровка (векторизация) каротажных кривых:**
- Есть сканированное изображение бумажного каротажного графика
- На изображении нарисованы кривые (линии) — каждая это физическая характеристика породы
- Нужно преобразовать пиксели линии в числа: глубина (метры) → значение (Ohmm, mV, etc.)
- Результат — LAS-файл: текстовый ASCII формат, колонки Depth + кривые

**Ключевое исправление понимания:**
- Скрипт НЕ должен открывать готовый .nlgx и "скроллить" его
- Скрипт ДОЛЖЕН создавать НОВЫЙ проект из изображения через File → New
- Данных в .nlgx изначально нет — они появляются только после трейсинга по изображению

---

## 2. Структура входных данных

```
F:\nds\projects\
└── Semeguniv_020\          ← папка одной скважины (указывает пользователь)
    ├── wlg\                ← здесь будут сохраняться .nlgx файлы
    └── img\  (или root)    ← исходные изображения (50-80 штук)
        ├── Semeguniv_20_BK+MBK_3080_3520_200_D1.tif
        ├── Semeguniv_20_GK_3080_3520_200_D1.tif
        ├── Semeguniv_20_SP_2800_3080_200_D1.tif
        └── ...
```

**Из имени файла можно извлечь:**
- Название скважины: `Semeguniv_20`
- Тип кривых: `BK+MBK`, `GK`, `SP`
- Ориентировочный диапазон глубин: `3080_3520` → from=3080, to=3520 (метры) (может физически отличаться на изображении)
- Разрешение: `200` DPI
- Номер части: `D1, D2`

---

## 3. Реальный рабочий пайплайн (новое понимание)

```
Пользователь указывает папку проекта, например, Semeguniv_020: F:\nds\projects\Semeguniv_020 
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  ДЛЯ КАЖДОГО изображения в папке:                           │
│                                                             │
│  БЛОК 1: Создать новый проект в NeuraLOG                    │
│  ─────────────────────────────────────────────────────────  │
│  File → New → Мастер создания проекта:                      │
│    • Кнопка "Далее/Next"                                    │
│    • "Browse for Image..." → выбрать .tif файл              │
│    • Кнопка "Готово/Finish"                                 │
│  → Открывается пустой .nlgx с загруженным изображением      │
│                                                             │
│  БЛОК 2: Анализ изображения (Computer Vision + AI)          │
│  ─────────────────────────────────────────────────────────  │
│  Сделать скриншот области с изображением в NeuraLOG         │
│  Передать Claude Vision API:                                │
│    → Сколько кривых на изображении?                         │
│    → Имена кривых (BK, MBK, GK, SP...)                      │
│    → Единицы измерения (Ohmm, mV, %)                        │
│    → Шкала: минимум и максимум (0-20 Ohmm, -25..+25 mV)     │
│    → Диапазон глубин (3080-3520м — из шкалы глубин)         │
│    → Цвет каждой кривой                                     │
│                                                             │
│  БЛОК 3: Калибровка в NeuraLOG                              │
│  ─────────────────────────────────────────────────────────  │
│  Установить параметры в интерфейсе:                         │
│    • Глубина начала / конца (Calibrate & Trace toolbar)     │
│    • Для каждой кривой: имя, единицы, min/max шкалы         │
│    • Привязка по двум точкам (верх/низ изображения)         │
│                                                             │
│  БЛОК 4: Трейсинг (оцифровка)                               │
│  ─────────────────────────────────────────────────────────  │
│  BTN_FORWARD → трейсер следует по кривой автоматически      │
│  Ждать пока depth достигнет depth_to (мониторинг)           │
│  BTN_STOP                                                   │
│                                                             │
│  БЛОК 5: Сохранение и экспорт                               │
│  ─────────────────────────────────────────────────────────  │
│  Ctrl+S → сохранить .nlgx с данными                         │
│  File → Export → Digital Log → LAS 2.0                      │
│  → F:\nds\output\имя_файла.las                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Что уже сделано и работает

### 4.1 Разведка UI (полностью готово)

**Файл: `neuralog_win32.py`** — контроллер NeuraLOG через Win32 API

| Метод                    | Статус      | Описание                                       |
| ------------------------ | ----------- | ---------------------------------------------- |
| `connect()`              | ✅ Работает  | Найти NeuraLOG по HWND (MDIClient как признак) |
| `ensure_visible()`       | ✅ Работает  | Восстановить окно с (-32000,-32000)            |
| `get_open_projects()`    | ✅ Работает  | MDI-children список                            |
| `switch_to_project()`    | ✅ Работает  | WM_MDIACTIVATE                                 |
| `start_trace()`          | ✅ Работает  | BM_CLICK BTN_FORWARD                           |
| `stop_trace()`           | ✅ Работает  | BM_CLICK BTN_STOP                              |
| `get_current_depth()`    | ✅ Работает  | Static "3013" → float                          |
| `trace_to_depth(target)` | ✅ Работает  | Мониторинг + стоп                              |
| `save()`                 | ✅ Работает  | Ctrl+S                                         |
| `get_curves_list()`      | ✅ Работает  | ListBox LB_GETTEXT (8 кривых)                  |
| `get_menu_structure()`   | ✅ Работает  | 157 пунктов, все CMD ID                        |
| `export_las()`           | ⚠️ В работе | Диалог находит, но CMD не всегда открывает     |
| `open_project()`         | ⚠️ Частично | Ctrl+O работает, MDI-switch добавлен           |

### 4.2 Известные HWND контролов (из uia_probe + dialog_probe)

```python
# Главное окно NeuraLOG
main_window: class="Afx:400000:8:10003:0:2640923"

# Ключевые контролы (всегда одинаковые по title)
BTN_FORWARD:  class="Button"  title="BTN_FORWARD"  (старт трейсера)
BTN_STOP:     class="Button"  title="BTN_STOP"
BTN_REVERSE:  class="Button"  title="BTN_REVERSE"
Slider1:      class="msctls_trackbar32"  (скорость, 0-100000, сейчас=max)
Static depth: class="Static"  title="3013"  (текущая глубина)
ListBox:      id=3013  (список кривых в проекте)

# Тулбары
Toolbar "Calibrate & Trace"  ← главный тулбар для калибровки
Toolbar "Main"
Toolbar "Color Selection"
Toolbar "Curve Edit"
Toolbar "Curve/Track Tools"
Toolbar "Zoom"
Toolbar "Quality Control"

# Диалог "Digital Curve Output" (экспорт LAS)
Edit id=3369:   путь к файлу (File)
Edit id=3371:   Start Depth (M)
Edit id=3372:   Stop Depth (M)  
Edit id=3373:   Depth Step (M) = 0.1
Button id=3387: View After Generation (checkbox)
Button id=1:    OK
ListBox id=3013: список кривых для экспорта
ComboBox id=3388: Curve Output Type (LAS 2.0)
```

### 4.3 Меню команды (CMD ID)

```python
CMD 3006  = File → Export → Digital Log... (открывает "Digital Curve Output")
CMD 3157  = первый пункт меню 1 (предположительно File → New или Open)
# Полная карта 157 пунктов в dump_menu_ids()
```

### 4.4 Наблюдения по трейсеру

- Скорость трейсинга: ~0.3 м/с при speed=100000 (максимум)
- За 5 секунд: глубина 3078 → 3088 (+10м)
- Диапазон 440м (3080→3520): ~24 мин
- Таймаут установлен: `max(1800, remaining/0.20)` секунд
- Нужно сделать трейсинг в зависимости от изображения и скорости распознования изображения

---

## 5. Что нужно сделать (новый план)

### БЛОК 1: Мастер создания проекта (New Project Wizard)

**Задача:** Автоматизировать File → New → выбрать изображение → Finish

**Нужно разведать:**
1. CMD ID для File → New (возможно CMD 3157)
2. Шаги мастера:
   - Какие окна появляются (title, class)
   - Кнопки "Далее", "Назад", "Готово"
   - Где поле выбора изображения
   - Вариант "Browse for Image..." vs "Hide Digitized Images"
3. После Finish: как выглядит пустой .nlgx с изображением

**Метод разведки:**
```python
# Запустить вручную File → New, фиксировать появляющиеся окна
python test_buttons.py wizard_probe
```

**Реализация:**
```python
def new_project_from_image(self, image_path: str, project_save_path: str) -> bool:
    # 1. SendMessage WM_COMMAND <New_CMD_ID>
    # 2. Ждать wizard окна
    # 3. Кликнуть Next
    # 4. Найти Browse button → click → выбрать файл
    # 5. Finish
    # 6. Ждать загрузки изображения
    # 7. Save As project_save_path
```

---

### БЛОК 2: Анализ изображения каротажа

**Задача:** Понять что нарисовано на изображении без участия человека

**Что нужно извлечь:**
- Количество кривых
- Имя каждой кривой (BK, MBK, GK, SP, ПС...)
- Единицы измерения (Ohmm, mV, %, мкР/ч...)
- Шкала по горизонтали: min и max значения
- Тип шкалы: линейная или логарифмическая
- Цвет/стиль каждой кривой (для идентификации на изображении)
- Диапазон глубин (по вертикальной шкале)

**Подходы (от простого к сложному):**

#### Вариант А: Claude Vision API (рекомендуется для MVP)
```python
import anthropic, base64

def analyze_log_image(image_path: str) -> dict:
    """
    Передать изображение в Claude Vision и получить структуру кривых.
    """
    client = anthropic.Anthropic()
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/tiff",  # или jpeg/png
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": """Это изображение каротажного графика скважины (well log).
Определи:
1. Количество кривых на графике
2. Для каждой кривой: имя, единицы измерения, минимальное значение шкалы, максимальное значение
3. Диапазон глубин (верхняя и нижняя отметки по вертикальной шкале)
4. Тип шкалы для каждой кривой (линейная/логарифмическая)

Ответь ТОЛЬКО JSON:
{
  "depth_from": 3080,
  "depth_to": 3520,
  "curves": [
    {
      "name": "BK",
      "unit": "Ohmm",
      "scale_min": 0,
      "scale_max": 20,
      "scale_type": "linear",
      "color": "black"
    }
  ]
}"""
                }
            ]
        }]
    )
    
    import json, re
    text = response.content[0].text
    json_str = re.search(r'\{.*\}', text, re.DOTALL).group()
    return json.loads(json_str)
```

**Почему Vision API:**
- Читает текстовые подписи кривых (OCR уже встроен)
- Понимает контекст ("это каротаж, единицы = Ohmm")
- Справляется с нестандартными шкалами (-25..+75 mV)
- Не требует отдельной настройки под каждый тип каротажа

#### Вариант Б: OpenCV + pytesseract (для точных координат)
```python
# Дополнительно к Vision: извлечь пиксельные координаты шкал
# для передачи в Calibrate & Trace
import cv2, pytesseract

def find_scale_pixels(image_path: str) -> dict:
    """Найти пиксельные координаты меток на шкале глубин."""
    img = cv2.imread(image_path)
    # Область шкалы глубин — левая колонка изображения
    # Template matching для числовых меток
    ...
```

**Итоговый подход:** Vision API → понять структуру, OpenCV → точные координаты для калибровки. Использовать собственные мощности, без использования облачных сервисов.

---

### БЛОК 3: Калибровка в NeuraLOG

**Задача:** Установить в NeuraLOG параметры до трейсинга

**Что нужно настроить:**
1. **Привязка по глубине** (Calibrate & Trace):
   - Кликнуть на верхнюю метку глубины на изображении
   - Ввести значение глубины (3080м)
   - Кликнуть на нижнюю метку
   - Ввести значение (3520м)

2. **Параметры кривой** (Curve/Track Dialog):
   - Имя: Определяется с названия, уточняется в мнемонике по изображению
   - Единицы: Определяются с мнемоники и изображения
   - Левый предел: Определяется с изображения
   - Правый предел: Определяется с изображения

3. **Цветовой порог трейсера** (Color Selection toolbar):
   - Настроить под цвет конкретной кривой

**Нужно разведать:**
- Какие CMD ID в тулбаре "Calibrate & Trace"
- Как добавить новую кривую (Curve/Track Dialog)
- Как задать min/max шкалы через интерфейс

```python
def calibrate_depth(self, pixel_top: int, depth_top: float,
                         pixel_bottom: int, depth_bottom: float):
    """Привязать пиксели изображения к значениям глубины."""
    # Нужен CMD ID из тулбара Calibrate & Trace
    # Предположительно: кликнуть точку на изображении → ввести глубину
    pass

def setup_curve(self, name: str, unit: str, 
                scale_min: float, scale_max: float):
    """Создать/настроить кривую в Curve/Track Dialog."""
    # Через ListBox id=3013 (список кривых)
    # Кнопки в Curve/Track Dialog: Add, Remove, Units
    pass
```

---

### БЛОК 4: Трейсинг (уже работает)

**Статус: ✅ ГОТОВО**

```python
nl.start_trace()  # BTN_FORWARD
nl.wait_for_trace_complete()  # или trace_to_depth(target)
nl.stop_trace()
```

**Уточнение по "шагу трейсера":**
- Скорость = 100000 (максимум слайдера) → уже максимально быстро
- "Шаг" в LAS = Depth Step в диалоге экспорта (id=3373, сейчас=0.1м)
- Если нужен крупнее шаг → изменить через WM_SETTEXT в Edit id=3373

---

### БЛОК 5: Сохранение и экспорт

**Статус: ⚠️ В РАБОТЕ** (диалог найден, автооткрытие нестабильно)

**Проблема:** `SendMessage WM_COMMAND 3006` иногда не открывает диалог

**Версии для тестирования:**
```python
# Попытка 1: SendMessage к main_window
win32api.SendMessage(main_window, WM_COMMAND, 3006, 0)

# Попытка 2: SendMessage к активному MDI-child  
win32api.SendMessage(mdi_child, WM_COMMAND, 3006, 0)

# Попытка 3: Ctrl+Shift+F через pyautogui
pyautogui.hotkey("ctrl", "shift", "f")
```

---

## 6. Неизвестные вещи (нужна разведка)

| Вопрос | Как узнать |
|--------|------------|
| CMD ID для File → New | `python test_buttons.py wizard_probe` — открыть вручную и поймать окно |
| Шаги мастера New Project | Записать последовательность окон/кнопок вручную |
| Как выглядит пустой проект с изображением | Скриншот после wizard_probe |
| CMD ID кнопок "Calibrate & Trace" тулбара | `explore_toolbar(toolbar_calibrate)` |
| Как задать параметры кривой | Записать действия в Curve/Track Dialog |
| Формат .nlgx — можно ли читать метаданные | HxD diff двух похожих файлов |
| Сколько кривых бывает на одном изображении | Изучить примеры файлов |
| Есть ли заголовок с именами кривых на самом изображении | Визуально |

---

## 7. Файлы проекта (текущий статус)

```
F:\nds\Auto\
├── neuralog_win32.py      ← Win32 контроллер (1400 строк)
│                             connect, open_project, trace, export...
├── batch_pipeline.py      ← Оркестратор батча (577 строк)
│                             find_projects, process_one, run_all...
├── test_buttons.py        ← Тесты и диагностика
│                             modes: state|trace|menu|status|export2|dialog_probe|full
├── uia_probe.py           ← Разведка UIAutomation (одноразовый)
└── architecture.html      ← Архитектурная схема
```

---

## 8. Среда разработки

```
Python:     3.14.5 64-bit
            C:\Users\Defou1t\AppData\Local\Python\pythoncore-3.14-64\python.exe
NeuraLOG:   v2010, 32-bit MFC, class "Afx:400000:..."
Проекты:    F:\nds\projects\<скважина>\
Вывод:      F:\nds\output\
Логи:       F:\nds\logs\
Скрипты:    F:\nds\Auto\
IDE:        Spyder 6.1.4 (F:\Soft\SpyderPython)

Зависимости:
  pip install pywinauto pywin32 pyautogui pillow opencv-python anthropic
```

**Предупреждение:**
```
UserWarning: 32-bit application should be automated using 32-bit Python
```
Это предупреждение — не ошибка. win32 backend работает корректно с 64-bit Python.
Для ReadProcessMemory нужен 32-bit Python (но он нужен только для toolbar button probe).

---

## 9. Первые шаги для нового диалога

### Шаг 1: Разведать мастер File → New

Открыть NeuraLOG вручную, запустить:
```python
python test_buttons.py wizard_probe
```
*(нужно добавить режим wizard_probe в test_buttons.py)*

Записать:
- Заголовки всех появляющихся окон
- Тексты кнопок
- Какое поле/кнопка позволяет выбрать изображение

### Шаг 2: Протестировать Vision анализ изображения

```python
# Взять один реальный .tif из F:\nds\projects\Semeguniv_020\
# Запустить analyze_log_image(path)
# Проверить правильность извлечённых данных
```

### Шаг 3: Разведать Calibrate & Trace тулбар

```python
nl = NeuraLog()
nl.explore_all_toolbars()  # выведет CMD ID всех кнопок тулбара Calibrate & Trace
```

### Шаг 4: Собрать полный пайплайн

Только после Шагов 1-3 — реализовать `auto_digitize_image(image_path)`.

---

## 10. Концептуальная схема итоговой системы

```
                     ПОЛЬЗОВАТЕЛЬ
                          │
                 Указывает папку скважины
                 F:\nds\projects\Semeguniv_020
                          │
                          ▼
              ┌─────────────────────────┐
              │   batch_pipeline.py     │
              │  Найти все .tif/.jpg    │
              │  Построить очередь      │
              └────────────┬────────────┘
                           │  для каждого изображения
                           ▼
         ┌─────────────────────────────────────┐
         │      auto_digitize_image()          │
         │                                     │
         │  ┌─────────────────────────────┐    │
         │  │  БЛОК 1: New Project        │    │
         │  │  neuralog_win32.py          │    │
         │  │  new_project_from_image()   │    │
         │  └─────────────┬───────────────┘    │
         │                │                    │
         │  ┌─────────────▼───────────────┐    │
         │  │  БЛОК 2: Image Analysis     │    │
         │  │  Claude Vision API          │    │
         │  │  analyze_log_image()        │    │
         │  │  → curves, units, scale     │    │
         │  └─────────────┬───────────────┘    │
         │                │                    │
         │  ┌─────────────▼───────────────┐    │
         │  │  БЛОК 3: Calibration        │    │
         │  │  neuralog_win32.py          │    │
         │  │  calibrate_depth()          │    │
         │  │  setup_curve()              │    │
         │  └─────────────┬───────────────┘    │
         │                │                    │
         │  ┌─────────────▼───────────────┐    │
         │  │  БЛОК 4: Tracing            │    │  ← Перепроверить
         │  │  start_trace()              │    │
         │  │  trace_to_depth(target)     │    │
         │  │  stop_trace()               │    │
         │  └─────────────┬───────────────┘    │
         │                │                    │
         │  ┌─────────────▼───────────────┐    │
         │  │  БЛОК 5: Export             │    │  ← ПОЧТИ РАБОТАЕТ
         │  │  save() → export_las()      │    │
         │  │  → F:\nds\output\имя.las    │    │
         │  └─────────────────────────────┘    │
         └─────────────────────────────────────┘
                          │
                          ▼
              F:\nds\output\Semeguniv_20_BK_3080_3520.las
              F:\nds\output\Semeguniv_20_GK_3080_3520.las
              F:\nds\output\Semeguniv_20_SP_2800_3080.las
              ...
```

---

*Файл создан для продолжения разработки в новом диалоге.*
*Все накопленные знания о HWND, CMD ID, контролах — актуальны и используются.*
Нужно добавить в скрипт использование собственных мощностей для анализа изображения и векторизации, либо обучить собственного агента с возможностью передавать его на другие устройства.

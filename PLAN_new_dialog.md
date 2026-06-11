# CarrotageAuto — оцифровка каротажа в LAS через NeuraLOG

Документ для продолжения разработки. Оптимизирован 11.06.2026 (чекпоинт).

---

## 1. Цель

Автоматически оцифровать каротажные кривые из растровых сканов (.jpg/.tif/.png)
в формат **LAS 2.0** с помощью **NeuraLOG v2010** (32-bit MFC). На скважину
50–80 изображений. Сдача заказчику: **nlgx + bck + las на каждое изображение**
(заказчик перепроверяет с исходником).

**Пайплайн на изображение:**
```
Изображение → [Блок 2] локальный анализ (CV+OCR+VLM): глубины, кривые, шкалы,
                       перевыносы, пиксельная калибровка
            → [Блок 3] калибровка в NeuraLOG: Depth Axis + Scale Axis + кривые
            → [Блок 4] трейсинг (NeuraLOG следует по линии)
            → [Блок 5] сохранение nlgx/bck + экспорт LAS
```

**Входные данные:** `F:\nds\projects\<well>\` → `img\` (сканы), `wlg\` (nlgx+bck),
`las\` (эталонные LAS эксперта). Имя файла: `Well_CURVES_from_to_scale_part.jpg`,
напр. `Semeguniv_20_BK+MBK_3080_3520_200_D1.jpg` → скважина, кривые, интервал
3080–3520 м, **масштаб 1:200** (НЕ DPI!), часть D1.

---

## 2. СТРАТЕГИЯ (выбор эксперта)

UI NeuraLOG нельзя водить синтетическими кликами по канве (см. §6). Поэтому:

- **Резистивные кривые (BK/MBK/BKZ/GZ…) → B4 human-in-the-loop:** Блок 2 считает
  всё (пиксели граней, глубины, шкалы), код армирует/заполняет диалоги/жмёт
  кнопки/экспортирует, **эксперт делает ТОЛЬКО канва-клики** (грани Depth Axis).
  NeuraLOG трейсит перевыносы нативно — эталонное качество. ← ОСНОВНОЙ ПУТЬ.
- **Запасной (одношкальные кривые, гамма/SP):** свой трейсер Блока 2 → LAS
  напрямую (`las_writer.py`), без NeuraLOG. Упирается в перевыносы (§5).

---

## 3. СТАТУС БЛОКОВ

| Блок | Статус | Кратко |
|------|--------|--------|
| **2. Анализ изображения** | ✅ Силён | Калибровка глубины точна (rmse 0.05–0.11 м); шкалы/тики VLM; трасса (срез перевыносов — гэп) |
| **2. LAS-писатель** | ✅ Готов | `las_writer.py` |
| **2. QC** | ✅ Готов | `compare_trace_las.py`, `batch_qc.py` |
| **3. B4 Depth Axis** | ✅ РАБОТАЕТ | `calibrate_depth_assisted` — рамка ставится (эксперт кликает, код вписывает) |
| **3. B4 Scale Axis/кривые** | ⬜ Следующее | Add New→Other→abbrev; поля Left/Right; Add Backup ×5 — код-онли |
| **4. Трейсинг** | ✅ Готов | `start_trace`/`stop_trace`/`trace_to_depth` (BM_CLICK) |
| **5. Экспорт LAS** | ⚠️ CMD найден | CMD **32837** (не 3006!); диалог «Digital Curve Output» |

---

## 4. КАНОН РАМКИ (правила оцифровки — эксперт, ЗАПОМНИТЬ)

Снято с эталона `Semeguniv_20_BK+MBK_4350_4610` (бинарь nlgx + панель + скрины).

1. **Сетка 1:200 → 1 жирная горизонталь = 10 клеток = 4 м.** Грани рамки идут
   …4348, 4352, 4356… Рукописные подписи (4350, 4360) НЕ совпадают с гранями —
   депт-линии ставить НА ЖИРНУЮ ГРАНЬ (чуть выше/ниже подписи), не резать клетки.
2. **Depth Axis создаётся ОДИН раз** = грани рамки по max границам всех линий
   (эталон: Top **4348**, Bottom **4612**; в nlgx — два double @938/@946).
   Короткие кривые внутри — норма. Снап глубин к сетке 4 м.
3. **На каждый масштаб — свой Scale Axis** (панель SCALE AXIS, кнопка `Add Backup`):
   base + перевыносы. Значения Left/Right = value кривой на КРАЯХ изображения
   (экстраполяция тиков). Эталон: BK 0–**19**→0–**95**→0–**475** (×5);
   MBK −55..40→−275..200→−1375..1000 + SCCH3 −11..8.
4. **Типы переходов масштаба:** множительные (2х/5х/10х/100х — `Add Backup`);
   линейные backup right/left (0-20, 20-40 — продолжение без умножения);
   `Scale Change` для прочего. ВАЖНО: при повторном ×5 значения НЕ перемножаются
   авто (scale3 останется 0-95 вместо 0-475) — править вручную.
5. **Пунктир справа от линии = скрытый переход масштаба.** Пример: MBK с 3077.9 м
   идёт сразу во 2-м масштабе.
6. **Оцифровка сверху вниз. На одной глубине — только одна трасса семейства**
   (DA1SA1 и 5X1_DA1SA1 перезаписывают друг друга). Приоритет МЕНЬШЕМУ масштабу.
7. **Пики доводить до конца** (до квадратика), стыковать на переходах. Никаких
   сглаживаний (НЕ использовать Smooth Curve CMD 32866).
8. **Порядок:** Depth Axis → Depth Grids (4 м) → Add New кривая → base-шкала →
   Add Backup ×5. `Add Backup` работает ТОЛЬКО после `Add New`.

---

## 5. БЛОК 2 — анализ изображения (`analyze_log_image.py`)

5 слоёв: filename → OpenCV-геометрия → OCR → LM Studio VLM → merger. Результат —
строгий JSON `*_analysis.json` рядом со сканом (+ `--debug-overlay`).

**Что работает:**
- **Калибровка глубины** (`pixel_calibration`): детект блобов-меток → VLM читает
  числа → робастная подгонка depth=a·y+b. rmse 0.05–0.11 м. `depth_axis_suggestion`
  и `depth_start/end` у кривых.
- **Детект меток** (исправлено): фильтр числоподобных блобов `w≥40 и w/h≥1.6`
  (отсекает узкие тики шкалы, берёт левую колонку чисел).
- **Шкалы:** OCR-строки + VLM («0 2 4 6 ОММ»), `value_ticks_px`, тип linear/log.
- **Трасса:** полоса трека → центроид строки → depth/value сэмплы.
- `_track_band_line` — континуити-трекер (готов, НЕ включён: на перевыносах
  залипает; нужен вместе с разворачиванием уровней).

**ГЛАВНЫЙ ГЭП — ПЕРЕВЫНОСЫ (research-grade, отдельная сессия):**
Трасса ловит только base-шкалу (BK 0–8), теряя ×5/×25/×125 (эталон BK→309).
Структура = непрерывное оборачивание пера: `value = base(x)·5^level`. Две попытки
развернуть (счётчик оборотов; log-непрерывность) ПРОВАЛИЛИСЬ — нужна ГЛОБАЛЬНАЯ
оптимизация уровней (Viterbi/DP: state=level, цена=гладкость+штраф перехода+
соответствие сегментам), не жадный проход. ВАЖНО: весь ground-truth (10 LAS)
резистивный → свой трейсер для резистивных кривых гейтится на этой задаче и не
валидируется чисто (нет single-scale эталона).

**Окружение Блока 2:** Python `C:\Users\Defou1t\AppData\Local\Python\pythoncore-3.14-64\python.exe`
(opencv/numpy/pillow/pywin32/pywinauto; tesseract/paddle НЕ ставились).
LM Studio: `C:\Users\Defou1t\.lmstudio\bin\lms.exe`, vision-модель
`google/gemma-4-26b-a4b` (reasoning-тип: ответ м.б. в `reasoning_content`,
max_tokens 3000–6000, чтение строки шкалы 1–3 мин). Запуск:
`lms server start && lms load google/gemma-4-26b-a4b -y`.

**Датасет ошибок (батч-QC 11.06.2026):** «BKZ»=зонды GZ4/5/6+OGZ, «MK»=MGZ+MPZ
(наложены в одном треке — нужно разделение); MK_3720 — детект меток взял не ту
колонку; BK+MBK_3690 — кривые разными глубинными сегментами.

---

## 6. БЛОК 3 — калибровка в NeuraLOG (B4) + ключевые Win32-находки

### 6.1 Что РАБОТАЕТ через код (примитивы)
- **Подключение:** `NeuraLog()` (win32 backend), `cm.main_window/mdi_client/status_bar/toolbar_calibrate`.
- **Канва-вид:** класс `AfxFrameOrView42` (потомок mdi_client). Скроллбары —
  ОТДЕЛЬНЫЕ контролы `ScrollBar` (вид своих не имеет).
- **Маппинг image↔client (аналитически, без статус-бара):**
  `Z = (GetScrollInfo(vbar,SB_CTL).nMax+1) / image_native_height` (= зум);
  `client_y = img_y·Z − scroll_pos`; `client_x ≈ x_margin(≈14) + img_x·Z`.
- **Скролл (надёжный):** `SetScrollInfo(vbar, want)` + `WM_VSCROLL` (HIWORD=want |
  SB_THUMBPOSITION) РОДИТЕЛЮ скроллбара. WM_MOUSEWHEEL НЕНАДЁЖЕН. После прыжка —
  нудж LINEDOWN+LINEUP + InvalidateRect/UpdateWindow (иначе вид «залипает»).
- **Арминг инструмента оси = «Create Depth and Scale Axes» = CMD 32840.**
  НЕ 3194 (Raster Calibration — НЕ создаёт Depth Axis!). Повторное WM_COMMAND
  по нажатой кнопке ВЫКЛЮЧАЕТ её (toggle) — слать только если `_is_tool_armed`==False
  (TBSTATE_CHECKED 0x01). Опознать активную кнопку: `detect_armed_tool.py`.
- **Диалог глубины** = «**Set Depth Axis Value**» (TOP/BOTTOM DEPTH + combo
  Feet/Meters + чекбокс Lithology + OK). Детект по подстроке «set depth»+«value».
  Заполнение: WM_SETTEXT в Edit, CB_SETCURSEL для единиц (Feet=0, Meters=1 —
  текст combo межпроцессно НЕ читается), BM_CLICK OK. **Фокус НЕ нужен.**
- **Foreground (если надо):** AttachThreadInput-трюк.
- **Flyby-тултипы кнопок видны ТОЛЬКО при активном окне NeuraLOG.**

### 6.2 ЖЁСТКИЙ БЛОКЕР — синтетический клик по КАНВЕ не регистрируется
PostMessage WM_LBUTTON*, реальный mouse_event (foreground+позиция+SetFocus+jiggle),
SendInput — ВСЕ не открывают диалог. DPI=100%, позиция верна. computer-use
(реальный OS-ввод) клик принимает, но фокус нестабилен. 2 монитора (virtual
5120×1754, origin y=-314) ломают SendInput-absolute; SetCursorPos позиционирует
верно. ⇒ канва-клики делает ЭКСПЕРТ (B4), остальное — код.

### 6.3 B4 — как работает (`b4_calibrate.py` + `calibrate_depth_assisted`)
1. `plan_from_analysis` читает калибровку Блока 2, снапит грани к сетке 4 м →
   Top/Bottom депт + пиксели + track_x.
2. `calibrate_depth_assisted`: arm 32840 → scroll к грани → курсор-наводка →
   эксперт кликает грань → `_fill_depth_interval_dialog` ловит «Set Depth Axis
   Value» и вписывает глубину+Meters+OK. Так Top, потом Bottom (120 с на клик).
3. **ПРОВЕРЕНО: Yatskivska → РЕЗУЛЬТАТ OK, Depth Axis 3180/3580 встал.**
Предусловие: в NeuraLOG ОТКРЫТО изображение (File>Open>Log Image).

### 6.4 Известные HWND/CMD (durable)
```
main_window: class "Afx:400000:8:..."; канва "AfxFrameOrView42"
Трейсер: Button "BTN_FORWARD"/"BTN_STOP"/"BTN_REVERSE"; speed = msctls_trackbar32
Тулбар "Calibrate & Trace" кнопки: 3194(Raster,НЕ юзать), 32840(Create Depth&
  Scale Axes ✓), 32841(Depth Grids), 32842, 32843, 32844, 32846
CMD: 32837 Export Digital Log (LAS!); 3106 Open Log Image; 57603 Save/57604 SaveAs;
  32851 Curve Name; 32852 Curve Scales; 32854 Scale Grid Values; 32853 Depth Grid
  Spacing; 3217 New Scale(перевыносы?); 32791 Area Tracer; 32884 Go to Depth;
  3134/3135/3136 Delete Depth/Scale/Curve. НЕ: 32866 Smooth Curve, 3006(=Save PDF)
Диалог «Digital Curve Output» (экспорт): Edit 3369 путь, 3371 Start, 3372 Stop,
  3373 Step(0.1), Button 3387 View, Button 1 OK, ComboBox 3388 тип (LAS 2.0)
.nlgx = TIFF («II*\0») + приватные теги 34768–34928; калибровка добавляет теги
  34908/34910/34912; глубины как double внутри блока данных.
```

### 6.5 СЛЕДУЮЩИЕ ШАГИ B4 (после рамки — почти всё код-онли)
- [ ] Кривая: `Add New`(CURVES) → диалог Select Curve → прокрутить к «Other...» →
      Create → диалог «Curve Abbreviation»: ТОЛЬКО аббревиатура (MBK) → OK.
      ВНИМАНИЕ: клик НИЖЕ «Other…» в списке закрывает NeuraLOG (эксперт).
- [ ] Scale Axis: поля Left/Right панели (WM_SETTEXT; Ctrl+A не работает — End+
      Backspace+ввод+Enter при ручном; кодом — WM_SETTEXT), единицы combo,
      `Add Backup` → диалог «Select Backup Curve type» (Backup Left/Right, 2x/5x/
      10x/100x, Scale Change) → 5x, с ручной правкой 3-го масштаба.
- [ ] Трейсинг: `start_trace()` (BM_CLICK BTN_FORWARD) → монитор глубины → stop.
- [ ] Экспорт: WM_COMMAND 32837 → заполнить «Digital Curve Output» → OK.
- [ ] Watchdog: NeuraLOG нестабилен (сам закрывается на кликах ниже Other / модалках)
      — reconnect/перезапуск проекта.
- [ ] Шкалы части файлов VLM не читает (Yatskivska MBK 0–5 дефолт) — добить чтение
      или брать из мнемоники.

---

## 7. ФАЙЛЫ ПРОЕКТА (`F:\nds\Auto\`)
```
analyze_log_image.py   Блок 2: анализ скана → *_analysis.json (+debug-overlay)
las_writer.py          LAS 2.0 из trace.samples → F:\nds\output\*_auto.las
compare_trace_las.py   QC: трасса vs эталонный LAS (corr/shift/график)
batch_qc.py            прогон Блока 2 по всем файлам с эталонными LAS
neuralog_win32.py      Win32-контроллер NeuraLOG (connect, trace, B4-калибровка)
b4_calibrate.py        B4 runner: ассистированная установка Depth Axis
detect_armed_tool.py   опознать активную кнопку калибр-тулбара
test_buttons.py        диагностика (state|trace|menu|analyze|…)
batch_pipeline.py      оркестратор батча (черновик)
mnemonics.json         мнемоника кривых (имена/единицы)
menu_map.txt/toolbar_map.txt  карты меню/тулбаров
PLAN_new_dialog.md     этот план
```

**Ground truth (Semeguniv_020):** 11 nlgx+bck в `wlg\`, 10 эталонных LAS в `las\`.

---

## 8. БЛИЖАЙШИЙ РОАДМАП
1. **B4 кривая+шкала+трейсинг+экспорт** (код-онли после рамки) — собрать
   end-to-end на Yatskivska MBK, получить nlgx/bck/las.
2. **Watchdog** на нестабильность NeuraLOG.
3. **Батч-оркестрация** 50–80 изображений (B4 для каждого: открыть, рамка
   эксперт-клик, остальное код).
4. **(R&D) Перевыносы Viterbi/DP** — для своего трейсера / валидации.
5. **(потом) Разделение наложенных кривых** (BKZ/MK), своя модель на датасете.

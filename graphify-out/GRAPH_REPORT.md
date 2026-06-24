# Graph Report - .  (2026-06-22)

## Corpus Check
- 75 files · ~81,789 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 907 nodes · 1621 edges · 43 communities (40 shown, 3 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 189 edges (avg confidence: 0.78)
- Token cost: 142,302 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Batch Pipeline Orchestration|Batch Pipeline Orchestration]]
- [[_COMMUNITY_QC & Track Evaluation|QC & Track Evaluation]]
- [[_COMMUNITY_Digitize to nlgx Writing|Digitize to nlgx Writing]]
- [[_COMMUNITY_Dataset Build & Level Decode|Dataset Build & Level Decode]]
- [[_COMMUNITY_NeuraLOG Agent Tooling|NeuraLOG Agent Tooling]]
- [[_COMMUNITY_Click-Replay Tracing (b4)|Click-Replay Tracing (b4)]]
- [[_COMMUNITY_Scale & Calibration Detection|Scale & Calibration Detection]]
- [[_COMMUNITY_nlgx Extract  LAS Export|nlgx Extract / LAS Export]]
- [[_COMMUNITY_NeuraLOG Win32 Automation|NeuraLOG Win32 Automation]]
- [[_COMMUNITY_Baseline Trace & Inference Eval|Baseline Trace & Inference Eval]]
- [[_COMMUNITY_Track Identity & Swaps|Track Identity & Swaps]]
- [[_COMMUNITY_NeuraLOG Panel & Wizard Control|NeuraLOG Panel & Wizard Control]]
- [[_COMMUNITY_U-Net Model & Training|U-Net Model & Training]]
- [[_COMMUNITY_Curve Dataset & Masks|Curve Dataset & Masks]]
- [[_COMMUNITY_Mask & Line Detection|Mask & Line Detection]]
- [[_COMMUNITY_Instance Extraction by Colour|Instance Extraction by Colour]]
- [[_COMMUNITY_Trace QC & Delivery|Trace QC & Delivery]]
- [[_COMMUNITY_Scale Calibration Dialogs|Scale Calibration Dialogs]]
- [[_COMMUNITY_Depth Calibration & Canvas Scroll|Depth Calibration & Canvas Scroll]]
- [[_COMMUNITY_LAS Export via Menu Command|LAS Export via Menu Command]]
- [[_COMMUNITY_OCR Layer Analysis|OCR Layer Analysis]]
- [[_COMMUNITY_Image Layout Detection (OpenCV)|Image Layout Detection (OpenCV)]]
- [[_COMMUNITY_Tracer Status & State Reading|Tracer Status & State Reading]]
- [[_COMMUNITY_Project Open  Save Control|Project Open / Save Control]]
- [[_COMMUNITY_Control Map & Connection|Control Map & Connection]]
- [[_COMMUNITY_Toolbar & Menu Exploration|Toolbar & Menu Exploration]]
- [[_COMMUNITY_Cross-Process Memory Reading|Cross-Process Memory Reading]]
- [[_COMMUNITY_UIA  MSAA Probing|UIA / MSAA Probing]]
- [[_COMMUNITY_Scale-from-OCR Extraction|Scale-from-OCR Extraction]]
- [[_COMMUNITY_Scale Reading via LM Studio|Scale Reading via LM Studio]]
- [[_COMMUNITY_Trace Button Control|Trace Button Control]]
- [[_COMMUNITY_Curve Trace Sampling|Curve Trace Sampling]]
- [[_COMMUNITY_LM Studio VLM Reading|LM Studio VLM Reading]]
- [[_COMMUNITY_Depth Axis Calibration Fit|Depth Axis Calibration Fit]]
- [[_COMMUNITY_Swap Resolution|Swap Resolution]]
- [[_COMMUNITY_Anthropic Legacy Analysis|Anthropic Legacy Analysis]]
- [[_COMMUNITY_Curve Normalization Helpers|Curve Normalization Helpers]]
- [[_COMMUNITY_Trace Wizard Dialog Flow|Trace Wizard Dialog Flow]]
- [[_COMMUNITY_nlgx  TIFF Parsing|nlgx / TIFF Parsing]]
- [[_COMMUNITY_Dialog Inspector|Dialog Inspector]]
- [[_COMMUNITY_Menu Resource Parser|Menu Resource Parser]]
- [[_COMMUNITY_Patch Applier|Patch Applier]]

## God Nodes (most connected - your core abstractions)
1. `NeuraLog` - 110 edges
2. `extract()` - 40 edges
3. `analyze_log_image()` - 27 edges
4. `BatchPipeline` - 20 edges
5. `ProjectInfo` - 16 edges
6. `digitize_one()` - 16 edges
7. `NeuraLog` - 15 edges
8. `main()` - 15 edges
9. `_calibrate_depth_axis()` - 13 edges
10. `_to_float()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `NeuraLOG Tracer digitizes (not scrolls)` --semantically_similar_to--> `Real OS click opens NeuraLOG canvas dialog (synthetic fails)`  [INFERRED] [semantically similar]
  architecture.html → PLAN_new_dialog.md
- `analyze_log_image()` --calls--> `Path`  [INFERRED]
  analyze_log_image.py → batch_qc.py
- `analyze_log_image_cached()` --calls--> `Path`  [INFERRED]
  analyze_log_image.py → batch_qc.py
- `analyze_batch()` --calls--> `Path`  [INFERRED]
  analyze_log_image.py → batch_qc.py
- `_validate_image_path()` --calls--> `Path`  [INFERRED]
  analyze_log_image.py → batch_qc.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Phase A — frame & scales modules** — digitizer_detect_scales, digitizer_set_depth_grid, digitizer_detect_calibration, digitizer_detect_masks [EXTRACTED 1.00]
- **Phase B — identity & behaviour modules** — digitizer_extract_instances, digitizer_behavior_priors, digitizer_track_identity, digitizer_digitize_b3 [EXTRACTED 1.00]
- **nlgx calibration model (axes + grid + traces)** — plan_new_dialog_nlgx_format, plan_new_dialog_depth_axis, plan_new_dialog_scale_axis, plan_new_dialog_depth_grid, plan_new_dialog_backup_scales [EXTRACTED 1.00]

## Communities (43 total, 3 thin omitted)

### Community 0 - "Batch Pipeline Orchestration"
Cohesion: 0.06
Nodes (42): BatchPipeline, main(), ProcessResult, ProjectInfo, batch_pipeline.py — Батч-оркестратор NeuraLOG v2015 ============================, Результат обработки одного проекта., Оркестратор батчевой обработки .nlgx проектов., nl          : экземпляр NeuraLog из neuralog_win32.py         export_cmd  : CMD (+34 more)

### Community 1 - "QC & Track Evaluation"
Cohesion: 0.05
Nodes (51): fix_backup_values(), load_analysis(), main(), plan(), NeuraLog, b4_digitize.py — B4: оцифровка изображения в NeuraLOG после рамки.  Цепочка (пре, Канон §4.4: при повторном ×5 NeuraLOG НЕ перемножает значения сам     (scale3 ос, find_image() (+43 more)

### Community 2 - "Digitize to nlgx Writing"
Cohesion: 0.06
Nodes (50): digitize_one(), main(), r""" digitize_auto.py — АВТОНОМНАЯ оцифровка (БЕЗ трассы-эталона): шаблон даёт т, Per-row трасса одного цвета: ведём по непрерывности (предсказание x+v, ближайший, _runs(), trace_color(), aggregate_xs(), curve_pred() (+42 more)

### Community 3 - "Dataset Build & Level Decode"
Cohesion: 0.06
Nodes (46): Dataset, features(), main(), r""" analyze_behavior.py — v2: различимы ли НАЛОЖЕННЫЕ кривые по ПОВЕДЕНИЮ (идея, curve_color(), main(), r""" analyze_colors.py — v2-разведка: разделимы ли НАЛОЖЕННЫЕ кривые по ЦВЕТУ? Д, Медианный RGB чернил вдоль трассы (берём самый ТЁМНЫЙ пиксель в окне ±win (+38 more)

### Community 4 - "NeuraLOG Agent Tooling"
Cohesion: 0.06
Nodes (36): main(), NeuraLogAgent, neuralog_agent.py — ИИ-агент для автоматизации NeuraLOG v2015 ==================, Найти все окна NeuraLOG и их состояние., Перевести фокус на главное окно NeuraLOG., Клик мышью по координатам., Ввести текст с клавиатуры (в активное поле)., Нажать горячую клавишу. Пример: ['ctrl', 's'] или ['alt', 'f', 'o'] (+28 more)

### Community 5 - "Click-Replay Tracing (b4)"
Cohesion: 0.10
Nodes (39): auto_click(), fill_dialog(), force_foreground(), auto_click_value.py — АВТОНОМНЫЙ канва-клик NeuraLOG (без эксперта, без Claude)., SetCursorPos-итерации к пикселю изображения. Возвращает (sx,sy,ix,iy)., Жёстко вывести окно вперёд (для seek нужен WM_MOUSEMOVE → live статус-бар)., read_cursor(), seek() (+31 more)

### Community 6 - "Scale & Calibration Detection"
Cohesion: 0.07
Nodes (37): Filename parsing (well/curves/depth/scale), curve_behavior(), curve_class(), main(), mnem(), r""" behavior_priors.py — B2 (Фаза B роадмапа §6.6.7): ПОВЕДЕНЧЕСКИЕ приоры лини, Поведенческие фичи кривой из пиксель-трассы (инвариантно к плотности кликов)., _summ() (+29 more)

### Community 7 - "nlgx Extract / LAS Export"
Cohesion: 0.09
Nodes (33): align_frac(), find_image(), main(), norm(), r""" Батч-проверка расшифровки nlgx по датасету Archive: для каждого nlgx (не _a, Векторизовано: доля точек трассы, у которых в окне ±rx,±ry есть чернила., export(), main() (+25 more)

### Community 8 - "NeuraLOG Win32 Automation"
Cohesion: 0.07
Nodes (31): batch_pipeline.py orchestrator (queue + report), Digital Curve Output dialog (LAS export, CMD 3006/32837), NeuraLOG Automation architecture (HTML), NeuraLOG Tracer digitizes (not scrolls), main(), plan_from_analysis(), b4_calibrate.py - B4 human-in-the-loop установка Depth Axis в NeuraLOG.  Код бер, Снап глубины к жирной грани сетки (4 м при масштабе 1:200). (+23 more)

### Community 9 - "Baseline Trace & Inference Eval"
Cohesion: 0.08
Nodes (33): corr_stats(), family_of(), level_fn(), main(), process_curve(), baseline_trace.py — ВАРИАНТ А: эвристика-baseline трассировки. Меряем КАЧЕСТВО А, trace_x (по строкам окна) -> (depth, value) через оракул-калибровку+level., Среднее x тёмных пикселей по строкам окна (NaN если пусто). (+25 more)

### Community 10 - "Track Identity & Swaps"
Cohesion: 0.09
Nodes (27): main(), r""" eval_swaps.py — метрика свопов: трасса _auto vs экспертная трасса orig, ТОЛ, traces(), main(), mnem(), r""" survey_levels.py — характеризация ПЕРЕВЫНОСОВ по ground-truth всего Archive, {idx: scale} -> список цепочек по next (35273)., scale_families() (+19 more)

### Community 11 - "NeuraLOG Panel & Wizard Control"
Cohesion: 0.09
Nodes (14): NeuraLog, Дамп всех пунктов меню с ID (без текста — owner-draw)., Высокоуровневый контроллер NeuraLOG v2015., Нажать кнопку тулбара по command_id через WM_COMMAND., HWND контролов панели Curve/Track Dialog (кэшируется)., Текст RICHEDIT user-prompt (состояние армированного инструмента)., Элементы ListBox: LB_GETTEXT, при owner-draw — item-data строка., LB_SETCURSEL + уведомление LBN_SELCHANGE родителю (MFC обновит панель). (+6 more)

### Community 12 - "U-Net Model & Training"
Cohesion: 0.11
Nodes (17): dice_of(), load_model(), main(), r""" eval_viz.py — диагностика «где датасет/модель мешают» (запускать после trai, per-tile Dice + предсказанные вероятности (для триптихов худших)., [img | img+GT(зел) | img+pred(красн)] горизонтально, T×(3T)×3., tile_dice(), triptych() (+9 more)

### Community 13 - "Curve Dataset & Masks"
Cohesion: 0.11
Nodes (21): Archive normalization report (39 wells, 1017 triples), Name normalization (dot vs underscore matching), curve_mask(), curve_polyline(), is_real_curve(), log_corr_vs_las(), match_las(), mnemonic() (+13 more)

### Community 14 - "Mask & Line Detection"
Cohesion: 0.16
Nodes (18): compute_masks(), detect_hlines(), detect_text(), detect_vlines(), gt_trace_mask(), ink_mask(), main(), r""" detect_masks.py — A3 (Фаза A роадмапа §6.6.7): маска НЕ-линейных зон планше (+10 more)

### Community 15 - "Instance Extraction by Colour"
Cohesion: 0.15
Nodes (18): _behavior_from_xs(), classify_ink(), extract_instances(), _gt_color(), _gt_xy(), instances_from_channel(), main(), r""" extract_instances.py — B1 (Фаза B роадмапа §6.6.7): инстанс-извлечение лини (+10 more)

### Community 16 - "Trace QC & Delivery"
Cohesion: 0.18
Nodes (18): _level_bnds(), main(), overlap_check(), _overlay(), _pick_color(), qc_curve(), qc_file(), r""" qc_trace.py — НЕЗАВИСИМАЯ аналитическая QC оцифровки: проверяет трассу _aut (+10 more)

### Community 17 - "Scale Calibration Dialogs"
Cohesion: 0.21
Nodes (8): Дождаться нового видимого top-level окна процесса (диалога)., [(hwnd, class, id, text), ...] всех потомков диалога., Нажать кнопку диалога по тексту (без учёта регистра/&) или по id., Закрыть месседжбокс процесса, вернуть его текст ('' если не было)., Создать base Scale Axis с участием эксперта (продолжение тула 32840         посл, Вписать значение в первый Edit диалога, опц. combo, нажать OK., Нажать «Add Backup» (3028) → диалог «Select Backup Curve type» →         выбрать, «Add New» (3019) → «Select Curve» → «More Curves...» (1234) →         «Curve Abb

### Community 18 - "Depth Calibration & Canvas Scroll"
Cohesion: 0.12
Nodes (8): HWND вида документа (AfxFrameOrView42) под mdi_client., HWND вертикального ScrollBar (вид своих скроллбаров не имеет)., (nMin, nMax, nPage, nPos) вертикального скроллбара., Прокрутить вид так, чтобы target_img_y оказался в видимой зоне.         Надёжный, Текущие экранные координаты пикселя изображения (или None вне вида)., Найти видимое окно (top-level или дочернее главного), чей заголовок         соде, Дождаться модального «Set Depth Interval Value», вписать глубину,         выстав, B4: установить Depth Axis с участием эксперта.         Код армирует, наводит кур

### Community 19 - "LAS Export via Menu Command"
Cohesion: 0.12
Nodes (9): Найти свежесозданный LAS (возраст ≤ max_age_sec) в hint_paths и папках проекта., Автоматически определить command_id для экспорта LAS.          Перебирает EXPORT, Текущий заголовок окна (содержит путь к проекту)., Извлечь путь к текущему проекту из заголовка окна., Выполнить команду меню по ID через WM_COMMAND.         target: "main"  — главное, Получить HWND активного MDI-child окна.         В MDI-приложении (NeuraLOG) кажд, Экспорт результата (LAS). command_id уточняется через dump_menu_ids()         ил, Найти открытый диалог Save As / Open — стандартный #32770 без родителя         и (+1 more)

### Community 20 - "OCR Layer Analysis"
Cohesion: 0.18
Nodes (16): _curve_from_name(), _extract_curve_names_from_text(), _extract_depth_pair_from_text(), _merge_filename_layer(), _merge_ocr_layer(), analyze_log_image.py - локальный анализ изображений каротажа.  MVP Блока 2:   1., Центры контрастных «прогонов» чернил в строке (x_local, width)., Континуити-трекер линии в полосе трека: на каждой строке выбирает     «прогон» ч (+8 more)

### Community 21 - "Image Layout Detection (OpenCV)"
Cohesion: 0.16
Nodes (16): analyze_log_image(), analyze_log_image_cached(), _cluster_positions(), _compact_lm_layer(), _cv2_read(), _detect_layout_opencv(), _estimate_result_confidence(), _filter_track_bounds() (+8 more)

### Community 22 - "Tracer Status & State Reading"
Cohesion: 0.13
Nodes (8): Включить «Create Depth and Scale Axes», ЕСЛИ ещё не включён.         Повторное W, Вывести текущее состояние в лог., Текст статус-бара (Ready / Tracing / координаты)., Текст из панели User Prompt (подсказки программы)., Прочитать текущую глубину (из Static контрола рядом с 'Current Depth:')., Прочитать текущее положение слайдера скорости., Прочитать позицию ползунка глубины (первый trackbar — позиция по глубине)., Ждать окончания трейсинга по стабилизации глубины.         Считается завершённым

### Community 23 - "Project Open / Save Control"
Cohesion: 0.12
Nodes (8): Отправить горячую клавишу в главное окно., Сохранить проект (Ctrl+S)., Открыть .nlgx проект в NeuraLOG.          Алгоритм:           1. Проверить — не, Создать новый проект: File > New, указать изображение и сохранить.         Требу, Скопировать текст в буфер обмена.         Первичный способ: win32clipboard (pywi, Восстановить и вывести окно NeuraLOG на передний план., Получить список всех открытых проектов (MDI-children).         Возвращает: [{"hw, Переключиться на уже открытый проект в MDI по части пути/имени.         Возвраща

### Community 24 - "Control Map & Connection"
Cohesion: 0.14
Nodes (7): ControlMap, Подключиться к запущенному NeuraLOG.         Стратегия: сначала ищем HWND через, Найти главное окно NeuraLOG через EnumWindows.         Критерии главного окна:, Вернуть окно на экран если оно свёрнуто или за пределами монитора.         Коорд, Обойти дочерние окна и заполнить ControlMap по классу и заголовку., Выбрать ListBox с наибольшим числом кривых (owner-draw, в Curve/Track Dialog)., HWND всех ключевых контролов NeuraLOG (из probe-отчёта).

### Community 25 - "Toolbar & Menu Exploration"
Cohesion: 0.14
Nodes (8): explore(), Получить список кнопок тулбара через TB_BUTTONCOUNT / TB_GETBUTTON.         Возв, Кнопка тулбара калибровки нажата (армирована)?, Вывести кнопки тулбара — для первичного исследования., Исследовать все тулбары — запустить один раз для маппинга команд., Распечатать структуру меню., Полная диагностика — запустить один раз для маппинга UI., Прочитать структуру главного меню через GetMenuString (надёжнее для MFC).

### Community 26 - "Cross-Process Memory Reading"
Cohesion: 0.17
Nodes (6): Открыть handle процесса окна для cross-process чтения/записи., Контекстный менеджер: VirtualAllocEx в процессе окна.         Возвращает (hproce, Прочитать size байт из памяти процесса окна., Прочитать часть статус-бара через SB_GETTEXTW (cross-process).          SB_GETTE, Прочитать имя кривой из owner-draw ListBox через LB_GETITEMDATA.         LB_GETT, Получить список кривых из owner-draw ListBox (LB_GETITEMDATA).

### Community 27 - "UIA / MSAA Probing"
Cohesion: 0.30
Nodes (11): main(), probe_controls(), probe_keyboard_shortcuts(), probe_win32_classes(), uia_probe.py — Диагностика UIAutomation / MSAA для NeuraLOG v2015 Запускать ПОСЛ, Ищем кнопки, меню, тулбары, диалоги, Прямой обход Win32 окон — работает всегда, без pywinauto, Ищем доступные горячие клавиши через меню (+3 more)

### Community 28 - "Scale-from-OCR Extraction"
Cohesion: 0.20
Nodes (10): _assign_scales_to_curves(), _bbox_to_xywh(), _classify_scale_ticks(), _extract_scales_from_ocr(), _normalized_ocr_blocks(), Найти в OCR-блоках строки-шкалы вида "0 2 4 6 8 ОММ".      Возвращает список шка, linear для арифметической прогрессии, log для геометрической, иначе None., Привязать найденные шкалы к кривым: по перекрытию с треками или по порядку. (+2 more)

### Community 29 - "Scale Reading via LM Studio"
Cohesion: 0.20
Nodes (10): _detect_depth_label_blobs(), Найти прямоугольники компактных чернильных блобов (рукописный текст).      Длинн, Найти прямоугольники рукописных меток глубины.      Метки — это 3-4-значные ЧИСЛ, Прочитать рукописные строки-шкалы ("0 2 4 6 8 ОММ") в шапках треков     через LM, Проверить, что тики ложатся на прямую value(x): защита от блобов-мусора., Самая верхняя горизонтальная строка из >=3 блобов с близкими cy., _read_scales_with_lm_studio(), _text_blob_boxes() (+2 more)

### Community 30 - "Trace Button Control"
Cohesion: 0.24
Nodes (5): Нажать BTN_FORWARD — начать трейсинг вперёд., Нажать BTN_STOP — остановить трейсинг., Нажать BTN_REVERSE — трейсинг назад., Отправить BM_CLICK напрямую кнопке по HWND., Запустить трейсер и ждать пока глубина достигнет target_depth.         Останавли

### Community 31 - "Curve Trace Sampling"
Cohesion: 0.25
Nodes (9): _clip_points_to_extent(), _detect_data_extent(), _extract_curve_traces(), _make_value_mapper(), Пересчитать сэмплы depth/value после обновления шкал кривых., Извлечь кривые как полилинии из чернильной маски.      Для каждой полосы трека:, Найти (start_row, end_row) непрерывных данных кривой.      Данные = плотная зали, Функция x_px -> значение кривой; None, если шкала неизвестна. (+1 more)

### Community 32 - "LM Studio VLM Reading"
Cohesion: 0.36
Nodes (8): _analyze_with_lm_studio(), _first_lm_studio_model(), _http_json(), _image_to_data_url_for_vlm(), Прочитать рукописные числа на кропах меток через LM Studio VLM., Прочитать одну строку шкалы. Возвращает (числа слева направо, unit)., _read_label_crops_lm_studio(), _read_scale_row_lm_studio()

### Community 33 - "Depth Axis Calibration Fit"
Cohesion: 0.25
Nodes (8): _calibrate_depth_axis(), _median_label_spacing(), _plausible_depth_window(), Робастная прямая v = m*y + b по парам (y, v).     Возвращает (m, b, индексы инла, Найти метки глубины на изображении и построить depth = a*y + b.      Источники з, _robust_linear_fit(), _sample_evenly(), _snap_label_step()

### Community 34 - "Swap Resolution"
Cohesion: 0.39
Nodes (7): main(), match_err(), r""" resolve_swaps.py — v2 пост-хок развязка свопов идентичности на пересечениях, для каждой GT — лучшая линия (min медиана |dx|) на общих строках., resolve(), rough(), to_array()

### Community 35 - "Anthropic Legacy Analysis"
Cohesion: 0.29
Nodes (7): _analyze_anthropic_legacy(), analyze_batch(), _empty_result(), _parse_filename_metadata(), _parse_json_response(), Анализ списка изображений., Извлечь подсказки из имени: Well_CURVE_3080_3520_200_D1.tif.      Четвертое числ

### Community 36 - "Curve Normalization Helpers"
Cohesion: 0.48
Nodes (7): _merge_model_layer(), _normalize_curve(), _normalize_curve_name(), _normalize_unit(), _parse_depth_label_text(), _to_float(), Any

### Community 37 - "Trace Wizard Dialog Flow"
Cohesion: 0.48
Nodes (6): find_child_by_id(), find_dialog_by_title(), handle_open_file_dialog(), main(), Скрипт сквозного прохода мастера создания проекта NeuraLOG (Шаг 1 -> Шаг 2 -> Ша, scan_window()

### Community 38 - "nlgx / TIFF Parsing"
Cohesion: 0.40
Nodes (5): describe(), dump_tag(), parse_tiff(), Парсер структуры .nlgx (NeuraLOG) — это TIFF-контейнер (II*\\0) с приватными тег, Подробный дамп одного тега: все значения + hex.

## Knowledge Gaps
- **10 isolated node(s):** `Depth Axis (depth<->pixel calibration)`, `Behavioural priors (SP smooth, CALI/RES peaky)`, `Band-detect (density-gap split for spaced curves)`, `adaptiveThreshold for faint pencil curves`, `Status-bar cursor seek (live readout navigation)` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Path` connect `QC & Track Evaluation` to `Batch Pipeline Orchestration`, `Digitize to nlgx Writing`, `Anthropic Legacy Analysis`, `Dataset Build & Level Decode`, `Click-Replay Tracing (b4)`, `Scale & Calibration Detection`, `nlgx Extract / LAS Export`, `NeuraLOG Win32 Automation`, `Baseline Trace & Inference Eval`, `Swap Resolution`, `Track Identity & Swaps`, `U-Net Model & Training`, `NeuraLOG Agent Tooling`, `Mask & Line Detection`, `Instance Extraction by Colour`, `Trace QC & Delivery`, `Image Layout Detection (OpenCV)`?**
  _High betweenness centrality (0.555) - this node is a cross-community bridge._
- **Why does `NeuraLog` connect `NeuraLOG Panel & Wizard Control` to `Batch Pipeline Orchestration`, `QC & Track Evaluation`, `Click-Replay Tracing (b4)`, `NeuraLOG Win32 Automation`, `Scale Calibration Dialogs`, `Depth Calibration & Canvas Scroll`, `LAS Export via Menu Command`, `Tracer Status & State Reading`, `Project Open / Save Control`, `Control Map & Connection`, `Toolbar & Menu Exploration`, `Cross-Process Memory Reading`, `Trace Button Control`?**
  _High betweenness centrality (0.357) - this node is a cross-community bridge._
- **Why does `main()` connect `QC & Track Evaluation` to `NeuraLOG Panel & Wizard Control`?**
  _High betweenness centrality (0.206) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `NeuraLog` (e.g. with `NeuraLog` and `NeuraLog`) actually correct?**
  _`NeuraLog` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 55 inferred relationships involving `Path` (e.g. with `_analyze_anthropic_legacy()` and `analyze_batch()`) actually correct?**
  _`Path` has 55 INFERRED edges - model-reasoned connections that need verification._
- **Are the 35 inferred relationships involving `extract()` (e.g. with `main()` and `main()`) actually correct?**
  _`extract()` has 35 INFERRED edges - model-reasoned connections that need verification._
- **What connects `analyze_log_image.py - локальный анализ изображений каротажа.  MVP Блока 2:   1.`, `Анализировать изображение каротажа.      provider:       - local      : filename`, `Анализ с кешированием JSON рядом с изображением или в cache_dir.` to the rest of the system?**
  _318 weakly-connected nodes found - possible documentation gaps or missing edges._
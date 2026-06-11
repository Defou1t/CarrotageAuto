"""
test_buttons.py — интерактивная разведка NeuraLOG UI
Запуск: python test_buttons.py [trace|modes|menu|all]
"""
import sys
import time
import logging
from neuralog_win32 import NeuraLog

logging.basicConfig(level=logging.INFO, format="%(message)s")


def test_state(nl: NeuraLog):
    print("\n=== Состояние ===")
    nl.print_state()
    print(f"  depth slider:  0x{nl.cm.slider_depth_position:08X}" if nl.cm.slider_depth_position else "  depth slider:  —")
    print(f"  speed slider:  0x{nl.cm.slider_tracer_speed:08X}" if nl.cm.slider_tracer_speed else "  speed slider:  —")
    print(f"  listbox:       0x{nl.cm.listbox_curves:08X}" if nl.cm.listbox_curves else "  listbox:       —")


def test_trace(nl: NeuraLog):
    print("\n=== Тест трейсера (5 сек) ===")
    d0 = nl.get_current_depth()
    print(f"  глубина до:    {d0}")
    nl.start_trace()
    time.sleep(5)
    d1 = nl.get_current_depth()
    print(f"  глубина после: {d1}")
    print(f"  статус:        {nl.get_status()!r}")
    nl.stop_trace()
    print(f"  остановлено,  глубина={nl.get_current_depth()}")


def test_trace_modes(nl: NeuraLog):
    print("\n=== Режимы трейсинга (Calibrate & Trace toolbar) ===")
    nl.explore_all_toolbars()
    print("\nПереключение режимов — наблюдайте UI NeuraLOG:")
    for cmd_id in [32840, 32841, 32842, 32843, 32844, 32846]:
        print(f"\n  CMD={cmd_id}...")
        nl.send_menu_command(cmd_id)
        time.sleep(1.5)
        input("  Что изменилось? Enter → следующая команда...")


def test_menu(nl: NeuraLog):
    print("\n=== ID меню (owner-draw, без текста) ===")
    items = nl.dump_menu_ids()
    export_ids = [it for it in items if it["command_id"] in NeuraLog.EXPORT_CMD_CANDIDATES]
    print(f"  всего пунктов: {len(items)}")
    print("  кандидаты экспорта:", export_ids)
    for it in items[:30]:
        print(f"    {it['path']:>12s}  ID={it['command_id']}")
    if len(items) > 30:
        print(f"    ... ещё {len(items) - 30}")


def test_status(nl: NeuraLog):
    """Проверить что get_status() теперь возвращает текст (не пусто)."""
    print("\n=== Статусбар ===")
    status = nl.get_status()
    print(f"  get_status() = {status!r}")
    if not status:
        print("  ⚠ Пустой ответ — статус-бар может использовать ANSI SB_GETTEXT (0x0402)")
        print("    Попробуй вручную: win32gui.SendMessage(nl.cm.status_bar, 0x0402, 0, remote)")
    else:
        print("  ✓ Работает!")


def test_export(nl: NeuraLog, output_path: str = r"F:\nds\output\test_export.las"):
    """
    Определить command_id экспорта LAS и выполнить тестовый экспорт.
    Запускай когда в NeuraLOG открыт проект с векторизованными кривыми.
    """
    print("\n=== Поиск команды экспорта LAS ===")
    cmd = nl.probe_export_command()
    if cmd is None:
        print("  ✗ Не найдено — открой NDSLOG.EXE в ResourceHacker или запусти parse_menu_resources.py")
        return
    print(f"  ✓ Команда экспорта: CMD={cmd}")
    print(f"\n  Тестовый экспорт → {output_path}")
    ok = nl.export_las(output_path, cmd_id=cmd)
    print(f"  {'✓ Файл создан' if ok else '✗ Файл не создан'}")


def test_export_dialog_probe(nl: NeuraLog):
    """Диагностика: показать все контролы диалога Digital Curve Output."""
    print("\n=== Probe Export Dialog Controls ===")
    print("Откроется диалог — смотри вывод контролов, потом Enter для закрытия")
    nl.probe_export_dialog_controls()


def test_export_new(nl: NeuraLog, output: str = r"F:\nds\output\test_export.las"):
    """
    Тест нового export_las: Ctrl+Shift+F → Digital Curve Output → WM_SETTEXT → OK
    Затем fallback: поиск в /las/ и копирование.
    """
    print(f"\n=== Тест нового export_las ===")
    print(f"  output: {output}")

    import time, os
    t0 = time.time()
    ok = nl.export_las(output, cmd_id=3006, dialog_timeout=10.0)
    elapsed = time.time() - t0

    print(f"  export_las вернул: {ok}  ({elapsed:.1f}с)")
    if ok and os.path.isfile(output):
        size = os.path.getsize(output) / 1024
        print(f"  ✓ Файл создан: {size:.1f} KB")
        # Показать первые строки LAS
        with open(output, encoding="utf-8", errors="replace") as f:
            head = [f.readline() for _ in range(12)]
        print("  --- LAS header (12 lines) ---")
        for line in head:
            print(f"    {line.rstrip()}")
    else:
        print("  ✗ Файл не создан")
        print("  Запусти: python test_buttons.py dialog_probe  — чтобы увидеть контролы")


def test_full_single(nl: NeuraLog,
                     project: str = r"F:\nds\projects\Semeguniv_020\wlg\Semeguniv_20_BK+MBK_3080_3520_200_D1.nlgx",
                     output:  str = r"F:\nds\output\Semeguniv_20_BK+MBK_3080_3520_200_D1.las"):
    """
    Полный тест одного проекта: открыть → трейсинг → сохранить → экспорт LAS.
    Аналог batch_pipeline --one, но с подробным выводом.
    """
    import time, os
    from batch_pipeline import ProjectInfo, BatchPipeline
    print(f"\n=== Full pipeline test (одна скважина) ===")
    print(f"  project: {project}")
    print(f"  output:  {output}")

    proj = ProjectInfo(path=project, name=os.path.splitext(os.path.basename(project))[0])
    proj.output_las = output
    print(f"  depth range: {proj.depth_from} → {proj.depth_to}  DPI={proj.dpi}")

    pipeline = BatchPipeline(nl, export_cmd=3006)
    t0 = time.time()
    result = pipeline.process_one(proj, resume=False)
    elapsed = time.time() - t0

    print(f"\nРезультат: status={result.status}  time={elapsed:.0f}с")
    if result.status == "ok":
        print(f"  ✓ LAS: {result.las_path}  ({result.las_size_kb:.1f} KB)")
    else:
        print(f"  ✗ Error: {result.error}")


def test_wizard_probe(nl: NeuraLog):
    """
    Разведка мастера File → New.
    Перебирает кандидатов CMD ID, фиксирует появившиеся диалоги
    и все их контролы (HWND / class / id / text).
    Запускать один раз для документирования UI мастера.
    """
    print("\n=== wizard_probe: точечная разведка ===")
    print("Ищем истинный ID в диапазоне 1120-1170 с задержкой 0.5 сек.")
    input("Нажмите Enter для запуска...")
    nl.probe_wizard_refined(1120, 1170)


def test_new_project(nl: NeuraLog,
                     image_path: str  = r"F:\nds\projects\Semeguniv_020\img\Semeguniv_20_BK+MBK_3080_3520_200_D1.tif",
                     project_path: str = r"F:\nds\projects\Semeguniv_020\wlg\test_new.nlgx"):
    """
    Тест создания нового проекта из изображения через File → New wizard.
    Запускать ПОСЛЕ wizard_probe (когда известны шаги мастера).
    """
    import os
    print(f"\n=== test_new_project ===")
    print(f"  image:   {image_path}")
    print(f"  project: {project_path}")

    if not os.path.isfile(image_path):
        print(f"  ✗ Изображение не найдено: {image_path}")
        return

    t0 = time.time()
    ok = nl.new_project_from_image(image_path, project_path)
    elapsed = time.time() - t0

    print(f"\n  new_project_from_image → {ok}  ({elapsed:.1f}с)")
    if ok:
        title = nl.get_title()
        depth = nl.get_current_depth()
        print(f"  Заголовок: {title}")
        print(f"  Глубина:   {depth}")
    else:
        print("  Проверь: правильный ли cmd_id в NEW_PROJECT_CMD_CANDIDATES?")
        print("  Запусти wizard_probe для уточнения.")


def test_analyze_image(image_path: str = r"F:\nds\projects\Semeguniv_020\img\Semeguniv_20_BK+MBK_3080_3520_200_D1.tif"):
    """
    Тест локального анализа изображения каротажа.
    Не требует запущенного NeuraLOG — работает автономно.
    """
    import os, json
    print(f"\n=== test_analyze_image ===")
    print(f"  file: {image_path}")

    if not os.path.isfile(image_path):
        print(f"  ✗ Файл не найден: {image_path}")
        return

    try:
        from analyze_log_image import analyze_log_image_cached
    except ImportError:
        print("  ✗ analyze_log_image.py не найден — скопируй рядом с test_buttons.py")
        return

    t0 = time.time()
    result = analyze_log_image_cached(image_path, force_refresh=True)
    elapsed = time.time() - t0

    print(f"\n  Источник:  {result.get('source')}")
    print(f"  Глубины:   {result.get('depth_from')} → {result.get('depth_to')} {result.get('depth_unit','m')}")
    print(f"  Треков:    {result.get('num_tracks')}")
    print(f"  Кривых:    {len(result.get('curves', []))}")
    print(f"  Время:     {elapsed:.1f}с")

    for c in result.get("curves", []):
        print(f"    {c.get('name','?'):8s} {c.get('unit','?'):8s} "
              f"{c.get('scale_min','?')}..{c.get('scale_max','?')}  "
              f"{c.get('color','')}")

    print(f"\n  JSON:\n{json.dumps(result, ensure_ascii=False, indent=4)}")


def main():
    modes_help = (
        "state | trace | modes | menu | status | export | export2 | "
        "dialog_probe | wizard_probe | new_project | analyze | full | all"
    )
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    # Режимы без NeuraLOG
    if mode == "analyze":
        image = sys.argv[2] if len(sys.argv) > 2 else r"F:\nds\projects\Semeguniv_020\img\Semeguniv_20_BK+MBK_3080_3520_200_D1.tif"
        test_analyze_image(image)
        return

    nl = NeuraLog()
    if not nl.cm.main_window:
        print("NeuraLOG не найден. Запустите NDSLOG.EXE.")
        sys.exit(1)

    nl.ensure_visible()
    time.sleep(0.5)

    if mode in ("all", "state"):
        test_state(nl)
    if mode in ("all", "trace"):
        test_trace(nl)
    if mode == "modes":
        test_trace_modes(nl)
    if mode in ("all", "menu"):
        test_menu(nl)
    if mode in ("all", "status"):
        test_status(nl)
    if mode == "export":
        output = sys.argv[2] if len(sys.argv) > 2 else r"F:\nds\output\test_export.las"
        test_export(nl, output)
    if mode == "export2":
        output = sys.argv[2] if len(sys.argv) > 2 else r"F:\nds\output\test_export.las"
        test_export_new(nl, output)
    if mode == "dialog_probe":
        test_export_dialog_probe(nl)
    if mode == "wizard_probe":
        test_wizard_probe(nl)
    if mode == "new_project":
        image   = sys.argv[2] if len(sys.argv) > 2 else r"F:\nds\projects\Semeguniv_020\img\Semeguniv_20_BK+MBK_3080_3520_200_D1.tif"
        project = sys.argv[3] if len(sys.argv) > 3 else r"F:\nds\projects\Semeguniv_020\wlg\test_new.nlgx"
        test_new_project(nl, image, project)
    if mode == "full":
        project = sys.argv[2] if len(sys.argv) > 2 else r"F:\nds\projects\Semeguniv_020\wlg\Semeguniv_20_BK+MBK_3080_3520_200_D1.nlgx"
        output  = sys.argv[3] if len(sys.argv) > 3 else r"F:\nds\output\Semeguniv_20_BK+MBK_3080_3520_200_D1.las"
        test_full_single(nl, project, output)

    if mode not in (
        "all", "state", "trace", "modes", "menu", "status",
        "export", "export2", "dialog_probe", "wizard_probe",
        "new_project", "analyze", "full",
    ):
        print(f"Неизвестный режим '{mode}'.")
        print(f"Доступные: {modes_help}")


if __name__ == "__main__":
    main()

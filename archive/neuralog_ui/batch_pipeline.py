"""
batch_pipeline.py — Батч-оркестратор NeuraLOG v2015
====================================================
Открывает .nlgx проекты → трейсинг → Save → Export LAS.
Поддерживает resume (пропуск уже готовых), CSV-отчёт, state.json checkpoint.

Зависимость: neuralog_win32.py (должен быть в той же папке)

Использование (CLI):
    python batch_pipeline.py --list                          # список проектов
    python batch_pipeline.py --one F:\\nds\\...\\file.nlgx  # один файл
    python batch_pipeline.py --dir F:\\nds\\projects         # весь батч
    python batch_pipeline.py --probe-export                  # найти CMD экспорта
    python batch_pipeline.py --check-sliders                 # состояние слайдеров

Использование (из кода):
    from batch_pipeline import ProjectInfo, BatchPipeline
    from neuralog_win32 import NeuraLog

    nl       = NeuraLog()
    pipeline = BatchPipeline(nl, export_cmd=3006)
    proj     = ProjectInfo(path=r"F:\\nds\\...\\file.nlgx", name="file")
    result   = pipeline.process_one(proj, resume=False)
    print(result.status, result.las_path)
"""

import os
import sys
import re
import csv
import json
import time
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────────────────────────

CFG = {
    "output_dir":       r"F:\nds\output",
    "report_dir":       r"F:\nds\logs",
    "state_file":       r"F:\nds\logs\batch_state.json",
    "export_cmd_file":  r"F:\nds\Auto\export_cmd.json",
    "default_export_cmd": 3006,
    # Трейсер ~0.20 м/с — для динамического таймаута
    "trace_speed_mps":  0.20,
    # Минимальный таймаут трейсинга (30 минут)
    "min_trace_timeout": 1800,
}

# ─────────────────────────────────────────────────────────────────
# МОДЕЛИ ДАННЫХ
# ─────────────────────────────────────────────────────────────────

@dataclass
class ProjectInfo:
    """Описание одного .nlgx проекта для обработки."""
    path:       str            # полный путь к .nlgx
    name:       str            # имя без расширения (для логов и state)
    depth_from: float = 0.0   # начало диапазона глубин (м)
    depth_to:   float = 0.0   # конец диапазона глубин (м)
    dpi:        int   = 200   # DPI сканирования
    part:       str   = "D1"  # часть изображения (D1, D2...)
    output_las: str   = ""    # куда сохранять LAS

    def __post_init__(self):
        self._parse_filename()
        if not self.output_las:
            self.output_las = os.path.join(CFG["output_dir"], self.name + ".las")

    def _parse_filename(self):
        """
        Разобрать имя файла: Well_A_BK+MBK_3080_3520_200_D1.nlgx
        → depth_from=3080, depth_to=3520, dpi=200, part="D1"
        Паттерн: три числа подряд = from, to, dpi.
        """
        m = re.search(r"_(\d{3,5})_(\d{3,5})_(\d{2,4})", self.name)
        if m:
            self.depth_from = float(m.group(1))
            self.depth_to   = float(m.group(2))
            self.dpi        = int(m.group(3))
        pm = re.search(r"_(D\d+)(?:_|$)", self.name, re.IGNORECASE)
        if pm:
            self.part = pm.group(1).upper()

    @property
    def depth_range(self) -> float:
        return abs(self.depth_to - self.depth_from)

    def __str__(self) -> str:
        return (f"{self.name}  [{self.depth_from}→{self.depth_to}м  "
                f"Δ={self.depth_range:.0f}м  DPI={self.dpi}]")


@dataclass
class ProcessResult:
    """Результат обработки одного проекта."""
    project:     str
    status:      str   = "pending"  # pending | ok | error | skipped
    las_path:    str   = ""
    las_size_kb: float = 0.0
    error:       str   = ""
    start_time:  float = field(default_factory=time.time)
    end_time:    float = 0.0
    depth_start: float = 0.0
    depth_end:   float = 0.0

    @property
    def elapsed(self) -> float:
        t = self.end_time or time.time()
        return t - self.start_time

    def finish_ok(self, las_path: str):
        self.status   = "ok"
        self.las_path = las_path
        self.end_time = time.time()
        if os.path.isfile(las_path):
            self.las_size_kb = os.path.getsize(las_path) / 1024

    def finish_error(self, error: str):
        self.status   = "error"
        self.error    = error
        self.end_time = time.time()

    def skip(self, reason: str = "already done"):
        self.status   = "skipped"
        self.error    = reason
        self.end_time = time.time()


# ─────────────────────────────────────────────────────────────────
# БАТЧ-ПАЙПЛАЙН
# ─────────────────────────────────────────────────────────────────

class BatchPipeline:
    """Оркестратор батчевой обработки .nlgx проектов."""

    def __init__(self, nl, export_cmd: Optional[int] = None):
        """
        nl          : экземпляр NeuraLog из neuralog_win32.py
        export_cmd  : CMD ID для экспорта LAS (из probe или export_cmd.json)
        """
        self.nl         = nl
        self.export_cmd = export_cmd or self._load_export_cmd()
        self.state      = self._load_state()

    # ─────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────

    def _load_export_cmd(self) -> int:
        try:
            with open(CFG["export_cmd_file"]) as f:
                return json.load(f).get("cmd_id", CFG["default_export_cmd"])
        except Exception:
            return CFG["default_export_cmd"]

    def _save_export_cmd(self, cmd_id: int):
        os.makedirs(os.path.dirname(CFG["export_cmd_file"]), exist_ok=True)
        with open(CFG["export_cmd_file"], "w") as f:
            json.dump({"cmd_id": cmd_id, "saved": datetime.now().isoformat()}, f)

    def _load_state(self) -> dict:
        try:
            with open(CFG["state_file"]) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_state(self):
        os.makedirs(os.path.dirname(CFG["state_file"]), exist_ok=True)
        with open(CFG["state_file"], "w") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    # ─────────────────────────────────────────────────────────────
    # ПОИСК И ФИЛЬТРАЦИЯ ПРОЕКТОВ
    # ─────────────────────────────────────────────────────────────

    def find_projects(self, directory: str) -> List[ProjectInfo]:
        """Рекурсивно найти все .nlgx в директории."""
        projects = []
        for p in sorted(Path(directory).rglob("*.nlgx")):
            projects.append(ProjectInfo(path=str(p), name=p.stem))
        log.info(f"Найдено {len(projects)} проектов в {directory}")
        return projects

    def filter_pending(self, projects: List[ProjectInfo]) -> List[ProjectInfo]:
        """Вернуть только проекты без LAS-файла и без записи status=ok в state."""
        pending = []
        for proj in projects:
            if os.path.isfile(proj.output_las):
                log.debug(f"  skip (LAS exists): {proj.name}")
            elif self.state.get(proj.name, {}).get("status") == "ok":
                log.debug(f"  skip (state=ok):   {proj.name}")
            else:
                pending.append(proj)
        return pending

    # ─────────────────────────────────────────────────────────────
    # ОБРАБОТКА ОДНОГО ПРОЕКТА
    # ─────────────────────────────────────────────────────────────

    def process_one(self, proj: ProjectInfo, resume: bool = True) -> ProcessResult:
        """
        Полный цикл для одного .nlgx:
          [01] open_project
          [02] switch_to_project (MDI focus)
          [03] rebuild_control_map
          [04] trace_to_depth
          [05] save
          [06] export_las
        """
        result = ProcessResult(project=proj.path)
        sep = "─" * 58
        log.info(f"\n{sep}")
        log.info(f"Проект: {proj}")
        log.info(sep)

        # ── Resume check ──────────────────────────────────────────
        if resume and os.path.isfile(proj.output_las):
            result.skip("LAS уже существует")
            log.info(f"  ⏭ {proj.output_las}")
            return result

        try:
            # ── [01] Открыть проект ───────────────────────────────
            log.info("  [01] open_project...")
            if not self.nl.open_project(proj.path):
                result.finish_error("open_project failed")
                return result
            time.sleep(1.0)

            # ── [02] MDI focus ────────────────────────────────────
            log.info("  [02] switch_to_project...")
            self.nl.switch_to_project(proj.path)
            time.sleep(0.5)

            # ── [03] Пересобрать ControlMap ───────────────────────
            # HWND контролов могут измениться после открытия нового документа
            log.info("  [03] rebuild ControlMap...")
            self.nl._build_control_map()

            # ── [04] Трейсинг ─────────────────────────────────────
            result.depth_start = self.nl.get_current_depth() or proj.depth_from
            log.info(f"  [04] trace  {result.depth_start} → {proj.depth_to} м...")
            ok = self._run_trace(proj)
            if not ok:
                self.nl.stop_trace()
                result.finish_error("trace_to_depth timeout/failed")
                return result
            result.depth_end = self.nl.get_current_depth() or proj.depth_to

            # ── [05] Сохранить проект ─────────────────────────────
            log.info("  [05] save...")
            self.nl.save()
            time.sleep(0.5)

            # ── [06] Экспорт LAS ──────────────────────────────────
            log.info(f"  [06] export_las → {proj.output_las}...")
            os.makedirs(os.path.dirname(proj.output_las), exist_ok=True)
            ok = self.nl.export_las(proj.output_las, cmd_id=self.export_cmd)

            if ok and os.path.isfile(proj.output_las):
                result.finish_ok(proj.output_las)
                log.info(f"  ✓ {result.las_size_kb:.1f} KB  {result.elapsed:.0f}с")
            else:
                result.finish_error(
                    f"export_las failed (cmd={self.export_cmd}, "
                    f"file={'exists' if os.path.isfile(proj.output_las) else 'missing'})"
                )

        except Exception as exc:
            import traceback
            result.finish_error(str(exc))
            log.error(f"  ✗ Исключение: {exc}")
            log.debug(traceback.format_exc())

        # Обновить checkpoint
        self.state[proj.name] = {
            "status": result.status,
            "las":    result.las_path,
            "error":  result.error,
            "time":   datetime.now().isoformat(),
        }
        self._save_state()

        return result

    def _run_trace(self, proj: ProjectInfo) -> bool:
        """
        Трейсинг с динамическим таймаутом.
        Скорость ~0.20 м/с (эмпирически из тестов).
        Таймаут = max(1800с, оставшийся_диапазон / скорость).
        """
        current = self.nl.get_current_depth()
        if current is None:
            log.warning("    Глубина недоступна, берём depth_from")
            current = proj.depth_from

        remaining = abs(proj.depth_to - current)
        timeout   = max(
            CFG["min_trace_timeout"],
            remaining / CFG["trace_speed_mps"],
        )
        log.info(
            f"    depth={current:.1f}  target={proj.depth_to:.1f}  "
            f"Δ={remaining:.0f}м  timeout={timeout:.0f}с "
            f"({timeout/60:.0f}мин)"
        )
        return self.nl.trace_to_depth(proj.depth_to, timeout=timeout)

    # ─────────────────────────────────────────────────────────────
    # ПОЛНЫЙ БАТЧ
    # ─────────────────────────────────────────────────────────────

    def run_all(
        self,
        projects: List[ProjectInfo],
        resume: bool = True,
    ) -> List[ProcessResult]:
        """Обработать все проекты последовательно. Записать CSV-отчёт."""
        results = []
        total   = len(projects)
        ok_cnt = err_cnt = skip_cnt = 0

        log.info(f"Батч: {total} проектов  resume={resume}")

        for i, proj in enumerate(projects, 1):
            log.info(f"\n[{i}/{total}] {proj.name}")
            result = self.process_one(proj, resume=resume)
            results.append(result)

            if result.status == "ok":
                ok_cnt += 1
            elif result.status == "skipped":
                skip_cnt += 1
            else:
                err_cnt += 1

            log.info(
                f"  Итог: ✓{ok_cnt}  ✗{err_cnt}  ⏭{skip_cnt}  "
                f"осталось {total - i}"
            )

        self._save_report(results)
        log.info(f"\n{'='*58}")
        log.info(f"БАТЧ ГОТОВ: ok={ok_cnt}  err={err_cnt}  skip={skip_cnt}")
        return results

    def _save_report(self, results: List[ProcessResult]):
        """Сохранить CSV-отчёт батча."""
        os.makedirs(CFG["report_dir"], exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CFG["report_dir"], f"batch_{ts}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "project", "status", "las_size_kb", "elapsed_sec",
                "depth_start", "depth_end", "error",
            ])
            for r in results:
                w.writerow([
                    r.project, r.status,
                    round(r.las_size_kb, 1),
                    round(r.elapsed, 1),
                    r.depth_start, r.depth_end, r.error,
                ])
        log.info(f"Отчёт: {path}")

    # ─────────────────────────────────────────────────────────────
    # УТИЛИТЫ
    # ─────────────────────────────────────────────────────────────

    def check_sliders(self):
        """Показать диапазон и текущее положение слайдеров (диагностика)."""
        import win32gui
        TBM_GETRANGEMIN = 0x0401
        TBM_GETRANGEMAX = 0x0402
        TBM_GETPOS      = 0x0400

        cm = self.nl.cm
        for label, hwnd in [
            ("speed_slider",      cm.slider_tracer_speed),
            ("depth_pos_slider",  cm.slider_depth_position),
        ]:
            if not hwnd:
                print(f"  {label:20s}: HWND не найден")
                continue
            lo  = win32gui.SendMessage(hwnd, TBM_GETRANGEMIN, 0, 0)
            hi  = win32gui.SendMessage(hwnd, TBM_GETRANGEMAX, 0, 0)
            cur = win32gui.SendMessage(hwnd, TBM_GETPOS, 0, 0)
            print(f"  {label:20s}: min={lo:6d}  max={hi:6d}  cur={cur:6d}")

    def probe_export(self) -> Optional[int]:
        """Определить CMD ID экспорта LAS и сохранить в export_cmd.json."""
        cmd = self.nl.probe_export_command()
        if cmd:
            self._save_export_cmd(cmd)
            log.info(f"CMD экспорта: {cmd}  → {CFG['export_cmd_file']}")
        else:
            log.warning("CMD не найден — используй ResourceHacker или probe вручную")
        return cmd


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def _setup_logging():
    os.makedirs(CFG["report_dir"], exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(CFG["report_dir"], "batch.log"),
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
    )


def main():
    _setup_logging()

    parser = argparse.ArgumentParser(description="NeuraLOG Batch Pipeline")
    parser.add_argument("--dir",  metavar="DIR",  help="Директория с .nlgx")
    parser.add_argument("--one",  metavar="PATH", help="Один .nlgx файл")
    parser.add_argument("--list", action="store_true",
                        help="Показать проекты без обработки")
    parser.add_argument("--resume",    dest="resume", default=True,
                        action="store_true",  help="Пропускать готовые (default)")
    parser.add_argument("--no-resume", dest="resume",
                        action="store_false", help="Переделать все, даже готовые")
    parser.add_argument("--check-sliders", action="store_true")
    parser.add_argument("--probe-export",  action="store_true")
    parser.add_argument("--export-cmd", type=int,
                        help="CMD ID экспорта (переопределить auto-detect)")
    args = parser.parse_args()

    from neuralog_win32 import NeuraLog
    nl = NeuraLog()
    if not nl.cm.main_window:
        print("NeuraLOG не найден. Запустите NDSLOG.EXE.")
        sys.exit(1)
    nl.ensure_visible()

    pipeline = BatchPipeline(nl, export_cmd=args.export_cmd)

    # ── Диагностика ───────────────────────────────────────────────
    if args.check_sliders:
        print("\n=== Слайдеры ===")
        pipeline.check_sliders()
        nl.print_state()
        return

    if args.probe_export:
        print("\n=== Поиск CMD ID экспорта ===")
        cmd = pipeline.probe_export()
        print(f"Результат: CMD={cmd}")
        return

    # ── Один файл ─────────────────────────────────────────────────
    if args.one:
        p    = args.one
        name = os.path.splitext(os.path.basename(p))[0]
        proj = ProjectInfo(path=p, name=name)
        print(f"Один проект: {proj}")
        result = pipeline.process_one(proj, resume=args.resume)
        print(f"\nРезультат: {result.status}  ({result.elapsed:.0f}с)")
        if result.status == "ok":
            print(f"  LAS: {result.las_path}  ({result.las_size_kb:.1f} KB)")
        elif result.error:
            print(f"  Ошибка: {result.error}")
        return

    # ── Батч ──────────────────────────────────────────────────────
    if not args.dir:
        parser.print_help()
        return

    projects = pipeline.find_projects(args.dir)
    if args.list:
        print(f"\nНайдено {len(projects)} проектов в {args.dir}:\n")
        for proj in projects:
            las_ok = "✓ LAS" if os.path.isfile(proj.output_las) else "     "
            st     = pipeline.state.get(proj.name, {}).get("status", "—")
            print(f"  [{las_ok}] [{st:7s}]  {proj}")
        return

    pending = pipeline.filter_pending(projects) if args.resume else projects
    print(f"\nВсего: {len(projects)},  к обработке: {len(pending)}")

    if not pending:
        print("Нечего делать — все проекты уже обработаны.")
        return

    results = pipeline.run_all(pending, resume=args.resume)
    ok  = sum(1 for r in results if r.status == "ok")
    err = sum(1 for r in results if r.status == "error")
    print(f"\nГотово: {ok}/{len(results)} успешно, {err} ошибок")


if __name__ == "__main__":
    main()

"""
neuralog_agent.py — ИИ-агент для автоматизации NeuraLOG v2015
=============================================================
Архитектура: Claude API (tool_use) + pywinauto/win32gui + файловая система

Агент получает задачу → планирует → вызывает инструменты → проверяет результат

Установка:
    pip install anthropic pywinauto pywin32 pillow pyautogui opencv-python

Запуск:
    python neuralog_agent.py
"""

import os
import sys
import json
import time
import base64
import glob
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import anthropic
import pyautogui
import win32gui
import win32con
import win32api
from PIL import Image, ImageGrab
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────
#  КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────────────────────────
CFG = {
    "projects_dir":    r"F:\nds\projects",
    "images_dir":      r"F:\nds\images",          # исходные изображения
    "output_dir":      r"F:\nds\output",           # экспорт результатов
    "screenshots_dir": r"F:\nds\ScreensAutoGUI",
    "log_file":        r"F:\nds\agent.log",
    "neuralog_exe":    r"C:\Program Files\NeuraLOG\NeuraLOG.exe",  # путь к exe
    "neuralog_window_re": r".*[Nn]eura[Ll][Oo][Gg].*",
    "anthropic_model": "claude-opus-4-6",
    "max_iterations":  30,   # максимум шагов агента на задачу
    "screenshot_w":    1920,
    "screenshot_h":    1080,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CFG["log_file"], encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
#  ИНСТРУМЕНТЫ АГЕНТА  (каждый — одна Python-функция + JSON-схема)
# ─────────────────────────────────────────────────────────────────

def tool_take_screenshot(region: Optional[dict] = None) -> dict:
    """Снять скриншот экрана (или области). Вернуть base64."""
    try:
        if region:
            img = ImageGrab.grab(bbox=(
                region["x"], region["y"],
                region["x"] + region["w"],
                region["y"] + region["h"]
            ))
        else:
            img = ImageGrab.grab()
        
        # Сохранить для отладки
        ts = datetime.now().strftime("%H%M%S_%f")[:10]
        path = os.path.join(CFG["screenshots_dir"], f"agent_{ts}.png")
        img.save(path)

        # Конвертировать в base64
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"ok": True, "base64": b64, "path": path,
                "width": img.width, "height": img.height}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_list_files(directory: str, pattern: str = "*") -> dict:
    """Список файлов в директории по паттерну."""
    try:
        path = Path(directory)
        if not path.exists():
            return {"ok": False, "error": f"Директория не найдена: {directory}"}
        files = []
        for f in sorted(path.glob(pattern)):
            stat = f.stat()
            files.append({
                "name":     f.name,
                "path":     str(f),
                "size_kb":  round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "is_dir":   f.is_dir()
            })
        return {"ok": True, "directory": str(directory),
                "pattern": pattern, "count": len(files), "files": files}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_get_neuralog_windows() -> dict:
    """Найти все окна NeuraLOG и их состояние."""
    import re
    windows = []

    def callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if re.search(r"[Nn]eura[Ll][Oo][Gg]", title):
            rect = win32gui.GetWindowRect(hwnd)
            cls  = win32gui.GetClassName(hwnd)
            windows.append({
                "hwnd":  hwnd,
                "hwnd_hex": f"0x{hwnd:08X}",
                "title": title,
                "class": cls,
                "rect":  {"x": rect[0], "y": rect[1],
                           "w": rect[2]-rect[0], "h": rect[3]-rect[1]},
                "is_active": hwnd == win32gui.GetForegroundWindow()
            })
        return True

    win32gui.EnumWindows(callback, None)
    return {"ok": True, "count": len(windows), "windows": windows}


def tool_focus_neuralog() -> dict:
    """Перевести фокус на главное окно NeuraLOG."""
    result = tool_get_neuralog_windows()
    if not result["ok"] or result["count"] == 0:
        return {"ok": False, "error": "NeuraLOG не найден"}
    hwnd = result["windows"][0]["hwnd"]
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.3)
        return {"ok": True, "hwnd": hwnd, "title": result["windows"][0]["title"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_click(x: int, y: int, button: str = "left", double: bool = False) -> dict:
    """Клик мышью по координатам."""
    try:
        tool_focus_neuralog()
        pyautogui.moveTo(x, y, duration=0.15)
        if double:
            pyautogui.doubleClick(x, y)
        else:
            pyautogui.click(x, y, button=button)
        time.sleep(0.2)
        return {"ok": True, "x": x, "y": y, "button": button, "double": double}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_type_text(text: str, with_enter: bool = False) -> dict:
    """Ввести текст с клавиатуры (в активное поле)."""
    try:
        pyautogui.typewrite(text, interval=0.04)
        if with_enter:
            pyautogui.press("enter")
        time.sleep(0.2)
        return {"ok": True, "text": text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_hotkey(keys: list) -> dict:
    """Нажать горячую клавишу. Пример: ['ctrl', 's'] или ['alt', 'f', 'o']"""
    try:
        tool_focus_neuralog()
        pyautogui.hotkey(*keys)
        time.sleep(0.4)
        return {"ok": True, "keys": keys}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_send_wm_command(hwnd: int, command_id: int) -> dict:
    """Отправить WM_COMMAND напрямую (без захвата фокуса)."""
    try:
        win32api.SendMessage(hwnd, win32con.WM_COMMAND, command_id, 0)
        time.sleep(0.3)
        return {"ok": True, "hwnd": f"0x{hwnd:08X}", "command_id": command_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_find_template(template_path: str, threshold: float = 0.8) -> dict:
    """Найти UI-элемент по шаблону (OpenCV). Вернуть координаты центра."""
    try:
        screenshot = np.array(ImageGrab.grab())
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            return {"ok": False, "error": f"Шаблон не найден: {template_path}"}
        result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < threshold:
            return {"ok": False, "found": False,
                    "confidence": round(max_val, 3),
                    "error": f"Шаблон не найден (confidence={max_val:.3f} < {threshold})"}
        h, w = template.shape
        cx = max_loc[0] + w // 2
        cy = max_loc[1] + h // 2
        return {"ok": True, "found": True, "x": cx, "y": cy,
                "confidence": round(max_val, 3)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_wait_for_template(template_path: str, timeout: float = 15.0,
                           threshold: float = 0.8, interval: float = 0.5) -> dict:
    """Ждать появления UI-элемента (шаблон). Полезно для диалогов."""
    start = time.time()
    while time.time() - start < timeout:
        r = tool_find_template(template_path, threshold)
        if r.get("found"):
            return r
        time.sleep(interval)
    return {"ok": False, "found": False,
            "error": f"Таймаут {timeout}с: {template_path}"}


def tool_open_project(project_path: str) -> dict:
    """Открыть .nlgx проект в NeuraLOG через File > Open."""
    try:
        # Ctrl+O
        tool_hotkey(["ctrl", "o"])
        time.sleep(1.0)

        # Ввести путь в диалог открытия файла
        pyautogui.hotkey("ctrl", "a")
        pyautogui.typewrite(project_path, interval=0.03)
        pyautogui.press("enter")
        time.sleep(2.0)  # NeuraLOG загружает проект

        return {"ok": True, "path": project_path}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_save_project(project_path: Optional[str] = None) -> dict:
    """Сохранить текущий проект (Ctrl+S или Save As)."""
    try:
        if project_path:
            # Save As
            pyautogui.hotkey("ctrl", "shift", "s")
            time.sleep(0.8)
            pyautogui.hotkey("ctrl", "a")
            pyautogui.typewrite(project_path, interval=0.03)
            pyautogui.press("enter")
        else:
            # Save
            tool_hotkey(["ctrl", "s"])
        time.sleep(1.0)
        return {"ok": True, "path": project_path or "current"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_export_result(output_path: str, format: str = "las") -> dict:
    """Экспортировать результат векторизации."""
    try:
        # Зависит от меню NeuraLOG — настроить под реальный UI
        # Пример: File > Export > LAS
        pyautogui.hotkey("alt", "f")
        time.sleep(0.4)
        # Дальше — навигация по меню (будет уточнено после uia_probe)
        return {"ok": True, "output_path": output_path, "format": format,
                "note": "Требует настройки под реальное меню NeuraLOG"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_copy_file(src: str, dst: str) -> dict:
    """Скопировать файл."""
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return {"ok": True, "src": src, "dst": dst}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_read_text_file(path: str) -> dict:
    """Прочитать текстовый файл (лог, csv, las-header)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read(10000)  # первые 10 КБ
        return {"ok": True, "path": path, "content": content,
                "truncated": len(content) == 10000}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_report_status(message: str, level: str = "info") -> dict:
    """Записать статус в лог (агент сообщает о прогрессе)."""
    getattr(log, level, log.info)(f"[AGENT] {message}")
    return {"ok": True, "logged": message}


# ─────────────────────────────────────────────────────────────────
#  РЕЕСТР ИНСТРУМЕНТОВ (схемы для Claude API)
# ─────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "take_screenshot",
        "description": "Снять скриншот экрана или области. Используй для проверки текущего состояния UI.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "object",
                    "description": "Опционально: область {x,y,w,h}. Без — весь экран.",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"},
                                   "w": {"type": "integer"}, "h": {"type": "integer"}},
                }
            }
        }
    },
    {
        "name": "list_files",
        "description": "Список файлов в директории. Используй для поиска изображений, проектов, LAS-файлов.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Путь к директории"},
                "pattern":   {"type": "string", "description": "Glob-паттерн, напр. *.nlgx, *.png, *.las"}
            },
            "required": ["directory"]
        }
    },
    {
        "name": "get_neuralog_windows",
        "description": "Найти открытые окна NeuraLOG, их HWND и координаты.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "focus_neuralog",
        "description": "Перевести фокус на главное окно NeuraLOG перед UI-действиями.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "click",
        "description": "Кликнуть мышью по координатам экрана.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x":      {"type": "integer"},
                "y":      {"type": "integer"},
                "button": {"type": "string", "enum": ["left", "right", "middle"]},
                "double": {"type": "boolean", "description": "Двойной клик"}
            },
            "required": ["x", "y"]
        }
    },
    {
        "name": "type_text",
        "description": "Ввести текст в активное поле ввода.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text":       {"type": "string"},
                "with_enter": {"type": "boolean"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "hotkey",
        "description": "Нажать комбинацию клавиш. Примеры: ['ctrl','s'], ['ctrl','o'], ['alt','f4']",
        "input_schema": {
            "type": "object",
            "properties": {
                "keys": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["keys"]
        }
    },
    {
        "name": "find_template",
        "description": "Найти UI-элемент на экране по PNG-шаблону. Вернуть координаты центра.",
        "input_schema": {
            "type": "object",
            "properties": {
                "template_path": {"type": "string", "description": "Путь к PNG-шаблону"},
                "threshold":     {"type": "number", "description": "Порог совпадения 0..1 (по умолчанию 0.8)"}
            },
            "required": ["template_path"]
        }
    },
    {
        "name": "wait_for_template",
        "description": "Ждать появления UI-элемента (диалог, кнопка) по шаблону. Для синхронизации.",
        "input_schema": {
            "type": "object",
            "properties": {
                "template_path": {"type": "string"},
                "timeout":       {"type": "number", "description": "Максимальное ожидание в секундах"},
                "threshold":     {"type": "number"}
            },
            "required": ["template_path"]
        }
    },
    {
        "name": "open_project",
        "description": "Открыть .nlgx проект в NeuraLOG по пути к файлу.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_path": {"type": "string", "description": "Полный путь к .nlgx файлу"}
            },
            "required": ["project_path"]
        }
    },
    {
        "name": "save_project",
        "description": "Сохранить текущий проект (Ctrl+S) или в новый файл.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_path": {"type": "string", "description": "Если указан — Save As"}
            }
        }
    },
    {
        "name": "export_result",
        "description": "Экспортировать результат векторизации в LAS или другой формат.",
        "input_schema": {
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "format":      {"type": "string", "enum": ["las", "csv", "dlis", "tiff"]}
            },
            "required": ["output_path"]
        }
    },
    {
        "name": "copy_file",
        "description": "Скопировать файл из src в dst.",
        "input_schema": {
            "type": "object",
            "properties": {
                "src": {"type": "string"},
                "dst": {"type": "string"}
            },
            "required": ["src", "dst"]
        }
    },
    {
        "name": "read_text_file",
        "description": "Прочитать содержимое текстового файла (лог, LAS-заголовок, CSV).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "report_status",
        "description": "Записать статусное сообщение в лог агента. Используй для отчётности о прогрессе.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "level":   {"type": "string", "enum": ["info", "warning", "error"]}
            },
            "required": ["message"]
        }
    },
]

# Диспетчер: имя инструмента → функция
TOOL_DISPATCH = {
    "take_screenshot":    lambda p: tool_take_screenshot(p.get("region")),
    "list_files":         lambda p: tool_list_files(p["directory"], p.get("pattern", "*")),
    "get_neuralog_windows": lambda p: tool_get_neuralog_windows(),
    "focus_neuralog":     lambda p: tool_focus_neuralog(),
    "click":              lambda p: tool_click(p["x"], p["y"],
                                               p.get("button", "left"), p.get("double", False)),
    "type_text":          lambda p: tool_type_text(p["text"], p.get("with_enter", False)),
    "hotkey":             lambda p: tool_hotkey(p["keys"]),
    "find_template":      lambda p: tool_find_template(p["template_path"], p.get("threshold", 0.8)),
    "wait_for_template":  lambda p: tool_wait_for_template(p["template_path"],
                                                            p.get("timeout", 15.0),
                                                            p.get("threshold", 0.8)),
    "open_project":       lambda p: tool_open_project(p["project_path"]),
    "save_project":       lambda p: tool_save_project(p.get("project_path")),
    "export_result":      lambda p: tool_export_result(p["output_path"], p.get("format", "las")),
    "copy_file":          lambda p: tool_copy_file(p["src"], p["dst"]),
    "read_text_file":     lambda p: tool_read_text_file(p["path"]),
    "report_status":      lambda p: tool_report_status(p["message"], p.get("level", "info")),
}


# ─────────────────────────────────────────────────────────────────
#  СИСТЕМНЫЙ ПРОМПТ АГЕНТА
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
Ты — автоматизационный агент для работы с программой NeuraLOG v2015 (Windows desktop).
NeuraLOG — геофизическая программа для векторизации изображений каротажных кривых (well logs).
Проектные файлы: .nlgx (бинарный C++ формат). Экспорт: LAS 2.0, CSV.

Рабочие директории:
- Изображения: F:\\nds\\images
- Проекты:     F:\\nds\\projects  
- Вывод:       F:\\nds\\output
- Скриншоты:   F:\\nds\\ScreensAutoGUI

Ключевые правила:
1. ВСЕГДА делай take_screenshot перед UI-действием, если не уверен в текущем состоянии экрана
2. ВСЕГДА вызывай focus_neuralog перед click/hotkey
3. После каждого важного шага — take_screenshot для подтверждения результата
4. При ошибке — опиши что пошло не так и попробуй альтернативный подход
5. Используй find_template для надёжного поиска кнопок вместо жёстких координат
6. report_status после каждого завершённого этапа задачи

Типовая последовательность для обработки одного изображения:
1. list_files → найти изображение и шаблон проекта
2. focus_neuralog + open_project
3. take_screenshot → убедиться что проект открылся
4. Выполнить векторизацию (через меню или горячие клавиши)
5. save_project
6. export_result → LAS файл в output
7. report_status о завершении

Ты работаешь автономно, но всегда проверяй результат каждого действия через скриншот или ответ инструмента.
При неопределённости — снимай скриншот и анализируй.
"""


# ─────────────────────────────────────────────────────────────────
#  ЦИКЛ АГЕНТА
# ─────────────────────────────────────────────────────────────────

class NeuraLogAgent:
    def __init__(self):
        self.client = anthropic.Anthropic()  # читает ANTHROPIC_API_KEY из env
        self.history = []
        self.iteration = 0

    def _execute_tool(self, name: str, params: dict) -> str:
        """Выполнить инструмент и вернуть JSON-строку результата."""
        log.info(f"TOOL CALL: {name}({json.dumps(params, ensure_ascii=False)[:200]})")
        fn = TOOL_DISPATCH.get(name)
        if fn is None:
            result = {"ok": False, "error": f"Неизвестный инструмент: {name}"}
        else:
            try:
                result = fn(params)
            except Exception as e:
                result = {"ok": False, "error": str(e)}
        
        log.info(f"TOOL RESULT: {json.dumps(result, ensure_ascii=False)[:300]}")
        return json.dumps(result, ensure_ascii=False)

    def _build_tool_result_content(self, tool_use_block, result_str: str):
        """Подготовить блок tool_result для следующего сообщения."""
        # Если инструмент вернул скриншот — добавляем изображение в content
        result = json.loads(result_str)
        if "base64" in result and result["base64"]:
            return [
                {"type": "tool_result",
                 "tool_use_id": tool_use_block.id,
                 "content": [
                     {"type": "text",
                      "text": json.dumps({k: v for k, v in result.items() if k != "base64"},
                                         ensure_ascii=False)},
                     {"type": "image",
                      "source": {"type": "base64",
                                 "media_type": "image/png",
                                 "data": result["base64"]}}
                 ]}
            ]
        else:
            return [{"type": "tool_result",
                     "tool_use_id": tool_use_block.id,
                     "content": result_str}]

    def run(self, task: str):
        """Запустить агента с задачей."""
        log.info(f"=== НОВАЯ ЗАДАЧА ===\n{task}")
        self.history = [{"role": "user", "content": task}]
        self.iteration = 0

        while self.iteration < CFG["max_iterations"]:
            self.iteration += 1
            log.info(f"--- Итерация {self.iteration}/{CFG['max_iterations']} ---")

            response = self.client.messages.create(
                model=CFG["anthropic_model"],
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self.history,
            )

            log.info(f"stop_reason: {response.stop_reason}")

            # Добавить ответ ассистента в историю
            self.history.append({
                "role": "assistant",
                "content": response.content
            })

            # Если агент завершил — выход
            if response.stop_reason == "end_turn":
                final_text = " ".join(
                    b.text for b in response.content
                    if hasattr(b, "text")
                )
                log.info(f"=== ЗАДАЧА ЗАВЕРШЕНА ===\n{final_text}")
                print(f"\n✓ Агент завершил задачу:\n{final_text}")
                return final_text

            # Обработать вызовы инструментов
            if response.stop_reason == "tool_use":
                tool_results_content = []
                for block in response.content:
                    if block.type == "tool_use":
                        result_str = self._execute_tool(block.name, block.input)
                        tool_results_content.extend(
                            self._build_tool_result_content(block, result_str)
                        )

                self.history.append({
                    "role": "user",
                    "content": tool_results_content
                })

        log.warning("Достигнут лимит итераций")
        return "LIMIT_REACHED"


# ─────────────────────────────────────────────────────────────────
#  ПРЕДУСТАНОВЛЕННЫЕ ЗАДАЧИ
# ─────────────────────────────────────────────────────────────────

TASKS = {
    "1": {
        "name": "Проверить состояние системы",
        "prompt": """
Проверь текущее состояние:
1. Сделай скриншот экрана
2. Найди окна NeuraLOG
3. Выведи список файлов в F:\\nds\\images (*.png, *.jpg, *.tif)
4. Выведи список файлов в F:\\nds\\projects (*.nlgx)
5. Сообщи сколько изображений ожидают обработки и сколько проектов уже создано
"""
    },
    "2": {
        "name": "Векторизировать одно изображение",
        "prompt": """
Векторизируй первое необработанное изображение из F:\\nds\\images.
Необработанным считается изображение для которого нет .nlgx файла в F:\\nds\\projects с тем же именем.
Шаги:
1. list_files — найти изображения и существующие проекты
2. Определить первое необработанное изображение
3. Открыть NeuraLOG, создать/открыть проект для этого изображения
4. Выполнить векторизацию
5. Сохранить проект
6. Экспортировать LAS в F:\\nds\\output
7. report_status о результате
"""
    },
    "3": {
        "name": "Пакетная векторизация всех изображений",
        "prompt": """
Выполни пакетную векторизацию ВСЕХ изображений из F:\\nds\\images которые ещё не обработаны.
Для каждого изображения:
1. Открыть проект (или создать новый)
2. Загрузить изображение
3. Выполнить векторизацию  
4. Сохранить .nlgx проект
5. Экспортировать в F:\\nds\\output как LAS
6. report_status об этом изображении
Продолжай пока все изображения не будут обработаны.
"""
    },
    "4": {
        "name": "Экспортировать все готовые проекты",
        "prompt": """
Экспортируй все .nlgx проекты из F:\\nds\\projects в LAS формат в F:\\nds\\output.
Пропусти проекты у которых уже есть соответствующий .las файл в output.
"""
    },
}


def main():
    print("=== NeuraLOG AI Agent ===\n")
    print("Доступные задачи:")
    for k, v in TASKS.items():
        print(f"  {k}. {v['name']}")
    print("  c. Своя задача (custom)\n")

    choice = input("Выбери задачу: ").strip()

    if choice in TASKS:
        task_prompt = TASKS[choice]["prompt"]
    elif choice == "c":
        task_prompt = input("Введи задачу: ").strip()
    else:
        print("Неверный выбор")
        return

    agent = NeuraLogAgent()
    agent.run(task_prompt)


if __name__ == "__main__":
    main()

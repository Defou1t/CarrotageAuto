r"""_lms.py — ДЕЛЕГИРОВАНИЕ РУТИНЫ ЛОКАЛЬНОЙ МОДЕЛИ (LM Studio, Эдуард 22.07).

Экономия контекста: длинные логи гейтов, покривые таблицы и однотипные сводки не обязаны
проходить через дорогую модель. LM Studio крутится локально (OpenAI-совместимый API на :1234).

★ ГРАНИЦА ОТВЕТСТВЕННОСТИ (важно): локальной модели отдаётся ТОЛЬКО пересказ/сжатие уже
посчитанных чисел. Ей НЕ поручается интерпретация («засчитана ли правка»), выбор параметров и
любые выводы, влияющие на решения — цена ошибки там несопоставима с экономией.

  python _lms.py summarize <файл.log> [--model ...]      # сжать лог в таблицу
  python _lms.py ask "вопрос" --file <f>                 # произвольный вопрос по файлу
"""
import sys, json, argparse, urllib.request

URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.6-35b-a3b"


def ask(prompt, text="", model=DEFAULT_MODEL, temperature=0.0, max_tokens=1200):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content":
             "Ты помощник инженера. Отвечай ПО-РУССКИ, кратко, только фактами из данных. "
             "Не додумывай числа, не давай советов и оценок, если не просят. "
             "Если данных не хватает — так и скажи. НЕ рассуждай вслух, сразу давай результат."},
            {"role": "user", "content": ((prompt + "\n\n" + text) if text else prompt) + "\n/no_think"},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # reasoning-модели (qwen3.x) иначе отдают пустой content и весь ответ кладут в
        # reasoning_content — на выходе получается поток мыслей вместо сводки
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read().decode("utf-8"))
    msg = out["choices"][0]["message"]
    txt = (msg.get("content") or "").strip()
    if not txt:
        # запасной путь: модель всё же ушла в рассуждение — берём ТОЛЬКО хвост, где обычно
        # лежит итог, иначе в вывод попадает весь поток мыслей
        txt = (msg.get("reasoning_content") or "").strip()[-1500:]
    return txt


SUMMARIZE = (
    "Ниже — вывод инженерного гейта. Сожми его в компактную таблицу:\n"
    "1) строки, где результат ИЗМЕНИЛСЯ между вариантами (с числами);\n"
    "2) строки, где НЕ изменился — только их количество, без перечисления;\n"
    "3) итоговые суммы, если они есть в тексте.\n"
    "Ничего не интерпретируй и не оценивай — только перенеси числа."
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summarize", "ask", "ping"])
    ap.add_argument("arg", nargs="?", default="")
    ap.add_argument("--file", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "ping":
        print(ask("Ответь одним словом: работает"))
    else:
        src = a.file or a.arg
        text = open(src, encoding="utf-8", errors="replace").read() if src else ""
        print(ask(SUMMARIZE if a.cmd == "summarize" else a.arg, text, a.model))

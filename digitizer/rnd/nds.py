r"""nds.py — ПУЛЬТ: одно слово вместо длинной команды (§6.186).

ЗАЧЕМ. Управление счётом разъехалось по трём стендам с длинными ключами (`_govern.py --pause`,
`_nice.py --keep`, `_run_status.py`), и чтобы притормозить машину, надо было помнить путь к
интерпретатору и имя ключа. Заказчик попросил простой способ — вот он.

    nds                  что сейчас происходит (то же, что `nds status`)
    nds pause            ПАУЗА: счёт замирает, НИЧЕГО не теряется
    nds go               продолжить; дальше решает регулятор
    nds cores 4          держать РОВНО 4 процесса, и в простое тоже (ручной режим)
    nds auto             вернуть автоматику: под рукой мало, в простое много
    nds stop             остановить совсем (теперь дёшево: теряются листы, не часы)
    nds speed            темп на каждом уровне активных процессов

★ ПАУЗА И «cores» ЖИВУТ ФАЙЛАМИ, А НЕ ПАМЯТЬЮ. Регулятор просыпается раз в минуту и вернул бы
шарды обратно: две воли на один ресурс всегда кончаются победой той, что чаще просыпается.
⚠ `pause` ничего не теряет, но приостановленные процессы ДЕРЖАТ ПАМЯТЬ. На час — ставьте паузу,
на сутки дешевле дать досчитать или `nds stop`.
⚠ `stop` с 05.09 стоит недорого: шард пишет промежуточный дамп раз в 20 листов и при следующем
запуске их ПРОПУСКАЕТ (§6.186). До этой правки снятый шард терял всё — так ушли 249 листов.
"""
import sys, json, subprocess
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

RND = Path(__file__).resolve().parent
PY = sys.executable
LOG = Path(r"F:/nds/output/taskS/govern.log")
CONF = LOG.with_suffix(".conf.json")
PAUSED = LOG.with_suffix(".paused")


def gov(*args):
    # ⚠⚠ ПУЛЬТ САМ БЫЛ ВТОРОЙ ВОЛЕЙ. Планировщик зовёт регулятор с `--base 3 --max 8`, а отсюда он
    # звался БЕЗ ключей — то есть с умолчаниями стенда (base=4). Каждый `nds status` расширял счёт
    # до 4, каждый тик через минуту сжимал до 3, и журнал заполнялся качелями. ⇒ Настройка живёт
    # ТОЛЬКО в `govern.conf.json`, который перебивает ключи у ОБОИХ вызовов (§6.186).
    subprocess.run([PY, str(RND / "_govern.py"), *args], cwd=str(RND))


def conf_set(**kw):
    d = {}
    if CONF.is_file():
        try:
            d = json.loads(CONF.read_text(encoding="utf-8"))
        except Exception:
            d = {}
    d.update({k: v for k, v in kw.items() if v is not None})
    for k, v in kw.items():
        if v is None:
            d.pop(k, None)
    CONF.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


TS = Path(r"F:/nds/output/taskS")


def live():
    """★ ЧТО СЧИТАЕТСЯ ПРЯМО СЕЙЧАС — по СВЕЖЕСТИ каталогов, а не по зашитому имени.
    `_run_status.py` знает один конкретный замер и на любом другом показывает прошлый — сводка,
    формально верная и вводящая в заблуждение. Здесь каталог определяется свежестью логов шардов.
    """
    import time
    now = time.time()
    best, best_t = None, 0
    for d in TS.glob("ab_*"):
        if not d.is_dir():
            continue
        lg = list((d / "logs").glob("sh*.log")) if (d / "logs").is_dir() else []
        t = max((q.stat().st_mtime for q in lg), default=0)
        if t > best_t:
            best, best_t = d, t
    if best is None or now - best_t > 1800:
        print("★ СЧЁТ НЕ ИДЁТ: ни один прогон не писал в логи за последние 30 минут")
        return
    mins = (now - best_t) / 60
    print(f"★ СЧИТАЕТСЯ: {best.name}   (последняя запись {mins:.0f} мин назад)")
    for m in sorted(q.name for q in best.iterdir() if q.is_dir() and q.name != "logs"):
        print(f"   режим {m}: выдано листов {len(list((best / m).iterdir()))}")
    fin = list(best.glob("ab_*of*.pkl"))
    part = list(best.glob("ab_*of*.part.pkl"))
    fin = [q for q in fin if not q.name.endswith(".part.pkl")]
    print(f"   дампов ГОТОВЫХ {len(fin)}, промежуточных {len(part)}"
          + ("   ⚠ промежуточных нет — остановка сейчас дороже" if not part and not fin else ""))
    # ★ СЧИТАТЬ НАДО ОБА ВИДА. Здесь считались только промежуточные, и пульт говорил «сохранено
    #   180», когда три шарда были ДОСЧИТАНЫ и на диске лежало вчетверо больше. Заниженная цифра
    #   пугает не меньше завышенной: по ней паузу откладывают там, где она давно безопасна.
    import pickle

    def _sheets(paths):
        t = 0
        for q in paths:
            try:
                t += sum(len(v[1]) for v in pickle.load(open(q, "rb")).get("res", {}).values())
            except Exception:
                pass
        return t

    sf, sp = _sheets(fin), _sheets(part)
    if sf or sp:
        print(f"   ★ СОХРАНЕНО листо-прогонов: {sf + sp} "
              f"(в готовых {sf}, в промежуточных {sp}) — столько НЕ пропадёт при остановке")


def verdict():
    """★ ПРИГОВОР ПРИЁМКИ — В ПУЛЬТ, А НЕ В ЛОГ. Вердикт, лежащий только в файле лога, человек
    узнаёт последним; пульт — то место, куда и так смотрят."""
    vf = Path(r"F:/nds/output/taskS/pregate_verdict.txt")
    if not vf.is_file():
        return
    head = (vf.read_text(encoding="utf-8", errors="replace").splitlines() or [""])[0]
    mark = "★★★" if head.startswith("ПРИНЯТО") else "⛔⛔"
    print(f"\n{mark} ПРИЁМКА ПРЕДГЕЙТА: {head}\n    подробности: {vf}")


def status():
    live()
    verdict()
    print()
    if PAUSED.is_file():
        print("⏸  СЧЁТ НА ПАУЗЕ. Продолжить: nds go")
    d = {}
    if CONF.is_file():
        try:
            d = json.loads(CONF.read_text(encoding="utf-8"))
        except Exception:
            pass
    if isinstance(d.get("manual"), int):
        print(f"🔧 РУЧНОЙ РЕЖИМ: держится ровно {d['manual']} процессов. Вернуть автоматику: nds auto")
    elif isinstance(d.get("base"), int) and isinstance(d.get("max"), int):
        print(f"🔄 АВТОМАТИКА: под рукой {d['base']}, в простое до {d['max']}. "
              f"Задать вручную: nds cores N")
    else:
        # ★ ЧЕСТНЕЕ НЕ ЗНАТЬ, ЧЕМ ПОДСТАВИТЬ ПРАВДОПОДОБНОЕ. Здесь стояло `d.get('base', 3)` —
        #   и пульт печатал «под рукой 3», когда регулятор держал 4: умолчание пульта разошлось с
        #   умолчанием стенда, а на экране разницы не было видно.
        print("🔄 АВТОМАТИКА: пределы не заданы в govern.conf.json — действуют умолчания "
              "регулятора. Задать: nds cores N (ручной режим) или прописать base/max в конфиг.")
    gov("--once")


cmd = (sys.argv[1] if len(sys.argv) > 1 else "status").lower()
arg = sys.argv[2] if len(sys.argv) > 2 else None

if cmd in ("status", "", "-h", "--help", "help"):
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
    else:
        status()
elif cmd == "pause":
    gov("--pause")
    print("⏸  Пауза. Прогресс ЦЕЛ. Продолжить: nds go")
elif cmd in ("go", "resume", "unpause"):
    gov("--unpause")
    print("▶  Продолжаю. Дальше решает регулятор — nds status покажет, сколько активно.")
elif cmd == "cores":
    if not (arg or "").isdigit():
        print("⛔ сколько процессов держать? Например: nds cores 4")
        sys.exit(1)
    n = int(arg)
    conf_set(manual=n)
    if PAUSED.is_file():
        gov("--unpause")
    print(f"🔧 Держу ровно {n} процессов — и под рукой, и в простое.")
    gov("--once")
elif cmd == "auto":
    conf_set(manual=None)
    print("🔄 Автоматика вернулась: под рукой мало, в простое много.")
    gov("--once")
elif cmd == "stop":
    # ⚠ Убиваем ТОЛЬКО свой счёт, по командной строке, а не по имени `python.exe`: тем же
    #   интерпретатором идут разборы и этот самый пульт (§6.170).
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -match '_trace_prod_ab\\.py' } | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }")
    r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                       capture_output=True, text=True, creationflags=0x08000000)
    killed = [x for x in (r.stdout or "").split() if x.isdigit()]
    if PAUSED.is_file():
        PAUSED.unlink()
    print(f"⏹  Остановлено процессов: {len(killed)}")
    print("   Потеряны только листы после последнего промежуточного дампа (раз в 20 листов).")
    print("   Продолжить с того же места: тик поднимет сам, либо запустите драйвер прогона.")
elif cmd == "speed":
    gov("--rate")
else:
    print(__doc__)
    print(f"⛔ не знаю команду {cmd!r}")
    sys.exit(1)

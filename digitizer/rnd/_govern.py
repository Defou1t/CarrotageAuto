r"""_govern.py — ДИНАМИЧЕСКИЙ РЕГУЛЯТОР НАГРУЗКИ: БОЛЬШЕ ШАРДОВ В ПРОСТОЕ, МЕНЬШЕ ПОД РУКОЙ (§6.176).

ОТКУДА ВОПРОС (заказчик, 03.09): «когда ПК простаивает — брать больше мощностей, когда работает
с другими программами — меньше; можно динамично?» Можно, и в ветке это уже писали: §6.106 завёл
`_load_governor.ps1` ровно с этой логикой. ⛔ Но он МОЛЧА НЕ РАБОТАЛ: свои процессы он искал по
СПИСКУ ИМЁН ПРОГОНОВ (`ab_wellmap|ab_pickmodel|degen_ab`), и каждый новый замер требовал правки
этого списка. Последний запуск 30.08 сразу написал «шардов нет — выхожу»: имена не совпали.
⇒ Здесь процессы опознаются по ИМЕНИ СКРИПТА в командной строке (`_trace_prod_ab.py`), как в
`_nice.py` и в тике, — тогда новый прогон подхватывается сам.

⚠⚠ ЧЕГО РЕГУЛЯТОР НЕ МОЖЕТ, И ЭТО ОПРЕДЕЛЯЕТ, КАК ЗАПУСКАТЬ ПРОГОН. Он умеет ПРИОСТАНАВЛИВАТЬ и
ВОЗОБНОВЛЯТЬ уже запущенные шарды, но не умеет создавать новые. ⇒ Запас должен быть заложен
ЗАРАНЕЕ: прогон запускается на ВЕРХНЕЙ границе (`--max`), а регулятор держит активными `--base`,
пока машиной пользуются, и отпускает остальных в простое. Запустишь на четырёх — в простое их и
останется четыре.

ЧТО СЧИТАЕТСЯ ПРОСТОЕМ — два независимых признака, оба должны сойтись (как в §6.106):
  1. пользователь не трогал ввод дольше `--idle-sec` (GetLastInputInfo);
  2. ЧУЖАЯ загрузка процессора (всё, кроме наших питонов) ниже `--foreign-pct`.
Второй обязателен: игра, сборка или рендер грузят машину и без участия рук.
★ Плюс пол по памяти (`--min-free-gb`): §6.106 — на 24 шардах осталось 5 ГБ из 61, машина ушла в
подкачку, и темп упал с 6.5 до 0.7 лист-режима в минуту. Жадность сделала хуже И человеку, И замеру.
⇒ Ниже пола сжимаемся НЕЗАВИСИМО от простоя.

★★ И ГЛАВНОЕ ОТЛИЧИЕ ОТ §6.106: РЕГУЛЯТОР ПИШЕТ ЖУРНАЛ, ПО КОТОРОМУ ЕГО МОЖНО ПРОВЕРИТЬ.
Каждое решение — строка с временем, числом активных, чужой загрузкой, простоем и свободной
памятью. `--rate` читает этот журнал вместе с временами готовых каталогов выдачи и печатает
ТЕМП (листов в минуту) НА КАЖДОМ УРОВНЕ активных шардов. ⇒ «больше шардов быстрее» перестаёт
быть допущением: на 4 шардах темп по фолдам 1.60 / 2.06 / 1.53 — разброс ±25%, и разницу меньше
трети одним сравнением не поймать. Нужны выборки на каждом уровне, а не одно сравнение.

  <ComfyUI>\python_embeded\python.exe _govern.py --once            # один шаг (для тика раз в минуту)
  <ComfyUI>\python_embeded\python.exe _govern.py --loop            # цикл, шаг раз в --every секунд
  <ComfyUI>\python_embeded\python.exe _govern.py --rate            # темп по уровням, из журнала
  <ComfyUI>\python_embeded\python.exe _govern.py --release         # снять регулирование, вернуть всех
"""
import os, sys, argparse, ctypes, subprocess, re, time, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--match", default=r"_trace_prod_ab\.py|_name_cost_prod\.py",
                help="регулярка по КОМАНДНОЙ СТРОКЕ: что считать нашим счётом")
ap.add_argument("--base", type=int, default=4, help="активных, пока машиной пользуются")
ap.add_argument("--max", type=int, default=99, help="потолок активных в простое")
ap.add_argument("--idle-sec", type=int, default=180, help="столько секунд без ввода = простой")
ap.add_argument("--foreign-pct", type=float, default=15.0, help="выше — машина занята чужим")
ap.add_argument("--min-free-gb", type=float, default=10.0, help="ниже — сжимаемся независимо")
ap.add_argument("--every", type=int, default=45)
ap.add_argument("--log", default=r"F:/nds/output/taskS/govern.log")
ap.add_argument("--out", default=r"F:/nds/output/taskS/ab_pickhonest_run",
                help="каталог прогона: по временам его выдач считается темп для --rate")
ap.add_argument("--once", action="store_true")
ap.add_argument("--loop", action="store_true")
ap.add_argument("--rate", action="store_true")
ap.add_argument("--release", action="store_true", help="возобновить всех и выйти")
ap.add_argument("--pause", action="store_true",
                help="ПАУЗА: приостановить ВСЕ шарды и запретить регулятору их возвращать")
ap.add_argument("--unpause", action="store_true", help="снять паузу и вернуть шарды")
a = ap.parse_args()

k32, ntd = ctypes.windll.kernel32, ctypes.windll.ntdll
PROC_SUSPEND = 0x0800


class LASTINPUT(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


def idle_seconds():
    li = LASTINPUT(); li.cbSize = ctypes.sizeof(LASTINPUT)
    ctypes.windll.user32.GetLastInputInfo(ctypes.byref(li))
    return max(0.0, (k32.GetTickCount() - li.dwTime) / 1000.0)


def ps(cmd):
    return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", cmd],
                          capture_output=True, text=True, timeout=90,
                          creationflags=0x08000000).stdout   # CREATE_NO_WINDOW: без окон (§6.168)


def snapshot():
    """Один заход к Windows: наши процессы с их временем ЦП, суммарное время ЦП, свободная память."""
    out = ps("$os=Get-CimInstance Win32_OperatingSystem; "
             "\"MEM`t$($os.FreePhysicalMemory)\"; "
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "ForEach-Object { \"P`t$($_.ProcessId)`t$($_.KernelModeTime + $_.UserModeTime)`t$($_.CommandLine)\" }; "
             # ★ ProcessId 0 — «Система бездействия», и её время ЦП РАВНО ПРОСТОЮ. Без этого
             # фильтра «чужая нагрузка» всегда выходит ≈100% — первая редакция дала 121%, то есть машина
             # числилась бы занятой ВСЕГДА и регулятор никогда бы не расширился.
             "Get-CimInstance Win32_Process -Filter 'ProcessId != 0' | "
             "Measure-Object -Property KernelModeTime,UserModeTime -Sum | "
             "ForEach-Object { \"T`t$($_.Sum)\" }")
    rx, ours, free, tot = re.compile(a.match), [], 0.0, 0.0
    for ln in out.splitlines():
        p = ln.split("\t")
        if p[0] == "MEM":
            free = float(p[1]) / 1048576.0
        elif p[0] == "P" and len(p) >= 4 and rx.search(p[3]):
            ours.append((int(p[1]), float(p[2])))
        elif p[0] == "T":
            tot += float(p[1])
    return ours, tot, free


# ⚠⚠ ЧИСЛО ЯДЕР БЕРЁТСЯ В СВОЁМ ЖЕ ПРОЦЕССЕ. Первая редакция спрашивала его ОТДЕЛЬНЫМ python —
# без `creationflags`, то есть С КОНСОЛЬНЫМ ОКНОМ, и регулятор моргал им РАЗ В МИНУТУ. Заказчик это
# и увидел. Урок общий: подпроцесс в фоновой задаче обязан идти с CREATE_NO_WINDOW, а подпроцесс,
# которого можно не запускать, — не запускаться вовсе.
NCPU = os.cpu_count() or 32


def foreign_pct(dt=2.0):
    """Чужая загрузка = (всё время ЦП − НАШЕ время ЦП) за интервал, в процентах от всех ядер.
    ⚠ Считается по РАЗНОСТИ счётчиков, а не по мгновенной величине: мгновенная у CIM шумит."""
    o1, t1, free = snapshot()
    time.sleep(dt)
    o2, t2, _ = snapshot()
    mine = sum(c for p, c in o2) - sum(c for p, c in o1 if p in {q for q, _ in o2})
    total = t2 - t1
    # времена в 100-нс единицах; знаменатель — dt на всех ядрах
    denom = dt * 1e7 * NCPU
    return max(0.0, 100.0 * (total - mine) / denom), o2, free


STATE = Path(a.log).with_suffix(".state.json")
# ★★ ПАУЗА — ФАЙЛОМ, А НЕ ПАМЯТЬЮ ПРОЦЕССА. Регулятор живёт отдельным запуском раз в минуту, и
# «приостановил вручную» он бы через минуту отменил, вернув шарды: две воли на один ресурс всегда
# кончаются тем, что побеждает та, что чаще просыпается. Признак на диске переживает и запуск
# регулятора, и снос сессии, и перезагрузку планировщика.
PAUSED = Path(a.log).with_suffix(".paused")
# ★★ НАСТРОЙКА — ТОЖЕ ФАЙЛОМ, ПО ТОЙ ЖЕ ПРИЧИНЕ, ЧТО И ПАУЗА. Регулятор живёт отдельным запуском
# раз в минуту: значение, переданное ключом, живёт до конца этого запуска и ни на что дальше не
# влияет. Человек же меняет «сколько ядер занимать» НАДОЛГО. ⇒ `govern.conf.json` перебивает ключи.
#   {"manual": N} — держать РОВНО N активных, не глядя на простой (ручной режим);
#   {"base": N, "max": M} — обычный режим: N под рукой, до M в простое.
CONF = Path(a.log).with_suffix(".conf.json")
_conf = {}
if CONF.is_file():
    try:
        _conf = json.loads(CONF.read_text(encoding="utf-8"))
    except Exception:
        _conf = {}
if isinstance(_conf.get("base"), int):
    a.base = _conf["base"]
if isinstance(_conf.get("max"), int):
    a.max = _conf["max"]
MANUAL = _conf.get("manual") if isinstance(_conf.get("manual"), int) else None


def load_state():
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {"susp": []}


def say(m):
    line = time.strftime("%m-%d %H:%M:%S") + "  " + m
    with open(a.log, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    # ⚠ ПЕЧАТЬ НЕ ИМЕЕТ ПРАВА УРОНИТЬ РЕГУЛЯТОР. Под планировщиком (wscript //B) и в конвейере
    # PowerShell стандартный вывод бывает недоступен, и `print` бросает OSError 22. Регулятор к
    # этому моменту уже ПРИОСТАНОВИЛ процессы — падение оставило бы их замороженными навсегда.
    # Журнал в файле обязателен, экран — удобство.
    try:
        print(line)
    except OSError:
        pass


def setrun(pid, resume):
    h = k32.OpenProcess(PROC_SUSPEND, False, pid)
    if not h:
        return False
    r = ntd.NtResumeProcess(h) if resume else ntd.NtSuspendProcess(h)
    k32.CloseHandle(h)
    return r == 0


def step():
    fp, ours, free = foreign_pct()
    if not ours:
        say("шардов нет — регулировать нечего")
        return
    if PAUSED.is_file():
        # ⚠ Пауза сильнее любых признаков простоя: её ставит человек, и снимать её сам регулятор
        #   не вправе. Догоняем только тех, кто ещё работает, — иначе новый шард, запущенный тиком,
        #   остался бы активным вопреки паузе.
        st = load_state()
        susp = set(st.get("susp", []))
        add = [pid for pid, _ in ours if pid not in susp and setrun(pid, False)]
        if add:
            susp |= set(add)
            STATE.write_text(json.dumps({"susp": sorted(susp)}), encoding="utf-8")
        say(f"ПАУЗА ({PAUSED.name}): приостановлено всего {len(susp)}"
            + (f", из них сейчас {len(add)}" if add else "") + ". Снять: --unpause")
        return
    st = load_state()
    susp = [p for p in st["susp"] if p in {q for q, _ in ours}]
    idle = idle_seconds()
    quiet = idle >= a.idle_sec and fp < a.foreign_pct
    # ★ РУЧНОЙ РЕЖИМ СИЛЬНЕЕ ПРОСТОЯ: человек сказал «столько», значит столько, и в простое тоже.
    want = MANUAL if MANUAL is not None else (min(a.max, len(ours)) if quiet else a.base)
    if free < a.min_free_gb:
        # ★ пол по памяти сильнее простоя: §6.106 — подкачка роняет темп в разы
        want = min(want, max(1, a.base - 2))
    want = max(1, min(want, len(ours)))
    active = [p for p, _ in ours if p not in susp]

    why = (f"простой {idle:.0f}с, чужой ЦП {fp:.0f}%, память {free:.1f} ГБ ⇒ "
           + (f"РУЧНОЙ РЕЖИМ: держу {MANUAL}" if MANUAL is not None
              else ("ПРОСТОЙ" if quiet else "машина занята")))
    if len(active) == want:
        say(f"активных {len(active)} из {len(ours)} — как надо; {why}")
        return
    if len(active) > want:                       # сжимаемся: приостановить лишние
        for p in active[want:]:
            if setrun(p, False):
                susp.append(p)
        msg = f"СЖИМАЮСЬ: {len(active)} → {want} активных (приостановлено {len(active)-want}); {why}"
    else:                                        # расширяемся: вернуть приостановленных
        need = want - len(active)
        for p in list(susp)[:need]:
            if setrun(p, True):
                susp.remove(p)
        msg = f"РАСШИРЯЮСЬ: {len(active)} → {len(ours)-len(susp)} активных; {why}"
    # ★★ СОСТОЯНИЕ ЗАПИСЫВАЕТСЯ РАНЬШЕ СЛОВ. Приостановка уже СЛУЧИЛАСЬ; если что-то помешает
    # между делом и записью, следующий тик не найдёт приостановленных в списке, не вернёт их и
    # будет замораживать дальше — счёт встанет молча. Дело фиксируем, потом рассказываем.
    STATE.write_text(json.dumps({"susp": susp}), encoding="utf-8")
    say(msg)


def release():
    _, ours, _ = foreign_pct(0.2)
    st = load_state()
    n = sum(1 for p in st.get("susp", []) if setrun(p, True))
    STATE.write_text(json.dumps({"susp": []}), encoding="utf-8")
    say(f"РЕГУЛИРОВАНИЕ СНЯТО: возобновлено {n}, активны все {len(ours)}")


def rate():
    """Темп на КАЖДОМ уровне активных шардов: сколько листов выдано за минуты, пока держался
    этот уровень. ⚠ Это ответ на «больше шардов быстрее?» ЗАМЕРОМ, а не допущением."""
    lg = Path(a.log)
    if not lg.is_file():
        sys.exit("⛔ журнала регулятора нет — темп считать не из чего")
    ev = []
    for ln in lg.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\d\d-\d\d \d\d:\d\d:\d\d)\s+.*?активных (\d+)", ln)
        if not m:
            m2 = re.match(r"(\d\d-\d\d \d\d:\d\d:\d\d)\s+.*?→ (\d+) активных", ln)
            if not m2:
                continue
            m = m2
        t = time.mktime(time.strptime(f"{time.strftime('%Y')}-{m.group(1)}", "%Y-%m-%d %H:%M:%S"))
        ev.append((t, int(m.group(2))))
    if len(ev) < 2:
        sys.exit("⛔ в журнале меньше двух решений — уровни не с чем сравнивать")
    sheets = sorted(q.stat().st_mtime for f in Path(a.out).glob("f*/M") for q in f.iterdir())
    per = {}
    for (t0, n), (t1, _) in zip(ev, ev[1:]):
        c = sum(1 for s in sheets if t0 <= s < t1)
        mins = (t1 - t0) / 60.0
        if mins > 0.5:
            d = per.setdefault(n, [0.0, 0.0])
            d[0] += c; d[1] += mins
    print(f"★ ТЕМП ПО УРОВНЯМ АКТИВНЫХ ШАРДОВ (выдач {len(sheets)}, решений в журнале {len(ev)})")
    print("| активных | листов | минут | листов/мин |")
    for n in sorted(per):
        c, mins = per[n]
        print(f"| {n} | {c:.0f} | {mins:.0f} | {c/mins if mins else 0:.2f} |")
    print("⚠ уровни набираются НЕОДИНАКОВЫМИ листами: сложность бланка гуляет, и на 4 шардах "
          "разброс темпа по фолдам был 1.53-2.06/мин (±25%). Меньшую разницу так не поймать.")


def pause(on):
    """Пауза: приостановить ВСЁ и запретить возврат. Прогресс цел — дамп шарда пишется в конце,
    приостановленный процесс его не теряет (§6.171). ⚠ Память шарды продолжают держать."""
    _, ours, free = foreign_pct(0.2)
    if on:
        PAUSED.write_text("пауза поставлена вручную; снять: _govern.py --unpause\n", encoding="utf-8")
        st = load_state()
        susp = set(st.get("susp", []))
        for pid, _ in ours:
            if pid not in susp and setrun(pid, False):
                susp.add(pid)
        STATE.write_text(json.dumps({"susp": sorted(susp)}), encoding="utf-8")
        say(f"★ ПАУЗА: приостановлено {len(susp)} шардов, регулятор их не вернёт. "
            f"Свободно памяти {free:.1f} ГБ — приостановленные её ДЕРЖАТ.")
        print("⚠ Прогресс цел полностью: приостановленный процесс ничего не теряет.")
        print("⚠ Но память шарды держат. Если пауза на сутки — дешевле дать досчитать.")
    else:
        if PAUSED.is_file():
            PAUSED.unlink()
        st = load_state()
        n = sum(1 for pid in st.get("susp", []) if setrun(pid, True))
        STATE.write_text(json.dumps({"susp": []}), encoding="utf-8")
        say(f"★ ПАУЗА СНЯТА: возобновлено {n}; дальше решает регулятор (база {a.base})")


if a.pause or a.unpause:
    pause(a.pause)
elif a.rate:
    rate()
elif a.release:
    release()
elif a.loop:
    say(f"регулятор запущен: база {a.base}, потолок {a.max}, простой ≥{a.idle_sec}с, "
        f"чужой ЦП <{a.foreign_pct}%, пол памяти {a.min_free_gb} ГБ, шаг {a.every}с")
    while True:
        try:
            step()
        except Exception as e:
            say(f"⚠ шаг не удался: {type(e).__name__}: {e}")
        time.sleep(a.every)
else:
    step()

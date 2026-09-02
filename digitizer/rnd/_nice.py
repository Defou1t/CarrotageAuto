r"""_nice.py — ОПУСТИТЬ ПРИОРИТЕТ НАШЕГО СЧЁТА, ЧТОБЫ МАШИНА ОСТАВАЛАСЬ ЖИВОЙ (§6.170).

ОТКУДА ВОПРОС. Замер идёт 8 шардами по 3 потока = 24 из 32 ядер, с ОБЫЧНЫМ приоритетом. Это
именно та патология, которую ветка уже описывала (§6.106): жадность к ресурсам делает хуже и
человеку, и замеру — машина уходит в подкачку, темп падает. Заказчик: «остальные процессы сильно
глючат».

ЧТО ДЕЛАЕТ. Находит НАШИ процессы счёта по КОМАНДНОЙ СТРОКЕ (не по имени `python.exe` — им
запускаются и разборы, и посторонние программы) и ставит им класс приоритета BELOW_NORMAL, а
`--idle` — IDLE. Планировщик Windows отдаёт им процессор только когда он свободен: на пустой
машине счёт идёт полным ходом, под нагрузкой — уступает.

⚠ ЭТО НЕ ЗАМЕДЛЯЕТ СЧЁТ НА ПРОСТОЙНОЙ МАШИНЕ. Приоритет влияет на КОНКУРЕНЦИЮ за процессор, а
не на скорость самих вычислений; когда конкурентов нет, разницы нет.
⚠ Ввод-вывод и память приоритет не трогает — если прогон упирается в них, поможет только меньше
шардов (`--shard i/N`) или меньше `OMP_NUM_THREADS`.

Без внешних зависимостей: OpenProcess/SetPriorityClass через ctypes, список процессов — через
WMI-запрос средствами самого Windows (`wmic` не нужен).

  <ComfyUI>\python_embeded\python.exe _nice.py              # понизить до BELOW_NORMAL
  <ComfyUI>\python_embeded\python.exe _nice.py --idle       # до IDLE (совсем в фон)
  <ComfyUI>\python_embeded\python.exe _nice.py --show       # только показать, не менять
"""
import sys, argparse, ctypes, subprocess, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ap = argparse.ArgumentParser()
ap.add_argument("--match", default=r"_trace_prod_ab\.py|_name_cost_prod\.py",
                help="регулярка по КОМАНДНОЙ СТРОКЕ: что считать нашим счётом")
ap.add_argument("--idle", action="store_true", help="IDLE вместо BELOW_NORMAL")
ap.add_argument("--show", action="store_true", help="только показать")
ap.add_argument("--keep", type=int, default=0,
                help="оставить работать N шардов, остальные ПРИОСТАНОВИТЬ (прогресс цел)")
ap.add_argument("--resume", action="store_true", help="возобновить приостановленные")
a = ap.parse_args()

BELOW_NORMAL, IDLE = 0x00004000, 0x00000040
PROC_SET_INFO, PROC_QUERY = 0x0200, 0x0400
NAMES = {0x100: "REALTIME", 0x80: "HIGH", 0x8000: "ABOVE_NORMAL",
         0x20: "NORMAL", 0x4000: "BELOW_NORMAL", 0x40: "IDLE"}

k32 = ctypes.windll.kernel32

# ★ Список процессов берём у самого Windows одной командой, без сторонних пакетов. `-NoProfile`
#   и `-NonInteractive` обязательны: профиль пользователя может печатать своё и ломать разбор.
ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
      "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
try:
    out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                         capture_output=True, text=True, timeout=60,
                         creationflags=0x08000000).stdout  # CREATE_NO_WINDOW
except Exception as e:
    sys.exit(f"⛔ не удалось получить список процессов: {type(e).__name__}: {e}")

rx = re.compile(a.match)
found = []
for line in out.splitlines():
    if "\t" not in line:
        continue
    pid, cmd = line.split("\t", 1)
    if not pid.strip().isdigit() or not rx.search(cmd):
        continue
    found.append((int(pid), cmd.strip()))

if not found:
    print("★ наших процессов счёта не найдено — менять нечего")
    sys.exit(0)

# ★★ ПРИОСТАНОВКА, А НЕ УБИЙСТВО. Дамп шарда пишется в САМОМ КОНЦЕ, поэтому убить шард —
# потерять часы счёта. Приостановленный процесс не ест процессор, но держит всю свою работу;
# так же поступает и `_load_governor.ps1` (§6.106).
if a.keep or a.resume:
    nt = ctypes.windll.ntdll
    PROC_SUSPEND = 0x0800
    act = "возобновляю" if a.resume else "приостанавливаю"
    tail = found if a.resume else found[a.keep:]
    print(f"★ {act} процессов: {len(tail)} (оставляю работать {0 if a.resume else a.keep})")
    for pid, cmd in tail:
        h = k32.OpenProcess(PROC_SUSPEND, False, pid)
        if not h:
            print(f"   pid {pid:>6}  ⛔ нет доступа")
            continue
        r = nt.NtResumeProcess(h) if a.resume else nt.NtSuspendProcess(h)
        print(f"   pid {pid:>6}  {'возобновлён' if a.resume else 'приостановлен'}"
              f"{'' if r == 0 else f'  ⛔ код {r}'}")
        k32.CloseHandle(h)
    print("⚠ Приостановленные шарды НЕ считают, но и НЕ ТЕРЯЮТ прогресс. Возобновить: --resume")
    sys.exit(0)

want = IDLE if a.idle else BELOW_NORMAL
want_name = "IDLE" if a.idle else "BELOW_NORMAL"
print(f"★ найдено процессов счёта: {len(found)}")
ok = fail = 0
for pid, cmd in found:
    h = k32.OpenProcess(PROC_SET_INFO | PROC_QUERY, False, pid)
    if not h:
        print(f"   pid {pid:>6}  ⛔ нет доступа (код {k32.GetLastError()})")
        fail += 1
        continue
    cur = k32.GetPriorityClass(h)
    if a.show:
        print(f"   pid {pid:>6}  сейчас {NAMES.get(cur, hex(cur))}")
    else:
        r = k32.SetPriorityClass(h, want)
        print(f"   pid {pid:>6}  {NAMES.get(cur, hex(cur))} → {want_name}"
              f"{'' if r else '   ⛔ НЕ УДАЛОСЬ'}")
        ok += bool(r); fail += (not r)
    k32.CloseHandle(h)
if not a.show:
    print(f"★ понижено {ok}, не удалось {fail}")
    print("⚠ приоритет влияет на КОНКУРЕНЦИЮ за процессор: на свободной машине счёт не замедлится.")
    print("⚠ если тормозит ввод-вывод или память — помогает только меньше шардов или OMP.")

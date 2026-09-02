r"""_drivers_sync.py — ДРАЙВЕРЫ ЗАМЕРОВ ПОД ВЕРСИЮ, С ПРОВЕРКОЙ РАСХОЖДЕНИЯ (§6.171).

ОТКУДА ВОПРОС. Все 29 драйверов замера (`_pickmodel_tick.ps1`, `_rdhonest_run.ps1`,
`_load_governor.ps1` и прочие) лежат в `F:/nds/output/taskS` — ВНЕ репозитория, на томе, который
подключён джанкшеном. Под версией не было НИ ОДНОГО. При этом журнал ссылается на их поведение
постоянно (§6.164 про снос дерева, §6.166 про дозапуск, §6.171 про стирание готовых дампов), а
воспроизвести замер без точного драйвера нельзя: в шапке каждого записано, ЗАЧЕМ он так устроен.
⇒ Потеря тома или неудачная правка стирали бы не файлы, а способ повторить всё замеренное.
Вес вопроса: 99 КБ. Цена бездействия: любой замер.

ЧТО ДЕЛАЕТ. Копирует `taskS/*.ps1` и `*.vbs` в `rnd/drivers/` и печатает, что изменилось.
⚠ Копия, которая молча расходится с оригиналом, ХУЖЕ отсутствия копии: она врёт о том, чем
считали. Поэтому расхождение здесь не прячется, а печатается, и `--check` не пишет ничего —
только сообщает и возвращает 1, если репозиторий отстал от того, что реально исполняется.

  <ComfyUI>\python_embeded\python.exe _drivers_sync.py            # синхронизировать
  <ComfyUI>\python_embeded\python.exe _drivers_sync.py --check    # только сказать, есть ли расхождение
"""
import sys, argparse, shutil
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=r"F:\nds\output\taskS")
ap.add_argument("--dst", default=str(Path(__file__).parent / "drivers"))
ap.add_argument("--check", action="store_true", help="не копировать, только сообщить о расхождении")
a = ap.parse_args()

src, dst = Path(a.src), Path(a.dst)
dst.mkdir(exist_ok=True)
live = sorted(f for p in ("*.ps1", "*.vbs") for f in src.glob(p))

new = changed = same = 0
for f in live:
    d = dst / f.name
    # ★ сравниваем БАЙТЫ, а не время: копирование обновляет mtime, и по времени всё всегда «свежо»
    if not d.exists():
        state, new = "НОВЫЙ", new + 1
    elif d.read_bytes() != f.read_bytes():
        state, changed = "РАСХОДИТСЯ", changed + 1
    else:
        state, same = "", same + 1
    if state:
        print(f"   {state:<12} {f.name}")
        if not a.check:
            shutil.copy2(f, d)

# ★ файл, исчезнувший из taskS, из репозитория НЕ удаляем: журнал может на него ссылаться,
#   а история — единственное, что от снятого драйвера остаётся.
gone = sorted(p.name for p in dst.glob("*") if not (src / p.name).exists())
if gone:
    print(f"   ★ только в репозитории ({len(gone)}): {', '.join(gone[:6])}"
          + (f" … ещё {len(gone) - 6}" if len(gone) > 6 else ""))

print(f"★ драйверов {len(live)}: новых {new}, разошлось {changed}, совпадает {same}"
      f"{'  (ничего не писал)' if a.check else ''}")
sys.exit(1 if (a.check and (new or changed)) else 0)

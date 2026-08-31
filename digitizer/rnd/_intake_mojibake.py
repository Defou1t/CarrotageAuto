r"""_intake_mojibake.py — ВОССТАНОВИТЬ ИМЕНА ФАЙЛОВ, ИСПОРЧЕННЫЕ ПРИ РАСПАКОВКЕ АРХИВОВ.

ЗАЧЕМ (запрос Эдуарда 08.08). Часть архивов сделана инструментами, кладущими имена в DOS-кодировке
cp866 (см. предупреждение в `_intake_sort.unpack_all`: python `zipfile` такие бракует, WinRAR
открывает). WinRAR прочитал байты имени как cp437 — и получилось `üé»104_1985.08.03_üèç.JPG`
вместо `БВп104_1985.08.03_БКЗ.JPG`.

ЧЕМ ЭТО ВРЕДНО. `_intake_sort.well_of` разбирает имя, а не содержимое: скважину `üé»104` он не
опознаёт, файл уходит в корзину «имя не разобрано» и в наборы не попадает. Замерено: 74 готовые
ПАРЫ «скан + разметка» лежат в отстойнике мёртвым грузом ровно по этой причине.

ПРАВИЛО. Имя чинится, если `имя.encode(cp437).decode(cp866)` даёт кириллицу. Обратное
преобразование однозначно: это ровно те же байты, прочитанные правильной таблицей.
⚠ Не трогаем имена, где кириллицы не получилось: значит имя латинское и портить его нечем.

★ ПО УМОЛЧАНИЮ — СУХОЙ ПРОГОН. Ничего не переименовывается без `--apply`.

  <python> _intake_mojibake.py [--apply] [--root ...\intake]
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", nargs="+", default=[r"E:\Carrotagki_auto\intake"])
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()


def fix(s):
    """Имя обратно в cp866. Пробуем обе DOS-таблицы, которыми WinRAR мог прочитать байты."""
    for enc in ("cp437", "cp850"):
        try:
            r = s.encode(enc).decode("cp866")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if any("А" <= c <= "я" or c in "ёЁ" for c in r):
            return r
    return None


todo = []
for root in a.root:
    for p in Path(root).rglob("*"):
        if not p.is_file():
            continue
        r = fix(p.name)
        if r and r != p.name:
            todo.append((p, r))
print(f"имён к восстановлению: {len(todo)}")
for p, r in todo[:8]:
    print(f"  {p.name[:46]}  →  {r[:46]}")
if len(todo) > 8:
    print(f"  … и ещё {len(todo)-8}")

if not a.apply:
    print("\n★ Это СУХОЙ ПРОГОН. Ничего не переименовано. Переименовать: тот же вызов с --apply")
    sys.exit()

n, failed = 0, 0
for p, r in todo:
    t = p.parent / r
    # ⚠ Столкновение возможно: испорченное и целое имя могут сосуществовать (файл пришёл из двух
    # архивов — из битого и из нормального). Затирать нельзя, разводим суффиксом.
    if t.exists():
        i = 2
        while (p.parent / f"{Path(r).stem} ({i}){Path(r).suffix}").exists():
            i += 1
        t = p.parent / f"{Path(r).stem} ({i}){Path(r).suffix}"
    try:
        p.rename(t); n += 1
    except OSError as e:
        failed += 1
        if failed <= 5:
            print(f"  ⛔ {p.name[:50]}: {e}")
print(f"\n★ восстановлено имён: {n}" + (f", ⛔ НЕ УДАЛОСЬ {failed}" if failed else ""))

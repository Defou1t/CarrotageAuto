r"""_stand_audit.py — ПРОВЕРКА УКАЗАТЕЛЯ СТЕНДОВ И ОБЯЗАТЕЛЬНОГО МИНИМУМА (§6.106).

⚠⚠ ЗАЧЕМ. В каталоге 183 стенда, роадмап ссылается на них поимённо, и дважды за одну сессию (03.08)
стенд молча мерил НЕ ТОТ ОБЪЁМ и выдавал правдоподобное число: `_slot_cause.py` печатал «106 листов»
СТРОКОЙ при прогоне по 702, `_slot_rules.py` по умолчанию брал один каталог из пяти (20 листов из
702) — и на той базе четыре правила из девятнадцати имели ДРУГОЙ ЗНАК. Оба прогона завершались
успешно. Указатель, который никто не проверяет, разъедется так же незаметно, поэтому он проверяется
машиной.

ЧТО ПРОВЕРЯЕТСЯ:
  1. ПОЛНОТА УКАЗАТЕЛЯ — каждый `*.py` каталога назван в `STANDS.md` и наоборот. Появился стенд без
     строки в указателе (или строка без файла) — падение.
  2. ОБЪЁМ ИЗ СЧЁТЧИКА — в печатаемых строках ДЕЙСТВУЮЩИХ стендов не должно быть числа, приклеенного
     к слову объёма («106 листов», «три набора»). Ловится именно ловушка `_slot_cause.py`.
  3. СВЕРКА СПИСКА — действующий стенд, который обходит список листов и умеет ПРОПУСКАТЬ (`continue`),
     обязан печатать «обработано + пропущено против длины списка» (образец `_pool_oracle.py`).
  4. ПИННИНГ ПУТИ — стенд, который гоняет пайплайн (`pipeline.run`) или собирает `Config()`, обязан
     присваивать `cv.seq_model` явно: при доступном torch прод ведёт линии селектором, и стенд,
     наследующий умолчание, меряет РАЗНЫЙ алгоритм на разных машинах (ловушка `_slot_prod_ab.py`).

Список ДЕЙСТВУЮЩИХ стендов берётся ИЗ САМОГО УКАЗАТЕЛЯ (строки со статусом «ДЕЙСТВ»), а не из
отдельного списка здесь: два списка разъехались бы.

  python _stand_audit.py            # обычный python, torch не нужен
  python _stand_audit.py --all      # линтовать не только действующие, но все
"""
import sys, re, ast, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

RND = Path(__file__).resolve().parent
IDX = RND / "STANDS.md"

ap = argparse.ArgumentParser()
ap.add_argument("--all", action="store_true", help="линтовать все стенды, не только действующие")
a = ap.parse_args()

if not IDX.exists():
    sys.exit(f"⛔ нет указателя {IDX}")
md = IDX.read_text(encoding="utf-8")
files = sorted(p.name for p in RND.glob("*.py"))

# ── 1. полнота указателя ──────────────────────────────────────────────────────────────────────
named = set(re.findall(r"`([A-Za-z_0-9]+\.py)`", md))
missing = [f for f in files if f not in named]
extra = sorted(named - set(files))
print(f"файлов в каталоге {len(files)}, названо в указателе {len(named & set(files))}")
if missing:
    print(f"⛔ НЕТ СТРОКИ В УКАЗАТЕЛЕ ({len(missing)}): {', '.join(missing)}")
if extra:
    print(f"⛔ В УКАЗАТЕЛЕ ЕСТЬ, В КАТАЛОГЕ НЕТ ({len(extra)}): {', '.join(extra)}")

# ── действующие: строки указателя, где рядом с именем стоит «ДЕЙСТВ» ──────────────────────────
ACTIVE = set()
for ln in md.splitlines():
    if "ДЕЙСТВ" in ln:
        ACTIVE.update(n for n in re.findall(r"`([A-Za-z_0-9]+\.py)`", ln) if n in set(files))
print(f"действующих по указателю {len(ACTIVE)}")

VOL = r"(лист|крив|набор|скважин|трасс|слот|пар[аы]?\b)"
bad = {}


def flag(f, code, msg):
    bad.setdefault(f, []).append(f"{code} {msg}")


for f in (files if a.all else sorted(ACTIVE)):
    src = (RND / f).read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        flag(f, "⛔СИНТАКСИС", str(e)); continue

    # 2. число, приклеенное к слову объёма, внутри ПЕЧАТАЕМОЙ строки
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print"):
            continue
        for part in ast.walk(node):
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                m = re.search(r"(\d+|дв[ае]|три|четыре|пять)\s+\w*" + VOL, part.value)
                if m:
                    flag(f, "⚠ОБЪЁМ-ТЕКСТОМ", f"строка {part.lineno}: «{m.group(0)}» — печатать из счётчика")

    # 3. сверка списка у стендов, которые обходят листы и умеют пропускать
    # ⚠ `continue` ищется В ТЕЛЕ САМОГО ЦИКЛА, а не по всему файлу: `_pick_gate.py` обходит кэш без
    # единого пропуска, а `continue` у него в других функциях — правило «есть слово continue»
    # ругалось бы на честный стенд, и на такое правило перестают смотреть.
    walks = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.For) and isinstance(node.iter, ast.Name)
                and node.iter.id in ("FILES", "sheets", "SHEETS", "files")):
            walks = walks or any(isinstance(x, ast.Continue) for x in ast.walk(node))
    if walks and "СВЕРКА" not in src:
        flag(f, "⚠НЕТ-СВЕРКИ", "обходит список и пропускает листы, но не печатает «обработано + "
                               "пропущено против длины списка» (образец `_pool_oracle.py`)")

    # 4. пиннинг пути трассировки
    if re.search(r"\bpipe_run\(|pipeline\.run\(|from auto\.pipeline import run", src):
        if not re.search(r"(cv|cfg\.cv)\.seq_model\s*=", src):
            flag(f, "⛔НЕ-ПИННИТ-SEQ", "гоняет пайплайн, но не присваивает cv.seq_model — путь "
                                       "трассировки унаследуется от машины (ловушка `_slot_prod_ab.py`)")

print(f"\n{'='*80}")
if bad:
    print(f"ЗАМЕЧАНИЙ У {len(bad)} СТЕНДОВ ИЗ {len(files) if a.all else len(ACTIVE)}:")
    for f in sorted(bad):
        print(f"\n  {f}")
        for m in bad[f]:
            print(f"      {m}")
else:
    print("★ ЗАМЕЧАНИЙ НЕТ")
print(f"{'='*80}")
print(f"ИТОГ: указатель {'⛔ РАСХОДИТСЯ' if (missing or extra) else '★ СОШЁЛСЯ'} "
      f"({len(files)} файлов); стендов с замечаниями {len(bad)}")
sys.exit(1 if (missing or extra or bad) else 0)

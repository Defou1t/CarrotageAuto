r"""_intake_tier.py — ЗАПИСАТЬ СОРТ ЭТАЛОНА В САМ НАБОР, А НЕ ДЕРЖАТЬ ЕГО В ГОЛОВЕ.

ЗАЧЕМ (archive/docs/INTAKE.md, шаг 2). Сорт **A** — чат Богдана: это то, что сдавалось заказчику после проверки
экспертом. Сорт **B** — группа и телеграм: «почти правильное», возможны ошибки оцифровки, имён
методов, шкал axis/depth. Учить можно на обоих, МЕРИТЬ — только на A (память `carrotage-truth-tiers`).
Если сорт нигде не записан, это правило существует только в переписке и будет нарушено первым же
прогоном, который «взял всё, что лежит в наборе».

⚠⚠ ПОЧЕМУ ЭТО НЕ ПРОСТО «ПОСМОТРЕТЬ, ОТКУДА ФАЙЛ». Раскладка шла ЖЁСТКИМИ ССЫЛКАМИ, а потом
`_intake_clean.py` удалил исходные имена, а `_intake_verify.py` — архивы. Путь источника у файла в
наборе больше не спросить: инод совпадает лишь у 156 файлов из 6284. Провенанс приходится
восстанавливать по следам:
  ПО ИМЕНИ (сильное)   — `cleaned*.txt` хранят полные пути 15 285 удалённых исходных имён;
  ПО СКВАЖИНЕ (слабое) — если имя не нашлось, берётся сорт скважины целиком.
⚠ Слабое доказательство помечается в отчёте отдельно: у 43 скважин сорта A и B СМЕШАНЫ (archive/docs/INTAKE.md
утверждал «источники не пересекаются» — это верно про имена файлов, но не про скважины), поэтому
вывод по скважине там ненадёжен и такие листы честнее считать B.

Результат — `<sorted>\_tier.tsv`: лист, сорт, сила доказательства. Имя с `_` в начале, поэтому
`_intake_pair.py` считает его служебным и не принимает за скважину.

  <python> _intake_tier.py [--sorted ...\intake\sorted]
"""
import sys, re, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--sorted", default=r"E:\Carrotagki_auto\intake\sorted")
ap.add_argument("--logs", nargs="+", default=[r"E:\Carrotagki_auto\intake\cleaned.txt",
                                              r"E:\Carrotagki_auto\intake\cleaned2.txt"])
ap.add_argument("--live", nargs="+", default=[r"E:\Carrotagki_auto\from Bodya",
                                              r"E:\Carrotagki_auto\intake\unp_bodya",
                                              r"E:\Carrotagki_auto\from group",
                                              r"E:\Carrotagki_auto\from telegram",
                                              r"E:\Carrotagki_auto\New_files"])
ap.add_argument("--out", default="")
a = ap.parse_args()
S = Path(a.sorted)
OUT = Path(a.out) if a.out else S / "_tier.tsv"

RE_DATE = re.compile(r"^\d{4}[._\-]\d{2}[._\-]\d{2}"
                     r"(?:\s*-\s*(?:\d{4}[._\-])?\d{2}[._\-]\d{2})?[_\-]+(.+)$")
RE_WELL = re.compile(r"^([A-Za-zА-Яа-я]+(?:[_\-][A-Za-zА-Яа-я]+)*)[_\-]?(\d{1,4})(?:[_\-]|$)")
RE_DUP = re.compile(r"^(.*?)\s*\((\d+)\)$")


def well_of(stem):
    s = stem
    m = RE_DATE.match(s)
    if m:
        s = m.group(1)
    m = RE_WELL.match(s)
    return f"{m.group(1)}_{m.group(2)}" if m else None


def sheet_of(stem):
    m = RE_DUP.match(stem)
    return m.group(1) if m else stem


def tier_of(path):
    s = str(path).lower()
    if "from bodya" in s or "unp_bodya" in s:
        return "A"
    if "from group" in s or "from telegram" in s or "new_files" in s:
        return "B"
    return None


name2, well2 = defaultdict(set), defaultdict(set)
for f in a.logs:
    p = Path(f)
    if not p.is_file():
        continue
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split(": ", 1)[-1].strip()
        if not line[1:3] == ":\\":
            continue
        t = tier_of(line)
        if not t:
            continue
        q = Path(line)
        name2[q.name.lower()].add(t)
        w = well_of(q.stem)
        if w:
            well2[w].add(t)
for d in a.live:
    p = Path(d)
    if not p.is_dir():
        continue
    for q in p.rglob("*"):
        if not q.is_file():
            continue
        t = tier_of(q)
        name2[q.name.lower()].add(t)
        w = well_of(q.stem)
        if w:
            well2[w].add(t)
mixed = {w for w, t in well2.items() if len(t) > 1}
print(f"карта имён {len(name2)}, карта скважин {len(well2)}, из них со смешанным сортом {len(mixed)}")

rows, stat = {}, defaultdict(int)
for f in S.rglob("*"):
    if not f.is_file() or f.parent == S:
        continue
    well = f.relative_to(S).parts[0]
    if well.startswith("_"):
        continue
    sh = sheet_of(f.stem)
    t = name2.get(f.name.lower())
    if t:
        # ⚠ Имя, встреченное в обоих источниках — это не «A и B сразу», а неопределённость.
        # Понижаем до B: завысить сорт опаснее, чем занизить (замер на B врёт молча).
        tier, how = ("A" if t == {"A"} else "B"), "имя"
    elif well in well2 and well not in mixed:
        tier, how = next(iter(well2[well])), "скважина"
    else:
        tier, how = "B", "по умолчанию"
    old = rows.get(sh)
    # У листа несколько файлов; сорт листа — САМЫЙ СЛАБЫЙ из его файлов.
    if old is None or (old[1] == "A" and tier == "B"):
        rows[sh] = (well, tier, how)
for sh, (w, t, how) in rows.items():
    stat[f"{t} ({how})"] += 1
OUT.write_text("sheet\twell\ttier\tevidence\n"
               + "\n".join(f"{sh}\t{w}\t{t}\t{how}" for sh, (w, t, how) in sorted(rows.items())),
               encoding="utf-8")
print(f"\nлистов в наборе: {len(rows)}")
for k, v in sorted(stat.items()):
    print(f"  {k:<24}{v:>6}")
na = sum(v for k, v in stat.items() if k.startswith("A"))
print(f"\n★ ЗАМЕРНАЯ ЧАСТЬ (сорт A): {na} листов; обучающая — все {len(rows)}")
print(f"→ {OUT}")

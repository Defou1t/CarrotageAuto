r"""_xlsx_check.py — СВЕРКА ОТЧЁТНОЙ ТАБЛИЦЫ (Додаток) С РАЗМЕТКОЙ nlgx.

ЗАЧЕМ (02.08). Эталон бывает двух сортов ([[carrotage-truth-tiers]]), и у сорта B возможны
**неверные имена методов**. Это опаснее, чем кажется: имя метода определяет СЛОТ, а отбор трассы под
слот — сегодняшнее узкое место (§6.103). Ошибка в имени систематически штрафует нас за правильную
работу, и по одной только разметке её не увидеть.

★ Таблицы `Додаток` (xlsx, передавались заказчику) дают НЕЗАВИСИМЫЙ перечень: `назва LAS | крива | м`
— какие кривые есть на листе и какой у них метраж. Это второй источник имён, и расхождение с nlgx
означает ошибку в одном из двух. Здесь оно ищется и печатается.

ЧТО СЧИТАЕТСЯ:
  нет в nlgx   — таблица обещает кривую, в разметке её нет (недооцифровано либо имя разошлось);
  нет в xlsx   — в разметке есть кривая, которой нет в отчёте (лишняя либо переименована);
  метраж       — |длина по таблице − длина по разметке| в метрах, если у листа есть шкала глубин.
⚠⚠ СОГЛАШЕНИЯ ИМЁН РАЗНЫЕ, И ЭТО ВЫЯСНЕНО ЗАМЕРОМ, А НЕ ДОГАДКОЙ. Первый прогон без нормализации дал
0 совпадений из 114 при полном совпадении имён ЛИСТОВ — значит расхождение систематическое. Разбор:
в nlgx имя вида `GZ11 DA1 SA1` = МЕТОД + порядковый номер экземпляра, в таблице тот же метод записан
как `GZ1`. То есть отличается ровно ОДНА последняя цифра: `GZ11`→`GZ1`, `OGZ1`→`OGZ`, `SP1`→`SP`.
⚠ Штатный `meta.mnem_root` здесь НЕ годится: он срезает ВСЕ хвостовые цифры (`GZ11`→`GZ`) и теряет
номер зонда, а в БКЗ `GZ1`…`GZ5` — РАЗНЫЕ зонды, их нельзя сливать. Снимается ровно один символ.

  python _xlsx_check.py --xlsx <файл.xlsx> --wlg <каталог с nlgx>
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
from collections import defaultdict
import openpyxl
from extract_nlgx import extract, NULL

ap = argparse.ArgumentParser()
ap.add_argument("--xlsx", required=True)
ap.add_argument("--wlg", required=True, help="каталог с .nlgx этой скважины")
ap.add_argument("--quiet", action="store_true", help="печатать только сводку")
a = ap.parse_args()

# ── таблица ───────────────────────────────────────────────────────────────────────────────────
want = defaultdict(dict)                       # stem листа → {кривая: метраж}
wb = openpyxl.load_workbook(a.xlsx, read_only=True, data_only=True)
for ws in wb.worksheets:
    for r in ws.iter_rows(values_only=True):
        cells = [c for c in r if c is not None]
        if len(cells) < 3:
            continue
        las, curve, m = r[0], r[1], r[2]
        if not isinstance(las, str) or not isinstance(curve, str):
            continue
        if las.strip().lower().startswith("назва"):     # шапка
            continue
        try:
            m = float(m)
        except (TypeError, ValueError):
            continue
        want[las.strip()][curve.strip()] = m
print(f"таблица: листов {len(want)}, кривых {sum(len(v) for v in want.values())}")

# ── разметка ──────────────────────────────────────────────────────────────────────────────────
got = {}
for p in sorted(Path(a.wlg).glob("*.nlgx")):
    if p.stem.endswith("_auto"):
        continue
    try:
        G = extract(str(p))
    except Exception as e:
        print(f"  ⛔ {p.name[:50]}: {type(e).__name__}"); continue
    cur = {}
    for c in G["curves"]:
        n = sum(1 for x in c["xs"] if x != NULL)
        if n < 50:
            continue
        tok = c["name"].split()[0]
        if tok.rstrip("0123456789").upper() == "DA":    # ось глубин, кривой не является
            continue
        # снять ПОРЯДКОВЫЙ НОМЕР ЭКЗЕМПЛЯРА — ровно один хвостовой символ, если он цифра
        key = tok[:-1] if len(tok) > 1 and tok[-1].isdigit() else tok
        cur[key] = n
    got[p.stem] = cur
print(f"разметка: листов {len(got)}, кривых {sum(len(v) for v in got.values())}")

# ── сверка ────────────────────────────────────────────────────────────────────────────────────
matched, only_x, only_n, no_sheet = 0, [], [], []
for las, curves in sorted(want.items()):
    g = got.get(las)
    if g is None:                              # имя листа в таблице не нашлось среди nlgx
        no_sheet.append(las); continue
    for cn in curves:
        if cn in g:
            matched += 1
        else:
            only_x.append((las, cn))
    for cn in g:
        if cn not in curves:
            only_n.append((las, cn))

print(f"\n{'★ совпало имя кривой':<34}{matched:>6}")
print(f"{'⚠ есть в таблице, НЕТ в nlgx':<34}{len(only_x):>6}")
print(f"{'⚠ есть в nlgx, НЕТ в таблице':<34}{len(only_n):>6}")
print(f"{'· лист таблицы не найден среди nlgx':<34}{len(no_sheet):>6}")
if not a.quiet:
    for tag, lst in (("есть в таблице, нет в разметке", only_x),
                     ("есть в разметке, нет в таблице", only_n)):
        if lst:
            print(f"\n{tag}:")
            for las, cn in lst[:15]:
                print(f"  {las[:46]:<48} {cn}")
            if len(lst) > 15:
                print(f"  … и ещё {len(lst)-15}")
    if no_sheet:
        print("\nлисты таблицы без nlgx:")
        for s in no_sheet[:10]:
            print(f"  {s[:64]}")
        if len(no_sheet) > 10:
            print(f"  … и ещё {len(no_sheet)-10}")

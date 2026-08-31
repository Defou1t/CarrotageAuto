r"""_intake_pair.py — ОСТАВИТЬ В НАБОРАХ ТОЛЬКО ПАРЫ «СКАН + РАЗМЕТКА», ОСТАЛЬНОЕ УВЕСТИ В unsorted.

ЗАЧЕМ (запрос Эдуарда 08.08). `_intake_sort.py` раскладывает комплект, если есть ХОТЬ ЧТО-ТО
полезное: `if "wlg" not in got and "img" not in got: continue`. Поэтому в `intake\sorted` рядом с
рабочими комплектами лежат половинки — скан без разметки и разметка без скана. Учиться и мериться
на них нельзя: без `.nlgx` нечем мерить, без изображения нечего распознавать.

ПАРА = один лист, у которого есть И скан (`.jpg/.jpeg/.tif/.tiff/.png`), И разметка эксперта
(`.nlgx`). Лист опознаётся ПО ИМЕНИ, и только по имени: связка «этот скан к этой разметке» иначе
недоказуема, а угаданная пара тихо портит и обучение, и замер. `.bck` и `.las` — спутники листа:
едут туда же, куда его пара, и в одиночку листа не образуют.

⚠ ХВОСТ `(1)`, `(2)` — ЭТО НЕ ЧАСТЬ ИМЕНИ ЛИСТА, а след копирования. Правило Эдуарда (08.08):
имя листа — то, что до хвоста; если файла с чистым именем нет, а есть только `… (1)`, то он и есть
правильный вариант и переименовывается в чистое имя. Когда вариантов несколько, берётся младший
номер (чистое имя старше `(1)`, `(1)` старше `(2)`), остальные уезжают в `unsorted` как
альтернативные версии — не удаляются, потому что расхождение версий здесь норма (§ `_intake_sort`:
разметка после правки эксперта уезжает в резервную копию, и какая версия свежее — решает человек).

⚠ ПЕРЕМЕЩЕНИЕ, А НЕ УДАЛЕНИЕ. Всё непарное уезжает в `intake\unsorted\<Скважина>\{wlg,img,las}` —
структура сохраняется, откат = перенести обратно.
⚠ Файлы — ЖЁСТКИЕ ССЫЛКИ на источники (`_intake_sort.py --mode link`). Перенос в пределах тома
ссылку не рвёт: «объём» в отчёте — объём данных, а не освобождаемое место. Поэтому и удалять
непарное смысла нет: байты держит исходник в `from telegram` / `unpacked`.

★ ПО УМОЛЧАНИЮ — СУХОЙ ПРОГОН. Ничего не двигается без `--apply`.

  <python> _intake_pair.py [--apply] [--sorted ...\intake\sorted] [--unsorted ...\intake\unsorted]
"""
import sys, os, re, argparse, shutil
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import defaultdict

IMG = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
KIND = {".nlgx": "nlgx", ".bck": "bck", ".las": "las"}
FOLDER = {"nlgx": "wlg", "bck": "wlg", "las": "las", "img": "img"}
RE_DUP = re.compile(r"^(.*?)\s*\((\d+)\)$")          # `имя (2)` → («имя», 2)
# ⚠ Имя от сканера: скважины в нём нет вообще, привязать такой скан к разметке уже нечем.
# Отдельная корзина в отчёте — чтобы было видно, сколько сканов потеряно безвозвратно, а не
# «просто не хватило разметки». В `unsorted` они едут наравне с прочим непарным.
RE_SCANNER = re.compile(r"^(?:\d{8,}[_\-]?\d*|[A-Z0-9]{6,8}|photo[_\-]?\d{4}|IMG[_\-]?\d+|"
                        r"Screenshot[_\-]?\d*|image[_\-]?\d+)$", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("--sorted", default=r"E:\Carrotagki_auto\intake\sorted")
ap.add_argument("--unsorted", default=r"E:\Carrotagki_auto\intake\unsorted")
ap.add_argument("--apply", action="store_true", help="реально переносить (без флага — только отчёт)")
ap.add_argument("--report", default=r"E:\Carrotagki_auto\intake\unpaired.txt")
a = ap.parse_args()

S, U = Path(a.sorted), Path(a.unsorted)
if not S.is_dir():
    sys.exit(f"нет каталога наборов {S}")


def split_dup(stem):
    m = RE_DUP.match(stem)
    return (m.group(1), int(m.group(2))) if m else (stem, 0)


# ── собрать листы из ОБЕИХ папок: (скважина, чистое имя) → вид → [(номер копии, файл)] ─────────
# ⚠ Смотрим и в `unsorted`: прошлый прогон судил по полному имени, и лист, у которого скан звался
# `X.jpg`, а разметка `X (1).nlgx`, был разорван и уехал туда целиком. Теперь он снова пара.
sets, alien, empty = defaultdict(lambda: defaultdict(list)), [], []
for root in (S, U):
    if not root.is_dir():
        continue
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in IMG and ext not in KIND:
            alien.append(p); continue
        # ⚠ ПУСТОЙ ФАЙЛ — НЕ ФАЙЛ. Два скана пришли нулевой длины ещё из источника (даты 2019/2020),
        # и по имени они образовывали «пару» с настоящей разметкой: лист числился рабочим, а
        # распознавать в нём нечего. Считаем такие файлы отсутствующими.
        if p.stat().st_size == 0:
            empty.append(p); continue
        well = p.relative_to(root).parts[0]
        # ⚠ Служебные каталоги приёмки (имя с `_`) — НЕ СКВАЖИНЫ. `unsorted\_orphan` это отстойник
        # спасённых файлов, разложенных отдельно `_intake_sort.py`; если считать его скважиной,
        # найденные внутри пары уезжают в `sorted\_orphan\…` и теряют имя скважины (поймано вживую:
        # 300 файлов). Отстойник разбирается раскладкой, а не этим шагом.
        if well.startswith("_"):
            continue
        base, idx = split_dup(p.stem)
        sets[(well, base)]["img" if ext in IMG else KIND[ext]].append((idx, p))

keep, move, why = [], [], defaultdict(list)
renamed = []
for (well, base), got in sorted(sets.items()):
    if "img" in got and "nlgx" in got:
        for k, lst in got.items():
            lst.sort(key=lambda q: (q[0], str(q[1])))
            idx, p = lst[0]
            tgt = S / well / FOLDER[k] / (base + p.suffix)
            keep.append((p, tgt))
            if idx:
                renamed.append((p, tgt))
            for _, q in lst[1:]:
                move.append(q); why["альтернативная версия того же листа"].append(q)
        continue
    files = [q for lst in got.values() for _, q in lst]
    move += files
    if "img" in got and RE_SCANNER.match(base):
        why["скан с именем от сканера — привязать не к чему"] += files
    elif "img" in got:
        why["скан без разметки (.nlgx нет)"] += files
    elif "nlgx" in got:
        why["разметка без скана (есть .nlgx, нет изображения)"] += files
    elif "bck" in got:
        why["только резервная копия разметки (.bck)"] += files
    else:
        why["только значения (.las)"] += files
if alien:
    move += alien; why["посторонний формат"] += alien
if empty:
    move += empty; why["файл нулевой длины (битый в источнике)"] += empty

GB = 1 << 30
vol = lambda lst: sum(p.stat().st_size for p in lst) / GB
pairs = sum(1 for (w, b), g in sets.items() if "img" in g and "nlgx" in g)
back = [p for p, _ in keep if U in p.parents]
wells_keep = {t.relative_to(S).parts[0] for _, t in keep}

print(f"листов найдено: {len(sets)}   из них ПАР (скан + разметка): {pairs}")
print(f"\n{'★ в sorted (пары со спутниками)':<52}{len(keep):>7} файлов   {vol([p for p, _ in keep]):>6.1f} ГБ")
print(f"{'  из них вернётся из unsorted':<52}{len(back):>7}")
print(f"{'  из них переименовано из `имя (N)` в `имя`':<52}{len(renamed):>7}")
for k, lst in sorted(why.items(), key=lambda q: -len(q[1])):
    print(f"{'→ в unsorted: ' + k:<52}{len(lst):>7} файлов   {vol(lst):>6.1f} ГБ")
print(f"{'→ в unsorted ВСЕГО':<52}{len(move):>7} файлов   {vol(move):>6.1f} ГБ")
print(f"\nскважин с парами: {len(wells_keep)}")
for p, t in renamed[:8]:
    print(f"  переименование: {p.name[:58]} → {t.name[:58]}")
if len(renamed) > 8:
    print(f"  … и ещё {len(renamed)-8}")

Path(a.report).write_text("\n".join(
    [f"# НЕПАРНОЕ → {U}", ""]
    + [line for k, lst in sorted(why.items())
       for line in [f"## {k} — {len(lst)}"] + [str(p) for p in lst] + [""]]
    + ["=" * 78, f"# ПЕРЕИМЕНОВАНО `имя (N)` → `имя` — {len(renamed)}", ""]
    + [f"{p.name}  →  {t.name}" for p, t in renamed]), encoding="utf-8")
print(f"подробности → {a.report}")

if not a.apply:
    print("\n★ Это СУХОЙ ПРОГОН. Ничего не перенесено. Перенести: тот же вызов с --apply")
    sys.exit()


def relocate(p, tgt):
    """Перенос с разведением столкновений: затирать существующий файл нельзя ни при каких условиях."""
    if p.resolve() == tgt.resolve():
        return 0
    tgt.parent.mkdir(parents=True, exist_ok=True)
    if tgt.exists():
        i = 2
        while (tgt.parent / f"{tgt.stem} ({i}){tgt.suffix}").exists():
            i += 1
        tgt = tgt.parent / f"{tgt.stem} ({i}){tgt.suffix}"
    shutil.move(str(p), str(tgt))
    return 1


# ⚠ Сначала уводим непарное, потом ставим пары на место: иначе переименование `X (1).nlgx` → `X.nlgx`
# может столкнуться с ещё не уехавшим чужим `X.nlgx` и получить лишний суффикс.
moved = kept = failed = 0
for p in move:
    well = p.relative_to(S if S in p.parents else U).parts[0]
    k = "img" if p.suffix.lower() in IMG else KIND[p.suffix.lower()]
    try:
        moved += relocate(p, U / well / FOLDER[k] / p.name)
    except OSError as e:
        failed += 1
        if failed <= 5:
            print(f"  ⛔ {p.name[:50]}: {e}")
for p, t in keep:
    try:
        kept += relocate(p, t)
    except OSError as e:
        failed += 1
        if failed <= 5:
            print(f"  ⛔ {p.name[:50]}: {e}")
for root in (S, U):
    for d in sorted(root.rglob("*"), key=lambda q: -len(q.parts)):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass
print(f"\n★ перенесено в unsorted: {moved}; поставлено/переименовано в sorted: {kept}"
      + (f"; ⛔ НЕ УДАЛОСЬ {failed}" if failed else ""))

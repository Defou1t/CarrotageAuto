r"""_intake_merge.py — ПЕРЕНОС ГОТОВЫХ ПАР ИЗ ПРИЁМКИ В БОЕВОЙ АРХИВ (шаг 5 INTAKE.md).

ЗАЧЕМ. В `intake\sorted` лежат 1580 пар «скан + разметка», из них 1567 в боевом архиве отсутствуют.
Это удвоение корпуса: 1181 → 2748 листов. Раскладка и отбор пар сделаны, сорта проставлены
(`_tier.tsv`), архивы снесены — остался перенос, и он делается ОТДЕЛЬНЫМ осознанным шагом
(INTAKE.md:86), потому что боевой архив читают все замеры ветки.

⚠⚠ ТОМ ОДИН. `F:\nds\projects\Archive` — это ТОЧКА СОЕДИНЕНИЯ на `E:\Carrotagki_auto\projects\Archive`
(проверено: robocopy `/MOVE` 28.06 увёл данные на E, после чего созданы junction'ы; регулярной задачи
зеркала нет). Значит приёмка и архив физически на одном томе E, и перенос делается ЖЁСТКИМИ
ССЫЛКАМИ: мгновенно, ноль байт, исходник цел. Предупреждение INTAKE.md:28 про «robocopy увезёт что
угодно» относится к той разовой миграции и сегодня неактуально.

⚠ ПЕРЕИМЕНОВАНИЯ — ТОЛЬКО ДОКАЗАННЫЕ. Одна скважина под разными написаниями ломает `--cap N` на
скважину и разрез по семействам. Но канонизировать имена ПО ЗАГОЛОВКУ LAS НЕЛЬЗЯ: у 51 скважины
заголовок расходится с папкой, и часть расхождений — ошибки заполнения (`Chutivska_65` → в заголовке
`CHUTIVSKA_007`, `Mashivska_7` → `MASHIVSKA_029`). Поэтому LAS используется как проверка КОНКРЕТНОЙ
гипотезы, а таблица ниже — закрытая, с доказательством на каждую строку.

⚠ ЗАЩИТА ЗАМЕРА. `_trace_prod_ab.py` строит выборку по пулам и сопоставляет с архивом ПО ИМЕНИ
ФАЙЛА, а скважину берёт из имени папки. Если после переноса какой-то файл сменит скважину, отбор
`--cap` поедет и перезапущенный шард разойдётся с остальными. Скрипт это проверяет и без
`--allow-flip` отказывается работать.

★ ПО УМОЛЧАНИЮ — СУХОЙ ПРОГОН.

  <python> _intake_merge.py [--apply]
"""
import sys, os, argparse, shutil, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import defaultdict

# ── таблица переименований: слева папка приёмки, справа канон в архиве ────────────────────────
# `Stanulska_2`   → в заголовке LAS `Stanulska_002`, и такая папка в архиве уже есть;
# `YULIV_007`     → в заголовке LAS `YULIIV_007` (пропущена буква в имени папки), папка в архиве есть;
# `Rozpashnizka_81` → в заголовке LAS `ROZPASHNIVSKA_081`, рядом лежит `Rozpashnivska_81`;
# `БВп_N`         → `Bilch_Volyts_N`: «БВп» = сокращение «Білче-Волиця» (зафиксировано в
#                   `_intake_sort.py`), номер в заголовке LAS совпадает (`#151`), пересечения имён
#                   листов между папками нет ни одного.
# ⚠ НЕ переименовываются: `Уп_198`/`Уп_199` — в заголовке `#198У`/`#199У`, суффикс У = другое
#   месторождение, и совпадение номера с `Bilch_Volyts_198/199` (`#198`/`#199`) СЛУЧАЙНО;
#   `БВп_166` против `BVU_BV_166` — заголовки `#166` и `BVU #BV166`, тождество не доказано.
RENAME = {
    "Stanulska_2": "Stanulska_002",
    "YULIV_007": "YULIIV_007",
    "Rozpashnizka_81": "Rozpashnivska_81",
    "БВп_151": "Bilch_Volyts_151", "БВп_152": "Bilch_Volyts_152", "БВп_153": "Bilch_Volyts_153",
    "БВп_160": "Bilch_Volyts_160", "БВп_202": "Bilch_Volyts_202", "БВп_204": "Bilch_Volyts_204",
}
# ⚠ `Semeguniv_1` целиком дублирует `F:\nds\projects\Semeguniv_001` (все 8 листов совпадают по
# имени), а замеры берут её оттуда под ярлыком `Semeguniv`. Перенос создал бы вторую папку и увёл
# бы один лист выборки A/B в другую скважину. Новых данных в ней нет — не переносим.
EXCLUDE = {"Semeguniv_1": "дубль F:\\nds\\projects\\Semeguniv_001"}

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=r"E:\Carrotagki_auto\intake\sorted")
ap.add_argument("--dst", default=r"E:\Carrotagki_auto\projects\Archive")
ap.add_argument("--extra-wells", nargs="+", default=[r"F:\nds\projects\Semeguniv_001"],
                help="каталоги скважин ВНЕ архива, которые замеры тоже читают")
ap.add_argument("--apply", action="store_true")
ap.add_argument("--allow-flip", action="store_true", help="разрешить смену скважины у файла")
ap.add_argument("--manifest", default="")
a = ap.parse_args()
S, D = Path(a.src), Path(a.dst)
if not S.is_dir() or not D.is_dir():
    sys.exit(f"нет каталога: {S if not S.is_dir() else D}")


def sha1(p, n=1 << 20):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()


def canon(w):
    return RENAME.get(w, w)


# ── план ──────────────────────────────────────────────────────────────────────────────────────
plan, skip_same, conflict, skipped_wells = [], [], [], defaultdict(int)
for wd in sorted(S.iterdir()):
    if not wd.is_dir() or wd.name.startswith("_"):
        continue
    if wd.name in EXCLUDE:
        skipped_wells[EXCLUDE[wd.name]] += len(list(wd.rglob("*")))
        continue
    for p in wd.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(wd)                       # {img,las,wlg}\имя
        t = D / canon(wd.name) / rel
        if t.exists():
            if t.stat().st_size == p.stat().st_size and sha1(t) == sha1(p):
                skip_same.append(p)
            else:
                conflict.append((p, t))
            continue
        plan.append((p, t))

# ── защита замера: не должен смениться ярлык скважины ни у одного имени ───────────────────────
WELL = {}
for wlg in D.glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WELL[q.name] = wlg.parent.name
for extra in a.extra_wells:
    e = Path(extra)
    for q in (e / "wlg").glob("*.nlgx"):
        WELL.setdefault(q.name, e.name.rstrip("0123456789_") or e.name)
flip = []
for p, t in plan:
    if p.suffix.lower() == ".nlgx":
        old = WELL.get(p.name)
        if old and old != t.parent.parent.name:
            flip.append((p.name, old, t.parent.parent.name))

GB = 1 << 30
vol = lambda lst: sum(p.stat().st_size for p, *_ in lst) / GB
new_wells = sorted({t.parent.parent.name for _, t in plan} - {d.name for d in D.iterdir() if d.is_dir()})
sheets = {t.stem for _, t in plan if t.suffix.lower() == ".nlgx"}
print(f"{'★ перенести':<44}{len(plan):>7} файлов   {vol(plan):>6.1f} ГБ")
print(f"{'  из них листов (nlgx)':<44}{len(sheets):>7}")
print(f"{'  новых скважин в архиве':<44}{len(new_wells):>7}")
print(f"{'· уже есть, совпало байт-в-байт':<44}{len(skip_same):>7}")
print(f"{'⚠ есть под тем же именем, ДРУГОЕ содержимое':<44}{len(conflict):>7}   не трогаю")
for k, v in skipped_wells.items():
    print(f"{'· пропущена скважина: ' + k:<44}{v:>7} файлов")
for p, t in conflict[:6]:
    print(f"   конфликт: {t.parent.parent.name}/{p.name[:52]}")
if flip:
    print(f"\n⛔ СМЕНА СКВАЖИНЫ У {len(flip)} ФАЙЛОВ — выборка A/B поедет:")
    for n, o, w in flip[:8]:
        print(f"   {n[:52]}  {o} → {w}")

if flip and not a.allow_flip:
    sys.exit("\n⛔ ОСТАНОВ: перенос сменил бы скважину у файлов выборки. "
             "Разобрать написания имён или запустить с --allow-flip.")
if not a.apply:
    print("\n★ Это СУХОЙ ПРОГОН. Ничего не перенесено. Перенести: тот же вызов с --apply")
    sys.exit()

linked = copied = failed = 0
for p, t in plan:
    t.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(p, t); linked += 1                     # один том ⇒ ноль байт
    except OSError:
        try:
            shutil.copy2(p, t); copied += 1
        except OSError as e:
            failed += 1
            if failed <= 5:
                print(f"  ⛔ {p.name[:50]}: {e}")
print(f"\n★ перенесено: {linked} ссылок, {copied} копий"
      + (f", ⛔ НЕ УДАЛОСЬ {failed}" if failed else ""))

# ── манифест: что и с каким сортом эталона добавлено ──────────────────────────────────────────
tier = {}
tf = S / "_tier.tsv"
if tf.is_file():
    for line in tf.read_text(encoding="utf-8").splitlines()[1:]:
        c = line.split("\t")
        if len(c) >= 3:
            tier[c[0]] = c[2]
man = Path(a.manifest) if a.manifest else D / "_intake_merged.tsv"
rows = sorted({(t.stem, t.parent.parent.name, tier.get(t.stem, "?"))
               for _, t in plan if t.suffix.lower() == ".nlgx"})
man.write_text("sheet\twell\ttier\n" + "\n".join("\t".join(r) for r in rows), encoding="utf-8")
nA = sum(1 for r in rows if r[2] == "A")
print(f"★ манифест: {len(rows)} листов (сорт A {nA}, B {len(rows)-nA}) → {man}")
print("  ⚠ замерять по-прежнему только на сорте A; общие счётчики архива после этого шага "
      "несравнимы с прежними числами роадмапа")

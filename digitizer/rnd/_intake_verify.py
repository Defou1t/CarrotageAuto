r"""_intake_verify.py — ДОКАЗАТЬ, ЧТО АРХИВЫ И РАСПАКОВКА БОЛЬШЕ НЕ НУЖНЫ, И ТОЛЬКО ПОТОМ СНОСИТЬ.

ЗАЧЕМ (запрос Эдуарда 08.08). После `_intake_sort.py` + `_intake_pair.py` рабочие данные лежат в
`intake\{sorted,unsorted}`. Источники держат 254 ГБ архивов и 86 ГБ распаковки. Освободить их надо,
но удаление необратимо, поэтому каждый шаг требует доказательства.

⚠⚠ ГЛАВНОЕ ПРО ЖЁСТКИЕ ССЫЛКИ. `_intake_sort.py --mode link` раскладывал НЕ КОПИИ, а вторые имена
тех же файлов. Значит:
  • удаление `unpacked` НЕ УДАЛЯЕТ данные, у которых есть имя в `sorted`/`unsorted` — байты держит
    вторая ссылка; освободится только то, что никуда не разложено;
  • и наоборот: файл из `unpacked`, которого нет ни в одном наборе, исчезнет НАВСЕГДА.
Поэтому распаковка делится по индексу файла на томе (st_ino): «есть ссылка» и «сирота».

ДОКАЗАТЕЛЬСТВО ДЛЯ АРХИВА — существует непустой каталог распаковки с его именем (та же конвенция,
что в `_intake_sort.unpack_all`: `unpacked\<имя архива без расширения>`, у многотомных — по основе
до `.partN`). Нет каталога или он пуст — архив НЕ УДАЛЯЕТСЯ и попадает в отчёт.

★ ПО УМОЛЧАНИЮ — ТОЛЬКО ОТЧЁТ. Шаги включаются по одному и в этом порядке:
  --rescue          сироты полезных форматов (img/nlgx/bck/las) → `unsorted\_orphan\<архив>\`
  --drop-archives   удалить доказанные архивы
  --drop-unpacked   удалить каталог распаковки (после --rescue!)

  <python> _intake_verify.py [--rescue] [--drop-archives] [--drop-unpacked]
"""
import sys, os, re, argparse, shutil
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import defaultdict

IMG = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
USEFUL = IMG | {".nlgx", ".bck", ".las"}
ARCH = {".zip", ".rar"}
RE_PART = re.compile(r"^(.*)\.part0*(\d+)$", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", default=[
    r"E:\Carrotagki_auto\from Bodya", r"E:\Carrotagki_auto\from group",
    r"E:\Carrotagki_auto\from telegram", r"E:\Carrotagki_auto\New_files",
    r"E:\Carrotagki_auto\intake\unpacked"])
ap.add_argument("--unpacked", default=r"E:\Carrotagki_auto\intake\unpacked")
ap.add_argument("--sets", nargs="+", default=[r"E:\Carrotagki_auto\intake\sorted",
                                              r"E:\Carrotagki_auto\intake\unsorted"])
ap.add_argument("--orphan", default=r"E:\Carrotagki_auto\intake\unsorted\_orphan")
ap.add_argument("--rescue", action="store_true")
ap.add_argument("--drop-archives", action="store_true")
ap.add_argument("--drop-unpacked", action="store_true")
ap.add_argument("--report", default=r"E:\Carrotagki_auto\intake\verify.txt")
a = ap.parse_args()
GB = 1 << 30

# ── что уже разложено: индексы файлов на томе ─────────────────────────────────────────────────
INO = set()
for s in a.sets:
    for p in Path(s).rglob("*"):
        if p.is_file():
            INO.add(p.stat().st_ino)
print(f"в наборах {len(INO)} файлов (по индексу на томе)")

# ── распаковка: ссылка или сирота ─────────────────────────────────────────────────────────────
UNP = Path(a.unpacked)
linked, orphan_useful, orphan_junk = [], [], []
for p in UNP.rglob("*"):
    if not p.is_file():
        continue
    if p.stat().st_ino in INO:
        linked.append(p)
    elif p.suffix.lower() in USEFUL:
        orphan_useful.append(p)
    else:
        orphan_junk.append(p)
vol = lambda lst: sum(p.stat().st_size for p in lst) / GB
print(f"\nРАСПАКОВКА {UNP}")
print(f"{'  есть имя в наборах (удаление байт не тронет)':<52}{len(linked):>7}   {vol(linked):>6.1f} ГБ")
print(f"{'⚠ СИРОТА полезного формата (исчезнет навсегда)':<52}{len(orphan_useful):>7}   {vol(orphan_useful):>6.1f} ГБ")
print(f"{'  сирота прочего формата (pdf/exe/ini/nlg/…)':<52}{len(orphan_junk):>7}   {vol(orphan_junk):>6.1f} ГБ")
if orphan_useful:
    ext = defaultdict(int)
    for p in orphan_useful:
        ext[p.suffix.lower()] += 1
    print("   по расширениям: " + ", ".join(f"{k} {v}" for k, v in sorted(ext.items(), key=lambda q: -q[1])))

# ── архивы: доказан / не доказан ──────────────────────────────────────────────────────────────
proved, unproved, volumes = [], [], []
for s in a.src:
    r = Path(s)
    if not r.is_dir():
        continue
    for p in r.rglob("*"):
        if p.suffix.lower() not in ARCH:
            continue
        mp = RE_PART.match(p.stem)
        base = mp.group(1) if mp else p.stem
        # ⚠ Доказательством служит И каталог спасённых сирот: `--rescue` уводит из распаковки файлы,
        # у которых нет имени в наборах, и каталог `unpacked\<архив>` после этого может опустеть
        # ЦЕЛИКОМ. Без этой второй проверки 345 архивов, чьё содержимое целиком состояло из сирот,
        # объявлялись недоказанными сразу после того, как их содержимое было надёжно сохранено.
        out = UNP / base
        alt = Path(a.orphan) / base
        ok = any(d.is_dir() and any(d.rglob("*")) for d in (out, alt))
        if mp and int(mp.group(2)) != 1:
            volumes.append((p, ok))            # том многотомного: удаляется вместе с первым
        elif ok:
            proved.append(p)
        else:
            unproved.append((p, "нет каталога распаковки" if not out.exists() else "каталог пуст"))
vol_ok = [p for p, ok in volumes if ok]
vol_bad = [p for p, ok in volumes if not ok]
print(f"\nАРХИВЫ")
print(f"{'★ доказан (есть непустая распаковка)':<52}{len(proved):>7}   {vol(proved):>6.1f} ГБ")
print(f"{'★ тома многотомных к доказанным':<52}{len(vol_ok):>7}   {vol(vol_ok):>6.1f} ГБ")
print(f"{'⛔ НЕ ДОКАЗАН — не удаляю':<52}{len(unproved)+len(vol_bad):>7}   {vol([p for p, _ in unproved] + vol_bad):>6.1f} ГБ")
for p, whyn in unproved[:10]:
    print(f"  ⛔ {p.name[:56]}: {whyn}")
if len(unproved) > 10:
    print(f"  … и ещё {len(unproved)-10}")

Path(a.report).write_text("\n".join(
    [f"# НЕ ДОКАЗАННЫЕ АРХИВЫ — {len(unproved)+len(vol_bad)} (не удаляются)"]
    + [f"{whyn}: {p}" for p, whyn in unproved] + [str(p) for p in vol_bad]
    + ["", f"# СИРОТЫ ПОЛЕЗНОГО ФОРМАТА В РАСПАКОВКЕ — {len(orphan_useful)}"]
    + [str(p) for p in orphan_useful]), encoding="utf-8")
print(f"подробности → {a.report}")

# ── шаг 1: спасти сирот ───────────────────────────────────────────────────────────────────────
if a.rescue and orphan_useful:
    O = Path(a.orphan)
    n = 0
    for p in orphan_useful:
        rel = p.relative_to(UNP)
        d = O / rel.parts[0]
        d.mkdir(parents=True, exist_ok=True)
        t = d / p.name
        if t.exists():
            i = 2
            while (d / f"{p.stem} ({i}){p.suffix}").exists():
                i += 1
            t = d / f"{p.stem} ({i}){p.suffix}"
        shutil.move(str(p), str(t)); n += 1
    print(f"\n★ спасено сирот в {O}: {n}")
    orphan_useful = []
elif orphan_useful and a.drop_unpacked:
    sys.exit(f"\n⛔ ОСТАНОВ: {len(orphan_useful)} сирот полезного формата исчезнут навсегда. "
             f"Сначала `--rescue`.")

# ── шаг 2: архивы ─────────────────────────────────────────────────────────────────────────────
if a.drop_archives:
    gone, freed, failed = 0, 0, 0
    for p in proved + vol_ok:
        try:
            sz = p.stat().st_size; p.unlink(); gone += 1; freed += sz
        except OSError as e:
            failed += 1
            if failed <= 5:
                print(f"  ⛔ {p.name[:50]}: {e}")
    print(f"\n★ удалено архивов: {gone}, освобождено {freed/GB:.1f} ГБ"
          + (f", ⛔ НЕ УДАЛОСЬ {failed}" if failed else ""))

# ── шаг 3: распаковка ─────────────────────────────────────────────────────────────────────────
if a.drop_unpacked:
    before = shutil.disk_usage(str(UNP.anchor)).free
    shutil.rmtree(UNP, ignore_errors=True)
    after = shutil.disk_usage(str(UNP.anchor)).free
    print(f"★ каталог распаковки удалён, освободилось {(after-before)/GB:.1f} ГБ "
          f"(остальное держат ссылки из наборов — так и задумано)")

if not (a.rescue or a.drop_archives or a.drop_unpacked):
    print("\n★ Это ОТЧЁТ. Ничего не изменено. Шаги: --rescue → --drop-archives → --drop-unpacked")

# ── пустые каталоги ───────────────────────────────────────────────────────────────────────────
if a.drop_archives or a.rescue:
    for s in a.src:
        for d in sorted(Path(s).rglob("*"), key=lambda q: -len(q.parts)):
            if d.is_dir():
                try:
                    d.rmdir()
                except OSError:
                    pass

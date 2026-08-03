r"""_intake_sort.py — РАЗБОР СВАЛКИ ФАЙЛОВ В СТРУКТУРУ ПРОЕКТА + АУДИТ НЕПОЛНЫХ КОМПЛЕКТОВ.

ЗАЧЕМ (запрос Эдуарда 02.08). Архивы скважин лежат вперемешку (выгрузка чата, общая папка). Нужно
разложить их в рабочую структуру `<Скважина>/{wlg,img,las}` и ОТДЕЛЬНО показать, чего не хватает,
чтобы недостающее искать прицельно, а не наугад.

КОМПЛЕКТ = один лист. Опознаётся по ИМЕНИ БЕЗ РАСШИРЕНИЯ (stem): `.nlgx` (разметка эксперта),
`.jpg/.jpeg/.tif/.tiff/.png` (скан), `.las` (значения), `.bck` (бэкап разметки). В архиве все четыре
файла листа имеют один и тот же stem — на этом и держится сборка.

СКВАЖИНА достаётся из имени, две конвенции (см. карту роадмапа):
  основная  (81%)  `BOGAT_011_AKC_0000-1200_500_1984-09-28_D_1_B_1` → BOGAT_011
  скобочная (19%)  `1966.01.12_Rybal_058_GK_(0012-1395)_NGK_..._500` → Rybal_058
⚠ Что не распозналось — НЕ раскладывается и попадает в отчёт отдельной корзиной. Молча угадывать
скважину нельзя: ошибка здесь тихо портит и обучение, и замер по семействам.

★ ПО УМОЛЧАНИЮ НИЧЕГО НЕ ПЕРЕМЕЩАЕТСЯ. Сначала отчёт, `--apply` — копирование (не перенос: исходная
свалка остаётся нетронутой, откат = удалить целевую папку).

  <python> _intake_sort.py --src <папка…> [--dst F:\nds\projects\Inbox] [--apply]
"""
import sys, argparse, re, shutil, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import defaultdict

IMG = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
# ⚠ `.bck` — ОТДЕЛЬНЫЙ вид, хотя кладётся в тот же `wlg/`. Если свалить его в один слот с `.nlgx`,
# каждый лист, у которого есть оба файла, ложно объявляется конфликтом (поймано вхолостую: 1072 шт).
KIND = {".nlgx": "wlg", ".bck": "bck", ".las": "las"}

ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", required=True, help="папки со свалкой (рекурсивно)")
# ⚠ Приёмка живёт на E, ВНЕ обоих зеркал robocopy. `E:\Carrotagki_auto\projects\Archive` — цель
# задания с флагом `/MOVE` из `F:\nds\projects\Archive`, то есть туда класть рабочие файлы нельзя:
# следующий прогон зеркала их увезёт. Соседний `intake\` ничему не принадлежит и безопасен.
# В боевой `Archive` комплекты переносятся ОТДЕЛЬНЫМ шагом, ПОСЛЕ анализа (`_intake_merge.py`).
ap.add_argument("--dst", default=r"E:\Carrotagki_auto\intake\sorted")
ap.add_argument("--apply", action="store_true", help="реально копировать (без флага — только отчёт)")
ap.add_argument("--report", default="", help="куда положить список неполных комплектов (txt)")
# ⚠ Выгрузка чата содержит rar/zip. Их надо РАСПАКОВАТЬ, иначе комплекты внутри невидимы. Распаковка
# идёт в отдельный каталог рядом с приёмкой и делается один раз (повторный запуск её пропускает).
ap.add_argument("--unpack", default=r"E:\Carrotagki_auto\intake\unpacked",
                help="куда распаковывать rar/zip; пусто = не распаковывать")
# ⚠⚠ Часть присланного УЖЕ ЕСТЬ в боевом архиве. Такие комплекты не должны попадать ни в приёмку,
# ни тем более в обучение второй раз. Сверяем по stem, а совпавшие — ещё и по содержимому.
ap.add_argument("--archive", default=r"F:\nds\projects\Archive",
                help="боевой архив для отсева уже имеющегося; пусто = не сверять")
a = ap.parse_args()

# ── скважина из имени ─────────────────────────────────────────────────────────────────────────
# Дата в начале скобочной конвенции бывает и ДИАПАЗОНОМ: `1965.12.29-1966.01.12_Rybal_058_…`
# и `1966.02.24-11.16_Rybal_057_…`. Одиночная дата покрывала лишь часть — вхолостую 388 файлов
# остались неразобранными именно из-за этого.
RE_DATE = re.compile(r"^\d{4}[.\-]\d{2}[.\-]\d{2}"
                     r"(?:\s*-\s*(?:\d{4}[.\-])?\d{2}[.\-]\d{2})?[_\-]+(.+)$")
# Имя скважины бывает и СОСТАВНЫМ: `Pn_Zavoda_1_AK_…` (вхолостую — 372 файла мимо разбора).
# Буквенных токенов может быть несколько, номер — первый числовой токен после них.
# Разделитель перед номером БЫВАЕТ ОПУЩЕН: архивы зовутся `Andriyashivska11`, `Voloshkivska-10`.
# Без этого папочный запасной вариант не срабатывал там, где он нужнее всего — на сканах с именами
# от сканера, у которых скважина есть ТОЛЬКО в имени архива.
RE_WELL = re.compile(r"^([A-Za-zА-Яа-я]+(?:[_\-][A-Za-zА-Яа-я]+)*)[_\-]?(\d{1,4})(?:[_\-]|$)")
# ⚠ Имена, которые Telegram выдаёт сам (`photo_1198.jpg`), под это правило подпадают случайно и
# были бы приписаны несуществующей скважине «photo_1198». Отсекаем явно.
RE_JUNK = re.compile(r"^(photo|image|video|media|back|section|IMG|DSC)[_\-]?\d*$", re.I)
# ⚠⚠ `_auto` — НАША СОБСТВЕННАЯ ВЫДАЧА, а не разметка эксперта. В корпус её пускать нельзя:
# измеряя себя по своему же файлу, получаем круговую оценку (тот же класс ошибки, что ловит
# `leaked()` в §6.36). Поймано вхолостую: три таких файла числились «эталоном без скана».
RE_AUTO = re.compile(r"_auto\d*$", re.I)
FOLDER = {"wlg": "wlg", "bck": "wlg", "las": "las", "img": "img"}   # .bck лежит рядом с .nlgx


def well_of(stem):
    if RE_JUNK.match(stem):
        return None
    s = stem
    m = RE_DATE.match(s)
    if m:
        s = m.group(1)
    m = RE_WELL.match(s)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}"


def well_of_path(p, roots):
    """Скважина по имени файла, а если не вышло — ПО ИМЕНИ ПАПКИ (архива), в которой он лежит.

    ⚠ ЗАЧЕМ. 427 из 473 неразобранных файлов — сканы с именами от сканера
    (`20191115120359_001.jpg`, `500000AV.JPG`): скважины в имени НЕТ ВООБЩЕ. Но лежат они внутри
    архива, названного по скважине (`Andriyashivska11.zip`), и каталог распаковки это имя хранит.
    ⚠ Поднимаемся вверх ДО КОРНЯ ИСТОЧНИКА, не выше: иначе можно поймать имя случайного общего
    каталога и молча приписать сотни листов не той скважине."""
    w = well_of(p.stem)
    if w:
        return w
    stop = {Path(r).resolve() for r in roots}
    cur = p.parent
    while True:
        rp = cur.resolve()
        if rp in stop or cur.parent == cur:
            return None
        w = well_of(cur.name)
        if w:
            return w
        cur = cur.parent


def sha1(p, n=1 << 20):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()


# ── распаковка архивов ────────────────────────────────────────────────────────────────────────
# ⚠ Для `.rar` вызывается ГОТОВЫЙ UnRAR.exe (D:\Soft\Winrar, 7.20), а не пакет `rarfile`: тому всё
# равно нужен внешний бинарь, так что python-зависимость была бы лишней прослойкой.
UNRAR = Path(r"D:\Soft\Winrar\UnRAR.exe")
WINRAR = Path(r"D:\Soft\Winrar\WinRAR.exe")           # запасной для zip, которые бракует python
RE_PART = re.compile(r"^(.*)\.part0*(\d+)$", re.I)    # `имя.part2` → (основа, номер тома)


def unpack_all(srcs, dst):
    """rar/zip → каталог распаковки. Уже распакованное пропускается (метка по имени архива).
    ⚠ Архив может содержать вложенные архивы (так бывает у пересланных пачек), поэтому проход
    повторяется, пока появляются новые распаковки, но не более 3 кругов — защита от бомбы."""
    import zipfile, subprocess
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    done, failed, n = 0, [], 0
    roots = list(srcs)
    for _round in range(3):
        found = 0
        for s in roots:
            for p in sorted(Path(s).rglob("*")):
                if p.suffix.lower() not in (".zip", ".rar"):
                    continue
                # ⚠ МНОГОТОМНЫЙ архив: `имя.part2.rar` открывать НЕЛЬЗЯ — распаковывается только
                # ПЕРВЫЙ том, остальные UnRAR подтягивает сам. Поймано на DRUZH_053.part2.rar.
                mp = RE_PART.match(p.stem)
                if mp and int(mp.group(2)) != 1:
                    continue
                out = dst / (mp.group(1) if mp else p.stem)
                if out.exists():
                    done += 1; continue
                out.mkdir(parents=True, exist_ok=True)
                try:
                    if p.suffix.lower() == ".zip":
                        try:
                            with zipfile.ZipFile(p) as z:
                                z.extractall(out)
                        except zipfile.BadZipFile:
                            # ⚠ Часть zip сделана инструментами, кладущими имена в cp866/cp1251:
                            # питоновский `zipfile` бракует их («File name in directory … differ»),
                            # а WinRAR открывает. Поймано на Andriyashivska11.zip и Melykh_034.zip.
                            if not WINRAR.is_file():
                                raise
                            r = subprocess.run([str(WINRAR), "x", "-y", "-ibck", str(p),
                                                str(out) + "\\"],
                                               capture_output=True, text=True, timeout=3600)
                            if r.returncode != 0:
                                raise RuntimeError(f"WinRAR код {r.returncode}")
                    elif UNRAR.is_file():
                        r = subprocess.run([str(UNRAR), "x", "-y", "-idq", str(p), str(out) + "\\"],
                                           capture_output=True, text=True, timeout=3600)
                        if r.returncode != 0:
                            raise RuntimeError((r.stderr or r.stdout or "").strip()[:120]
                                               or f"UnRAR код {r.returncode}")
                    else:
                        raise RuntimeError(f"не найден {UNRAR}")
                    done += 1; n += 1; found += 1
                except Exception as e:
                    try:
                        out.rmdir()            # пустой каталог не должен маскировать неудачу
                    except OSError:
                        pass
                    failed.append((p.name, f"{type(e).__name__}: {e}"))
        if not found:
            break
        roots = [dst]                          # следующий круг — только по распакованному
    return done, n, failed


if a.unpack:
    d, n, failed = unpack_all(a.src, a.unpack)
    if d or failed:
        print(f"архивов rar/zip: {d} распаковано (из них новых {n})"
              + (f"; ⛔ НЕ ОТКРЫЛОСЬ {len(failed)}" if failed else ""))
        for nm, why in failed[:5]:
            print(f"  ⛔ {nm[:52]}: {why[:60]}")
        if failed and any("не найден" in w for _, w in failed):
            print(f"  ⚠ не найден распаковщик {UNRAR} — .rar останутся нетронутыми")
    if d:
        a.src = list(a.src) + [a.unpack]

# ── сбор ──────────────────────────────────────────────────────────────────────────────────────
files, skipped_auto = [], 0
for s in a.src:
    for p in Path(s).rglob("*"):
        if not p.is_file() or not (p.suffix.lower() in KIND or p.suffix.lower() in IMG):
            continue
        if RE_AUTO.search(p.stem):
            skipped_auto += 1; continue
        files.append(p)
print(f"найдено файлов: {len(files)} в {len(a.src)} источник(ах)"
      + (f"; ⚠ пропущено НАШЕЙ выдачи `_auto`: {skipped_auto}" if skipped_auto else ""))

sets, unknown, dupes = defaultdict(dict), [], []
for p in files:
    stem = p.stem
    w = well_of_path(p, a.src)
    if w is None:
        unknown.append(p); continue
    ext = p.suffix.lower()
    k = "img" if ext in IMG else KIND[ext]
    slot = sets[(w, stem)]
    if k in slot and slot[k] != p:
        # одинаковый stem и вид, но разные файлы — решать должен человек
        if slot[k].stat().st_size != p.stat().st_size or sha1(slot[k]) != sha1(p):
            dupes.append((w, stem, k, slot[k], p)); continue
    slot.setdefault(k, p)

# ── что УЖЕ есть в боевом архиве ──────────────────────────────────────────────────────────────
# ⚠ Сверка двухступенчатая: сначала по stem (дёшево), и только совпавшие — по содержимому. Лист,
# у которого stem тот же, а файл ДРУГОЙ, это не дубликат, а расхождение версий: его нельзя ни
# молча добавить, ни молча выбросить — он идёт в отдельную корзину для решения человеком.
HAVE = {}
if a.archive and Path(a.archive).is_dir():
    for p in Path(a.archive).rglob("*"):
        if p.is_file() and (p.suffix.lower() in KIND or p.suffix.lower() in IMG):
            HAVE.setdefault(p.stem, {})[p.suffix.lower()] = p
    print(f"в боевом архиве листов: {len({k for k in HAVE})}")

# ── аудит ─────────────────────────────────────────────────────────────────────────────────────
full, no_img, no_nlgx, no_las, already, differs = [], [], [], [], [], []
for (w, stem), got in sorted(sets.items()):
    if stem in HAVE:
        same = True
        for k, p in got.items():
            q = HAVE[stem].get(p.suffix.lower())
            if q is None:
                same = False; break
            if q.stat().st_size != p.stat().st_size or sha1(q) != sha1(p):
                same = False; break
        (already if same else differs).append((w, stem, got))
        continue
    has_n, has_i, has_l = "wlg" in got, "img" in got, "las" in got
    if has_n and has_i:
        (full if has_l else no_las).append((w, stem, got))
    elif has_n:
        no_img.append((w, stem, got))
    elif has_i:
        no_nlgx.append((w, stem, got))

wells = sorted({w for w, _ in sets})
print(f"скважин {len(wells)}, комплектов (листов) {len(sets)}\n")
print(f"{'★ полный (nlgx + img + las)':<38}{len(full):>6}")
print(f"{'  nlgx + img, НЕТ las':<38}{len(no_las):>6}   годен для распознавания, без значений")
print(f"{'⚠ nlgx БЕЗ img':<38}{len(no_img):>6}   ← искать скан В ПЕРВУЮ ОЧЕРЕДЬ (есть эталон)")
print(f"{'  img БЕЗ nlgx':<38}{len(no_nlgx):>6}   нечем мерить: скан без разметки эксперта")
print(f"{'⛔ имя не разобрано':<38}{len(unknown):>6}   скважина не определена, не раскладываю")
print(f"{'⛔ конфликт (один stem, разные файлы)':<38}{len(dupes):>6}   решать вручную")
print(f"{'· уже в боевом архиве (совпал байт-в-байт)':<38}{len(already):>6}   пропускаю")
print(f"{'⚠ тот же лист, но ДРУГАЯ версия файла':<38}{len(differs):>6}   ← смотреть вам: что новее")
if differs:
    print(f"\n⚠ РАСХОЖДЕНИЕ ВЕРСИЙ с архивом — {len(differs)} листов:")
    for w, stem, _ in differs[:12]:
        print(f"  {w:<16} {stem[:60]}")
    if len(differs) > 12:
        print(f"  … и ещё {len(differs)-12}")

if no_img:
    print(f"\n⚠ NLGX БЕЗ СКАНА — {len(no_img)} листов, вот они (это ваш список поиска):")
    byw = defaultdict(list)
    for w, stem, _ in no_img:
        byw[w].append(stem)
    for w in sorted(byw):
        print(f"  {w:<16} {len(byw[w]):>3}: {byw[w][0][:62]}" + (" …" if len(byw[w]) > 1 else ""))
for p in unknown[:10]:
    print(f"  не разобрано: {p.name[:70]}")
if len(unknown) > 10:
    print(f"  … и ещё {len(unknown)-10}")
for w, stem, k, x, y in dupes[:10]:
    print(f"  конфликт {k}: {stem[:50]} — {x.parent.name}/ против {y.parent.name}/")

if a.report:
    Path(a.report).write_text(
        "\n".join(["# NLGX БЕЗ СКАНА (искать в первую очередь)"] + [s for _, s, _ in no_img]
                  + ["", "# IMG БЕЗ NLGX"] + [s for _, s, _ in no_nlgx]
                  + ["", "# ИМЯ НЕ РАЗОБРАНО"] + [p.name for p in unknown]), encoding="utf-8")
    print(f"\nсписок недостающего → {a.report}")

# ── раскладка ─────────────────────────────────────────────────────────────────────────────────
if not a.apply:
    print("\n★ Это ОТЧЁТ. Ничего не скопировано. Разложить: тот же вызов с --apply")
    sys.exit()
n = 0
for (w, stem), got in sets.items():
    if stem in HAVE and (w, stem) in {(x, y) for x, y, _ in already}:
        continue
    if "wlg" not in got and "img" not in got:
        continue
    for k, p in got.items():
        d = Path(a.dst) / w / FOLDER[k]
        d.mkdir(parents=True, exist_ok=True)
        dst = d / p.name
        if not dst.exists():
            shutil.copy2(p, dst); n += 1
print(f"\n★ скопировано файлов: {n} → {a.dst} (исходники не тронуты)")

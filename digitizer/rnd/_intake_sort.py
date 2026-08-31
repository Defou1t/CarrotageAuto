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
import sys, os, argparse, re, shutil, hashlib
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
ap.add_argument("--apply", action="store_true", help="реально раскладывать (без флага — только отчёт)")
# ⚠⚠ РАСКЛАДКА ЖЁСТКИМИ ССЫЛКАМИ, А НЕ КОПИЕЙ. Источников набралось 240 ГБ, из них 902 архива на
# 220 ГБ; после распаковки свободного остаётся ~160 ГБ, и копия разложенного туда НЕ ВЛЕЗАЕТ.
# Жёсткая ссылка на том же томе не занимает места и создаётся мгновенно, а исходник остаётся цел —
# удаление разложенного каталога данные не трогает. `copy` оставлен для разных томов.
ap.add_argument("--mode", default="link", choices=["link", "copy"],
                help="link — жёсткие ссылки (по умолчанию), copy — копирование")
ap.add_argument("--min-free", type=float, default=25.0,
                help="ГБ: ниже этого распаковка останавливается")
ap.add_argument("--report", default="", help="куда положить список неполных комплектов (txt)")
# Приоритет источников при расхождении версий одного листа. Чем раньше — тем главнее.
ap.add_argument("--priority", default="for bodya_cherk,from bodya,from group,unpacked,from telegram",
                help="подстроки путей через запятую, по убыванию доверия")
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
# Разделитель внутри даты бывает точкой, дефисом И ПОДЧЁРКИВАНИЕМ: `1973_06_09_Rybal_137_…`
# (найдено на полном разборе — часть из 997 неразобранных имён).
RE_DATE = re.compile(r"^\d{4}[._\-]\d{2}[._\-]\d{2}"
                     r"(?:\s*-\s*(?:\d{4}[._\-])?\d{2}[._\-]\d{2})?[_\-]+(.+)$")
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
                # ⚠ СТОРОЖ МЕСТА: архивов 902 на 220 ГБ, распаковка их удваивает. Забить диск на
                # машине, где идут расчёты, — отдельная поломка; лучше остановиться и сказать.
                if shutil.disk_usage(str(dst)).free / (1 << 30) < a.min_free:
                    failed.append((p.name, f"остановлено: свободно < {a.min_free} ГБ"))
                    return done, n, failed
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

# ── приоритет источников при расхождении версий ───────────────────────────────────────────────
# ⚠⚠ ПОРЯДОК ЗАДАЁТ ВЛАДЕЛЕЦ, А НЕ СКРИПТ. Правило Эдуарда (02.08): при совпадении листа берётся
# версия из `for Bodya_Cherk` — это то, что реально сдавалось заказчику ПОСЛЕ проверки экспертом.
# Остальное — «почти правильное»: возможны ошибки оцифровки, имён методов, шкал axis/depth.
# Чем РАНЬШЕ источник в списке, тем он главнее. Неизвестный источник — в самый конец.
def PRIO(p):
    s = str(p).replace("\\", "/").lower()
    for i, key in enumerate(PRIO_KEYS):
        if key in s:
            return i
    return len(PRIO_KEYS)


PRIO_KEYS = [k.strip().lower() for k in a.priority.split(",") if k.strip()]
sets, unknown, dupes = defaultdict(dict), [], []
for p in files:
    stem = p.stem
    w = well_of_path(p, a.src)
    if w is None:
        unknown.append(p); continue
    ext = p.suffix.lower()
    k = "img" if ext in IMG else KIND[ext]
    slot = sets[(w, stem)]
    old = slot.get(k)
    if old is not None and old != p:
        # ⚠⚠ ОДИН ЛИСТ В НЕСКОЛЬКИХ ВЕРСИЯХ — это НОРМА для этого архива, а не сбой. Проверено на
        # `Bilch_Volyts_203_…_AK_1076_1493`: nlgx 136974 / 134462 / 136936 байт в трёх источниках,
        # причём `.bck` из группы совпадает по размеру с `.nlgx` из экспорта ⇒ экспорт СТАРЕЕ, его
        # разметка после правки эксперта уехала в резервную копию.
        # ⇒ Молча пропускать нельзя (потеряем лист) и молча брать первый попавшийся тоже нельзя.
        # Берём версию из источника ВЫШЕ ПО ПРИОРИТЕТУ, а проигравшую записываем в отчёт.
        if old.stat().st_size != p.stat().st_size or sha1(old) != sha1(p):
            if PRIO(p) < PRIO(old):
                dupes.append((w, stem, k, old, p)); slot[k] = p
            else:
                dupes.append((w, stem, k, p, old))
            continue
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
print(f"{'· версий одного листа разошлось':<38}{len(dupes):>6}   взята из источника выше по приоритету")
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
    # ⚠ ГЛАВНОЕ В ОТЧЁТЕ — СПИСОК СКВАЖИН, А НЕ ФАЙЛОВ (запрос Эдуарда 02.08: «чтобы не теряться в
    # каждом подфайле»). Семь с половиной тысяч имён листов невозможно использовать как список
    # поиска; список из полусотни скважин — можно. Поэтому сначала скважины, потом уже подробности.
    # ⚠⚠ СКАН МОЖЕТ УЖЕ ЛЕЖАТЬ В БОЕВОМ АРХИВЕ. Без этой проверки отчёт писал «BOGAT_011 — нет 179
    # из 180», хотя у этой скважины в `F:\nds\projects\Archive\BOGAT_011\img` сканы есть. Список
    # поиска, отправляющий искать уже имеющееся, хуже отсутствия списка.
    HAVE_IMG = {stem for stem, kinds in HAVE.items() if any(e in IMG for e in kinds)}
    per_well = defaultdict(lambda: [0, 0])          # скважина → [листов всего, из них со сканом]
    for (w, stem), got in sets.items():
        per_well[w][0] += 1
        per_well[w][1] += ("img" in got or stem in HAVE_IMG)
    none_img = sorted(w for w, (t, i) in per_well.items() if i == 0)
    part_img = sorted(((w, t - i, t) for w, (t, i) in per_well.items() if 0 < i < t),
                      key=lambda q: -(q[1]))
    lines = [f"# СКВАЖИНЫ БЕЗ ЕДИНОГО СКАНА — {len(none_img)} шт "
             f"(есть разметка, изображений нет вообще)", ""]
    for i in range(0, len(none_img), 8):            # по 8 в строку: список для глаз, не для машины
        lines.append(", ".join(none_img[i:i + 8]))
    lines += ["", f"# СКВАЖИНЫ, ГДЕ СКАНОВ НЕ ХВАТАЕТ ЧАСТИЧНО — {len(part_img)} шт "
                  f"(формат: скважина — нет N из M листов)", ""]
    lines += [f"{w} — нет {miss} из {tot}" for w, miss, tot in part_img]
    lines += ["", "=" * 78, f"# ПОДРОБНО: NLGX БЕЗ СКАНА, {len(no_img)} листов", ""]
    lines += [s for _, s, _ in no_img]
    lines += ["", f"# IMG БЕЗ NLGX, {len(no_nlgx)} (часто это скан с именем от сканера — "
                  f"по имени в пару не встанет, хотя лист может быть тот же)", ""]
    lines += [s for _, s, _ in no_nlgx]
    lines += ["", f"# ИМЯ НЕ РАЗОБРАНО, {len(unknown)}", ""]
    lines += [p.name for p in unknown]
    # ⚠ ОДНА СКВАЖИНА ПОД РАЗНЫМИ НАПИСАНИЯМИ — `BVp_32` / `bvp_32` / `БВп_32`, `SKVORTS_012` /
    # `SKVRTS_012`. Сливать их автоматически НЕЛЬЗЯ: латиница и кириллица могут оказаться разными
    # скважинами, а ошибка слияния тихо испортит и обучение, и разрез по семействам. Поэтому —
    # только СПИСОК ПОДОЗРЕНИЙ для человека. Группируем по регистру и по номеру скважины.
    byl = defaultdict(set)
    for w in per_well:
        byl[w.lower()].add(w)
    same_case = [sorted(v) for v in byl.values() if len(v) > 1]
    # ⚠ ГРУППИРОВКА ПО НОМЕРУ ДАЁТ ЛОЖНЫЕ СОВПАДЕНИЯ: `KOBZIV_107, MARKIV_107, YULIIV_107, БВп_107`
    # — это РАЗНЫЕ скважины с одинаковым номером. Полезен только случай, когда совпадает и номер,
    # и ПЕРВАЯ БУКВА основы после транслитерации кириллицы: `Bilch_Volyts_403` ↔ `БВп_403`.
    TR = str.maketrans("абвгдежзиклмнопрстуфхцчшы", "abvgdejziklmnoprstufhcchy")
    def key(w):
        m = re.match(r"^(.*?)[_\-]?(\d+)$", w)
        if not m:
            return None
        base = m.group(1).lower().translate(TR)
        base = re.sub(r"[^a-z]", "", base)
        return (base, str(int(m.group(2))))

    def dist(x, y):                                 # расстояние редактирования, потолок 3
        if abs(len(x) - len(y)) > 2:
            return 9
        prev = list(range(len(y) + 1))
        for i, cx in enumerate(x, 1):
            cur = [i]
            for j, cy in enumerate(y, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cx != cy)))
            prev = cur
        return prev[-1]

    # ⚠ Двух первых букв основы МАЛО: `KREMEN_063` и `KRUZH_063` — разные месторождения, а список
    # с третью ложных срабатываний перестают читать. Требуем совпадение номера И близость основ.
    bynum = defaultdict(list)
    for w in per_well:
        k = key(w)
        if k:
            bynum[k[1]].append((k[0], w))
    susp = []
    for num, items in bynum.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (b1, w1), (b2, w2) = items[i], items[j]
                if w1.lower() != w2.lower() and dist(b1, b2) <= 2:
                    susp.append(sorted([w1, w2]))
    if same_case or susp:
        lines += ["", "=" * 78,
                  "# ⚠ ВОЗМОЖНО ОДНА И ТА ЖЕ СКВАЖИНА ПОД РАЗНЫМИ ИМЕНАМИ (не сливал — решать вам)",
                  "# ⚠ АББРЕВИАТУРЫ ЗДЕСЬ НЕ НАЙДУТСЯ: `БВп_403` и `Bilch_Volyts_403` — это одна",
                  "#   скважина, но «БВп» = сокращение «Білче-Волиця», а не другое написание.",
                  "#   Строковым правилом такую пару не поймать; их надо назвать вручную.",
                  ""]
        lines += ["различие только в регистре: " + ", ".join(g) for g in same_case]
        lines += ["та же основа и номер (кириллица/латиница): " + ", ".join(g) for g in susp[:60]]
    Path(a.report).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n★ СКВАЖИН БЕЗ ЕДИНОГО СКАНА: {len(none_img)}; частично не хватает у {len(part_img)}")
    if none_img:
        print("  " + ", ".join(none_img[:14]) + (" …" if len(none_img) > 14 else ""))
    print(f"список недостающего → {a.report}")

# ── раскладка ─────────────────────────────────────────────────────────────────────────────────
if not a.apply:
    print("\n★ Это ОТЧЁТ. Ничего не скопировано. Разложить: тот же вызов с --apply")
    sys.exit()
SKIP = {(x, y) for x, y, _ in already}
n, linked, copied, failed = 0, 0, 0, 0
for (w, stem), got in sets.items():
    if (w, stem) in SKIP:                       # уже есть в боевом архиве байт-в-байт
        continue
    if "wlg" not in got and "img" not in got:
        continue
    for k, p in got.items():
        d = Path(a.dst) / w / FOLDER[k]
        d.mkdir(parents=True, exist_ok=True)
        dst = d / p.name
        if dst.exists():
            continue
        try:
            if a.mode == "link":
                os.link(p, dst); linked += 1
            else:
                shutil.copy2(p, dst); copied += 1
        except OSError:
            # другой том либо ФС без жёстких ссылок: молча терять файл нельзя, копируем
            try:
                shutil.copy2(p, dst); copied += 1
            except OSError as e:
                failed += 1
                if failed <= 5:
                    print(f"  ⛔ не разложен {p.name[:50]}: {e}")
        n += 1
print(f"\n★ разложено файлов: {n} → {a.dst}  (ссылок {linked}, копий {copied}"
      + (f", ⛔ НЕ УДАЛОСЬ {failed}" if failed else "") + ")")
print("  исходники не тронуты; жёсткая ссылка места не занимает, удаление каталога данные не удалит")

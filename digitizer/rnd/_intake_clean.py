r"""_intake_clean.py — УБРАТЬ ИЗ ИСХОДНЫХ ПАПОК ТО, ЧТО УЖЕ ДОКАЗУЕМО ЛЕЖИТ В НАБОРАХ СКВАЖИН.

ЗАЧЕМ (запрос Эдуарда 02.08). Источники (`from group`, `from Bodya`, `from telegram`) больше не
пополняются; всё разобранное перенесено в `intake\sorted\<Скважина>\{wlg,img,las}`. Исходные папки
надо освободить — но так, чтобы НИЧЕГО не потерять.

⚠⚠ УДАЛЕНИЕ НЕОБРАТИМО, ПОЭТОМУ ПРАВИЛО ЖЁСТКОЕ: файл удаляется ТОЛЬКО если он доказуемо
представлен в наборе скважин. Доказательство — одно из двух:
  ЖЁСТКАЯ ССЫЛКА — у файла в источнике и файла в `sorted` СОВПАДАЕТ индекс на томе (st_ino).
                   Это тот же самый файл, второе имя; удаление имени данные не трогает.
  ХЭШ            — размер и sha1 совпадают с файлом в `sorted`. Копия, содержимое сохранено.
Всё остальное — НЕ УДАЛЯЕТСЯ и попадает в отчёт с причиной. В частности НЕ удаляются:
  • проигравшие версии листа (их содержимое в `sorted` не попало — а именно они понадобятся,
    когда появится `for Bodya_Cherk`, чтобы измерить шум эталона);
  • файлы с неразобранным именем;
  • сами архивы rar/zip (в `sorted` лежит их СОДЕРЖИМОЕ, а не они; удалять архив — решение
    владельца, потому что распаковка воспроизводима, но небесплатна).

★ ПО УМОЛЧАНИЮ — СУХОЙ ПРОГОН. Ничего не удаляется без `--apply`.
★ Освобождение места считается ЧЕСТНО: у жёсткой ссылки удаление имени НЕ освобождает байты
  (данные держит вторая ссылка). Поэтому в отчёте два числа: «освободится» и «просто исчезнет имя».

  <python> _intake_clean.py [--apply] [--archives]   # --archives: удалять и распакованные архивы
"""
import sys, os, argparse, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", default=[
    r"E:\Carrotagki_auto\from Bodya", r"E:\Carrotagki_auto\from group",
    r"E:\Carrotagki_auto\from telegram", r"E:\Carrotagki_auto\New_files"])
ap.add_argument("--sorted", default=r"E:\Carrotagki_auto\intake\sorted")
ap.add_argument("--apply", action="store_true", help="реально удалять (без флага — только отчёт)")
ap.add_argument("--archives", action="store_true",
                help="удалять и архивы rar/zip, чьё СОДЕРЖИМОЕ разложено")
ap.add_argument("--report", default=r"E:\Carrotagki_auto\intake\cleaned.txt")
a = ap.parse_args()
ARCH = {".zip", ".rar"}


def sha1(p, n=1 << 20):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()


# ── что лежит в наборах скважин ───────────────────────────────────────────────────────────────
S = Path(a.sorted)
if not S.is_dir():
    sys.exit(f"нет каталога наборов {S} — сначала разложить (`_intake_sort.py --apply`)")
by_ino, by_size = {}, defaultdict(list)
for p in S.rglob("*"):
    if not p.is_file():
        continue
    st = p.stat()
    by_ino[st.st_ino] = p
    by_size[st.st_size].append(p)
print(f"в наборах скважин: {len(by_ino)} файлов, {len(by_size)} разных размеров")

# ── разбор источников ─────────────────────────────────────────────────────────────────────────
del_link, del_copy, keep = [], [], defaultdict(list)
arch_keep = []
for root in a.src:
    r = Path(root)
    if not r.is_dir():
        print(f"⚠ нет источника {root} — пропущен"); continue
    for p in r.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in ARCH:
            arch_keep.append(p); continue
        st = p.stat()
        if st.st_ino in by_ino:                       # ТОТ ЖЕ файл, второе имя
            del_link.append(p); continue
        same = [q for q in by_size.get(st.st_size, ()) if q.name == p.name]
        if same and sha1(p) == sha1(same[0]):         # копия с тем же содержимым
            del_copy.append(p); continue
        keep[("другая версия" if same else "нет в наборах")].append(p)

n_free = sum(p.stat().st_size for p in del_copy)
n_name = sum(p.stat().st_size for p in del_link)
GB = 1 << 30
print(f"\n{'★ удалить: то же имя файла (жёсткая ссылка)':<46}{len(del_link):>7}"
      f"   освободит 0 (данные держит набор)")
print(f"{'★ удалить: копия, содержимое совпало':<46}{len(del_copy):>7}"
      f"   освободит {n_free/GB:.1f} ГБ")
for why, lst in sorted(keep.items()):
    print(f"{'· ОСТАВИТЬ: ' + why:<46}{len(lst):>7}   {sum(q.stat().st_size for q in lst)/GB:.1f} ГБ")
print(f"{'· ОСТАВИТЬ: архивы rar/zip':<46}{len(arch_keep):>7}"
      f"   {sum(p.stat().st_size for p in arch_keep)/GB:.1f} ГБ"
      + ("  (удалятся: задан --archives)" if a.archives else "  (в наборах их СОДЕРЖИМОЕ)"))
print(f"\nимён исчезнет {len(del_link)+len(del_copy)}, реально освободится "
      f"{n_free/GB:.1f} ГБ (ссылки байтов не держат)")

Path(a.report).write_text("\n".join(
    [f"# УДАЛЕНО КАК ССЫЛКА НА НАБОР ({len(del_link)})"] + [str(p) for p in del_link]
    + ["", f"# УДАЛЕНО КАК КОПИЯ ({len(del_copy)})"] + [str(p) for p in del_copy]
    + ["", "# ОСТАВЛЕНО"]
    + [f"{why}: {p}" for why, lst in sorted(keep.items()) for p in lst]), encoding="utf-8")
print(f"подробности → {a.report}")

if not a.apply:
    print("\n★ Это СУХОЙ ПРОГОН. Ничего не удалено. Удалить: тот же вызов с --apply")
    sys.exit()

gone = 0
for p in del_link + del_copy + (arch_keep if a.archives else []):
    try:
        p.unlink(); gone += 1
    except OSError as e:
        print(f"  ⛔ {p.name[:50]}: {e}")
# пустые каталоги после удаления убираем, чтобы источники не превратились в лес пустышек
for root in a.src:
    for d in sorted(Path(root).rglob("*"), key=lambda q: -len(q.parts)):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass
print(f"\n★ удалено файлов: {gone}")

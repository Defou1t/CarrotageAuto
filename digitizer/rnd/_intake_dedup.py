r"""_intake_dedup.py — ДУБЛИ ПРИЁМКИ ПРОТИВ АРХИВА → КАРАНТИН (26.09, аудит проекта).

`E:\Carrotagki_auto\intake` (104 ГБ, 49 тыс. файлов) — сырьё приёмки 10.08 («корпус удвоен», §6.1xx): листы разобраны и
влиты в `projects\Archive`. Замер 26.09: 11 138 файлов (61.8 ГБ) совпадают с архивом по имени+размеру, ещё 2 348 (2.7 ГБ) — по
размеру и содержимому под другим именем. Файл считается дублем, только если в архиве есть файл ТОГО ЖЕ РАЗМЕРА с тем же хэшем
первого и последнего мегабайта (имени мало). Дубли переносятся в `--quarantine` с тем же относительным путём + манифест;
уникальное (то, чего в архиве нет) остаётся на месте. По умолчанию — сухой прогон.

  _intake_dedup.py [--root E:/Carrotagki_auto/intake] [--apply --quarantine E:/Carrotagki_auto/_quarantine]
"""
import os, sys, argparse, hashlib, shutil, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import defaultdict, Counter

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=r"E:/Carrotagki_auto/intake")
ap.add_argument("--archive", default=r"E:/Carrotagki_auto/projects/Archive")
ap.add_argument("--quarantine", default="")
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()


def phash(p):
    with open(p, "rb") as fh:
        head = fh.read(1 << 20)
        n = os.path.getsize(p)
        fh.seek(max(0, n - (1 << 20)))
        return hashlib.md5(head + fh.read(1 << 20)).hexdigest()


by_size = defaultdict(list)
for dp, dn, fn in os.walk(a.archive):
    for f in fn:
        p = os.path.join(dp, f)
        try:
            by_size[os.path.getsize(p)].append(p)
        except OSError:
            pass
HC = {}


def arch_hashes(sz):
    if sz not in HC:
        HC[sz] = {phash(q) for q in by_size[sz]}
    return HC[sz]


dup = []; uniq = Counter(); uniq_sz = Counter(); dsz = 0
for dp, dn, fn in os.walk(a.root):
    for f in fn:
        p = os.path.join(dp, f)
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        top = Path(p).relative_to(a.root).parts[0]
        if sz > 0 and sz in by_size and phash(p) in arch_hashes(sz):
            dup.append((p, sz)); dsz += sz
        else:
            uniq[top] += 1; uniq_sz[top] += sz
print(f"★ дублей архива: {len(dup)} файлов, {dsz/2**30:.1f} ГБ")
print("★ уникальное (остаётся): " + ", ".join(f"{k}: {uniq[k]} ф., {uniq_sz[k]/2**30:.1f} ГБ" for k in sorted(uniq)))
if a.apply and a.quarantine:
    q = Path(a.quarantine); q.mkdir(parents=True, exist_ok=True)
    with open(q / "_manifest.tsv", "a", encoding="utf-8") as man:
        for p, sz in dup:
            dst = q / Path(*Path(p).resolve().parts[1:])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(p, str(dst))
            man.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{p}\t{dst}\t{sz/2**20:.1f} МБ\tдубль архива (размер + хэш 1+1 МБ)\n")
    print(f"★ перенесено в карантин: {len(dup)}")
else:
    print("⚠ сухой прогон — ничего не перенесено")

"""ЗАПИСЬ kind=6 в nlgx («описание кривой») — extract() её НЕ ЧИТАЕТ ВООБЩЕ.
Замер 19.07: записей ровно столько же, сколько кривых, и они несут:
  35368 полное имя слота («GZ41 DA1 SA1»), 35370 ЧИСТУЮ МНЕМОНИКУ («GZ4»),
  35380 некий индекс (у GZ4=0, GZ5=2, CALI=5, SP=6), плюс TIFF-теги 256/257/258/273/279 —
  растр 8x8 1 бит = узор пера.
Эдуард (19.07): цвета/толщины в словаре мнемоник НЕТ, их задаёт ЭКСПЕРТ при подготовке рамки.
Значит источник цвета — не словарь, а сама рамка. Проверяем, что означает 35380 и растр.

  python _curve_desc_probe.py [N листов]
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
from collections import Counter, defaultdict
from extract_nlgx import read_ifds, val
from auto import meta as M

MN = r"F:\nds\Auto\mnemonics.json"
ARCHIVE = Path(r"F:\nds\projects\Archive")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def descs(path):
    """Записи kind=6 файла: [{name, mnem, i35372, i35378, i35380, i35382, bitmap}]."""
    data = open(path, "rb").read()
    out = []
    for t in read_ifds(data):
        if val(t, 34768) != 6:
            continue
        off, cnt = val(t, 273), val(t, 279)
        bm = data[off:off + cnt] if off and cnt else b""
        out.append({"name": val(t, 35368), "mnem": val(t, 35370),
                    "i372": val(t, 35372), "i378": val(t, 35378),
                    "i380": val(t, 35380), "i382": val(t, 35382),
                    "w": val(t, 256), "h": val(t, 257), "bits": val(t, 258),
                    "bitmap": bm})
    return out


by_mnem = defaultdict(Counter)      # мнемоника → какие i380 встречаются
by_380 = defaultdict(Counter)       # i380 → какие мнемоники
patterns = Counter()
n_files = n_desc = 0
examples = []
for wlg in sorted(ARCHIVE.glob("*/wlg")):
    for f in sorted(wlg.glob("*.nlgx")):
        if "_auto" in f.stem or n_files >= N:
            continue
        try:
            ds = descs(str(f))
        except Exception:
            continue
        if not ds:
            continue
        n_files += 1
        for d in ds:
            n_desc += 1
            mn = M.mnem_root(d["mnem"] or "") or (d["mnem"] or "?")
            by_mnem[mn][d["i380"]] += 1
            by_380[d["i380"]][mn] += 1
            patterns[d["bitmap"].hex()] += 1
            if len(examples) < 6 and d["bitmap"]:
                examples.append((f.stem[:34], d["mnem"], d["i380"], d["bitmap"].hex()))

print(f"файлов с записями kind=6: {n_files}, описаний кривых: {n_desc}")
print(f"\nРАЗНЫХ УЗОРОВ 8x8: {len(patterns)}  (топ)")
for h, c in patterns.most_common(6):
    print(f"   {h}  ×{c}")
print(f"\nЗНАЧЕНИЙ 35380: {len(by_380)}")
for v, c in sorted(by_380.items(), key=lambda kv: -sum(kv[1].values()))[:12]:
    top = ", ".join(f"{m}×{k}" for m, k in c.most_common(6))
    print(f"   35380={v!r:<6} всего {sum(c.values()):<5} мнемоники: {top}")
print("\nПО МНЕМОНИКАМ (стабилен ли индекс у одной мнемоники):")
for m, c in sorted(by_mnem.items(), key=lambda kv: -sum(kv[1].values()))[:14]:
    tot = sum(c.values()); top, n = c.most_common(1)[0]
    print(f"   {m:<8} n={tot:<5} чаще всего 35380={top!r} ({100*n//tot}%)  варианты: {dict(c)}")
print("\nпримеры:")
for e in examples:
    print("  ", e)

r"""_fold_missing.py — КАКИХ ЛИСТОВ В ФОЛДЕ НЕ ХВАТАЕТ, И СПИСОК ДЛЯ ДОЗАПУСКА (§6.166).

ОТКУДА ВОПРОС. Полнота фолда проверялась ПО ФАЙЛАМ ШАРДОВ: `ab_*of8.pkl` восемь штук ⇒ «готов».
Это неверно. Шард может доработать до конца и записать дамп, УРОНИВ по дороге отдельные листы:
`_trace_prod_ab.py` ловит исключение листа, печатает «ПАДЕНИЕ» и идёт дальше. Так фолд 2 замера
§6.159 оказался на 206 листах из 208 — два листа упали `MemoryError` (массивы 1.95 ГиБ и 704 МиБ),
и НИ ОДИН контролёр этого не увидел: тик считал дампы, сводка печатала «шардов 8 из 8».

⚠⚠ И ПОТЕРЯ НЕ СЛУЧАЙНА — ЭТО УЖЕ ЗАПИСАННЫЙ УРОК (§6.157). Память кончается на САМЫХ КРУПНЫХ
листах, то есть выборка смещается по размеру бланка — по признаку, который влияет и на трудность
листа, и на поведение декодера. §6.157 лечил это отдельным пересчётом (`ab_wellmap_fix`,
`ab_fold0_fix`); здесь лечение встроено в тик.

ЧТО ДЕЛАЕТ. Читает дампы фолда, собирает МНОЖЕСТВО РАЗОБРАННЫХ ЛИСТОВ (а не число шардов),
сверяет со списком `--sheets` и печатает недостачу. С `--out` пишет список недостающих в файл —
его тик скармливает дозапуску с МЕНЬШИМ числом шардов (больше памяти на процесс).

  <ComfyUI>\python_embeded\python.exe _fold_missing.py --dir <фолд> --sheets <список> [--out <файл>]

Код возврата: 0 — фолд полон; 1 — есть недостача (тик по этому коду и решает, дозапускать ли).
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="каталог фолда (в нём лежат ab_*of<N>.pkl)")
ap.add_argument("--mode", default="M")
ap.add_argument("--sheets", required=True, help="список листов, который просили посчитать")
ap.add_argument("--out", default="", help="куда записать список НЕДОСТАЮЩИХ листов")
# ★ 20.09: полнота ПО ФАЙЛАМ ВЫДАЧИ, а не только по дампам. A/B §6.213 дошёл до конца с полными дампами
#   и без файлов `_auto.nlgx` режимов G и R (дефект чистки в `_trace_prod_ab.py`) — эта проверка сказала
#   «ПОЛОН», а ведущему счёту читать было нечего. С `--files` лист считается разобранным, только если
#   есть и запись в дампе, и файл выдачи `<dir>/<mode>/<стем40_md5-8>/*_auto.nlgx`.
ap.add_argument("--files", action="store_true", help="требовать и файл выдачи, не только запись в дампе")
a = ap.parse_args()

want = [l.strip() for l in open(a.sheets, encoding="utf-8") if l.strip()]
d = Path(a.dir)
files = sorted(f for f in d.glob("ab_*of*.pkl") if not f.name.endswith(".part.pkl"))
if not files:                                   # ★ 20.09: готовых нет — смотрим промежуточные (ход прогона)
    files = sorted(d.glob("ab_*of*.part.pkl"))
if not files:
    print(f"⛔ в {d} нет дампов — фолд не считан вовсе")
    sys.exit(1)

# ★ знаменатель берётся из ИМЁН дампов, а не задаётся: разные прогоны шардуются по-разному
den = sorted({f.name.split("of")[1].split(".")[0] for f in files},
             key=lambda q: max(f.stat().st_mtime for f in files if f.name.split("of")[1].split(".")[0] == q))[-1]
files = [f for f in files if f.name.split("of")[1].split(".")[0] == den]
got = set()
for f in files:
    dd = pickle.load(open(f, "rb"))
    tot, per = dd["res"].get(a.mode, ({}, {}))
    got |= set(per)

# ★ ДОЗАПИСАННЫЕ ЛИСТЫ ЛЕЖАТ В ОТДЕЛЬНОМ КАТАЛОГЕ `<фолд>_fix` (дампы нельзя мешать: сводка
# берёт ОДИН знаменатель of<N>). Без их учёта фолд навсегда числится неполным, и тик на каждом
# тике заново копирует выдачи — нашла состязательная проверка §6.169.
fix = d.parent / (d.name + "_fix")
if fix.is_dir():
    for f in fix.glob("ab_*of*.pkl"):
        try:
            got |= set(pickle.load(open(f, "rb"))["res"].get(a.mode, ({}, {}))[1])
        except Exception:
            pass

# в списке имена без расширения либо с ним — сверяем по стему
norm = lambda s: s[:-5] if s.endswith(".nlgx") else s
gotn = {norm(x) for x in got}
missing = [w for w in want if norm(w) not in gotn]
nofile = []
if a.files:
    import hashlib
    for w in want:
        if norm(w) not in gotn:
            continue
        stem = norm(w)
        pd = d / a.mode / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
        if not (pd.is_dir() and any(pd.glob("*_auto.nlgx"))):
            nofile.append(w)
    missing = missing + nofile

print(f"★ ФОЛД {d.name}: дампов {len(files)} из {den}; "
      f"листов в списке {len(want)}, РАЗОБРАНО {len(got)}, НЕДОСТАЁТ {len(missing)}"
      f"   {'★ ПОЛОН' if not missing else '⛔ НЕПОЛОН'}"
      + (f"; из них без файла выдачи {len(nofile)}" if a.files else ""))
if missing:
    print("  недостающие листы (память кончается на КРУПНЫХ — §6.157, смещение по размеру бланка):")
    for m in missing[:20]:
        print(f"    {m}")
    if len(missing) > 20:
        print(f"    … ещё {len(missing) - 20}")
    if a.out:
        Path(a.out).write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"  ★ список для дозапуска записан: {a.out}")
sys.exit(1 if missing else 0)

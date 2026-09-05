r"""_handoff_check.py — СВЕРКА ТАБЛИЦЫ «ЧТО СЕЙЧАС В ПРОДЕ» С САМИМ `auto/config.py` (§6.183).

ОТКУДА ВОПРОС. 05.09 в `HANDOFF.md`, в разделе «ЧТО СЕЙЧАС В ПРОДЕ», нашлись ДВЕ ВРАНУШКИ:
`slot_gate = frac0.2` и `slot_sib = 0.0` с пометками «⇒ ВЫКЛЮЧИТЬ» и «⇒ ВКЛЮЧИТЬ». Решение §6.132
внесено 20.08, то есть таблица описывала прод ДВУХНЕДЕЛЬНОЙ давности — а это ТОЧКА ВХОДА, её
читают первой. Новая сессия взяла бы неверную конфигурацию и не узнала бы об этом.
⇒ Тот же класс, что аудит §6.173 нашёл в шести местах ПРОДА («комментарий пережил замер»), только
теперь в документе о проде. Значит проверять надо машиной, а не глазами.

ЧТО ДЕЛАЕТ. Читает таблицу раздела «ЧТО СЕЙЧАС В ПРОДЕ» и для каждой ручки сверяет значение с
умолчанием в `auto/config.py`. Расхождение — код возврата 1.
⚠ Сверяются только те строки, чьё имя есть в `CVParams`: в таблице встречаются и не-ручки.

★★ ПРОВЕРОК ТРИ, И ДВЕ ИЗ НИХ ДОБАВЛЕНЫ ПОСЛЕ 05.09 ПО ЗАМЕЧАНИЮ ЗАКАЗЧИКА: одной сверки
значений мало, потому что она ловит уже СЛУЧИВШЕЕСЯ расхождение, а надо, чтобы блок не мог
устареть незаметно.
  1. ЗНАЧЕНИЯ: каждая ручка таблицы против умолчания `auto/config.py`. Отказ при расхождении.
  2. ★ СВЕЖЕСТЬ: дата последнего коммита, тронувшего `auto/`, против даты в заголовке блока.
     Тронули прод и не обновили блок — блок врёт по построению, даже если значения ещё сходятся.
  3. ★ НАЗВАННОСТЬ РУЧЕК с ХРАПОВИКОМ: §6.173 намерил, что 37 из 64 полей `CVParams` не
     упомянуты в журнале ни разу. Требовать «назвать все» сегодня — значит получить вечно
     красную проверку, а её перестанут читать. Поэтому проверяется НЕ УХУДШЕНИЕ: число
     неназванных не должно расти. Новая ручка обязана быть названа хоть где-то.
⚠ Почему храповик, а не порог: ложный или всегда-красный сторож — это отсутствующий сторож
(тот же урок, что стоил двух ложных тревог самой этой сверке).

  <ComfyUI>\python_embeded\python.exe _handoff_check.py
"""
import sys, re, io, argparse
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ap = argparse.ArgumentParser()
ap.add_argument("--handoff", default=r"F:/nds/Auto/digitizer/rnd/HANDOFF.md")
ap.add_argument("--config", default=r"F:/nds/Auto/auto/config.py")
ap.add_argument("--head", default="## 2. ЧТО СЕЙЧАС В ПРОДЕ")
ap.add_argument("--journal", default=r"F:/nds/Auto/ROADMAP_RECOGNITION.md")
ap.add_argument("--repo", default=r"F:/nds/Auto")
# ★ ХРАПОВИК: столько неназванных ручек было 05.09. Уменьшать можно и нужно, увеличивать — нельзя.
ap.add_argument("--unnamed-max", type=int, default=31)
a = ap.parse_args()

cfg = io.open(a.config, encoding="utf-8").read()
h = io.open(a.handoff, encoding="utf-8").read()
if a.head not in h:
    print(f"⛔ в {a.handoff} нет раздела {a.head!r}")
    sys.exit(1)
i = h.index(a.head)
j = h.find("\n## ", i + 1)
sec = h[i:j if j > 0 else len(h)]


# ★★ ПОСТОЯННЫЕ РАЗЫМЕНОВЫВАЮТСЯ, ВКЛЮЧАЯ ПЕРЕИМЕНОВАННЫЕ ПРИ ИМПОРТЕ. `seq_model` объявлен как
#   `= DEFAULT_SEQ_MODEL`, а он сам — `from .trace_seq import DEFAULT_MODEL as DEFAULT_SEQ_MODEL`.
#   Без разыменования сверка ругалась бы на каждую ручку, заданную через имя, — то есть кричала бы
#   ложно, а ЛОЖНЫЙ СТОРОЖ ПЕРЕСТАЮТ ЧИТАТЬ, и он хуже отсутствующего.
_CFG = Path(a.config)
CONST = {}
for q in sorted(_CFG.parent.glob("*.py")):
    CONST.update(dict(re.findall(r'^([A-Z][A-Z0-9_]*)\s*=\s*"([^"]*)"\s*(?:#.*)?$',
                                 io.open(q, encoding="utf-8").read(), re.M)))
for src, orig, alias in re.findall(r"^from \.(\w+) import (\w+) as (\w+)", cfg, re.M):
    if orig in CONST:
        CONST[alias] = CONST[orig]


def norm(v):
    """★ Сравнение по СМЫСЛУ, а не по написанию: `2.0` и `2`, `"x"` и `x`, `**жирное**` — одно.
    ⚠ Из ячейки HANDOFF берётся ПЕРВОЕ значение в обратных кавычках, если оно есть: за ним в
    той же ячейке может идти пояснение («= гейт ВЫКЛЮЧЕН»), и оно не часть значения."""
    m = re.search(r"`([^`]*)`", v)
    if m:
        v = m.group(1)
    v = v.strip().strip("*").strip().strip("`").strip().strip('"')
    v = CONST.get(v, v)
    try:
        return f"{float(v):g}"
    except ValueError:
        return v


rows = re.findall(r"^\| `([a-z_]+)` \| ([^|]+) \|", sec, re.M)
print(f"★ строк в таблице: {len(rows)}")
bad, skipped = [], []
for name, val in rows:
    m = re.search(r"^    " + name + r"\s*:\s*[^=]+=\s*(.+?)\s*$", cfg, re.M)
    if not m:
        skipped.append(name)
        continue
    real = m.group(1).split("#")[0]
    if norm(val) != norm(real):
        bad.append((name, val.strip(), real.strip()))
    print(f"   {name:20} HANDOFF {norm(val):<26} config {norm(real):<26}"
          f"{'★' if norm(val) == norm(real) else '  ⛔ РАСХОДИТСЯ'}")
if skipped:
    print(f"⚠ не поля `CVParams`, не сверялись ({len(skipped)}): {', '.join(skipped)}")
if bad:
    print(f"\n⛔⛔ ТОЧКА ВХОДА ВРЁТ О ПРОДЕ: расходится строк {len(bad)}")
    for n, v, r in bad:
        print(f"   `{n}`: в HANDOFF «{v}», в config.py «{r}»")
    sys.exit(1)
print("\n★ ПРОВЕРКА 1 — таблица «ЧТО СЕЙЧАС В ПРОДЕ» сходится с `auto/config.py` до значения")

# ── ПРОВЕРКА 2: не старше ли блок, чем последняя правка прода ──────────────────────────────────
import subprocess, datetime
fail2 = False
g = subprocess.run(["git", "log", "-1", "--format=%ad", "--date=short", "--", "auto/"],
                   cwd=a.repo, capture_output=True, text=True, encoding="utf-8",
                   errors="replace", creationflags=0x08000000)
touched = (g.stdout or "").strip()
m = re.search(r"##\s*★*\s*СОСТОЯНИЕ НА (\d\d)\.(\d\d)\.(\d{4})", h)
print("\n★ ПРОВЕРКА 2 — свежесть блока против последней правки прода")
if not touched or not m:
    print(f"   ⚠ не с чем сравнивать (прод тронут: {touched!r}, дата блока найдена: {bool(m)}) — пропуск")
else:
    block = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    print(f"   `auto/` тронут {touched}, блок датирован {block}   "
          f"{'★' if block >= touched else '⛔ БЛОК СТАРШЕ ПРАВКИ ПРОДА'}")
    if block < touched:
        print("   ⇒ прод изменён ПОЗЖЕ, чем обновлялась точка входа: она врёт по построению")
        fail2 = True

# ── ПРОВЕРКА 3: названность ручек, храповиком ────────────────────────────────────────────────
jr = io.open(a.journal, encoding="utf-8").read()
fields = list(dict.fromkeys(re.findall(r"^    ([a-z_][a-z0-9_]*)\s*:\s*[A-Za-z\[]", cfg, re.M)))
unnamed = [k for k in fields if k not in jr and k not in h]
print(f"\n★ ПРОВЕРКА 3 — названность ручек (§6.173), храповик {a.unnamed_max}")
print(f"   полей `CVParams` {len(fields)}, НЕ упомянуты ни в журнале, ни в точке входа: {len(unnamed)}")
fail3 = len(unnamed) > a.unnamed_max
if fail3:
    print(f"   ⛔ СТАЛО ХУЖЕ: было {a.unnamed_max}, стало {len(unnamed)}. Новые: "
          f"{', '.join(unnamed[:8])}")
    print("   ⇒ ручку заводят вместе с доводом, иначе через месяц никто не скажет, откуда цифра (§6.154)")
elif len(unnamed) < a.unnamed_max:
    print(f"   ★ стало ЛУЧШЕ ({len(unnamed)} < {a.unnamed_max}) — опустите храповик: --unnamed-max {len(unnamed)}")
else:
    print("   ★ не ухудшилось")

if fail2 or fail3:
    sys.exit(1)
# ── ПРОВЕРКА 4: ручки, определяющие ОТГРУЖАЕМОЕ поведение, обязаны быть НАЗВАНЫ в таблице ────
# ⚠ Проверка 1 сверяет то, что в таблице ЕСТЬ, и потому слепа к тому, чего в ней НЕТ. 05.09 в
# прод внесли три ручки, и таблица молчала о них, оставаясь формально верной. Умолчание — тоже
# способ соврать: читатель видит семь строк и думает, что это весь прод.
REQUIRED = ("seq_model", "trace_wide_run", "slot_model", "slot_gate", "slot_order", "slot_sib",
            "row_decoder", "rowdec_pick", "rowdec_pregate")
have = {n for n, _ in rows}
lost = [k for k in REQUIRED if k not in have]
print(f"\n★ ПРОВЕРКА 4 — обязательные ручки в таблице ({len(REQUIRED)} шт.)")
if lost:
    print(f"   ⛔ В ТАБЛИЦЕ НЕТ: {', '.join(lost)}")
    print("   ⇒ таблица формально верна и при этом умалчивает о части прода")
    sys.exit(1)
print("   ★ все на месте")

print("\n★★ ВСЕ ЧЕТЫРЕ ПРОВЕРКИ ПРОЙДЕНЫ")

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

  <ComfyUI>\python_embeded\python.exe _handoff_check.py
"""
import sys, re, io, argparse
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ap = argparse.ArgumentParser()
ap.add_argument("--handoff", default=r"F:/nds/Auto/digitizer/rnd/HANDOFF.md")
ap.add_argument("--config", default=r"F:/nds/Auto/auto/config.py")
ap.add_argument("--head", default="## 2. ЧТО СЕЙЧАС В ПРОДЕ")
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
print("\n★ таблица «ЧТО СЕЙЧАС В ПРОДЕ» сходится с `auto/config.py` до значения")

r"""_roadmap_index.py — ОГЛАВЛЕНИЕ ЖУРНАЛА ЗАМЕРОВ, КОТОРОЕ НЕ УСТАРЕВАЕТ (01.09).

ОТКУДА ВОПРОС. `ROADMAP_RECOGNITION.md` — источник истины по числам, 174 раздела и ~590 тысяч
знаков. Найти в нём нужный §6.x можно только поиском по номеру, который надо откуда-то знать.
Заказчик сформулировал это прямо: «не понимаю, какой файл за что отвечает». Резать журнал надвое
нельзя — получится два файла с одной задачей, то есть ровно та раздробленность, от которой уходим.
⇒ Файл остаётся один, но получает ОГЛАВЛЕНИЕ, и оглавление собирается машиной, а не руками.

ЧТО ДЕЛАЕТ. Читает заголовки `### <номер> <текст>`, вытаскивает у каждого:
  • номер и заголовок;
  • ЗВЁЗДЫ (★) — как автор раздела сам оценил его важность;
  • МЕТКУ СОСТОЯНИЯ: ⛔ отозван/закрыт отрицательно, ⚠ с оговоркой, ★ действующий.
и переписывает блок между маркерами в НАЧАЛЕ файла. Всё вне маркеров не трогается.

ПРОВЕРКИ, КОТОРЫЕ ПАДАЮТ (оглавление обязано быть верным, иначе оно вреднее отсутствия):
  1. ПОВТОР НОМЕРА — два раздела с одним §; в журнале, куда дописывают руками, это бывает.
  2. ДЫРА В НУМЕРАЦИИ внутри §6 — пропущенный номер печатается (не падение: раздел могли
     сознательно не заводить, но знать об этом надо).
  3. РАЗДЕЛ БЕЗ НОМЕРА — печатается отдельным списком, чтобы не потерялся.

  <ComfyUI>\python_embeded\python.exe _roadmap_index.py            # обновить оглавление
  <ComfyUI>\python_embeded\python.exe _roadmap_index.py --check    # только проверить, не писать
"""
import sys, argparse, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--file", default=r"F:\nds\Auto\ROADMAP_RECOGNITION.md")
ap.add_argument("--check", action="store_true", help="только проверить, файл не менять")
a = ap.parse_args()

BEG = "<!-- ОГЛАВЛЕНИЕ: собирается `_roadmap_index.py`, руками не править -->"
END = "<!-- /ОГЛАВЛЕНИЕ -->"

src = Path(a.file).read_text(encoding="utf-8")
body = src.split(END, 1)[1] if END in src else src

heads = [(m.start(), m.group(1).strip()) for m in re.finditer(r"^###\s+(.*)$", body, re.M)]
print(f"★ разделов найдено: {len(heads)}")

rows, unnumbered, seen = [], [], {}
for pos, title in heads:
    m = re.match(r"(\d+(?:\.\d+[a-z]?)?)\s+(.*)", title)
    if not m:
        unnumbered.append(title)
        continue
    num, rest = m.group(1), m.group(2)
    stars = len(re.match(r"[★]*", rest).group(0))
    rest = rest.lstrip("★ ").strip()
    mark = "⛔" if rest.startswith("⛔") or "ОТОЗВАН" in rest.upper() else (
        "⚠" if rest.startswith("⚠") else ("★" * min(stars, 3) if stars else ""))
    rest = rest.lstrip("⛔⚠ ").strip()
    if num in seen:
        print(f"⛔ ПОВТОР НОМЕРА §{num}: «{seen[num][:50]}» и «{rest[:50]}»")
    seen[num] = rest
    rows.append((num, mark, rest, stars))

# ⚠ номер может нести буквенный суффикс (6.11b) — он появляется, когда в журнале
# обнаружился ЗАНЯТЫЙ номер и разводить их пришлось задним числом.
def key(n):
    out = []
    for x in n.split("."):
        d = re.match(r"(\d+)([a-z]?)", x)
        out += [int(d.group(1)), d.group(2)]
    return tuple(out)
rows.sort(key=lambda r: key(r[0]), reverse=True)

six = sorted({key(n)[2] for n, *_ in rows if key(n)[0] == 6 and "." in n})
holes = [i for i in range(min(six), max(six) + 1) if i not in six] if six else []
print(f"★ §6: от 6.{min(six)} до 6.{max(six)}; пропущено номеров {len(holes)}"
      + (f": {holes[:20]}{'...' if len(holes) > 20 else ''}" if holes else ""))
if unnumbered:
    print(f"⚠ разделов БЕЗ номера: {len(unnumbered)} — в оглавление идут отдельным списком")
dups = [n for n, v in seen.items() if sum(1 for r in rows if r[0] == n) > 1]
if dups:
    print(f"⛔ ПОВТОРЯЮЩИЕСЯ НОМЕРА: {dups}")

out = [BEG, "",
       "## Оглавление журнала",
       "",
       f"Разделов: **{len(rows)}**, новые сверху. Метка: ★ действующий (звёзды — оценка автора),",
       "⚠ с оговоркой, ⛔ отозван или закрыт отрицательно — такой раздел ЦИТИРОВАТЬ НЕЛЬЗЯ,",
       "он оставлен, чтобы ошибку не повторили. Собирается `digitizer/rnd/_roadmap_index.py`.",
       "",
       "| § | | о чём |",
       "|---|---|---|"]
for num, mark, rest, _ in rows:
    t = rest.replace("|", "\\|")
    out.append(f"| **{num}** | {mark} | {t[:150]} |")
if unnumbered:
    out += ["", "**Разделы без номера** (ранние, до введения нумерации):", ""]
    out += [f"- {t[:150]}" for t in unnumbered]
out += ["", END]
block = "\n".join(out)

if a.check:
    print("★ --check: файл не изменён")
    sys.exit(0)
head = src.split(BEG, 1)[0] if BEG in src else (src.split("\n", 1)[0] + "\n\n" if src.startswith("#") else "")
if BEG not in src and src.startswith("#"):
    first, rest_txt = src.split("\n", 1)
    new = first + "\n\n" + block + "\n" + rest_txt
else:
    new = head + block + body
Path(a.file).write_text(new, encoding="utf-8", newline="\n")
print(f"★ оглавление записано: {len(block)} знаков, {len(rows)} строк")
print(f"★ файл: {len(new)} знаков (было {len(src)})")

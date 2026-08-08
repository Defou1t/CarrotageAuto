r"""_intake_survey.py — ЧТО МЫ ПОЛУЧИЛИ В НОВЫХ СКВАЖИНАХ И СКОЛЬКО ИЗ ЭТОГО ПРИГОДНО (§6.107).

⚠⚠ ЗАЧЕМ ИМЕННО ЭТОТ ЗАМЕР ПЕРВЫМ. §6.89 упёрся в то, что чистую проверку делать НЕЧЕМ: все 42
скважины архива уже в пулах, и обученная раскладка проверялась ЗАМЕНИТЕЛЯМИ (LOWO, деление на
половины). В `intake\sorted` пришли скважины, которых в `projects\Archive` нет вовсе, — то есть
впервые появляется НАСТОЯЩИЙ держанный набор. Но прежде чем на нём что-то мерить, надо знать, какая
его часть пайплайну вообще по зубам: §6.10 намерил, что 19% имён архива не разбираются ВООБЩЕ, а
разбор имени даёт приор (сколько кривых и какие) — без него лист идёт вслепую.

ЧТО СЧИТАЕТСЯ (без прогона пайплайна — только XML, имена и наличие пар):
  • пара лист↔картинка↔LAS: `find_image` / `find_las` (та же раскладка `<скв>/{wlg,img,las}`);
  • РАЗБОР ИМЕНИ: `meta.parse_name` → ожидаемые кривые (приор). Не разобралось = слепой лист;
  • ЭКСПЕРТНЫЕ КРИВЫЕ в листе: сколько кривых с реальными точками (это будущий эталон);
  • ПЕРЕСЕЧЕНИЕ СО СТАРЫМ НАБОРОМ по имени скважины — держанность обязана быть доказанной,
    а не предполагаемой: скважина, встречавшаяся в обучении, из держанного набора выбывает.

⚠ Объём печатается ИЗ СЧЁТЧИКОВ, и в конце — сверка «обработано + пропущено против длины списка»
(§6.106). Прогон по неполной выборке иначе выглядит совершенно так же.

  python _intake_survey.py [--root E:\Carrotagki_auto\intake\sorted] [--csv out.csv]
"""
import sys, re, argparse, io, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from extract_nlgx import extract, NULL
from dataset_build import find_image, find_las
import dataset as ds
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=r"E:\Carrotagki_auto\intake\sorted")
ap.add_argument("--archive", default=r"F:\nds\projects\Archive")
ap.add_argument("--csv", default="")
ap.add_argument("--limit", type=int, default=0, help="ограничить число листов (отладка)")
a = ap.parse_args()

ROOT = Path(a.root)
if not ROOT.is_dir():
    sys.exit(f"нет каталога {ROOT}")


def norm_well(s):
    """Имя скважины к сравнимому виду: регистр, дефис/подчерк, ведущие нули номера."""
    s = s.upper().replace("-", "_")
    return re.sub(r"_0*(\d+)$", lambda m: "_" + m.group(1).lstrip("0").zfill(3), s)


ARCH = Path(a.archive)
seen_wells = set()
if ARCH.is_dir():
    seen_wells = {norm_well(d.name) for d in ARCH.iterdir()
                  if d.is_dir() and (d / "wlg").is_dir()}
seen_wells.add(norm_well("Semeguniv_001"))
seen_wells.add(norm_well("Semeguniv_020"))
print(f"скважин в старом наборе (projects/Archive + Semeguniv): {len(seen_wells)}")

FILES = []
for d in sorted(ROOT.iterdir()):
    if d.is_dir() and (d / "wlg").is_dir():
        FILES += sorted((d / "wlg").glob("*.nlgx"))
if a.limit:
    FILES = FILES[: a.limit]
print(f"список: {len(FILES)} листов из {len(set(f.parent.parent.name for f in FILES))} скважин")

T = dict(листов=0, спарено=0, сLAS=0, имя_разобрано=0, кривых=0, слепых=0)
SKIP = {}
per_well = {}
rows = []

for f in FILES:
    well = f.parent.parent.name
    nw = norm_well(well)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m = extract(str(f))
    except Exception as e:
        SKIP[f"не читается nlgx ({type(e).__name__})"] = SKIP.get(
            f"не читается nlgx ({type(e).__name__})", 0) + 1
        continue
    img = find_image(f)
    las = find_las(f)
    try:
        curves = [c for c in ds.real_curves(m) if any(x != NULL for x in c["xs"])]
    except Exception:
        curves = []
    # ПРИОР ИЗ ИМЕНИ: разобралось ли имя в набор ожидаемых кривых
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fm = M.parse_filename(f)
        exp = list(getattr(fm, "expected_curves", []) or [])
    except Exception:
        exp = []
    T["листов"] += 1
    T["спарено"] += bool(img)
    T["сLAS"] += bool(las)
    T["имя_разобрано"] += bool(exp)
    T["слепых"] += (not exp)
    T["кривых"] += len(curves)
    w = per_well.setdefault(nw, dict(имя=well, листов=0, кривых=0, спарено=0, разобрано=0,
                                     держ=nw not in seen_wells))
    w["листов"] += 1; w["кривых"] += len(curves)
    w["спарено"] += bool(img); w["разобрано"] += bool(exp)
    rows.append((well, f.name, len(curves), len(exp), bool(img), bool(las)))

held = {k: v for k, v in per_well.items() if v["держ"]}
old = {k: v for k, v in per_well.items() if not v["держ"]}

W = 92
print(f"\n{'='*W}\nИТОГО: листов {T['листов']}, скважин {len(per_well)}, экспертных кривых {T['кривых']}")
# ⚠⚠ СВЕРКА СПИСКА — обязательная печать (§6.106, образец `_pool_oracle.py`).
_sk = sum(SKIP.values())
print(f"  СВЕРКА СПИСКА: обработано {T['листов']} + пропущено {_sk} = {T['листов'] + _sk} "
      f"против длины списка {len(FILES)}"
      f"   {'★ СОШЛОСЬ' if T['листов'] + _sk == len(FILES) else '⛔ НЕ СОШЛОСЬ'}")
for k, v in sorted(SKIP.items(), key=lambda q: -q[1]):
    print(f"    пропущено «{k}»: {v}")

pc = lambda n: f"{100*n/max(1,T['листов']):.0f}%"
print(f"  спарено с картинкой  {T['спарено']:>5} ({pc(T['спарено'])})")
print(f"  есть LAS             {T['сLAS']:>5} ({pc(T['сLAS'])})")
print(f"  ★ имя разобрано      {T['имя_разобрано']:>5} ({pc(T['имя_разобрано'])})   "
      f"⚠ слепых (приора нет) {T['слепых']} ({pc(T['слепых'])})")

print(f"\n{'='*W}\n★★ ДЕРЖАННЫЙ НАБОР (скважин, которых НЕТ в старом): {len(held)}")
print(f"   листов {sum(v['листов'] for v in held.values())}, "
      f"экспертных кривых {sum(v['кривых'] for v in held.values())}, "
      f"спарено {sum(v['спарено'] for v in held.values())}, "
      f"имя разобрано {sum(v['разобрано'] for v in held.values())}")
print(f"   ⚠ пересечение со старым набором: {len(old)} скважин, "
      f"{sum(v['листов'] for v in old.values())} листов — В ДЕРЖАННЫЙ НЕ БРАТЬ")
if old:
    print("     " + ", ".join(f"{v['имя']}({v['листов']})" for v in old.values()))

print(f"\n{'скважина':<26}{'листов':>7}{'кривых':>8}{'спарено':>9}{'имя разобр.':>13}")
for k, v in sorted(held.items(), key=lambda q: -q[1]["листов"])[:25]:
    print(f"{v['имя']:<26}{v['листов']:>7}{v['кривых']:>8}{v['спарено']:>9}{v['разобрано']:>13}")
if len(held) > 25:
    print(f"… и ещё {len(held)-25} скважин")

if a.csv:
    import csv
    with open(a.csv, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["well", "nlgx", "curves", "expected", "has_img", "has_las"])
        wr.writerows(rows)
    print(f"\n-> {a.csv}")

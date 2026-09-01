r"""_run_status.py — ГДЕ СЕЙЧАС ЗАМЕР, ОДНОЙ КОМАНДОЙ И БЕЗ КОНСОЛЬНЫХ ОКОН (§6.168).

ОТКУДА ВОПРОС. Состояние прогона проверялось россыпью команд: `Get-Process`, `Get-ScheduledTask`,
`ls` по каталогам, `grep` по логам. Каждый вызов `powershell.exe` открывает КОНСОЛЬНОЕ ОКНО на
экране пользователя — заказчик попросил этого не делать. И правильно: всё, что нужно, выводится
из ФАЙЛОВ, без опроса процессов.

ЧТО СЧИТАЕТ, И ПОЧЕМУ ИМЕННО ТАК:
  • ПОЛНОТА — по ЛИСТАМ, а не по числу дампов (§6.166: шард дописывает дамп, даже уронив листы;
    так фолд 2 вышел на 206 из 208, и «8 из 8 дампов» это скрыло);
  • ПАДЕНИЯ — считаются из логов шардов отдельной строкой, с указанием причины;
  • ЖИВОСТЬ — по СВЕЖЕСТИ логов шардов, а не по списку процессов: лог, тронутый минуту назад,
    означает работающий счёт вернее, чем наличие процесса с именем python.exe;
  • ТЕМП и ОСТАТОК — из меток времени каталогов выдачи.

  <ComfyUI>\python_embeded\python.exe _run_status.py
"""
import sys, argparse, pickle, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:/nds/output/taskS/ab_pickmodel")
ap.add_argument("--sheets", default=r"F:/nds/output/taskS/pickfolds/sheets_f{f}.txt")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--mode", default="M")
ap.add_argument("--alive-min", type=float, default=5.0, help="минут: лог свежее — счёт идёт")
ap.add_argument("--log", default=r"F:/nds/output/taskS/_pickmodel_tick.log")
a = ap.parse_args()

NOW = time.time()
root = Path(a.dir)
print(f"★ ЗАМЕР {root.name}  ({time.strftime('%d.%m %H:%M:%S')})")
# ⚠ ДВА РАЗНЫХ СЧЁТА, И ПУТАТЬ ИХ НЕЛЬЗЯ. «разобрано» — листы из ДАМПОВ: это истина о
# полноте, но у идущего фолда дампы ещё не сданы (шард пишет свой в самом конце).
# «выдано» — каталоги выдачи: это прогресс прямо сейчас. Первая редакция показывала
# только «разобрано» и у фолда на 259 выданных листах печатала 37.
print(f"{'фолд':<6}{'разобрано':>13}{'выдано':>9}{'дампов':>8}{'падений':>9}{'состояние':>26}")

done_sheets = want_all = 0
alive_fold = None
for f in range(a.folds):
    fo = root / f"f{f}"
    try:
        want = sum(1 for l in open(a.sheets.format(f=f), encoding="utf-8") if l.strip())
    except OSError:
        want = 0
    want_all += want
    dumps = sorted(fo.glob("ab_*of*.pkl"))
    got = set()
    if dumps:
        den = sorted({x.stem.split("of")[1] for x in dumps},
                     key=lambda q: max(y.stat().st_mtime for y in dumps if y.stem.endswith("of" + q)))[-1]
        for x in [y for y in dumps if y.stem.endswith("of" + den)]:
            try:
                got |= set(pickle.load(open(x, "rb"))["res"].get(a.mode, ({}, {}))[1])
            except Exception:
                pass
    fx = root / f"f{f}_fix"
    if fx.is_dir():
        for x in fx.glob("ab_*of*.pkl"):
            try:
                got |= set(pickle.load(open(x, "rb"))["res"].get(a.mode, ({}, {}))[1])
            except Exception:
                pass
    logs = list((fo / "logs").glob("sh*.log"))
    lines = fails = 0
    fresh = 0.0
    for L in logs:
        try:
            t = L.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines += t.count("ЧЕСТНЫХ")
        fails += t.count("ПАДЕНИЕ")
        fresh = max(fresh, L.stat().st_mtime)
    live = logs and (NOW - fresh) < a.alive_min * 60
    if live:
        alive_fold = f
    emitted = len([x for x in (fo / a.mode).iterdir() if x.is_dir()]) if (fo / a.mode).is_dir() else 0
    done_sheets += max(len(got), emitted)
    st = ("★ ГОТОВ" if want and len(got) >= want else
          ("⏳ СЧИТАЕТСЯ" if live else
           (f"⛔ НЕПОЛОН: {len(got)} из {want}" if got and want else
            ("— не начат" if not logs else "⚠ ОСТАНОВЛЕН"))))
    n_txt = f"{len(got)} из {want}" if want else "—"
    print(f"{f:<6}{n_txt:>13}{emitted:>9}{len(dumps):>8}{fails:>9}{st:>26}")

print(f"\n★ ВСЕГО: {done_sheets} листов из {want_all} ({100*done_sheets/max(1,want_all):.1f}%)")
if alive_fold is not None:
    fo = root / f"f{alive_fold}"
    md = sorted((x.stat().st_mtime for x in (fo / a.mode).iterdir() if x.is_dir()),
                reverse=True) if (fo / a.mode).is_dir() else []
    if len(md) > 20:
        span = md[0] - md[min(len(md) - 1, 200)]
        n = min(len(md) - 1, 200)
        rate = n / (span / 60) if span > 0 else 0
        try:
            want = sum(1 for l in open(a.sheets.format(f=alive_fold), encoding="utf-8") if l.strip())
        except OSError:
            want = 0
        left = max(0, want - len(md))
        print(f"★ ИДЁТ фолд {alive_fold}: темп {rate:.2f} листов/мин, осталось {left} "
              f"⇒ ≈{left/rate/60:.1f} ч" if rate else f"★ ИДЁТ фолд {alive_fold}")
else:
    print("⚠ СЧЁТ НЕ ИДЁТ: ни один лог шарда не тронут за последние "
          f"{a.alive_min:.0f} мин. Тик поднимет его по расписанию (раз в 20 мин).")

tl = Path(a.log)
if tl.is_file():
    ln = [x for x in tl.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
    print("\n★ ЛОГ ТИКА (последние 3):")
    for x in ln[-3:]:
        print(f"   {x}")

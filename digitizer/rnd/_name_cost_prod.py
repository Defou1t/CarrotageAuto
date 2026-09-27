r"""_name_cost_prod.py — ЦЕНА ИМЕНИ НА ОТГРУЖАЕМОМ ПУТИ (§6.126 мерил по пулам и просил это отдельно).

§6.126 намерил на пуловых дампах: если разрешить перестановку подписей внутри трека, честных
становится 1286 вместо 832 (+454). Но `written` в пулах — это раскладка ПРАВИЛОМ (§4), а прод
раскладывает моделью и часть перестановок уже забирает. Здесь то же самое считается по РЕАЛЬНО
ВЫДАННЫМ ФАЙЛАМ прогона: пайплайн не запускается, читается готовая выдача.

  1. С ИМЕНЕМ — как считает ветка: кривая под именем N против эталона имени N;
  2. БЕЗ ИМЕНИ — максимальное 1:1 внутри ОДНОГО ТРЕКА, кто бы как ни назывался.

⚠ Трек берётся из пуловых дампов (slot → track): переставлять подписи между треками нельзя, это
разные дорожки бланка. Ограничение делает оценку КОНСЕРВАТИВНОЙ.

  <ComfyUI>\python_embeded\python.exe _name_cost_prod.py --dir F:/nds/output/taskS/ab_gate_fix --mode A
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:/nds/output/taskS/ab_gate_fix")
ap.add_argument("--mode", default="A", help="подкаталог режима (A = прод)")
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
# ★★ ПАРНОЕ СРАВНЕНИЕ ДВУХ РЕЖИМОВ ОДНОГО ПРОГОНА, С ШУМОМ (§6.118, §6.120 — правило 3 §6).
# ⚠⚠ ЗАЧЕМ ЗАВЕДЕНО 24.08. Оба числа §6.141 («−41 с именем», «+61 без имени») переписывались в
# роадмап РУКАМИ из разных источников, и пересчёт одним проходом дал −41 и **+2**, а не +61.
# Знаковый тест при этом остаётся значимым: направление есть, а СУММЫ нет. Пока оба счёта не
# считаются ОДНИМ проходом по одним и тем же каталогам, такие расхождения невидимы.
ap.add_argument("--mode2", default="", help="второй режим: ПАРНОЕ сравнение по листам с тестами шума")
ap.add_argument("--perm", type=int, default=20000, help="перестановок в тесте")
ap.add_argument("--seed", type=int, default=20260824)
# ★ ВЫГРУЗКА ПОЛИСТНЫХ СЧЁТОВ. Ведущий счёт ветки — БЕЗЫМЯННЫЙ, и живёт он только здесь: дампы
# `_trace_prod_ab` держат ИМЕННОЙ (§6.126). Любому разбору, которому нужен ведущий счёт по листам
# (например §6.156 — разложение по держанности скважины), пришлось бы ПЕРЕПИСАТЬ привязку 1:1, а
# две реализации одной метрики неизбежно разъедутся. ⇒ отдаём посчитанное, а не повторяем его.
ap.add_argument("--dump", default="", help="pickle с {лист: (с_именем, без_имени, кривых)}")
a = ap.parse_args()
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


def match(rows, cols, ok):
    pair = {}

    def try_(r, seen):
        for c in cols:
            if not ok.get((r, c)) or c in seen:
                continue
            seen.add(c)
            if c not in pair or try_(pair[c], seen):
                pair[c] = r
                return True
        return False
    return sum(1 for r in rows if try_(r, set()))


# треки из пуловых дампов: имя листа -> {кривая: трек}
TRACK, seen = {}, set()
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        if f.stem in seen:
            continue
        seen.add(f.stem)
        d = pickle.load(open(f, "rb"))
        TRACK[d["name"]] = {s["name"]: s["track"] for s in d["slots"]}
print(f"треков загружено для {len(TRACK)} листов")

# исходные разметки: имя -> путь
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q

# каталог назван <stem[:40]>_<md5(stem)[:8]> — восстанавливаем по всем известным листам
BYDIR = {}
for nm, q in SRC.items():
    BYDIR[f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"] = nm


def count(mode):
    """→ (per_sheet {лист: (с именем, без имени, экспертных)}, разобрано, пропущено, каталогов)."""
    MD = Path(a.dir) / mode
    dirs = [d for d in MD.iterdir() if d.is_dir()] if MD.is_dir() else []
    print(f"каталогов выдачи в {MD}: {len(dirs)}")
    per, sheets, skipped = {}, 0, 0
    for d in dirs:
        nm = BYDIR.get(d.name)
        if nm is None:
            skipped += 1; continue
        got = next(iter(sorted(d.glob("*_auto.nlgx"))), None)
        if got is None:
            skipped += 1; continue
        src = SRC[nm]
        gts = {c["name"]: dense(c) for c in extract(str(src))["curves"]
               if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        wr = {c["name"]: dense(c) for c in extract(str(got))["curves"]
              if M.mnem_root(c["name"]) != "DA"}
        tr_map = TRACK.get(nm, {})
        sheets += 1
        nmd = sum(1 for k, gt in gts.items() if k in wr and HON(*err(wr[k], gt)))
        fre = 0
        for t in set(tr_map.get(k) for k in gts) | set(tr_map.get(k) for k in wr):
            if t is None:
                continue
            G = [k for k in gts if tr_map.get(k) == t]
            W = [k for k in wr if tr_map.get(k) == t]
            if not G or not W:
                continue
            ok = {(g, w): HON(*err(wr[w], gts[g])) for g in G for w in W}
            fre += match(G, W, ok)
        per[nm] = (nmd, fre, len(gts))
    print(f"★ СВЕРКА {mode}: листов разобрано {sheets} + пропущено {skipped} = {sheets+skipped} "
          f"против {len(dirs)} каталогов   "
          f"{'★ СОШЛОСЬ' if sheets+skipped == len(dirs) else '⛔ НЕ СОШЛОСЬ'}")
    return per, sheets, skipped, len(dirs)


def tests(diff):
    """Парный перестановочный (смена знака) + знаковый по расходящимся листам."""
    d = np.array([v for v in diff if v], float)
    if not len(d):
        return 0.0, 0.0, 1.0, 1.0
    rng = np.random.default_rng(a.seed)
    obs = float(d.sum())
    sign = rng.integers(0, 2, size=(a.perm, len(d))) * 2 - 1
    null = (sign * d).sum(1)
    p_perm = float((np.abs(null) >= abs(obs)).mean())
    up = int((d > 0).sum()); dn = int((d < 0).sum())
    n = up + dn
    if n:
        from math import comb
        k = max(up, dn)
        p_sign = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
    else:
        p_sign = 1.0
    return obs, float(null.std()), p_perm, p_sign


perA, shA, skA, dA = count(a.mode)
named = sum(v[0] for v in perA.values()); free = sum(v[1] for v in perA.values())
curves = sum(v[2] for v in perA.values())
print(f"экспертных кривых {curves}")
print(f"\n  честных С ИМЕНЕМ (как считает ветка):  {named}  ({100*named/max(1,curves):.1f}%)")
print(f"  честных БЕЗ ИМЕНИ (1:1 внутри трека):  {free}  ({100*free/max(1,curves):.1f}%)")
print(f"\n★★ ЦЕНА ИМЕНИ НА ОТГРУЗКЕ: {free-named:+d} кривых "
      f"({100*(free-named)/max(1,named):+.1f}% к нынешнему счёту)")
print(f"⚠ Оракульный потолок (лучшую перестановку знает только эталон), не прирост.")

if a.dump:
    import pickle as _pk
    _d = {a.mode: perA}
    if a.mode2:
        _d[a.mode2] = count(a.mode2)[0]
    _tmp = a.dump + ".tmp"                  # ⛔ 27.09 (разбор): атомарно — снятый на записи процесс не оставит обрезок
    with open(_tmp, "wb") as _fh:
        _pk.dump(_d, _fh)
    import os as _os
    _os.replace(_tmp, a.dump)
    print("")
    print(f"★ полистные счёты выгружены: {a.dump}  "
          f"({', '.join(f'{k}: {len(v)} листов' for k, v in _d.items())})")

if a.mode2:
    perB, shB, skB, dB = count(a.mode2)
    both = sorted(set(perA) & set(perB))
    print(f"\n{'='*78}\n★★ ПАРНОЕ СРАВНЕНИЕ {a.mode} → {a.mode2} НА {len(both)} ЛИСТАХ, "
          f"ОБРАБОТАННЫХ ОБОИМИ\n{'='*78}")
    print(f"⚠ {a.mode}: {shA} листов, {a.mode2}: {shB}; общих {len(both)}")
    for j, what in ((0, "С ИМЕНЕМ (метрика ветки)"), (1, "БЕЗ ИМЕНИ (критерий Эдуарда)")):
        A = sum(perA[n][j] for n in both); B = sum(perB[n][j] for n in both)
        diff = [perB[n][j] - perA[n][j] for n in both]
        obs, sd, pp, ps = tests(diff)
        up = sum(1 for v in diff if v > 0); dn = sum(1 for v in diff if v < 0)
        print(f"\n  {what}:  {A} → {B}   Δ = {B-A:+d}")
        print(f"    листов ↑{up} / ↓{dn} (совпало {len(both)-up-dn}); SD нуля {sd:.1f}; "
              f"перестановочный p = {pp:.4f}; знаковый p = {ps:.5f}")
    print(f"\n⚠ Оба счёта посчитаны ОДНИМ проходом по одним каталогам — суммы и тесты сходятся "
          f"по построению. Переписывать их в роадмап порознь из разных прогонов нельзя (§6 правило 2).")

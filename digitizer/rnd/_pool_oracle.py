r"""_pool_oracle.py — СКОЛЬКО ЧЕСТНЫХ КРИВЫХ БЫЛО В ПУЛЕ, А СКОЛЬКО ПОПАЛО В ФАЙЛ.

ЗАЧЕМ (§6.78). На выданных файлах трассы почти не связаны с назначенными им кривыми (корреляция
0.15), а перестановка УЖЕ ЗАПИСАННЫХ почти ничего не даёт (§6.76: 6 → 7 честных). Значит в файл
попадают не те трассы. При этом на пути 2 хорошие трассы В ПУЛЕ есть (потолок 122 → 152 из 227).
Здесь то же самое меряется НА ОТГРУЖАЕМОМ ПУТИ, где этого не делали ни разу.

КАК. `emit._map_lines_to_slots` перехватывается: он получает ПУЛ всех трасс листа и возвращает
раскладку по слотам. Считаем три числа одной метрикой §6.36:
  ЗАПИСАНО     — честные среди того, что реально ушло в файл (слот X против кривой X);
  ОПТИМУМ 1:1  — лучшее взаимно-однозначное назначение ПУЛА на кривые (что мог бы дать идеальный
                 отбор при том же контракте «одна трасса — один слот»);
  ПОТОЛОК      — лучшая трасса пула для каждой кривой без требования 1:1 (что вообще есть в пуле).
Разрыв ЗАПИСАНО → ОПТИМУМ и есть цена отбора K из N на отгрузке.

⚠ БЕЗ МОСТИКА (§6.75): в проде разрывы пишутся как NULL, поэтому метрика тоже без него —
иначе считалось бы то, чего в файле нет.
⚠ classify НЕ подменяется: это отгружаемый путь, а не путь 2.
⚠ Режим селектора пиннится флагом --seq, а не наследуется из умолчаний (§6.71).

  <ComfyUI>\python_embeded\python.exe _pool_oracle.py --files <nlgx…> [--seq имя.pt]
"""
import sys, io, argparse, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from _multi_replica_probe import dense
from auto import meta as M
from auto import emit as emit_mod
from auto.pipeline import run as pipe_run
from auto.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="+", default=[])
ap.add_argument("--files-from", default="",
                help="файл со списком путей, ПО ОДНОМУ НА СТРОКУ (utf-8). ⚠ Списки надо передавать "
                     "так или массивом оболочки — подстановка рвёт имена с пробелом и запятой")
ap.add_argument("--seq", default="", help="чекпойнт селектора; пусто = жадный выбор")
ap.add_argument("--out", default=r"F:\nds\output\taskS\pool_oracle")
ap.add_argument("--dump", default="", help="каталог: сохранить ПУЛ трасс + рамку + эталон на лист. "
                                           "Тогда правила раскладки перебираются ОФЛАЙН за секунды, "
                                           "а не полным прогоном пайплайна на каждый вариант")
a = ap.parse_args()

POOL = {}
_orig_map = emit_mod._map_lines_to_slots


CAPM = {}


# ⚠ `*rest` — ПРОБРОС ХВОСТА СИГНАТУРЫ. `emit._map_lines_to_slots` получил пятым аргументом `cv`
# (§6.105, ручка `slot_order`), и перехват с жёсткими четырьмя падал бы `TypeError` ВНУТРИ
# `try/except` пайплайна: прогон бы отработал, напечатал «ПАДЕНИЕ» на каждый лист и выдал пустой пул.
def capture(traces, model, frame, mnemonics_path, *rest):
    POOL["all"] = list(traces)
    CAPM["model"], CAPM["frame"], CAPM["mn"] = model, frame, mnemonics_path
    r = _orig_map(traces, model, frame, mnemonics_path, *rest)
    POOL["written"] = dict(r)
    return r


emit_mod._map_lines_to_slots = capture


def err(ours, gt):
    com = [y for y in ours if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(ours[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


def leaked(tr, raw):
    """Контроль копирования эксперта (§6.36): совпавшая сетка строк и значения = утечка."""
    er = np.array([raw["top_y"] + i for i, x in enumerate(raw["xs"]) if x != NULL])
    ex = np.array([x for x in raw["xs"] if x != NULL], float)
    oy = np.array(sorted(tr)); ox = np.array([tr[y] for y in oy], float)
    if len(oy) == len(er) and np.array_equal(oy, er) and np.allclose(ox, ex):
        return True
    com = np.intersect1d(oy, er)
    if len(com) >= 30:
        om = dict(zip(oy.tolist(), ox.tolist())); em = dict(zip(er.tolist(), ex.tolist()))
        if np.mean([abs(om[y] - em[y]) < 1e-9 for y in com]) > 0.5:
            return True
    return False


HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9
T = {"написано": 0, "оптимум": 0, "потолок": 0, "кривых": 0, "пул": 0, "листов": 0}
SKIP = {}

# ⚠⚠ СПИСОК ПУТЕЙ — ФАЙЛОМ ИЛИ МАССИВОМ, НИКОГДА ПОДСТАНОВКОЙ В ОБОЛОЧКУ. Дважды за ветку
# `--files $(cat list)` ломал прогон, и ОБА раза прогон завершался успешно с правдоподобными
# числами: (1) список, записанный из Python на Windows, нёс `\r` в конце каждой строки → `OSError`
# на каждом листе; (2) имена вида `KREMEN_083_BK, IK_0800-…` содержат ПРОБЕЛ И ЗАПЯТУЮ, и 118 путей
# оболочка разбила на 178 обрывков — 80 «нет картинки», а посчитались только имена без пробелов.
# Поэтому `\r` снимается явно, а в конце ОБЯЗАТЕЛЬНО печатается сверка «обработано + пропущено
# против длины списка»: без неё обе поломки невидимы.
FILES = list(a.files)
if a.files_from:
    for ln in Path(a.files_from).read_text(encoding="utf-8").splitlines():
        ln = ln.strip().strip('"')
        if ln:
            FILES.append(ln)
if not FILES:
    sys.exit("не задан ни --files, ни --files-from")
absent = [f for f in FILES if not Path(f).exists()]
if absent:
    print(f"⚠ НЕТ НА ДИСКЕ: {len(absent)} путей из {len(FILES)}")
    for f in absent[:5]:
        print(f"    {f}")
print(f"список: {len(FILES)} путей" + (f" (из {a.files_from})" if a.files_from else ""))

for f in FILES:
    n = Path(f)
    img = find_image(n)
    if not img:
        SKIP["нет картинки"] = SKIP.get("нет картинки", 0) + 1
        print(f"  {n.stem[:44]:<46} нет картинки"); continue
    cfg = Config(); cfg.out = Path(a.out) / n.stem[:40]
    cfg.cv.seq_model = a.seq                      # §6.71: режим задаёт стенд
    POOL.clear()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
    except Exception as e:
        SKIP["падение"] = SKIP.get("падение", 0) + 1
        print(f"  {n.stem[:44]:<46} ПАДЕНИЕ {type(e).__name__}: {e}"); continue
    G = extract(str(n))
    raws = {c["name"]: c for c in G["curves"]
            if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    gts = {nm: dense(c) for nm, c in raws.items()}
    if not gts:
        SKIP["нет экспертных кривых"] = SKIP.get("нет экспертных кривых", 0) + 1
        continue
    pool = [tr for _, tr in POOL.get("all", [])]
    written = {nm: tr for nm, (_, tr) in POOL.get("written", {}).items()}

    wr = sum(1 for nm, gd in gts.items()
             if nm in written and HON(*err(written[nm], gd)) and not leaked(written[nm], raws[nm]))
    # ПОТОЛОК: лучшая трасса пула на каждую кривую, без 1:1
    cap = 0
    pairs = []
    for nm, gd in gts.items():
        best = None
        for ti, tr in enumerate(pool):
            m, c = err(tr, gd)
            if m is None or leaked(tr, raws[nm]):
                continue
            pairs.append((m, c, nm, ti))
            if best is None or (c >= 0.9, -m) > (best[1] >= 0.9, -best[0]):
                best = (m, c)
        cap += bool(best and HON(*best))
    # ОПТИМУМ 1:1: жадно по возрастанию ошибки среди пар, проходящих порог покрытия
    pairs.sort(key=lambda q: (q[1] < 0.9, q[0]))
    got, used, opt = set(), set(), 0
    for m, c, nm, ti in pairs:
        if nm in got or ti in used:
            continue
        got.add(nm); used.add(ti)
        opt += HON(m, c)
    if a.dump:
        # ★ Сохраняем ровно то, что нужно раскладке: трассы с их признаками (цвет, класс,
        # x_center, трек), слоты рамки и эталон. Пайплайн больше не нужен — перебор правил
        # становится офлайновым и стоит секунды вместо часа на вариант.
        import pickle
        d = Path(a.dump); d.mkdir(parents=True, exist_ok=True)
        lines = [{"track": L.track_index, "color": L.color, "behavior": L.behavior,
                  "x_center": float(L.x_center), "conf": L.confidence, "tr": tr}
                 for L, tr in POOL.get("all", [])]
        # ⚠ В дамп кладутся ТОЛЬКО чистые данные: объект рамки тянул бы за собой пакет `auto` при
        # чтении. Всё, что раскладке нужно от рамки, — это трек слота, поэтому он вычисляется
        # здесь и сохраняется числом. Приор цвета/класса из словаря — тоже здесь.
        slots = []
        for c in CAPM["model"].get("curves", []):
            nmc = c.get("name", "")
            if M.mnem_root(nmc) == "DA":
                continue
            info = M.curve_info(nmc, CAPM["mn"])
            slots.append({"name": nmc, "track": emit_mod._slot_track(CAPM["model"], c, CAPM["frame"]),
                          "color": info["color"], "class": info["class"]})
        pickle.dump({"lines": lines, "slots": slots, "gts": gts, "raws": raws,
                     "written": {k: v[1] for k, v in POOL.get("written", {}).items()},
                     "name": n.name},
                    open(d / f"{n.stem[:60]}.pkl", "wb"))
    print(f"  {n.stem[:42]:<44} кривых {len(gts):>2}  пул {len(pool):>3}  "
          f"записано {wr}  оптимум1:1 {opt}  потолок {cap}")
    T["написано"] += wr; T["оптимум"] += opt; T["потолок"] += cap
    T["кривых"] += len(gts); T["пул"] += len(pool); T["листов"] += 1

print(f"\n{'='*84}\nИТОГО: листов {T['листов']}, кривых {T['кривых']}, трасс в пулах {T['пул']}, "
      f"режим seq={a.seq or '—'}")
# ⚠⚠ СВЕРКА СПИСКА — ОБЯЗАТЕЛЬНАЯ ПЕЧАТЬ, а не отладка: обе поломки передачи списка (см. выше)
# давали прогон, который ОТРАБАТЫВАЛ и выдавал правдоподобные числа по неполной выборке.
_skipped = sum(SKIP.values())
_ok = (T["листов"] + _skipped) == len(FILES)
print(f"  СВЕРКА СПИСКА: обработано {T['листов']} + пропущено {_skipped} = "
      f"{T['листов'] + _skipped} против длины списка {len(FILES)}"
      f"   {'★ СОШЛОСЬ' if _ok else '⛔ НЕ СОШЛОСЬ'}")
for k, v in sorted(SKIP.items(), key=lambda q: -q[1]):
    print(f"    пропущено «{k}»: {v}")
print(f"  ЗАПИСАНО в файл      {T['написано']:>3} честных")
print(f"  ОПТИМУМ 1:1 из пула  {T['оптимум']:>3}   ← цена отбора K из N: "
      f"{T['оптимум'] - T['написано']:+d}")
print(f"  ПОТОЛОК пула         {T['потолок']:>3}   ← цена требования 1:1: "
      f"{T['потолок'] - T['оптимум']:+d}")

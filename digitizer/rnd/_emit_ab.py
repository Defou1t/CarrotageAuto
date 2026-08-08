r"""_emit_ab.py — ДВА ПУТИ ЭМИССИИ ЛОБ В ЛОБ, ПО ВЫДАННЫМ ФАЙЛАМ (§6.55).

§6.54 показал, что гейт уверенности пропускает к трассировке 23% линий. Но §6.45 сделал ВТОРОЙ
путь эмиссии (`_emit_traces`, постановка §6.33 «имена не нужны»), и он форсирует AUTO. Возник
вопрос, на который до сих пор отвечали качественно: НАСКОЛЬКО эти два пути расходятся в ВЫДАЧЕ.

ЧТО СРАВНИВАЕТСЯ (оба раза — итоговый .nlgx, а не трассы внутри пайплайна):
  A) ПРОД: `auto.pipeline.run` целиком → `emit.emit_into_frame` → <stem>_auto.nlgx.
     Настоящая `classify`: часть линий уходит в FLAG и до выдачи не доходит.
  B) §6.33: тот же прогон, но `classify` форсирует AUTO, а слоты заполняет `emit_traces`
     (очистка всех слотов, своё окно строк, отбор §6.49, мостик §6.53).

МЕТРИКА — §6.36 без изменений: назначение 1-к-1, честная = med<=3 И cov>=0.9, контроль
копирования эксперта. Имена не проверяются (§6.33).

⚠ Сравнение честное только по ФАЙЛАМ: у пути A свои слоты и своё окно строк, у B свои, и
именно это и есть предмет спора.

  python _emit_ab.py [--n 12]
"""
import sys, io, argparse, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image
from _multi_replica_probe import dense
from _emit_traces import emit_traces
from auto import confidence as CM, emit as EM
from auto.config import Config
from auto.pipeline import run as pipe_run

OUT = Path(r"F:\nds\output\taskS\emit_ab")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--cache", default=r"F:\nds\output\taskS\pick_gate\cache")
ap.add_argument("--seq", default="", help="чекпойнт селектора; пусто = ЖАДНЫЙ trace2d (§6.106)")
a = ap.parse_args()


def Cfg():
    """★ §6.106: ПУТЬ ТРАССИРОВКИ ЗАДАЁТ СТЕНД, А НЕ УМОЛЧАНИЕ. `CVParams.seq_model` по умолчанию
    непустой, и при доступном torch пайплайн ведёт линии ОБУЧЕННЫМ СЕЛЕКТОРОМ. Стенд, наследующий
    умолчание, меряет РАЗНЫЙ алгоритм на разных машинах — ровно ловушка `_slot_prod_ab.py`."""
    c = Config(); c.cv.seq_model = a.seq
    return c


print(f"★ режим трассировки: {a.seq or '— ЖАДНЫЙ trace2d (--seq пусто)'}")

_classify = CM.classify
TR = {}
_map = EM._map_lines_to_slots


def cap(t, mo, fr, mn, *rest):                      # *rest: §6.105, пятый аргумент `cv`
    TR["all"] = list(t)
    return _map(t, mo, fr, mn, *rest)


EM._map_lines_to_slots = cap


def allauto(sheet, *ar, **kw):
    r = _classify(sheet, *ar, **kw)
    for L in sheet.lines:
        L.confidence = "AUTO"
    return r


def honest_of_file(path, GM, raw):
    """Честные кривые в ВЫДАННОМ файле. Трассы берём из него же, не из пайплайна."""
    A = extract(str(path))
    got_tr = {}
    for c in A.get("curves", []):
        axs = {c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL}
        if len(axs) >= 30:
            got_tr[c["name"]] = axs
    pairs = []
    for nm, gm in GM.items():
        for sn, tr in got_tr.items():
            com = [y for y in tr if y in gm]
            if len(com) < 30:
                continue
            dd = np.array([abs(tr[y] - gm[y]) for y in com])
            pairs.append((float(np.median(dd)), len(com) / len(gm), nm, sn))
    pairs.sort(key=lambda q: (q[1] < 0.9, q[0]))
    asg, used = {}, set()
    for md, cv, nm, sn in pairs:
        if nm in asg or sn in used:
            continue
        asg[nm] = (md, cv, sn); used.add(sn)
    hon = 0
    for nm, (md, cv, sn) in asg.items():
        g = raw[nm]
        er = np.array([g["top_y"] + i for i, x in enumerate(g["xs"]) if x != NULL])
        oy = np.array(sorted(got_tr[sn]))
        if len(oy) == len(er) and np.array_equal(oy, er):
            continue                                   # контроль: сетка = экспертная
        if md <= 3 and cv >= 0.9:
            hon += 1
    return hon, len(got_tr)


ARCH = Path(r"F:\nds\projects\Archive")
idx = {}
for wlg in ARCH.glob("*/wlg"):
    for f in wlg.glob("*.nlgx"):
        idx.setdefault(f.name, f)
import pickle
sheets = []
for fp in sorted(Path(a.cache).glob("*.pkl")):
    q = pickle.load(open(fp, "rb"))
    src = idx.get(q["name"])
    if src and q["GM"]:
        sheets.append(src)
sheets = sheets[:a.n]

print(f"листов: {len(sheets)}   метрика §6.36 по ВЫДАННЫМ файлам\n")
# ★ ВАРИАНТ C РАЗЛАГАЕТ ВЫИГРЫШ: настоящая classify (гейт РАБОТАЕТ) + эмиссия `emit_traces`.
# A→C = вклад ЭМИССИИ (очистка слотов, своё окно строк, отбор §6.49, мостик §6.53),
# C→B = вклад ВЫКЛЮЧЕННОГО ГЕЙТА. Это решает, что можно вносить без смены постановки.
print(f"{'лист':<42}{'кривых':>7}{'A прод':>8}{'C эмис':>8}{'D флаг':>8}{'B оба':>7}")
tA = tB = tC = tD = tc = 0
SKIP, done = {}, 0                      # §6.106: сверка списка — обязательная печать, не отладка
for n in sheets:
    img = find_image(n)
    if not img:
        SKIP["нет картинки"] = SKIP.get("нет картинки", 0) + 1
        continue
    m = extract(str(n))
    raw = {c["name"]: c for c in ds.real_curves(m) if any(x != NULL for x in c["xs"])}
    GM = {}
    for nm, g in raw.items():
        d = dense(g)
        if len(d) >= 50:
            GM[nm] = {y: d[y] for y in sorted(d)}
    if not GM:
        SKIP["нет эталонных кривых"] = SKIP.get("нет эталонных кривых", 0) + 1
        continue
    done += 1
    hA = hB = 0
    # A: прод целиком, настоящая classify
    CM.classify = _classify
    cfg = Cfg(); cfg.out = OUT / "A" / n.stem[:30]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, res = pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
        pa = Path(res.get("nlgx", ""))
        if pa.is_file():
            hA, _ = honest_of_file(pa, GM, raw)
    except Exception as e:
        print(f"{n.name[:40]:<42} A ПАДЕНИЕ {type(e).__name__}")
    # C: настоящая classify + emit_traces
    CM.classify = _classify
    TR.clear()
    cfg = Cfg(); cfg.out = OUT / "C" / n.stem[:30]
    hC = 0
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
            ours = [tr for _, tr in TR.get("all", [])]
            pc, _w = emit_traces(n, ours, OUT / "C" / n.stem[:30], model=m, image=img)
        hC, _ = honest_of_file(pc, GM, raw)
    except Exception as e:
        print(f"{n.name[:40]:<42} C ПАДЕНИЕ {type(e).__name__}: {e}")
    # D: гейт ВЫКЛЮЧЕН флагом конфига, но эмиссия ШТАТНАЯ (`emit_into_frame`).
    # ★ Недостающая клетка матрицы: если D ≈ B, весь выигрыш берётся одним флагом, без смены
    # формата выдачи и без вопроса об именах.
    CM.classify = _classify
    cfg = Cfg(); cfg.out = OUT / "D" / n.stem[:30]; cfg.cv.trace_flagged = True
    hD = 0
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, res = pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
        pd_ = Path(res.get("nlgx", ""))
        if pd_.is_file():
            hD, _ = honest_of_file(pd_, GM, raw)
    except Exception as e:
        print(f"{n.name[:40]:<42} D ПАДЕНИЕ {type(e).__name__}: {e}")
    # B: §6.33 — форсированный AUTO + emit_traces
    CM.classify = allauto
    TR.clear()
    cfg = Cfg(); cfg.out = OUT / "B" / n.stem[:30]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
            ours = [tr for _, tr in TR.get("all", [])]
            pb, _w = emit_traces(n, ours, OUT / "B" / n.stem[:30], model=m, image=img)
        hB, _ = honest_of_file(pb, GM, raw)
    except Exception as e:
        print(f"{n.name[:40]:<42} B ПАДЕНИЕ {type(e).__name__}: {e}")
    tA += hA; tB += hB; tC += hC; tD += hD; tc += len(GM)
    print(f"{n.name[:40]:<42}{len(GM):>7}{hA:>8}{hC:>8}{hD:>8}{hB:>7}"
          + ("   ★" if hB > hA else ("   ✗" if hB < hA else "")))
CM.classify = _classify
# ⚠⚠ СВЕРКА СПИСКА (§6.106, образец `_pool_oracle.py`): без неё прогон по НЕПОЛНОЙ выборке
# завершается успешно и печатает правдоподобные числа.
_sk = sum(SKIP.values())
print(f"\nСВЕРКА СПИСКА: обработано {done} + пропущено {_sk} = {done + _sk} против длины списка "
      f"{len(sheets)}   {'★ СОШЛОСЬ' if done + _sk == len(sheets) else '⛔ НЕ СОШЛОСЬ'}")
for k, v in sorted(SKIP.items(), key=lambda q: -q[1]):
    print(f"    пропущено «{k}»: {v}")
print(f"\n{'ИТОГО':<42}{tc:>7}{tA:>8}{tB:>9}")
if tc:
    print(f"честных: ПРОД {100*tA/tc:.0f}%, путь §6.33 {100*tB/tc:.0f}%")

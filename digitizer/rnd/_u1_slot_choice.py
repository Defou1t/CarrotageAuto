r"""_u1_slot_choice.py — ПОСЛЕДНЕЕ ЗВЕНО: САЖАЕТ ЛИ `emit` В СЛОТ ЛУЧШУЮ ИЗ ДОСТУПНЫХ ЛИНИЙ.

Что уже исключено как причина разрыва «стенд 15.9px / прод 586px»: смещение полосы U1 (§6.30:
med 65px, в допуске §6.29), ширина полосы (med 608px — по оси 1 терпимо), выбор цвета (стоит ~7%
на базе), эскалация полосы до трека (заглушение НЕ помогает, селектору даже хуже).

Осталось измеренное, но не доведённое: 67% записанных трасс ближе к ЧУЖОЙ кривой
(`_u1_map_audit.py`). На стенде это невозможно — там у каждой кривой полоса из её же GT. В проде
слот получает линию по классу/цвету/порядку (`emit._map_lines_to_slots`), и вопрос ровно один:

  линия, которую emit посадил в слот, — ЛУЧШАЯ ли из доступных по попаданию в кривую?

Если нет (а лучшая при этом была) — узкое место в ВЫБОРЕ ЛИНИИ, и это чинится геометрией
(у слота есть своя scale-axis), без всякого ML.

  <ComfyUI>\python_embeded\python.exe _u1_slot_choice.py     # (torch не нужен, но пути общие)
"""
import sys, io, json, contextlib
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
from auto.pipeline import run as pipe_run
from auto.config import Config
from auto import confidence as confidence_mod
from auto import emit as emit_mod
from _relatch_bench import SH, ARCH

OUT = Path(r"F:\nds\output\taskS\slot_choice")

_classify = confidence_mod.classify


def classify_all_auto(sheet, *args, **kw):
    r = _classify(sheet, *args, **kw)
    for L in sheet.lines:
        L.confidence = "AUTO"                 # без этого доступна 1 линия и выбирать не из чего
    return r


confidence_mod.classify = classify_all_auto

_map = emit_mod._map_lines_to_slots
LOG = {}


def map_logged(traces, model, frame, mnemonics_path):
    m = _map(traces, model, frame, mnemonics_path)
    LOG["chosen"] = {nm: float(L.x_center) for nm, (L, _) in m.items()}
    LOG["offered"] = [float(L.x_center) for L, _ in traces]
    return m


emit_mod._map_lines_to_slots = map_logged

print(f"{'скважина':<12}{'слот':<8}{'взята x':>9}{'GT x':>8}{'|взята-GT|':>11}"
      f"{'лучшая|Δ|':>10}{'ранг':>6}  вердикт")
rows = []
for rel in SH:
    n = ARCH / rel
    well = n.parent.parent.name
    img = find_image(n)
    LOG.clear()
    cfg = Config(); cfg.out = OUT / well
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
    except Exception as e:
        print(f"{well:<12} ERR {type(e).__name__}: {e}"); continue
    mo = extract(str(n))
    gts = [c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])]
    offered = np.array(LOG.get("offered", []))
    for g in gts:
        d = dense(g)
        if len(d) < 50:
            continue
        gmed = float(np.median([d[y] for y in sorted(d)]))
        ch = LOG.get("chosen", {}).get(g["name"])
        if ch is None or not len(offered):
            print(f"{well:<12}{g['name'].split()[0]:<8}   слот не заполнен"); continue
        d_ch = abs(ch - gmed)
        dd = np.abs(offered - gmed)
        d_best = float(dd.min())
        rank = int((dd < d_ch - 1e-6).sum()) + 1        # 1 = взята лучшая из предложенных
        rows.append((d_ch, d_best, rank, len(offered)))
        v = "✓ взята лучшая" if rank == 1 else f"✗ была линия в {d_best:.0f}px"
        print(f"{well:<12}{g['name'].split()[0]:<8}{ch:>9.0f}{gmed:>8.0f}{d_ch:>11.0f}"
              f"{d_best:>10.0f}{rank:>6}  {v}")

if rows:
    dch = np.array([r[0] for r in rows]); dbe = np.array([r[1] for r in rows])
    rk = np.array([r[2] for r in rows])
    print(f"\nслотов: {len(rows)}")
    print(f"  взята ЛУЧШАЯ из предложенных линий: {100*(rk==1).mean():.0f}%")
    print(f"  |взята − GT|: med {np.median(dch):.0f}px      |лучшая − GT|: med {np.median(dbe):.0f}px")
    print(f"  в допуске §6.29 (<=100px): взятая {100*(dch<=100).mean():.0f}%   "
          f"лучшая доступная {100*(dbe<=100).mean():.0f}%")
    print("\nЧитать: если «лучшая доступная» в допуске СИЛЬНО чаще взятой — виноват выбор линии")
    print("под слот (emit), а не U1 и не трассировщик. Чинится геометрией, без ML.")

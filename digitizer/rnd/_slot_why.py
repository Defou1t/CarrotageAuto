r"""_slot_why.py — ПОЧЕМУ верная трасса не попала в слот.

§6.79 показал: в пуле 15 честных кривых, в файл уходит 5. Отбор теряет две трети, причём
требование 1:1 не стоит ничего (оптимум = потолок). Значит дело в правиле `_map_lines_to_slots`.
Правило простое (`auto/emit.py:124-155`): внутри трека пары (слот, линия) сортируются по
(совпадение класса, `x_center` линии), плюс ФИЛЬТР ЦВЕТА — если на треке есть линия того цвета,
который слот ожидает по словарю, то линии другого цвета в этот слот не допускаются.

Отсюда ровно три способа потерять верную трассу, и они различимы:
  ЦВЕТ    — верная линия отсеяна фильтром цвета (слот ждёт red, линия чёрная);
  КЛАСС   — верная линия проиграла по совпадению класса (SP против RES по `behavior`);
  ПОРЯДОК — оба признака совпали, но `x_center` поставил вперёд другую линию.
⚠ Четвёртый случай — верная линия вообще не дошла до `_map_lines_to_slots` (не трассировалась):
считается отдельно, к правилу отбора отношения не имеет.

  <ComfyUI>\python_embeded\python.exe _slot_why.py --files <nlgx…> [--seq имя.pt]
"""
import sys, io, argparse, contextlib, collections
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
ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--seq", default="")
ap.add_argument("--out", default=r"F:\nds\output\taskS\slot_why")
a = ap.parse_args()

CAP = {}
_orig = emit_mod._map_lines_to_slots


def capture(traces, model, frame, mnemonics_path, *rest):   # *rest: §6.105, пятый аргумент `cv`
    CAP["traces"] = list(traces)
    CAP["model"] = model
    CAP["frame"] = frame
    CAP["mn"] = mnemonics_path
    r = _orig(traces, model, frame, mnemonics_path, *rest)
    CAP["written"] = dict(r)
    return r


emit_mod._map_lines_to_slots = capture


def err(ours, gt):
    com = [y for y in ours if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(ours[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9
reasons = collections.Counter()
detail = []

for f in a.files:
    n = Path(f)
    img = find_image(n)
    if not img:
        continue
    cfg = Config(); cfg.out = Path(a.out) / n.stem[:40]
    cfg.cv.seq_model = a.seq
    CAP.clear()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
    except Exception:
        continue
    G = extract(str(n))
    gts = {c["name"]: dense(c) for c in G["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    traces = CAP.get("traces", [])
    written = CAP.get("written", {})
    for nm, gd in gts.items():
        # ЛУЧШАЯ трасса пула для этой кривой — та, которую отбор ОБЯЗАН был выбрать
        best = None
        for L, tr in traces:
            m, c = err(tr, gd)
            if HON(m, c) and (best is None or m < best[0]):
                best = (m, L, tr)
        if best is None:
            continue                                   # честной трассы в пуле нет — не наш случай
        got = written.get(nm)
        if got is not None and HON(*err(got[1], gd)):
            reasons["записана верно"] += 1
            continue
        _, L, _ = best
        info = M.curve_info(nm, CAP["mn"])
        colors = {q.color for q, _ in traces if q.track_index == L.track_index}
        strict = info["color"] is not None and info["color"] in colors
        cls = "SP" if L.behavior == "smooth" else "RES"
        if strict and info["color"] != L.color:
            why = "ЦВЕТ"
        elif info["class"] not in (cls, "OTHER", "CALI"):
            why = "КЛАСС"
        elif got is None:
            why = "СЛОТ ПУСТ"
        else:
            why = "ПОРЯДОК x_center"
        reasons[why] += 1
        detail.append((why, nm, info["color"], L.color, info["class"], cls, n.stem[:30]))

print(f"\n{'='*88}\nПОЧЕМУ ЧЕСТНАЯ ТРАССА ИЗ ПУЛА НЕ ПОПАЛА В СЛОТ (режим seq={a.seq or '—'})")
tot = sum(reasons.values())
for k, v in reasons.most_common():
    print(f"  {k:<20} {v:>3} ({100*v/max(1,tot):>3.0f}%)")
print(f"\nпримеры (слот ждёт цвет/класс — у линии цвет/класс):")
for w, nm, sc, lc, scl, lcl, sh in detail[:12]:
    print(f"  {w:<18} {nm[:14]:<15} ждёт {str(sc):<7}/{scl:<5} — линия {str(lc):<7}/{lcl:<5} {sh}")

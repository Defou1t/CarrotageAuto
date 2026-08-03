r"""_wide_run_cost.py — ОТКУДА БЕРЁТСЯ ЗАМЕДЛЕНИЕ ПРИ ВЫКЛЮЧЕННОЙ ВЕРШИНЕ (§6.102).

ЗАМЕР A/B показал: пайплайн с `wide_run=10**6` идёт вдвое дольше (2.4 против 1.2 мин на лист при
12 процессах). Гипотеза: центр рана даёт более полную трассу, из-за чего в `refine_trace` чаще
срабатывает ветка ЭСКАЛАЦИИ полосы (`beyond/len(rows) >= 0.03` → перетрасс с `x_range` до трека),
а это второй полный проход по МНОГО более широкой полосе.

Проверяется прямо: считаются вызовы `trace_line` — всего и с `x_range` (эскалация), — плюс время.
Разница именно в доле эскалаций подтвердит или опровергнет гипотезу.

  <ComfyUI>\python_embeded\python.exe _wide_run_cost.py [--n 4]
"""
import sys, io, time, argparse, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from dataset_build import find_image
from auto import trace2d as T
from auto.pipeline import run as pipe_run
from auto.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--out", default=r"F:\nds\output\taskS\wide_run_cost")
a = ap.parse_args()

WLG = []
for wlg in sorted(Path(r"F:\nds\projects\Archive").glob("*/wlg")):
    WLG += sorted(wlg.glob("*.nlgx"))
sheets = [q for q in WLG if find_image(q)][: a.n]
print(f"листов {len(sheets)}")

_orig = T.trace_line
CNT = {}


def make(wr):
    def wrapped(fg, line, frame, p, **kw):
        CNT["all"] = CNT.get("all", 0) + 1
        if kw.get("x_range") is not None:
            CNT["esc"] = CNT.get("esc", 0) + 1
        if wr is not None:
            kw.setdefault("wide_run", wr)
        return _orig(fg, line, frame, p, **kw)
    return wrapped


for tag, wr in (("A вершина ВКЛ (14)", None), ("B вершина ВЫКЛ (10**6)", 10 ** 6)):
    T.trace_line = make(wr)
    CNT.clear()
    t0 = time.time()
    for q in sheets:
        cfg = Config(); cfg.cv.seq_model = ""      # §6.71: жадный путь пиннится явно
        cfg.out = Path(a.out) / tag.split()[0] / q.stem[:40]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pipe_run(str(find_image(q)), frame_nlgx=str(q), cfg=cfg, stages=False)
        except Exception as e:
            print(f"  ПАДЕНИЕ {q.stem[:40]}: {type(e).__name__}: {e}")
    al, es = CNT.get("all", 0), CNT.get("esc", 0)
    print(f"{tag:<24} {time.time()-t0:>6.1f} с  вызовов trace_line {al:>4}, "
          f"из них эскалаций {es:>4} ({100*es/max(1,al):.0f}%)")
T.trace_line = _orig

r"""_tone_gate.py — есть ли ТОН БУМАГИ, по которому §6.52 можно включать выборочно.

ЧТО ИЗВЕСТНО (§6.73). `v_hi = min(grid_v_hi, max(dark_v+10, paper-50))` задумывался как спасение
пожелтевших сканов (19 листов давали 0 трасс на кривую), но у реальной БЕЛОЙ бумаги p90(V)=243-251,
поэтому `paper-50` < 205 и ветка `min` почти не срабатывает: правка действует на ВСЕХ листах.
На валидации она не помогла ни одному листу и стоила двух (15 против 17 прод-путём).

ВОПРОС. Если польза §6.52 сидит ТОЛЬКО на тёмных листах, а вред — только на светлых, то между
группами должен быть РАЗРЫВ по `paper`, и сторож берётся по его середине — ровно так §6.65 брал
`color_max_frac` (бимодальное распределение, пустая зона 12 п.п., порог по центру).
Если разрыва нет — порог подбирать НЕЛЬЗЯ (§6.47/§6.49: подгонка под популяцию), и вопрос
закрывается иначе.

★ ВСЕ ТРИ ПАРЫ ЧИСТЫЕ: обе стороны каждой пары собраны СЕГОДНЯШНИМ кодом и различаются ровно
одним `--set grid_v_rel=false`. Брать сюда `cache`/`wide` (сборки 24.07) НЕЛЬЗЯ — это была бы
ровно ошибка §6.71, только в другом месте:
    гейт       gate_g2  (§6.52 вкл) против gate_abl (выкл)
    валидация  hold_g2  против hold_abl
    третий     wide60   против wide_abl

  python _tone_gate.py
"""
import sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.argv = ["_pick_gate.py"]
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
import cv2
import _pick_gate as G
from dataset_build import find_image
from pathlib import Path

R = Path(r"F:\nds\output\taskS\pick_gate")
G.a.bridge = 20
G.a.dedup_tol = 50
PICK = "nl+npts"

PAIRS = [("гейт", "gate_g2", "gate_abl"),
         ("валидация", "hold_g2", "hold_abl"),
         ("третий", "wide60", "wide_abl")]

idx = {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        idx.setdefault(q.name, q)
for extra in (r"F:\nds\projects\Semeguniv_001\wlg", r"F:\nds\projects\Semeguniv_020\wlg"):
    if Path(extra).is_dir():
        for q in Path(extra).glob("*.nlgx"):
            idx.setdefault(q.name, q)

_tone = {}


def paper_p90(pkl_name, src_name):
    """p90(V) листа — ровно та величина, по которой §6.52 считает порог."""
    if pkl_name in _tone:
        return _tone[pkl_name]
    q = idx.get(src_name)
    img = find_image(q) if q else None
    val = None
    if img:
        im = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if im is not None:
            v = cv2.cvtColor(im[::8, ::8], cv2.COLOR_BGR2HSV)[:, :, 2]
            val = int(np.percentile(v, 90))
    _tone[pkl_name] = val
    return val


def honest(d, n):
    dd = pickle.load(open(R / d / f"{n}.pkl", "rb"))
    c = [G.bridge(t, 20) for t in dd["traces"] if len(t) >= G.MINPTS]
    h = G.score(G.PICKERS[PICK](c, max(1, dd["K"] or len(dd["GM"])), G.a), dd["GM"], dd["raw"])[0]
    return h, dd["name"]


rows = []
for lbl, don, doff in PAIRS:
    if not (R / doff).is_dir():
        print(f"{lbl}: кэш абляции {doff} ещё не готов")
        continue
    names = sorted({f.stem for f in (R / don).glob("*.pkl")} & {f.stem for f in (R / doff).glob("*.pkl")})
    for n in names:
        a, src = honest(don, n)
        b, _ = honest(doff, n)
        rows.append((lbl, n, paper_p90(n, src), a, b, a - b))   # d>0 = §6.52 ПОМОГ

print(f"\n{'набор':<20}{'p90(V)':>8}{'вкл':>5}{'выкл':>6}{'вклад §6.52':>13}  лист")
for r in sorted(rows, key=lambda q: (q[2] is None, q[2] or 0)):
    if r[5]:
        print(f"{r[0]:<20}{str(r[2]):>8}{r[3]:>5}{r[4]:>6}{r[5]:>+13d}  {r[1][:44]}")

print(f"\n{'='*96}\nРАЗДЕЛЯЕТ ЛИ ТОН ПОЛЬЗУ И ВРЕД")
good = [r[2] for r in rows if r[5] > 0 and r[2] is not None]     # §6.52 помог
bad = [r[2] for r in rows if r[5] < 0 and r[2] is not None]      # §6.52 навредил
neu = [r[2] for r in rows if r[5] == 0 and r[2] is not None]
for nm, g in (("ПОМОГ", good), ("НАВРЕДИЛ", bad), ("не изменил", neu)):
    if g:
        print(f"  {nm:<12} листов {len(g):>3}   p90(V): мин {min(g)}  медиана {int(np.median(g))}  макс {max(g)}")
if good and bad:
    if max(good) < min(bad):
        lo, hi = max(good), min(bad)
        print(f"\n★ РАЗРЫВ ЕСТЬ: польза до p90={lo}, вред от p90={hi} — пустая зона {hi-lo} ед.")
        print(f"  сторож по её середине: применять §6.52 при p90(V) < {(lo+hi)//2}")
    else:
        print(f"\n⛔ РАЗРЫВА НЕТ: пользa до {max(good)}, вред от {min(bad)} — диапазоны ПЕРЕСЕКАЮТСЯ.")
        print("  Сторож по тону подбирать НЕЛЬЗЯ (§6.47/§6.49): это была бы подгонка.")

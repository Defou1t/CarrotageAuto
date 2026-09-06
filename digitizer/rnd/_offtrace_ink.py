r"""_offtrace_ink.py — ЧТО ИМЕННО ВЕДЁТ ДЕКОДЕР, КОГДА ВЕДЁТ «НЕ ТО» (§6.189 → следующий шаг B1).

ОТКУДА ВОПРОС. §6.189 свёл весь провал декодера на пучке к одной клетке: 368 трасс из 721 больше
чем на 20% строк не лежат ни на одной ОЦИФРОВАННОЙ кривой. И там же честно сказано, чего тот разбор
знать не мог: «вне линий» — это «не совпало с эталоном», а НЕ «не на туши». Разница решает правку:

  ★ трасса идёт по НАСТОЯЩЕЙ туши, которую эксперт не оцифровал ⇒ дефект ВЫБОРА: линий на треке
    больше, чем нужно оцифровать, и декодер берёт не те. Информация для выбора есть (число
    ожидаемых кривых, цвет, положение) и сейчас не используется — правка осмысленная;
  ★ трасса идёт по РАМКЕ, СЕТКЕ или ни по чему ⇒ дефект КАНДИДАТОВ: декодер принимает за линию то,
    что линией не является. Это ближе к переднему плану и к тому, что ветка уже закрывала.

ЧТО ДЕЛАЕТ. Берёт трассы, ВЫГРУЖЕННЫЕ тем же разбором (`_k34_why.py --dump-off`) — не отобранные
заново, иначе объяснялся бы другой набор, — и спрашивает у картинки три маски ПРОДА:
  `trace2d._color_fg` по всем цветам (§6.100: маска ровно та, которой трассирует прод),
  мягкое чернило (V ≤ p90 строки − DELTA, минус структура) — та же, что в `_row_ceiling.py`,
  `imaging.structure_mask` — рамка и сетка.

⚠ САНИТАРНЫЙ СЧЁТЧИК (§6.65): печатается доля листа, принятая мягкой маской. Тушь занимает единицы
процентов; десятки — значит маска ловит бумагу, и числа недействительны.

  <ComfyUI>\python_embeded\python.exe _offtrace_ink.py --dump offtrace_ДЕКОДЕР_3_4.pkl --shard 0/4
  <ComfyUI>\python_embeded\python.exe _offtrace_ink.py --sum --tag offink
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from dataset_build import find_image
from auto import imaging as im, trace2d as T
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dump", default="offtrace_ДЕКОДЕР_3_4.pkl")
ap.add_argument("--wlg-roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--shard", default="0/1")
ap.add_argument("--delta", type=float, default=90.0)
ap.add_argument("--tol", type=int, default=3, help="допуск ±px, как §6.26")
ap.add_argument("--near", type=float, default=0.8, help="доля строк, чтобы назвать трассу лежащей")
ap.add_argument("--cap", type=int, default=0)
ap.add_argument("--tag", default="offink")
ap.add_argument("--sum", action="store_true")
a = ap.parse_args()
TS = Path(a.ts)
COLORS = ["black", "red", "orange", "green", "blue"]

if a.sum:
    tot, sane = Counter(), []
    for p in sorted(TS.glob(f"{a.tag}_*of*.pkl")):
        d = pickle.load(open(p, "rb"))
        tot.update(d["cnt"])
        sane += d["soft_frac"]
    n = sum(tot.values())
    print(f"★★ ЧТО ВЕДЁТ ТРАССА, КОГДА ВЕДЁТ «НЕ ТО» (трасс {n})")
    for k, v in tot.most_common():
        print(f"| {k} | {v} | {100*v/max(1,n):.0f}% |")
    if sane:
        s = np.array(sane, float)
        print(f"\n⚠ САНИТАРНЫЙ (§6.65): мягкая маска берёт {100*s.mean():.1f}% листа "
              f"(медиана {100*np.median(s):.1f}%, максимум {100*s.max():.1f}%). "
              f"{'★ единицы процентов — маска ловит тушь' if np.median(s) < 0.10 else '⛔ ДЕСЯТКИ ПРОЦЕНТОВ — маска ловит БУМАГУ, числа недействительны'}")
    sys.exit(0)

i, n = (int(x) for x in a.shard.split("/"))
OFF = pickle.load(open(TS / a.dump, "rb"))
by_sheet = defaultdict(list)
for sh, gname, wname, tr in OFF:
    by_sheet[sh].append((gname, wname, tr))
sheets = sorted(by_sheet)
if a.cap:
    sheets = sheets[:a.cap]
mine = [s for k, s in enumerate(sheets) if k % n == i]
print(f"★ ШАРД {i}/{n}: листов {len(mine)} из {len(sheets)}, трасс {sum(len(by_sheet[s]) for s in mine)}")

WLG = {}
for root in a.wlg_roots:
    for wlg in Path(root).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            WLG.setdefault(q.name, q)
for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
    WLG.setdefault(q.name, q)

cnt, soft_frac, skip = Counter(), [], Counter()
p = DEFAULT.cv
for k, sh in enumerate(mine, 1):
    q = WLG.get(sh)
    img = find_image(q) if q else None
    if not img:
        skip["нет картинки"] += len(by_sheet[sh]); continue
    try:
        # ⚠ ЧИТАТЬ ТЕМ ЖЕ, ЧЕМ ЧИТАЕТ ПРОД (§6.115): `cv2.imread` молча терял 10.6% листов.
        rgb = im.load_rgb(str(img))
    except Exception as e:
        skip[f"{type(e).__name__}"] += len(by_sheet[sh]); continue
    H, W = rgb.shape[:2]
    fg = np.zeros((H, W), bool)
    for c in COLORS:
        fg |= T._color_fg(rgb, c, p)          # «есть ли тушь ЛЮБОГО цвета» — цвет здесь не судим
    V = im.value_channel(rgb)
    paper = np.empty(H, np.float32)
    for y0 in range(0, H, 4096):              # §6.65: перцентиль блоками, иначе 0.7 ГБ на процесс
        y1 = min(H, y0 + 4096)
        paper[y0:y1] = np.percentile(V[y0:y1, ::4], 90, axis=1)
    struct = im.structure_mask(rgb, p)
    soft = (V <= (paper - a.delta)[:, None]) & ~struct
    soft_frac.append(float(soft.mean()))

    for gname, wname, tr in by_sheet[sh]:
        ys = [y for y in tr if 0 <= y < H]
        if len(ys) < 30:
            skip["трасса короче 30 строк"] += 1; continue
        h_fg = h_soft = h_st = 0
        for y in ys:
            x = int(tr[y])
            lo, hi = max(0, x - a.tol), min(W, x + a.tol + 1)
            if lo >= hi:
                continue
            if fg[y, lo:hi].any():
                h_fg += 1
            if soft[y, lo:hi].any():
                h_soft += 1
            if struct[y, lo:hi].any():
                h_st += 1
        m = len(ys)
        f_fg, f_soft, f_st = h_fg / m, h_soft / m, h_st / m
        # ★ ПОРЯДОК ВОПРОСОВ — ОТ САМОГО ДОРОГОГО ВЫВОДА К САМОМУ ДЕШЁВОМУ. Если трасса лежит на
        #   прод-туши, то дело НЕ в переднем плане и не в кандидатах: линия настоящая, её просто
        #   не оцифровали. Это и есть дефект выбора, и признать его надо в первую очередь.
        if f_fg >= a.near:
            cnt["★ на НАСТОЯЩЕЙ туши (прод-маска) — линия есть, но не оцифрована"] += 1
        elif f_soft >= a.near:
            cnt["на БЛЕДНОЙ туши (только мягкая маска)"] += 1
        elif f_st >= 0.5:
            cnt["на СТРУКТУРЕ (рамка/сетка)"] += 1
        else:
            cnt["ни на чём (шум/обрывки)"] += 1
    if k % 20 == 0 or k == len(mine):
        print(f"  {k}/{len(mine)}  разобрано трасс {sum(cnt.values())}")

out = TS / f"{a.tag}_{i}of{n}.pkl"
pickle.dump(dict(cnt=dict(cnt), soft_frac=soft_frac, skip=dict(skip)), open(out, "wb"))
print(f"★ ГОТОВО: трасс {sum(cnt.values())}, пропущено {sum(skip.values())} → {out.name}")
for k, v in cnt.most_common():
    print(f"   {k}: {v}")
for k, v in skip.items():
    print(f"   ⚠ {k}: {v}")

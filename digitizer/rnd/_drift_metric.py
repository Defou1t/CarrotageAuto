r"""_drift_metric.py — МЕТРИКА ДРЕЙФА: сколько строк трассировщик держит личность и чего стоит срыв.

⚠⚠ ЗАЧЕМ (§6.94). Замер скептика показал главное число ветки: ран, ближайший к ПРАВИЛЬНОМУ
предыдущему x, попадает в эталон ≤3px в **95% строк**, а ближайший к СВОЕМУ предыдущему положению —
лишь в **26%**. Разница 95 → 26 — это накопление ошибки, то есть ДРЕЙФ. Но «доля строк на цели» не
говорит, дрейф ли это (уход навсегда) или дребезг (срывы и возвраты), а от этого зависит выбор
инструмента: gap-closing и глобальная линковка (§6.92) лечат первое, сглаживание — второе.
Здесь мерится то, чего в проекте не было: ДЛИНА СЕРИИ до первого срыва и ЦЕНА восстановления.

ТРИ РЕЖИМА ВЕДЕНИЯ, одни и те же раны, отличается только ИСТОРИЯ:
  oracle  — на каждой строке берётся ран, ближайший к ЭТАЛОННОМУ x этой строки (верхняя граница);
  hist    — ран, ближайший к ПРАВИЛЬНОМУ x ПРЕДЫДУЩЕЙ строки (что даёт идеальная история — 95%);
  ★ self  — ран, ближайший к СВОЕМУ предыдущему выбору (как ведёт реальный трассировщик).
Разница `hist` и `self` — цена накопления, и это метрика ветки.

⚠⚠ ДВЕ РАЗНЫЕ ВЕЛИЧИНЫ, КОТОРЫЕ НЕЛЬЗЯ СМЕШИВАТЬ (первая редакция смешала и дала ложный вердикт):
  on_*  — ★ ВЫБРАН ЛИ ПРАВИЛЬНЫЙ РАН (эталонная точка строки лежит внутри выбранного рана). Это
          ИДЕНТИЧНОСТЬ, и именно это сравнимо с независимыми 95%/26%;
  px_*  — попал ли ВЫДАННЫЙ x (центр рана) в ±tol от эталона. Здесь потолок задаёт ШИРИНА рана
          (медиана 11px ⇒ ±5px), то есть это зона УТОЧНЕНИЯ (refine/деспайк), а не идентичности.

ЧТО СЧИТАЕТСЯ (на кривую, по каждому режиму):
  len/maxlen/medlen — длина первой, самой длинной и медианной серии подряд ВЕРНЫХ ранов;
  br        — число срывов на 1000 строк (переход с верного рана на неверный);
  back      — доля срывов, после которых ведение САМО вернулось на верный ран (дребезг, не уход);
  relatch   — минимальный интервал принудительного возврата к эталону, при котором держится ≥90%
              верных ранов (0 = не достигается даже каждые 25 строк).

⚠ Раны берутся в ОДНОЙ маске (см. `--mask`, по умолчанию объединение чернил): урок §6.94 —
смешение масок между ступенями рождает ложные корзины.
⚠ Это НЕ прод-трассировщик (у того есть полоса, refine, деспайк, селектор ранов). Это чистый
эксперимент на ведение: сколько даёт голая непрерывность по ранам и где она рвётся.

  <ComfyUI>\python_embeded\python.exe _drift_metric.py [--shard 0/8] [--step 5]
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from _multi_replica_probe import dense
from auto import imaging as im, trace2d as T, meta as M
from auto.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="F:/nds/output/taskS/_slot_abstain_cache_v2.pkl")
ap.add_argument("--step", type=int, default=5)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--mask", default="ink", choices=["ink", "black"])
ap.add_argument("--tol", type=float, default=3.0, help="порог «на цели», px")
ap.add_argument("--band", type=float, default=8.0, help="полоса поиска для режима band (у прода 8)")
ap.add_argument("--out", default=r"F:\nds\output\taskS\drift_metric")
a = ap.parse_args()
SH_I, SH_N = (int(v) for v in a.shard.split("/"))
P = Config().cv
BAND = a.band                      # полоса режима "band" (у прода band_pad=8)

C = pickle.load(open(a.cache, "rb"))
PHON = {}
for (si, nm), v in C["phon"].items():
    PHON[(C["names"][si], nm)] = v
WELL = dict(zip(C["names"], C["wells"]))
WLG = {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WLG[q.name] = q
for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
    WLG.setdefault(q.name, q)

todo = sorted(n for n in C["names"] if n in WLG)
todo = [n for i, n in enumerate(todo) if i % SH_N == SH_I]
print(f"шард {SH_I}/{SH_N}: листов {len(todo)}, маска {a.mask}, порог {a.tol}px")


def lead(runs_by_row, gxs, mode):
    """Провести кривую по ранам. Возвращает (xs, hits): xs — выданный x (ЦЕНТР выбранного рана),
    hits — попал ли ВЫБРАННЫЙ РАН на эталонную точку строки.

    ⚠⚠ ПОЧЕМУ ДВА ЧИСЛА, А НЕ ОДНО. Первая редакция ставила x = предыдущий эталонный x, если тот
    попадал в ран, и тогда ошибка равнялась СОБСТВЕННОМУ НАКЛОНУ кривой между строками выборки: при
    шаге 5 строк любая кривая быстрее 0.6px/строку «срывалась» сама собой, давая ложные 61% против
    независимых 95% и ложный вердикт «дребезг» (медианная серия 2 строки, самовозвраты 100%).
    Правильно разделять: `hits` — ВЫБОР РАНА, то есть идентичность (это и сравнимо с 95%);
    `xs` — выданное значение, где точность ограничена ШИРИНОЙ рана (медиана 11px ⇒ ±5px), и это
    зона уточнения (refine/деспайк), а не идентичности. Смешивать их нельзя."""
    xs = np.full(len(gxs), np.nan)
    hits = np.zeros(len(gxs), bool)
    prev = gxs[0]
    for i, rr in enumerate(runs_by_row):
        if not rr:
            continue
        ref = gxs[i] if mode == "oracle" else (gxs[i - 1] if (mode == "hist" and i) else prev)
        if mode == "band":
            # ★★ РЕЖИМ ПРОДА: поиск ограничен ПОЛОСОЙ вокруг предыдущего решения (band_pad, у прода 8).
            # Требование приговора §6.96: разрыв 88%/36% измерен БЕЗ полосы, поэтому часть его прод уже
            # забирает. Если ран в полосу не попал — ведение НЕ ПРЫГАЕТ далеко, а держит позицию
            # (у прода это FLAG/пропуск), и именно это отличает его от жадного «ближайший в ±150px».
            near_rr = [t for t in rr if t[0] - BAND <= ref <= t[1] + BAND]
            if not near_rr:
                xs[i] = prev
                hits[i] = False
                continue
            rr = near_rr
        r = min(rr, key=lambda t: 0 if t[0] <= ref <= t[1] else min(abs(t[0] - ref), abs(t[1] - ref)))
        xs[i] = float(r[2])                     # выдаём ЦЕНТР рана, как делает трассировщик
        hits[i] = (r[0] - 1) <= gxs[i] <= (r[1] + 1)
        prev = xs[i]                            # состояние — своё же предыдущее решение
    return xs, hits


def lead_dp(runs_by_row, gxs, band=None, jump=40.0):
    """★ ГЛОБАЛЬНОЕ решение о пути: ДП (Виттерби) по ранам вместо жадного шага.

    Состояние строки — выбранный ран; цена перехода — |Δx| между центрами, обрезанная `jump`
    (дальше — запрещено); цена состояния — 0 (все раны равноправны, признаков сюда сознательно НЕ
    добавляем, чтобы мерить чистый эффект глобальности). Возвращает (xs, hits), как `lead`.

    ⚠ ЗАЧЕМ ИМЕННО ЭТО. §6.95-§6.98: при ПРАВИЛЬНОЙ истории ближайший ран верен в 92% строк, при своей —
    в 40%, а прод (жадный + полоса) даёт 8.0% честных кривых против 59.0% при идеальной истории. Жадность
    решает построчно и не может передумать; ДП выбирает путь ЦЕЛИКОМ и потому не обязан цепляться за
    ошибку. Это самый дешёвый представитель класса «глобальная линковка» (§6.92: laptrack/Stone Soup) —
    если он не двигает цифру, то и LAP не двинет, и наоборот.
    ⚠ `band` здесь НЕ окно поиска, а ограничение перехода: у прода полоса ±8px на строку, поэтому
    сравнение честное только при том же ограничении.
    """
    n = len(gxs)
    xs = np.full(n, np.nan)
    hits = np.zeros(n, bool)
    lim = band if band else jump
    prev_idx, prev_cost, prev_runs = None, None, None
    back = []                                   # (для каждой строки) откуда пришли
    for i, rr in enumerate(runs_by_row):
        if not rr:
            back.append(None)
            continue
        cen = np.array([t[2] for t in rr], float)
        if prev_runs is None:
            cost = np.abs(cen - gxs[0])          # старт — от известной первой точки, как у `lead`
            ptr = np.full(len(rr), -1, int)
        else:
            d = np.abs(cen[:, None] - prev_cen[None, :])
            pen = np.where(d <= lim, d, 1e6)     # прыжок дальше полосы запрещён
            tot = prev_cost[None, :] + pen
            ptr = np.argmin(tot, axis=1)
            cost = tot[np.arange(len(rr)), ptr]
        back.append((rr, ptr))
        prev_cost, prev_cen, prev_runs = cost, cen, rr
    if prev_runs is None:
        return xs, hits
    # обратный проход
    j = int(np.argmin(prev_cost))
    for i in range(len(back) - 1, -1, -1):
        if back[i] is None:
            continue
        rr, ptr = back[i]
        if j >= len(rr):
            j = int(np.clip(j, 0, len(rr) - 1))
        r = rr[j]
        xs[i] = float(r[2])
        hits[i] = (r[0] - 1) <= gxs[i] <= (r[1] + 1)
        j = int(ptr[j]) if ptr[j] >= 0 else 0
    return xs, hits


rows = []
for k, name in enumerate(todo, 1):
    n = WLG[name]
    img = find_image(n)
    if not img:
        continue
    try:
        G = extract(str(n))
        raws = {c["name"]: c for c in G["curves"]
                if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        gts = {nm: dense(c) for nm, c in raws.items()}
        if not gts:
            continue
        rgb = im.load_rgb(str(img))
        H, W = rgb.shape[:2]
        msk = (im.ink_foreground(rgb, P) > 0) if a.mask == "ink" else T._color_fg(rgb, "black", P)
    except Exception as e:
        print(f"  ПАДЕНИЕ {name[:40]}: {type(e).__name__}: {e}")
        continue

    for nm, gt in gts.items():
        ys = [y for y in sorted(gt)[::a.step] if 0 <= y < H]
        if len(ys) < 30:
            continue
        gxs = np.array([gt[y] for y in ys], float)
        lo = max(0, int(gxs.min()) - 150); hi = min(W, int(gxs.max()) + 151)
        rbr = []
        for y in ys:
            rr = im.row_runs(msk[y, lo:hi])
            rbr.append([(p + lo, q + lo, c + lo) for p, q, c in rr])
        rec = dict(sheet=name, curve=nm, well=WELL.get(name, "?"),
                   fam=(WELL.get(name) or "?").split("_")[0], nrows=len(ys),
                   norun=float(np.mean([not r for r in rbr])), prod=PHON.get((name, nm), 0))
        for mode in ("oracle", "hist", "self", "band", "dp"):
            xs, hits = (lead_dp(rbr, gxs, band=BAND) if mode == "dp"
                        else lead(rbr, gxs, mode))
            ok = hits.copy()                    # ★ «на цели» = ВЫБРАН ПРАВИЛЬНЫЙ РАН (идентичность)
            rec[f"on_{mode}"] = float(ok.mean())
            near = np.abs(xs - gxs) <= a.tol    # а это уже точность внутри рана (уточнение)
            near[np.isnan(xs)] = False
            rec[f"px_{mode}"] = float(near.mean())
            # ⚠ «первая серия от старта» хрупка: одна неудачная стартовая строка даёт 0. Поэтому
            # рядом считаются САМАЯ ДЛИННАЯ и МЕДИАННАЯ серии «на цели» — они и отвечают на вопрос
            # «сколько строк ведение держит личность».
            first = 0
            while first < len(ok) and ok[first]:
                first += 1
            streaks, cur = [], 0
            for v in ok:
                if v:
                    cur += 1
                elif cur:
                    streaks.append(cur); cur = 0
            if cur:
                streaks.append(cur)
            rec[f"len_{mode}"] = first
            rec[f"lenf_{mode}"] = first / max(1, len(ok))
            rec[f"maxlen_{mode}"] = max(streaks) if streaks else 0
            rec[f"medlen_{mode}"] = float(np.median(streaks)) if streaks else 0.0
            rec[f"nstreak_{mode}"] = len(streaks)
            # срывы и самовозвраты
            br = int(np.sum(ok[:-1] & ~ok[1:]))
            rb = int(np.sum(~ok[:-1] & ok[1:]))
            rec[f"br_{mode}"] = 1000.0 * br / max(1, len(ok))
            rec[f"back_{mode}"] = rb / max(1, br) if br else 0.0
            rec[f"med_{mode}"] = float(np.nanmedian(np.abs(xs - gxs)))
        # цена восстановления: возврат к эталону каждые G строк, минимальное G под 90% ПОПАДАНИЙ В РАН
        need = None
        for G_ in (10000, 2000, 1000, 500, 300, 200, 100, 50, 25):
            hits = np.zeros(len(gxs), bool)
            prev = gxs[0]
            for i, rr in enumerate(rbr):
                if i % G_ == 0:
                    prev = gxs[i]
                if not rr:
                    continue
                r = min(rr, key=lambda t: 0 if t[0] <= prev <= t[1]
                        else min(abs(t[0] - prev), abs(t[1] - prev)))
                hits[i] = (r[0] - 1) <= gxs[i] <= (r[1] + 1)
                prev = float(r[2])
            if hits.mean() >= 0.9:
                need = G_
                break
        rec["relatch_every"] = need if need else 0
        rec["relatch_n"] = (len(gxs) // need) if need else -1
        rows.append(rec)
    del rgb, msk
    if k % 10 == 0:
        print(f"  {k}/{len(todo)} листов, кривых {len(rows)}")

Path(a.out).mkdir(parents=True, exist_ok=True)
dst = Path(a.out) / f"rows_{SH_I}of{SH_N}.pkl"
pickle.dump(rows, open(dst, "wb"))
print(f"\nзаписано {len(rows)} кривых → {dst}")

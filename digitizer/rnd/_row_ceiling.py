r"""_row_ceiling.py — ЧТО 95.5% §6.131 СТОЯТ В ЧЕСТНЫХ КРИВЫХ (замер потолка ДО постройки модели).

§6.131 намерил потолок ПОСТАНОВКИ построчной задачи: 4.5% строк неразрешимы по положению, значит
95.5% разрешимы. Но метрика ветки считает не строки, а КРИВЫЕ (med|Δx| ≤ 3px И cov ≥ 0.9), и
перевода одного в другое нет: локальная ошибка медиану переживает, а отсутствие чернил бьёт по cov.
Пока перевод не сделан, «95.5%» не обещает ни одной честной кривой (⚠ прямо сказано в §6.131).

ЧТО СЧИТАЕТСЯ. Три потолка на одном и том же корпусе и одной и той же метрике:

  V0 «ЧИСТАЯ ПОСТАНОВКА» — ровно §6.131, без картинки: положения берутся из ЭТАЛОНА, ошибка
     возникает только там, где соседи по треку ближе NEAR=3px (там декодер путает их местами —
     циклический сдвиг внутри слипшейся группы). Контроль: если V0 ≈ весь корпус, то 95.5% строк
     действительно «почти всё» — и тогда решает НЕ постановка, а то, что видно на скане.

  V1 «ПО ЧЕРНИЛАМ» — положение засчитывается, только если под точкой эталона ЕСТЬ ран той же
     цветовой маски, которой трассирует прод (`trace2d._color_fg`, §6.100: маску брать ту же, что
     берёт прод, а не «какую-нибудь»). Строка без рана — НЕ ПОКРЫТА (бьёт по cov, §6.112).
     Ошибка внутри рана = 0: декодер вправе выдать любую точку рана, это ПОТОЛОК.

  V2 «ПО ЧЕРНИЛАМ + 1:1» — то же, плюс ограничение прода: две кривые одного трека не могут
     занять ОДИН ран (§6.115: конкуренция за линию стоит мало, но она есть). В слипшейся группе
     одна кривая получает свой x, остальные — x соседа по группе (циклический сдвиг). Это
     ПЕССИМИСТИЧНАЯ граница: реальный декодер иногда угадает, и тогда результат между V1 и V2.

  V3 «ПО ПИКСЕЛЯМ + 1:1» — то же, что V2, но чернило ищется МЯГКИМ критерием, а не прод-маской.
     ⚠⚠ ЗАЧЕМ ОТДЕЛЬНЫЙ ВАРИАНТ. Смоук показал, что на непокрытых строках тушь ЕСТЬ, но БЛЕДНАЯ:
     RYBAL_168, под эталоном V медиана 123 при бумаге p90 = 243 и прод-пороге `dark_v = 110`.
     То есть V1/V2 — потолок НЫНЕШНЕГО ПЕРЕДНЕГО ПЛАНА (бинарь по абсолютной темноте), а не
     потолок задачи: декодер, читающий пиксели, видит серое. Это ровно §6.112 («под половиной
     пропущенных строк тушь есть») и мотив recall-модели §6.110.
     Критерий: V ≤ p90(V по строке) − DELTA, минус структура. Порог ОТНОСИТЕЛЬНЫЙ — на пожелтевших
     сканах бумага 194, а не 243 (§6.52), и абсолютный порог там накрывает лист целиком.
     DELTA=90 выбран между СВЕТЛОЙ СЕТКОЙ (V≈166, §6.6.13) и бледной тушью (V≈123): 243−90 = 153.
     ⚠ САНИТАРНЫЙ СЧЁТЧИК обязателен (§6.65): печатается доля листа, принятая мягкой маской.
     Тушь занимает единицы процентов; если мягкая маска берёт десятки — она ловит бумагу, и
     соответствующие числа недействительны.

⚠⚠ ЧЕГО ЭТОТ ЗАМЕР НЕ ГОВОРИТ (читать до цитирования):
  • Это ПОТОЛОК ПОСТАНОВКИ, а не результат модели. Ни одна модель его не достигнет.
  • Цвет кривой выбирается ОРАКУЛЬНО (где под эталоном больше туши) — тем же приёмом, что в
    `_decoder_seq_data.extract_sheet`. Прод берёт цвет из U1 и ошибается; значит V1/V2 завышены.
  • Раны считаются по ВСЕЙ строке маски, без полосы линии: полоса — это уже механизм ведения,
    у построчного декодера её нет по построению.
  • Путь замера — ПУЛЫ (дампы `_pool_oracle`), не отгрузка. §6.123: пуловый выигрыш может не
    дожить до выдачи. Для потолка это допустимо (он и не претендует на прод-число), но переносить
    вывод «столько-то будет в файле» НЕЛЬЗЯ.

  <ComfyUI>\python_embeded\python.exe _row_ceiling.py --shard 0/6
  <ComfyUI>\python_embeded\python.exe _row_ceiling.py --sum
"""
import sys, argparse, pickle, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from dataset_build import find_image
from auto import imaging as im, trace2d as T
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--wlg-roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--shard", default="0/6")
ap.add_argument("--cap", type=int, default=0, help="листов всего (0 = все); для смоука")
ap.add_argument("--out", default=r"F:/nds/output/taskS")
ap.add_argument("--tag", default="rowceil")
ap.add_argument("--delta", type=float, default=90.0, help="V3: насколько тушь темнее бумаги строки")
ap.add_argument("--sum", action="store_true")
a = ap.parse_args()
OUT = Path(a.out)
NEAR = 3.0                     # §6.131: ближе этого положение не различает кривые
TOL = 3.0                      # §6.26/_decoder_seq_data: ран накрывает эталон с допуском ±3px
COLORS = ["black", "red", "orange", "green", "blue"]
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def curve_stats(err_by_row, n_gt):
    """med по ПОКРЫТЫМ строкам и cov = покрытые/всего — как `err()` в `_trace_prod_ab.py`."""
    if len(err_by_row) < 30:
        return None, len(err_by_row) / max(1, n_gt)
    return float(np.median(np.array(list(err_by_row.values()), float))), len(err_by_row) / max(1, n_gt)


def cyc(xs):
    """Циклический сдвиг: каждой кривой слипшейся группы достаётся x СОСЕДА (пессимизм V0/V2)."""
    return xs[1:] + xs[:1]


def run_sheet(dump, img):
    """→ list[(имя, cov0, med0, cov1, med1, cov2, med2)] по кривым листа."""
    # ⚠⚠ ЧИТАТЬ ТЕМ ЖЕ, ЧЕМ ЧИТАЕТ ПРОД (`imaging.load_rgb`, PIL + MAX_IMAGE_PIXELS=None и
    # доигрывание обрезанных файлов). Первая редакция стенда брала `cv2.imread` — и МОЛЧА
    # потеряла 285 листов из 2676 (10.6%): 224 TIFF, 50 JPEG, 10 PNG, все с `imread=None`.
    # Пропуск был виден только счётчиком ошибок, а сами числа выглядели полными. Это ровно
    # ловушка «стенд повторяет прод неточно» (§6.115) в самом дешёвом её месте — на загрузке.
    rgb = im.load_rgb(str(img))
    H, W = rgb.shape[:2]
    p = DEFAULT.cv
    track = {s["name"]: s["track"] for s in dump["slots"]}
    gts = {nm: g for nm, g in dump["gts"].items() if g and len(g) >= 30}
    if not gts:
        return []

    # ── маски цветов: ровно то, чем трассирует прод (`trace2d._color_fg`) ────────────────────
    fgs = {c: T._color_fg(rgb, c, p) for c in COLORS}
    # ── V3: мягкая маска «темнее бумаги СВОЕЙ строки на DELTA», минус структура ─────────────
    V = im.value_channel(rgb)
    # ⚠ ПАМЯТЬ: `np.percentile` по всей матрице поднимает копию в float64 (на ленте 60000×1390 это
    # 0.7 ГБ на процесс × 6 шардов). Считаем блоками строк и по каждой 4-й колонке — уровень бумаги
    # это оценивает с запасом, а пик памяти держится в десятках мегабайт.
    paper = np.empty(H, np.float32)
    for y0 in range(0, H, 4096):
        y1 = min(H, y0 + 4096)
        paper[y0:y1] = np.percentile(V[y0:y1, ::4], 90, axis=1)
    soft = (V <= (paper - a.delta)[:, None]) & ~im.structure_mask(rgb, p)
    soft_frac = float(soft.mean())

    # ── цвет кривой ОРАКУЛЬНО (как в `_decoder_seq_data`): где под эталоном больше туши ──────
    best = {}
    for nm, g in gts.items():
        rows = sorted(g)
        probe = rows[::max(1, len(rows) // 50)]
        bc, bh = "black", -1
        for c in COLORS:
            fg = fgs[c]
            hit = sum(1 for y in probe
                      if 0 <= y < H and fg[y, max(0, int(g[y]) - 3):int(g[y]) + 4].any())
            if hit > bh:
                bh, bc = hit, c
        best[nm] = bc

    # ── раны строки кэшируются на (цвет, строка): у трека их делят все кривые ────────────────
    runs_cache = {}

    def runs_at(c, y):
        k = (c, y)
        if k not in runs_cache:
            src = soft if c == "__soft__" else fgs[c]
            runs_cache[k] = im.row_runs(src[y]) if 0 <= y < H else []
        return runs_cache[k]

    # ── строки трека: кто на ней стоит ──────────────────────────────────────────────────────
    by_track = defaultdict(list)
    for nm in gts:
        by_track[track.get(nm)].append(nm)

    err0 = {nm: {} for nm in gts}       # V0 чистая постановка
    err1 = {nm: {} for nm in gts}       # V1 по чернилам
    err2 = {nm: {} for nm in gts}       # V2 по чернилам + 1:1
    err3 = {nm: {} for nm in gts}       # V3 по пикселям + 1:1
    # ⚠ ТОЧКА = ЦЕНТР РАНА, а не «любая точка рана». V1/V3 засчитывают попадание в ран ошибкой 0,
    # то есть допускают, что декодер выберет внутри рана идеально. На слипшемся пучке ран широкий,
    # и это допущение щедрое; §6.26 прямо намерил, что весь остаток медианной ошибки — выбор ТОЧКИ
    # внутри верного рана (полуширины 3-9px объясняли ошибку 4-11px). Здесь — консервативная
    # граница: точка берётся центром, ровно как её берёт прод-селектор (`trace_seq`: nx = C[k]).
    err1c = {nm: {} for nm in gts}      # V1ц прод-маска, точка = центр рана
    err3c = {nm: {} for nm in gts}      # V3ц по пикселям, точка = центр рана
    # ★ V3м: то же, что V3ц, но РАЗРЫВЫ МОСТЯТСЯ. ⚠ Первая редакция потолка считала строку без
    # чернил как «нет ответа» (бьёт по cov) — это пессимизм: кривая непрерывна, и декодер вправе
    # выдать положение на разрыве, а не молчать. §6.112: под половиной пропущенных строк тушь есть;
    # P1-5: 43% точек эксперта лежат ВНЕ туши (пунктир, выцветшее). Мостим линейно между ближайшими
    # покрытыми строками — это НИЖНЯЯ оценка мостика (интерполяция ПО ФОРМЕ была бы точнее, P1-5),
    # то есть потолок с ней всё ещё занижен, но уже не отбрасывает разрывы целиком.
    pos3c = {nm: {} for nm in gts}      # положения (центры ранов) для мостика

    for t, names in by_track.items():
        rows = defaultdict(list)
        for nm in names:
            for y, x in gts[nm].items():
                rows[y].append((nm, float(x)))
        for y, items in rows.items():
            # ── V0: слипание по ПОЛОЖЕНИЮ (эталон, картинка не нужна) ───────────────────────
            order = sorted(items, key=lambda it: it[1])
            grp, cur = [], [order[0]]
            for prev, nxt in zip(order, order[1:]):
                if nxt[1] - prev[1] <= NEAR:
                    cur.append(nxt)
                else:
                    grp.append(cur); cur = [nxt]
            grp.append(cur)
            for g0 in grp:
                if len(g0) == 1:
                    err0[g0[0][0]][y] = 0.0
                else:
                    xs = [it[1] for it in g0]
                    for (nm, x), xn in zip(g0, cyc(xs)):
                        err0[nm][y] = abs(x - xn)

            # ── V1/V2/V3: слипание ПО РАНУ (V1/V2 — прод-маска цвета, V3 — мягкая) ─────────
            xof = dict(items)

            def hits(colorer, cen=None):
                """→ {имя: (цвет, x0, x1)}; в `cen` (если дан) кладётся центр попавшего рана."""
                h = {}
                for nm, x in items:
                    for x0, x1, c in runs_at(colorer(nm), y):
                        if x0 - TOL <= x <= x1 + TOL:
                            h[nm] = (colorer(nm), x0, x1)
                            if cen is not None:
                                cen[nm][y] = abs(x - c)
                                if cen is err3c:
                                    pos3c[nm][y] = float(c)
                            break
                return h

            def one_to_one(h, dst):
                """Ран занимает ОДНА кривая; остальным группы достаётся x соседа (пессимизм)."""
                same = defaultdict(list)
                for nm, key in h.items():
                    same[key].append(nm)
                for key, nms in same.items():
                    if len(nms) == 1:
                        dst[nms[0]][y] = 0.0
                    else:
                        nms = sorted(nms, key=lambda q: xof[q])
                        for nm, xn in zip(nms, cyc([xof[q] for q in nms])):
                            dst[nm][y] = abs(xof[nm] - xn)

            h1 = hits(lambda nm: best[nm], err1c)
            for nm in h1:
                err1[nm][y] = 0.0                      # ран есть ⇒ декодер вправе попасть точно
            one_to_one(h1, err2)
            one_to_one(hits(lambda nm: "__soft__", err3c), err3)

    out = []
    for nm, g in gts.items():
        n = len(g)
        m0, c0 = curve_stats(err0[nm], n)
        m1, c1 = curve_stats(err1[nm], n)
        m2, c2 = curve_stats(err2[nm], n)
        m3, c3 = curve_stats(err3[nm], n)
        m1c, c1c = curve_stats(err1c[nm], n)
        m3c, c3c = curve_stats(err3c[nm], n)
        # ── V3м: мостик по покрытым строкам, ошибка считается на ВСЕХ строках эталона ──────
        pr = sorted(pos3c[nm])
        if len(pr) >= 2:
            allr = np.array(sorted(g), dtype=np.int64)
            px = np.array(pr, dtype=np.int64); pv = np.array([pos3c[nm][y] for y in pr], float)
            interp = np.interp(allr, px, pv)
            gtv = np.array([g[int(y)] for y in allr], float)
            # ⚠ за краями покрытия np.interp ДЕРЖИТ КОНСТАНТУ — это не мостик, а выдумка;
            # такие строки в счёт не идут (иначе потолок надувается хвостами).
            inside = (allr >= px[0]) & (allr <= px[-1])
            errm = {int(y): float(abs(v - t))
                    for y, v, t, ok in zip(allr, interp, gtv, inside) if ok}
            m3m, c3m = curve_stats(errm, n)
            gapmax = int(np.max(np.diff(px))) if len(px) > 1 else 0
        else:
            m3m, c3m, gapmax = None, 0.0, 0
        out.append((nm, best[nm], n, c0, m0, c1, m1, c2, m2, c3, m3, soft_frac,
                    c1c, m1c, c3c, m3c, m3m, c3m, gapmax))
    return out


def collect(i, n):
    WLG = {}
    for root in a.wlg_roots:
        for wlg in Path(root).glob("*/wlg"):
            for q in wlg.glob("*.nlgx"):
                WLG.setdefault(q.name, q)
    for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
        WLG.setdefault(q.name, q)
    BY_STEM = {q.stem[:60]: q for q in WLG.values()}

    seen, files = set(), []
    for root in a.pools:
        for f in sorted(Path(root).glob("*.pkl")):
            if f.stem in seen or f.stem not in BY_STEM:
                continue
            seen.add(f.stem); files.append((f, BY_STEM[f.stem]))
    # ⚠ ШАРД ПО РАЗМЕРУ ДАМПА (§6.118): блочное разбиение отдаёт одному шарду все дорогие листы.
    bins, load = [[] for _ in range(n)], [0] * n
    for f, q in sorted(files, key=lambda z: (-z[0].stat().st_size, z[0].name)):
        k = min(range(n), key=lambda t: (load[t], t))
        bins[k].append((f, q)); load[k] += f.stat().st_size
    mine = sorted(bins[i], key=lambda z: z[0].name)
    if a.cap:
        mine = mine[:a.cap]
    print(f"★ ШАРД {i}/{n}: листов {len(mine)} (в пулах {len(files)}), "
          f"пул {load[i]/2**20:.0f} МБ, перекос {max(load)/max(1,min(load)):.2f}×")

    rows, skip = [], defaultdict(int)
    for k, (f, q) in enumerate(mine, 1):
        img = find_image(q)
        if not img:
            skip["нет картинки"] += 1; continue
        try:
            d = pickle.load(open(f, "rb"))
            r = run_sheet(d, img)
        except Exception as e:
            skip[f"{type(e).__name__}: {e}"[:60]] += 1; continue
        for it in r:
            rows.append((q.name,) + it)
        if k % 25 == 0 or k == len(mine):
            print(f"  {k}/{len(mine)}  кривых {len(rows)}")
    p = OUT / f"{a.tag}_{i}of{n}.pkl"
    pickle.dump(dict(rows=rows, skip=dict(skip), sheets=len(mine)), open(p, "wb"))
    print(f"★ готово: листов {len(mine)}, кривых {len(rows)}, пропусков {sum(skip.values())} → {p}")
    for s, c in skip.items():
        print(f"    ⚠ {s}: {c}")


def summarise():
    fs = sorted(OUT.glob(f"{a.tag}_*of*.pkl"))
    if not fs:
        sys.exit("нет дампов")
    den = fs[0].stem.split("of")[1]
    fs = [f for f in fs if f.stem.endswith("of" + den)]
    rows, sheets, skip = [], 0, defaultdict(int)
    for f in fs:
        d = pickle.load(open(f, "rb"))
        rows += d["rows"]; sheets += d["sheets"]
        for s, c in d["skip"].items():
            skip[s] += c
    print(f"шардов {len(fs)} из {den}" + ("  ⚠⚠ НЕПОЛНЫЙ — числа не цитировать" if len(fs) < int(den) else ""))
    print(f"листов {sheets}, кривых с эталоном {len(rows)}, пропусков {sum(skip.values())}")
    for s, c in sorted(skip.items(), key=lambda z: -z[1])[:6]:
        print(f"    ⚠ {s}: {c}")

    def tally(ci, mi):
        hon = sum(1 for r in rows if HON(r[mi], r[ci]))
        nocov = sum(1 for r in rows if r[ci] < 0.9)
        badmed = sum(1 for r in rows if r[ci] >= 0.9 and not HON(r[mi], r[ci]))
        return hon, nocov, badmed

    N = max(1, len(rows))
    print(f"\n{'вариант':<34}{'ЧЕСТНЫХ':>9}{'доля':>8}{'cov<0.9':>10}{'med>3':>8}")
    for nm, ci, mi in (("V0 чистая постановка (§6.131)", 4, 5),
                       ("V1 по чернилам прод-маской", 6, 7),
                       ("V2 прод-маска + 1:1", 8, 9),
                       ("★ V3 по пикселям + 1:1", 10, 11),
                       ("V1ц прод-маска, точка=центр", 13, 14),
                       ("★ V3ц по пикселям, точка=центр", 15, 16),
                       ("★★ V3м то же + мостик разрывов", 18, 17)):
        h, nc, bm = tally(ci, mi)
        print(f"{nm:<34}{h:>9}{100*h/N:>7.1f}%{nc:>10}{bm:>8}")
    # ── что мостик даёт и чем он рискует: длина самого длинного разрыва ────────────────────
    gm = np.array([r[19] for r in rows if r[19]], float)
    if len(gm):
        print(f"\nмостик: медиана длиннейшего разрыва в кривой {np.median(gm):.0f} строк, "
              f"p90 {np.percentile(gm, 90):.0f}, максимум {gm.max():.0f}")
        for lim in (50, 200, 1000):
            sel = [r for r in rows if r[19] and r[19] <= lim]
            h = sum(1 for r in sel if HON(r[17], r[18]))
            if sel:
                print(f"   у кривых с разрывами ≤{lim:>4} строк ({len(sel):>4}): честных {h} "
                      f"({100*h/len(sel):.1f}%)")

    # ⚠ САНИТАРНЫЙ СЧЁТЧИК МЯГКОЙ МАСКИ (§6.65): тушь — единицы процентов листа.
    sf = np.array([r[12] for r in rows], float)
    bad = float((sf > 0.15).mean())
    print(f"\nмягкая маска V3 покрывает листа: медиана {100*np.median(sf):.1f}%, "
          f"p90 {100*np.percentile(sf, 90):.1f}%   "
          + ("★ в норме" if bad < 0.05 else f"⛔ у {100*bad:.0f}% кривых >15% — там V3 ловит бумагу"))
    print(f"\nдля сравнения (§6.115, тот же корпус 6863 кривых): правило 832 (12.1%), "
          f"прод 989 (14.4%), потолок раскладки 1975 (28.8%)")

    # ── ГДЕ ИМЕННО ТЕРЯЕТ V2: покрытие или ошибка ──────────────────────────────────────────
    cov = np.array([r[8] for r in rows], float)
    print(f"\nпокрытие чернилами (V2): медиана {np.median(cov):.3f}, "
          f"≥0.9 у {int((cov >= 0.9).sum())} кривых, ≥0.5 у {int((cov >= 0.5).sum())}")
    by_c = defaultdict(lambda: [0, 0])
    for r in rows:
        by_c[r[2]][0] += 1
        by_c[r[2]][1] += int(HON(r[9], r[8]))
    print("по цвету (оракульный выбор): " + ", ".join(
        f"{c} {v[1]}/{v[0]}" for c, v in sorted(by_c.items(), key=lambda z: -z[1][0])))


if a.sum:
    summarise()
else:
    i, n = (int(v) for v in a.shard.split("/"))
    collect(i, n)

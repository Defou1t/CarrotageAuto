r"""_cov_extend.py — ПРОТОТИП МЕХАНИЗМА: продолжение трассы НА КОНЦАХ по одиночному узкому рану.

§6.112.1 сузил задачу до одного механизма: на концах тушь есть у 99% строк и в 65% случаев это
одиночный узкий ран; в дырах внутри половина строк вообще без чернил. Осторожный потолок — 98 кривых
из 378. ⚠ Но 98 посчитаны ОРАКУЛОМ: строка засчитывалась, если тушь есть В ±4px ОТ ТОЧКИ ЭКСПЕРТА.
Настоящий механизм эксперта не видит и идёт от последней своей точки. Здесь он и написан.

МЕХАНИЗМ (только концы, дыры не трогаются вовсе):
  от последней точки трассы шагаем на строку наружу и берём ран, если ВСЕ условия разом:
    • ран ровно один в ±`--jump` px от текущей x;                  (иначе — толчея, стоп)
    • в ±`--near` px нет других ранов;                             (иначе — толчея, стоп)
    • ширина рана ≤ `--wide` × собственной ширины линии;           (иначе — слияние, стоп)
    • строка внутри окна анализа.                                   (иначе — край, стоп)
  новая x — центр рана; он же становится якорем. Один отказ — конец продолжения (без «коаста»).

★★ ТРИ ПРОВЕРКИ, БЕЗ КОТОРЫХ ЧИСЛО НЕЛЬЗЯ ЦИТИРОВАТЬ:
 1. **ТОЧНОСТЬ ДОБИТОГО** — у нас есть эталон, поэтому меряется НАПРЯМУЮ: медиана |x_добитой −
    x_эксперта| по всем добавленным строкам. Если механизм уходит на соседа, это видно сразу.
 2. **НУЛЬ-КОНТРОЛЬ §6.20.3** — тот же прогон, но добавленные точки сдвинуты на `--null-shift` px.
    `cov` при этом растёт ТАК ЖЕ. Если и в этом варианте кривые «становятся честными», значит
    метрика покупается, а не улучшается, и весь прирост — артефакт. Печатается рядом, всегда.
 3. **MED ПЕРЕСЧИТЫВАЕТСЯ ПО ВСЕМ строкам** (свои + добитые), а не берётся из §6.111: добивка
    обязана портить med, если она врёт, — иначе контроль №2 бессмыслен.

⚠ Прод НЕ ТРОГАЕТСЯ: это офлайн-прототип по сохранённым пулам и картинкам. Если числа устоят,
следующий шаг — ручка в `auto/` и A/B на отгрузке (`_trace_prod_ab.py`), потому что цитировать
ветка позволяет только отгрузку.

  python _cov_extend.py [--shard 0/6] [--jump 6] [--near 40] [--wide 3.0] [--null-shift 1000]
"""
import sys, io, argparse, contextlib, pickle, collections
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from dataset_build import find_image
from auto import frame as F, meta as M, imaging as im
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--list", default=r"F:\nds\output\taskS\cov_written_short.tsv")
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--jump", type=int, default=6, help="на сколько px может уйти x за строку")
ap.add_argument("--near", type=int, default=40, help="окно «толчеи» по x, px")
ap.add_argument("--wide", type=float, default=3.0, help="во сколько раз шире своей линии = слияние")
ap.add_argument("--gate", choices=("strict", "margin"), default="strict",
                help="strict — никого в ±near; margin — сосед дальше запаса")
ap.add_argument("--margin", type=float, default=3.0, help="во сколько раз сосед должен быть дальше")
ap.add_argument("--bridge", type=int, default=0,
                help="на сколько строк разрешено ПЕРЕШАГНУТЬ разрыв чернил (точки в разрыве НЕ ставятся)")
ap.add_argument("--null-shift", type=int, default=1000, help="нуль-контроль §6.20.3")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--out", default=r"F:\nds\output\taskS\cov_extend.tsv")
a = ap.parse_args()
SH_I, SH_N = (int(v) for v in a.shard.split("/"))
p = DEFAULT.cv
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def runs(row):
    d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def step(fg, y, anchor, my_w, H, W):
    """Одна строка продолжения: (центр рана, причина) — центр None, если гейт не пропустил.
    ★ Причина возвращается ВСЕГДА: без неё непонятно, что чинить, если механизм встаёт рано."""
    if not (0 <= y < H):
        return None, "край картинки"
    s, e = runs(fg[y])
    if not len(s):
        return None, "нет туши"
    cen = (s + e - 1) / 2.0
    cand = np.flatnonzero((e > anchor - a.jump) & (s < anchor + a.jump + 1))
    if len(cand) == 0:
        return None, "нет туши под якорем"
    if len(cand) > 1:
        return None, "два рана под якорем"
    k = int(cand[0])
    # ⚠⚠ ПОРОГ СЛИЯНИЯ — ОТНОСИТЕЛЬНЫЙ ПЛЮС АБСОЛЮТНЫЙ ЗАПАС. Чистое `wide × своя ширина` у тонкой
    # линии вырождается: своя ширина 1px ⇒ ран в 4px уже «слияние», и продолжение вставало через
    # две строки (замер на GZ11 Rybal_135: 3,3,4,3,3,2,2px — обычное дрожание толщины штриха).
    # В `_cov_stop` того же дефекта почти нет: там класс ставится по МЕДИАНЕ по пропуску, а здесь
    # одна строка останавливает всё, поэтому чувствительность к порогу несоизмеримо выше.
    if my_w > 0 and (e[k] - s[k]) > max(a.wide * my_w, my_w + 4.0):
        return None, "широкий ран (слияние)"
    if a.gate == "strict":
        if int(((s < anchor + a.near) & (e > anchor - a.near)).sum()) > 1:
            return None, "толчея в ±near"
    else:
        # ★ ВАРИАНТ «ЗАПАС»: соседи рядом допускаются, если выбранный ран ЯВНО ближе прочих.
        # Строгий гейт («никого в ±40px») на плотных бланках BKZ запрещает почти всё — соседняя
        # кривая там ближе 40px постоянно. Смысл замера: отличить «механизм невозможен» от
        # «невозможен ИМЕННО ЭТОТ гейт», иначе приговор направлению будет вынесен по одной ручке.
        d = np.maximum(s - anchor, np.maximum(anchor - (e - 1), 0))
        d1 = float(d[k])
        oth = np.delete(d, k)
        # ⚠ Запас считается от ШИРИНЫ СВОЕЙ ЛИНИИ, а не от нуля: при `d1 = 0` (якорь внутри рана)
        # чистое `margin × d1` пропустило бы соседа в трёх пикселях, то есть гейта не было бы вовсе.
        if len(oth) and float(oth.min()) < a.margin * max(d1, my_w, 1.0):
            return None, "сосед не дальше запаса"
    return float(cen[k]), ""


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


rows = [l.rstrip("\n").split("\t") for l in open(a.list, encoding="utf-8")][1:]
want = collections.defaultdict(list)
for r in rows:
    want[r[0]].append(r[1])
sheets = sorted(want)
mine = [s for i, s in enumerate(sheets) if i % SH_N == SH_I]
print(f"список: {len(rows)} кривых на {len(sheets)} листах; ★ ШАРД {SH_I}/{SH_N}: {len(mine)} листов")

DUMP = {}
for root in a.pools:
    for f in Path(root).glob("*.pkl"):
        DUMP.setdefault(f.stem, f)
WELL = {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WELL[q.name] = q

res, SKIP, STOP, BR = [], collections.Counter(), collections.Counter(), []
add_err = []
nsheet = ncurve = 0

for name in mine:
    dmp, nl = DUMP.get(Path(name).stem[:60]), WELL.get(name)
    if dmp is None or nl is None:
        SKIP["нет дампа/nlgx"] += len(want[name]); continue
    img = find_image(nl)
    if not img:
        SKIP["нет картинки"] += len(want[name]); continue
    try:
        d = pickle.load(open(dmp, "rb"))
        rgb = im.load_rgb(str(img))
        with contextlib.redirect_stdout(io.StringIO()):
            m = M.parse_filename(nl.name, r"F:\nds\Auto\mnemonics.json")
            fr = F.frame_from_nlgx(str(nl), m, p, rgb=rgb)
            if getattr(fr, "row_shift", None) is not None:
                rgb = F.apply_row_shift(rgb, fr.row_shift)
            fg = im.ink_foreground(rgb, p) > 0
    except Exception as e:
        SKIP[f"падение {type(e).__name__}"] += len(want[name]); continue
    H, W = fg.shape
    nsheet += 1
    for nm in want[name]:
        gt, wtr = d["gts"].get(nm), d["written"].get(nm)
        if gt is None or wtr is None:
            SKIP["кривой нет в дампе"] += 1; continue
        ncurve += 1
        m0, c0 = err(wtr, gt)
        # своя ширина — по строкам, которые трасса взяла
        sw = []
        ys_have = sorted(wtr)
        for y in ys_have[::max(1, len(ys_have) // 60)]:
            if 0 <= y < H:
                s, e = runs(fg[y])
                x = int(round(wtr[y]))
                k = np.searchsorted(s, x, "right") - 1
                if k >= 0 and x < e[k]:
                    sw.append(e[k] - s[k])
        my_w = float(np.median(sw)) if sw else 0.0
        ext = dict(wtr)
        added = []
        for end, dirn in ((max(ys_have), +1), (min(ys_have), -1)):
            anchor, y = ext[end], end + dirn
            while True:
                if not (fr.top_y <= y < fr.bottom_y):
                    STOP["край окна анализа"] += 1; break
                x, why = step(fg, y, anchor, my_w, H, W)
                if x is None and a.bridge and why.startswith("нет туши"):
                    # ★★ ПЕРЕШАГ ЧЕРЕЗ РАЗРЫВ ЧЕРНИЛ — И ЭТО НЕ КОАСТ §6.20.3. Коаст ВЫДУМЫВАЛ точки
                    # внутри разрыва (и провалил нуль-контроль: сдвиг добитых на 1000px ничего не
                    # менял). Здесь в разрыве НЕ ставится ни одной точки: трасса просто возобновляется
                    # там, где тушь вернулась, поэтому каждая добавленная точка стоит на чернилах.
                    for k in range(1, a.bridge + 1):
                        yy = y + dirn * k
                        if not (fr.top_y <= yy < fr.bottom_y):
                            break
                        x2, _ = step(fg, yy, anchor, my_w, H, W)
                        if x2 is not None:
                            BR.append(k); x, y = x2, yy
                            break
                if x is None:
                    STOP[why] += 1; break
                ext[y] = x; added.append(y); anchor = x
                y += dirn
        m1, c1 = err(ext, gt)
        # ★ прямая точность добитого — эталон у нас есть
        de = [abs(ext[y] - gt[y]) for y in added if y in gt]
        add_err += de
        # ★ нуль-контроль §6.20.3: те же строки, но x сдвинут
        null = dict(wtr)
        for y in added:
            null[y] = ext[y] + a.null_shift
        mN, cN = err(null, gt)
        res.append((name, nm, len(gt), len(added), c0, m0, c1, m1, c1 >= 0.9 and (m1 or 9) <= 3.0,
                    cN, mN, cN >= 0.9 and (mN or 9) <= 3.0,
                    float(np.median(de)) if de else -1.0))

W_ = 100
print(f"\n{'='*W_}")
print(f"ОБРАБОТАНО листов {nsheet}, кривых {ncurve}; пропущено {sum(SKIP.values())}")
for k, v in SKIP.items():
    print(f"    пропуск «{k}»: {v}")
want_n = sum(len(want[s]) for s in mine)
print(f"  СВЕРКА СПИСКА: {ncurve} + {sum(SKIP.values())} = {ncurve+sum(SKIP.values())} против "
      f"{want_n}   " + ("★ СОШЛОСЬ" if ncurve + sum(SKIP.values()) == want_n else "⛔ РАСХОЖДЕНИЕ"))
if res:
    add = np.array([r[3] for r in res], float)
    hon = sum(r[8] for r in res)
    honN = sum(r[11] for r in res)
    print(f"{'='*W_}")
    print(f"ДОБАВЛЕНО СТРОК: всего {int(add.sum())}, медиана на кривую {np.median(add):.0f}, "
          f"кривых без единой добавки {int((add == 0).sum())}")
    if add_err:
        ae = np.array(add_err)
        print(f"\n★ ТОЧНОСТЬ ДОБИТОГО (|добитая − эксперт|, {len(ae)} строк): медиана {np.median(ae):.1f}px, "
              f"90-й перцентиль {np.percentile(ae, 90):.0f}px, доля ≤3px {100*np.mean(ae <= 3):.1f}%")
    if BR:
        b = np.array(BR)
        print(f"\nПЕРЕШАГОВ ЧЕРЕЗ РАЗРЫВ: {len(b)}, медиана длины {np.median(b):.0f} строк, "
              f"90-й перцентиль {np.percentile(b, 90):.0f}")
    st = sum(STOP.values()) or 1
    print(f"\nПОЧЕМУ ПРОДОЛЖЕНИЕ ВСТАЛО ({st} остановов, по два на кривую — верх и низ)")
    for k, v in STOP.most_common():
        print(f"  {k:<26}{v:>7}{100*v/st:>7.1f}%  " + "█" * int(36 * v / st))
    print(f"\nCOV: медиана {np.median([r[4] for r in res]):.2f} → {np.median([r[6] for r in res]):.2f}")
    print(f"MED: медиана {np.median([r[5] for r in res]):.2f} → "
          f"{np.median([r[7] for r in res if r[7] is not None]):.2f}px")
    print(f"\n{'='*W_}\n★★ СТАЛИ ЧЕСТНЫМИ (med≤3 И cov≥0.9): {hon} из {len(res)}\n{'='*W_}")
    print(f"  ⚠ НУЛЬ-КОНТРОЛЬ §6.20.3 (те же строки, сдвиг {a.null_shift}px): {honN} из {len(res)}")
    # ⚠ При hon = 0 сравнивать нечего: «артефакт» тут был бы ложным приговором механизму,
    # который просто ничего не добавил. Приговор нуль-контроля имеет смысл только при hon > 0.
    print("  ⇒ " + ("ПРИРОСТА НЕТ ВОВСЕ — нуль-контролю нечего проверять" if hon == 0 else
                    "★ ПРИРОСТ НАСТОЯЩИЙ: сдвиг убивает его" if honN * 3 < hon else
                    "⛔ ПРИРОСТ АРТЕФАКТ: метрика покупается покрытием, как в §6.20.3"))
    dump = Path(a.out if SH_N == 1 else a.out.replace(".tsv", f"_{SH_I}of{SH_N}.tsv"))
    with open(dump, "w", encoding="utf-8") as fh:
        fh.write("лист\tкривая\tстрокGT\tдобавлено\tcov0\tmed0\tcov1\tmed1\tчестная\t"
                 "covN\tmedN\tчестная_нуль\tmed_добитого\n")
        for r in res:
            fh.write("\t".join("" if v is None else
                               (f"{v:.3f}" if isinstance(v, float) else str(int(v) if isinstance(v, bool) else v))
                               for v in r) + "\n")
    print(f"\n★ выгружено покривой → {dump}")

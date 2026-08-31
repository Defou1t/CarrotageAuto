r"""_cov_ink.py — ЕСТЬ ЛИ ТУШЬ ПОД ПРОПУЩЕННЫМИ СТРОКАМИ: верхняя граница рычага покрытия (§6.111).

§6.111 нашёл адрес покрытия: **378 кривых записаны ТОЧНО (med≤3px), но КОРОТКО (cov<0.9)**, и отбор
им проходить не надо — честными их делает только удлинение. Осталось узнать, есть ли что удлинять.

ВОПРОС (тот же, что у §6.23, и та же механика гейта): для каждой строки эксперта, которой НЕТ в
записанной трассе, — есть ли под ней тушь в ±tol px?
  • ВНЕ ОКНА  — строка за [frame.top_y, frame.bottom_y): обрезало окно анализа;
  • ЕСТЬ ТУШЬ — трасса оборвалась на ЖИВОЙ кривой ⇒ восстановимо по туши;
  • НЕТ ТУШИ  — чернил нет (пунктир/выцвет) ⇒ ловушка §6.2, добивать нельзя (§6.20.3).
⚠ Прежние ответы НЕ переносятся на этот список: §6.23 мерил 19 кривых ОДНОЙ скважины (46% есть
тушь), §6.28.1 — те же строки на ОРАКУЛЬНОЙ полосе (0%). Здесь 378 кривых, 324 листа, 104 скважины.

★★ ГЛАВНОЕ ЧИСЛО — НЕ ДОЛЯ СТРОК, А СКОЛЬКО КРИВЫХ ПЕРЕХОДЯТ ПОРОГ. Кривой с cov 0.76 нужно добрать
14 пунктов; если тушь есть под половиной пропущенных строк, порога 0.9 она всё равно не достигнет.
Поэтому считается `cov_после = (покрыто + строки_с_тушью) / строк_GT` и сколько таких кривых
дотягивают до 0.9. Это и есть ВЕРХНЯЯ ГРАНИЦА рычага — верхняя, потому что предполагает идеальную
добивку по всей туши и сохранение med≤3px.
⚠ Стенд НИЧЕГО НЕ ДОБИВАЕТ и ничего не меняет: он только меряет потенциал. Любой механизм, который
потом этот потенциал начнёт брать, обязан пройти нуль-контроль §6.20.3 (сдвиг добитых точек на
1000px не должен оставлять метрику на месте).

КОНТРОЛИ: (1) `cov` пересчитывается из дампа и сверяется с колонкой входного TSV — если разошлось,
читается не тот объект; (2) сверка «обработано + пропущено = длина списка» (§6.106); (3) доля туши
считается сразу при трёх допусках (2/4/8 px), чтобы вывод не висел на одном пороге.

  python _cov_ink.py [--list ...] [--tol 4] [--shard 0/1]
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
ap.add_argument("--tol", type=int, default=4, help="допуск по x, px (§6.23 брал 4)")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--out", default=r"F:\nds\output\taskS\cov_ink.tsv")
a = ap.parse_args()
SH_I, SH_N = (int(v) for v in a.shard.split("/"))
TOLS = sorted({2, a.tol, 8})
p = DEFAULT.cv

# ---- вход: список кривых и указатель дампов -------------------------------------------------
rows = [l.rstrip("\n").split("\t") for l in open(a.list, encoding="utf-8")][1:]
want = collections.defaultdict(list)          # лист → [(кривая, cov_из_списка)]
for r in rows:
    want[r[0]].append((r[1], float(r[3])))
sheets = sorted(want)
mine = [s for i, s in enumerate(sheets) if i % SH_N == SH_I]
print(f"список: {len(rows)} кривых на {len(sheets)} листах; ★ ШАРД {SH_I}/{SH_N}: {len(mine)} листов")

DUMP = {}
for root in a.pools:
    for f in Path(root).glob("*.pkl"):
        DUMP.setdefault(f.stem, f)            # первый каталог по порядку — как в §6.111
WELL = {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WELL[q.name] = q

TOT = {t: collections.Counter() for t in TOLS}
per_curve, SKIP = [], collections.Counter()
nsheet = ncurve = 0
cov_mismatch = 0

for name in mine:
    key = Path(name).stem[:60]
    dmp = DUMP.get(key)
    nl = WELL.get(name)
    if dmp is None or nl is None:
        SKIP["нет дампа/nlgx"] += len(want[name]); continue
    img = find_image(nl)
    if not img:
        SKIP["нет картинки"] += len(want[name]); continue
    try:
        d = pickle.load(open(dmp, "rb"))
        # ⚠⚠ ЗАГРУЗЧИК — ПРОДОВСКИЙ, А НЕ `cv2.imread`. Две причины, обе замерены: (1) `imread` на
        # Windows не открывает пути с кириллицей и молча возвращает None — партия Бильче-Волыця
        # (`БВп_*`, `Уп_*`) выпадала целиком, 7 кривых из 378; (2) у той же партии биты карты полос,
        # и `load_rgb` читает такой TIFF частично и ГРОМКО (§6.108) — то есть стенд обязан видеть
        # ровно ту картинку, что видел прод, включая её обрезанность.
        rgb = im.load_rgb(str(img))
        with contextlib.redirect_stdout(io.StringIO()):
            m = M.parse_filename(nl.name, r"F:\nds\Auto\mnemonics.json")
            fr = F.frame_from_nlgx(str(nl), m, p, rgb=rgb)
            # ⚠ СДВИГ СТРОК ОБЯЗАТЕЛЕН И ИМЕННО ЗДЕСЬ: трассы пула и GT живут в СДВИНУТОЙ системе
            # (пайплайн `_pool_oracle` считал их после сдвига), а картинка с диска — в исходной.
            # Без этого «есть тушь» мерилось бы по чужим строкам и дало бы правдоподобный мусор.
            if getattr(fr, "row_shift", None) is not None:
                rgb = F.apply_row_shift(rgb, fr.row_shift)
            fg = im.ink_foreground(rgb, p) > 0
    except Exception as e:
        SKIP[f"падение {type(e).__name__}"] += len(want[name]); continue
    H, W = fg.shape
    nsheet += 1
    for nm, cov_list in want[name]:
        gt, wtr = d["gts"].get(nm), d["written"].get(nm)
        if gt is None or wtr is None:
            SKIP["кривой нет в дампе"] += 1; continue
        ncurve += 1
        have = sum(1 for y in gt if y in wtr)
        if abs(have / max(1, len(gt)) - cov_list) > 0.005:
            cov_mismatch += 1
        missing = [y for y in gt if y not in wtr]
        lo_t, hi_t = min(wtr), max(wtr)
        cls = {t: collections.Counter() for t in TOLS}
        ink_at = {t: 0 for t in TOLS}
        ink_hole = {t: 0 for t in TOLS}
        for y in missing:
            if y < fr.top_y or y >= fr.bottom_y or not (0 <= y < H):
                for t in TOLS:
                    cls[t]["вне_окна"] += 1
                continue
            x = int(round(gt[y]))
            for t in TOLS:
                if fg[y, max(0, x - t):min(W, x + t + 1)].any():
                    cls[t]["есть_тушь"] += 1
                    ink_at[t] += 1
                    ink_hole[t] += (lo_t <= y <= hi_t)
                else:
                    cls[t]["нет_туши"] += 1
        for t in TOLS:
            TOT[t].update(cls[t])
        cov_after = {t: (have + ink_at[t]) / max(1, len(gt)) for t in TOLS}
        per_curve.append((name, nm, len(gt), len(missing), cov_list,
                          {t: (ink_at[t], cov_after[t], ink_hole[t]) for t in TOLS}))

W_ = 96
print(f"\n{'='*W_}")
print(f"ОБРАБОТАНО листов {nsheet}, кривых {ncurve}; пропущено {sum(SKIP.values())}")
for k, v in SKIP.items():
    print(f"    пропуск «{k}»: {v}")
print(f"  СВЕРКА СПИСКА: {ncurve} + {sum(SKIP.values())} = {ncurve+sum(SKIP.values())} против "
      f"{sum(len(want[s]) for s in mine)}   "
      + ("★ СОШЛОСЬ" if ncurve + sum(SKIP.values()) == sum(len(want[s]) for s in mine)
         else "⛔ РАСХОЖДЕНИЕ"))
print(f"  КОНТРОЛЬ cov: расходится со списком у {cov_mismatch} кривых из {ncurve}   "
      + ("★ СОШЛОСЬ" if cov_mismatch == 0 else "⛔ ЧИТАЕТСЯ НЕ ТОТ ОБЪЕКТ"))
print(f"{'='*W_}")

for t in TOLS:
    s = sum(TOT[t].values()) or 1
    mark = " ★(порог §6.23)" if t == 4 else ""
    print(f"\nПРОПУЩЕННЫЕ СТРОКИ, допуск ±{t}px{mark}: всего {s}")
    for k in ("вне_окна", "есть_тушь", "нет_туши"):
        print(f"  {k:<12}{TOT[t][k]:>10}{100*TOT[t][k]/s:>7.1f}%  " + "█" * int(46 * TOT[t][k] / s))

if per_curve:
    print(f"\n{'='*W_}\n★★ СКОЛЬКО КРИВЫХ ПЕРЕХОДЯТ ПОРОГ 0.9, ЕСЛИ ДОБИТЬ ТОЛЬКО СТРОКИ С ТУШЬЮ\n{'='*W_}")
    print(f"  {'допуск':<10}{'дотянут':>9}{'из':>6}{'доля':>8}   медиана cov после")
    for t in TOLS:
        ok = [c for c in per_curve if c[5][t][1] >= 0.9]
        med = np.median([c[5][t][1] for c in per_curve])
        print(f"  ±{t}px{'':<6}{len(ok):>9}{len(per_curve):>6}{100*len(ok)/len(per_curve):>7.1f}%"
              f"        {med:.2f}")
    t = a.tol
    ok = [c for c in per_curve if c[5][t][1] >= 0.9]
    ink_tot = sum(c[5][t][0] for c in per_curve)
    hole_tot = sum(c[5][t][2] for c in per_curve)
    print(f"\n  при ±{t}px строк с тушью {ink_tot}, из них ВНУТРИ пролёта трассы {hole_tot} "
          f"({100*hole_tot/max(1,ink_tot):.0f}%) — остальное концы")
    print(f"  ⇒ верхняя граница рычага: {len(ok)} кривых к нынешним 832 честным "
          f"({100*len(ok)/832:+.1f}% к счёту честных на этой выборке)")

    dump = Path(a.out if SH_N == 1 else a.out.replace(".tsv", f"_{SH_I}of{SH_N}.tsv"))
    with open(dump, "w", encoding="utf-8") as fh:
        fh.write("лист\tкривая\tстрокGT\tпропущено\tcov\tстрок_с_тушью\tв_дырах\tcov_после\tдотянул\n")
        for name, nm, ng, nmis, cov, dd in per_curve:
            ink, cova, hole = dd[a.tol]
            fh.write(f"{name}\t{nm}\t{ng}\t{nmis}\t{cov:.3f}\t{ink}\t{hole}\t{cova:.3f}\t"
                     f"{int(cova >= 0.9)}\n")
    print(f"\n★ выгружено покривой → {dump}")

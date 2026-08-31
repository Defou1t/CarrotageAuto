r"""_cov_stop.py — ЧТО ИМЕННО МОЖНО ДОБРАТЬ БЕЗОПАСНО: деление резерва §6.112 на чинимое и опасное.

§6.112 показал: под **49.3%** пропущенных строк тушь есть, и 202 кривые из 378 перешли бы порог 0.9.
Но «есть тушь» ≠ «надо было продолжать»: там, где линия СЛИВАЕТСЯ с соседом, продолжение — это латч
(§6.28.2), и добивка купит метрику, а не качество (нуль-контроль §6.20.3). Значит потолок §6.112
надо разделить на часть, которую можно брать, и часть, которую трогать нельзя.

ЕДИНИЦА СЧЁТА — ПРОПУСК ЦЕЛИКОМ, А НЕ «ТОЧКА ОСТАНОВА». ⚠ Первая редакция брала первую строку
каждого пропуска и получила «93.8% без туши»: у одной кривой таких границ ОКОЛО ТЫСЯЧИ, потому что
трасса роняет одиночные строки там, где чернил нет. Это верно, но отвечает на другой вопрос — масса
покрытия лежит не в тысяче однострочных провалов, а в длинных пропусках. Поэтому здесь пропуски
берутся ЦЕЛИКОМ и всё взвешивается СТРОКАМИ.

КЛАССЫ ПРОПУСКА (по туши внутри него):
  • НЕТ ТУШИ   — предел данных (§6.2), добирать нечего;
  • ★ ЧИСТЫЙ   — тушь есть, ран узкий (≤ `--wide` × своей ширины) и рядом никого ⇒ БЕЗОПАСНО;
  • ⚠ ШИРОКИЙ  — ран много шире своей линии ⇒ слияние/пересечение, продолжать нельзя;
  • ⚠ ТОЛЧЕЯ   — в ±`--near` px ещё раны ⇒ выбор неоднозначен, это задача селектора, не добивки.
Признак ширины взят не с потолка: §6.20.4 замерил 8050 моментов ухода, у своего рана med 10.0px,
у ошибочно взятого 2.0px — ширина оказалась самым информативным признаком ветки.
★ «Своя ширина» меряется по строкам, которые трасса ВЗЯЛА: у CALI линия 3px, у GZ 15px, и общий
порог в пикселях сравнивал бы разное.

★★ ГЛАВНЫЙ ВЫВОД СТЕНДА — ОСТОРОЖНЫЙ ПОТОЛОК: сколько кривых переходят 0.9, если добрать ТОЛЬКО
чистые пропуски. Разница с 202 из §6.112 и есть цена запрета трогать слияния.

  python _cov_stop.py [--shard 0/6] [--tol 4] [--wide 3.0] [--near 40]
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
ap.add_argument("--tol", type=int, default=4)
ap.add_argument("--near", type=int, default=40, help="окно «толчеи» по x, px")
ap.add_argument("--wide", type=float, default=3.0, help="во сколько раз шире своей линии = слияние")
ap.add_argument("--probe", type=int, default=15, help="сколько строк пропуска щупать на ширину")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--out", default=r"F:\nds\output\taskS\cov_stop.tsv")
a = ap.parse_args()
SH_I, SH_N = (int(v) for v in a.shard.split("/"))
p = DEFAULT.cv
NO, CLEAN, WIDE, CROWD = ("1 НЕТ ТУШИ — предел данных", "2 ★ ЧИСТЫЙ — брать безопасно",
                          "3 ⚠ ШИРОКИЙ — слияние, не трогать", "4 ⚠ ТОЛЧЕЯ — задача выбора")
CLS = [NO, CLEAN, WIDE, CROWD]


def run_at(row, x, tol):
    """(ширина рана под x, сколько ранов рядом) или (0, 0), если туши нет."""
    d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    if not len(s):
        return 0, 0
    k = np.searchsorted(s, x + tol, "right") - 1
    if k < 0 or (x - tol) >= e[k]:
        return 0, 0
    return int(e[k] - s[k]), int(((s < x + a.near) & (e > x - a.near)).sum())


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

seg_n = collections.Counter()          # пропусков по классам
seg_rows = collections.Counter()       # СТРОК по классам (главный вес)
ink_rows = collections.Counter()       # строк С ТУШЬЮ по классам
kind_rows = {"конец": collections.Counter(), "дыра": collections.Counter()}
per_curve, SKIP = [], collections.Counter()
nsheet = ncurve = 0
widths, ownw = [], []

for name in mine:
    dmp, nl = DUMP.get(Path(name).stem[:60]), WELL.get(name)
    if dmp is None or nl is None:
        SKIP["нет дампа/nlgx"] += len(want[name]); continue
    img = find_image(nl)
    if not img:
        SKIP["нет картинки"] += len(want[name]); continue
    try:
        d = pickle.load(open(dmp, "rb"))
        rgb = im.load_rgb(str(img))       # ⚠ прод-загрузчик: кириллица + обрезанные TIFF (§6.112)
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
        ys = sorted(gt)
        have = set(wtr)
        lo_t, hi_t = min(wtr), max(wtr)
        sw = []
        for y in ys[::max(1, len(ys) // 60)]:
            if y in have and 0 <= y < H:
                w, _ = run_at(fg[y], int(round(wtr[y])), a.tol)
                if w:
                    sw.append(w)
        my_w = float(np.median(sw)) if sw else 0.0
        ownw.append(my_w)
        # пропуски ЦЕЛИКОМ: максимальные серии подряд идущих строк GT без трассы
        segs, cur = [], []
        for y in ys:
            if y in have:
                if cur:
                    segs.append(cur); cur = []
            else:
                cur.append(y)
        if cur:
            segs.append(cur)
        cc = collections.Counter()
        clean_rows = 0
        for seg in segs:
            seg = [y for y in seg if fr.top_y <= y < fr.bottom_y and 0 <= y < H]
            if not seg:
                continue
            kind = "конец" if (seg[-1] < lo_t or seg[0] > hi_t) else "дыра"
            step = max(1, len(seg) // a.probe)
            ws, nears, ink = [], [], 0
            for y in seg[::step]:
                w, near = run_at(fg[y], int(round(gt[y])), a.tol)
                if w:
                    ws.append(w); nears.append(near); ink += 1
            share = ink / max(1, len(seg[::step]))
            if not ws:
                c = NO
            else:
                mw = float(np.median(ws))
                widths.append(mw)
                if my_w > 0 and mw > a.wide * my_w:
                    c = WIDE
                elif float(np.median(nears)) > 1:
                    c = CROWD
                else:
                    c = CLEAN
            seg_n[c] += 1
            seg_rows[c] += len(seg)
            ink_rows[c] += int(round(share * len(seg)))
            kind_rows[kind][c] += len(seg)
            cc[c] += len(seg)
            if c == CLEAN:
                clean_rows += int(round(share * len(seg)))
        cov_now = len(have & set(ys)) / max(1, len(ys))
        cov_safe = (len(have & set(ys)) + clean_rows) / max(1, len(ys))
        per_curve.append((name, nm, len(ys), len(segs), my_w, round(cov_now, 3),
                          clean_rows, round(cov_safe, 3), int(cov_safe >= 0.9)))

W_ = 96
print(f"\n{'='*W_}")
print(f"ОБРАБОТАНО листов {nsheet}, кривых {ncurve}; пропущено {sum(SKIP.values())}")
for k, v in SKIP.items():
    print(f"    пропуск «{k}»: {v}")
want_n = sum(len(want[s]) for s in mine)
print(f"  СВЕРКА СПИСКА: {ncurve} + {sum(SKIP.values())} = {ncurve+sum(SKIP.values())} против "
      f"{want_n}   " + ("★ СОШЛОСЬ" if ncurve + sum(SKIP.values()) == want_n else "⛔ РАСХОЖДЕНИЕ"))
print(f"{'='*W_}")
S, R, I = sum(seg_n.values()) or 1, sum(seg_rows.values()) or 1, sum(ink_rows.values()) or 1
print(f"ПРОПУСКОВ {S}, строк в них {R}, из них с тушью {I} "
      f"(порог «широкий» = {a.wide}× своей ширины, толчея в ±{a.near}px)")
print(f"\n{'класс пропуска':<38}{'пропусков':>10}{'строк':>10}{'доля строк':>12}{'строк с тушью':>15}")
for c in CLS:
    print(f"  {c:<36}{seg_n[c]:>10}{seg_rows[c]:>10}{100*seg_rows[c]/R:>11.1f}%{ink_rows[c]:>15}")
print(f"\nКОНЦЫ ПРОТИВ ДЫР (в строках)")
for kind in ("конец", "дыра"):
    k = kind_rows[kind]; ks = sum(k.values()) or 1
    print(f"  {kind:<8}{ks:>10} строк:  " + "  ".join(f"{c.split()[0]} {100*k[c]/ks:.0f}%" for c in CLS))
if widths:
    print(f"\nШИРИНА РАНА в пропуске: медиана {np.median(widths):.0f}px; "
          f"своя линия {np.median([w for w in ownw if w > 0]):.0f}px")
if per_curve:
    ok = sum(r[8] for r in per_curve)
    print(f"\n{'='*W_}\n★★ ОСТОРОЖНЫЙ ПОТОЛОК: добираем ТОЛЬКО чистые пропуски\n{'='*W_}")
    print(f"  переходят порог 0.9: {ok} из {len(per_curve)} ({100*ok/len(per_curve):.1f}%)")
    print(f"  (у §6.112, где добиралась ВСЯ тушь без разбора, было 202 из 378 = 53.4%)")
    dump = Path(a.out if SH_N == 1 else a.out.replace(".tsv", f"_{SH_I}of{SH_N}.tsv"))
    with open(dump, "w", encoding="utf-8") as fh:
        fh.write("лист\tкривая\tстрокGT\tпропусков\tсвоя_ширина\tcov\tчистых_строк\tcov_безопасн\t"
                 "дотянул\n")
        for r in per_curve:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"\n★ выгружено покривой → {dump}")

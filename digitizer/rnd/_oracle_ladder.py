r"""_oracle_ladder.py — ЛЕСТНИЦА ОРАКУЛОВ: детекция против идентичности, по ВСЕЙ выборке.

⚠⚠ ЗАЧЕМ (§6.93). §6.91 показал: 1543 кривые из 1889 (81.7%) не берутся ничем, кроме распознавания,
и у 40% лучшая доступная ТРАССА лежит дальше 100px. Но «трасса» — уже продукт нашего трассировщика,
поэтому по ней нельзя понять, чего не хватает: сигнала или выбора. Здесь мерится то, чего в проекте
не мерили НИ РАЗУ — есть ли тушь экспертной кривой в наших МАСКАХ и можно ли попасть в кривую,
всего лишь правильно выбирая РАН на каждой строке.

СТУПЕНИ (на кривую, шаг по строкам --step):
  L0-ink  — доля строк GT, где в `imaging.ink_foreground` (детекция, гейт §6.65) есть чернило ±3px;
  L0-blk  — то же в МАСКЕ ТРАССИРОВКИ `trace2d._color_fg(rgb,'black')` (без гейта). Расхождение
            L0-ink и L0-blk ловит асимметрию гейтов между детекцией и трассировкой;
  L1      — ОРАКУЛЬНЫЙ выбор рана: ближайший к точке GT ран маски трассировки, медиана |Δx|.
            Это верхняя граница ЛЮБОГО трассировщика, который лишь выбирает ран на строке;
  контроль — ширина рана под точкой GT и число ранов на строку. ⚠ БЕЗ НЕГО L1 БЕССМЫСЛЕН: если ран
            во весь трек, L1=0 достигается тривиально и ничего не обещает;
  merge   — доля строк, где ран с точкой GT содержит ТАКЖЕ точку ДРУГОЙ экспертной кривой (прокси
            физической неразличимости);
  str/deg — доля точек GT, съеденных `structure_mask`, и попавших в вырожденный цветовой канал.

РАЗВИЛКА, КОТОРУЮ ЭТО РЕШАЕТ:
  L0 низкое                  ⇒ ДЕТЕКЦИЯ (порог темноты/гейт). Лечится константой, дни.
  L0 высокое и L1 ≤3px       ⇒ НАЗНАЧЕНИЕ: сигнал есть и отделён, ошибается наш выбор. Это адрес
                               многоцелевого трекинга (§6.92: laptrack/Stone Soup) и декодера пути.
  L0 высокое, L1 велико      ⇒ ран не тот даже локально: мерить ширину/слияние ранов.
⚠ Основа зонда — `scratchpad\_mask_probe.py`, написанный при разборе направления 31.07 на 6 листах
(22 кривые). Здесь он масштабирован на 702 листа, шардирован и агрегирован; выводы того разбора
до этого прогона остаются ГИПОТЕЗОЙ на 6 листах (проект дважды обжигался на выводах с 5-24 кривых).

  <ComfyUI>\python_embeded\python.exe _oracle_ladder.py [--shard 0/8] [--step 5]
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
ap.add_argument("--step", type=int, default=5, help="каждая N-я строка эталона")
ap.add_argument("--shard", default="0/1")
# ⚠⚠ §6.94, вторая правка: маска, в которой ищутся И тушь, И РАНЫ, должна быть ОДНА. В первой
# редакции наличие туши считалось по маске цвета, а раны — по чёрной, и от этого появилась ложная
# корзина «ран не тот локально» (20 кривых, L1 210px) — артефакт смешения, а не находка.
#   ink   — ★ по умолчанию: `imaging.ink_foreground`, объединение всех обнаруженных чернил. Это
#           верхняя граница «тушь вообще найдена детекцией» и единственный вариант, не зависящий от
#           словарного цвета;
#   black — чёрная маска трассировки (как считала первая лестница);
#   color — маска СЛОВАРНОГО цвета кривой. ⚠ Замер показал, что словарь врёт: у 139 «красных»
#           кривых чёрная маска находит тушь в 42% случаев, а красная — в 19% (§6.13).
ap.add_argument("--mask", default="ink", choices=["ink", "black", "color"])
ap.add_argument("--out", default=r"F:\nds\output\taskS\oracle_ladder")
a = ap.parse_args()
SH_I, SH_N = (int(v) for v in a.shard.split("/"))
P = Config().cv

C = pickle.load(open(a.cache, "rb"))
WELL = dict(zip(C["names"], C["wells"]))
PHON = {}                                        # (лист, кривая) → прод записал честно
for (si, nm), v in C["phon"].items():
    PHON[(C["names"][si], nm)] = v

WLG = {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WLG[q.name] = q
for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
    WLG.setdefault(q.name, q)

todo = sorted(n for n in C["names"] if n in WLG)
todo = [n for i, n in enumerate(todo) if i % SH_N == SH_I]
print(f"шард {SH_I}/{SH_N}: листов {len(todo)} из {len(C['names'])}, шаг по строкам {a.step}")

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
        paper = int(np.percentile(im.value_channel(rgb)[::4, ::4], 90))
        ink = im.ink_foreground(rgb, P) > 0
        blk = T._color_fg(rgb, "black", P)
        # ⚠⚠ ИСПРАВЛЕНИЕ §6.94: первая редакция брала ЧЁРНУЮ маску для ВСЕХ кривых, а прод трассирует
        # линию в маске ЕЁ цвета (`trace2d`, fg_cache[L.color]). Красная тушь (V≈200) не проходит
        # dark_v=110 в принципе, поэтому все SP/PS (139 кривых, словарный red) ложно попадали в
        # корзину «туши нет». Здесь для каждой кривой берётся маска её СЛОВАРНОГО цвета, как у прода.
        mn = Config().mnemonics
        fgc = {"black": blk}
        for nm0 in gts:
            c0 = (M.curve_info(nm0, mn) or {}).get("color")
            if c0 and c0 not in fgc:
                try:
                    fgc[c0] = T._color_fg(rgb, c0, P)
                except Exception:
                    fgc[c0] = blk
        stc = im.structure_mask(rgb, P)
        chans = im.color_channels(rgb, P)
        dmask = None
        for c, m in chans.items():
            if float(m.mean()) > P.color_max_frac:
                dmask = m if dmask is None else (dmask | m)
    except Exception as e:
        print(f"  ПАДЕНИЕ {name[:44]}: {type(e).__name__}: {e}")
        continue

    for nm, gt in gts.items():
        ys = sorted(gt)[::a.step]
        if len(ys) < 30:
            continue
        xs = np.array([gt[y] for y in ys], float)
        lo = max(0, int(xs.min()) - 150); hi = min(W, int(xs.max()) + 151)
        others = [g for kk, g in gts.items() if kk != nm]
        colname = (M.curve_info(nm, mn) or {}).get("color") or "black"
        cfg_m = fgc.get(colname, blk)
        # ★ ЕДИНАЯ маска для наличия туши И для ранов (см. --mask)
        run_m = ink if a.mask == "ink" else (blk if a.mask == "black" else cfg_m)
        c_ink = c_blk = c_str = c_deg = c_mrg = c_col = 0
        d1, wid, nrun = [], [], []
        for y, x in zip(ys, xs):
            if not (0 <= y < H):
                continue
            xi = int(round(x)); x0 = max(0, xi - 3); x1 = min(W, xi + 4)
            c_ink += bool(ink[y, x0:x1].any())
            c_blk += bool(blk[y, x0:x1].any())
            c_col += bool(cfg_m[y, x0:x1].any())          # маска ЦВЕТА кривой, как у прода
            c_str += bool(stc[y, max(0, xi - 1):xi + 2].any())
            if dmask is not None:
                c_deg += bool(dmask[y, max(0, xi - 1):xi + 2].any())
            rr = im.row_runs(run_m[y, lo:hi])
            if not rr:
                continue
            rr = [(p + lo, q + lo, cc + lo) for p, q, cc in rr]
            nrun.append(len(rr))
            r = min(rr, key=lambda r: 0 if r[0] <= x <= r[1] else min(abs(r[0] - x), abs(r[1] - x)))
            inside = r[0] <= x <= r[1]
            d1.append(0.0 if inside else min(abs(r[0] - x), abs(r[1] - x)))
            if inside:
                wid.append(r[1] - r[0] + 1)
                for g in others:
                    xo = g.get(y)
                    if xo is not None and r[0] <= xo <= r[1] and abs(xo - x) > 3:
                        c_mrg += 1
                        break
        kk = max(1, len(ys))
        rows.append(dict(sheet=name, curve=nm, well=WELL.get(name, "?"),
                         fam=(WELL.get(name) or "?").split("_")[0], nrows=len(ys),
                         ink3=c_ink / kk, blk3=c_blk / kk, col3=c_col / kk, colname=colname,
                         maskmode=a.mask,
                         strf=c_str / kk, degf=c_deg / kk,
                         merge=c_mrg / kk, paper=paper,
                         l1med=float(np.median(d1)) if d1 else np.nan,
                         l1p90=float(np.percentile(d1, 90)) if d1 else np.nan,
                         widmed=float(np.median(wid)) if wid else np.nan,
                         nrunmed=float(np.median(nrun)) if nrun else np.nan,
                         prod=PHON.get((name, nm), 0)))
    del rgb, ink, blk, stc, chans, dmask
    if k % 10 == 0:
        print(f"  {k}/{len(todo)} листов, кривых {len(rows)}")

Path(a.out).mkdir(parents=True, exist_ok=True)
dst = Path(a.out) / f"rows_{SH_I}of{SH_N}.pkl"
pickle.dump(rows, open(dst, "wb"))
print(f"\nзаписано {len(rows)} кривых → {dst}")

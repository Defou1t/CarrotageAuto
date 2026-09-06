r"""_nodes_size.py — РАЗМЕР ЗАДАЧИ «РАЗРЕШИТЬ УЗЛЫ»: СКОЛЬКО ИХ НА КОРПУСЕ, А НЕ НА ОДНОМ ЛИСТЕ.

ОТКУДА ВОПРОС. Разбор одного листа (BKZ, 4 кривые) показал: кривые расходятся почти везде
(медианный зазор 93px) и сближаются лишь на 8.2% строк, а эти строки собираются в 71 УЗЕЛ.
Отсюда предложение: не вести линию 16000 строк, а собрать однозначные куски и разрешить десятки
узлов ГЛОБАЛЬНО. Но один лист — это один лист (§6.88: вывод по одному объекту не переносится).
Здесь та же величина считается по всему честному полю.

⚠ ЭТО ЗАМЕР ЗАДАЧИ, А НЕ РЕШЕНИЯ. Он не говорит, что глобальная сборка сработает; он говорит,
СКОЛЬКО решений ей пришлось бы принять. Если узлов на лист десятки — точный решатель уместен;
если тысячи — предложение отпадает, не будучи написанным.

★ Считается по ЭТАЛОНУ и только по нему: узел — это место, где две оцифрованные кривые ОДНОГО
ТРЕКА сходятся ближе `--near`. Картинка не нужна, прогон не нужен.
⚠ Узел по эталону — НИЖНЯЯ оценка: на скане рядом могут проходить и неоцифрованные линии (§6.190),
и они тоже создают развилку. Поэтому число «узлов на лист» ниже реального, и это сказано прямо.

  <ComfyUI>\python_embeded\python.exe _nodes_size.py
  … --near 12 --min-len 3
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--rows", default="pick_learn_honest2.pkl.dump")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--near", type=float, default=12.0, help="ближе этого — узел")
ap.add_argument("--min-len", type=int, default=3, help="узел короче — не узел, а касание пикселя")
ap.add_argument("--cap", type=int, default=0)
a = ap.parse_args()
TS = Path(a.ts)

trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]): int(r[3]["n_prod"]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q

sheets = sorted({r[0] for r in trk})
if a.cap:
    sheets = sheets[:a.cap]
print(f"★ ЛИСТОВ {len(sheets)}, треков {len(trk)}; узел = сближение ближе {a.near:g}px "
      f"на ≥{a.min_len} строк подряд")

per_sheet, per_track, amb_rows, tot_rows = Counter(), [], 0, 0
node_len, bykind = [], defaultdict(list)
skip = Counter()
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        skip["нет разметки"] += 1
        continue
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    bytrack = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None:
            bytrack[t].append(nm)
    for t, ns in bytrack.items():
        K = KOF.get((sh, t))
        if K is None or len(ns) < 2:
            continue
        common = sorted(set(gts[ns[0]]).intersection(*[set(gts[n]) for n in ns[1:]]))
        if len(common) < 100:
            skip["мало общих строк"] += 1
            continue
        gaps = np.array([min(np.diff(sorted(gts[n][y] for n in ns))) for y in common])
        close = gaps < a.near
        tot_rows += len(common)
        amb_rows += int(close.sum())
        ev, run = 0, None
        for i, v in enumerate(close):
            if v and run is None:
                run = i
            elif not v and run is not None:
                if i - run >= a.min_len:
                    ev += 1
                    node_len.append(i - run)
                run = None
        if run is not None and len(close) - run >= a.min_len:
            ev += 1
            node_len.append(len(close) - run)
        per_sheet[sh] += ev
        per_track.append(ev)
        kb = "2" if K == 2 else ("3-4" if 3 <= K <= 4 else ("5+" if K >= 5 else "1"))
        bykind[kb].append(ev)
    if si % 200 == 0:
        print(f"  {si}/{len(sheets)}  узлов накоплено {sum(per_sheet.values())}")

ps = np.array([per_sheet[s] for s in per_sheet]) if per_sheet else np.array([0])
pt = np.array(per_track) if per_track else np.array([0])
print(f"\n★★ РАЗМЕР ЗАДАЧИ ПО КОРПУСУ (листов с многокривыми треками {len(per_sheet)})")
print(f"| величина | медиана | среднее | 90-й перцентиль | максимум |")
print(f"| УЗЛОВ НА ЛИСТ | **{np.median(ps):.0f}** | {ps.mean():.1f} | {np.percentile(ps,90):.0f} | {ps.max()} |")
print(f"| узлов на трек | {np.median(pt):.0f} | {pt.mean():.1f} | {np.percentile(pt,90):.0f} | {pt.max()} |")
if node_len:
    nl = np.array(node_len)
    print(f"| длина узла, строк | {np.median(nl):.0f} | {nl.mean():.0f} | {np.percentile(nl,90):.0f} | {nl.max()} |")
print(f"\n★ СТРОК В УЗЛАХ: {amb_rows} из {tot_rows} = **{100*amb_rows/max(1,tot_rows):.1f}%** "
      f"⇒ однозначно {100-100*amb_rows/max(1,tot_rows):.1f}% материала")
print(f"\n★ УЗЛОВ НА ТРЕК ПО ПЛОТНОСТИ ПУЧКА")
for k in ("2", "3-4", "5+"):
    if bykind[k]:
        v = np.array(bykind[k])
        print(f"| K = {k} | треков {len(v)} | медиана {np.median(v):.0f} | среднее {v.mean():.1f} | максимум {v.max()} |")
for k, v in skip.items():
    print(f"⚠ пропущено ({k}): {v}")
print("\n⚠ ЭТО НИЖНЯЯ ОЦЕНКА: узлы считаны по ОЦИФРОВАННЫМ кривым, а на скане рядом проходят и "
      "неоцифрованные линии (§6.190) — они тоже создают развилку.")

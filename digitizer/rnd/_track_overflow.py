"""ЗАМЕР B: как часто ЭКСПЕРТНАЯ (GT) кривая ВЫХОДИТ ЗА ГРАНИЦЫ СВОЕГО ТРЕКА.

Мотив (RYBAL_058, 19.07): NGK1 имеет размах x 213..608 при треке 190..501 — треть размаха
вне трека, кривая не детектируется вовсе. Вопрос: это единичная аномалия конкретного листа
или системное свойство архива (и особенно — есть ли то же на K=1 листах, которые СЕЙЧАС
работают на med 1-2px и которые ломать нельзя).

Считаем ПО ТОЧКАМ: доля отсчётов GT вне [x_left, x_right] трека + величина вылета в px
(по перцентилям 2/98, а не min/max — единичный выброс не должен объявлять кривую вылезающей).

ВАЖНО про скорость: frame_from_nlgx(rgb=None) строит треки ИЗ scale_axes — пиксели нужны
только для grid_period и для фолбэка «нет осей вовсе» (такие листы мы и так пропускаем).
Поэтому идёт ПОЛНЫЙ проход по архиву без чтения картинок; картинка проверяется только на
СУЩЕСТВОВАНИЕ (find_image, обращение к файловой системе), как требует постановка.

  python _track_overflow.py [--thr 2.0] [--json out.json] [--per-well 0]
"""
import sys, json, argparse, collections
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from auto import meta as M
from auto.frame import frame_from_nlgx
from auto.emit import _slot_track

MN = r"F:\nds\Auto\mnemonics.json"
ARCHIVE = Path(r"F:\nds\projects\Archive")

ap = argparse.ArgumentParser()
ap.add_argument("--thr", type=float, default=2.0, help="%% точек вне трека, выше — «вылезает»")
ap.add_argument("--per-well", type=int, default=0, help="0 = все листы (полный проход)")
ap.add_argument("--json", default="")
a = ap.parse_args()

sheets = []
for wlg in sorted(ARCHIVE.glob("*/wlg")):
    well = wlg.parent.name
    k = 0
    for n in sorted(wlg.glob("*.nlgx")):
        if "_auto" in n.stem:
            continue
        if a.per_well and k >= a.per_well:
            break
        img = find_image(n)
        if not img:
            continue
        sheets.append((well, n))
        k += 1
print(f"листов с картинкой: {len(sheets)}")

rows, skipped = [], collections.Counter()
for i, (well, n) in enumerate(sheets):
    if i % 200 == 0:
        print(f"  ...{i}/{len(sheets)}", flush=True)
    try:
        model = extract(str(n))
    except Exception as e:
        skipped[f"extract:{type(e).__name__}"] += 1
        continue
    # реальные GT-кривые: >=100 непустых точек, не ось глубин
    gts = [c for c in model["curves"]
           if sum(1 for x in c["xs"] if x != NULL) >= 100 and M.mnem_root(c["name"]) != "DA"]
    if not gts:
        skipped["нет GT-кривых"] += 1
        continue
    if not model.get("scale_axes"):
        skipped["нет scale_axes (трек неизвестен)"] += 1
        continue
    try:
        m = M.parse_filename(n.name, MN)
        frame = frame_from_nlgx(n, meta=m, rgb=None)
    except Exception as e:
        skipped[f"frame:{type(e).__name__}"] += 1
        continue
    if not frame.tracks:
        skipped["нет треков"] += 1
        continue
    K = len(gts)
    for c in gts:
        xs = np.array([x for x in c["xs"] if x != NULL], float)
        try:
            ti = _slot_track(model, c, frame)
            t = frame.tracks[ti]
        except Exception as e:
            skipped[f"slot_track:{type(e).__name__}"] += 1
            continue
        # размах по перцентилям 2/98 — выбросы не должны решать
        p2, p98 = float(np.percentile(xs, 2)), float(np.percentile(xs, 98))
        out_l = (xs < t.x_left).sum()
        out_r = (xs > t.x_right).sum()
        pct_out = 100.0 * (out_l + out_r) / len(xs)
        rows.append({
            "well": well, "file": n.name, "tok": m.curves_token, "K": K,
            "name": c["name"].split()[0], "root": M.mnem_root(c["name"]),
            "npts": int(len(xs)), "ntracks": len(frame.tracks), "track": ti,
            "tl": int(t.x_left), "tr": int(t.x_right), "tw": int(t.width),
            "p2": round(p2, 1), "p98": round(p98, 1),
            "pct_out": round(pct_out, 2),
            "pct_l": round(100.0 * out_l / len(xs), 2),
            "pct_r": round(100.0 * out_r / len(xs), 2),
            # вылет в px по перцентилям (0 если внутри)
            "ovf_l": round(max(0.0, t.x_left - p2), 1),
            "ovf_r": round(max(0.0, p98 - t.x_right), 1),
        })

for k, v in skipped.most_common():
    print(f"  пропущено [{k}]: {v}")

N = len(rows)
bad = [r for r in rows if r["pct_out"] > a.thr]
print(f"\n=== ВСЕГО: {len(set(r['file'] for r in rows))} листов, {N} GT-кривых ===")
print(f"вылезает (>{a.thr}% точек вне трека): {len(bad)} = {100.0*len(bad)/max(1,N):.1f}%")


# ── РАЗДЕЛЕНИЕ ДВУХ РАЗНЫХ ЯВЛЕНИЙ ───────────────────────────────────────────────
# «Кривая целиком мимо трека» (нет пересечения вообще) — это НЕ вылет кривой, а промах
# _slot_track: слот привязался не к своему треку. Смешивать нельзя, лечится разным.
def _disjoint(r):
    return r["p2"] > r["tr"] or r["p98"] < r["tl"]


dis = [r for r in bad if _disjoint(r)]
par = [r for r in bad if not _disjoint(r)]
print(f"\n=== ИЗ НИХ ДВА РАЗНЫХ ЯВЛЕНИЯ ===")
print(f"  ДИЗЪЮНКТ (кривая целиком вне трека => промах _slot_track): {len(dis)} "
      f"({100.0*len(dis)/max(1,N):.1f}% всех кривых)")
print(f"    треков на листе: {collections.Counter(r['ntracks'] for r in dis).most_common()}")
print(f"  ЧАСТИЧНЫЙ вылет (настоящий overflow):                      {len(par)} "
      f"({100.0*len(par)/max(1,N):.1f}% всех кривых)")
print(f"    треков на листе: {collections.Counter(r['ntracks'] for r in par).most_common()}")
print(f"    (ntracks=1 => промах маппинга невозможен, вылет реальный)")
if par:
    v = np.array([max(r["ovf_l"], r["ovf_r"]) for r in par])
    f = np.array([max(r["ovf_l"], r["ovf_r"]) / max(1, r["tw"]) for r in par])
    print(f"    вылет px  med={np.median(v):.0f} p90={np.percentile(v,90):.0f} max={v.max():.0f}")
    print(f"    в долях трека med={np.median(f)*100:.0f}% p90={np.percentile(f,90)*100:.0f}%")
    print(f"    сторона: только влево {sum(1 for r in par if r['ovf_l']>0 and r['ovf_r']==0)}, "
          f"только вправо {sum(1 for r in par if r['ovf_r']>0 and r['ovf_l']==0)}, "
          f"обе {sum(1 for r in par if r['ovf_l']>0 and r['ovf_r']>0)}")
    print(f"    листов хотя бы с одним частичным вылетом: {len(set(r['file'] for r in par))} "
          f"из {len(set(r['file'] for r in rows))}")


def hist(items, edges, key):
    lab = []
    for j in range(len(edges) - 1):
        lo, hi = edges[j], edges[j + 1]
        cnt = sum(1 for r in items if lo <= r[key] < hi)
        lab.append((f"{lo:g}..{hi:g}", cnt))
    lab.append((f">={edges[-1]:g}", sum(1 for r in items if r[key] >= edges[-1])))
    return lab


print("\n--- распределение ДОЛИ точек вне трека (все кривые) ---")
for l, c in hist(rows, [0, 0.5, 2, 5, 10, 20, 40], "pct_out"):
    print(f"  {l:>10}% точек : {c:5d} ({100.0*c/max(1,N):5.1f}%)")

print(f"\n--- ВЕЛИЧИНА вылета в px (только вылезающие, n={len(bad)}) ---")
for side, key in (("влево", "ovf_l"), ("вправо", "ovf_r")):
    vals = [r[key] for r in bad if r[key] > 0]
    print(f"  {side}: кривых {len(vals)}", end="")
    if vals:
        v = np.array(vals)
        print(f"  med={np.median(v):.0f}px  p90={np.percentile(v,90):.0f}px  max={v.max():.0f}px")
    else:
        print()
print("  вылет в ДОЛЯХ ширины трека (max(ovf_l,ovf_r)/tw):")
if bad:
    fr = np.array([max(r["ovf_l"], r["ovf_r"]) / max(1, r["tw"]) for r in bad])
    for lo, hi in ((0, .05), (.05, .15), (.15, .30), (.30, .60), (.60, 99)):
        print(f"    {lo*100:3.0f}..{hi*100:3.0f}% ширины: {int(((fr>=lo)&(fr<hi)).sum()):5d}")
    print(f"    med={np.median(fr)*100:.0f}% ширины трека")

print("\n--- ТОП скважин (по числу вылезающих кривых) ---")
byw = collections.Counter(r["well"] for r in bad)
totw = collections.Counter(r["well"] for r in rows)
for w, c in byw.most_common(15):
    print(f"  {w:<18} {c:4d} из {totw[w]:4d} ({100.0*c/totw[w]:5.1f}%)")

print("\n--- ТОП токенов имени ---")
byt = collections.Counter(r["tok"] for r in bad)
tott = collections.Counter(r["tok"] for r in rows)
for t, c in byt.most_common(15):
    print(f"  {(t or '?'):<18} {c:4d} из {tott[t]:4d} ({100.0*c/tott[t]:5.1f}%)")

print("\n--- ТОП мнемоник (root) ---")
byr = collections.Counter(r["root"] for r in bad)
totr = collections.Counter(r["root"] for r in rows)
for t, c in byr.most_common(15):
    print(f"  {(t or '?'):<10} {c:4d} из {totr[t]:4d} ({100.0*c/totr[t]:5.1f}%)")

print("\n=== РАЗРЕЗ ПО K (число GT-кривых в файле) ===")
print(f"{'K':>3} {'кривых':>7} {'вылезает':>9} {'%':>6}  {'med вылет px':>12}")
for K in sorted(set(r["K"] for r in rows)):
    sub = [r for r in rows if r["K"] == K]
    sb = [r for r in sub if r["pct_out"] > a.thr]
    mv = np.median([max(r["ovf_l"], r["ovf_r"]) for r in sb]) if sb else 0
    tag = "  <- K=1: работают сейчас" if K == 1 else ""
    print(f"{K:>3} {len(sub):>7} {len(sb):>9} {100.0*len(sb)/len(sub):>5.1f}%  {mv:>12.0f}{tag}")

k1 = [r for r in rows if r["K"] == 1]
k1b = [r for r in k1 if r["pct_out"] > a.thr]
k1p = [r for r in k1b if not _disjoint(r)]
print(f"\nK=1 подробно: {len(set(r['file'] for r in k1))} листов, {len(k1)} кривых, "
      f"вылезает {len(k1b)} ({100.0*len(k1b)/max(1,len(k1)):.1f}%), "
      f"из них частичных {len(k1p)} ({100.0*len(k1p)/max(1,len(k1)):.1f}%)")
# сколько на K=1 кривых ТОЙ ЖЕ тяжести, что RYBAL_058/NGK1 (20% точек вне, 34% ширины трека)
sev = [r for r in k1 if r["pct_out"] >= 15
       and max(r["ovf_l"], r["ovf_r"]) / max(1, r["tw"]) >= 0.20]
print(f"  K=1 уровня NGK1 (>=15% точек вне И >=20% ширины трека): {len(sev)} из {len(k1)}")
for r in sorted(sev, key=lambda r: -r["pct_out"]):
    fr = max(r["ovf_l"], r["ovf_r"]) / max(1, r["tw"]) * 100
    print(f"    {r['well']:<14} {r['name']:<7} трек[{r['tl']},{r['tr']}]w{r['tw']} "
          f"кривая[{r['p2']:.0f},{r['p98']:.0f}] вне={r['pct_out']:5.1f}% "
          f"L{r['ovf_l']:.0f}/R{r['ovf_r']:.0f}px = {fr:.0f}% ширины")
if k1b:
    v = np.array([max(r["ovf_l"], r["ovf_r"]) for r in k1b])
    f = np.array([max(r["ovf_l"], r["ovf_r"]) / max(1, r["tw"]) for r in k1b])
    print(f"  вылет med={np.median(v):.0f}px p90={np.percentile(v,90):.0f}px max={v.max():.0f}px; "
          f"в долях трека med={np.median(f)*100:.0f}% max={f.max()*100:.0f}%")
    print("  худшие 10 K=1:")
    for r in sorted(k1b, key=lambda r: -r["pct_out"])[:10]:
        print(f"    {r['well']:<14} {r['name']:<7} трек[{r['tl']},{r['tr']}] "
              f"кривая[{r['p2']:.0f},{r['p98']:.0f}] вне={r['pct_out']:5.1f}% "
              f"(L{r['ovf_l']:.0f}/R{r['ovf_r']:.0f}px)")

print("\n--- худшие 15 по доле точек вне трека (весь архив) ---")
for r in sorted(rows, key=lambda r: -r["pct_out"])[:15]:
    print(f"  {r['well']:<14} K={r['K']} {r['name']:<7} трек[{r['tl']},{r['tr']}] "
          f"кривая[{r['p2']:.0f},{r['p98']:.0f}] вне={r['pct_out']:5.1f}% "
          f"(L{r['ovf_l']:.0f}/R{r['ovf_r']:.0f}px) {r['file'][:44]}")

# контроль: сам RYBAL_058, с которого началось
print("\n--- контроль RYBAL_058 (лист, с которого начался разбор) ---")
for r in rows:
    if "Rybal_058" in r["file"] and "GK_(0012-1395)_NGK" in r["file"]:
        print(f"  {r['name']:<7} трек[{r['tl']},{r['tr']}] кривая[{r['p2']:.0f},{r['p98']:.0f}] "
              f"вне={r['pct_out']:.1f}% (L{r['ovf_l']:.0f}/R{r['ovf_r']:.0f}px)")

if a.json:
    Path(a.json).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print("\njson →", a.json)

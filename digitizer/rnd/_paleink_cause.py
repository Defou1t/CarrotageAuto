r"""_paleink_cause.py — ПРИЧИНА, А НЕ РАЗМЕР: РАСТЁТ ЛИ ЦЕНА БЛЕДНОЙ ТУШИ С ПЛОТНОСТЬЮ ПУЧКА (§6.185).

ОТКУДА ВОПРОС. §6.185 намерил РАЗМЕР направления: на треках K ≥ 3 лежит 104 кривые из 179 — 58%
всего потрекового недобора, и там декодер слабее прода в полтора раза. Но там же прямо сказано,
чего этот замер НЕ говорит: «бледная тушь» — гипотеза §6.133, а не измеренная причина. Проверять
её предлагалось отдельным прогоном на `softfg=`.

★★ ЗДЕСЬ ОНА ПРОВЕРЯЕТСЯ БЕЗ ЕДИНОГО ПРОГОНА. `_row_ceiling.py` уже посчитал по всему корпусу два
потолка, различающихся ТОЛЬКО критерием чернила:
    V2 — ран прод-маски (`trace2d._color_fg`, абсолютная темнота `dark_v`) + ограничение 1:1;
    V3 — то же самое 1:1, но чернило ищется МЯГКО: темнее бумаги СВОЕЙ строки на `--delta`.
Разница V3 − V2 и есть ЦЕНА БЛЕДНОЙ ТУШИ в честных кривых: всё, что прод-передний план теряет, а
мягкий критерий находит. Гипотеза §6.133 предсказывает, что эта цена растёт с K. Если не растёт —
направление названо неверно, и лучше узнать это разбором, чем сутками счёта.

⚠⚠ ЧТО ЭТО ЗА ЧИСЛО И ЧЕМ ОНО НЕ ЯВЛЯЕТСЯ. Это ПОТОЛОК, а не обещание: V3 говорит, сколько кривых
стало бы достижимо, если бы передний план видел бледную тушь ИДЕАЛЬНО. Реальный мягкий передний
план поднимает и лишние пиксели (§6.135: лишние кандидаты — работа отбору), поэтому взятое здесь
число — верхняя граница выигрыша, а не его оценка.

★ ПРЕДПОСЫЛКА, КОТОРАЯ ПРОВЕРЯЕТСЯ, А НЕ ПРЕДПОЛАГАЕТСЯ. Потолки посчитаны по КРИВЫМ, а K живёт
на ТРЕКАХ; связь «кривая → трек» берётся из тех же пуловых дампов (`slots`), по которым считались
и потолки. Стенд печатает, какая доля кривых состыковалась, и отказывается работать, если
состыковалось меньше `--need` (умолчание 0.9): разбор по неполной стыковке — это §6.166 в профиль.

  <ComfyUI>\python_embeded\python.exe _paleink_cause.py
  … --rebuild-map        # пересобрать кэш «лист → {кривая: трек}» (читает пулы, ~2.5 ГБ)
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--rows", default="pick_learn_honest2.pkl.dump")
ap.add_argument("--ceil", default="rowceil_*of6.pkl")
ap.add_argument("--map", default="slotmap.pkl", help="кэш «лист → {кривая: трек}»")
ap.add_argument("--rebuild-map", action="store_true")
ap.add_argument("--need", type=float, default=0.9, help="минимальная доля состыкованных кривых")
ap.add_argument("--perm", type=int, default=200000)
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
a = ap.parse_args()
TS = Path(a.ts)

HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9   # ровно критерий ветки и §6.133

# ───────────────────────────────────────────────────────────────────── потрековая честная пара
trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]): int(r[3]["n_prod"]) for r in trk}
HA = {(r[0], r[2]): int(r[4]) for r in trk}
HB = {(r[0], r[2]): int(r[5]) for r in trk}
sheets = {r[0] for r in trk}
print(f"★ ЧЕСТНАЯ ПАРА: треков {len(trk)} на {len(sheets)} листах; "
      f"прод {sum(HA.values())}, декодер {sum(HB.values())}")

# ───────────────────────────────────────────────────────────────────────────── потолки по кривым
CEIL = sorted(TS.glob(a.ceil))
if not CEIL:
    sys.exit(f"⛔ нет дампов потолков {a.ceil} — сначала `_row_ceiling.py` (драйвер _rowceil_run.ps1)")
rows = []
for p in CEIL:
    rows += pickle.load(open(p, "rb"))["rows"]
print(f"★ ПОТОЛКИ: кривых {len(rows)} из {len(CEIL)} дампов")

# ──────────────────────────────────────────────── связь «кривая → трек» (кэш, пулы читаются раз)
MAP = TS / a.map
if a.rebuild_map or not MAP.is_file():
    stem2f = {}
    for root in a.pools:
        for f in sorted(Path(root).glob("*.pkl")):
            stem2f.setdefault(f.stem, f)
    smap, miss = {}, 0
    for k, sh in enumerate(sorted(sheets), 1):
        f = stem2f.get(Path(sh).stem)
        if not f:
            miss += 1
            continue
        try:
            d = pickle.load(open(f, "rb"))
            smap[sh] = {s["name"]: int(s["track"]) for s in d["slots"]}
        except Exception:
            miss += 1
        if k % 200 == 0:
            print(f"  пулы {k}/{len(sheets)}")
    pickle.dump(smap, open(MAP, "wb"))
    print(f"★ КЭШ СОБРАН: листов {len(smap)}, не нашлось {miss} → {MAP.name}")
smap = pickle.load(open(MAP, "rb"))

# ─────────────────────────────────────────────────────────── стыковка и проверка предпосылки
buck, unmatched, outside = defaultdict(list), 0, 0
for r in rows:
    sh, nm = r[0], r[1]
    # ⚠ ДВА РАЗНЫХ СЛУЧАЯ, И СМЕШИВАТЬ ИХ НЕЛЬЗЯ. Потолки посчитаны по ВСЕМУ корпусу (2671 лист),
    #   а честная пара живёт на 1123: кривая с чужого листа — это не брак стыковки, а другое поле.
    #   Первая редакция считала их вместе и объявила стыковку 39.7% при фактических ~98%.
    if sh not in sheets:
        outside += 1
        continue
    t = smap.get(sh, {}).get(nm)
    if t is None or (sh, t) not in KOF:
        unmatched += 1
        continue
    buck[(sh, t)].append(r)
matched = sum(len(v) for v in buck.values())
share = matched / max(1, matched + unmatched)
print(f"★★ СТЫКОВКА (только по честному полю): кривых с треком {matched}, без {unmatched} ⇒ "
      f"доля {share:.1%}; вне поля {outside} кривых (это не брак); "
      f"треков с кривыми {len(buck)} из {len(trk)}")
if share < a.need:
    sys.exit(f"⛔ состыковалось меньше {a.need:.0%} — разбор по неполной стыковке недействителен")

# ─────────────────────────────────────────────────────────────────────────────────── таблица
BUCKETS = [(1, 1), (2, 2), (3, 4), (5, 10 ** 6)]
print(f"\n★★★ ЦЕНА БЛЕДНОЙ ТУШИ ПО ПЛОТНОСТИ ПУЧКА (K = линий ведёт прод, {matched} кривых)")
print("| K | треков | кривых | честных V2 прод-маска | честных V3 мягкое чернило | +бледная тушь | на трек |")
tot2 = tot3 = 0
per_bucket = []
for lo, hi in BUCKETS:
    ks = [k for k in buck if lo <= KOF[k] <= hi]
    cs = [r for k in ks for r in buck[k]]
    h2 = sum(1 for r in cs if HON(r[9], r[8]))
    h3 = sum(1 for r in cs if HON(r[11], r[10]))
    tot2 += h2
    tot3 += h3
    tag = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+")
    per_bucket.append((tag, len(ks), len(cs), h2, h3))
    print(f"| {tag} | {len(ks)} | {len(cs)} | {h2} | {h3} | **{h3-h2:+d}** | "
          f"{(h3-h2)/max(1,len(ks)):.3f} |")
print(f"| ВСЕГО | {len(buck)} | {matched} | {tot2} | {tot3} | **{tot3-tot2:+d}** | "
      f"{(tot3-tot2)/max(1,len(buck)):.3f} |")

# ★★★ ВТОРОЙ СЧЁТ, БЕЗ КОТОРОГО ПЕРВЫЙ ВВОДИТ В ЗАБЛУЖДЕНИЕ. «Бледная тушь» — не единственное, что
# отделяет кривую от потолка. Между V1 (чернило есть, 1:1 НЕ наложено) и V2 (то же чернило плюс
# запрет двум кривым занять один ран) лежит цена ИДЕНТИЧНОСТИ: не «где линия», а «чья она».
# Ровно эта цена обязана расти с плотностью пучка — и если на K = 3-4 она больше цены чернила,
# то лечить надо не передний план, а раздачу ранов.
print(f"\n★★★ ЧТО ИМЕННО ДЕРЖИТ КАЖДУЮ КОРЗИНУ: ЧЕРНИЛО ИЛИ ИДЕНТИЧНОСТЬ")
print("| K | кривых | V1 чернило без 1:1 | V2 +запрет 1:1 | цена идентичности | цена бледной туши |")
for (lo, hi), (tag, nk, nc, h2, h3) in zip(BUCKETS, per_bucket):
    cs = [r for k in buck if lo <= KOF[k] <= hi for r in buck[k]]
    h1 = sum(1 for r in cs if HON(r[7], r[6]))
    print(f"| {tag} | {len(cs)} | {h1} | {h2} | **{h2-h1:+d}** | **{h3-h2:+d}** |")

# ★★★★ КОЛОНКА, КОТОРАЯ И РЕШАЕТ ВОПРОС. Поднимать потолок имеет смысл ТОЛЬКО там, где в него
# упираются. Насыщение = что декодер берёт СЕЙЧАС, делённое на потолок нынешнего переднего плана.
# Близко к 1 — фронт исчерпан, и мягкое чернило будет ПОЧУВСТВОВАНО. Далеко от 1 — мешает сам
# декодер, и поднятый потолок останется неиспользованным (§6.135: лишние кандидаты ещё и вредят).
print(f"\n★★★★ ГДЕ ПОДНЯТЫЙ ПОТОЛОК ВООБЩЕ БУДЕТ ПОЧУВСТВОВАН")
print("| K | прод | декодер | потолок V2 | насыщение декодера | запас до потолка | даст мягкое чернило |")
for (lo, hi), (tag, nk, nc, h2, h3) in zip(BUCKETS, per_bucket):
    ks = [k for k in buck if lo <= KOF[k] <= hi]
    pa = sum(HA[k] for k in ks)
    pb = sum(HB[k] for k in ks)
    print(f"| {tag} | {pa} | {pb} | {h2} | **{pb/max(1,h2):.0%}** | {h2-pb:+d} | {h3-h2:+d} |")

ge3 = sum(h3 - h2 for tag, nk, nc, h2, h3 in per_bucket if tag in ("3-4", "5+"))
gain = tot3 - tot2
print(f"\n★ ДОЛЯ ПРИБАВКИ НА K ≥ 3: {ge3} из {gain} = {ge3/max(1,gain):.1%} "
      f"(недобора там лежит 58% — §6.185)")

# ── ПРОВЕРКА, А НЕ ВПЕЧАТЛЕНИЕ: растёт ли цена бледной туши с K на уровне ТРЕКА ──────────────
# ⚠ Доля сама по себе ничего не доказывает: на K ≥ 3 и кривых больше. Считаем на ТРЕК и проверяем
#   связь рангов; единица независимости — трек, а не кривая (кривые одного бланка делят дефекты).
kk = np.array([KOF[k] for k in buck], float)
gg = np.array([sum(1 for r in buck[k] if HON(r[11], r[10])) -
               sum(1 for r in buck[k] if HON(r[9], r[8])) for k in buck], float)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))


rho = spearman(kk, gg)
rng = np.random.default_rng(0)
null = np.array([spearman(kk, rng.permutation(gg)) for _ in range(2000)])
p = float((np.abs(null) >= abs(rho) - 1e-12).mean())
print(f"★★ СВЯЗЬ «ПЛОТНЕЕ ПУЧОК → ДОРОЖЕ БЛЕДНАЯ ТУШЬ»: ρ = {rho:+.3f}, p = {p:.4f} "
      f"(по {len(kk)} трекам, 2000 перестановок)")
# ⚠⚠ ОДНОГО ρ МАЛО, И ЧИТАТЬ ЕГО В ОТРЫВЕ ОТ ТАБЛИЦЫ НЕЛЬЗЯ. Здесь стояло «ρ > 0 ⇒ причина названа
# верно» — вывод, которому таблица выше ПРОТИВОРЕЧИТ: связь значима, но держит её корзина 5+, а на
# 3-4 цена бледной туши НИЖЕ, чем у однолинейных треков. Монотонности нет, и ρ её не показывает.
per_track = [(tag, (h3 - h2) / max(1, nk)) for tag, nk, nc, h2, h3 in per_bucket]
mono = all(per_track[i][1] <= per_track[i + 1][1] for i in range(len(per_track) - 1))
print("   на трек по корзинам: " + ", ".join(f"{t} → {v:.3f}" for t, v in per_track)
      + f"  ⇒ рост с K {'ЕСТЬ' if mono else 'НЕ монотонен'}")
print("   ★ РЕШАЕТ НЕ ρ, А НАСЫЩЕНИЕ: поднятый потолок чувствуется там, где декодер в него упёрся.")

# ── контроль: тот же контраст в «центровом» варианте (V1ц против V3ц) ───────────────────────
c2 = sum(1 for k in buck for r in buck[k] if HON(r[14], r[13]))
c3 = sum(1 for k in buck for r in buck[k] if HON(r[16], r[15]))
print(f"\n★ КОНТРОЛЬ (центровой вариант, ошибка в центре рана): V1ц {c2} → V3ц {c3} "
      f"({c3-c2:+d}) — знак обязан совпасть с основным, иначе эффект держит не чернило, а 1:1")

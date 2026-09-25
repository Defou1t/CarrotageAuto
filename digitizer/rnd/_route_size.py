r"""_route_size.py — СКОЛЬКО ДАЁТ ВЫБОР ПУТИ ПО СЛОТУ, И ДОСТИЖИМО ЛИ ЭТО (§6.200 → шаг 2).

ОТКУДА ВОПРОС. `_u1_vs_dec.py` (§6.200) показал, что прод и обученный детектор берут РАЗНЫЕ
кривые: там, где линия U1 неверна, декодер берёт 309 против 94 у прода-965, а покривое объединение
пары даёт 1332 против 1107 у нынешнего прода. Значит рычаг — не «перестроить U1», а смешать два
готовых источника мельче, чем сейчас. Гранулярности три, и две уже в проде:

  * ПО ЛИСТУ — предгейт `rowdec_pregate` (§6.184);
  * ПО ТРЕКУ — `rowdec_pick` (§6.157); оракул §6.175 = 1257;
  * ★ ПО СЛОТУ — НЕ ПРОБОВАЛОСЬ. Считается здесь.

★★ ОСУЩЕСТВИМОСТЬ ПРОВЕРЕНА ПО КОДУ ДО ЗАМЕРА: `emit.py:371-375` уже сливает выдачу ПОСЛОТНО, и
источник навязывает лишь строка `src = m_alt if use.get(ti) else mapping` — `use` ключуется треком.
Оба пути на гейтованном листе к этому моменту посчитаны (`trace2d.py:167-184`, `.alt`), то есть
переход на выбор по слоту НЕ СТОИТ СЧЁТА.

⛔ ПОЧЕМУ НЕЛЬЗЯ ВЗЯТЬ ЧИСЛО ИЗ `_u1_vs_dec.py`. Та колонка кладёт в 1:1 кандидатов обоих путей
сразу, то есть разрешает двум кривым ОДНОГО слота уйти к двум разным эталонным кривым. Эмиттер так
не может. ⇒ Потолок считается ПЕРЕБОРОМ вектора выбора по слотам трека (2^S), и он ниже.

⛔⛔ И ПОЧЕМУ ПАРА — ИМЕННО (A, B), А НЕ (НЫНЕШНИЙ ПРОД, B). На треке, где выбор пути УЖЕ
переключился на декодер, нынешняя выдача равна декодеровой, и вариант прода из пары исчезает.
Первая редакция стенда мерила пару (G, B) и получила потолок 1220; правильная пара — ЧИСТЫЙ прод
`ab_wellmap/A` и ЧИСТЫЙ декодер `ab_rdhonest/B`, а `ab_pregate/G` служит ОПОРОЙ (1107), которую
надо обогнать. Тот же дефект виден в `_u1_vs_dec.py`: `CUR∪DEC` = 1246 ниже `P965∪DEC` = 1332.

ЧТО СЧИТАЕТ (по замороженным выдачам, счёта не тратит):
  опора G                — обязана дать 1107 (§6.188);
  ★ ОРАКУЛ ПО ТРЕКУ (A,B) — обязан дать 1257 (§6.175). ЭТО ПРОВЕРКА МАШИНЕРИИ, а не результат;
  ★ ОРАКУЛ ПО СЛОТУ (A,B) — потолок направления, максимум по 2^S;
  ★ то же, но декодер только на ГЕЙТОВАННЫХ листах (`color_top_frac ≥ 0.80`, `trace2d.py:146-154`,
    признак пересчитывается из `<лист>_understanding.json` той же формулой) — потолок БЕЗ доплаты
    за счёт: на отсеянном листе второй путь и сегодня не считается;
  три правила БЕЗ ЭТАЛОНА — согласие / размах / всегда декодер. Правило ниже опоры значит «оракул
  этим признаком недостижим» (§6.199: «разумный» отбор проиграл «не отбирать вообще»).

★ Обе стороны считаются ОДНОЙ функцией `match` из `_name_cost_prod.py` (§6.198).

  <ComfyUI>\python_embeded\python.exe _route_size.py
"""
import sys, argparse, pickle, hashlib, itertools, json
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
ap.add_argument("--a", default="ab_wellmap/A", help="ЧИСТЫЙ прод — кандидат 1")
ap.add_argument("--b", default="ab_rdhonest/B", help="ЧИСТЫЙ декодер — кандидат 2")
ap.add_argument("--g", default="ab_pregate/G", help="нынешний прод — ОПОРА")
ap.add_argument("--pregate", type=float, default=0.80)
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--maxs", type=int, default=12)
ap.add_argument("--cap", type=int, default=0)
a = ap.parse_args()
TS = Path(a.ts)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def st(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    return float(np.median([abs(tr[y] - gt[y]) for y in com])), len(com) / max(1, len(gt))


def match(rows, cols, ok):
    """Безымянный счёт: максимальное 1:1 внутри трека (§6.143), механика из `_name_cost_prod.py`."""
    pair = {}

    def try_(r, seen):
        for c in cols:
            if not ok.get((r, c)) or c in seen:
                continue
            seen.add(c)
            if c not in pair or try_(pair[c], seen):
                pair[c] = r
                return True
        return False
    return sum(1 for r in rows if try_(r, set()))


def _dir(root, sh):
    stem = Path(sh).stem
    return TS / root / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"


def read_out(root, sh):
    pd = _dir(root, sh)
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    if not got:
        return {}
    return {c["name"]: dense(c) for c in extract(str(got))["curves"]
            if M.mnem_root(c["name"]) != "DA"}


def gate_open(sh):
    """Тот же признак, что `trace2d._pregate_ok`: доля самого частого цвета среди ВСЕХ линий листа,
    без фильтров (там это оговорено отдельно — фильтр менял бы измеряемую величину)."""
    pd = _dir(a.g, sh)
    f = next(iter(sorted(pd.glob("*_understanding.json"))), None) if pd.is_dir() else None
    if not f:
        return True
    L = json.loads(f.read_text(encoding="utf-8")).get("lines") or []
    if not L:
        return True
    c = Counter(str(x.get("color")) for x in L)
    return max(c.values()) / len(L) >= a.pregate


trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]): int(r[3]["n_prod"]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q

RULES = ["★ опора G (нынешний прод)", "чистый прод A", "чистый декодер B",
         "правило: согласие → A, иначе B", "правило: согласие → B, иначе A", "правило: размах",
         "оракул ПО ТРЕКУ (A,B) — сверка 1257", "★ ОРАКУЛ ПО СЛОТУ (A,B)",
         "★ то же, декодер только на гейтованных"]
tot = Counter()
big = ngate = 0
sheets = sorted({r[0] for r in trk})
if a.cap:
    sheets = sheets[:a.cap]
done, miss = [], Counter()
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        miss["нет исходника"] += 1
        continue
    done.append(sh)
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    bytr = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None:
            bytr[t].append(nm)
    WA, WB, WG = read_out(a.a, sh), read_out(a.b, sh), read_out(a.g, sh)
    og = gate_open(sh)
    ngate += og
    for t, ns in bytr.items():
        if (sh, t) not in KOF:
            continue
        S = sorted({k for k in WA if tm.get(k) == t} | {k for k in WB if tm.get(k) == t})
        SG = [k for k in WG if tm.get(k) == t]
        okg = {(g, k): HON(*st(WG[k], gts[g])) for g in ns for k in SG}
        tot["★ опора G (нынешний прод)"] += match(ns, SG, okg)
        if not S:
            continue
        ok, have = {}, set()
        for s in S:
            for tag, W in (("A", WA), ("B", WB)):
                if s in W:
                    have.add((s, tag))
                    for g in ns:
                        ok[(g, (s, tag))] = HON(*st(W[s], gts[g]))

        def score(pick):
            """pick: слот → 'A'/'B'. В слот уходит РОВНО ОДНА кривая — в этом отличие от
            объединения кандидатов в `_u1_vs_dec.py`, которое потолок завышает."""
            return match(ns, [(s, pick[s]) for s in S if (s, pick[s]) in have], ok)

        allA = {s: "A" for s in S}
        allB = {s: "B" for s in S}
        vA, vB = score(allA), score(allB)
        tot["чистый прод A"] += vA
        tot["чистый декодер B"] += vB
        tot["оракул ПО ТРЕКУ (A,B) — сверка 1257"] += max(vA, vB)
        # ⚠ ЗНАК ПРАВИЛА — ВЫБОР АВТОРА, поэтому печатаются ОБЕ ориентации (ревизия §6.201):
        #   первая редакция показывала только «согласие → A» (−196), зеркало даёт −123.
        r1, r1m, r2 = {}, {}, {}
        for s in S:
            c, d = WA.get(s), WB.get(s)
            if c is None:
                r1[s] = r1m[s] = r2[s] = "B"
                continue
            if d is None:
                r1[s] = r1m[s] = r2[s] = "A"
                continue
            m, _ = st(c, d)
            agree = m is not None and m <= 3.0
            r1[s] = "A" if agree else "B"
            r1m[s] = "B" if agree else "A"
            r2[s] = "A" if len(c) >= len(d) else "B"
        tot["правило: согласие → A, иначе B"] += score(r1)
        tot["правило: согласие → B, иначе A"] += score(r1m)
        tot["правило: размах"] += score(r2)
        # ★ Ревизия §6.201 заметила: тот же максимум даёт ОДНО паросочетание по OR-графу (колонка =
        #   слот, ребро «честна у A или у B») — перебор 2^S и жадность излишни. Оставлен перебор как
        #   первичный, OR-граф — как сверка: расхождение обязано быть нулём.
        okor = {(g, s): (ok.get((g, (s, "A"))) or ok.get((g, (s, "B")))) for g in ns for s in S}
        best_or = match(ns, S, okor)
        if len(S) <= a.maxs:
            best = max(score(dict(zip(S, v))) for v in itertools.product("AB", repeat=len(S)))
        else:
            big += 1
            cur, best = dict(allA), vA
            for s in S:
                cur[s] = "B"
                v = score(cur)
                if v >= best:
                    best = v
                else:
                    cur[s] = "A"
        tot["★ ОРАКУЛ ПО СЛОТУ (A,B)"] += best
        tot["оракул по OR-графу (сверка)"] += best_or
        tot["расхождений перебор/OR-граф"] += int(best != best_or)
        tot["★ то же, декодер только на гейтованных"] += best if og else vA
        tot["слотов"] += len(S); tot["треков"] += 1; tot["кривых"] += len(ns)
    if si % 200 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

_sk = sum(miss.values())
print(f"\nСВЕРКА СПИСКА: обработано {len(done)} + пропущено {_sk} = {len(done)+_sk} "
      f"против длины списка {len(sheets)}   "
      f"{'★ СОШЛОСЬ' if len(done)+_sk == len(sheets) else '⛔ НЕ СОШЛОСЬ'}")
print(f"треков {tot['треков']}, слотов {tot['слотов']}, эталонных кривых {tot['кривых']}, "
      f"гейт открыт на {ngate} листах из {len(done)} ({100*ngate/max(1,len(done)):.1f}%)"
      + (f", жадно (S > {a.maxs}) {big}" if big else ""))

print("\n★★★ ТРИ СВЕРКИ С ИЗВЕСТНЫМИ ЧИСЛАМИ (без них таблица недействительна)")
bad = 0
for key, exp, src in (("★ опора G (нынешний прод)", 1107, "§6.188"),
                      ("чистый прод A", 965, "§6.196"),
                      ("чистый декодер B", 934, "§6.175"),
                      ("оракул ПО ТРЕКУ (A,B) — сверка 1257", 1257, "§6.175")):
    g = tot[key]
    okk = abs(g - exp) <= 2
    bad += not okk
    print(f"  {key}: {g}, ожидалось {exp} ({src}) — {'OK' if okk else '⛔ РАСХОЖДЕНИЕ'}")
if bad:
    print("⛔⛔ СВЕРКА НЕ СОШЛАСЬ — числа ниже цитировать нельзя.")

base = tot["★ опора G (нынешний прод)"]
print(f"  оракул по OR-графу: {tot['оракул по OR-графу (сверка)']} против перебора "
      f"{tot['★ ОРАКУЛ ПО СЛОТУ (A,B)']}, треков с расхождением {tot['расхождений перебор/OR-граф']} — "
      f"{'OK' if tot['расхождений перебор/OR-граф'] == 0 else '⛔ ЖАДНОСТЬ ЗАНИЗИЛА'}")
print("\n★★ ВЫБОР ПУТИ ПО СЛОТУ (безымянный 1:1 внутри трека)")
print("| правило | честных | к опоре 1107 |")
print("|---|---|---|")
for r in RULES:
    print(f"| {r} | {tot[r]} | {tot[r]-base:+d} |")

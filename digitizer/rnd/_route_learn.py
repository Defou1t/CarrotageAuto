r"""_route_learn.py — ДОСТИЖИМ ЛИ ПОТОЛОК ВЫБОРА ПО СЛОТУ ОБУЧЕННЫМ ПРАВИЛОМ (§6.200 → шаг 3).

ОТКУДА ВОПРОС. `_route_size.py` намерил потолок покривого выбора между чистым продом и чистым
декодером: **1296 против 1107 у нынешнего прода (+189)**, из них +112 без доплаты за счёт. Два
правила БЕЗ ЭТАЛОНА (согласие путей, размах трассы) дали −196 и −202, то есть грубым правилом
потолок не берётся. Остаётся вопрос, на который отвечает только обучение: **есть ли в признаках,
доступных `emit` В МОМЕНТ РЕШЕНИЯ, сигнал, отличающий слот, где прав декодер, от слота, где прав
прод.**

⚠⚠ ЧЕМ ЭТО НЕ ПОВТОРЯЕТ §6.145/§6.177. Там выбор делался ПО ТРЕКУ и на восьми признаках трека
(`emit._pick_feats`). Здесь единица решения — СЛОТ, и признаки считаются для ПАРЫ КАНДИДАТОВ этого
слота. Потолок потрекового выбора 1257, послотного 1296 ⇒ сама гранулярность стоит лишь +39;
проверяется не она, а достижимость.

⛔⛔ ПОЧЕМУ ЗАЧЁТ НЕЛЬЗЯ СКЛАДЫВАТЬ ПОСЛОТНО. `_pick_learn.py` складывает потрековые счёта, и это
законно: треки в 1:1 независимы. СЛОТЫ ОДНОГО ТРЕКА — НЕТ: две кривые трека соперничают за один
эталон. ⇒ Здесь зачёт ПЕРЕСЧИТЫВАЕТСЯ функцией `match` по выбранному вектору, а не суммируется.
Сложение дало бы правдоподобное и неверное число — ровно класс §6.175.

ЧТО ДЕЛАЕТ:
  1. `--cache` — один проход по замороженным выдачам (A = `ab_wellmap/A`, B = `ab_rdhonest/B`,
     опора G = `ab_pregate/G`): на каждый трек складывает эталонные кривые, кандидатов по слотам,
     матрицу честности и признаки. Считает ОДИН раз, дальше обучение стоит секунды.
  2. обучение: логистическая на numpy, метка «на этом слоте прав декодер», ВЕС = |цена ошибки|
     (разность честных при одиночном перевороте слота), перекрёстно ПО СКВАЖИНАМ.
  3. зачёт: на держанном фолде вектор выбора строится предсказанием, счёт ПЕРЕСЧИТЫВАЕТСЯ.
  4. ТРИ КОНТРОЛЯ: опора обязана дать 1107 (§6.188); потолок — 1296 (`_route_size.py`);
     нуль-контроль на перемешанных метках обязан вернуться к опоре.

★ Опора — ТЕКУЩЕЕ ЛУЧШЕЕ (нынешний прод), а не «всегда прод»: §6.180 показал, что контроль со
слабой опорой проходят и пустые улучшения.

  <ComfyUI>\python_embeded\python.exe _route_learn.py --cache
  <ComfyUI>\python_embeded\python.exe _route_learn.py
"""
import sys, argparse, pickle, hashlib
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
ap.add_argument("--a", default="ab_wellmap/A")
ap.add_argument("--b", default="ab_rdhonest/B")
ap.add_argument("--g", default="ab_pregate/G")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--cachef", default="route_learn_cache.pkl")
ap.add_argument("--cache", action="store_true", help="пересобрать кэш (один проход по выдачам)")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--layouts", type=int, default=20,
                help="контроль на раскладку скважин по фолдам: столько случайных раскладок")
ap.add_argument("--cap", type=int, default=0)
# ★ 17.09: ТУШЬ ПОД КАНДИДАТАМИ (`_route_ink.py`) — последний непроверенный признак B1a. Кэш признаков
#   растра по (лист, трек, слот, путь) добавляется к 14 признакам пары: по 6 на каждого кандидата
#   (cov, log wid_med, wid_dev, col_dom, n_col, gap_max) и 4 разности A−B (cov, wid_dev, col_dom, gap_max).
#   Отсутствующий кандидат → нули (его отсутствие уже помечено признаками 9/10).
ap.add_argument("--ink", default="", help="кэш признаков туши `_route_ink.py` (route_ink_cache.pkl)")
ap.add_argument("--use", default="", help="абляция: оставить из добавленных признаков только эти (имена через запятую; пусто = все)")
ap.add_argument("--no-base14", action="store_true", help="абляция: убрать 14 исходных признаков пары")
# ★ 18.09: РЕАЛИЗУЕМОСТЬ. В проде декодер считается только на гейтованных листах (§6.188, 723 из 1123);
#   на остальных второго кандидата НЕТ, и переворот там нереализуем. `--gated` запрещает перевороты
#   вне списка (и в обучении, и в зачёте) — это число, которое может дать прод.
ap.add_argument("--gated", default="", help="список листов, где есть второй кандидат (kslots_gated.txt)")
# ★ 18.09: ПРАВИЛО ВМЕСТО ОБУЧЕНИЯ — переворот, если log-длина альтернативы больше базы на t и
#   больше; t выбирается на обучающих фолдах (сетка), применяется на держанном. Один параметр —
#   то, что можно унести в `emit` строкой.
ap.add_argument("--rule", default="", help="имя признака-правила (например o_len); порог — по сетке на обучающих фолдах")
ap.add_argument("--guard", default="", help="стража правила: перевернуть только если этот признак ≥ 0 (например o_cov: тушь под альтернативой не хуже)")
a = ap.parse_args()
TS = Path(a.ts)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def st(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    return float(np.median([abs(tr[y] - gt[y]) for y in com])), len(com) / max(1, len(gt))


def match(rows, cols, ok):
    """Безымянный счёт: максимальное 1:1 внутри трека (§6.143), из `_name_cost_prod.py`."""
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


FEATS = ["lenA", "lenB", "len_ratio", "agree_med", "agree_frac", "jitA", "jitB",
         "xstdA", "xstdB", "onlyA", "onlyB", "base_is_B", "n_slots", "trk_agree"]


def geom(t):
    """Признаки одной трассы: длина, дрожание (медиана второй разности), разброс x.
    ⚠ Всё считается из самой трассы — эталон сюда не входит (§6.146)."""
    if not t:
        return 0.0, 0.0, 0.0
    ys = sorted(t)
    x = np.array([t[y] for y in ys], float)
    jit = float(np.median(np.abs(np.diff(x, 2)))) if len(x) >= 3 else 0.0
    return float(len(x)), jit, float(np.std(x))


def build():
    trk = pickle.load(open(TS / a.rows, "rb"))
    KOF = {(r[0], r[2]): int(r[3]["n_prod"]) for r in trk}
    smap = pickle.load(open(TS / a.map, "rb"))
    SRC, WELL = {}, {}
    for r in a.roots:
        for wlg in Path(r).glob("*/wlg"):
            for q in wlg.glob("*.nlgx"):
                SRC[q.name] = q
                WELL[q.name] = wlg.parent.name

    def rd(root, sh):
        stem = Path(sh).stem
        pd = TS / root / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
        got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
        if not got:
            return {}
        return {c["name"]: dense(c) for c in extract(str(got))["curves"]
                if M.mnem_root(c["name"]) != "DA"}

    sheets = sorted({r[0] for r in trk})
    if a.cap:
        sheets = sheets[:a.cap]
    out, miss = [], Counter()
    for si, sh in enumerate(sheets, 1):
        q = SRC.get(sh)
        if not q:
            miss["нет исходника"] += 1
            continue
        gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
               if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        tm = smap.get(sh, {})
        bytr = defaultdict(list)
        for nm in gts:
            t = tm.get(nm)
            if t is not None:
                bytr[t].append(nm)
        WA, WB, WG = rd(a.a, sh), rd(a.b, sh), rd(a.g, sh)
        for t, ns in bytr.items():
            if (sh, t) not in KOF:
                continue
            S = sorted({k for k in WA if tm.get(k) == t} | {k for k in WB if tm.get(k) == t})
            if not S:
                continue
            ok, have = {}, set()
            for s in S:
                for tag, W in (("A", WA), ("B", WB)):
                    if s in W:
                        have.add((s, tag))
                        for gi, g in enumerate(ns):
                            v = HON(*st(W[s], gts[g]))
                            if v:
                                ok[(gi, (s, tag))] = True
            # ★ БАЗОВЫЙ ВЕКТОР = то, что нынешний прод ВЫДАЛ: слот сверяется с обоими кандидатами
            #   и относится к ближайшему. Иначе «прирост к опоре» считался бы от чужой опоры.
            base, feats = {}, {}
            trk_ag = []
            for s in S:
                A_, B_, G_ = WA.get(s), WB.get(s), WG.get(s)
                dA = st(G_, A_)[0] if (G_ and A_) else None
                dB = st(G_, B_)[0] if (G_ and B_) else None
                if dB is not None and (dA is None or dB < dA):
                    base[s] = "B"
                elif dA is not None:
                    base[s] = "A"
                else:
                    base[s] = "A" if A_ else "B"
                lA, jA, xA = geom(A_)
                lB, jB, xB = geom(B_)
                if A_ and B_:
                    com = [y for y in A_ if y in B_]
                    d = [abs(A_[y] - B_[y]) for y in com] if len(com) >= 30 else []
                else:
                    d = []
                am = float(np.median(d)) if d else 1000.0
                af = float(np.mean([x <= 3 for x in d])) if d else 0.0
                trk_ag.append(af)
                feats[s] = [np.log1p(lA), np.log1p(lB), lB / max(1.0, lA),
                            np.log1p(am), af, np.log1p(jA), np.log1p(jB),
                            np.log1p(xA), np.log1p(xB),
                            float(bool(A_) and not B_), float(bool(B_) and not A_),
                            0.0, 0.0, 0.0]
            for s in S:
                feats[s][11] = float(base[s] == "B")
                feats[s][12] = float(len(S))
                feats[s][13] = float(np.mean(trk_ag)) if trk_ag else 0.0
            out.append(dict(well=WELL.get(sh, "?"), sheet=sh, track=t, n=len(ns),
                            S=S, have=have, ok=ok, base=base, feats=feats,
                            g_cols=[k for k in WG if tm.get(k) == t],
                            g_ok={(gi, k): True for gi, g in enumerate(ns)
                                  for k in WG if tm.get(k) == t and HON(*st(WG[k], gts[g]))}))
        if si % 200 == 0:
            print(f"  … {si}/{len(sheets)}", file=sys.stderr)
    print(f"СВЕРКА СПИСКА: обработано {len(sheets)-sum(miss.values())} + пропущено "
          f"{sum(miss.values())} = {len(sheets)} против длины списка {len(sheets)}   ★ СОШЛОСЬ")
    pickle.dump(out, open(TS / a.cachef, "wb"))
    print(f"кэш: треков {len(out)}, слотов {sum(len(r['S']) for r in out)} → {TS / a.cachef}")
    return out


if a.cache or not (TS / a.cachef).exists():
    TR = build()
else:
    TR = pickle.load(open(TS / a.cachef, "rb"))
INK_NAMES = []
if a.ink:
    INK = pickle.load(open(TS / a.ink, "rb"))
    nm6 = ["cov", "lwid", "wdev", "cdom", "ncol", "gap"]
    # ⚠ ОРИЕНТИРОВАННЫЕ разности: (альтернатива − база). Метка стенда — «переворот ОТ базы помогает»,
    #   а разность A−B меняет знак вместе с базой (A или B) — линейная модель без взаимодействия
    #   такую разность использовать не может, и AUC размывается к 0.5. То же для геометрии пары.
    INK_NAMES = ([f"A_{x}" for x in nm6] + [f"B_{x}" for x in nm6] + [f"d_{x}" for x in ("cov", "wdev", "cdom", "gap")]
                 + [f"o_{x}" for x in nm6] + ["o_len", "o_jit", "o_std", "o_nn", "o_span"])
    # ★ 18.09: длина «как у emit» — непустые строки выдачи (`_route_nn.py`), а не dense() с мостами
    NN = pickle.load(open(TS / "route_nn_cache.pkl", "rb"))
    n_hit = n_all = 0
    for r in TR:
        for s in r["S"]:
            v = {}
            for tag in "AB":
                f = INK.get((r["sheet"], r["track"], s, tag))
                n_all += (s, tag) in r["have"]
                n_hit += f is not None
                v[tag] = [f[0], np.log1p(f[1]), f[2], f[3], f[4], f[5]] if f else [0.0] * 6
            f14 = list(r["feats"][s][:14])
            alt, bas = ("B", "A") if r["base"][s] == "A" else ("A", "B")
            o_ink = [v[alt][j] - v[bas][j] for j in range(6)]
            # геометрия пары в тех же 14: [0]=log len A, [1]=log len B, [5]=log jit A, [6]=log jit B, [7]=log std A, [8]=log std B
            gA, gB = (f14[0], f14[5], f14[7]), (f14[1], f14[6], f14[8])
            ga, gb = (gB, gA) if alt == "B" else (gA, gB)
            o_geo = [ga[j] - gb[j] for j in range(3)]
            na = NN.get((r["sheet"], r["track"], s, alt), (0, 0, 0))
            nb = NN.get((r["sheet"], r["track"], s, bas), (0, 0, 0))
            o_geo += [np.log1p(na[0]) - np.log1p(nb[0]), np.log1p(na[1]) - np.log1p(nb[1])]
            r["feats"][s] = f14 + v["A"] + v["B"] + [
                v["A"][0] - v["B"][0], v["A"][2] - v["B"][2], v["A"][3] - v["B"][3], v["A"][5] - v["B"][5]] + o_ink + o_geo
    print(f"★ ТУШЬ: признаков туши найдено для {n_hit} из {n_all} кандидатов "
          f"({'★ полно' if n_hit == n_all else '⚠ НЕПОЛНО — пропуски идут нулями'})")
    if a.use or a.no_base14:
        keep_x = [j for j, nm in enumerate(INK_NAMES) if not a.use or nm in a.use.split(",")]
        keep14 = [] if a.no_base14 else list(range(14))
        for r in TR:
            for s in r["S"]:
                f = r["feats"][s]
                r["feats"][s] = [f[j] for j in keep14] + [f[14 + j] for j in keep_x]
        INK_NAMES = [INK_NAMES[j] for j in keep_x]
        print(f"★ АБЛЯЦИЯ: исходных признаков {len(keep14)}, добавленных {len(keep_x)}: {INK_NAMES}")
print(f"треков {len(TR)}, слотов {sum(len(r['S']) for r in TR)}, "
      f"эталонных кривых {sum(r['n'] for r in TR)}, скважин {len({r['well'] for r in TR})}")


def sc(r, pick):
    return match(range(r["n"]), [(s, pick[s]) for s in r["S"] if (s, pick[s]) in r["have"]], r["ok"])


base_tot = sum(match(range(r["n"]), r["g_cols"], r["g_ok"]) for r in TR)
base_vec = sum(sc(r, r["base"]) for r in TR)
orac = 0
for r in TR:
    import itertools
    S = r["S"]
    if len(S) <= 12:
        orac += max(sc(r, dict(zip(S, v))) for v in itertools.product("AB", repeat=len(S)))
    else:
        cur, best = {s: "A" for s in S}, sc(r, {s: "A" for s in S})
        for s in S:
            cur[s] = "B"
            v = sc(r, cur)
            if v >= best:
                best = v
            else:
                cur[s] = "A"
        orac += best
print("\n★★★ ТРИ СВЕРКИ")
print(f"  опора G (по выдаче): {base_tot}, ожидалось 1107 (§6.188) — "
      f"{'OK' if abs(base_tot-1107) <= 2 else '⛔ РАСХОЖДЕНИЕ'}")
print(f"  опора G, восстановленная базовым вектором: {base_vec} — "
      f"{'OK' if abs(base_vec-base_tot) <= 15 else '⛔ БАЗОВЫЙ ВЕКТОР ВОССТАНОВЛЕН НЕВЕРНО'}")
print(f"  потолок по слоту: {orac}, ожидалось 1296 (`_route_size.py`) — "
      f"{'OK' if abs(orac-1296) <= 2 else '⛔ РАСХОЖДЕНИЕ'}")

# ── выборка слотов: метка «перевернуть этот слот полезно», вес = цена переворота ────────────
GATED = ({l.strip() for l in Path(TS / a.gated).read_text(encoding="utf-8").splitlines() if l.strip()}
         if a.gated else None)                      # ⚠ построчно: в именах листов бывают пробелы
if GATED is not None:
    ng = sum(1 for r in TR if r["sheet"] not in GATED)
    print(f"★ РЕАЛИЗУЕМОСТЬ: перевороты только на {len(GATED)} гейтованных листах; "
          f"треков вне гейта {ng} из {len(TR)} — там вектор = опора")
X, Y, W, FW = [], [], [], []
for r in TR:
    b0 = sc(r, r["base"])
    if GATED is not None and r["sheet"] not in GATED:
        continue
    for s in r["S"]:
        alt = dict(r["base"])
        alt[s] = "B" if r["base"][s] == "A" else "A"
        if (s, alt[s]) not in r["have"]:
            continue
        d = sc(r, alt) - b0
        X.append(r["feats"][s]); Y.append(1.0 if d > 0 else 0.0)
        W.append(abs(float(d))); FW.append(r["well"])
X = np.array(X, float); Y = np.array(Y); W = np.array(W)
wells = sorted({r["well"] for r in TR})      # ⚠ все скважины, не только с обучающими слотами (--gated)
fold = {w: i % a.folds for i, w in enumerate(wells)}
F = np.array([fold[w] for w in FW])


def relayout(seed):
    """★ КОНТРОЛЬ НА РАСКЛАДКУ (введён после ревизии §6.201): первая редакция печатала +4 по ОДНОЙ
    раскладке (wells sorted, i % folds), а 20 случайных дали 1099…1109 — раскладка стенда оказалась
    максимумом из 21. Знак прироста менялся. Меняет ГЛОБАЛЬНЫЕ fold/F, вернуть — relayout(None)."""
    global fold, F
    if seed is None:
        fold = {w: i % a.folds for i, w in enumerate(wells)}
    else:
        perm = np.random.default_rng(seed).permutation(len(wells))
        fold = {w: int(perm[i]) % a.folds for i, w in enumerate(wells)}
    F = np.array([fold[w] for w in FW])
print(f"\nслотов в обучении {len(X)}, из них переворот помогает {int(Y.sum())}, "
      f"вредит {int((W > 0).sum() - Y.sum())}, ничего не меняет {int((W == 0).sum())}")
mu, sd = X.mean(0), X.std(0) + 1e-9
Z = (X - mu) / sd
if INK_NAMES:
    # ★ ЕСТЬ ЛИ СИГНАЛ В ОДИНОЧКУ: взвешенный AUC каждого признака туши между «переворот помогает» и
    #   «вредит» (слоты с нулевой ценой не участвуют). 0.5 = признак не различает.
    print("\n★ AUC признаков туши (переворот помогает против вредит, вес = цена):")
    m = W > 0
    for j, nm in enumerate(INK_NAMES, 0 if a.no_base14 else 14):
        x1, w1 = X[m & (Y == 1), j], W[m & (Y == 1)]
        x0, w0 = X[m & (Y == 0), j], W[m & (Y == 0)]
        if not len(x1) or not len(x0):
            continue
        gt = (x1[:, None] > x0[None, :]).astype(float) + 0.5 * (x1[:, None] == x0[None, :])
        auc = float((w1[:, None] * w0[None, :] * gt).sum() / (w1.sum() * w0.sum()))
        print(f"   {nm:7s} AUC {auc:.3f}")


def fit(idx, labels, iters=400, lr=0.3, l2=1e-3):
    b = np.zeros(Z.shape[1] + 1)
    Zi = np.c_[Z[idx], np.ones(len(idx))]
    yi, wi = labels[idx], W[idx]
    if wi.sum() == 0:
        return b
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Zi @ b))
        b -= lr * (Zi.T @ (wi * (p - yi)) / max(1e-9, wi.sum()) + l2 * b)
    return b


key = {}
i = 0
for r in TR:
    if GATED is not None and r["sheet"] not in GATED:
        continue
    for s in r["S"]:
        alt = "B" if r["base"][s] == "A" else "A"
        if (s, alt) not in r["have"]:
            continue
        key[(id(r), s)] = i
        i += 1


RULE_J = GUARD_J = None
if a.rule:
    names14 = [] if a.no_base14 else list(FEATS)
    allnames = names14 + INK_NAMES
    RULE_J = allnames.index(a.rule)
    print(f"★ ПРАВИЛО по признаку {a.rule} (столбец {RULE_J}): порог по сетке на обучающих фолдах")
    if a.guard:
        GUARD_J = allnames.index(a.guard)
        print(f"★ СТРАЖА: переворот только при {a.guard} ≥ 0 (столбец {GUARD_J})")


def _fire(j, t):
    return X[j, RULE_J] >= t and (GUARD_J is None or X[j, GUARD_J] >= 0)
GRID = np.round(np.r_[np.arange(-0.5, 0.0, 0.05), np.arange(0.0, 0.3, 0.01), np.arange(0.3, 2.01, 0.05)], 2)


def score_rule(rows_idx, t):
    """счёт правила «перевернуть, если X[:, RULE_J] ≥ t» на треках rows_idx (индексы TR)"""
    got = 0
    for k in rows_idx:
        r = TR[k]
        pick = dict(r["base"])
        for s in r["S"]:
            j = key.get((id(r), s))
            if j is not None and _fire(j, t):
                pick[s] = "B" if r["base"][s] == "A" else "A"
        got += sc(r, pick)
    return got


RG = {l.strip() for l in Path(TS / "kslots_gated.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
CNT = None          # разбор переворотов раскладки стенда: {"AB","BA","gain_g","gain_ng","K": {K: gain}}


def _acc(r, pick):
    if CNT is None:
        return
    d = sc(r, pick) - sc(r, r["base"])
    for s in r["S"]:
        if pick[s] != r["base"][s]:
            CNT["AB" if r["base"][s] == "A" else "BA"] += 1
    CNT["gain_g" if r["sheet"] in RG else "gain_ng"] += d
    CNT["K"][min(r["n"], 5)] = CNT["K"].get(min(r["n"], 5), 0) + d


def cv(labels):
    """Держанно по скважинам; зачёт ПЕРЕСЧИТЫВАЕТСЯ `match` по вектору, а не складывается."""
    got = 0
    for f in range(a.folds):
        tr = np.where(F != f)[0]
        if not len(tr) or not len(np.where(F == f)[0]):
            continue
        if RULE_J is not None:
            tr_rows = [k for k, r in enumerate(TR) if fold[r["well"]] != f]
            te_rows = [k for k, r in enumerate(TR) if fold[r["well"]] == f]
            # ⚠ порог выбирается ТОЛЬКО на обучающих фолдах (иначе это подгонка на держанном)
            best_t = max(GRID, key=lambda t: score_rule(tr_rows, t))
            got += score_rule(te_rows, best_t)
            if CNT is not None:
                CNT["thr"].append(float(best_t))
                for k in te_rows:
                    r = TR[k]
                    pick = dict(r["base"])
                    for s in r["S"]:
                        j = key.get((id(r), s))
                        if j is not None and _fire(j, best_t):
                            pick[s] = "B" if r["base"][s] == "A" else "A"
                    _acc(r, pick)
            continue
        b = fit(tr, labels)
        for r in TR:
            if fold[r["well"]] != f:
                continue
            pick = dict(r["base"])
            for s in r["S"]:
                j = key.get((id(r), s))
                if j is None:
                    continue
                p = 1 / (1 + np.exp(-(np.r_[Z[j], 1.0] @ b)))
                if p >= 0.5:
                    pick[s] = "B" if r["base"][s] == "A" else "A"
            got += sc(r, pick)
            _acc(r, pick)
    return got


if RULE_J is not None:
    # ★ вся сетка на ВСЕХ треках — для чтения формы кривой (не держанная оценка)
    allrows = list(range(len(TR)))
    curve = [(t, score_rule(allrows, t)) for t in GRID]
    print("★ сетка порога (все треки, НЕ держанно): " +
          "  ".join(f"{t:+.2f}→{v}" for t, v in curve if (0.0 <= t < 0.3) or (abs(t * 10 - round(t * 10)) < 1e-6)))


CNT = {"AB": 0, "BA": 0, "gain_g": 0, "gain_ng": 0, "K": {}, "thr": []}
held = cv(Y)
print(f"★ РАЗБОР (раскладка стенда): переворотов A→B {CNT['AB']}, B→A {CNT['BA']}; прирост на гейтованных "
      f"листах {CNT['gain_g']:+d}, вне гейта {CNT['gain_ng']:+d}; по K трека " +
      ", ".join(f"{k}{'+' if k == 5 else ''}: {v:+d}" for k, v in sorted(CNT["K"].items())) +
      (f"; пороги по фолдам {CNT['thr']}" if CNT["thr"] else ""))
CNT = None
rng = np.random.default_rng(0)
nul = [cv(rng.permutation(Y)) for _ in range(3)] if RULE_J is None else []
lay = []
for sd in range(1, a.layouts + 1):
    relayout(sd); lay.append(cv(Y))
relayout(None)
print(f"\n★★ ОБУЧЕННЫЙ ВЫБОР ПО СЛОТУ, перекрёстно по скважинам ({a.folds} фолдов): {held}")
print(f"   против опоры {base_vec} ({held-base_vec:+d}), потолок {orac} "
      f"({100*(held-base_vec)/max(1,orac-base_vec):.0f}% зазора)")
if lay:
    print(f"★★ КОНТРОЛЬ НА РАСКЛАДКУ: {len(lay)} случайных раскладок скважин по фолдам → "
          f"{min(lay)}…{max(lay)}, среднее {np.mean(lay):.0f} ({np.mean(lay)-base_vec:+.0f} к опоре); "
          f"раскладка стенда {held} — "
          f"{'внутри разброса' if min(lay) <= held <= max(lay) else '⛔ ВНЕ разброса: цитировать разброс, а не её'}")
if nul:
    m = float(np.mean(nul))
    good = m - base_vec < 0.2 * max(1, held - base_vec)
    print(f"⚠ нуль-контроль (метки перемешаны, 3 повтора): {m:.0f} при опоре {base_vec} — "
          f"{'★ шум прироста не даёт' if good else '⛔ ПРОЦЕДУРА ЗАВЫШАЕТ — числа выше негодны'}")
else:
    print("⚠ нуль-контроль для правила не определён (меток нет); контроль — раскладки и сетка порога")

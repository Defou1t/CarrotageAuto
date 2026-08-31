r"""_rowdec_transfer.py — КУДА ДЕВАЕТСЯ ПРЕИМУЩЕСТВО ДЕКОДЕРА МЕЖДУ ПУЛОМ И ВЫДАННЫМ ФАЙЛОМ (§6.143 → §6.144)

ВОПРОС. На пуловом пути построчный декодер берёт **44.0%** против **27.9%** у прода на той же
выборке (§6.140/§6.141) — шестнадцать пунктов. На отгружаемом пути, безымянным счётом, оба дают
**35.4%** (976 против 978 из 2760, §6.143). Преимущество исчезает целиком, и никто не мерил, где.
Пока это неизвестно, ЛЮБОЙ следующий механизм ведения потеряет там же.

ЧТО ДЕЛАЕТ СТЕНД. Читает ГОТОВЫЕ выдачи двух режимов одного прогона (пайплайн не запускается) и
раскладывает КАЖДУЮ экспертную кривую по клеткам «честна у A / честна у B», а для расходящихся
называет ПРИЧИНУ проигравшего:
    нет кандидата  — на треке этой кривой режим не написал НИЧЕГО;
    med > 3px      — трасса есть и полная, но идёт не там (ГЕОМЕТРИЯ/личность);
    cov < 0.9      — трасса идёт верно, но коротка (ПОКРЫТИЕ/обрыв);
    и то и другое.
⚠ Кандидат ищется по ТРЕКУ и БЕЗ учёта имени — это безымянный счёт, ведущий с 20.08. Причина
берётся у ЛУЧШЕГО кандидата трека, а не у одноимённого: иначе разбор мерил бы подпись, а не ведение.

★ ЗАЧЕМ ОТДЕЛЬНЫЙ СТЕНД, А НЕ СТРОКА В `_name_cost_prod.py`. Тот считает СКОЛЬКО, этот — ПОЧЕМУ.
§6.143 показал, чем кончается, когда две разные величины снимаются одним инструментом и сводятся
руками.

  python _rowdec_transfer.py --dir F:/nds/output/taskS/ab_rowdec_pair --a A --b B
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import json
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:/nds/output/taskS/ab_rowdec_pair")
ap.add_argument("--a", default="A", help="базовый режим (прод)")
ap.add_argument("--b", default="B", help="сравниваемый режим (декодер)")
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
a = ap.parse_args()
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


def why(med, cov):
    if med is None:
        return "нет кандидата"
    bad_m = med > 3.0
    bad_c = cov < 0.9
    if bad_m and bad_c:
        return "и то и другое"
    return "med > 3px" if bad_m else ("cov < 0.9" if bad_c else "честна")


SRC, WELL = {}, {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC.setdefault(q.name, q); WELL.setdefault(q.name, wlg.parent.name)
print(f"разметок найдено: {len(SRC)}")

TRACK, seen = {}, set()
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        if f.stem in seen:
            continue
        seen.add(f.stem)
        d = pickle.load(open(f, "rb"))
        TRACK[d["name"]] = {s["name"]: s["track"] for s in d["slots"]}
print(f"треков загружено для {len(TRACK)} листов")

BYDIR = {f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}": nm
         for nm, q in SRC.items()}


def read_mode(mode):
    """→ {лист: ({кривая: трасса}, {трек: линий U1})}, разобрано, пропущено, каталогов."""
    MD = Path(a.dir) / mode
    dirs = sorted((d for d in MD.iterdir() if d.is_dir()), key=lambda p: p.name) if MD.is_dir() else []
    out, done, skip = {}, 0, 0
    for d in dirs:
        nm = BYDIR.get(d.name)
        got = next(iter(sorted(d.glob("*_auto.nlgx"))), None)
        if nm is None or got is None:
            skip += 1; continue
        wr = {c["name"]: dense(c) for c in extract(str(got))["curves"]
              if M.mnem_root(c["name"]) != "DA"}
        lines = Counter()
        u = next(iter(sorted(d.glob("*_understanding.json"))), None)
        if u is not None:
            try:
                for L in json.loads(u.read_text(encoding="utf-8")).get("lines", []):
                    if L.get("confidence") == "AUTO":
                        lines[L.get("track")] += 1
            except Exception:
                pass
        out[nm] = (wr, lines); done += 1
    print(f"★ СВЕРКА {mode}: разобрано {done} + пропущено {skip} = {done+skip} против "
          f"{len(dirs)} каталогов   {'★ СОШЛОСЬ' if done+skip == len(dirs) else '⛔ НЕ СОШЛОСЬ'}")
    return out, done, skip, len(dirs)


A, dA, sA, nA = read_mode(a.a)
B, dB, sB, nB = read_mode(a.b)
both = sorted(set(A) & set(B))
print(f"листов у {a.a} {dA}, у {a.b} {dB}, ОБЩИХ {len(both)}")

cell = Counter()                 # (честна у A, честна у B) → кривых
whyB = Counter()                 # причина проигрыша B там, где A честна
whyA = Counter()                 # причина проигрыша A там, где B честна
medB_lost, covB_lost = [], []
gaps = Counter()                 # трек: линий U1 против написанных кривых
sig = Counter()                  # (согласие путей, клетка) → кривых
sigk = Counter()                 # (кривых на треке ПО ЭТАЛОНУ, клетка) → кривых
sigl = Counter()                 # (линий U1 на треке — НАБЛЮДАЕМО, клетка)
sigw = Counter()                 # (написано кривых на треке — НАБЛЮДАЕМО, клетка)
mismatch = [0]                   # у скольких кривых число линий U1 разошлось между режимами
rows = []                        # (скважина, написано кривых на треке, честна у A, честна у B)
curves = 0
for nm in both:
    src = SRC[nm]
    gts = {c["name"]: dense(c) for c in extract(str(src))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    if not gts:
        continue
    tmap = TRACK.get(nm, {})
    wrA, linA = A[nm]
    wrB, linB = B[nm]
    for ti in {tmap.get(g) for g in gts}:
        nl = linB.get(ti, 0) if ti is not None else 0
        nw = sum(1 for k in wrB if tmap.get(k) == ti)
        if ti is not None:
            gaps[(nl, nw)] += 1
    for g, gt in gts.items():
        curves += 1
        ti = tmap.get(g)

        def best(wr):
            """→ (медиана, покрытие, ИМЯ лучшего кандидата своего трека)."""
            cand = [k for k in wr if ti is None or tmap.get(k) == ti] or list(wr)
            bm, bc, bk = None, 0.0, None
            for k in cand:
                m, c = err(wr[k], gt)
                if m is None:
                    continue
                if bm is None or m < bm:
                    bm, bc, bk = m, c, k
            return bm, bc, bk
        mA, cA, kA = best(wrA)
        mB, cB, kB = best(wrB)
        hA, hB = HON(mA, cA), HON(mB, cB)
        cell[(hA, hB)] += 1
        if hA and not hB:
            whyB[why(mB, cB)] += 1
            if mB is not None:
                medB_lost.append(mB); covB_lost.append(cB)
        if hB and not hA:
            whyA[why(mA, cA)] += 1
        # ── ★ ЕСТЬ ЛИ СИГНАЛ, ПО КОТОРОМУ МОЖНО ВЫБРАТЬ ПУТЬ, НЕ ЗНАЯ ЭТАЛОНА ────────────
        # ⚠ Всё, что здесь считается, ДОСТУПНО НА ВЫДАЧЕ: согласие двух путей между собой и
        # число кривых на треке. Эталон нужен только чтобы узнать, КТО был прав, — то есть это
        # проверка разделимости, а не механизм.
        agree = None
        if kA is not None and kB is not None:
            com = [y for y in wrA[kA] if y in wrB[kB]]
            if len(com) >= 30:
                agree = float(np.median([abs(wrA[kA][y] - wrB[kB][y]) for y in com]))
        nk = sum(1 for x in gts if tmap.get(x) == ti) if ti is not None else 0
        bucket = ("нет пары" if agree is None else
                  "сходятся ≤3px" if agree <= 3 else
                  "расходятся 3-30px" if agree <= 30 else "расходятся >30px")
        sig[(bucket, (hA, hB))] += 1
        sigk[(min(nk, 6), (hA, hB))] += 1
        # ⚠⚠ НАБЛЮДАЕМЫЙ АНАЛОГ. `nk` СЧИТАН ПО ЭТАЛОНУ — на выдаче его нет, и правило по нему
        # непостроимо. Наблюдаемых аналога два, оба берутся из того же `_understanding.json`,
        # который пайплайн и так кладёт рядом с выдачей: сколько линий нашёл U1 и сколько кривых
        # режим написал. Правило можно строить только по ним.
        nl = linB.get(ti, 0) if ti is not None else 0
        nw = sum(1 for k in wrB if tmap.get(k) == ti) if ti is not None else 0
        sigl[(min(nl, 6), (hA, hB))] += 1
        sigw[(min(nw, 6), (hA, hB))] += 1
        if linA.get(ti, 0) != nl:
            mismatch[0] += 1
        rows.append((WELL.get(nm, "?"), min(nw, 6), hA, hB))

print(f"\nэкспертных кривых разобрано {curves}")
print(f"\n{'='*78}\n★★ ГДЕ КАЖДАЯ КРИВАЯ (безымянно, лучший кандидат СВОЕГО трека)\n{'='*78}")
print(f"  честна у обоих:              {cell[(True, True)]}")
print(f"  честна только у {a.a} (прод):     {cell[(True, False)]}")
print(f"  честна только у {a.b} (декодер):  {cell[(False, True)]}")
print(f"  не берёт никто:              {cell[(False, False)]}")
print(f"  ⇒ нетто {a.b}−{a.a}: {cell[(False, True)] - cell[(True, False)]:+d} при "
      f"{cell[(True, False)] + cell[(False, True)]} расходящихся")

print(f"\n★★ ПОЧЕМУ {a.b} ТЕРЯЕТ ТАМ, ГДЕ {a.a} БЕРЁТ ({sum(whyB.values())} кривых)")
for k, v in whyB.most_common():
    print(f"    {k:<16} {v:>5}  ({100*v/max(1,sum(whyB.values())):4.1f}%)")
if medB_lost:
    md = np.array(medB_lost); cv = np.array(covB_lost)
    print(f"    у проигранных: медиана med {np.median(md):.1f}px, доля med≤3 {100*(md<=3).mean():.0f}%; "
          f"медиана cov {np.median(cv):.2f}, доля cov≥0.9 {100*(cv>=0.9).mean():.0f}%")
print(f"\n★ ПОЧЕМУ {a.a} ТЕРЯЕТ ТАМ, ГДЕ {a.b} БЕРЁТ ({sum(whyA.values())} кривых)")
for k, v in whyA.most_common():
    print(f"    {k:<16} {v:>5}  ({100*v/max(1,sum(whyA.values())):4.1f}%)")

un = cell[(True, True)] + cell[(True, False)] + cell[(False, True)]
print(f"\n{'='*78}\n★★★ ОБЪЕДИНЕНИЕ ДВУХ ПУТЕЙ (оракул выбора, знает эталон): {un} кривых "
      f"({100*un/max(1,curves):.1f}%)\n{'='*78}")
print(f"   против {cell[(True,True)]+cell[(True,False)]} у {a.a} и "
      f"{cell[(True,True)]+cell[(False,True)]} у {a.b} по отдельности "
      f"⇒ **{un - max(cell[(True,True)]+cell[(True,False)], cell[(True,True)]+cell[(False,True)]):+d} "
      f"кривых**, если бы путь выбирался ПОКРИВОЙ")
print(f"⚠ Это ОРАКУЛ: кто прав, знает только эталон. Ниже — есть ли наблюдаемый признак,\n"
      f"  по которому выбор можно сделать БЕЗ него.")


def tab(counter, title, key_name):
    print(f"\n★★ {title}")
    print(f"   {key_name:<20}{'обе':>6}{'только '+a.a:>12}{'только '+a.b:>12}{'никто':>8}"
          f"{'  доля ' + a.b + ' среди расходящихся':>10}")
    for k in sorted({k for k, _ in counter}, key=str):
        tt = counter[(k, (True, True))]; ta = counter[(k, (True, False))]
        tb = counter[(k, (False, True))]; tn = counter[(k, (False, False))]
        d = ta + tb
        print(f"   {str(k):<20}{tt:>6}{ta:>12}{tb:>12}{tn:>8}"
              f"{('   ' + f'{100*tb/d:.0f}%') if d else '     —':>10}")


tab(sig, "СИГНАЛ 1 — СОГЛАСИЕ ДВУХ ПУТЕЙ МЕЖДУ СОБОЙ (наблюдаемо на выдаче)", "согласие")
tab(sigk, "СИГНАЛ 2 — СКОЛЬКО ЭКСПЕРТНЫХ КРИВЫХ НА ТРЕКЕ (⛔ ПО ЭТАЛОНУ, правило по нему НЕ ПОСТРОИТЬ)", "кривых на треке")
tab(sigl, "СИГНАЛ 3 — СКОЛЬКО ЛИНИЙ НАШЁЛ U1 НА ТРЕКЕ (★ НАБЛЮДАЕМО)", "линий U1")
tab(sigw, "СИГНАЛ 4 — СКОЛЬКО КРИВЫХ РЕЖИМ НАПИСАЛ НА ТРЕКЕ (★ НАБЛЮДАЕМО)", "написано кривых")
print(f"\n★ КОНТРОЛЬ: число линий U1 разошлось между режимами у {mismatch[0]} кривых из {curves}  "
      f"{'★ U1 ОДИН И ТОТ ЖЕ' if not mismatch[0] else '⚠⚠ РЕЖИМЫ ВИДЯТ РАЗНЫЕ ЛИНИИ — признак негоден'}")
print(f"\n⚠ Признак ГОДЕН, если доля {a.b} среди расходящихся заметно уходит от 50% в разных "
      f"строках: тогда по нему можно выбирать путь. Ровно 50% везде = признак пустой.")

print(f"\n★ СКОЛЬКО ТРАЕКТОРИЙ {a.b} ДОЕЗЖАЕТ ДО ФАЙЛА (линий U1 на треке → написано кривых)")
lost = sum(n for (nl, nw), n in gaps.items() if nw < nl)
tot = sum(gaps.values())
print(f"    треков всего {tot}; где написано МЕНЬШЕ, чем линий: {lost} ({100*lost/max(1,tot):.1f}%)")
for (nl, nw), n in sorted(gaps.items())[:14]:
    if nl:
        print(f"      линий {nl:>2} → написано {nw:>2}: {n}")


# ═══ ПРАВИЛО ВЫБОРА ПУТИ ПО ТРЕКУ, ПРОВЕРЕННОЕ НА ДЕРЖАННЫХ СКВАЖИНАХ ═════════════════════════
# ⚠⚠ ЗАЧЕМ ПЕРЕКРЁСТНАЯ ПРОВЕРКА, А НЕ ПРОСТО СУММА ПО ТАБЛИЦЕ. Порог «декодер до N кривых на
# треке, дальше прод» выбран ГЛЯДЯ НА ЭТИ ЖЕ ЧИСЛА — а менять правило после того, как увидел
# число, есть ровно та ошибка, на которой ветка горела четырежды (§6 правило 1). Поэтому порог
# подбирается на 4/5 скважин и применяется к пятой, и в зачёт идёт ТОЛЬКО держанная часть.
print(f"\n{'='*78}\n★★★ ПРАВИЛО: НА КАКИХ ТРЕКАХ БРАТЬ ДЕКОДЕР, А НА КАКИХ ПРОД\n{'='*78}")
wells = sorted({w for w, _, _, _ in rows})
fold = {w: i % 5 for i, w in enumerate(wells)}


def take(sub, thr):
    """Сколько честных даст правило «nw ≤ thr → декодер, иначе прод» на наборе sub."""
    return sum((hB if nw <= thr else hA) for _, nw, hA, hB in sub)


allA = sum(hA for _, _, hA, _ in rows)
allB = sum(hB for _, _, _, hB in rows)
print(f"   на всех {len(rows)} кривых:  {a.a} {allA},  {a.b} {allB}")
print(f"\n   порог  в лоб (тот же набор)")
for thr in range(0, 7):
    print(f"     ≤{thr}      {take(rows, thr)}")

held = 0
picked = Counter()
for f in range(5):
    tr = [r for r in rows if fold[r[0]] != f]
    te = [r for r in rows if fold[r[0]] == f]
    if not te:
        continue
    thr = max(range(0, 7), key=lambda t: take(tr, t))
    picked[thr] += 1
    held += take(te, thr)
print(f"\n   ★ ПЕРЕКРЁСТНАЯ ПРОВЕРКА ПО СКВАЖИНАМ (5 фолдов, порог с 4/5, зачёт с 1/5):")
print(f"     честных по правилу {held}   против {allA} у {a.a} и {allB} у {a.b}")
print(f"     ⇒ {held - max(allA, allB):+d} кривых к лучшему из двух путей "
      f"({100*(held-max(allA,allB))/max(1,max(allA,allB)):+.1f}%)")
print(f"     порог, выбранный фолдами: {dict(picked)}")
print(f"\n⚠ Счёт здесь — ЛУЧШИЙ КАНДИДАТ ТРЕКА, а не паросочетание 1:1, поэтому он на 1-2% выше,")
print(f"  чем даёт `_name_cost_prod.py`. Сравнивать можно только внутри этой таблицы.")
print(f"⚠ И правило меняет ПУТЬ ВЕДЕНИЯ, то есть требует прогона обоих — цена вдвое.")


# ═══ ТО ЖЕ ПРАВИЛО, НО ОФИЦИАЛЬНОЙ МЕТРИКОЙ ВЕТКИ — ПАРОСОЧЕТАНИЕМ 1:1 ═══════════════════════
# ⚠⚠ ЗАЧЕМ ВТОРОЙ СЧЁТ. Всё выше считано «лучшим кандидатом трека»: одна траектория могла оказаться
# лучшей сразу для двух экспертных кривых и зачлась дважды. Ведущий счёт ветки (§6.126, §6.143) —
# максимальное 1:1 внутри трека, и цитировать надо его. Здесь правило применяется ПО ТРЕКУ (берём
# кривые того режима, который правило выбрало), а затем считается то же паросочетание, что в
# `_name_cost_prod.py`.
def match(rows_, cols, ok):
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
    return sum(1 for r in rows_ if try_(r, set()))


def free_count(thr):
    """Честных 1:1 внутри трека, если путь выбирается правилом «nw ≤ thr → декодер».
    thr = -1 → всегда прод, thr = 99 → всегда декодер."""
    per = Counter()
    for nm in both:
        src = SRC[nm]
        gts = {c["name"]: dense(c) for c in extract(str(src))["curves"]
               if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        if not gts:
            continue
        tmap = TRACK.get(nm, {})
        wrA, _ = A[nm]; wrB, _ = B[nm]
        tot = 0
        for ti in {tmap.get(g) for g in gts}:
            if ti is None:
                continue
            G = [g for g in gts if tmap.get(g) == ti]
            nw = sum(1 for k in wrB if tmap.get(k) == ti)
            wr = wrB if nw <= thr else wrA
            W = [k for k in wr if tmap.get(k) == ti]
            if not G or not W:
                continue
            ok = {(g, w): HON(*err(wr[w], gts[g])) for g in G for w in W}
            tot += match(G, W, ok)
        per[WELL.get(nm, "?")] += tot
    return per


print(f"\n{'='*78}\n★★★ ТО ЖЕ ПРАВИЛО ОФИЦИАЛЬНОЙ МЕТРИКОЙ (1:1 внутри трека, как §6.143)\n{'='*78}")
PER = {t: free_count(t) for t in (-1, 1, 2, 3, 4, 99)}
for t in (-1, 1, 2, 3, 4, 99):
    nm_ = "всегда прод" if t == -1 else ("всегда декодер" if t == 99 else f"декодер при nw ≤ {t}")
    print(f"   {nm_:<24} {sum(PER[t].values())}")
held = 0
picked = Counter()
for f in range(5):
    tr_w = [w for w in wells if fold[w] != f]
    te_w = [w for w in wells if fold[w] == f]
    thr = max((-1, 1, 2, 3, 4, 99), key=lambda t: sum(PER[t][w] for w in tr_w))
    picked[thr] += 1
    held += sum(PER[thr][w] for w in te_w)
base = max(sum(PER[-1].values()), sum(PER[99].values()))
print(f"\n   ★ ПЕРЕКРЁСТНАЯ ПРОВЕРКА ПО СКВАЖИНАМ: {held} против {base} у лучшего из двух путей")
print(f"     ⇒ {held - base:+d} кривых ({100*(held-base)/max(1,base):+.1f}%), порог по фолдам {dict(picked)}")


# ═══ ШУМ ПРАВИЛА: ПАРНЫЙ ТЕСТ ПРОТИВ ПРОДА ПО ЛИСТАМ (§6 правило 3, §6.118) ═══════════════════
# ⚠⚠ Без измеренного шума прирост нецитируем: «+13» три недели читалось как довод, пока не
# посчитали разброс (§6.118). Единица — ЛИСТ, метка «прод/правило» под H0 произвольна.
def free_per_sheet(thr):
    per = {}
    for nm in both:
        src = SRC[nm]
        gts = {c["name"]: dense(c) for c in extract(str(src))["curves"]
               if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        if not gts:
            continue
        tmap = TRACK.get(nm, {})
        wrA, _ = A[nm]; wrB, _ = B[nm]
        tot = 0
        for ti in {tmap.get(g) for g in gts}:
            if ti is None:
                continue
            G = [g for g in gts if tmap.get(g) == ti]
            nw = sum(1 for k in wrB if tmap.get(k) == ti)
            wr = wrB if nw <= thr else wrA
            W = [k for k in wr if tmap.get(k) == ti]
            if not G or not W:
                continue
            ok = {(g, w): HON(*err(wr[w], gts[g])) for g in G for w in W}
            tot += match(G, W, ok)
        per[nm] = tot
    return per


PS = {t: free_per_sheet(t) for t in (-1, 3)}
sheets_ = sorted(set(PS[-1]) & set(PS[3]))
d = np.array([PS[3][n] - PS[-1][n] for n in sheets_], float)
nz = d[d != 0]
rng = np.random.default_rng(20260824)
obs = float(d.sum())
null = ((rng.integers(0, 2, size=(20000, len(nz))) * 2 - 1) * nz).sum(1)
p_perm = float((np.abs(null) >= abs(obs)).mean())
up = int((d > 0).sum()); dn = int((d < 0).sum())
from math import comb
n_ = up + dn
p_sign = min(1.0, 2 * sum(comb(n_, i) for i in range(max(up, dn), n_ + 1)) / 2 ** n_) if n_ else 1.0
print(f"\n{'='*78}\n★★★ ПРАВИЛО (порог 3) ПРОТИВ ПРОДА, ПАРНО ПО ЛИСТАМ\n{'='*78}")
print(f"   листов {len(sheets_)};  прод {int(sum(PS[-1].values()))} → правило {int(sum(PS[3].values()))}"
      f"   Δ = {int(obs):+d}")
print(f"   листов ↑{up} / ↓{dn} (совпало {len(sheets_)-up-dn}); SD нуля {null.std():.1f}; "
      f"перестановочный p = {p_perm:.5f}; знаковый p = {p_sign:.6f}")
print(f"⚠ Порог 3 здесь ФИКСИРОВАН (он же выбран перекрёстной проверкой выше) — тест меряет шум "
      f"эффекта, а не подбор порога; держанная оценка эффекта = +88, не {int(obs):+d}.")


# ═══ САМАЯ БОЛЬШАЯ КОРЗИНА: КРИВЫЕ, КОТОРЫХ НЕ БЕРЁТ НИ ОДИН ПУТЬ ════════════════════════════
# ⚠⚠ ЗАЧЕМ. Клетка «не берёт никто» — 1396 кривых из 2760 (50.6%), больше всех остальных вместе.
# Пока она не вскрыта, любой разговор о том, куда идти после выбора пути, — угадывание: неизвестно
# даже, лежит там недоведение, промах по личности или отсутствие трассы вообще.
# ⚠ Разбирается по ЛУЧШЕМУ из двух путей: если уж и он не берёт, то дело не в выборе.
print(f"\n{'='*78}\n★★★ ЧТО В КОРЗИНЕ «НЕ БЕРЁТ НИКТО»\n{'='*78}")
none_med, none_cov, why_none = [], [], Counter()
band = Counter()
for nm in both:
    src = SRC[nm]
    gts = {c["name"]: dense(c) for c in extract(str(src))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    if not gts:
        continue
    tmap = TRACK.get(nm, {})
    wrA, _ = A[nm]
    wrB, _ = B[nm]
    for g, gt in gts.items():
        ti = tmap.get(g)

        def best2(wr):
            cand = [k for k in wr if ti is None or tmap.get(k) == ti] or list(wr)
            bm, bc = None, 0.0
            for k in cand:
                m, c = err(wr[k], gt)
                if m is None:
                    continue
                if bm is None or m < bm:
                    bm, bc = m, c
            return bm, bc
        mA, cA = best2(wrA)
        mB, cB = best2(wrB)
        if HON(mA, cA) or HON(mB, cB):
            continue
        # ЛУЧШИЙ из двух: тот, у кого медиана меньше (при равенстве — с бо́льшим покрытием)
        cand = [(m, c) for m, c in ((mA, cA), (mB, cB)) if m is not None]
        if not cand:
            why_none["трассы на треке НЕТ ВОВСЕ"] += 1
            continue
        m, c = min(cand, key=lambda z: (z[0], -z[1]))
        none_med.append(m); none_cov.append(c)
        why_none[why(m, c)] += 1
        band["≤3px (мешает только покрытие)" if m <= 3 else
             "3-10px" if m <= 10 else "10-50px" if m <= 50 else
             "50-200px" if m <= 200 else ">200px"] += 1
tot_none = sum(why_none.values())
print(f"кривых в корзине: {tot_none}")
for k, v in why_none.most_common():
    print(f"   {k:<28} {v:>5}  ({100*v/max(1,tot_none):4.1f}%)")
if none_med:
    md = np.array(none_med); cv = np.array(none_cov)
    print(f"\n   ошибка ЛУЧШЕГО из двух путей: медиана {np.median(md):.1f}px, "
          f"p25 {np.percentile(md,25):.1f}, p75 {np.percentile(md,75):.1f}")
    print(f"   покрытие лучшего: медиана {np.median(cv):.2f}, доля cov≥0.9 {100*(cv>=0.9).mean():.0f}%")
print(f"\n   как далеко промахивается лучший путь:")
for k in ("≤3px (мешает только покрытие)", "3-10px", "10-50px", "50-200px", ">200px"):
    if band[k]:
        print(f"     {k:<30} {band[k]:>5}  ({100*band[k]/max(1,sum(band.values())):4.1f}%)")
print(f"\n⚠ Читать так: «≤3px» — геометрия уже верна, кривую держит ПОКРЫТИЕ; «3-10px» — полоса,")
print(f"  закрытая §6.119 постфильтрами; «>50px» — путь идёт по ЧУЖОЙ кривой, это личность.")

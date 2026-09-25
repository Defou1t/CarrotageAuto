r"""_u1_vs_dec.py — ДАЁТ ЛИ ОБУЧЕННЫЙ ДЕТЕКТОР ТО, ЧЕГО НЕ ДАЁТ U1 (§6.199 → первый шаг B1).

ОТКУДА ВОПРОС. §6.197 разложил остаток по исходу U1 и показал: **все +509 кривых лежат там, где
линия U1 неверна больше чем на 3px**, а тушь под нужной кривой есть (потолок V2 в провальных
корзинах 39-48%). HANDOFF назвал первый, ещё не сделанный шаг: **проверить, даёт ли ОБУЧЕННЫЙ
детектор линий на тех же сканах то, чего не даёт нынешний U1.**

★ ТАКОЙ ДЕТЕКТОР УЖЕ ЕСТЬ, И ЕГО ВЫДАЧА ЗАМОРОЖЕНА. Построчный декодер (`auto/rowdec.py`) строит
траектории ИЗ КАРТИНКИ: от U1 он берёт только ЧИСЛО линий трека `K = len(lines)` и вертикальный
диапазон `y0..y1` (`rowdec.py:368-371`), а x-полосы линий U1 в траекторию не входят — они нужны
лишь для сопоставления готовых трасс с линиями ради цвета/класса при раскладке (`rowdec.py:392-429`).
⇒ Вопрос «даёт ли обученный детектор другое» отвечается ПО ЗАМОРОЖЕННЫМ ДАННЫМ, без прогона.

ЧТО СЧИТАЕТ. Та же разбивка эталонных кривых по исходу U1, что у `_u1_vs_prod.py` (§6.196), но
колонок не одна, а четыре — и все считаются ОДНОЙ функцией `match`, скопированной из
`_name_cost_prod.py` (§6.198: подстановка именного счёта вместо безымянного стоила трёх разделов):

  P965  `ab_wellmap/A`   — прод без декодера          (сверка: обязан дать 965)
  DEC   `ab_rdhonest/B`  — ЧЕСТНЫЙ ДЕКОДЕР ОДНИМ ПУТЁМ (сверка: обязан дать 934, §6.175)
  CUR   `ab_pregate/G`   — ТЕКУЩИЙ ПРОД с предгейтом   (сверка: обязан дать 1107, §6.188)
  ∪     потрековый 1:1 по ОБЪЕДИНЁННОМУ набору кандидатов CUR ∪ DEC — потолок ПОКРИВОГО выбора
        между нынешним продом и чистым детектором.

★ ОПОРА — ТЕКУЩАЯ ЛУЧШАЯ (CUR, 1107), а не прод-965: §6.180 показал, что контроль со слабой
опорой проходят и пустые улучшения. Колонка P965 оставлена только ради сверки с §6.196.

★ Счёта не тратит: пуловые дампы (линии U1), три замороженные выдачи, разметка.

  <ComfyUI>\python_embeded\python.exe _u1_vs_dec.py
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
ap.add_argument("--p965", default="ab_wellmap/A")
ap.add_argument("--dec", default="ab_rdhonest/B")
ap.add_argument("--cur", default="ab_pregate/G")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--cap", type=int, default=0)
ap.add_argument("--dump", default="", help="выгрузить покривой исход в pickle")
a = ap.parse_args()
TS = Path(a.ts)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def st(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    return float(np.median([abs(tr[y] - gt[y]) for y in com])), len(com) / max(1, len(gt))


def match(rows, cols, ok):
    """⚠⚠ ВЕДУЩИЙ СЧЁТ ВЕТКИ — БЕЗЫМЯННЫЙ: максимальное 1:1 внутри трека (§6.143). Механика взята
    из `_name_cost_prod.py` дословно; возвращает СПИСОК взятых эталонных кривых."""
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
    return [r for r in rows if try_(r, set())]


def read_out(root, sh):
    """Кривые ЗАМОРОЖЕННОЙ выдачи для листа — тем же именем каталога, что и у `_u1_vs_prod.py`."""
    stem = Path(sh).stem
    dn = f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    pd = TS / root / dn
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    if not got:
        return {}
    return {c["name"]: dense(c) for c in extract(str(got))["curves"]
            if M.mnem_root(c["name"]) != "DA"}


trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]): int(r[3]["n_prod"]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
stem2f = {}
for root in ("pools", "pools_gate", "pools_wide", "pools_more", "pools_div",
             "pools_heldout", "pools_all"):
    for f in sorted((TS / root).glob("*.pkl")):
        stem2f.setdefault(f.stem, f)
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q

ORDER = ["★ накрыта", "короткая (med≤3, cov<0.9)", "мелкий промах 3-10px",
         "промах 10-100px", "крупный >100px", "линии нет / нет пересечения"]
PATHS = [("P965", a.p965), ("DEC", a.dec), ("CUR", a.cur)]
UNI = {"CUR∪DEC": ("CUR", "DEC"), "P965∪DEC": ("P965", "DEC"),
       "ВСЕ ТРИ": ("P965", "DEC", "CUR")}
COLS = ["P965", "DEC", "CUR", "CUR∪DEC", "P965∪DEC", "ВСЕ ТРИ"]
tab = defaultdict(Counter)
SENS = [None] + [Counter() for _ in range(3)]        # [0] — естественный порядок = сама таблица
RNG = [np.random.default_rng(s) for s in (1, 2, 3)]
DUMP = []

sheets = sorted({r[0] for r in trk})
if a.cap:
    sheets = sheets[:a.cap]
miss, done = Counter(), []
for si, sh in enumerate(sheets, 1):
    f, q = stem2f.get(Path(sh).stem), SRC.get(sh)
    if not f or not q:
        miss["нет пула/исходника"] += 1
        continue
    done.append(sh)
    d = pickle.load(open(f, "rb"))
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    bytr = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None:
            bytr[t].append(nm)
    lines = defaultdict(list)
    for L in d["lines"]:
        lines[int(L["track"])].append(L["tr"])
    out = {}
    for tag, root in PATHS:
        w = read_out(root, sh)
        if not w:
            # ⛔ ПУСТАЯ ВЫДАЧА — ФАТАЛЬНО, А НЕ ЗАМЕТКА. Лист остаётся в поле, его эталонные кривые
            # идут в знаменатель, а колонка занижается — и сверка по другим путям всё равно
            # «сойдётся». Первая редакция считала это в `miss` и печатала, но в вердикт не вносила.
            miss[f"⛔ пустая выдача {tag}"] += 1
        out[tag] = w
    for t, ns in bytr.items():
        if (sh, t) not in KOF:
            continue
        LL = lines.get(t, [])
        took, okk = {}, {}
        for tag, _ in PATHS:
            W = [k for k in out[tag] if tm.get(k) == t]
            ok = {(g, k): HON(*st(out[tag][k], gts[g])) for g in ns for k in W}
            okk[tag] = (W, ok)
            took[tag] = set(match(ns, W, ok))
        # ★ ОБЪЕДИНЁННЫЙ набор кандидатов: 1:1 по колонкам ОБОИХ путей сразу, с пространством
        #   имён — иначе одноимённые кривые двух выдач слились бы в одну колонку.
        for uname, tags in UNI.items():
            cols, oku = [], {}
            for tag in tags:
                W, ok = okk[tag]
                for k in W:
                    cols.append((tag, k))
                    for g in ns:
                        oku[(g, (tag, k))] = ok[(g, k)]
            took[uname] = set(match(ns, cols, oku))
        buck = {}
        for nm in ns:
            g = gts[nm]
            sts = [st(tr, g) for tr in LL]
            if any(HON(m, c) for m, c in sts):
                buck[nm] = ORDER[0]
            else:
                cand = [(m, c) for m, c in sts if m is not None]
                if not cand:
                    buck[nm] = ORDER[5]
                else:
                    m, c = min(cand)
                    buck[nm] = (ORDER[1] if m <= 3.0 else ORDER[2] if m <= 10 else
                                ORDER[3] if m <= 100 else ORDER[4])
        for nm in ns:
            k = buck[nm]
            tab[k]["n"] += 1
            for col in COLS:
                if nm in took[col]:
                    tab[k][col] += 1
            if nm in took["DEC"] and nm not in took["CUR"]:
                tab[k]["ТОЛЬКО DEC"] += 1
            if nm in took["CUR"] and nm not in took["DEC"]:
                tab[k]["ТОЛЬКО CUR"] += 1
            if a.dump:
                DUMP.append((sh, t, nm, k, {c: nm in took[c] for c in COLS}))
        # ★★ ЧУВСТВИТЕЛЬНОСТЬ К ПОРЯДКУ ПАРОСОЧЕТАНИЯ. Максимальное 1:1 задаёт РАЗМЕР однозначно,
        # а СОСТАВ взятых кривых — нет: при нескольких максимумах побеждает та, что раньше в файле
        # разметки. Итоги по путям от этого не зависят, но «взял именно ЭТУ кривую» — зависит,
        # а на нём стоят и разбивка по корзинам, и «только DEC». ⇒ Тот же счёт повторяется на
        # перемешанном порядке эталонных кривых, и разброс печатается рядом с числом.
        for si_ in range(1, len(SENS)):
            perm = list(RNG[si_ - 1].permutation(np.array(ns, dtype=object)))
            tk = {tag: set(match(perm, *okk[tag])) for tag, _ in PATHS}
            for nm in ns:
                k = buck[nm]
                for tag, _ in PATHS:
                    if nm in tk[tag]:
                        SENS[si_][(k, tag)] += 1
                if nm in tk["DEC"] and nm not in tk["CUR"]:
                    SENS[si_][(k, "ТОЛЬКО DEC")] += 1
                if nm in tk["CUR"] and nm not in tk["DEC"]:
                    SENS[si_][(k, "ТОЛЬКО CUR")] += 1
    if si % 200 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

tot = Counter()
for k in ORDER:
    for kk, v in tab[k].items():
        tot[kk] += v

print("\n★★★ СВЕРКА С ИЗВЕСТНЫМИ ЧИСЛАМИ (без неё таблица ниже недействительна)")
EXP = {"P965": (965, "§6.196"), "DEC": (934, "§6.175"), "CUR": (1107, "§6.188")}
bad = 0
for tag, (exp, src) in EXP.items():
    got = tot[tag]
    flag = "OK" if abs(got - exp) <= 2 else "⛔ РАСХОЖДЕНИЕ"
    bad += flag != "OK"
    print(f"  {tag}: получено {got}, ожидалось {exp} ({src}) — {flag}")
print(f"  кривых честного поля: {tot['n']} (§6.196: 2737)")
# ★ ОБЯЗАТЕЛЬНЫЙ МИНИМУМ, правило 2 (STANDS.md): объём — из счётчика, и сверка со списком.
_sk = miss["нет пула/исходника"]
_empty = sum(v for k, v in miss.items() if k.startswith("⛔"))
_ok = (len(done) + _sk) == len(sheets) and _empty == 0
print(f"  СВЕРКА СПИСКА: обработано {len(done)} + пропущено {_sk} = {len(done) + _sk} "
      f"против длины списка {len(sheets)}, листов с пустой выдачей {_empty}"
      f"   {'★ СОШЛОСЬ' if _ok else '⛔ НЕ СОШЛОСЬ'}")
for k, v in sorted(miss.items(), key=lambda q: -q[1]):
    print(f"    «{k}»: {v}")
if not _ok:
    bad += 1
if bad:
    print("⛔⛔ СВЕРКА НЕ СОШЛАСЬ — числа ниже цитировать нельзя.")

# ★ ЖЁСТКАЯ ПРОВЕРКА ОБЪЕДИНЯЮЩЕЙ КОЛОНКИ: покривой выбор не может быть ХУЖЕ потрекового.
# §6.175 намерил оракул ПО ТРЕКУ на паре (P965, DEC) = 1257 ⇒ P965∪DEC обязан быть не ниже.
print(f"  P965∪DEC: {tot['P965∪DEC']} против потрекового оракула 1257 (§6.175) — "
      f"{'OK' if tot['P965∪DEC'] >= 1257 else '⛔ НИЖЕ ПОТРЕКОВОГО: объединение считает не то'}")

print("\n★★ ИСХОД У U1 ПРОТИВ КОЛОНОК (эталонных кривых, безымянный 1:1 внутри трека)")
print("⚠ Колонки с ∪ — ОБЪЕДИНЕНИЕ КАНДИДАТОВ, эмиттером НЕ РЕАЛИЗУЕМО (в слот уходит одна кривая):"
      " реализуемый потолок по слоту считает `_route_size.py` и он НИЖЕ.")
print("⚠ ОПОРА — CUR (текущее лучшее). P965 показан для сверки с §6.196, доводом он быть не может.")
print("| исход у U1 | кривых | P965 | DEC | ★ CUR | CUR∪DEC | P965∪DEC | только DEC | только CUR |")
print("|---|---|---|---|---|---|---|---|---|")
for k in ORDER:
    n = tab[k]["n"]
    if not n:
        continue
    c = tab[k]
    print(f"| {k} | {n} | {c['P965']} | {c['DEC']} | **{c['CUR']}** | {c['CUR∪DEC']} | "
          f"{c['P965∪DEC']} | +{c['ТОЛЬКО DEC']} | **−{c['ТОЛЬКО CUR']}** |")
print(f"| **ВСЕГО** | **{tot['n']}** | {tot['P965']} | {tot['DEC']} | **{tot['CUR']}** | "
      f"{tot['CUR∪DEC']} | {tot['P965∪DEC']} | +{tot['ТОЛЬКО DEC']} | **−{tot['ТОЛЬКО CUR']}** |")

print("\n★★ ЧУВСТВИТЕЛЬНОСТЬ К ПОРЯДКУ ПАРОСОЧЕТАНИЯ (3 перемешивания эталонных кривых)")
print("| величина | естественный порядок | разброс по перемешиваниям |")
print("|---|---|---|")
for key in ("DEC", "CUR", "ТОЛЬКО DEC", "ТОЛЬКО CUR"):
    v0 = tot[key] if key in tot else sum(tab[k][key] for k in ORDER)
    vs = [sum(S[(k, key)] for k in ORDER) for S in SENS[1:]]
    print(f"| {key} | {v0} | {min(vs)}…{max(vs)} |")
bads_ = ORDER[2:]
v0 = sum(tab[k]["ТОЛЬКО DEC"] for k in bads_)
vs = [sum(S[(k, "ТОЛЬКО DEC")] for k in bads_) for S in SENS[1:]]
print(f"| «только DEC» в провальных корзинах | {v0} | {min(vs)}…{max(vs)} |")

bads = ORDER[2:]
nb = sum(tab[k]["n"] for k in bads)
print(f"\n★ ТАМ, ГДЕ ЛИНИЯ U1 ПРОМАХИВАЕТСЯ БОЛЬШЕ 3px ({nb} кривых, весь запас §6.197 = +509):")
for col in COLS:
    v = sum(tab[k][col] for k in bads)
    print(f"    {col:>8}: {v} ({100*v/max(1,nb):.1f}%)")
print(f"    ★ взято ТОЛЬКО декодером (CUR не взял): {sum(tab[k]['ТОЛЬКО DEC'] for k in bads)}")
print(f"    ⚠ и НАОБОРОТ, только CUR (декодер не взял): {sum(tab[k]['ТОЛЬКО CUR'] for k in bads)}")
print(f"    ⚠ прибавка ОБЪЕДИНЕНИЯ к текущему проду: "
      f"+{sum(tab[k]['CUR∪DEC'] for k in bads) - sum(tab[k]['CUR'] for k in bads)} — "
      f"НЕ реализуемо, см. `_route_size.py`")

if a.dump:
    pickle.dump(DUMP, open(TS / a.dump, "wb"))
    print(f"\nвыгружено {len(DUMP)} кривых → {TS / a.dump}")

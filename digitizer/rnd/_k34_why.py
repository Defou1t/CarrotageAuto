r"""_k34_why.py — ЧЕМ ИМЕННО ДЕКОДЕР ТЕРЯЕТ 333 КРИВЫЕ НА ТРЕКАХ K = 3-4 (§6.187 → B1).

ОТКУДА ВОПРОС. §6.187 показал, что на K = 3-4 декодер стоит на 51% своего потолка: 333 кривые
доступны ПРИ НЫНЕШНЕМ переднем плане и не берутся. Это самый большой резерв ветки. Но «не берутся»
— не диагноз: чинить нечего, пока не названо, ЧТО именно ломается. Три разных дефекта требуют трёх
разных правок, и перепутать их дорого:

  ★ НЕ ПОКРЫТА  — вдоль эталона не выдано ни одной достаточно длинной трассы (cov < 0.9).
                  Лечится ведением/длиной, а не точностью.
  ★ УВЕДЕНО     — трасса есть и длинная, но med > 3px. Лечится точностью, а не покрытием.
  ★ ОТНЯТО      — годная трасса есть, но её забрала ДРУГАЯ кривая трека в 1:1. Значит декодер
                  выдал РАЗЛИЧИМЫХ кривых меньше, чем их на треке: дефект РАЗДЕЛЕНИЯ пучка.

⚠ Считается в БЕЗЫМЯННОЙ рамке (максимальное 1:1 внутри трека) — ведущий счёт ветки (§6.143).
Поэтому «перепутал подписи» здесь не бывает по построению: его уже съело 1:1. Остаётся ровно то,
что 1:1 съесть не может.

★ СЧЁТА НЕ ТРАТИТ: читает ЗАМОРОЖЕННЫЕ выдачи (`ab_wellmap/A` — прод, `ab_rdhonest/B` — честный
декодер одним путём) и сравнивает их на ОДНИХ И ТЕХ ЖЕ треках. Механика 1:1 и `err` скопированы из
`_name_cost_prod.py` дословно: считать надо тем же, чем считается метрика, иначе разложение
объясняет чужое число.

  <ComfyUI>\python_embeded\python.exe _k34_why.py
  … --lo 3 --hi 4        # другая корзина K (5 999 = «5+»)
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--prod", default="ab_wellmap/A", help="выдача прода")
ap.add_argument("--dec", default="ab_rdhonest/B", help="выдача декодера одним путём (честная)")
ap.add_argument("--rows", default="pick_learn_honest2.pkl.dump")
ap.add_argument("--map", default="slotmap.pkl", help="кэш «лист → {кривая: трек}» из _paleink_cause")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--lo", type=int, default=3)
ap.add_argument("--hi", type=int, default=4)
ap.add_argument("--cap", type=int, default=0)
a = ap.parse_args()
TS = Path(a.ts)

HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9        # ровно как в _name_cost_prod


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


def match(rows, cols, ok):
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


# ─────────────────────────────────────────────────────────────────── K по трекам и карта треков
trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]): int(r[3]["n_prod"]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
print(f"★ ТРЕКИ: {len(trk)} на {len({r[0] for r in trk})} листах; карта кривых на {len(smap)} листах")

SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q
BYDIR = {f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}": nm
         for nm, q in SRC.items()}


def why(moddir):
    """→ Counter причин по кривым корзины K и число взятых."""
    MD = Path(a.ts) / moddir
    dirs = sorted([d for d in MD.iterdir() if d.is_dir()]) if MD.is_dir() else []
    if a.cap:
        dirs = dirs[:a.cap]
    c, sheets, tracks = Counter(), 0, 0
    for d in dirs:
        nm = BYDIR.get(d.name)
        if nm is None:
            c["⚠ каталог не опознан"] += 1
            continue
        got = next(iter(sorted(d.glob("*_auto.nlgx"))), None)
        if got is None:
            c["⚠ нет файла выдачи"] += 1
            continue
        gts = {x["name"]: dense(x) for x in extract(str(SRC[nm]))["curves"]
               if M.mnem_root(x["name"]) != "DA" and sum(1 for v in x["xs"] if v != NULL) >= 50}
        wr = {x["name"]: dense(x) for x in extract(str(got))["curves"]
              if M.mnem_root(x["name"]) != "DA"}
        tm = smap.get(nm, {})
        sheets += 1
        for t in {tm.get(k) for k in gts} | {tm.get(k) for k in wr}:
            if t is None or not (a.lo <= KOF.get((nm, t), -1) <= a.hi):
                continue
            G = [k for k in gts if tm.get(k) == t]
            W = [k for k in wr if tm.get(k) == t]
            if not G:
                continue
            tracks += 1
            if not W:
                c["не покрыта"] += len(G)          # выдачи на треке нет вовсе — это крайний случай
                continue
            st = {(g, w): err(wr[w], gts[g]) for g in G for w in W}
            ok = {k: HON(*v) for k, v in st.items()}
            taken = set(match(G, W, ok))
            for g in G:
                if g in taken:
                    c["★ взята"] += 1
                    continue
                # ⚠ ПОРЯДОК РАЗБОРА ВАЖЕН. Сначала спрашиваем, была ли ГОДНАЯ трасса вообще: если
                #   была, но досталась соседке — это дефект РАЗДЕЛЕНИЯ, а не ведения, и лечится он
                #   совсем другим. Только потом смотрим на лучшее покрытие и лучшую точность.
                if any(ok[(g, w)] for w in W):
                    c["отнято конкуренцией"] += 1
                    continue
                covs = [st[(g, w)][1] for w in W]
                meds = [st[(g, w)][0] for w in W if st[(g, w)][0] is not None
                        and st[(g, w)][1] >= 0.9]
                if max(covs) < 0.9:
                    c["не покрыта"] += 1
                elif meds:
                    c["уведено"] += 1
                    # ⚠⚠ БЕЗ ЭТОГО РАЗБИЕНИЯ ВЫВОД БУДЕТ ЛОЖНЫМ. Трасса, ушедшая на СОСЕДНЮЮ линию
                    # пучка, покрывает эталон так же полно, как своя, и садится в ту же корзину,
                    # что промах на 4px. Но это разные дефекты: первое — РАЗДЕЛЕНИЕ (декодер ведёт
                    # не ту линию), второе — ТОЧНОСТЬ. Величина промаха их и различает.
                    # ★ Полоса 3-10px отдельно: §6.119 закрыл ВЕСЬ класс постфильтров на ней —
                    #   если недобор сидит там, дешёвого лечения у него нет и это надо знать сразу.
                    m = min(meds)
                    b = ("3-10px (§6.119: класс закрыт)" if m <= 10 else
                         "10-30px" if m <= 30 else "30-100px" if m <= 100 else ">100px")
                    c["   ↳ " + b] += 1
                    # ★★ «>100px» ЕЩЁ НЕ ДИАГНОЗ, А ДВА РАЗНЫХ ДЕФЕКТА. Трасса могла уйти на
                    # СОСЕДНЮЮ кривую того же трека (дефект РАЗДЕЛЕНИЯ: линию ведёт, но не ту) —
                    # или не совпасть ни с одной экспертной кривой вовсе (дефект ВЫБОРА: ведёт
                    # рамку, сетку, шум). Лечится это разным, и различить их ничего не стоит:
                    # берём ту самую трассу и спрашиваем, честна ли она для КАКОЙ-НИБУДЬ кривой.
                    if m > 100:
                        best_w = min((w for w in W if st[(g, w)][0] is not None
                                      and st[(g, w)][1] >= 0.9), key=lambda w: st[(g, w)][0])
                        if any(ok[(g2, best_w)] for g2 in G if g2 != g):
                            c["      ↳↳ ведёт СОСЕДНЮЮ кривую трека"] += 1
                        # ★★★ ПРОВЕРКА ВНУТРИ ТРЕКА НЕДОСТАТОЧНА, И ЭТО НЕ ПРИДИРКА. Лист состоит
                        # из НЕСКОЛЬКИХ дорожек; трасса, ушедшая в соседний трек, тоже «не совпала
                        # ни с одной кривой ЭТОГО трека» и села бы в корзину «шум». Между тем это
                        # совсем другой дефект: уход за границу лечится границами, а не моделью.
                        # ⇒ Спрашиваем ВЕСЬ лист, прежде чем говорить «ведёт не кривую».
                        elif any(HON(*err(wr[best_w], gts[g2])) for g2 in gts if g2 not in G):
                            c["      ↳↳ ушёл в ЧУЖОЙ ТРЕК"] += 1
                        else:
                            # ★★★ ПОСЛЕДНЕЕ РАЗЛИЧЕНИЕ, И ДЛЯ ПОСТРОЧНОГО ДЕКОДЕРА — ГЛАВНОЕ.
                            # Трасса может идти ПО ЛИНИЯМ и при этом не быть честной ни для кого:
                            # сверху ведёт одну кривую пучка, снизу другую. Такая СКЛЕЙКА не
                            # ловится ни проверкой соседа (для соседа она тоже нечестна), ни
                            # проверкой чужого трека. Отличаем её от настоящего шума двумя
                            # признаками: доля строк, где трасса лежит на КАКОЙ-НИБУДЬ кривой
                            # трека (±3px), и число кривых, которым она принадлежит по строкам.
                            t = wr[best_w]
                            ys = [y for y in t if any(y in gts[g2] for g2 in G)]
                            on, own = 0, Counter()
                            for y in ys:
                                cand = [(abs(t[y] - gts[g2][y]), g2) for g2 in G if y in gts[g2]]
                                d0, g0 = min(cand)
                                if d0 <= 3.0:
                                    on += 1
                                    own[g0] += 1
                            share = on / max(1, len(ys))
                            many = sum(1 for v in own.values() if v >= 0.2 * max(1, on)) >= 2
                            if share >= 0.8 and many:
                                c["      ↳↳ СКЛЕЙКА: меняет линию по глубине"] += 1
                            elif share >= 0.8:
                                c["      ↳↳ на линии, но систематически смещена"] += 1
                            else:
                                c["      ↳↳ вне линий (рамка/сетка/шум)"] += 1
    return c, sheets, tracks


print(f"\n★★ КОРЗИНА K = {a.lo}-{a.hi}. Разбираю ЗАМОРОЖЕННЫЕ выдачи, счёта не трачу.")
res = {}
for tag, moddir in (("ПРОД", a.prod), ("ДЕКОДЕР", a.dec)):
    c, sheets, tracks = why(moddir)
    res[tag] = c
    n = sum(v for k, v in c.items() if not k.startswith(" "))
    print(f"   {tag} ({moddir}): листов {sheets}, треков корзины {tracks}, кривых {n}")

KEYS = ["★ взята", "не покрыта", "уведено",
        "   ↳ 3-10px (§6.119: класс закрыт)", "   ↳ 10-30px", "   ↳ 30-100px", "   ↳ >100px",
        "      ↳↳ ведёт СОСЕДНЮЮ кривую трека", "      ↳↳ ушёл в ЧУЖОЙ ТРЕК",
        "      ↳↳ СКЛЕЙКА: меняет линию по глубине",
        "      ↳↳ на линии, но систематически смещена",
        "      ↳↳ вне линий (рамка/сетка/шум)",
        "отнято конкуренцией", "⚠ прочее", "⚠ каталог не опознан", "⚠ нет файла выдачи"]
# ⚠ Подкорзины «↳» — РАЗБИЕНИЕ «уведено», а не отдельные кривые. Складывать их со строкой-родителем
#   нельзя: печать «кривых N» на такой сумме врала (1079 против 688 на том же поле).
SUB = tuple(k for k in KEYS if k.startswith(" "))
print(f"\n| причина | ПРОД | ДЕКОДЕР | разница |")
for k in KEYS:
    p, d = res["ПРОД"][k], res["ДЕКОДЕР"][k]
    if not p and not d:
        continue
    print(f"| {k} | {p} | {d} | **{d-p:+d}** |")

miss_d = sum(res["ДЕКОДЕР"][k] for k in ("не покрыта", "уведено", "отнято конкуренцией"))
print(f"\n★ У ДЕКОДЕРА НЕ ВЗЯТО {miss_d} кривых. Из них:")
for k in ("не покрыта", "уведено", "отнято конкуренцией"):
    v = res["ДЕКОДЕР"][k]
    print(f"   {k}: {v} ({100*v/max(1,miss_d):.0f}%)")
print("★ Самая крупная доля и есть адрес правки B1 — три причины лечатся РАЗНЫМ, "
      "и выбирать надо по числу, а не по правдоподобию.")

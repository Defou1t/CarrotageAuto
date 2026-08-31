r"""_slot_prod_cause.py — РАЗЛОЖЕНИЕ ПОТЕРЬ ОТБОРА ПОД ТОЙ КОНФИГУРАЦИЕЙ, ЧТО СТОИТ В ПРОДЕ.

⚠⚠ ЗАЧЕМ. Число, которым ветка описывает своё узкое место — «82% потерь даёт ключ порядка
`x_center`» (§6.109) — посчитано `_slot_cause.py` для РАСКЛАДКИ ПРАВИЛОМ. А с 10.08 в проде стоит
ОБУЧЕННАЯ раскладка (`slot_model_g250` + `frac0.2`, §6.108), и пулы собираются с `slot_model=""`
намеренно, чтобы дампы до и после были сравнимы. ⇒ Опорное разложение описывает конфигурацию,
которой в проде больше нет, и никто не мерил, во что оно превратилось.

ВОПРОСЫ:
  1. сколько кривых записано честно ПОД ПРОД-КОНФИГУРАЦИЕЙ против 832 у правила (тот же корпус);
  2. насколько усохла корзина отбора (у правила 1159 кривых, 16.9%);
  3. ★ из чего теперь состоит остаток: лист ОТКАЗАН гейтом (⇒ писало правило) или модель ВЫБРАЛА
     другую линию;
  4. ★★ когда модель промахнулась — насколько: каким по скору был честный кандидат в своём слоте и
     каков разрыв до победителя. Если честный обычно второй с малым отрывом, лечится признаками и
     калибровкой; если он далеко внизу, у ранжировщика нет сигнала, и это другая работа.

МЕТОД. Прод-функция `slot_model.assign` применяется ОФЛАЙН по сохранённым дампам — ровно как в
`_slot_parity.py` (там же проверено, что признаки прода совпадают с обучающей матрицей, max|Δ|=0).
Отказ гейта воспроизводится как в проде: `assign` вернул None ⇒ лист пишет ПРАВИЛО, то есть берётся
`written` из дампа.
⚠ Это ПУЛОВЫЙ замер, не отгрузка: `_pool_oracle` берёт трассу в момент раскладки, а выданный файл
проходит ещё уровни и ветку масштаба (§6.108: разница ~4.5 п.п.). Цитировать как «на раскладке».

  python _slot_prod_cause.py [--model slot_model_g250.npz] [--gate frac0.2]
"""
import sys, argparse, pickle, collections
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from auto import slot_model as SM

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--model", default="slot_model_g250.npz", help="как в `auto/config.py`")
ap.add_argument("--gate", default="frac0.2", help="как в `auto/config.py`")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    """Дословно из `_accuracy_status.py` — на нём стоит разложение §6.104/§6.111."""
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


class FakeLine:
    """Ровно то, что прод-код читает у Line (как в `_slot_parity.py`)."""
    __slots__ = ("color", "behavior", "x_center")

    def __init__(self, color, behavior, x_center):
        self.color, self.behavior, self.x_center = color, behavior, x_center


w = SM.load(a.model)
kind, thr = SM._parse_gate(a.gate)
print(f"вес {a.model}, гейт {kind}<{thr}")

CAUSE = ["1 нет слота / нет кандидатов", "2 честной трассы в пуле нет",
         "3 ★ ОТБОР: лист отказан гейтом (писало правило)",
         "4 ★ ОТБОР: модель выбрала другую линию", "5 ★★ записана честно"]
cnt = collections.Counter()
rule_hon = 0                      # сколько было бы честных, если бы писало правило (контроль 832)
free_hon = 0                      # сколько было бы, если бы модель писала ВСЕГДА (гейт выключен)
GATE = collections.Counter()      # цена гейта на отказанных листах: правило против модели
rank_of_honest, gap_to_winner = [], []
nsheet = nabstain = ndup = nfile = 0
seen = set()

for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        nfile += 1
        if f.stem in seen:
            ndup += 1; continue
        seen.add(f.stem)
        if a.limit and nsheet >= a.limit:
            continue
        d = pickle.load(open(f, "rb"))
        nsheet += 1
        gts = d["gts"]
        by_track, lidx = {}, {}
        for i, ln in enumerate(d["lines"]):
            L = FakeLine(ln["color"], ln["behavior"], ln["x_center"])
            lidx[id(L)] = i
            by_track.setdefault(ln["track"], []).append((L, ln["tr"]))
        slots_all = [dict(name=s["name"], track=s["track"], color=s["color"], cls=s["class"])
                     for s in d["slots"]]
        tl = {}
        for i, ln in enumerate(d["lines"]):
            tl.setdefault(ln["track"], []).append(i)
        strk = {s["name"]: s["track"] for s in d["slots"]}

        got = SM.assign(slots_all, by_track, w, kind, thr) if slots_all else None
        abstain = got is None
        nabstain += abstain
        # ★ ЧТО МОДЕЛЬ НАПИСАЛА БЫ БЕЗ ГЕЙТА. `frac0.0` пропускает лист всегда (§6.108 мерил его как
        # крайнюю точку свипа), поэтому это ровно раскладка ДО отказа — и цена гейта считается прямо,
        # а не свипом порога: на отказанных листах сравниваются правило и модель на одних данных.
        free = SM.assign(slots_all, by_track, w, "frac", 0.0) if slots_all else None
        # скоры нужны отдельно: `assign` отдаёт только индекс выбранной пары
        sc, pairs = None, None
        if not abstain:
            X, pairs = SM.rows(slots_all, by_track)
            if len(X):
                sc = SM.predict(w, X)

        for nm, gt in gts.items():
            r_ok = bool(nm in d["written"] and HON(*err(d["written"][nm], gt)))
            f_ok = bool(free and nm in free and HON(*err(free[nm][1], gt)))
            rule_hon += r_ok
            free_hon += f_ok
            if abstain:                       # ★ цена гейта считается ТОЛЬКО на отказанных листах
                GATE["правило"] += r_ok
                GATE["модель без гейта"] += f_ok
                GATE["кривых"] += 1
            trk = strk.get(nm)
            cand = tl.get(trk, []) if trk is not None else []
            if trk is None or not cand:
                cnt[CAUSE[0]] += 1; continue
            # ⚠⚠ ОТКАТ К ПРАВИЛУ — ПОЛИСТНЫЙ, А НЕ ПОСЛОТНЫЙ. `emit.py:274-279`: если модель лист
            # ПРИНЯЛА, её словарь заменяет правило ЦЕЛИКОМ, и слот, которому она ничего не
            # назначила, остаётся ПУСТЫМ. Первая редакция подставляла такому слоту трассу правила и
            # тем завышала прод на 13 кривых (1002 против верных 989) — стенд приписывал проду
            # поведение, которого у него нет.
            if abstain:
                wr = d["written"].get(nm)
            else:
                wr = got[nm][1] if nm in got else None
            if wr is not None and HON(*err(wr, gt)):
                cnt[CAUSE[4]] += 1; continue
            # честный кандидат на треке есть?
            hon_idx = [i for i in cand if HON(*err(d["lines"][i]["tr"], gt))]
            if not hon_idx:
                cnt[CAUSE[1]] += 1; continue
            if abstain:
                cnt[CAUSE[2]] += 1; continue
            cnt[CAUSE[3]] += 1
            # ★ насколько модель промахнулась: место честного кандидата по скору в СВОЁМ слоте
            if sc is None:
                continue
            ks = [k for k, (pnm, _, _) in enumerate(pairs) if pnm == nm]
            if not ks:
                continue
            hset = {id(d["lines"][i]["tr"]) for i in hon_idx}
            kh = [k for k in ks if id(pairs[k][2]) in hset]
            if not kh:
                continue
            order = sorted(ks, key=lambda q: -sc[q])
            best_h = max(kh, key=lambda q: sc[q])
            rank_of_honest.append(order.index(best_h) + 1)
            gap_to_winner.append(float(sc[order[0]] - sc[best_h]))

W = 96
tot = sum(cnt.values())
print(f"{'='*W}")
print(f"ВЫБОРКА: файлов {nfile}, прочитано {nsheet}, дублей {ndup}"
      + (f"  ⚠ ОГРАНИЧЕНО --limit {a.limit}" if a.limit else ""))
ok = (nsheet + ndup == nfile) or bool(a.limit)
print(f"  СВЕРКА СПИСКА: {nsheet} + {ndup} = {nsheet+ndup} против {nfile}   "
      + ("★ СОШЛОСЬ" if ok else "⛔ РАСХОЖДЕНИЕ"))
print(f"  экспертных кривых {tot}; листов отказано гейтом {nabstain} "
      f"({100*nabstain/max(1,nsheet):.1f}%) ⇒ там писало правило")
print(f"{'='*W}")
print(f"★ КОНТРОЛЬ: честных ПРАВИЛОМ на этой же выборке {rule_hon} "
      f"(§6.111 на полном корпусе: 832)")
print(f"\n{'причина':<48}{'кривых':>8}{'доля':>8}")
for c in CAUSE:
    print(f"  {c:<46}{cnt[c]:>8}{100*cnt[c]/max(1,tot):>7.1f}%")
sel = cnt[CAUSE[2]] + cnt[CAUSE[3]]
print(f"\n★★ ПОД ПРОД-КОНФИГУРАЦИЕЙ: честных {cnt[CAUSE[4]]} "
      f"({100*cnt[CAUSE[4]]/max(1,tot):.1f}%) против {rule_hon} у правила "
      f"({100*rule_hon/max(1,tot):.1f}%) ⇒ {cnt[CAUSE[4]]-rule_hon:+d}")
print(f"★★ КОРЗИНА ОТБОРА: {sel} кривых ({100*sel/max(1,tot):.1f}%), из них "
      f"{cnt[CAUSE[2]]} по отказу гейта и {cnt[CAUSE[3]]} по выбору модели")

print(f"\n{'='*W}\n★★ ЦЕНА ГЕЙТА: что было бы на ОТКАЗАННЫХ листах ({GATE['кривых']} кривых)\n{'='*W}")
print(f"  писало правило (так и есть сейчас):  {GATE['правило']:>5}")
print(f"  написала бы модель без гейта:        {GATE['модель без гейта']:>5}   "
      f"⇒ {GATE['модель без гейта']-GATE['правило']:+d}")
print(f"\nПО ВСЕЙ ВЫБОРКЕ: правило {rule_hon}, прод (гейт {a.gate}) {cnt[CAUSE[4]]}, "
      f"модель всегда {free_hon}")
print("⇒ " + ("★ ГЕЙТ ОКУПАЕТСЯ: без него было бы хуже" if free_hon < cnt[CAUSE[4]] else
              "⚠⚠ ГЕЙТ СТОИТ ДОРОЖЕ, ЧЕМ ДАЁТ: модель без него пишет больше честных — "
              "порог подобран на втрое меньшем наборе (§6.108), проверить свипом на корпусе"))

if rank_of_honest:
    r = np.array(rank_of_honest); g = np.array(gap_to_winner)
    print(f"\n{'='*W}\nКОГДА МОДЕЛЬ ПРОМАХНУЛАСЬ: место честного кандидата в своём слоте "
          f"({len(r)} случаев)\n{'='*W}")
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10**6)):
        n = int(((r >= lo) & (r <= hi)).sum())
        lab = f"{lo}-е" if lo == hi else (f">{lo-1}-го" if hi > 10**5 else f"{lo}-{hi}-е")
        print(f"  {lab:<10}{n:>8}{100*n/len(r):>7.1f}%  " + "█" * int(44 * n / len(r)))
    print(f"  ⚠ «1-е место» = честный кандидат победил по скору, но слот занял другой: "
          f"это жадное 1:1, а не ошибка ранжировки")
    print(f"\nРАЗРЫВ СКОРА до победителя: медиана {np.median(g):.2f}, "
          f"25-й {np.percentile(g,25):.2f}, 75-й {np.percentile(g,75):.2f}")
    near = int((g <= 0.3).sum())
    print(f"  честный отстал меньше чем на 0.3 (порог margin в гейте `frac`): {near} "
          f"({100*near/len(g):.1f}%)")
    print("⇒ " + ("★ БОЛЬШИНСТВО ПРОМАХОВ — БЛИЗКИЕ: лечится признаками и калибровкой"
                  if np.median(g) <= 1.0 else
                  "⚠ ПРОМАХИ ДАЛЁКИЕ: у ранжировщика нет сигнала, признаками не закрыть"))

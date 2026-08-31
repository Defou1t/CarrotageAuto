r"""_slot_feat_gap.py — РАЗЛИЧИМЫ ЛИ ЧЕСТНЫЙ И ВЫБРАННЫЙ КАНДИДАТЫ В ПРИЗНАКАХ РАНЖИРОВЩИКА.

§6.115 оставил раскладке остаток в 986 кривых и закрыл к нему два подхода: гейт (§6.115) и способ
раздачи линий (§6.115.1, −1 кривая). Значит остаток держит САМ СКОР. Прежде чем учить новый вес,
стоит спросить то же, что §6.109 спросил про корзину «КЛАСС»: **а различимы ли эти линии вообще?**
Там ответ был «нет» — признаки формы у ошибочно и верно помеченных не различались (ВЧ-дрожь 0.500
против 0.530), и «подкрутить классификатор» оказалось тупиком.

ЧТО СЧИТАЕТСЯ. Для каждого промаха модели (честный кандидат на треке ЕСТЬ, а записан другой) берутся
14 признаков ОБОИХ кандидатов — честного и выбранного — прод-функцией `slot_model.rows`, и сравнивается:
  • ★ доля промахов, где векторы признаков СОВПАДАЮТ ПОЛНОСТЬЮ — это жёсткий потолок: такие пары не
    различит НИКАКАЯ модель на этих признаках, сколько её ни учи;
  • по каждому признаку — медиана у честного и у выбранного и доля случаев «честный больше»;
    0.5 значит «признак не несёт сигнала», 0.0 или 1.0 — «несёт, но модель им не пользуется».

⚠ Стенд ничего не обучает и не меняет: он отвечает, ЕСТЬ ЛИ ЧТО учить на нынешних признаках.

  python _slot_feat_gap.py [--limit N]
"""
import sys, argparse, pickle, json, collections
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
ap.add_argument("--model", default="slot_model_g250.npz")
ap.add_argument("--gate", default="frac0.2")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


class FakeLine:
    __slots__ = ("color", "behavior", "x_center")

    def __init__(self, color, behavior, x_center):
        self.color, self.behavior, self.x_center = color, behavior, x_center


w = SM.load(a.model)
kind, thr = SM._parse_gate(a.gate)
man = json.loads((Path(r"F:\nds\Auto\auto\models") / a.model.replace(".npz", ".json"))
                 .read_text(encoding="utf-8"))
FEAT = man["features"]
print(f"вес {a.model}, гейт {kind}<{thr}, признаков {len(FEAT)}")

H, C = [], []          # признаки честного и выбранного
same = 0
nfile = nsheet = ndup = 0
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
        by_track = {}
        for ln in d["lines"]:
            by_track.setdefault(ln["track"], []).append(
                (FakeLine(ln["color"], ln["behavior"], ln["x_center"]), ln["tr"]))
        slots = [dict(name=s["name"], track=s["track"], color=s["color"], cls=s["class"])
                 for s in d["slots"]]
        if not slots:
            continue
        got = SM.assign(slots, by_track, w, kind, thr)
        if got is None:                       # лист отказан ⇒ писало правило, промаха модели нет
            continue
        X, pairs = SM.rows(slots, by_track)
        if not len(X):
            continue
        for nm, gt in gts.items():
            if nm not in got:
                continue
            chosen = got[nm][1]
            if HON(*err(chosen, gt)):
                continue                      # записана честно — не промах
            ks = [k for k, (pnm, _, _) in enumerate(pairs) if pnm == nm]
            kh = [k for k in ks if HON(*err(pairs[k][2], gt))]
            if not kh:
                continue                      # честного кандидата не было — это не отбор
            kc = next((k for k in ks if pairs[k][2] is chosen), None)
            if kc is None:
                continue
            xh = X[max(kh, key=lambda q: 0)]  # любой честный: их обычно один
            xc = X[kc]
            H.append(xh); C.append(xc)
            same += bool(np.array_equal(xh, xc))

W = 96
print(f"{'='*W}")
print(f"ВЫБОРКА: файлов {nfile}, прочитано {nsheet}, дублей {ndup}"
      + (f"  ⚠ ОГРАНИЧЕНО --limit {a.limit}" if a.limit else ""))
print(f"промахов модели с честным кандидатом на треке: {len(H)}")
if H:
    Hm, Cm = np.array(H), np.array(C)
    print(f"★★ ВЕКТОРЫ ПРИЗНАКОВ СОВПАЛИ ПОЛНОСТЬЮ: {same} из {len(H)} "
          f"({100*same/len(H):.1f}%) — этих не различит никакая модель на этих признаках")
    print(f"\n{'признак':<18}{'честный':>10}{'выбранный':>12}{'честный>':>10}   сигнал")
    for i, nm in enumerate(FEAT):
        h, c = Hm[:, i], Cm[:, i]
        gt_share = float(np.mean(h > c) + 0.5 * np.mean(h == c))
        sig = "—" if abs(gt_share - 0.5) < 0.05 else ("★ есть" if abs(gt_share - 0.5) > 0.15 else "слабый")
        print(f"  {nm:<16}{np.median(h):>10.3f}{np.median(c):>12.3f}{gt_share:>10.2f}   {sig}")
    print("\n⇒ колонка «честный>» — доля промахов, где у честного признак БОЛЬШЕ (0.5 = признак не "
          "различает).\n  Если различающих признаков нет вовсе — переобучение веса бессмысленно, "
          "нужны ДРУГИЕ признаки (урок §6.109 про корзину КЛАСС).")

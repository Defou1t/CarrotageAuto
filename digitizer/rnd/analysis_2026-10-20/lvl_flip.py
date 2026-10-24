"""Перестановка меток уровней кривой по непрерывности значения на стыках пробегов (24.10, §6.343 разведка).
Для каждой кривой, честной в пикселях по имени: пробеги уровней выдачи; на каждом стыке — значение трассы по ≤ 15 строкам до и после
при уровне пробега; цена — сумма |Δv| / ширина шкалы уровня 0. Лучшая перестановка меток (K ≤ 4) против тождественной; если она
дешевле в `--ratio` раз — переставить. Счёт: честных в значениях до / после, по наборам.
lvl_flip.py <каталог режима> [ratio …]"""
import sys, pickle, hashlib, re, json, itertools
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
sys.argv, _argv = sys.argv[:1] + ["F:/nds/output/taskS/rp_lvl/__none__", "nul"], sys.argv
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M, level_ink as LI
import decode_levels as DL
src = open(Path(__file__).with_name("lvl_loss.py"), encoding="utf-8").read()
exec(src[:src.index("HON = lambda")])                    # tree, err, lv_rows, relevel, SRC, HLD
HON = lambda e: e[0] <= 3.0 and e[1] >= 0.9
MD = Path(_argv[1]); RAT = [float(r) for r in _argv[2:]] or [1.5, 2.0, 3.0]
TS = Path(r"F:/nds/output/taskS")
lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
FLD = lst("wellmap_sheets.txt")
EVO = set(json.loads((TS / "line_choice_v6e" / "line_choice_folds.json").read_text(encoding="utf-8"))) - FLD


def val(ch, l, x):
    s = ch[l]
    return s["v_left"] + (x - s["x_left"]) * (s["v_right"] - s["v_left"]) / ((s["x_right"] - s["x_left"]) or 1)


def cost(w, lw, ch, pm, W=15):
    ys = np.array(sorted(w)); L = np.array([lw.get(int(y), 0) for y in ys])
    sw = np.flatnonzero(np.diff(L) != 0)
    span = abs(ch[0]["v_right"] - ch[0]["v_left"]) or 1.0
    c = 0.0
    for i in sw:
        a = ys[max(0, i - W + 1):i + 1]; b = ys[i + 1:i + 1 + W]
        la, lb = pm[L[i]], pm[L[i + 1]]
        if la >= len(ch) or lb >= len(ch):
            return np.inf
        va = np.median([val(ch, la, w[int(y)]) for y in a]); vb = np.median([val(ch, lb, w[int(y)]) for y in b])
        c += abs(va - vb) / span
    return c, len(sw)


R = {r: Counter() for r in RAT}; N = Counter()
for d in sorted(x for x in MD.iterdir() if x.is_dir()):
    q = SRC.get(d.name)
    got = next(iter(sorted(d.glob("*_auto.nlgx"))), None)
    if q is None or got is None:
        continue
    sets = [n for n, S in (("поле", FLD), ("вне", EVO), ("сорт A", HLD)) if q.name in S]
    if not sets:
        continue
    mt = extract(str(q)); mw = extract(got.as_posix())
    gts = {c["name"]: c for c in mt["curves"] if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    wr = {c["name"]: c for c in mw["curves"] if M.mnem_root(c["name"]) != "DA"}
    for n, cg in gts.items():
        if n not in wr:
            continue
        gt = dense(cg); w = dense(wr[n])
        if not w or not HON(err(w, gt)):
            continue
        ch = DL.build_family(mt, cg)
        lg = lv_rows(cg); lw = lv_rows(wr[n])
        h0 = HON(err(relevel(w, lw, lg, ch), gt))
        for s_ in sets:
            N[(s_, "base")] += h0
        if len(ch) < 2 or LI.is_shift(ch):
            continue
        used = sorted({lw.get(int(y), 0) for y in w})
        if len(used) < 2:
            continue
        K = min(len(ch), 4)
        if max(used) >= K:
            continue
        c0 = cost(w, lw, ch, tuple(range(K)))
        if c0 == np.inf:
            continue
        best = None
        for pm in itertools.permutations(range(K)):
            if pm == tuple(range(K)):
                continue
            c1 = cost(w, lw, ch, pm)
            if c1 != np.inf and (best is None or c1[0] < best[0][0]):
                best = (c1, pm)
        if best is None:
            continue
        lw2 = {y: best[1][l] for y, l in lw.items()}
        h1 = None
        for r in RAT:
            if best[0][0] * r < c0[0]:
                if h1 is None:
                    h1 = HON(err(relevel(w, lw2, lg, ch), gt))
                for s_ in sets:
                    R[r][(s_, "flip")] += 1; R[r][(s_, "gain")] += int(h1) - int(h0)
for s_ in ("поле", "вне", "сорт A"):
    print(f"★ {s_}: честных в значениях (из честных в пикселях) {N[(s_, 'base')]}")
    for r in RAT:
        print(f"    ratio {r}: перестановок {R[r][(s_, 'flip')]}, Δ {R[r][(s_, 'gain')]:+d}")

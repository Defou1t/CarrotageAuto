"""Границы уровня без скачка трассы: эталон и выдача NSE8T; правило «слить границу без скачка x» — пересчёт именных в значениях.
lvl_prune.py <каталог режима> <out.pkl> [jump_frac] [win]"""
import sys, pickle, hashlib, re
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M, level_ink as LI
import decode_levels as DL
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lvl_common import err, lv_rows, relevel, HON                # мера plane2 и пересчёт уровней — как `_name_cost_prod.py`

MD = Path(sys.argv[1]); JF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3; WIN = int(sys.argv[4]) if len(sys.argv) > 4 else 8
MINRUN = int(sys.argv[5]) if len(sys.argv) > 5 else 0
TS = Path(r"F:/nds/output/taskS")
lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
HLD = lst("holdoutA_sheets.txt"); FLD = lst("wellmap_sheets.txt")
SRC = {f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}": q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}


def has_jump(tr, y, wd):
    """скачок x ≥ JF·ширины за ≤ WIN строк в окне ±WIN у строки y"""
    xs = [tr[yy] for yy in range(y - WIN, y + WIN + 1) if yy in tr]
    return len(xs) >= 2 and (max(xs) - min(xs)) >= JF * wd


def prune(segs, tr, wd):
    """слить соседние сегменты, если на границе нет скачка x; уровень слитого — уровень более длинного"""
    S = [list(s) for s in segs]
    changed = True
    while changed and len(S) > 1:
        changed = False
        for i in range(1, len(S)):
            if S[i][2] != S[i - 1][2] and ((MINRUN == 0 and not has_jump(tr, S[i][0], wd)) or
                                           (MINRUN > 0 and min(S[i][1] - S[i][0], S[i - 1][1] - S[i - 1][0]) < MINRUN)):
                a_, b_ = S[i - 1], S[i]
                lv = a_[2] if (a_[1] - a_[0]) >= (b_[1] - b_[0]) else b_[2]
                S[i - 1:i + 1] = [[a_[0], b_[1], lv]]
                changed = True
                break
    return S


G = Counter(); B = Counter(); D = Counter(); PER = {}
for d in sorted(x for x in MD.iterdir() if x.is_dir()):
    q = SRC.get(d.name)
    got = next(iter(sorted(d.glob("*_auto.nlgx"))), None)
    if q is None or got is None:
        continue
    grp = "A" if q.name in HLD else ("поле" if q.name in FLD else "вне")
    mt = extract(str(q)); mw = extract(got.as_posix())
    gts = {c["name"]: c for c in mt["curves"] if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    wr = {c["name"]: c for c in mw["curves"] if M.mnem_root(c["name"]) != "DA"}
    dd = 0
    for n, cg in gts.items():
        if n not in wr:
            continue
        ch = DL.build_family(mt, cg)
        if len(ch) < 2:
            continue
        gt = dense(cg); w = dense(wr[n])
        if not w:
            continue
        wd = abs(ch[0]["x_right"] - ch[0]["x_left"]) or 1
        # границы эталона: со скачком или без
        for s in (cg.get("segments") or [])[1:]:
            B[("эталон", has_jump(gt, s[0], wd))] += 1
        ws = wr[n].get("segments") or []
        for s in ws[1:]:
            B[("выдача", has_jump(w, s[0], wd))] += 1
        lg = lv_rows(cg)
        h0 = HON(err(relevel(w, lv_rows(wr[n]), lg, ch), gt))
        ps = prune(ws, w, wd)
        h1 = HON(err(relevel(w, lv_rows({"segments": ps}), lg, ch), gt)) if len(ps) != len(ws) else h0
        G[(grp, h0, h1)] += 1; dd += int(h1) - int(h0)
    PER[q.name] = (grp, dd)
print(f"★ MINRUN {MINRUN}; {MD.name}, скачок ≥ {JF}·ширины за ±{WIN} строк")
for who in ("эталон", "выдача"):
    t = B[(who, True)] + B[(who, False)]
    print(f"   границ {who}: {t}, со скачком {B[(who, True)]} ({B[(who, True)] / max(1, t):.2f})")
for grp in ("поле", "вне", "A"):
    up = G[(grp, False, True)]; dn = G[(grp, True, False)]
    sh = [v for g_, v in PER.values() if g_ == grp]
    print(f"   {grp}: кривых с цепочкой {sum(v for k, v in G.items() if k[0] == grp)}, честных было {G[(grp, True, True)] + dn}, "
          f"стало {G[(grp, True, True)] + up} (+{up} / −{dn}); листов ↑{sum(1 for v in sh if v > 0)}/↓{sum(1 for v in sh if v < 0)}")
pickle.dump(PER, open(sys.argv[2], "wb"))

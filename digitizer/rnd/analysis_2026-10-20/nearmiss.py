"""Почти-честные (покрытие ≥ 0.9, медиана 3–6 px) против честных: систематический сдвиг или разброс? Кандидаты prod/dec из tcache."""
import sys, pickle, glob, hashlib
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from _cand_ceiling import dense_tr

TS = Path(r"F:/nds/output/taskS")
FLD = {l.strip() for l in (TS / "wellmap_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
R = {"near": [], "hon": []}
n_sh = 0
for f in sorted(glob.glob(str(TS / "line_choice_i" / "*.pkl"))):
    r = pickle.load(open(f, "rb"))
    q = SRC.get(r["sheet"])
    if r["sheet"] not in FLD or q is None:
        continue
    key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
    tcp = TS / "tcache" / (key + ".pkl")
    if not tcp.exists():
        continue
    want = []
    for s in r["slots"]:
        if not s["has_truth"]:
            continue
        labs = [(c["k"], c["lab"]) for c in s["cands"] if c["lab"] is not None and r["cands"][c["k"]]["idx"] < 1000]
        if not labs:
            continue
        hon = [x for x in labs if x[1][0]]
        if hon:
            k = min(hon, key=lambda x: x[1][2])[0]; want.append(("hon", s, k))
        elif not any(c["lab"] is not None and c["lab"][0] for c in s["cands"]):
            near = [x for x in labs if x[1][1] >= 0.9 and 3.0 < x[1][2] <= 6.0]
            if near:
                k = min(near, key=lambda x: x[1][2])[0]; want.append(("near", s, k))
    if not want:
        continue
    n_sh += 1
    c = pickle.load(open(tcp, "rb"))
    mt = {cc["name"]: cc for cc in extract(str(q))["curves"]}
    for kind, s, k in want:
        cd = r["cands"][k]
        t = (c["traces"] if cd["src"] == "prod" else c["alt"])[cd["idx"]][1]
        ct = dense_tr(un(t)); gt = dense(mt[s["name"]])
        com = np.array([y for y in gt if y in ct])
        if len(com) < 100:
            continue
        dx = np.array([ct[y] - gt[y] for y in com])
        g = np.array([gt[y] for y in com]); sl = np.abs(np.gradient(g))
        flat = sl < 0.5
        R[kind].append((float(np.median(dx)), float(np.median(np.abs(dx - np.median(dx)))), float(np.mean(np.abs(dx) <= 3)),
                        float(np.mean(flat)), float(np.median(np.abs(dx[flat]))) if flat.sum() > 30 else np.nan,
                        float(np.median(np.abs(dx[~flat]))) if (~flat).sum() > 30 else np.nan))
print("листов", n_sh)
for kind, L in R.items():
    A = np.array(L)
    print(f"★ {kind}: {len(A)} слотов; медиана знакового сдвига q10/50/90 {np.round(np.nanpercentile(A[:, 0], [10, 50, 90]), 2)}; "
          f"|сдвиг| > 2 px у {np.mean(np.abs(A[:, 0]) > 2):.2f}; разброс (MAD) медиана {np.nanmedian(A[:, 1]):.2f}; доля строк |dx| ≤ 3 — {np.nanmedian(A[:, 2]):.2f}; "
          f"доля пологих строк {np.nanmedian(A[:, 3]):.2f}; |dx| на пологих {np.nanmedian(A[:, 4]):.2f}, на крутых {np.nanmedian(A[:, 5]):.2f}")
pickle.dump(R, open(TS / "_nearmiss.pkl", "wb"))

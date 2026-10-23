"""§6.337: упростить выдачу ломаной, как оцифровывает эксперт (вершины через 6–23 строки), — пересчёт именных честных в значениях.
Варианты: dec<k> — каждая k-я строка + прямые; rdp<ε> — Дуглас–Пекер с допуском ε px. Уровни выдачи не трогаются.
simplify_whatif.py <каталог режима> <out.pkl>"""
import sys, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "F:/nds/Auto"); sys.path.insert(0, "F:/nds/Auto/digitizer"); sys.path.insert(0, "F:/nds/Auto/digitizer/rnd")
sys.path.insert(0, "F:/nds/Auto/digitizer/rnd/analysis_2026-10-20")
from pathlib import Path
import numpy as np

VARS = ["dec8", "dec15", "rdp2", "rdp4"]


def rdp_idx(y, x, eps):
    """индексы вершин Дугласа–Пекера (итеративно)"""
    keep = np.zeros(len(y), bool); keep[0] = keep[-1] = True
    st = [(0, len(y) - 1)]
    while st:
        a, b = st.pop()
        if b - a < 2:
            continue
        ya, xa, yb, xb = y[a], x[a], y[b], x[b]
        yy, xx = y[a + 1:b], x[a + 1:b]
        dy, dx = yb - ya, xb - xa
        nrm = np.hypot(dy, dx) or 1.0
        d = np.abs(dx * (yy - ya) - dy * (xx - xa)) / nrm
        i = int(np.argmax(d))
        if d[i] > eps:
            m = a + 1 + i; keep[m] = True
            st += [(a, m), (m, b)]
    return np.flatnonzero(keep)


def simplify(tr, var):
    ys = np.array(sorted(tr)); xs = np.array([tr[y] for y in ys], float)
    if len(ys) < 3:
        return tr
    out = {}
    # куски без разрывов строк (> 1) и без скачков (разворот уровня) — упрощаются отдельно
    br = np.flatnonzero((np.diff(ys) > 1) | (np.abs(np.diff(xs)) > 40)) + 1
    for seg in np.split(np.arange(len(ys)), br):
        y, x = ys[seg], xs[seg]
        if len(y) < 3:
            out.update({int(a): float(b) for a, b in zip(y, x)}); continue
        if var.startswith("dec"):
            k = int(var[3:]); iv = np.unique(np.r_[np.arange(0, len(y), k), len(y) - 1])
        else:
            iv = rdp_idx(y.astype(float), x, float(var[3:]))
        xi = np.interp(y, y[iv], x[iv])
        out.update({int(a): float(b) for a, b in zip(y, xi)})
    return out


def one(job):
    from extract_nlgx import extract, NULL
    from _multi_replica_probe import dense
    from auto import meta as M
    import decode_levels as DL
    from lvl_common import err, lv_rows, relevel, HON
    d, q = job
    got = next(iter(sorted(Path(d).glob("*_auto.nlgx"))), None)
    if got is None:
        return None
    mt = extract(str(q)); mw = extract(got.as_posix())
    gts = {c["name"]: c for c in mt["curves"] if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    wr = {c["name"]: c for c in mw["curves"] if M.mnem_root(c["name"]) != "DA"}
    cnt = {v: 0 for v in ["base"] + VARS}
    for n, cg in gts.items():
        if n not in wr:
            continue
        w = dense(wr[n])
        if not w:
            continue
        gt = dense(cg); ch = DL.build_family(mt, cg); lg = lv_rows(cg); lw = lv_rows(wr[n])
        cnt["base"] += HON(err(relevel(w, lw, lg, ch), gt))
        for v in VARS:
            cnt[v] += HON(err(relevel(simplify(w, v), lw, lg, ch), gt))
    return q.name, cnt


if __name__ == "__main__":
    MD = Path(sys.argv[1])
    TS = Path("F:/nds/output/taskS")
    SRC = {f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}": q for q in Path("F:/nds/projects/Archive").glob("*/wlg/*.nlgx")}
    jobs = [(str(d), SRC[d.name]) for d in sorted(MD.iterdir()) if d.is_dir() and d.name in SRC]
    import multiprocessing as mp
    R = {}
    with mp.Pool(4) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            if r:
                R[r[0]] = r[1]
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(jobs)}")
    pickle.dump(R, open(sys.argv[2], "wb"))
    import json
    lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
    F, H = lst("wellmap_sheets.txt"), lst("holdoutA_sheets.txt")
    E = set(json.loads((TS / "line_choice_v6e" / "line_choice_folds.json").read_text(encoding="utf-8"))) - F
    rng = np.random.default_rng(0)
    for gn, S in (("поле+вне", F | E), ("сорт A", H)):
        S = [s for s in S if s in R]
        b = sum(R[s]["base"] for s in S)
        for v in VARS:
            d_ = np.array([R[s][v] - R[s]["base"] for s in S]); nz = d_[d_ != 0]
            p = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)) >= abs(d_.sum()))) if len(nz) else 1.0
            print(f"★ {gn} {v}: именных {b} → {b + int(d_.sum())} (Δ {int(d_.sum()):+d}, листов ↑{int((d_ > 0).sum())}/↓{int((d_ < 0).sum())}, p = {p:.4f})")

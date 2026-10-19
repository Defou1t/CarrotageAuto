r"""_link_data.py — ДАННЫЕ ДЛЯ ОБУЧЕННОГО ВЕДЕНИЯ: ПЕРЕХОДЫ «ПИК СТРОКИ → ПИК СЛЕДУЮЩЕЙ» С МЕТКОЙ «ТА ЖЕ КРИВАЯ» (19.10, §6.318).

Пики карты декодера (`peaks_dev`, 299 листов поля, ≥ 0.2·max строки) размечаются по эталону листа: пик принадлежит кривой эталона,
если её x в этой строке ≤ 3 px (ближайшей; иначе — «ничей»). Для каждой пары строк (t, t + d), d ∈ 1…4, и каждого пика
строки t, принадлежащего кривой: все пики строки t + d в окне ±`--win` px — пример перехода; метка 1, если пик той же кривой.
Признаки перехода: Δx/d, |Δx|/d, наклон кривой до t (по пикам той же метки — НЕТ, по пути не знаем) → наклон по ближайшему
предыдущему пику, ближайшему к прямому продолжению (как в инерции), p обоих пиков, ранг p в строке, число пиков в окне,
расстояние эмбеддингов (если есть), d. Пишет выборку `--out` (X, y, лист) для `_link_train.py`.

  _link_data.py --workers 4
"""
import sys, argparse, pickle, glob
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
_SRC = None


def one(f):
    global _SRC
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from auto import meta as M
    if _SRC is None:
        _SRC = {q.stem[:60]: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    D = pickle.load(open(f, "rb"))
    q = _SRC.get(D["stem"][:60])
    if q is None:
        return None
    G = [dense(c) for c in extract(str(q))["curves"] if M.mnem_root(c["name"]) != "DA"]
    G = [g for g in G if len(g) >= 50]
    y0, x0 = D["y0"], D["x0"]
    rows = D["rows"]; n = len(rows)
    # метки пиков: индекс кривой эталона или -1
    LAB = []
    for i, (mx, idx, pr, em) in enumerate(rows):
        y = y0 + i; xs = idx.astype(np.int64) + x0
        lab = np.full(len(xs), -1, np.int32)
        for gi, g in enumerate(G):
            gx = g.get(y)
            if gx is None:
                continue
            d = np.abs(xs - gx)
            j = int(np.argmin(d)) if len(d) else -1
            if j >= 0 and d[j] <= 3:
                lab[j] = gi
        LAB.append(lab)
    X, Y = [], []
    rng = np.random.default_rng(0)
    for t in range(1, n - 4, A.stride):
        mx, idx, pr, em = rows[t]
        for a_ in range(len(idx)):
            if LAB[t][a_] < 0:
                continue
            xa = float(idx[a_])
            # наклон по предыдущей строке: пик строки t-1, ближайший к xa
            pidx = rows[t - 1][1]
            if len(pidx):
                xp = float(pidx[int(np.argmin(np.abs(pidx.astype(float) - xa)))]); v = xa - xp
            else:
                v = 0.0
            for d in (1, 2, 3, 4):
                mx2, idx2, pr2, em2 = rows[t + d]
                if not len(idx2):
                    continue
                dx = idx2.astype(float) - xa
                m = np.abs(dx) <= A.win
                if not m.any():
                    continue
                rk = np.argsort(np.argsort(-pr2.astype(float)))
                for b_ in np.flatnonzero(m):
                    ed = float(np.sqrt(((em[a_].astype(np.float32) - em2[b_].astype(np.float32)) ** 2).sum())) if (em is not None and em2 is not None) else np.nan
                    X.append([dx[b_] / d, abs(dx[b_]) / d, abs(dx[b_] / d - v), v, float(pr[a_]) / max(mx, 1e-6), float(pr2[b_]) / max(mx2, 1e-6),
                              float(rk[b_]), float(m.sum()), ed, d, float(D["k"])])
                    Y.append(1 if LAB[t + d][b_] == LAB[t][a_] else 0)
                break                                       # только ближайшая строка, где есть пики в окне
    return np.array(X, np.float32), np.array(Y, np.int8), D["stem"]


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=int, default=60)
    ap.add_argument("--stride", type=int, default=3, help="строки t — через одну-две (выборка)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=r"F:/nds/output/taskS/link_data.pkl")
    A = ap.parse_args()
    import multiprocessing as mp
    fs = sorted(glob.glob(str(TS / "peaks_dev" / "*.pkl")))
    Xs, Ys, S = [], [], []
    with mp.Pool(A.workers, initializer=_init, initargs=(A.win, A.stride)) as pool:
        for r in pool.imap_unordered(one, fs):
            if r is None or not len(r[1]):
                continue
            Xs.append(r[0]); Ys.append(r[1]); S += [r[2]] * len(r[1])
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    pickle.dump(dict(X=X, y=Y, stem=np.array(S)), open(A.out, "wb"), protocol=4)
    print(f"переходов {len(Y)}, доля «та же кривая» {Y.mean():.3f}, треков {len(fs)}")


def _init(win, stride):
    global A
    import types
    A = types.SimpleNamespace(win=win, stride=stride)


A = None
if __name__ == "__main__":
    main()

r"""_level_depth.py — §6.379 РАЗВЕДКА: дефицит ВЕРХНИХ уровней масштаба (15.11).

Разбор глазами 15.11 (лист 24, шкала 2 / 10 / 50 / 250 / 1250): эталон доходит до уровней 3–4, наша выдача — 0↔1. Модель уровня
(`auto.level_ink`, KMAX = 4) выдаёт уровни 0–3. Здесь по всем листам стенда (выдача NSE26 вне фолда): кривые эталона с цепочкой шкал
≥ 2, наша кривая того же имени честна в пикселях (медиана |dx| ≤ 3, покрытие ≥ 0.9); матрица строк «уровень эталона × наш», доля
неверных строк по уровню эталона, кривые с > 10% строк неверного уровня — по максимальному уровню эталона и по длине цепочки K.

  _level_depth.py [--mode NSE26]
"""
import sys, argparse, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
from collections import Counter
import numpy as np

TS = Path(r"F:/nds/output/taskS")


def sheet_dir(n):
    return f"{n.stem[:40]}_{hashlib.md5(n.stem.encode('utf-8')).hexdigest()[:8]}"


def lev_at(segs, rows):
    out = np.full(len(rows), -1, np.int64)
    for y0, y1, l in segs:
        out[(rows >= y0) & (rows <= y1)] = int(l)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="NSE26")
    a = ap.parse_args()
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    import decode_levels as DL
    from auto import meta as M
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    names = set()
    for f in ("tcache_sheets.txt", "outside_sheets.txt"):
        names |= {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
    C = np.zeros((6, 6), np.int64)
    bad_by_tmax, all_by_tmax, bad_by_K, all_by_K = Counter(), Counter(), Counter(), Counter()
    ncur = nsh = 0
    for nm in sorted(names):
        if nm not in SRC:
            continue
        n = SRC[nm]
        od = TS / "rp_lvl" / a.mode / sheet_dir(n)
        fs = list(od.glob("*_auto.nlgx")) if od.exists() else []
        if not fs:
            continue
        try:
            m = extract(str(n)); mo = extract(str(fs[0]))
        except Exception:
            continue
        nsh += 1
        ours = {c["name"]: c for c in mo["curves"]}
        for c in m["curves"]:
            if M.mnem_root(c["name"]) == "DA" or not c.get("segments") or c["name"] not in ours:
                continue
            ch = DL.build_family(m, c)
            if len(ch) < 2:
                continue
            oc = ours[c["name"]]
            gt, ot = dense(c), dense(oc)
            com = np.array(sorted(y for y in gt if y in ot))
            if len(com) < 200 or len(com) / len(gt) < 0.9:
                continue
            if np.median([abs(ot[y] - gt[y]) for y in com]) > 3:
                continue
            Lt = lev_at(sorted(c["segments"]), com); Lo = lev_at(sorted(oc.get("segments") or []), com)
            ok = Lt >= 0
            Lt, Lo = Lt[ok], np.where(Lo[ok] < 0, 0, Lo[ok])
            if not len(Lt):
                continue
            ncur += 1
            np.add.at(C, (np.clip(Lt, 0, 5), np.clip(Lo, 0, 5)), 1)
            tmax = int(Lt.max()); K = len(ch)
            bad = float(np.mean(Lt != Lo)) > 0.10
            all_by_tmax[min(tmax, 5)] += 1; bad_by_tmax[min(tmax, 5)] += bad
            all_by_K[min(K, 6)] += 1; bad_by_K[min(K, 6)] += bad
    print(f"★ листов {nsh}; кривых с цепочкой ≥ 2, честных в пикселях: {ncur}")
    print("   строки: уровень эталона (строка) × наш (столбец), тыс.:")
    for i in range(6):
        if C[i].sum():
            print(f"     эталон {i}: " + "  ".join(f"{C[i, j] / 1000:8.1f}" for j in range(6)) + f"   | верно {C[i, i] / C[i].sum():.1%}")
    print("   кривых с > 10% строк неверного уровня — по макс. уровню эталона:")
    for t in sorted(all_by_tmax):
        print(f"     макс. уровень {t}{'+' if t == 5 else ''}: {bad_by_tmax[t]} / {all_by_tmax[t]} = {bad_by_tmax[t] / all_by_tmax[t]:.0%}")
    print("   — по длине цепочки K:")
    for k in sorted(all_by_K):
        print(f"     K = {k}{'+' if k == 6 else ''}: {bad_by_K[k]} / {all_by_K[k]} = {bad_by_K[k] / all_by_K[k]:.0%}")


if __name__ == "__main__":
    main()

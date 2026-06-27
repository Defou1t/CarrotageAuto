r"""
_band_multi_test.py — «ДАЛЬШЕ»: по-строчный N-трек для ТЕСНОГО ПАРАЛЛЕЛЬНОГО кластера, где колоночная
плотность не разводит (горбы сливаются), но по строкам кривые почти всегда раздельны.
N параллельных треков из чистой сид-строки (ровно N прогонов) → распространение вверх/вниз эксклюзивным
назначением прогон↔трек по предсказанию (min-jerk), коаст в слияниях. Тест на KREMEN_083 NNKM/NNKB/GK.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from extract_nlgx import extract, NULL
from dataset_build import find_image

F = r"F:\nds\projects\Archive\KREMEN_083\wlg\KREMEN_083_RK, AK, DS_0800-1560_200_1998-07-08_D_1.nlgx"
PROB = r"F:\nds\output\_bandprob_KREMEN_083_RK, AK, DS_08.npy"
XA, XB = 486, 1359
SL, GATE = 22.0, 24.0


def runs_at(prob, weight, y, xa, xb, cols):
    xs = np.nonzero(prob[y, xa:xb] > 0.4)[0]
    if not len(xs):
        return []
    out = []
    for r in np.split(xs, np.nonzero(np.diff(xs) > 3)[0] + 1):
        w = weight[y, xa:xb][r]
        out.append(float((cols[r] @ w / w.sum()) if w.sum() > 0 else cols[r].mean()))
    return out


def estimate_n(prob, xa, xb, ty, by):
    c = []
    for y in range(ty, by, 3):
        xs = np.nonzero(prob[y, xa:xb] > 0.4)[0]
        if len(xs):
            c.append(len(np.split(xs, np.nonzero(np.diff(xs) > 3)[0] + 1)))
    return int(np.percentile(c, 75)) if c else 0


def trace_multi(prob, weight, xa, xb, ty, by, N):
    cols = np.arange(xa, xb)
    mid = (ty + by) // 2; seedy = None
    for off in range(0, (by - ty) // 2):                  # сид-строка ровно с N прогонами, ближе к центру
        for yy in (mid + off, mid - off):
            if ty <= yy < by and len(runs_at(prob, weight, yy, xa, xb, cols)) == N:
                seedy = yy; break
        if seedy:
            break
    if seedy is None:
        return None, None
    seeds = sorted(runs_at(prob, weight, seedy, xa, xb, cols))
    tracks = {i: {"x": seeds[i], "v": 0.0, "pts": {seedy: seeds[i]}} for i in range(N)}

    def propagate(rng):
        for y in rng:
            rc = runs_at(prob, weight, y, xa, xb, cols)
            preds = {i: tracks[i]["x"] + float(np.clip(tracks[i]["v"], -SL, SL)) for i in tracks}
            if not rc:
                for i in tracks:
                    tracks[i]["x"] = preds[i]; tracks[i]["v"] *= 0.5; tracks[i]["pts"][y] = preds[i]
                continue
            pairs = sorted((abs(preds[i] - c), i, j) for i in tracks for j, c in enumerate(rc))
            ut = set(); ur = set(); amap = {}
            for d, i, j in pairs:
                if i in ut or j in ur:
                    continue
                amap[i] = rc[j]; ut.add(i); ur.add(j)
            for i in tracks:
                t = tracks[i]
                nx = amap[i] if (i in amap and abs(amap[i] - preds[i]) <= GATE) else preds[i]
                nx = float(np.clip(nx, xa, xb))           # кламп к полосе (без улёта)
                t["v"] = 0.6 * t["v"] + 0.4 * (nx - t["x"]); t["x"] = nx; t["pts"][y] = nx
    propagate(range(seedy + 1, by))
    for i in tracks:
        tracks[i]["x"] = seeds[i]; tracks[i]["v"] = 0.0
    propagate(range(seedy - 1, ty - 1, -1))
    return {i: tracks[i]["pts"] for i in tracks}, seedy


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    prob = np.load(PROB); m = extract(F); da = m["depth_axis"]
    H, W = prob.shape; ty, by = da["top_y"], min(H, da["bottom_y"])
    img = find_image(Path(F)) or m.get("img_path")
    gray = np.asarray(Image.open(img).convert("L"))[:H, :W]; bg = int(np.median(gray[::7, ::7]))
    weight = ((bg - gray.astype(np.int16)).clip(0).astype(np.float32)) * (prob > 0.4)
    gt = {c["name"].split()[0]: {c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL}
          for c in m["curves"] if c["name"].split()[0] in ("NNKM1", "NNKB1", "GK1")}
    N = 3                                                 # форсируем верное N (изолируем трекер от N-оценки)
    print(f"кластер x[{XA}..{XB}] N={N} (форс; оценка дала {estimate_n(prob, XA, XB, ty, by)}) | GT: "
          + ", ".join(f"{k}@{int(np.median(list(v.values())))}" for k, v in gt.items()))
    tracks, seedy = trace_multi(prob, weight, XA, XB, ty, by, N)
    if not tracks:
        print("сид-строка не найдена"); return
    print(f"сид-строка y={seedy}, треков={len(tracks)}")
    # сопоставить треки кривым по медиане x
    tmed = {i: np.median(list(p.values())) for i, p in tracks.items()}
    for k, gd in sorted(gt.items(), key=lambda kv: np.median(list(kv[1].values()))):
        gmed = np.median(list(gd.values()))
        i = min(tmed, key=lambda t: abs(tmed[t] - gmed))
        common = [y for y in gd if y in tracks[i]]
        if common:
            e = np.array([abs(tracks[i][y] - gd[y]) for y in common])
            print(f"  {k}@{int(gmed)} → трек{i}(med {int(tmed[i])}): |Δx|мед={np.median(e):.1f}px ≤5px={np.mean(e<=5)*100:.0f}% (n={len(common)})")
        else:
            print(f"  {k}: нет пересечения с треком")


if __name__ == "__main__":
    main()

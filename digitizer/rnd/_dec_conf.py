r"""_dec_conf.py — УВЕРЕННОСТЬ ТРАСС ДЕКОДЕРА ЗАДНИМ ЧИСЛОМ ДЛЯ КЭША (сайдкар, §6.240, 27.09).

`_dec_joint.py --pathstats`: медиана вероятности карты вдоль пути декодера предсказывает, честен ли путь, с AUC 0.95, а длина
(её сравнивает правило слота `emit`, §6.213) — 0.69. Карта зависит только от скана, трека, чекпойнта и окна по строкам —
не от выбора путей ⇒ уверенность считается для ЛЮБОГО кэша без пересборки трасс. Окно по строкам — как в
`rowdec.trace_auto`: [min y0, max y1] линий прод-пути трека, без линий — рамка (паритет проверен в `_dec_joint.py`).
Пишет `<ts>/conf_<имя кэша>.pkl`: {имя файла кэша: [(медиана p, средний −log p) по порядку `alt`]}. Возобновляемый: уже
посчитанные листы пропускаются; сохранение каждые 20 листов (атомарно).

  _dec_conf.py --cache F:/nds/output/taskS/tcache [--shard 0/2]
"""
import sys, argparse, pickle, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict
import numpy as np
from auto import imaging as im, rowdec as RD
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--cache", required=True)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--max-hours", type=float, default=0.0)
a = ap.parse_args()
TS = Path(a.ts); P = DEFAULT.cv; CACHE = Path(a.cache)
WMAP = json.loads((TS / "rowdec_wellmap.json").read_text(encoding="utf-8"))
SI, SN = (int(v) for v in a.shard.split("/"))
# ⚠ НЕ в каталоге кэша: стенды перебирают там `*.pkl` как листы и упали бы на чужом файле
OUT = TS / (f"conf_{CACHE.name}_{SI}of{SN}.pkl" if SN > 1 else f"conf_{CACHE.name}.pkl")
done = pickle.load(open(OUT, "rb")) if OUT.exists() else {}


def prob_map(rgb, track, ck, y0, y1):
    """Копия прохода окнами `rowdec.trace_track` (только вероятность)."""
    import torch
    net, dev = RD._load(ck)
    x0, x1 = int(track.x_left), int(track.x_right) + 1
    band = RD._band(rgb, P, x0, x1)
    H, Wb = band.shape
    y0 = max(0, int(y0)); y1 = min(H, int(y1))
    if y1 - y0 < 64 or Wb < 16:
        return None
    prob = np.zeros((y1 - y0, Wb), np.float32)
    STEP, OV, WIN = 512, 64, 512
    xcuts = [(c, min(Wb, c + WIN)) for c in range(0, max(1, Wb - 1), WIN - 64)]
    for (cx0, cx1) in xcuts:
        for gy in range(y0, y1, STEP - 2 * OV):
            gy2 = min(y1, gy + STEP)
            if gy2 - gy < 32:
                continue
            sub = band[gy:gy2, cx0:cx1].astype(np.float32) / 255.0
            with torch.no_grad():
                pp, _ = net(torch.from_numpy(sub)[None, None].to(dev))
                pp = torch.sigmoid(pp)[0, 0].float().cpu().numpy()
            v0 = gy + (OV if gy > y0 else 0); v1 = gy2 - (OV if gy2 < y1 else 0)
            wx0 = cx0 + (32 if cx0 > 0 else 0); wx1 = cx1 - (32 if cx1 < Wb else 0)
            if wx1 <= wx0:
                wx0, wx1 = cx0, cx1
            prob[v0 - y0:v1 - y0, wx0:wx1] = pp[v0 - gy:v1 - gy, wx0 - cx0:wx1 - cx0]
    return prob, x0, y0


def save():
    tmp = OUT.with_suffix(".tmp")
    pickle.dump(done, open(tmp, "wb")); tmp.replace(OUT)


files = sorted(CACHE.glob("*.pkl"))
files = [f for f in files if not f.name.startswith("_")][SI::SN]
T0 = time.time(); n_new = 0
for fi, f in enumerate(files, 1):
    if f.name in done:
        continue
    v = pickle.load(open(f, "rb"))
    if not v.get("alt"):
        done[f.name] = []; continue
    stem = Path(v["frame_nlgx"]).stem
    w = WMAP.get(Path(v["image"]).stem) or WMAP.get(v["stem"]) or WMAP.get(stem)
    fold = RD.fold_of_well(w) if w is not None else None
    ck = f"rowdec_of5_f{0 if fold is None else fold}_s0.pt"
    tracks = v["frame"].tracks
    by_t = defaultdict(list)
    for i, (L, t) in enumerate(v["alt"]):
        by_t[L.track_index].append(i)
    res = [None] * len(v["alt"])
    rgb = im.load_rgb(v["image"])
    for ti, idx in by_t.items():
        lines = [L for L, _ in v["traces"] if L.track_index == ti]
        if lines:
            y0 = min(L.y0 for L in lines); y1 = max(L.y1 for L in lines)
        else:
            y0, y1 = int(v["frame"].top_y), int(v["frame"].bottom_y)
        mp = prob_map(rgb, tracks[ti], ck, y0, y1)
        for i in idx:
            rows, xs = v["alt"][i][1]
            if mp is None or not len(rows):
                res[i] = (0.0, 99.0); continue
            prob, x0, py0 = mp
            r = rows - py0; c = xs.astype(int) - x0
            ok = (r >= 0) & (r < prob.shape[0]) & (c >= 0) & (c < prob.shape[1])
            pv = prob[r[ok], c[ok]] if ok.any() else np.zeros(1, np.float32)
            res[i] = (float(np.median(pv)), float(np.mean(-np.log(np.clip(pv, 1e-6, 1.0)))))
    del rgb
    done[f.name] = res; n_new += 1
    if n_new % 20 == 0:
        save(); print(f"  … {fi}/{len(files)} [{time.time() - T0:.0f} с]", flush=True)
    if a.max_hours and (time.time() - T0) / 3600 > a.max_hours:
        save(); print("★ ПАРТИЯ ОКОНЧЕНА — выхожу кодом 75"); sys.exit(75)
save()
print(f"★ УВЕРЕННОСТЬ ДЕКОДЕРА: листов {len(done)} в {OUT} (новых {n_new})")

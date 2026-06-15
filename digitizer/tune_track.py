r"""
tune_track.py — подбор параметров харднинга track() на ОДНОМ прогоне U-Net.
Метрики на конфиг: swap% (строки с экспертным GT), GZ2 own_px (регресс-индикатор),
out-of-track% (доля точек линии вне полосы своего трека = перескоки/выбросы).

python tune_track.py <ckpt> --nlgx <o.nlgx>   (venv ComfyUI)
"""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from infer import load_model, predict_prob
import track_multi as TM
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image

Image.MAX_IMAGE_PIXELS = None


def exp_traces(m):
    out = {}
    for c in m["curves"]:
        nm = c["name"].split()[0]
        if nm.startswith("DA"):
            continue
        ty = c["top_y"]
        d = {ty+i: x for i, x in enumerate(c["xs"]) if x != NULL}
        if len(d) >= 20:
            out[nm] = d
    return out


def metrics(lines, EO, track, tband):
    # сопоставить каждую экспертную кривую с ближайшей линией (по её строкам)
    Ls = [L["tr"] for L in lines]
    res = {}
    tot_rows = tot_swap = 0
    for nm, gd in EO.items():
        best = (1e9, None)
        for L in Ls:
            common = [y for y in gd if y in L]
            if len(common) < 15:
                continue
            e = np.median([abs(L[y]-gd[y]) for y in common])
            if e < best[0]:
                best = (e, L)
        if best[1] is None:
            continue
        L = best[1]
        rows = [y for y in gd if y in L]
        own = [abs(L[y]-gd[y]) for y in rows]
        peers = [p for p in EO if p != nm and track[p] == track[nm]]
        swap = 0
        for y in rows:
            ax = L[y]; d_own = abs(gd[y]-ax)
            if any(y in EO[p] and abs(EO[p][y]-ax) < d_own-3 for p in peers):
                swap += 1
        lo, hi = tband[track[nm]]
        allx = list(L.values())
        oot = np.mean([(x < lo-80) or (x > hi+80) for x in allx])
        res[nm] = (np.median(own), swap/max(1, len(rows)), oot, len(rows))
        tot_rows += len(rows); tot_swap += swap
    return res, (tot_swap, tot_rows)


def main():
    ckpt = sys.argv[1]; nlgx = sys.argv[sys.argv.index("--nlgx")+1]
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, _ = load_model(ckpt, device)
    m = extract(nlgx)
    rgb = np.asarray(Image.open(find_image(Path(nlgx))).convert("RGB"))
    H, W, _ = rgb.shape
    da = m["depth_axis"]; y0, y1 = da["top_y"], min(H, da["bottom_y"])
    print("U-Net прогон…")
    prob = predict_prob(net, rgb, device, y0=y0, y1=y1)

    EO = exp_traces(m)
    med = {n: np.median(list(EO[n].values())) for n in EO}
    order = sorted(EO, key=lambda n: med[n])
    track = {}; t = 0
    for k, n in enumerate(order):
        if k and med[n]-med[order[k-1]] > 250:
            t += 1
        track[n] = t
    tband = {}
    for n in order:
        xs = list(EO[n].values()); tt = track[n]
        lo, hi = tband.get(tt, (1e9, -1e9))
        tband[tt] = (min(lo, min(xs)), max(hi, max(xs)))

    configs = [
        ("OLDrepl mj106 band∞   ", dict(wlam=0.0, band=1e9, slmax=1e9, max_jump=106)),
        ("mj100 band400 a.997   ", dict(wlam=0.0, band=400, slmax=12, max_jump=100, anc_a=0.997)),
        ("mj100 band500 a.997   ", dict(wlam=0.0, band=500, slmax=12, max_jump=100, anc_a=0.997)),
        ("mj100 band400 wid0.3  ", dict(wlam=0.3, band=400, slmax=12, max_jump=100, anc_a=0.997)),
        ("mj100 band350 wid0.5  ", dict(wlam=0.5, band=350, slmax=12, max_jump=100, anc_a=0.997)),
        ("mj120 band450 wid0.3  ", dict(wlam=0.3, band=450, slmax=14, max_jump=120, anc_a=0.997)),
    ]
    print(f"seed/N будут учтены в track; полосы треков: {tband}")
    print(f"\n{'config':<24}{'swap%':>7}{'GZ2own':>8}{'GZ1own':>8}{'oot%max':>9}")
    for label, kw in configs:
        lines, N = TM.track(prob, rgb, y0, y1, **kw)
        res, (ts, tr) = metrics(lines, EO, track, tband)
        gz2 = res.get("GZ21", res.get("GZ2", (float('nan'),)*4))
        gz1 = res.get("GZ11", res.get("GZ1", (float('nan'),)*4))
        ootmax = max((v[2] for v in res.values()), default=0)
        print(f"{label:<24}{100*ts/max(1,tr):>6.1f}%{gz2[0]:>8.1f}{gz1[0]:>8.1f}{100*ootmax:>8.1f}%")
        det = "   ".join(f"{n}:own{res[n][0]:.0f}/sw{100*res[n][1]:.0f}/oot{100*res[n][2]:.0f}" for n in order if n in res)
        print(f"      {det}")


if __name__ == "__main__":
    main()

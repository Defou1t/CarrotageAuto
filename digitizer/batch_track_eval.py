r"""
batch_track_eval.py — валидация харднинга track() по val-набору: для каждого файла ОДИН
прогон U-Net, затем track СТАРЫМ конфигом (mj106, без band/width/clamp) и НОВЫМ (дефолты).
На кривую: own_px (медиана vs экспертный GT на его строках), swap% (соседи того же трека),
out-of-track% (точки линии вне полосы трека). Классы: separable (1 кривая в треке) vs overlapped.
Цель: separable own НЕ регрессировал; overlapped swap/oot улучшились.

python batch_track_eval.py <ckpt> [--limit 12] [--csv out.csv]   (venv ComfyUI)
"""
import sys, json, csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from infer import load_model, predict_prob
import track_multi as TM
from extract_nlgx import extract, NULL, depth_axis_ok
import dataset as ds
from dataset_build import find_image

Image.MAX_IMAGE_PIXELS = None
ARCHIVE = Path(r"F:\nds\projects\Archive")
OLD = dict(wlam=0.0, band=1e9, slmax=1e9, max_jump=106)
NEW = {}  # дефолты track()


def exp_traces(m, y0, y1):
    out = {}
    for c in ds.real_curves(m):
        nm = c["name"].split()[0]
        if nm.startswith("DA"):
            continue
        ty = c["top_y"]
        d = {ty+i: x for i, x in enumerate(c["xs"]) if x != NULL and y0 <= ty+i < y1}
        if len(d) >= 50:
            out[nm] = d
    return out


def tracks_of(EO):
    med = {n: np.median(list(EO[n].values())) for n in EO}
    order = sorted(EO, key=lambda n: med[n])
    tr = {}; t = 0
    for k, n in enumerate(order):
        if k and med[n] - med[order[k-1]] > 250:
            t += 1
        tr[n] = t
    band = {}
    for n in order:
        xs = list(EO[n].values())
        lo, hi = band.get(tr[n], (1e9, -1e9))
        band[tr[n]] = (min(lo, min(xs)), max(hi, max(xs)))
    size = {t: sum(1 for n in tr if tr[n] == t) for t in set(tr.values())}
    return tr, band, size


def curve_metrics(lines, EO, tr, band):
    Ls = [L["tr"] for L in lines]
    res = {}
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
        own = np.median([abs(L[y]-gd[y]) for y in rows])
        peers = [p for p in EO if p != nm and tr[p] == tr[nm]]
        swap = sum(1 for y in rows if any(y in EO[p] and abs(EO[p][y]-L[y]) < abs(gd[y]-L[y])-3 for p in peers))
        lo, hi = band[tr[nm]]
        oot = np.mean([(x < lo-80) or (x > hi+80) for x in L.values()])
        res[nm] = dict(own=float(own), swap=swap/max(1, len(rows)), oot=float(oot), rows=len(rows))
    return res


def main():
    ckpt = sys.argv[1]; args = sys.argv[2:]
    limit = 12; csvp = r"F:\nds\output\track_val_report.csv"
    i = 0
    while i < len(args):
        if args[i] == "--limit": limit = int(args[i+1]); i += 2
        elif args[i] == "--csv": csvp = args[i+1]; i += 2
        else: i += 1
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(r"F:\nds\output\unet_data")
    idx = json.loads((data / "index.json").read_text(encoding="utf-8"))
    val_wells = sorted({r["well"] for r in idx if r["split"] == "val"})

    files = []
    for w in val_wells:
        wlg = ARCHIVE / w / "wlg"
        if not wlg.is_dir():
            continue
        for nlgx in sorted(wlg.glob("*.nlgx")):
            if "_auto" in nlgx.stem:
                continue
            try:
                m = extract(str(nlgx))
            except Exception:
                continue
            if depth_axis_ok(m) and len(ds.real_curves(m)) >= 1 and find_image(nlgx):
                files.append(nlgx)
    # ровная выборка, разнообразие по числу кривых
    files.sort(key=lambda p: len(ds.real_curves(extract(str(p)))))
    if len(files) > limit:
        sel = np.linspace(0, len(files)-1, limit).round().astype(int)
        files = [files[k] for k in sorted(set(sel))]
    print(f"val-файлов к прогону: {len(files)}")

    net, _ = load_model(ckpt, device)
    rows_csv = []
    sep_old, sep_new = [], []
    ov_old_sw, ov_new_sw, ov_old_oot, ov_new_oot = [], [], [], []
    for nlgx in files:
        m = extract(str(nlgx))
        try:
            rgb = np.asarray(Image.open(find_image(nlgx)).convert("RGB"))
        except Exception as e:
            print(f"  пропуск {nlgx.stem[:40]}: {e}"); continue
        H, W, _ = rgb.shape
        da = m["depth_axis"]; y0, y1 = da["top_y"], min(H, da["bottom_y"])
        if y1 - y0 < 200:
            continue
        EO = exp_traces(m, y0, y1)
        if not EO:
            continue
        tr, band, size = tracks_of(EO)
        prob = predict_prob(net, rgb, device, y0=y0, y1=y1)
        lo, _ = TM.track(prob, rgb, y0, y1, **OLD)
        ln, _ = TM.track(prob, rgb, y0, y1, **NEW)
        ro = curve_metrics(lo, EO, tr, band)
        rn = curve_metrics(ln, EO, tr, band)
        ncur = len(EO)
        print(f"\n{nlgx.stem[:46]:<46} curves={ncur}")
        for nm in sorted(EO, key=lambda n: np.median(list(EO[n].values()))):
            o = ro.get(nm); n2 = rn.get(nm)
            if not o or not n2:
                continue
            sepc = size[tr[nm]] == 1
            tag = "sep" if sepc else "ovl"
            print(f"   {nm:<7}[{tag}] own {o['own']:.1f}->{n2['own']:.1f}  "
                  f"swap {100*o['swap']:.0f}->{100*n2['swap']:.0f}%  "
                  f"oot {100*o['oot']:.0f}->{100*n2['oot']:.0f}%")
            rows_csv.append([nlgx.stem, nm, tag, ncur, round(o['own'],1), round(n2['own'],1),
                             round(o['swap'],3), round(n2['swap'],3), round(o['oot'],3), round(n2['oot'],3)])
            if sepc:
                sep_old.append(o['own']); sep_new.append(n2['own'])
            else:
                ov_old_sw.append(o['swap']); ov_new_sw.append(n2['swap'])
                ov_old_oot.append(o['oot']); ov_new_oot.append(n2['oot'])

    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem","curve","class","ncurves","own_old","own_new","swap_old","swap_new","oot_old","oot_new"])
        w.writerows(rows_csv)

    def med(a): return float(np.median(a)) if a else float('nan')
    print("\n================ ИТОГО ================")
    print(f"SEPARABLE кривых: {len(sep_old)}  own медиана OLD {med(sep_old):.1f}px -> NEW {med(sep_new):.1f}px")
    reg = sum(1 for o, n in zip(sep_old, sep_new) if n > o + 2)
    print(f"  регрессий (NEW own > OLD+2px): {reg}/{len(sep_old)}")
    print(f"OVERLAPPED кривых: {len(ov_old_sw)}  swap% медиана OLD {100*med(ov_old_sw):.1f} -> NEW {100*med(ov_new_sw):.1f}")
    print(f"  oot% медиана OLD {100*med(ov_old_oot):.1f} -> NEW {100*med(ov_new_oot):.1f}")
    print(f"  swap mean OLD {100*np.mean(ov_old_sw):.1f} -> NEW {100*np.mean(ov_new_sw):.1f}  "
          f"oot mean OLD {100*np.mean(ov_old_oot):.1f} -> NEW {100*np.mean(ov_new_oot):.1f}")
    print(f"\nCSV -> {csvp}")


if __name__ == "__main__":
    main()

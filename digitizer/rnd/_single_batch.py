r"""
_single_batch.py — PRODUCTION автотрасса ОДИНОЧНЫХ кривых по списку рамок.
Для каждой кривой: U-Net → per-row центроид в её треке → инъекция в КОПИЮ рамки (_auto.nlgx+bck).
Если есть трасса эксперта — печатает точность (|Δx|, ≤5px, off_ink). Авто-ВЕРДИКТ:
  OK (одиночная, сдаётся) | ФЛАГ (мультилиния/обороты → ручная/декод).
Запуск ComfyUI-python:
  python _single_batch.py <nlgx|папка> [--ckpt MODEL] [--out DIR]
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
import torch, struct
from unet import UNet
from extract_nlgx import extract, NULL
from write_nlgx import read_full, write_full, write_bck, set_tag
from dataset_build import find_image


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev)
    net = UNet(in_ch=3, n_classes=1, base=ck.get("base", 32)); net.load_state_dict(ck["model"])
    net.to(dev).eval(); return net


@torch.no_grad()
def predict_prob(net, rgb, dev, tile=256, ov=96, bs=24, y0=0, y1=None):
    H, W, _ = rgb.shape; y1 = H if y1 is None else min(H, y1); step = tile - ov
    ys = list(range(0, max(1, H - tile + 1), step)); xs = list(range(0, max(1, W - tile + 1), step))
    if H > tile and ys[-1] != H - tile: ys.append(H - tile)
    if W > tile and xs[-1] != W - tile: xs.append(W - tile)
    ys = [y for y in ys if y + tile > y0 and y < y1]
    prob = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    w1 = np.hanning(tile); win = np.outer(w1, w1).astype(np.float32) + 1e-3
    batch = []; pos = []
    def flush():
        if not batch: return
        t = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().div(255).to(dev)
        with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
            p = torch.sigmoid(net(t)).float().cpu().numpy()[:, 0]
        for (yy, xx), pm in zip(pos, p):
            prob[yy:yy+tile, xx:xx+tile] += pm * win; wsum[yy:yy+tile, xx:xx+tile] += win
        batch.clear(); pos.clear()
    for yy in ys:
        for xx in xs:
            batch.append(rgb[yy:yy+tile, xx:xx+tile]); pos.append((yy, xx))
            if len(batch) >= bs: flush()
    flush()
    return prob / np.maximum(wsum, 1e-6)


def sa_for(curve, m):
    """трек (x_left,x_right) скейл-оси этой кривой (по суффиксу имени 'DA1 SA1')."""
    suf = " ".join(curve["name"].split()[1:])
    for s in m["scale_axes"]:
        if s["name"].endswith(suf):
            return s["x_left"], s["x_right"]
    s = m["scale_axes"][0]; return s["x_left"], s["x_right"]


def trace_curve(prob, xL, xR, ty, by, W):
    cols = np.arange(max(0, xL), min(W, xR)); out = {}
    for y in range(ty, by):
        seg = prob[y, max(0, xL):min(W, xR)]; mk = seg > 0.4
        if mk.any():
            out[y] = float((cols[mk] * seg[mk]).sum() / seg[mk].sum())
    return out


def process(nlgx, net, dev, out):
    m = extract(nlgx)
    img = find_image(Path(nlgx)) or m.get("img_path")
    if not img or not Path(img).is_file():
        return []
    rgb = np.asarray(Image.open(img).convert("RGB")); H, W = rgb.shape[:2]
    gray = np.asarray(Image.open(img).convert("L")); bg = int(np.median(gray))
    da = m["depth_axis"]; ty, by = da["top_y"], min(H, da["bottom_y"])
    prob = predict_prob(net, rgb, dev, y0=ty, y1=by)
    curves = [c for c in m["curves"] if not c["name"].split()[0].startswith("DA")]
    ifds = read_full(open(nlgx, "rb").read()); rows = []
    for c in curves:
        short = c["name"].split()[0]; xL, xR = sa_for(c, m)
        auto = trace_curve(prob, xL, xR, ty, by, W)
        if len(auto) < 30:
            rows.append((Path(nlgx).stem, short, None, None, None, "ПУСТО")); continue
        gt = {c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL}
        on_ink = np.mean([gray[y, int(round(auto[y]))] < bg - 40 for y in auto if 0 <= int(round(auto[y])) < W]) * 100
        offink = 100 - on_ink
        med5 = None; cov = None
        if len(gt) >= 30:
            common = [y for y in gt if y in auto]
            if common:
                d = np.array([abs(auto[y] - gt[y]) for y in common])
                med5 = (float(np.median(d)), float(np.mean(d <= 5) * 100)); cov = len(common) / len(gt) * 100
        verdict = "OK" if offink < 8 and (med5 is None or med5[0] < 5) else "ФЛАГ (мульти/обороты)"
        # инъекция
        for i, ifd in enumerate(ifds):
            tg = {t: raw for t, typ, cc, raw in ifd["entries"]}
            if 35470 in tg and tg[35470].split(b"\x00")[0].decode("latin1").startswith(short + " "):
                n = struct.unpack("<I", tg[35488][:4])[0]; ty0 = struct.unpack("<I", tg[35474][:4])[0]
                xs = [int(round(auto[ty0 + j])) if (ty0 + j) in auto else NULL for j in range(n)]
                set_tag(ifds, i, 35490, 4, xs)
                for t, v in [(35492, 1), (35494, ty0), (35496, ty0 + n - 1), (35498, 0)]:
                    set_tag(ifds, i, t, 4, [v])
                vx = [(j, x) for j, x in enumerate(xs) if x != NULL]
                if vx:
                    rw = [ty0 + j for j, _ in vx]; xv = [x for _, x in vx]
                    for t, v in [(35478, min(xv)), (35480, min(rw)), (35482, max(xv)), (35484, max(rw))]:
                        set_tag(ifds, i, t, 4, [v])
                break
        rows.append((Path(nlgx).stem, short, med5, cov, offink, verdict))
    data = write_full(ifds)
    open(out / f"{Path(nlgx).stem}_auto.nlgx", "wb").write(data)
    open(out / f"{Path(nlgx).stem}_auto.bck", "wb").write(write_bck(data))
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    a = sys.argv[1:]
    src = Path(a[0])
    ckpt = a[a.index("--ckpt") + 1] if "--ckpt" in a else r"F:\nds\output\unet_data\unet_best.pt"
    out = Path(a[a.index("--out") + 1] if "--out" in a else r"F:\nds\output\single_out"); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nlgxs = sorted(src.glob("*.nlgx")) if src.is_dir() else [src]
    nlgxs = [p for p in nlgxs if "_auto" not in p.stem and "framed" not in p.stem]
    net = load_model(ckpt, dev)
    print(f"device={dev} ckpt={Path(ckpt).name} файлов={len(nlgxs)} -> {out}\n")
    print(f"{'скан':<40}{'кривая':<7}{'|Δx|мед':>8}{'≤5px':>6}{'off_ink':>8}  вердикт")
    allrows = []
    for p in nlgxs:
        try:
            rows = process(str(p), net, dev, out)
        except Exception as e:
            print(f"  ! {p.stem[:38]}: {str(e)[:50]}"); continue
        for stem, short, med5, cov, offink, verdict in rows:
            md = f"{med5[0]:.1f}" if med5 else "-"; p5 = f"{med5[1]:.0f}%" if med5 else "-"
            oi = f"{offink:.0f}%" if offink is not None else "-"
            print(f"{stem[:40]:<40}{short:<7}{md:>8}{p5:>6}{oi:>8}  {verdict}")
            allrows.append((short, med5, offink, verdict))
    ok = [r for r in allrows if r[3] == "OK"]
    okm = [r[1][0] for r in ok if r[1]]
    print(f"\n=== ИТОГ: {len(ok)}/{len(allrows)} OK (одиночные сдаются)" +
          (f"; |Δx| медиана по OK с GT = {np.median(okm):.1f}px" if okm else "") + " ===")


if __name__ == "__main__":
    main()

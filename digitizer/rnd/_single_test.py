r"""
_single_test.py — ТЕСТ автомата на ОДИНОЧНОЙ кривой с GROUND TRUTH (трасса эксперта).
Старая U-Net → per-row центроид вероятности в треке → сравнение с трассой эксперта (|Δx|, %в допуске,
off_ink) + инъекция в КОПИЮ рамки + оверлей (эксперт зелёный / автомат красный).
Запуск ComfyUI-python (torch).

python _single_test.py <nlgx>
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

OUT = Path(r"F:\nds\output\test_single")
CKPT = r"F:\nds\output\unet_data\unet_best.pt"


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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    nlgx = sys.argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = extract(nlgx)
    img = find_image(Path(nlgx)) or m.get("img_path")
    rgb = np.asarray(Image.open(img).convert("RGB")); H, W = rgb.shape[:2]
    gray = np.asarray(Image.open(img).convert("L")); bg = int(np.median(gray))
    da = m["depth_axis"]; ty, by = da["top_y"], min(H, da["bottom_y"])
    cur = [c for c in m["curves"] if not c["name"].split()[0].startswith("DA")][0]
    short = cur["name"].split()[0]
    sa = m["scale_axes"][0]; xL, xR = sa["x_left"], min(W, sa["x_right"])
    # GT трасса эксперта
    gt = {cur["top_y"] + i: x for i, x in enumerate(cur["xs"]) if x != NULL}
    print(f"{Path(nlgx).stem}: кривая {short}, GT-точек {len(gt)}, трек x[{xL}..{xR}], depth y[{ty}..{by}]")

    net = load_model(CKPT, dev)
    prob = predict_prob(net, rgb, dev, y0=ty, y1=by)
    # АВТОТРАССА: per-row взвешенный центроид prob в треке
    auto = {}
    cols = np.arange(xL, xR)
    for y in range(ty, by):
        seg = prob[y, xL:xR]; mk = seg > 0.4
        if mk.any():
            auto[y] = float((cols[mk] * seg[mk]).sum() / seg[mk].sum())

    # СРАВНЕНИЕ vs GT
    common = [y for y in gt if y in auto]
    d = np.array([abs(auto[y] - gt[y]) for y in common])
    on_ink = np.mean([gray[y, int(round(auto[y]))] < bg - 40 for y in auto if 0 <= int(round(auto[y])) < W]) * 100
    print(f"\n=== ТОЧНОСТЬ vs ЭКСПЕРТ ({len(common)} общих строк) ===")
    print(f"  |Δx|: медиана={np.median(d):.1f}px  p90={np.percentile(d,90):.1f}px  макс={d.max():.0f}px")
    print(f"  в допуске: <=2px {np.mean(d<=2)*100:.0f}%  <=5px {np.mean(d<=5)*100:.0f}%  <=10px {np.mean(d<=10)*100:.0f}%")
    print(f"  покрытие автотрассой GT: {len(common)/len(gt)*100:.0f}%  | off_ink автотрассы: {100-on_ink:.1f}%")

    # инъекция в КОПИЮ рамки
    ifds = read_full(open(nlgx, "rb").read())
    for i, ifd in enumerate(ifds):
        tags = {t: raw for t, typ, c, raw in ifd["entries"]}
        if 35470 in tags and tags[35470].split(b"\x00")[0].decode("latin1").startswith(short + " "):
            n = struct.unpack("<I", tags[35488][:4])[0]; ty0 = struct.unpack("<I", tags[35474][:4])[0]
            xs = [int(round(auto[ty0 + j])) if (ty0 + j) in auto else NULL for j in range(n)]
            set_tag(ifds, i, 35490, 4, xs)
            for t, v in [(35492, 1), (35494, ty0), (35496, ty0 + n - 1), (35498, 0)]:
                set_tag(ifds, i, t, 4, [v])
            vx = [(j, x) for j, x in enumerate(xs) if x != NULL]
            if vx:
                rws = [ty0 + j for j, _ in vx]; xv = [x for _, x in vx]
                for t, v in [(35478, min(xv)), (35480, min(rws)), (35482, max(xv)), (35484, max(rws))]:
                    set_tag(ifds, i, t, 4, [v])
            break
    data = write_full(ifds); stem = Path(nlgx).stem
    open(OUT / f"{stem}_single_auto.nlgx", "wb").write(data)
    open(OUT / f"{stem}_single_auto.bck", "wb").write(write_bck(data))
    print(f"\n-> F:\\nds\\output\\test_single\\{stem}_single_auto.nlgx (+bck)")

    # оверлей: эксперт зелёный, автомат красный
    im = Image.open(img).convert("RGB")
    for tag, frac in [("top", 0.25), ("mid", 0.5), ("bot", 0.75)]:
        yc = int(ty + frac * (by - ty)); y0, y1 = max(0, yc - 220), min(H, yc + 220)
        crop = im.crop((0, y0, W, y1)).convert("RGB"); dr = ImageDraw.Draw(crop)
        for tr, col in [(gt, (0, 180, 0)), (auto, (230, 0, 0))]:
            pts = sorted((y, x) for y, x in tr.items() if y0 <= y < y1)
            for (ya, xa), (yb, xb) in zip(pts, pts[1:]):
                if abs(yb - ya) <= 3:
                    dr.line([(xa, ya - y0), (xb, yb - y0)], fill=col, width=2)
        crop.save(OUT / f"single_{stem}_{tag}.png")
    print(f"оверлей (эксперт зелёный / автомат красный) -> F:\\nds\\output\\test_single\\single_{stem}_[top,mid,bot].png")


if __name__ == "__main__":
    main()

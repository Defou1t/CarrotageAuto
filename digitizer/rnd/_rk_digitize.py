r"""
_rk_digitize.py — оцифровка RK (GK слева + NGK справа, разнесённые) с ДЕТЕКТОМ ПОЛОСЫ.
U-Net → плотность чернила по столбцам → разрыв между кривыми → 2 полосы → трасса каждой
в своей полосе (per-row центроид). Инъекция в КОПИЮ рамки + оверлей vs частичная трасса эксперта.
Запуск ComfyUI-python.

python _rk_digitize.py
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

FRAME = r"F:\nds\output\test_single\Yatskivska_1_RK_3180_4080_200_D1.nlgx"
CKPT = r"F:\nds\output\unet_data\unet_best.pt"
OUT = Path(r"F:\nds\output\test_single")


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


def trace_band(prob, xa, xb, ty, by, thr=0.4):
    cols = np.arange(xa, xb); out = {}
    for y in range(ty, by):
        seg = prob[y, xa:xb]; mk = seg > thr
        if mk.any():
            out[y] = float((cols[mk] * seg[mk]).sum() / seg[mk].sum())
    return out


def inject(ifds, short, tr, W):
    for i, ifd in enumerate(ifds):
        tags = {t: raw for t, typ, c, raw in ifd["entries"]}
        if 35470 in tags and tags[35470].split(b"\x00")[0].decode("latin1").startswith(short + " "):
            n = struct.unpack("<I", tags[35488][:4])[0]; ty0 = struct.unpack("<I", tags[35474][:4])[0]
            xs = []
            for j in range(n):
                v = tr.get(ty0 + j)
                xs.append(int(round(v)) if (v is not None and 0 <= round(v) < W) else NULL)
            set_tag(ifds, i, 35490, 4, xs)
            for t, v in [(35492, 1), (35494, ty0), (35496, ty0 + n - 1), (35498, 0)]:
                set_tag(ifds, i, t, 4, [v])
            vx = [(j, x) for j, x in enumerate(xs) if x != NULL]
            if vx:
                rw = [ty0 + j for j, _ in vx]; xv = [x for _, x in vx]
                for t, v in [(35478, min(xv)), (35480, min(rw)), (35482, max(xv)), (35484, max(rw))]:
                    set_tag(ifds, i, t, 4, [v])
            return sum(1 for x in xs if x != NULL)
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = extract(FRAME); rgb = np.asarray(Image.open(m["img_path"]).convert("RGB")); H, W = rgb.shape[:2]
    da = m["depth_axis"]; ty, by = da["top_y"], min(H, da["bottom_y"])
    xL, xR = m["scale_axes"][0]["x_left"], min(W, m["scale_axes"][0]["x_right"])
    # GT эксперта (для сравнения + назначения полос)
    gt = {}
    for c in m["curves"]:
        sh = c["name"].split()[0]
        if sh.startswith("DA"): continue
        gt[sh] = {c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL}
    print(f"RK {W}x{H} band y[{ty}..{by}] трек x[{xL}..{xR}]; GT: " +
          ", ".join(f"{k}={len(v)}тчк" for k, v in gt.items()))

    net = load_model(CKPT, dev)
    prob = predict_prob(net, rgb, dev, y0=ty, y1=by)
    np.save(OUT / "rk_prob.npy", prob)

    # ДЕТЕКТ ПОЛОС: плотность чернила по столбцам → разрыв между левой и правой кривой
    colsum = prob[ty:by, xL:xR].sum(0)
    k = 41; colsm = np.convolve(colsum, np.ones(k) / k, mode="same")
    # разрыв = самый глубокий провал в средних 30-70% трека
    mid = colsm[int(0.25 * len(colsm)):int(0.75 * len(colsm))]
    split = xL + int(0.25 * len(colsm)) + int(np.argmin(mid))
    print(f"разрыв между кривыми: x={split} → GK-полоса [{xL}..{split}], NGK-полоса [{split}..{xR}]")

    bands = {"GK1": (xL, split), "NGK1": (split, xR)}
    auto = {}
    for sh, (xa, xb) in bands.items():
        auto[sh] = trace_band(prob, xa, xb, ty, by)
    # точность vs частичный GT
    for sh in ("GK1", "NGK1"):
        g = gt.get(sh, {}); common = [y for y in g if y in auto[sh]]
        if common:
            d = np.array([abs(auto[sh][y] - g[y]) for y in common])
            print(f"  {sh}: автотрасса {len(auto[sh])}тчк | vs GT({len(common)} общих): |Δx| мед={np.median(d):.1f}px ≤5px={np.mean(d<=5)*100:.0f}%")
        else:
            print(f"  {sh}: автотрасса {len(auto[sh])}тчк (GT мало для сравнения)")

    ifds = read_full(open(FRAME, "rb").read())
    for sh in ("GK1", "NGK1"):
        nn = inject(ifds, sh, auto[sh], W); print(f"  инъекция {sh}: {nn} точек")
    data = write_full(ifds)
    open(OUT / "Yatskivska_1_RK_3180_4080_200_D1_rk_auto.nlgx", "wb").write(data)
    open(OUT / "Yatskivska_1_RK_3180_4080_200_D1_rk_auto.bck", "wb").write(write_bck(data))
    print("-> F:\\nds\\output\\test_single\\Yatskivska_1_RK_3180_4080_200_D1_rk_auto.nlgx")

    # оверлей: авто GK зелёный / NGK синий; GT эксперта красным (где есть)
    img = Image.open(m["img_path"]).convert("RGB")
    for tag, frac in [("a", 0.06), ("b", 0.5), ("c", 0.9)]:
        yc = int(ty + frac * (by - ty)); y0, y1 = max(0, yc - 240), min(H, yc + 240)
        crop = img.crop((0, y0, W, y1)).convert("RGB"); dr = ImageDraw.Draw(crop)
        for sh, col in [("GK1", (0, 170, 0)), ("NGK1", (0, 90, 255))]:
            pts = sorted((y, x) for y, x in auto[sh].items() if y0 <= y < y1)
            for (ya, xa), (yb, xb) in zip(pts, pts[1:]):
                if abs(yb - ya) <= 3: dr.line([(xa, ya - y0), (xb, yb - y0)], fill=col, width=2)
        for sh in ("GK1", "NGK1"):
            pts = sorted((y, x) for y, x in gt.get(sh, {}).items() if y0 <= y < y1)
            for (ya, xa), (yb, xb) in zip(pts, pts[1:]):
                if abs(yb - ya) <= 3: dr.line([(xa, ya - y0), (xb, yb - y0)], fill=(230, 0, 0), width=2)
        dr.line([(split, 0), (split, y1 - y0)], fill=(255, 160, 0), width=1)   # граница полос
        crop.save(OUT / f"rk_crop_{tag}.png")
    print("оверлей (GK зелёный/NGK синий/эксперт красный/граница оранж) -> rk_crop_[a,b,c].png")


if __name__ == "__main__":
    main()

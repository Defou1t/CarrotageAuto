r"""
_ak_unet.py — прогон ОБУЧЕННОЙ U-Net на AK-скане Pn_Zavoda: находит ли сеть линию (vs сетку)?
Карта вероятности штриха → overlay поверх скана (красным где сеть «горит») + наивная трасса
в рамку. Запуск ИНТЕРПРЕТАТОРОМ ComfyUI (torch+CUDA).

D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe _ak_unet.py [ckpt]
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
import torch, struct
from unet import UNet
from extract_nlgx import extract, NULL
from write_nlgx import read_full, write_full, write_bck, set_tag, find_ifd

FRAME = r"F:\nds\projects\Pn_Zavoda_001\wlg\Pn_Zavoda_1_AK_5508_5716_200_D1.nlgx"
OUT = Path(r"F:\nds\output\test_single")
CKPT = sys.argv[1] if len(sys.argv) > 1 else r"F:\nds\output\unet_data\unet_best.pt"


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    net = UNet(in_ch=3, n_classes=1, base=ck.get("base", 32))
    net.load_state_dict(ck["model"]); net.to(device).eval()
    return net


@torch.no_grad()
def predict_prob(net, rgb, device, tile=256, ov=96, bs=24, y0=0, y1=None):
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
        t = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().div(255).to(device)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
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


def inject(ifds, short, xs_arr):
    for i, ifd in enumerate(ifds):
        tags = {t: raw for t, typ, c, raw in ifd["entries"]}
        if 35470 in tags and tags[35470].split(b"\x00")[0].decode("latin1").startswith(short + " "):
            n = struct.unpack("<I", tags[35488][:4])[0]
            ty = struct.unpack("<I", tags[35474][:4])[0]
            xs = [int(x) if not np.isnan(x) else NULL for x in xs_arr[:n]] + [NULL] * max(0, n - len(xs_arr))
            set_tag(ifds, i, 35490, 4, xs[:n])
            for t, v in [(35492, 1), (35494, ty), (35496, ty + n - 1), (35498, 0)]:
                set_tag(ifds, i, t, 4, [v])
            vx = [(j, x) for j, x in enumerate(xs[:n]) if x != NULL]
            if vx:
                rws = [ty + j for j, _ in vx]; xv = [x for _, x in vx]
                for t, v in [(35478, min(xv)), (35480, min(rws)), (35482, max(xv)), (35484, max(rws))]:
                    set_tag(ifds, i, t, 4, [v])
            return sum(1 for x in xs[:n] if x != NULL)
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = extract(FRAME); rgb = np.asarray(Image.open(m["img_path"]).convert("RGB")); H, W = rgb.shape[:2]
    da = m["depth_axis"]; ty, by = da["top_y"], min(H, da["bottom_y"])
    print(f"device={dev} ckpt={Path(CKPT).name} scan {W}x{H} band y[{ty}..{by}]")
    net = load_model(CKPT, dev)
    prob = predict_prob(net, rgb, dev, y0=ty, y1=by)
    band = prob[ty:by]
    print(f"prob: max={band.max():.2f} mean={band.mean():.3f} | покрытие >0.4 = {(band>0.4).mean()*100:.1f}% строк-с-сигналом={(band.max(1)>0.4).mean()*100:.0f}%")
    np.save(OUT / "ak_prob.npy", prob)        # кэш карты — трекер крутим без GPU
    print("prob-карта -> F:\\nds\\output\\test_single\\ak_prob.npy")

    # overlay: красным где prob>0.3, поверх серого скана; кропы на тех же глубинах
    g = np.asarray(Image.open(m["img_path"]).convert("L"))
    for tag, dc in [("a", 5540.0), ("b", 5600.0), ("c", 5660.0)]:
        yc = int(ty + (dc - da["top_depth"]) / (da["bottom_depth"] - da["top_depth"]) * (by - ty))
        y0, y1 = max(0, yc - 220), min(H, yc + 220)
        base = np.stack([g[y0:y1]] * 3, -1).astype(np.uint8)
        pp = prob[y0:y1]
        base[..., 0] = np.maximum(base[..., 0], (pp * 255).astype(np.uint8))   # красный = prob
        base[..., 1] = np.where(pp > 0.3, base[..., 1] // 2, base[..., 1])
        base[..., 2] = np.where(pp > 0.3, base[..., 2] // 2, base[..., 2])
        Image.fromarray(base).save(OUT / f"unet_prob_{tag}.png")
    print("overlay-маски -> F:\\nds\\output\\test_single\\unet_prob_[a,b,c].png")

    # наивная трасса: per-row взвешенный центроид prob в треке (покажет, что центроид мульти-линии плывёт)
    sa = m["scale_axes"][0]; x0, x1 = max(0, sa["x_left"] - 25), min(W, sa["x_right"] + 110)
    n = by - ty; out = np.full(n, np.nan); cols = np.arange(x0, x1)
    for i in range(n):
        seg = prob[ty + i, x0:x1]; mk = seg > 0.4
        if mk.any(): out[i] = (cols[mk] * seg[mk]).sum() / seg[mk].sum()
    ifds = read_full(open(FRAME, "rb").read())
    nm = inject(ifds, "MGZ1", out)
    data = write_full(ifds); open(OUT / "Pn_Zavoda_1_AK_5508_5716_200_D1_unet.nlgx", "wb").write(data)
    open(OUT / "Pn_Zavoda_1_AK_5508_5716_200_D1_unet.bck", "wb").write(write_bck(data))
    print(f"наивная U-Net трасса в MGZ1: {nm} точек -> F:\\nds\\output\\test_single\\..._unet.nlgx")


if __name__ == "__main__":
    main()

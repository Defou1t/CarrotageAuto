r"""Трек 2 / вариант 4 инференс — UNetEmb: prob (полоса/детекция) + ЭМБЕДДИНГ
(идентичность). Логика: эмбеддинг РАЗДЕЛЯЕТ 2 позиции в строке (в т.ч. в наложении,
где prob размыт), prob НАЗНАЧАЕТ идентичность (MGZ/MPZ). Эмбеддинг локально-
консистентен (disc-loss per-tile) ⇒ кластеризуем ПО СТРОКЕ, не глобально.
  <venv>\python.exe infer_mk_emb.py <mk_emb.pt> <img> --nlgx <f.nlgx> [--out DIR]
Выдаёт <stem>_emb_traces.npz (MGZ/MPZ) для gate_npz/eval_mk.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unet_emb import UNetEmb
from extract_nlgx import extract, NULL
Image.MAX_IMAGE_PIXELS = None


def load(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev)
    net = UNetEmb(in_ch=ck.get("in_ch", 4), n_prob=ck.get("n_prob", 2),
                  emb_dim=ck.get("emb_dim", 8), base=ck.get("base", 48))
    net.load_state_dict(ck["model"]); net.to(dev).eval()
    return net, ck


@torch.no_grad()
def predict(net, rgb, dev, y0, y1, tile=256, ov=96, bs=12, edim=8):
    """prob (2,H,W) хэннинг-стич + emb (E,H,W) от БЛИЖАЙШЕГО тайла (не усредняем —
    эмбеддинг-пространство не глобально-консистентно)."""
    import cv2
    H, W, _ = rgb.shape; step = tile - ov
    ys = sorted(set(list(range(0, max(1, H - tile + 1), step)) + [H - tile]))
    ys = [y for y in ys if y + tile > y0 and y < y1]
    xs = sorted(set(list(range(0, max(1, W - tile + 1), step)) + [W - tile]))
    prob = np.zeros((2, H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    emb = np.zeros((edim, H, W), np.float32); ebest = np.full((H, W), -1.0, np.float32)
    win = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32) + 1e-3
    dark = (rgb.max(2) < 110).astype(np.uint8)
    DTf = np.clip(cv2.distanceTransform(dark, cv2.DIST_L2, 5) / 8.0, 0, 1).astype(np.float32)
    rgbf = rgb.astype(np.float32) / 255.0
    batch, pos = [], []
    def flush():
        if not batch: return
        t = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().to(dev)
        with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
            lo, em = net(t)
            pp = torch.sigmoid(lo).float().cpu().numpy()
            ee = em.float().cpu().numpy()
        for (yy, xx), pm, emm in zip(pos, pp, ee):
            prob[:, yy:yy+tile, xx:xx+tile] += pm * win
            wsum[yy:yy+tile, xx:xx+tile] += win
            m = win > ebest[yy:yy+tile, xx:xx+tile]      # ближайший тайл (макс hann) владеет emb
            for c in range(emb.shape[0]):
                sub = emb[c, yy:yy+tile, xx:xx+tile]; sub[m] = emm[c][m]
            eb = ebest[yy:yy+tile, xx:xx+tile]; eb[m] = win[m]
        batch.clear(); pos.clear()
    for yy in ys:
        for xx in xs:
            t3 = rgbf[yy:yy+tile, xx:xx+tile]
            batch.append(np.concatenate([t3, DTf[yy:yy+tile, xx:xx+tile, None]], -1))
            pos.append((yy, xx))
            if len(batch) >= bs: flush()
    flush()
    return prob / np.maximum(wsum, 1e-6)[None], emb


def trace_emb(prob, emb, y0, y1, x0, x1, thr=0.35):
    """Per-row: кандидатные x (prob любого канала > thr) → 2-means по эмбеддингам →
    2 центроида; назначение MGZ/MPZ по среднему prob-каналу кластера."""
    mgz, mpz = {}, {}
    E = emb.shape[0]
    for y in range(y0, y1):
        p0 = prob[0, y, x0:x1]; p1 = prob[1, y, x0:x1]
        act = np.nonzero(np.maximum(p0, p1) > thr)[0]
        if len(act) == 0:
            continue
        xs = act + x0
        if len(act) == 1:
            x = float(xs[0])
            (mgz if p0[act[0]] >= p1[act[0]] else mpz)[y] = x
            continue
        V = emb[:, y, xs].T                              # n×E
        # 2-means (несколько итераций) по эмбеддингам строки
        a, b = V[np.argmin(xs)], V[np.argmax(xs)]        # инициализация крайними
        for _ in range(8):
            da = ((V - a) ** 2).sum(1); db = ((V - b) ** 2).sum(1)
            ga = da <= db
            if ga.all() or (~ga).all():
                break
            a, b = V[ga].mean(0), V[~ga].mean(0)
        da = ((V - a) ** 2).sum(1); db = ((V - b) ** 2).sum(1); ga = da <= db
        if ga.all() or (~ga).all():                     # один кластер — по prob
            x = float((xs * np.maximum(p0[act], p1[act])).sum() / np.maximum(p0[act], p1[act]).sum())
            (mgz if p0[act].mean() >= p1[act].mean() else mpz)[y] = x
            continue
        xa = float(xs[ga].mean()); xb = float(xs[~ga].mean())
        # назначение: кластер с большим средним p0 → MGZ
        s0a = p0[act][ga].mean(); s0b = p0[act][~ga].mean()
        if s0a >= s0b:
            mgz[y] = xa; mpz[y] = xb
        else:
            mgz[y] = xb; mpz[y] = xa
    return mgz, mpz


def main():
    a = sys.argv[1:]
    ckpt, image = a[0], a[1]
    nlgx = a[a.index("--nlgx") + 1] if "--nlgx" in a else None
    out = Path(a[a.index("--out") + 1] if "--out" in a else r"F:\nds\output\mk_data")
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, ck = load(ckpt, dev)
    S = int(ck.get("scale", 1)); tile = int(ck.get("tile", 256))
    print(f"UNetEmb epoch={ck.get('epoch')} dice={ck.get('val_dice'):.3f} emb_sep={ck.get('emb_sep',0):.2f} "
          f"emb_dim={ck.get('emb_dim')} scale={S} tile={tile} dev={dev}")
    rgb = np.asarray(Image.open(image).convert("RGB")); H, W, _ = rgb.shape
    m = extract(nlgx) if nlgx else {}
    da = m.get("depth_axis")
    TY = int(da["top_y"]) if da else 0
    BY = int(da.get("bottom_y") or H - 1) if da else H - 1
    if S != 1:
        import cv2
        rgbP = cv2.resize(rgb, (W * S, H * S), interpolation=cv2.INTER_LINEAR)
        prob, emb = predict(net, rgbP, dev, TY * S, BY * S, tile=tile, ov=tile * 3 // 8,
                            bs=max(2, int(12 * (256 / tile) ** 2)), edim=ck.get("emb_dim", 8))
        # даунсэмпл в натив
        prob = prob[:, ::S, ::S][:, :H, :W]; emb = emb[:, ::S, ::S][:, :H, :W]
    else:
        prob, emb = predict(net, rgb, dev, TY, BY, tile=tile, ov=tile * 3 // 8, edim=ck.get("emb_dim", 8))
    # полоса MK из шкал шаблона (как infer_mk) или по prob-массе
    comb = (prob[0] + prob[1])[TY:BY].sum(0)
    if comb.max() > 0:
        xsnz = np.nonzero(comb > 0.08 * comb.max())[0]
        segs = np.split(xsnz, np.nonzero(np.diff(xsnz) > 30)[0] + 1)
        best = max(segs, key=lambda s: comb[s].sum())
        X0, X1 = max(0, int(best[0]) - 20), min(W, int(best[-1]) + 20)
    else:
        X0, X1 = 0, W
    print(f"MK-полоса x[{X0}..{X1}] prob>0.35 MGZ={100*(prob[0]>0.35).mean():.2f}%")
    mgz, mpz = trace_emb(prob, emb, TY, BY, X0, X1)
    stem = Path(image).stem[:40]
    np.savez(out / f"{stem}_emb_traces.npz",
             mgz_y=np.array(list(mgz)), mgz_x=np.array(list(mgz.values())),
             mpz_y=np.array(list(mpz)), mpz_x=np.array(list(mpz.values())))
    print(f"MGZ={len(mgz)} MPZ={len(mpz)} -> {stem}_emb_traces.npz")


if __name__ == "__main__":
    main()

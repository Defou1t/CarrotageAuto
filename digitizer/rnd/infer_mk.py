r"""Трек 2 / шаг 3 — инференс РАЗДЕЛИТЕЛЯ MK: 2-канальная prob (MGZ к0 / MPZ к1) → 2 трассы.
Канальная идентичность РЕШАЕТ свопы: к0 всегда MGZ (толстая), к1 MPZ — модель учила контекст.
Запуск интерпретатором venv ComfyUI (torch):
  python infer_mk.py <mk_sep.pt> <image.jpg> --nlgx <f.nlgx> [--out DIR]
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unet import UNet
from extract_nlgx import extract
Image.MAX_IMAGE_PIXELS = None


def load2(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev)
    net = UNet(in_ch=ck.get("in_ch", 3), n_classes=ck.get("n_classes", 2), base=ck.get("base", 32))
    net.load_state_dict(ck["model"]); net.to(dev).eval()
    return net, ck


@torch.no_grad()
def predict2(net, rgb, dev, tile=256, ov=96, bs=24, y0=0, y1=None, softmax=False):
    """2-канальная prob (2×H×W float32) тайлово с хэннинг-стичингом."""
    H, W, _ = rgb.shape; y1 = H if y1 is None else min(H, y1); step = tile - ov
    ys = list(range(0, max(1, H - tile + 1), step)) + ([H - tile] if H > tile else [])
    xs = list(range(0, max(1, W - tile + 1), step)) + ([W - tile] if W > tile else [])
    ys = sorted(set(y for y in ys if y + tile > y0 and y < y1)); xs = sorted(set(xs))
    prob = np.zeros((2, H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    win = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32) + 1e-3
    import cv2                                                     # DT-канал (толщина) как при обучении
    dark = (rgb.max(2) < 110).astype(np.uint8)
    DTf = np.clip(cv2.distanceTransform(dark, cv2.DIST_L2, 5) / 8.0, 0, 1).astype(np.float32)
    rgbf = rgb.astype(np.float32) / 255.0
    in4 = net.d1[0].in_channels == 4                              # модель ждёт RGB+DT?
    batch, pos = [], []
    def flush():
        if not batch: return
        t = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().to(dev)
        with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
            o = net(t)
            if softmax:                                            # {фон,MGZ,MPZ} → каналы [MGZ,MPZ]
                p = torch.softmax(o, 1)[:, 1:].float().cpu().numpy()
            else:
                p = torch.sigmoid(o).float().cpu().numpy()         # B×2×T×T
        for (yy, xx), pm in zip(pos, p):
            prob[:, yy:yy+tile, xx:xx+tile] += pm * win
            wsum[yy:yy+tile, xx:xx+tile] += win
        batch.clear(); pos.clear()
    for yy in ys:
        for xx in xs:
            t3 = rgbf[yy:yy+tile, xx:xx+tile]
            tile_in = np.concatenate([t3, DTf[yy:yy+tile, xx:xx+tile, None]], -1) if in4 else t3
            batch.append(tile_in); pos.append((yy, xx))
            if len(batch) >= bs: flush()
    flush()
    return prob / np.maximum(wsum, 1e-6)[None]


def trace_ch(probc, y0, y1, x0, x1, thr=0.4, win=8):
    """per-row x = ПИК канала: argmax prob + субпиксельный центроид в окне ±win вокруг него.
    NB: центроид по ВСЕЙ полосе утягивает посторонней тушью (LEVEN: 268px при разбросной туши на
    всю ширину планшета) — берём именно пик кривого канала, а не среднее массы полосы."""
    out = {}
    for y in range(y0, y1):
        seg = probc[y, x0:x1]
        if seg.max() <= thr:
            continue
        a = int(np.argmax(seg)); lo, hi = max(0, a - win), min(len(seg), a + win + 1)
        w = seg[lo:hi]; out[y] = float(((np.arange(lo, hi) + x0) * w).sum() / w.sum())
    return out


def repair_outliers(probc, tr, S=1, med_win=45, out_thr=15.0, cap_win=12.0, ratio=0.85, thr=0.30,
                    mad_max=8.0):
    """Чинит транзиентные выбросы argmax (диагностика 02.07: 75-90% свопов = одиночные строки,
    точка мимо ОБЕИХ кривых): строка-выброс = |x - локальная медиана канала| > out_thr; замена на
    пик канала В ОКНЕ вокруг медианы — только если там есть сопоставимый prob (ratio от prob
    выброса). Реальный спайк сохраняется: возле медианы prob слабый → замены нет.
    mad_max: где трасса локально нестабильна (MAD окна велик — модель «гуляет», BOGAT),
    медиане верить нельзя — не чиним (иначе ложные починки портят следование)."""
    ys = np.array(sorted(tr)); xs = np.array([tr[y] for y in ys], float)
    if len(ys) < 20:
        return tr, 0
    W = probc.shape[1]
    fixed = 0; new = dict(tr)
    for i, y in enumerate(ys):
        lo, hi = np.searchsorted(ys, y - med_win), np.searchsorted(ys, y + med_win + 1)
        med = float(np.median(xs[lo:hi]))
        if abs(xs[i] - med) <= out_thr * S:
            continue
        if float(np.median(np.abs(xs[lo:hi] - med))) > mad_max * S:
            continue                                             # окно нестабильно — медиана ненадёжна
        w0, w1 = max(0, int(med - cap_win * S)), min(W, int(med + cap_win * S) + 1)
        seg = probc[y, w0:w1]
        if not seg.size:
            continue
        cur_p = float(probc[y, min(W - 1, max(0, int(round(xs[i]))))])
        if seg.max() < max(thr, ratio * cur_p):
            continue                                             # у траектории нет туши — спайк реален
        a = int(np.argmax(seg)); l2, h2 = max(0, a - 8), min(len(seg), a + 9)
        w = seg[l2:h2]
        new[int(y)] = float(((np.arange(l2, h2) + w0) * w).sum() / w.sum())
        fixed += 1
    return new, fixed


def predict_tta(net, rgb, dev, y0, y1, softmax, tta):
    """prob с TTA-ансамблем ориентаций (h/v-флипы не портят толщину-различитель)."""
    ps = [predict2(net, rgb, dev, y0=y0, y1=y1, softmax=softmax)]
    if tta:
        H = rgb.shape[0]
        ps.append(predict2(net, rgb[:, ::-1].copy(), dev, y0=y0, y1=y1, softmax=softmax)[:, :, ::-1])
        ps.append(predict2(net, rgb[::-1].copy(), dev, y0=H - y1, y1=H - y0, softmax=softmax)[:, ::-1, :])
    return np.mean(ps, 0) if len(ps) > 1 else ps[0]


def main():
    a = sys.argv[1:]
    ckpt, image = a[0], a[1]
    nlgx = a[a.index("--nlgx") + 1] if "--nlgx" in a else None
    out = Path(a[a.index("--out") + 1] if "--out" in a else r"F:\nds\output\mk_data")
    save_prob = "--save-prob" in a                               # debug: 227MB prob-карта (по умолчанию выкл)
    continuity = "--continuity" in a                             # пост #1: совместный трекинг 2 прядей
    repair = "--repair" in a                                     # пост #2: constrained re-peak выбросов
    tta = "--tta" in a                                           # ансамбль ориентаций (id+hflip+vflip)
    ens = "--ens" in a                                           # ансамбль чекпойнтов (v4+v5 рядом с ckpt)
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cks = [ckpt]
    if ens:                                                      # чекпойнт-ансамбль: разные лоссы (v4 BCE /
        for sib in ("mk_sep_v4.pt", "mk_sep_v5.pt"):             # v5 softmax) = декоррелированные ошибки
            p = Path(ckpt).with_name(sib)
            if p.exists() and str(p) != str(Path(ckpt)):
                cks.append(str(p))
        cks = list(dict.fromkeys(cks))
    nets = [load2(c, dev) for c in cks]
    net, ck = nets[0]
    print(f"model epoch={ck.get('epoch')} val_dice={ck.get('val_dice'):.3f} device={dev}"
          + (f" ens={len(nets)}" if len(nets) > 1 else "") + (" tta" if tta else ""))
    rgb = np.asarray(Image.open(image).convert("RGB")); H, W, _ = rgb.shape
    if nlgx is None:                                             # авто-поиск парной рамки <stem>.nlgx рядом
        sib = Path(image).with_suffix(".nlgx")
        nlgx = str(sib) if sib.exists() else None
    da = extract(nlgx)["depth_axis"] if nlgx else None           # frameless: без рамки — только полное наложение
    TY = int(da["top_y"]) if da else 0
    BY = int(da.get("bottom_y") or H - 1) if da else H - 1
    def yof(d): return int(da["top_y"] + (d - da["top_depth"]) * (da["bottom_y"] - da["top_y"]) / da["span_depth"])  # span_px=0 дефект
    S = int(ck.get("scale", 1))                                  # модель обучена на ×S апскейле → инференс тоже на ×S
    if S != 1:
        import cv2
        rgbP = cv2.resize(rgb, (W * S, H * S), interpolation=cv2.INTER_LINEAR)
        TYp, BYp, Wp = TY * S, BY * S, W * S
    else:
        rgbP, TYp, BYp, Wp = rgb, TY, BY, W
    prob = np.mean([predict_tta(n_, rgbP, dev, TYp, BYp, bool(c_.get("softmax")), tta)
                    for n_, c_ in nets], 0)
    print(f"prob MGZ>0.4={100*(prob[0]>0.4).mean():.2f}% MPZ>0.4={100*(prob[1]>0.4).mean():.2f}% scale={S}")
    # PER-TRACK band: MK-полоса = плотнейший непрерывный кластер обоих каналов (исключает MBK/чужие треки)
    comb = (prob[0] + prob[1])[TYp:BYp].sum(0)
    if comb.max() > 0:
        xs = np.nonzero(comb > 0.08 * comb.max())[0]
        segs = np.split(xs, np.nonzero(np.diff(xs) > 30 * S)[0] + 1)
        best = max(segs, key=lambda s: comb[s].sum())
        X0, X1 = max(0, int(best[0]) - 20 * S), min(Wp, int(best[-1]) + 20 * S)
    else:
        X0, X1 = 0, Wp
    print(f"MK-полоса x[{X0}..{X1}] (scale-px)")
    def to_native(d):                                            # ×S координаты → нативные (усредняем S под-строк)
        if S == 1: return d
        acc = {}
        for yP, xP in d.items(): acc.setdefault(yP // S, []).append(xP / S)
        return {r: float(np.mean(v)) for r, v in acc.items()}
    stem = Path(image).stem[:40]
    def save_npz(path, dm, dp):
        np.savez(path, mgz_y=np.array(list(dm)), mgz_x=np.array(list(dm.values())),
                 mpz_y=np.array(list(dp)), mpz_x=np.array(list(dp.values())))
    if continuity:                                               # пост #1: DP-пары + линкер скоростью + голос
        from mk_continuity import continuity_traces
        cm, cp, info = continuity_traces(prob, TYp, BYp, X0, X1, S=S)
        mgz, mpz = to_native(cm), to_native(cp)
        print(f"continuity: rows={info.get('rows')} merged={info.get('merged_frac')} "
              f"vote_rows={info.get('vote_rows')} margin={info.get('vote_margin')}")
        pk_m = to_native(trace_ch(prob[0], TYp, BYp, X0, X1))    # baseline из ТОЙ ЖЕ prob — честный A/B
        pk_p = to_native(trace_ch(prob[1], TYp, BYp, X0, X1))
        save_npz(out / f"{stem}_traces_peak.npz", pk_m, pk_p)
    else:
        tm = trace_ch(prob[0], TYp, BYp, X0, X1)
        tp = trace_ch(prob[1], TYp, BYp, X0, X1)
        if repair:
            save_npz(out / f"{stem}_traces_peak.npz", to_native(tm), to_native(tp))  # база для A/B
            tm, fm = repair_outliers(prob[0], tm, S=S)
            tp, fp = repair_outliers(prob[1], tp, S=S)
            print(f"repair: MGZ исправлено {fm} строк, MPZ {fp}")
        mgz, mpz = to_native(tm), to_native(tp)
    if save_prob:
        np.save(out / "mk_prob2.npy", prob.astype(np.float16))
    save_npz(out / f"{stem}_traces.npz", mgz, mpz)               # для объективного eval_mk vs GT
    print(f"MGZ точек={len(mgz)} MPZ={len(mpz)} | трассы -> {stem}_traces.npz")
    COLM, COLP = (220, 0, 0), (0, 110, 230)
    def draw(ov, ox, oy):
        for d, col in ((mgz, COLM), (mpz, COLP)):
            for y, x in d.items():
                xi = int(x)
                if 0 <= y - oy < ov.shape[0] and 0 <= xi - ox < ov.shape[1]:
                    ov[y - oy, max(0, xi - ox - 1):xi - ox + 2] = col
    try: FONT = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 24)
    except Exception: FONT = ImageFont.load_default()
    bx0, bx1 = max(0, X0 // S - 15), min(W, X1 // S + 15)         # полоса MK в НАТИВНЫХ координатах
    full = rgb.copy(); draw(full, 0, 0)
    Image.fromarray(full[TY:BY, bx0:bx1]).save(out / f"{stem}_overlay.png")
    print(f"наложение -> {stem}_overlay.png")
    if da:                                                        # кропы по глубине — только при наличии рамки
        td = da["top_depth"]; bd = td + da["span_depth"]          # глубины равномерно по диапазону скважины
        for d in np.linspace(td, bd, 8)[1:-1]:                    # 6 кропов внутри [top,bottom]
            d = round(float(d)); yc = yof(d); y0, y1 = yc - 130, yc + 130
            if not (0 <= y0 and y1 < H): continue
            crop = rgb[y0:y1, bx0:bx1].copy(); draw(crop, bx0, y0)
            im_c = Image.fromarray(crop).resize(((bx1 - bx0) * 3, 260 * 3), Image.NEAREST)
            ImageDraw.Draw(im_c).text((6, 4), f"{d}м NN MGZ=красн MPZ=син", fill=(0, 0, 0), font=FONT)
            im_c.save(out / f"{stem}_{d}m.png"); print(f"  кроп -> {stem}_{d}m.png")


if __name__ == "__main__":
    main()

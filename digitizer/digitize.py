r"""
digitize.py — ЕДИНЫЙ ПРОХОД: скан → сдаваемый _auto.nlgx за один вызов (без pickle-кэша).
Сливает cache_tracks.py (U-Net + трекер) и inject_trace.py (вписать трассы в nlgx-шаблон).

Конвейер:
  extract(template.nlgx)         калибровка/сегменты/определения кривых + GT-трасса эксперта
  find_image / --image           растровый планшет
  predict_prob (U-Net)           вероятностная карта штриха по окну Depth Axis
  track_multi.track              разделение НАЛОЖЕННЫХ кривых геометрическим трекером
  match (близость к GT шаблона)   сопоставить линии трекера с кривыми
  set_tag(35490)                 заменить X-по-строкам каждой кривой трассой трекера
  write_full → _auto.nlgx (+bck) + верификация (переоткрыть, сверить xs)

Шаблон даёт калибровку/уровни/значения (трекер даёт ФОРМУ). GT-трасса шаблона нужна
только для матчинга линий к кривым. Результат открывается в NeuraLOG для визуального QC.
Обобщение inject_trace: nlgx-шаблон берётся ПО ПУТИ (--nlgx), а не глобом по Archive.

ЗАПУСК (venv ComfyUI python, torch):
  python digitize.py <ckpt> --nlgx <template.nlgx> [--image <f.jpg>] [--out DIR]
                     [--lam 0.15] [--device cuda|cpu]
"""
import sys, struct
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from infer import load_model, predict_prob
from track_multi import track
from write_nlgx import read_full, write_full, write_bck, set_tag, find_ifd
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image

Image.MAX_IMAGE_PIXELS = None


def build_gts(m, y0, y1):
    """name(short) -> {row: x} по реальным кривым шаблона в окне [y0,y1)."""
    gts = {}
    for c in ds.real_curves(m):
        ty = c["top_y"]
        d = {ty+i: x for i, x in enumerate(c["xs"]) if x != NULL and y0 <= ty+i < y1}
        if len(d) >= 20:
            gts[c["name"].split()[0]] = d
    return gts


def best_line(lines, gd):
    """Индекс линии трекера, ближайшей к GT-трассе кривой (медиана |dx|)."""
    best = (1e9, None)
    for li, L in enumerate(lines):
        common = [y for y in gd if y in L]
        if len(common) < 15:
            continue
        err = float(np.median([abs(L[y]-gd[y]) for y in common]))
        if err < best[0]:
            best = (err, li)
    return best


def curve_pred(short):
    """Предикат IFD кривой type7 с именем, начинающимся с short."""
    def is_this(tags):
        if 34768 not in tags or struct.unpack("<I", tags[34768][2][:4])[0] != 7:
            return False
        if 35470 not in tags:
            return False
        nm = tags[35470][2].split(b"\x00")[0].decode("latin1")
        return nm.startswith(short + " ") or nm == short
    return is_this


def digitize(ckpt, nlgx, image=None, out=None, lam=0.15, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nlgx = str(nlgx)
    out = Path(out) if out else Path(r"F:\nds\output")
    out.mkdir(parents=True, exist_ok=True)

    m = extract(nlgx)
    img = Path(image) if image else find_image(Path(nlgx))
    if not img or not Path(img).is_file():
        print(f"не найдена картинка для {nlgx} (укажите --image)"); return None
    rgb = np.asarray(Image.open(img).convert("RGB"))
    H, W, _ = rgb.shape
    da = m["depth_axis"]
    y0, y1 = da["top_y"], min(H, da["bottom_y"])

    net, ck = load_model(ckpt, device)
    print(f"model {Path(ckpt).name} (epoch {ck.get('epoch')}, val_dice {ck.get('val_dice')}) device={device}")
    print(f"скан {Path(img).name}  {W}x{H}  окно глубины y[{y0}:{y1}]")

    prob = predict_prob(net, rgb, device, y0=y0, y1=y1)
    raw_lines, N = track(prob, rgb, y0, y1, lam=lam)
    lines = [dict(L["tr"]) for L in raw_lines]
    gts = build_gts(m, y0, y1)
    print(f"трекер: N={N} линий={len(lines)}  GT-кривых={len(gts)}")

    # вписать в копию шаблона
    ifds = read_full(open(nlgx, "rb").read())
    written = []
    for c in ds.real_curves(m):
        short = c["name"].split()[0]
        gd = gts.get(short)
        if not gd:
            continue
        err, li = best_line(lines, gd)
        if li is None:
            continue
        L = lines[li]
        top_y = c["top_y"]; n = len(c["xs"])
        new_xs = [int(round(L[top_y+i])) if (top_y+i) in L else NULL for i in range(n)]
        idxs = find_ifd(ifds, curve_pred(short))
        if not idxs:
            continue
        set_tag(ifds, idxs[0], 35490, 4, new_xs)
        written.append((short, err, sum(1 for x in new_xs if x != NULL)))

    data = write_full(ifds)
    stem = Path(nlgx).stem
    dst = out / f"{stem}_auto.nlgx"
    open(dst, "wb").write(data)
    open(out / f"{stem}_auto.bck", "wb").write(write_bck(data))

    print(f"\nвписано трасс трекера: {len(written)}")
    for short, err, npts in written:
        print(f"  {short:<9} (track err≈{err:.1f}px, {npts} точек)")

    # верификация: переоткрыть и сверить xs с трассой трекера
    m2 = extract(str(dst))
    ok = 0
    for c2 in ds.real_curves(m2):
        short = c2["name"].split()[0]
        gd = gts.get(short)
        if not gd:
            continue
        err, li = best_line(lines, gd)
        if li is None:
            continue
        L = lines[li]
        common = [(c2["xs"][y - c2["top_y"]], L[y]) for y in gd
                  if y in L and 0 <= y-c2["top_y"] < len(c2["xs"]) and c2["xs"][y-c2["top_y"]] != NULL]
        if common and np.median([abs(a-b) for a, b in common]) < 1.5:
            ok += 1
    print(f"верификация: {ok}/{len(written)} кривых в _auto.nlgx соответствуют трассе трекера")
    print(f"-> {dst}")
    return dst


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    ckpt = sys.argv[1]; args = sys.argv[2:]
    nlgx = image = out = None
    lam = 0.15; device = "cuda" if torch.cuda.is_available() else "cpu"
    i = 0
    while i < len(args):
        if args[i] == "--nlgx": nlgx = args[i+1]; i += 2
        elif args[i] == "--image": image = args[i+1]; i += 2
        elif args[i] == "--out": out = args[i+1]; i += 2
        elif args[i] == "--lam": lam = float(args[i+1]); i += 2
        elif args[i] == "--device": device = args[i+1]; i += 2
        else: i += 1
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if not nlgx:
        print("нужен --nlgx <template.nlgx>"); return
    digitize(ckpt, nlgx, image, out, lam, device)


if __name__ == "__main__":
    main()

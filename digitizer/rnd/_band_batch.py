r"""
_band_batch.py — БАТЧ-ВАЛИДАЦИЯ band-детекта (A): сколько кривых становится АВТО благодаря разводу полос.
Для каждого combo-лога: U-Net prob → band-детект → классификация → трасса single → сверка с GT каждой кривой.
Метрика per-curve: AUTO (своя single-полоса, |Δx|≤5px) / FLAG (multi-полоса) / MISS (полоса не разделила/неточно).

ComfyUI-python _band_batch.py
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\output")
import _band_digitize as B
from extract_nlgx import extract, NULL
from dataset_build import find_image

FILES = [
    r"F:\nds\output\test_single\Yatskivska_1_RK_3180_4080_200_D1.nlgx",
    r"F:\nds\projects\Archive\KREMEN_083\wlg\KREMEN_083_RK, AK, DS_0800-1560_200_1998-07-08_D_1.nlgx",
    r"F:\nds\projects\Archive\YULIIV_062\wlg\YULIIV_062_RK, AK, DS, PS_3220-3760_200_1999-10-01_D_1.nlgx",
    r"F:\nds\projects\Archive\BEZLUD_051\wlg\BEZLUD_051_RK, AK, DS_2688-3142_500_1998-08-05_D_1.nlgx",
    r"F:\nds\projects\Archive\KREMEN_089\wlg\KREMEN_089_RK, AK, DS, SP_0785-1493_200_1999-08-10_D_1.nlgx",
]


def process(nlgx):
    m = extract(nlgx); da = m["depth_axis"]
    img = find_image(Path(nlgx)) or m.get("img_path")
    if not img or not Path(img).is_file():
        return None
    gray = np.asarray(Image.open(img).convert("L")); H, W = gray.shape
    ty, by = da["top_y"], min(H, da["bottom_y"]); bg = int(np.median(gray[::7, ::7]))
    prob = B.get_prob(nlgx, m, img, ty, by, [])[:H, :W]
    gate = prob > 0.30; ink_g = (gray < bg - 45) & gate
    weight = ((bg - gray.astype(np.int16)).clip(0).astype(np.float32)) * gate * (prob > 0.4)
    curves = [c for c in m["curves"] if not c["name"].split()[0].startswith("DA")]
    xL = min(s["x_left"] for s in m["scale_axes"]); xR = min(W, max(s["x_right"] for s in m["scale_axes"]))
    gt = {c["name"].split()[0]: {c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL} for c in curves}
    gt = {k: v for k, v in gt.items() if len(v) >= 20}
    gtmed = {k: float(np.median(list(v.values()))) for k, v in gt.items()}
    dens = prob[ty:by, xL:xR].sum(0)
    bands = B.band_detect(dens, xL)
    bands = B.refine_bands(bands, dens, xL, ink_g, ty, by)  # C: добить тесные кластеры
    binfo = []
    for (xa, xb) in bands:
        kind, mf, mw = B.classify(ink_g, xa, xb, ty, by)
        tr = B.trace_single(weight, prob, xa, xb, ty, by) if kind == "single" else None
        binfo.append((xa, xb, kind, tr))
    # per-curve статус
    out = []
    for nm, mx in sorted(gtmed.items(), key=lambda kv: kv[1]):
        bi = next((b for b in binfo if b[0] <= mx < b[1]), None)
        if bi is None:
            out.append((nm, mx, "MISS", None, "нет полосы")); continue
        xa, xb, kind, tr = bi
        if kind != "single":
            out.append((nm, mx, "FLAG", None, f"multi [{xa}..{xb}]")); continue
        common = [y for y in gt[nm] if y in tr] if tr else []
        if not common:
            out.append((nm, mx, "MISS", None, "трасса пуста")); continue
        e = np.array([abs(tr[y] - gt[nm][y]) for y in common])
        med = float(np.median(e))
        st = "AUTO" if med <= 5 else "MISS"
        out.append((nm, mx, st, med, f"[{xa}..{xb}]"))
    return Path(nlgx).stem, len(bands), out


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    A = T = F = M = 0
    for f in FILES:
        try:
            r = process(f)
        except Exception as e:
            print(f"! {Path(f).stem[:40]}: {str(e)[:60]}"); continue
        if r is None:
            print(f"! {Path(f).stem[:40]}: нет картинки"); continue
        stem, nb, rows = r
        print(f"\n=== {stem[:50]} | полос={nb}, кривых={len(rows)} ===")
        for nm, mx, st, med, info in rows:
            md = f"{med:.1f}px" if med is not None else "-"
            print(f"   {nm:<7}@{int(mx):<5} {st:<5} {md:>7}  {info}")
            T += 1; A += (st == "AUTO"); F += (st == "FLAG"); M += (st == "MISS")
    print(f"\n===== ИТОГ: кривых={T} | AUTO={A} ({100*A/max(1,T):.0f}%) | FLAG_multi={F} | MISS={M} =====")
    print("AUTO = разведена в свою полосу и трассирована ≤5px (рост авто-бакета за счёт band-детекта)")


if __name__ == "__main__":
    main()

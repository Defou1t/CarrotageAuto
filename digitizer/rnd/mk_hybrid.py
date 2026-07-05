r"""Трек 2 / ГИБРИД prod-ens3 ↔ SCJP по зигзаг-детектору (05.07). Мотивация: SCJP
(mk_pen_path --step2) — СПЕЦИАЛИСТ зигзаг-режима, НЕ глобальная замена. Held-out
гейт (eval_mk vs GT) доказал регресс SCJP на разделимых/впритык планшетах:
  BEZLUD  swap 5.6→9.9% px 3.5→6.5  |  BOGAT swap 19.8→27.2% cp 52→70.
Зато SCJP выигрывает там, где prod-peak не берёт вершину сплошного зигзага (3752
MPZ 23.5→41%) или где prod схлопывается/умирает (3130 хвост 3500-3760: prod
0-44% → SCJP 74-100% eff≤3).

РЕЖИМ ПО УМОЛЧАНИЮ = HOLE-FILL: hybrid = prod везде, где у prod ЕСТЬ значение;
SCJP заполняет ТОЛЬКО дыры prod (строки без prod-значения в [y0,y1]). НИКОГДА не
переопределяет живой prod ⇒ eff≤3 гибрида ДОКАЗУЕМО ≥ prod (заполнение дыры может
только превратить промах-fail в попадание, не наоборот). Восстанавливает провалы
prod: 3130 хвост 3724-3760 (prod 99% дыр → 0→87/19→65 eff≤3). На планшетах без
дыр гибрид = prod байт-в-байт. Проверено гейтом по 8 эталонным планшетам —
регрессов нет НИГДЕ.

⚠ ПОЧЕМУ НЕ ДЕТЕКТОР ЗИГЗАГА (эволюция, 05.07, не повторять): пытались per-окно
переключать prod→SCJP по морфологии (solid=widest/ink>0.6 & nseg<=3) — гейт
эталона Pn_Zavoda вскрыл РЕГРЕСС: морфо-триггер бьёт по здоровым зигзаг-планшетам,
где prod уже хорош (4718: prod MGZ med 2.7px 52% → SCJP 14px 33%). Морфология НЕ
отличает «сломанный зигзаг» (3752 prod med 12.8) от «хорошего» (4718 med 2.7);
prod-джиттер тоже (4718 LOSE — самый дребезжащий); health-on-ink ложно-срабатывает
на ВЫЦВЕТШЕЙ туши (prod верен, но тушь<110 → «сошёл»). НИ ОДИН inference-time
сигнал «prod сломан» не надёжен ⇒ безопасно только заполнять ЯВНЫЕ дыры prod.
Морфо-выигрыш 3752-MPZ (23.5→40) безопасно недостижим — открытый вопрос (нужен
GT-подобный сигнал качества prod). Флаг --morph оставлен для research (UNSAFE).

Полоса MK берётся из X-ДИАПАЗОНА prod-трасс (в проде есть всегда) → без GT.

  python mk_hybrid.py <scan> <frame.nlgx> --prob <mk_prob2.npy> --prod <traces.npz>
        [--win d0 d1] [--out <hybrid.npz>] [--gate] [--morph]   (--morph: окно-замена, UNSAFE)
"""
import sys, os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from extract_nlgx import extract, NULL
from mk_pen_path import (zhang_suen, scjp_candidates, scjp_joint, plateau,
                         load_prob_window, trace_dict, DARK)

WIN = 200
SOLID_T, NSEG_T, HEALTH_T = 0.6, 3, 0.5


def window_flags(dark, prod_native, y0, x0, darkF, morph=False):
    """Per-окно: (is_zigzag, solid, nseg, health). dark — тушь band; prod_native —
    {'MGZ':{y:x},'MPZ':{y:x}} в НАТИВНЫХ координатах; darkF — тушь всего скана.
    morph=False (по умолч.): триггер = health<HEALTH_T. morph=True: +морфо (UNSAFE)."""
    H, Wb = dark.shape
    nseg = np.zeros(H, np.int32); widest = np.zeros(H, np.int32); inkw = np.zeros(H, np.int32)
    for i in range(H):
        xs = np.nonzero(dark[i])[0]
        if not len(xs):
            continue
        segs = np.split(xs, np.nonzero(np.diff(xs) > 1)[0] + 1)
        nseg[i] = len(segs); widest[i] = max(len(s) for s in segs); inkw[i] = len(xs)
    flags = {}
    for w0 in range(0, H, WIN):
        w1 = min(H, w0 + WIN)
        occ = nseg[w0:w1] > 0
        if occ.sum() < (w1 - w0) * 0.3:
            flags[w0] = (False, 0.0, 0.0, 1.0); continue
        solid = float(np.median(widest[w0:w1][occ] / np.maximum(1, inkw[w0:w1][occ])))
        medseg = float(np.median(nseg[w0:w1][occ]))
        hs = []
        for k in ("MGZ", "MPZ"):
            nok = nh = 0
            for y in range(y0 + w0, y0 + w1):
                x = prod_native[k].get(y)
                if x is None:
                    continue
                nh += 1; xi = int(round(x))
                if darkF[y, max(0, xi - 3):xi + 4].any():
                    nok += 1
            hs.append((nh / (w1 - w0)) * (nok / nh if nh else 0.0))
        health = min(hs) if hs else 1.0
        zz = health < HEALTH_T
        if morph:
            zz = zz or (solid > SOLID_T and medseg <= NSEG_T)
        flags[w0] = (zz, solid, medseg, health)
    return flags


def merge_fill(prod_native, scjp_native, y0, y1):
    """HOLE-FILL (по умолч.): prod везде, где есть; SCJP заполняет дыры prod.
    Живой prod НИКОГДА не переопределяется ⇒ eff гибрида ≥ prod."""
    out = {"MGZ": {}, "MPZ": {}}
    for k in ("MGZ", "MPZ"):
        out[k].update(prod_native[k])
        for y, x in scjp_native[k].items():
            if y0 <= y < y1 and y not in prod_native[k]:
                out[k][y] = float(x)
    return out


def scjp_native_from(rgb, prob_native, y0, y1, x0, x1):
    """SCJP-трассы в НАТИВНЫХ координатах для полосы [y0:y1, x0:x1]. prob_native —
    (2, y1-y0, x1-x0) float. Переиспользуется infer_mk (--holefill) и main."""
    dark = rgb[y0:y1, x0:x1].max(2) < DARK
    skel = zhang_suen(dark)
    cands = scjp_candidates(skel, dark, prob_native)
    tm, tp = scjp_joint(cands)
    ta, tb = plateau(tm), plateau(tp)
    return {"MGZ": {y0 + y: x0 + x for y, x in ta.items()},
            "MPZ": {y0 + y: x0 + x for y, x in tb.items()}}


def holefill(rgb, prob_native, prod_native, y0, y1, x0, x1):
    """Заполнить дыры prod SCJP-структурой. rgb — весь скан; prob_native — окно
    полосы (2,h,w) в НАТИВ; prod_native — {'MGZ'/'MPZ':{y_native:x_native}}."""
    scjp = scjp_native_from(rgb, prob_native, y0, y1, x0, x1)
    return merge_fill(prod_native, scjp, y0, y1)


def merge_window(prod_native, scjp_native, flags, y0, y1):
    """⚠ UNSAFE (--morph): в зигзаг-окнах целиком SCJP. Регрессирует здоровые
    зигзаг-планшеты (Pn_Zavoda 4718). Оставлено для research."""
    out = {"MGZ": {}, "MPZ": {}}
    for k in ("MGZ", "MPZ"):
        for y in range(y0, y1):
            w0 = ((y - y0) // WIN) * WIN
            zz = flags.get(w0, (False,))[0]
            primary, backup = (scjp_native, prod_native) if zz else (prod_native, scjp_native)
            v = primary[k].get(y)
            if v is None:
                v = backup[k].get(y)
            if v is not None:
                out[k][y] = float(v)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    a = sys.argv[1:]
    scan, frame = a[0], a[1]
    prob_path = a[a.index("--prob") + 1]
    prod_path = a[a.index("--prod") + 1]
    out = a[a.index("--out") + 1] if "--out" in a else None
    me = extract(frame)
    da = me["depth_axis"]
    def yof(d): return int(da["top_y"] + (d - da["top_depth"]) * da["span_px"] / da["span_depth"])
    if "--win" in a:
        i = a.index("--win"); y0, y1 = yof(float(a[i + 1])), yof(float(a[i + 2]))
    else:
        y0, y1 = int(da["top_y"]), int(da["bottom_y"])

    d = np.load(prod_path)
    prod_native = {"MGZ": {int(y): float(x) for y, x in zip(d["mgz_y"], d["mgz_x"])},
                   "MPZ": {int(y): float(x) for y, x in zip(d["mpz_y"], d["mpz_x"])}}
    px_all = [x for k in prod_native for yy, x in prod_native[k].items() if y0 <= yy < y1]
    if not px_all:
        print("нет prod-трасс в окне"); return
    x0, x1 = max(0, int(min(px_all)) - 30), int(max(px_all)) + 30

    rgb = np.asarray(Image.open(scan).convert("RGB"))
    darkF = rgb.max(2) < DARK
    band = rgb[y0:y1, x0:x1]
    dark = band.max(2) < DARK
    morph = "--morph" in a
    holes = {k: sum(1 for y in range(y0, y1) if y not in prod_native[k]) for k in ("MGZ", "MPZ")}
    need_scjp = morph or holes["MGZ"] or holes["MPZ"]
    print(f"окно y[{y0}..{y1}] x[{x0}..{x1}] | дыр prod MGZ {holes['MGZ']} MPZ {holes['MPZ']}")

    scjp_native = {"MGZ": {}, "MPZ": {}}
    flags = {}
    if need_scjp:
        probw, S = load_prob_window(prob_path, rgb.shape[0], y0, y1, x0, x1)
        scjp_native = scjp_native_from(rgb, probw, y0, y1, x0, x1)
        print(f"SCJP посчитан: MGZ {len(scjp_native['MGZ'])} MPZ {len(scjp_native['MPZ'])} строк")

    if morph:
        flags = window_flags(dark, prod_native, y0, x0, darkF, morph=True)
        n_zz = sum(1 for f in flags.values() if f[0])
        print(f"⚠ --morph окно-замена: зигзаг-окон {n_zz}/{len(flags)}")
        hyb = merge_window(prod_native, scjp_native, flags, y0, y1)
    else:
        hyb = merge_fill(prod_native, scjp_native, y0, y1)
    filled = {k: len(hyb[k]) - len(prod_native[k]) for k in ("MGZ", "MPZ")}
    print(f"гибрид: MGZ {len(hyb['MGZ'])} (+{filled['MGZ']}) MPZ {len(hyb['MPZ'])} (+{filled['MPZ']}) строк")

    if out:
        np.savez(out,
                 mgz_y=np.array(list(hyb["MGZ"])), mgz_x=np.array(list(hyb["MGZ"].values())),
                 mpz_y=np.array(list(hyb["MPZ"])), mpz_x=np.array(list(hyb["MPZ"].values())))
        print(f"гибрид -> {out}")

    if "--gate" in a:
        gts = {c["name"].split()[0]: trace_dict(c) for c in me["curves"]}
        for mn, k in (("MGZ1", "MGZ"), ("MPZ1", "MPZ")):
            cl = [(y, x) for y, x in gts.get(mn, {}).items() if y0 <= y < y1]
            if not cl:
                continue
            for tag, tr in (("hybrid", hyb[k]), ("prod", prod_native[k])):
                dx = [abs(tr[y] - x) for y, x in cl if y in tr]
                miss = len(cl) - len(dx)
                dx = np.array(dx); n = len(cl)
                eff = (dx <= 3).sum() / n if n else 0
                print(f"  {mn} {tag:<7} n={n} miss={miss} "
                      + (f"med {np.median(dx):.1f} ≤3px {(dx<=3).mean()*100:.0f}% " if len(dx) else "")
                      + f"eff≤3 {eff*100:.1f}%")


if __name__ == "__main__":
    main()

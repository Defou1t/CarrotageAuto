"""BKZ, намеченный рычаг: СТРУКТУРНОЕ ×1↔×5 паррование на уровне СТРАНДОВ (не per-row).

Факт (STEP-1C): каждый градиент-зонд физически нарисован ДВУМЯ нитями на всю глубину —
×1-база и ×5-перевынос; эксперт трассирует ту, что ON-SCALE, вторую рисуют но не трассируют.
Геометрия шкал даёт жёсткую связь: обе нити одного зонда несут ОДНО значение, поэтому
x_base и x_5x связаны как x5 = xL + (x1 - xL)/5 (шкалы делят пиксельный диапазон).

Идея: странды M1 → для каждой ПАРЫ странд(a,b) проверить гипотезу «это одна кривая на двух
шкалах»: на общих строках value_k(a) ≈ value_j(b) при (k,j) из семейства. Скор = доля строк
с согласием. Так собираются ГРУППЫ странд = один зонд, и назначение делается по группам,
а не per-row (что и ломало: swap на level-скачках).

Диагностика: сколько странд, сколько парных гипотез подтверждается, покрывают ли группы GT.
python bkz_pair.py <nlgx> [--win y0 y1]"""
import sys; sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from extract_nlgx import extract, NULL
from decode_levels import build_family, gt_levels, scale_map
import bkz_track as BT
from dataset_build import find_image

ET = Path(r"F:\nds\projects\Archive\Yatskivska_001\wlg")


def strand_values(st, fam, k):
    """значения странда, прочитанные на шкале k"""
    mp = scale_map(fam[k])
    return {y: mp(x) for y, x in st.items()}


def pair_score(a, b, fam, ka, kb, tol=0.12, min_rows=40):
    """доля общих строк, где значение странда a на шкале ka ≈ значению b на шкале kb."""
    common = [y for y in a if y in b]
    if len(common) < min_rows:
        return 0.0, 0
    va = strand_values(a, fam, ka); vb = strand_values(b, fam, kb)
    ok = 0
    for y in common:
        x, z = abs(va[y]), abs(vb[y])
        m = max(x, z, 1e-6)
        if abs(x - z) / m <= tol:
            ok += 1
    return ok / len(common), len(common)


def main():
    nlgx = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ET / "Yatskivska_1_BKZ_3160_3570_200_D1.nlgx"
    if not nlgx.is_file():
        cand = sorted(ET.glob("*BKZ*D1.nlgx"))
        if not cand:
            print("нет BKZ-эталона"); return
        nlgx = cand[0]
    m = extract(str(nlgx)); img = find_image(nlgx)
    print(f"планшет: {nlgx.stem}")
    rgb = np.asarray(Image.open(img).convert("RGB"))
    mask = BT.ink_mask(rgb)
    tok = BT.PAIRS["D1"] if "_D1" in nlgx.stem else BT.PAIRS["D2"]
    rng = BT.track_x_range(m, tok)
    if rng is None:
        print("нет целевых кривых", tok); return
    x0, x1 = rng
    da = m["depth_axis"]; y0, y1 = da["top_y"], da["bottom_y"]
    if "--win" in sys.argv:
        i = sys.argv.index("--win"); y0, y1 = int(sys.argv[i+1]), int(sys.argv[i+2])
    print(f"трек x[{x0}..{x1}] строки {y0}..{y1}, кривые {tok}")
    strands = BT.link_strands(mask, x0, x1, y0, y1)
    print(f"странд M1: {len(strands)} (длиннейшие: {[len(s) for s in strands[:8]]})")
    curves = {c["name"].split()[0]: c for c in m["curves"]}
    for t in tok:
        c = curves.get(t)
        if c is None:
            continue
        fam = build_family(m, c); gl = gt_levels(c)
        gt = BT.curve_rows(c)
        lv = {y: gl.get(y, 0) for y in gt}
        print(f"\n--- {t}: fam={len(fam)}, GT-строк={len(gt)}, уровни={sorted(set(lv.values()))}")
        # какие странды покрывают GT (и на каком уровне)
        cover = []
        for i, st in enumerate(strands[:14]):
            common = [y for y in st if y in gt]
            if len(common) < 40:
                continue
            d = np.array([abs(st[y] - gt[y]) for y in common])
            frac3 = float((d <= 3).mean())
            if frac3 > 0.2:
                lvs = {}
                for y in common:
                    lvs[lv.get(y, 0)] = lvs.get(lv.get(y, 0), 0) + 1
                cover.append((i, len(common), round(frac3, 2), lvs))
        print(f"  странды, лежащие на GT (>20% строк ≤3px): {cover}")
        # ПАРНЫЕ ГИПОТЕЗЫ ×1<->×5 между странами
        print("  парные гипотезы (score, общих строк) для (k_a,k_b):")
        best = []
        for i in range(min(10, len(strands))):
            for j in range(i + 1, min(10, len(strands))):
                for ka in range(len(fam)):
                    for kb in range(len(fam)):
                        if ka == kb:
                            continue
                        s, n = pair_score(strands[i], strands[j], fam, ka, kb)
                        if s > 0.5 and n >= 60:
                            best.append((round(s, 2), n, i, j, ka, kb))
        best.sort(reverse=True)
        for b in best[:10]:
            print(f"     score={b[0]} rows={b[1]}  странд{b[2]}(шкала{b[4]}) ~ странд{b[3]}(шкала{b[5]})")
        if not best:
            print("     (нет подтверждённых пар)")


if __name__ == "__main__":
    main()

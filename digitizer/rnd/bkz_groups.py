"""BKZ шаг 2: СБОРКА ГРУПП странд (= один зонд на двух шкалах) и назначение кривых по группам.

Из bkz_pair: пары странд связываются геометрией шкал (значение на шкале k ≈ значению соседа на j).
Здесь: строим граф пар → компоненты связности = ГРУППЫ (кривая целиком, обе шкалы) → каждой
группе присваиваем on-scale x-per-row (странд + его шкала) → сверяем группы с GT обеих кривых.

Ключ: назначение по ГРУППАМ (а не per-row) должно убрать свопы на level-скачках — главный
дефект sequential/joint DP (bkz-separator-handoff).
python bkz_groups.py <nlgx> [--win y0 y1]"""
import sys; sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
sys.path.insert(0, r"C:\Users\Defou1t\AppData\Local\Temp\claude\F--nds\34d43bd5-fb8e-4e03-a36a-8c353bc90cfd\scratchpad")
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from extract_nlgx import extract, NULL
from decode_levels import build_family, gt_levels, scale_map
import bkz_track as BT
from bkz_pair import pair_score
from dataset_build import find_image


def build_groups(strands, fam, thr=0.65, min_rows=60, topn=14):
    """Граф пар (i,ki)-(j,kj) → компоненты. Узел = (индекс странда, шкала)."""
    edges = []
    N = min(topn, len(strands))
    for i in range(N):
        for j in range(i + 1, N):
            best = None
            for ka in range(len(fam)):
                for kb in range(len(fam)):
                    if ka == kb:
                        continue
                    s, n = pair_score(strands[i], strands[j], fam, ka, kb)
                    if n >= min_rows and s >= thr and (best is None or s > best[0]):
                        best = (s, ka, kb, n)
            if best:
                edges.append((i, j, best[1], best[2], best[0], best[3]))
    # union-find по узлам (странд, шкала)
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i, j, ka, kb, s, n in edges:
        union((i, ka), (j, kb))
    groups = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    return edges, [g for g in groups.values() if len(g) >= 2]


def group_trace(strands, group, fam):
    """x-per-row группы, приведённый к ЗНАЧЕНИЮ (шкала каждого узла своя)."""
    val = {}
    for (i, k) in group:
        mp = scale_map(fam[k])
        for y, x in strands[i].items():
            val.setdefault(y, []).append(abs(mp(x)))
    return {y: float(np.median(v)) for y, v in val.items()}


def main():
    nlgx = Path(sys.argv[1])
    m = extract(str(nlgx)); img = find_image(nlgx)
    rgb = np.asarray(Image.open(img).convert("RGB"))
    mask = BT.ink_mask(rgb)
    tok = BT.PAIRS["D1"] if "_D1" in nlgx.stem else BT.PAIRS["D2"]
    x0, x1 = BT.track_x_range(m, tok)
    da = m["depth_axis"]; y0, y1 = da["top_y"], da["bottom_y"]
    if "--win" in sys.argv:
        i = sys.argv.index("--win"); y0, y1 = int(sys.argv[i+1]), int(sys.argv[i+2])
    strands = BT.link_strands(mask, x0, x1, y0, y1)
    curves = {c["name"].split()[0]: c for c in m["curves"]}
    c0 = curves[tok[0]]
    fam = build_family(m, c0)
    edges, groups = build_groups(strands, fam)
    print(f"{nlgx.stem}  странд={len(strands)}  пар-рёбер={len(edges)}  групп={len(groups)}")
    for gi, g in enumerate(sorted(groups, key=lambda g: -sum(len(strands[i]) for i, _ in g))[:6]):
        rows = group_trace(strands, g, fam)
        print(f"  группа{gi}: узлы={sorted(g)} строк={len(rows)}")
        # сверка с GT обеих кривых ПО ЗНАЧЕНИЮ
        for t in tok:
            c = curves.get(t)
            if c is None:
                continue
            gl = gt_levels(c); gt = BT.curve_rows(c)
            famc = build_family(m, c)
            mps = [scale_map(s) for s in famc]
            gval = {y: abs(mps[min(gl.get(y, 0), len(famc)-1)](x)) for y, x in gt.items()}
            common = [y for y in rows if y in gval]
            if len(common) < 50:
                continue
            rel = np.array([abs(rows[y] - gval[y]) / max(gval[y], 1e-6) for y in common])
            print(f"     vs {t}: строк={len(common)} доля|Δv|/v<=0.05: {(rel<=0.05).mean()*100:.0f}%"
                  f"  <=0.12: {(rel<=0.12).mean()*100:.0f}%")


if __name__ == "__main__":
    main()

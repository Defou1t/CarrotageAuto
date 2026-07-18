r"""bkz_gate_groups.py — ШИРОКИЙ ГЕЙТ группового подхода BKZ на ВСЕХ эталонных планшетах.

Урок трека 2 (durable): вывод по 1-3 планшетам ЛОЖЕН — гейт обязателен на всех.
Меряет для каждого планшета (D1+D2): сколько собралось групп, и насколько каждая группа
совпадает по ЗНАЧЕНИЮ с каждой из целевых кривых (|Δv|/v ≤ 0.05 / 0.12), плюс покрытие строк.
Ключевой критерий — ЧИСТОТА: группа должна бить в ОДНУ кривую и в 0% в другую (нет свопов).

python bkz_gate_groups.py [--thr 0.55] [--win y0 y1] [--limit N]
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_nlgx import extract, NULL
from decode_levels import build_family, gt_levels, scale_map
from dataset_build import find_image
import bkz_track as BT
from bkz_pair import pair_score
from bkz_groups import build_groups, group_trace

ET = Path(r"F:\nds\projects\Archive\Yatskivska_001\wlg")


def predicted_twin_x(fam, ka, kb, x):
    """x-позиция ДВОЙНИКА той же точки на другой шкале: то же значение, другая шкала.
    Все шкалы листа делят один пиксельный диапазон ⇒ двойник детерминирован."""
    sa, sb = fam[ka], fam[kb]
    v = sa["v_left"] + (x - sa["x_left"]) * (sa["v_right"] - sa["v_left"]) / \
        ((sa["x_right"] - sa["x_left"]) or 1)
    return sb["x_left"] + (v - sb["v_left"]) * (sb["x_right"] - sb["x_left"]) / \
        ((sb["v_right"] - sb["v_left"]) or 1)


def find_twin_strands(strands, fam, tol=6.0, min_rows=40, topn=25):
    """ЦЕЛЕВОЙ поиск двойника (вместо O(N²) перебора пар): для странда i и пары шкал (ka,kb)
    считаем ПРЕДСКАЗАННУЮ траекторию двойника и ищем странд, лежащий на ней (медиана |Δx| ≤ tol).
    Зачем: перебор пар не находит короткие двойники — они не попадают в топ-N по длине
    (BKZ D2 3474/3866: 0 рёбер при том, что странды на GT лежат точно). Гейт 18.07."""
    edges = []
    N = min(topn, len(strands))
    for i in range(N):
        si = strands[i]
        rows_i = sorted(si)
        if len(rows_i) < min_rows:
            continue
        for ka in range(len(fam)):
            for kb in range(len(fam)):
                if ka == kb:
                    continue
                pred = {y: predicted_twin_x(fam, ka, kb, si[y]) for y in rows_i}
                for j in range(len(strands)):
                    if j == i:
                        continue
                    sj = strands[j]
                    common = [y for y in pred if y in sj]
                    if len(common) < min_rows:
                        continue
                    d = np.array([abs(pred[y] - sj[y]) for y in common])
                    if float(np.median(d)) <= tol:
                        edges.append((i, j, ka, kb, float(np.median(d)), len(common)))
    return edges


def gt_values(m, c):
    gl = gt_levels(c); gt = BT.curve_rows(c)
    fam = build_family(m, c)
    mps = [scale_map(s) for s in fam]
    return {y: abs(mps[min(gl.get(y, 0), len(fam) - 1)](x)) for y, x in gt.items()}


def run(nlgx, thr, win, min_rows=40, topn=20):
    m = extract(str(nlgx))
    img = find_image(nlgx)
    if not img:
        return None
    rgb = np.asarray(Image.open(img).convert("RGB"))
    mask = BT.ink_mask(rgb)
    tok = BT.PAIRS["D1"] if "_D1" in nlgx.stem else BT.PAIRS["D2"]
    rng = BT.track_x_range(m, tok)
    if rng is None:
        return None
    x0, x1 = rng
    da = m["depth_axis"]
    y0, y1 = (win if win else (da["top_y"], da["bottom_y"]))
    strands = BT.link_strands(mask, x0, x1, y0, y1)
    curves = {c["name"].split()[0]: c for c in m["curves"]}
    if tok[0] not in curves:
        return None
    # ★ ФИКС 18.07 (v2): семейства шкал РАЗНЫЕ у разных кривых листа (D2: GZ41 fam=3
    # v[0..19/95/475], OGZ1 fam=2; на 3474 у OGZ1 вообще другая база). Применять семейство ОДНОЙ
    # кривой ко ВСЕМ странам неверно (D2 давал 0 рёбер). НО и «все оси листа» ХУЖЕ (гейт: 5→4
    # чистых): лишние кандидаты порождают ЛОЖНЫЕ пары — почти любые два странда сходятся на
    # какой-нибудь паре шкал. ⇒ паррование ОТДЕЛЬНО ДЛЯ КАЖДОЙ кривой её СОБСТВЕННЫМ семейством.
    gvals = {t: gt_values(m, curves[t]) for t in tok if t in curves}
    out = []
    n_edges_tot = 0
    for t in tok:
        if t not in curves:
            continue
        famc = build_family(m, curves[t])
        if len(famc) < 2:
            continue
        edges, groups = build_groups(strands, famc, thr=thr, min_rows=min_rows, topn=topn)
        n_edges_tot += len(edges)
        best = None
        for g in sorted(groups, key=lambda g: -sum(len(strands[i]) for i, _ in g)):
            rows = group_trace(strands, g, famc)
            rec = {"for": t, "nodes": len(g), "rows": len(rows), "match": {}}
            for t2, gv in gvals.items():
                common = [y for y in rows if y in gv]
                if len(common) < 50:
                    continue
                rel = np.array([abs(rows[y] - gv[y]) / max(gv[y], 1e-6) for y in common])
                rec["match"][t2] = (round(float((rel <= 0.05).mean()) * 100), len(common))
            own = rec["match"].get(t, (0, 0))[0]
            if best is None or own > best[0]:
                best = (own, rec)
        if best:
            out.append(best[1])
    return dict(stem=nlgx.stem, tok=tok, n_strands=len(strands), n_edges=n_edges_tot, groups=out)


def main():
    a = sys.argv[1:]
    thr = float(a[a.index("--thr") + 1]) if "--thr" in a else 0.55
    win = None
    if "--win" in a:
        i = a.index("--win"); win = (int(a[i + 1]), int(a[i + 2]))
    limit = int(a[a.index("--limit") + 1]) if "--limit" in a else 99
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    files = [f for f in sorted(ET.glob("*BKZ*.nlgx")) if "_auto" not in f.stem][:limit]
    print(f"планшетов: {len(files)}  thr={thr}  окно={win or 'весь'}")
    clean = dirty = 0
    for f in files:
        try:
            r = run(f, thr, win)
        except Exception as e:
            print(f"{f.stem[:38]:<40} ERR {type(e).__name__}: {e}")
            continue
        if not r:
            print(f"{f.stem[:38]:<40} — пропуск"); continue
        print(f"\n{r['stem'][:44]}  {r['tok']}  странд={r['n_strands']} рёбер={r['n_edges']} групп={len(r['groups'])}")
        for g in r["groups"]:
            parts = " ".join(f"{t}:{v[0]}%(n={v[1]})" for t, v in g["match"].items())
            t = g["for"]
            own = g["match"].get(t, (0, 0))[0]
            other = max([v[0] for k, v in g["match"].items() if k != t] or [0])
            ok = own >= 40 and other <= 5          # своя кривая поймана, чужая не загрязняет
            clean += int(ok); dirty += int(not ok)
            print(f"   для {t}: узлов={g['nodes']:2d} строк={g['rows']:5d}  {parts}  {'ЧИСТО' if ok else ''}")
    print(f"\nИТОГ: планшетов с ЧИСТОЙ главной группой: {clean}, без: {dirty}")


if __name__ == "__main__":
    main()

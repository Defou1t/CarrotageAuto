r"""_header_colors.py — §6.384 РАЗВЕДКА: хроматические цвета шапки по треку — автоматически (20.11).

Шаг 1 сборки §6.383. Шапка — строки выше начала кривых эталона; полоса трека — объединение шкал рамки с тем же токеном DA. Хроматический
пиксель: разброс каналов ≥ 40 и min канала < 200; семейство оттенка по углу тона (красный, оранжевый, жёлтый, зелёный, бирюзовый, синий,
фиолетовый, розовый). Цвет сетки — самое частое хроматическое семейство поля кривых (строки после начала кривых) вдали (> 6 px) от трасс
эталона, если у него ≥ 50% хроматических пикселей поля; исключается. Семейство в шапке трека засчитано при ≥ N пикселей. Сравнение с
семействами хроматических кривых эталона трека (`taskS/color_clash/rows.jsonl`, §6.378, 400 листов).

  _header_colors.py [--min-px 200]   → taskS/header_colors/rows.jsonl (продолжаемо), сводка
"""
import sys, argparse, json, pickle, hashlib, time, colorsys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
OUT = TS / "header_colors"
FAM = [("красный", 345, 15), ("оранжевый", 15, 45), ("жёлтый", 45, 70), ("зелёный", 70, 165), ("бирюзовый", 165, 200),
       ("синий", 200, 255), ("фиолетовый", 255, 300), ("розовый", 300, 345)]


def sheet_dir(n):
    return f"{n.stem[:40]}_{hashlib.md5(n.stem.encode('utf-8')).hexdigest()[:8]}"


def fam_of_hue(h):
    """h — градусы [0, 360) → индекс семейства"""
    for i, (_, a, b) in enumerate(FAM):
        if (a <= h < b) if a < b else (h >= a or h < b):
            return i
    return 0


def hue_deg(rgb):
    r, g, b = (np.asarray(rgb, np.float64) / 255.0).T
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b); d = np.where(mx - mn == 0, 1e-9, mx - mn)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
    return np.mod(h, 360.0)


def chrom_mask(px, paper=None):
    """хроматическая ТУШЬ: разброс каналов ≥ 40 и (если дана бумага строки) темнее бумаги на ≥ 50 (иначе — min канала < 200)"""
    px = px.astype(np.int16)
    ok = (px.max(-1) - px.min(-1)) >= 40
    if paper is None:
        return ok & (px.min(-1) < 200)
    g = px.mean(-1)
    return ok & (g <= paper[:, None] - 50)


def fam_hist(px):
    """px [N, 3] хроматические → счёт по семействам"""
    if not len(px):
        return np.zeros(len(FAM), np.int64)
    h = hue_deg(px); f = np.array([fam_of_hue(v) for v in h]) if len(h) < 5000 else np.vectorize(fam_of_hue)(h)
    return np.bincount(f, minlength=len(FAM))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-px", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hrun", type=int, default=0, help="§6.384 (б): семейство засчитано только при горизонтальном отрезке туши ≥ hrun px (0 = по пикселям)")
    ap.add_argument("--win", type=int, default=0, help="строк шапки над началом кривых (0 = вся шапка)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from auto import imaging as im
    im.EXIF_TRANSPOSE = False
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    jf = od / "rows.jsonl"
    done = {json.loads(l)["sheet"] for l in jf.read_text(encoding="utf-8").splitlines()} if jf.exists() else set()
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    T0 = time.time(); k = 0
    for l in (TS / "color_clash" / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(l); nm = q["sheet"]
        if a.limit and k >= a.limit:
            break
        if nm in done or nm not in SRC:
            continue
        n = SRC[nm]
        src = next((TS / d / (sheet_dir(n) + ".pkl") for d in ("tcache", "tcache_out") if (TS / d / (sheet_dir(n) + ".pkl")).exists()), None)
        if src is None:
            continue
        m = extract(str(n))
        tops = [c["top_y"] for c in m["curves"] if c.get("top_y") is not None]
        if not tops:
            continue
        top = int(min(tops))
        rgb = im.load_rgb(pickle.load(open(src, "rb"))["image"])          # координаты nlgx — исходного скана
        H, W = rgb.shape[:2]
        # цвет сетки: поле кривых (до 3000 строк после начала), вдали от трасс эталона
        y0b, y1b = top, min(H, top + 3000)
        body = rgb[y0b:y1b:2, ::2]
        gray = rgb.mean(2)
        paper = np.percentile(gray[:, ::4], 90, axis=1)                  # бумага по строке (как level_ink.paper_of)
        near = np.zeros(body.shape[:2], bool)
        for c in m["curves"]:
            tr = dense(c)
            for y, x in tr.items():
                if y0b <= y < y1b and y % 2 == 0:
                    xi = int(round(x)) // 2
                    near[(y - y0b) // 2, max(0, xi - 3):xi + 4] = True
        bm = chrom_mask(body, paper[y0b:y1b:2]) & ~near
        bh = fam_hist(body[bm][::7])
        bg = [i for i in range(len(FAM)) if bh.sum() > 0 and bh[i] >= 0.25 * bh.sum()]      # сетка / печать поля
        grid = bg[0] if bg else -1
        # треки: полоса по шкалам рамки
        tracks = {}
        for s in m["scale_axes"]:
            tk = (s["name"].split() + [""])[0]
            if not tk.startswith("DA"):
                continue
            lo, hi = sorted((s["x_left"], s["x_right"]))
            a_, b_ = tracks.get(tk, (lo, hi)); tracks[tk] = (min(a_, lo), max(b_, hi))
        recs = []
        h0 = max(0, top - a.win) if a.win else 0
        hdr = rgb[h0:max(h0 + 1, top)]
        for tk, (lo, hi) in tracks.items():
            seg = hdr[:, int(max(0, lo)):int(min(W, hi))]
            if not seg.size:
                continue
            pm = chrom_mask(seg, paper[h0:h0 + seg.shape[0]])
            hh = fam_hist(seg[pm][::3]) * 3
            if a.hrun:
                # (б) горизонтальные цветные штрихи (шкалы, образцы линий): по семейству — карта пикселей, закрытие разрывов ≤ 2 px,
                #     самый длинный горизонтальный отрезок в строке ≥ hrun
                ys_, xs_ = np.nonzero(pm)
                fams = np.vectorize(fam_of_hue)(hue_deg(seg[ys_, xs_])) if len(ys_) else np.zeros(0, int)
                det = []
                for i in range(len(FAM)):
                    if i in bg:
                        continue
                    sel = fams == i
                    if sel.sum() < a.hrun:
                        continue
                    M = np.zeros(pm.shape, bool); M[ys_[sel], xs_[sel]] = True
                    M = M | np.roll(M, 1, 1) | np.roll(M, -1, 1)          # разрывы ≤ 2 px
                    best = 0
                    for row in M[np.flatnonzero(M.sum(1) >= a.hrun)]:
                        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
                        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
                        if len(st):
                            best = max(best, int((en - st).max()))
                    if best >= a.hrun:
                        det.append(i)
            else:
                det = [i for i in range(len(FAM)) if hh[i] >= a.min_px and i not in bg]
            tru = sorted({fam_of_hue(float(hue_deg(np.array([r["ct"]]))[0])) for r in q["recs"] if r["track"] == tk and r["ct"] is not None
                          and (max(r["ct"]) - min(r["ct"])) >= 40})
            recs.append(dict(track=tk, det=det, tru=tru, grid=bg, hist=hh.tolist()))
        with open(jf, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(sheet=nm, recs=recs), ensure_ascii=False) + "\n")
        k += 1
        del rgb, hdr, body, gray
        if k % 50 == 0:
            print(f"  листов {k}, {time.time() - T0:.0f} с")
    # ── сводка ──
    tp = fn = fp = 0; trk_chrom = trk_black = fp_black = 0
    for l in jf.read_text(encoding="utf-8").splitlines():
        for r in json.loads(l)["recs"]:
            det, tru = set(r["det"]), set(r["tru"])
            if tru:
                trk_chrom += 1; tp += len(det & tru); fn += len(tru - det); fp += len(det - tru)
            else:
                trk_black += 1; fp_black += bool(det)
    rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
    print(f"★ треков с хроматическими кривыми эталона {trk_chrom}: полнота семейств {rec:.1%} ({tp}/{tp + fn}), точность {prec:.1%} ({tp}/{tp + fp})")
    print(f"   треков без хроматических кривых {trk_black}: хоть одно ложное семейство в шапке — {fp_black} ({fp_black / max(trk_black, 1):.1%})")


if __name__ == "__main__":
    main()

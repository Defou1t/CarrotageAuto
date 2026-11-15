r"""_level_twin_net.py — §6.377 РАЗВЕДКА: различает ли СЕТЬ ТУШИ верный уровень масштаба по «близнецу» лучше серого (15.11).

Модель уровня (`auto.level_ink.features`) видит тушь на месте близнеца — где было бы перо соседнего масштаба при той же величине:
TWX[0] = x уровня 1 для величины, если наша линия — уровень 0 (перо 5× при нашем 1×), TWX[1] = x уровня 0, если наша линия — уровень 1.
Мера туши там — серая темнота (бумага − скан), и её глушат сетка, печать, соседи (§6.265 / 6.289 / 6.271). Сеть туши (`ink_seg_f`, вне
фолда) обучена на туши ОЦИФРОВАННЫХ кривых ±2 px — сетку и печать не видит. Здесь по блокам 8 строк выдачи NSE26 (вне фолда), где наша
трасса честна в пикселях (|x − x эталона| ≤ 3) и цепочка шкал ≥ 2: метка y = (уровень эталона ≠ нашему, оба ∈ {0, 1}); признак
«против нашей метки» s = тушь(x_b) − тушь(x_a), если мы на 0, и наоборот, если на 1 (x_a = TWX[0], x_b = TWX[1]); тушь — max по блоку
и ±3 px: серая (как прод) и сеть. AUC(s → y) серой и сети; то же по кривым (доля блоков).

  _level_twin_net.py [--n 150] [--seed 5]   → taskS/level_twin_net/rows.jsonl (продолжаемо), сводка
"""
import sys, argparse, pickle, json, random, hashlib, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
SEG = TS / "ink_seg_f"
OUT = TS / "level_twin_net"


def sheet_dir(n):
    return f"{n.stem[:40]}_{hashlib.md5(n.stem.encode('utf-8')).hexdigest()[:8]}"


def lev_at(segs, rows):
    out = np.full(len(rows), -1, np.int64)
    for y0, y1, l in segs:
        out[(rows >= y0) & (rows <= y1)] = int(l)
    return out


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # средние ранги при равенствах
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--mode", default="NSE26")
    a = ap.parse_args()
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    import decode_levels as DL
    from auto import meta as M, imaging as im, frame as FR, ink_seg as IS, level_ink as LI
    im.EXIF_TRANSPOSE = False
    OUT.mkdir(parents=True, exist_ok=True)
    jf = OUT / "rows.jsonl"
    done = {json.loads(l)["sheet"] for l in jf.read_text(encoding="utf-8").splitlines()} if jf.exists() else set()
    folds = json.loads((SEG / "folds.json").read_text(encoding="utf-8"))
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    names = []
    for f in ("tcache_sheets.txt", "outside_sheets.txt"):
        names += [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    names = sorted(set(names)); random.Random(a.seed).shuffle(names)
    T0 = time.time(); nsh = len(done)
    for nm in names:
        if nsh >= a.n:
            break
        if nm in done or nm not in SRC:
            continue
        n = SRC[nm]
        od = TS / "rp_lvl" / a.mode / sheet_dir(n)
        fs = list(od.glob("*_auto.nlgx")) if od.exists() else []
        src = next((TS / d / (sheet_dir(n) + ".pkl") for d in ("tcache", "tcache_out") if (TS / d / (sheet_dir(n) + ".pkl")).exists()), None)
        if not fs or src is None:
            continue
        try:
            m = extract(str(n)); mo = extract(str(fs[0]))
        except Exception:
            continue
        ours = {c["name"]: c for c in mo["curves"]}
        cur = []
        for c in m["curves"]:
            if M.mnem_root(c["name"]) == "DA" or not c.get("segments") or c["name"] not in ours:
                continue
            ch = DL.build_family(m, c)
            if len(ch) < 2:
                continue
            cur.append((c, ch))
        if not cur:
            continue
        cc = pickle.load(open(src, "rb"))
        rgb = im.load_rgb(cc["image"])                  # координаты nlgx (эталон и наша выдача) — исходного скана: row_shift НЕ применять
        gray = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.uint8)
        paper = LI.paper_of(gray)
        H, W = gray.shape
        fo = folds.get(n.parent.parent.name)
        net = str(SEG / (f"f{fo}.pt" if fo is not None else "all.pt"))
        recs = []
        for c, ch in cur:
            oc = ours[c["name"]]
            to = dense(oc); tt = dense(c)
            if len(to) < 200 or not oc.get("segments"):
                continue
            ys = np.array(sorted(to)); P = ys[(ys % 8) == 4]
            P = P[(P >= 4) & (P < H - 4)]
            if len(P) < 20:
                continue
            xo = np.array([to[y] for y in P], float)
            xt = np.array([tt.get(int(y), np.nan) for y in P], float)
            Lo = lev_at(sorted(oc["segments"]), P); Lt = lev_at(sorted(c["segments"]), P)
            ok = np.isfinite(xt) & (np.abs(xt - xo) <= 3) & np.isin(Lo, (0, 1)) & np.isin(Lt, (0, 1))
            if ok.sum() < 10:
                continue
            xa = LI._pix(ch[1], LI._val(ch[0], xo)); xb = LI._pix(ch[0], LI._val(ch[1], xo))
            xl = min(min(s["x_left"], s["x_right"]) for s in ch); xr = max(max(s["x_left"], s["x_right"]) for s in ch)
            wd = max(8.0, xr - xl); bx0 = int(max(0, np.floor(xl - 0.05 * wd))); bx1 = int(min(W, np.ceil(xr + 0.05 * wd)))
            Pk = P[ok]; y0 = int(Pk.min()) - 4; y1 = int(Pk.max()) + 4
            pm = IS.prob_map(rgb, y0, y1, bx0, bx1, net)
            if pm.ndim == 3:
                pm = pm[0]
            G = np.clip(paper[y0:y1, None].astype(np.int16) - gray[y0:y1, bx0:bx1].astype(np.int16), 0, 255)

            def blk(Mx, yy, xx):
                out = np.full(len(yy), np.nan)
                for i, (y, x) in enumerate(zip(yy, xx)):
                    if not np.isfinite(x):
                        continue
                    cx = int(round(x)) - bx0
                    if 3 <= cx < Mx.shape[1] - 3:
                        out[i] = float(Mx[y - 4 - y0:y + 4 - y0, cx - 3:cx + 4].max())
                return out
            ga = blk(G, Pk, xa[ok]); gb = blk(G, Pk, xb[ok]); na = blk(pm, Pk, xa[ok]); nb_ = blk(pm, Pk, xb[ok])
            go = blk(G, Pk, xo[ok]); no = blk(pm, Pk, xo[ok])
            recs.append(dict(curve=c["name"], K=len(ch), Lo=Lo[ok].tolist(), Lt=Lt[ok].tolist(),
                             ga=np.round(ga, 1).tolist(), gb=np.round(gb, 1).tolist(), na=np.round(na, 3).tolist(),
                             nb=np.round(nb_, 3).tolist(), go=np.round(go, 1).tolist(), no=np.round(no, 3).tolist()))
        with open(jf, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(sheet=nm, recs=recs), ensure_ascii=False) + "\n")
        nsh += 1
        del rgb, gray
        if nsh % 10 == 0:
            print(f"  листов {nsh}, {time.time() - T0:.0f} с")
    # ── сводка ──
    Lo, Lt, ga, gb, na, nb_, cv = [], [], [], [], [], [], []
    k = 0
    for l in jf.read_text(encoding="utf-8").splitlines():
        for r in json.loads(l)["recs"]:
            Lo += r["Lo"]; Lt += r["Lt"]; ga += r["ga"]; gb += r["gb"]; na += r["na"]; nb_ += r["nb"]; cv += [k] * len(r["Lo"]); k += 1
    Lo, Lt, ga, gb, na, nb_, cv = map(np.array, (Lo, Lt, ga, gb, na, nb_, cv))
    ga, gb, na, nb_ = (np.nan_to_num(v.astype(float), nan=0.0) for v in (ga, gb, na, nb_))
    y = Lo != Lt
    sg = np.where(Lo == 0, gb - ga, ga - gb); sn = np.where(Lo == 0, nb_ - na, na - nb_)
    print(f"★ блоков (честных в пикселях, цепочка ≥ 2, уровни 0/1): {len(y)} у {k} кривых; уровень неверен у {y.mean():.1%} блоков")
    print(f"   AUC «против нашей метки» → неверный уровень: серая {auc(sg, y):.3f} | сеть туши {auc(sn, y):.3f}")
    for L in (0, 1):
        s_ = Lo == L
        print(f"   наш уровень {L} ({s_.sum()} блоков, неверных {y[s_].mean():.1%}): серая {auc(sg[s_], y[s_]):.3f} | сеть {auc(sn[s_], y[s_]):.3f}")
    # по кривым: доля неверных блоков против средней улики
    cu = np.unique(cv)
    yc = np.array([y[cv == c].mean() for c in cu]); gc = np.array([sg[cv == c].mean() for c in cu]); nc = np.array([sn[cv == c].mean() for c in cu])
    print(f"   по кривым (≥ 50% блоков неверны): {np.mean(yc >= 0.5):.1%}; AUC средней улики: серая {auc(gc, yc >= 0.5):.3f} | сеть {auc(nc, yc >= 0.5):.3f}")


if __name__ == "__main__":
    main()

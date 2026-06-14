"""
Валидация расшифровки nlgx: реконструировать value(depth) из пиксель-трассы
эксперта и сверить с эталонным LAS (corr / RMSE / по перцентилям).

Модель реконструкции:
  depth(y) = top_depth + (y-top_y)*span_depth/span_px
  Для кривой семейства F (SA1/SA2) = упорядоченный по idx список scale-axes.
  Для строки y: pixel x = xs[i]; level L = уровень сегмента, содержащего y;
  scale = F[L]; value = v_left + (x-x_left)/(x_right-x_left)*(v_right-v_left).
"""
import sys, struct, math
from extract_nlgx import extract, depth_of, NULL


def family_of(model, curve_name):
    """SA1/SA2 — последний токен имени кривой ('BK1 DA1 SA1' -> 'SA1')."""
    key = curve_name.strip().split()[-1]
    fam = [s for s in model["scale_axes"] if s["name"].strip().split()[-1] == key]
    fam.sort(key=lambda s: s["idx"])
    return fam


def reconstruct(model, curve):
    fam = family_of(model, curve["name"])
    if not fam:
        return []
    top_y = curve["top_y"]
    xs = curve["xs"]
    segs = curve["segments"]

    def level_at(y):
        last = None
        for s, e, l in segs:
            if s <= y <= e:
                return l
            if e < y:
                last = l
        return last  # carry-forward last segment before y

    out = []
    for i, x in enumerate(xs):
        if x == NULL:
            continue
        y = top_y + i
        L = level_at(y)
        if L is None or L >= len(fam):
            continue
        sa = fam[L]
        frac = (x - sa["x_left"]) / (sa["x_right"] - sa["x_left"])
        v = sa["v_left"] + frac * (sa["v_right"] - sa["v_left"])
        out.append((depth_of(model, y), v))
    return out


def load_las(path):
    rows = []
    with open(path, "r", encoding="latin1") as f:
        in_data = False
        cols = None
        for line in f:
            s = line.strip()
            if s.startswith("~A"):
                in_data = True
                cols = s[2:].split()
                continue
            if in_data and s and not s.startswith("#"):
                parts = s.split()
                if len(parts) >= 2:
                    rows.append([float(p) for p in parts])
    return cols, rows


def resample(recon, depths, null=-999.25):
    """Линейная интерполяция recon (list of (d,v), по возрастанию d) на depths."""
    recon = sorted(recon, key=lambda t: t[0])
    ds = [d for d, _ in recon]
    vs = [v for _, v in recon]
    out = []
    j = 0
    for d in depths:
        if d < ds[0] or d > ds[-1]:
            out.append(None); continue
        while j+1 < len(ds) and ds[j+1] < d:
            j += 1
        # find bracketing
        k = 0
        # binary-ish linear scan from start (depths monotonic so could keep j, but be safe)
        lo, hi = 0, len(ds)-1
        while lo+1 < hi:
            mid = (lo+hi)//2
            if ds[mid] <= d: lo = mid
            else: hi = mid
        d0, d1 = ds[lo], ds[hi]
        v0, v1 = vs[lo], vs[hi]
        out.append(v0 if d1==d0 else v0 + (v1-v0)*(d-d0)/(d1-d0))
    return out


def stats(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None
             and not math.isnan(x) and not math.isnan(y) and abs(y) < 1e6]
    if len(pairs) < 5:
        return None
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    n = len(pairs)
    mx, my = sum(xs)/n, sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs); syy = sum((y-my)**2 for y in ys)
    sxy = sum((x-mx)*(y-my) for x, y in pairs)
    corr = sxy/math.sqrt(sxx*syy) if sxx>0 and syy>0 else 0
    rmse = math.sqrt(sum((x-y)**2 for x, y in pairs)/n)
    # log-domain (резистивность) corr — для перевыносов важнее
    lp = [(math.log10(max(x,1e-3)), math.log10(max(y,1e-3))) for x,y in pairs if x>0 and y>0]
    lcorr = None
    if len(lp) > 5:
        lx=[p[0] for p in lp]; ly=[p[1] for p in lp]; ln=len(lp)
        lmx=sum(lx)/ln; lmy=sum(ly)/ln
        lsxx=sum((x-lmx)**2 for x in lx); lsyy=sum((y-lmy)**2 for y in ly)
        lsxy=sum((x-lmx)*(y-lmy) for x,y in lp)
        lcorr = lsxy/math.sqrt(lsxx*lsyy) if lsxx>0 and lsyy>0 else 0
    return {"n": n, "corr": corr, "log_corr": lcorr, "rmse": rmse,
            "recon_range": (min(xs), max(xs)), "las_range": (min(ys), max(ys))}


def main():
    nlgx = sys.argv[1]
    las = sys.argv[2]
    m = extract(nlgx)
    cols, rows = load_las(las)
    print(f"LAS cols: {cols}, {len(rows)} rows")
    depths = [r[0] for r in rows]

    for ci, name in enumerate(cols[1:], start=1):
        # match curve by mnemonic prefix
        cur = None
        for c in m["curves"]:
            cn = c["name"].strip().split()[0]  # 'BK1'
            base = cn.rstrip("0123456789")     # 'BK'
            if base.upper() == name.upper():
                cur = c; break
        if cur is None:
            print(f"\n[{name}] no matching curve in nlgx"); continue
        recon = reconstruct(m, cur)
        if not recon:
            print(f"\n[{name}] reconstruction empty"); continue
        las_vals = [r[ci] for r in rows]
        res_vals = resample(recon, depths)
        st = stats(res_vals, las_vals)
        print(f"\n[{name}] curve '{cur['name']}' recon_pts={len(recon)}")
        if st:
            print(f"   corr={st['corr']:.4f}  log_corr={st['log_corr'] and round(st['log_corr'],4)}"
                  f"  rmse={st['rmse']:.3f}")
            print(f"   recon range {st['recon_range'][0]:.2f}..{st['recon_range'][1]:.2f}"
                  f"   LAS range {st['las_range'][0]:.2f}..{st['las_range'][1]:.2f}")
        # spot samples
        print("   sample depth: recon vs las")
        for dd in (depths[0], depths[len(depths)//4], depths[len(depths)//2],
                   depths[3*len(depths)//4], depths[-1]):
            i = depths.index(dd)
            rv = res_vals[i]
            print(f"     {dd:.1f}: {('%.2f'%rv) if rv is not None else 'None':>9} vs {las_vals[i]:.2f}")


if __name__ == "__main__":
    main()

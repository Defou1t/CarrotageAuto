r"""
_graph_resolve.py — ЗАМЕР-ГЕЙТ Стадии 1 (PLAN §6.6.11): разрешение X-узлов по непрерывности
кривизны на ОДНОЦВЕТНОЙ мультилинии BKZ GZ1-3 (YULIIV_055) с плотным GT эксперта.

ВОПРОС гейта: если идентичности заданы вверху верно, держит ли их ЭКСТРАПОЛЯЦИЯ НАКЛОНА
через пересечения? Изолирует Стадию 1 (узлы), не Стадию 0 (детект): ведём РОВНО N нитей из
GT-стартов в коридоре ±band, без фрагментации. Контраст resolve (pred=наклон) vs greedy (x).
Чистый numpy (нет skimage/scipy в py3.14).

python _graph_resolve.py [grp=L] [thr=45] [band=22]
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from extract_nlgx import extract, NULL
from dataset_build import find_image

NLGX = r"F:\nds\projects\Archive\YULIIV_055\wlg\YULIIV_055_BKZ_2960-3748_200_1998-08-03_D_1.nlgx"
OUT = Path(r"F:\nds\output")
GROUPS = {"L": ["GZ11", "GZ21", "GZ31"], "R": ["GZ41", "GZ51"]}
SL, GATE = 22.0, 16.0          # макс шаг/строку, гейт захвата центроида
CROSS, EVGAP, WIN = 12.0, 30, 25   # |Δx| пересечения, склейка событий, окно вокруг события
TOL = 12.0                     # допуск своп-теста vs GT
COLS = [(220, 40, 40), (40, 130, 230), (40, 180, 70), (210, 150, 0), (170, 60, 200)]


def dense(d, ty, by):
    ys = np.array(sorted(d)); xs = np.array([d[y] for y in ys])
    return np.interp(np.arange(ty, by), ys, xs, left=np.nan, right=np.nan)   # индекс = y-ty


def row_cents(rowmask, gap=2):
    xs = np.nonzero(rowmask)[0]
    if not len(xs):
        return []
    return [float(s.mean()) for s in np.split(xs, np.nonzero(np.diff(xs) > gap)[0] + 1)]


def trace(gray, ty, by, x0, gtd, mode, thr, band):
    """ведём по нити на GT-линию из её старта; resolve/greedy отличаются базой назначения."""
    xs_grid = np.arange(x0, x0 + gray.shape[1])
    bg = int(np.median(gray[::7, ::7]))
    names = list(gtd)
    th = {}
    for nm in names:
        a = gtd[nm]; idx = np.nonzero(~np.isnan(a))[0]
        y0 = ty + int(idx[0]); th[nm] = {"x": float(a[idx[0]]), "v": 0.0, "y0": y0, "pts": {}}
    for y in range(ty, by):
        ink = gray[y] < bg - thr
        cor = np.zeros(len(xs_grid), bool)
        for a in gtd.values():
            gx = a[y - ty]
            if not np.isnan(gx):
                cor |= np.abs(xs_grid - gx) <= band
        cents = [x0 + c for c in row_cents(ink & cor)]   # → абсолютные x
        act = [nm for nm in names if y >= th[nm]["y0"]]
        ref = {}
        for nm in act:
            t = th[nm]
            pr = t["x"] + float(np.clip(t["v"], -SL, SL))
            ref[nm] = pr if mode == "resolve" else t["x"]  # resolve=наклон, greedy=липнет к x
        assigned = {}
        if cents:
            order = sorted((abs(ref[nm] - cents[j]), nm, j) for nm in act for j in range(len(cents)))
            usedc = set()
            for d, nm, j in order:                       # ЭКСКЛЮЗИВНО: 1 нить ↔ 1 центроид
                if nm in assigned or j in usedc:
                    continue
                assigned[nm] = j; usedc.add(j)
            for nm in act:                               # центроидов < нитей → остаток делит ближайший (слипание)
                if nm not in assigned:
                    assigned[nm] = min(range(len(cents)), key=lambda j: abs(ref[nm] - cents[j]))
        for nm in act:
            t = th[nm]
            if nm in assigned and abs(t["x"] - cents[assigned[nm]]) <= 60:
                nx = cents[assigned[nm]]
            else:
                nx = t["x"]                              # далёкое назначение / нет ink → держим
            dx = nx - t["x"]; t["v"] = 0.6 * t["v"] + 0.4 * dx
            t["x"] = nx; t["pts"][y] = nx
    return th, bg


def crossing_events(gtd, ty, by):
    """интервалы глубины, где любая пара GT-линий ближе CROSS = настоящие пересечения."""
    ys = []
    for y in range(ty, by):
        gs = [a[y - ty] for a in gtd.values()]
        gs = [g for g in gs if not np.isnan(g)]
        if any(abs(gs[i] - gs[j]) < CROSS for i in range(len(gs)) for j in range(i + 1, len(gs))):
            ys.append(y)
    ev = []
    for y in ys:
        if ev and y - ev[-1][1] <= EVGAP:
            ev[-1][1] = y
        else:
            ev.append([y, y])
    return ev


def metrics(th, gtd, ty, events, mode):
    names = list(gtd)
    # своп per (nm,y): ближайшая GT не своя
    swap = {nm: set() for nm in names}
    tot = 0; dxs = []
    for nm in names:
        a = gtd[nm]
        for y, xv in th[nm]["pts"].items():
            gx = a[y - ty]
            if np.isnan(gx):
                continue
            tot += 1; dxs.append(abs(xv - gx))
            best = min(names, key=lambda n2: abs(xv - gtd[n2][y - ty])
                       if not np.isnan(gtd[n2][y - ty]) else 1e9)
            if best != nm and abs(xv - gtd[best][y - ty]) < abs(xv - gx) - 3:
                swap[nm].add(y)
    swrows = sum(len(s) for s in swap.values())
    # событие «провалено», если в окне ±WIN есть своп любой нити
    bad = 0
    for a, b in events:
        win = range(a - WIN, b + WIN + 1)
        if any(y in swap[nm] for nm in names for y in win):
            bad += 1
    okev = len(events) - bad
    return okev, len(events), swrows, tot, (np.median(dxs) if dxs else 0)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    grp = sys.argv[1] if len(sys.argv) > 1 else "L"
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
    band = float(sys.argv[3]) if len(sys.argv) > 3 else 22.0
    names = GROUPS[grp]
    m = extract(NLGX); da = m["depth_axis"]; ty, by = da["top_y"], da["bottom_y"]
    raw = {}
    for c in m["curves"]:
        nm = c["name"].split()[0]
        if nm in names:
            raw[nm] = {c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL}
    gtd = {nm: dense(d, ty, by) for nm, d in raw.items() if len(d) >= 20}
    allx = [x for a in gtd.values() for x in a if not np.isnan(x)]
    x0, x1 = int(min(allx)) - 15, int(max(allx)) + 15
    img = Image.open(find_image(Path(NLGX))).convert("RGB"); arr = np.asarray(img)
    gray = np.asarray(img.convert("L"))[:by, x0:x1]
    events = crossing_events(gtd, ty, by)
    print(f"YULIIV_055 BKZ группа {grp}={list(gtd)} | трек y[{ty}..{by}] H={by-ty} полоса x[{x0}..{x1}]"
          f" thr={thr} band={band}")
    print(f"  GT: " + ", ".join(f"{nm}:{int(np.isfinite(a).sum())}тчк" for nm, a in gtd.items())
          + f" | настоящих пересечений (|Δx|<{CROSS:.0f}px): {len(events)}")

    for mode in ("greedy", "resolve"):
        th, bg = trace(gray, ty, by, x0, gtd, mode, thr, band)
        okev, nev, swrows, tot, mdx = metrics(th, gtd, ty, events, mode)
        rate = 100 * okev / nev if nev else 0
        print(f"\n[{mode:7}] bg={bg} | пересечений пройдено без свопа {okev}/{nev} ({rate:.0f}%) "
              f"| строк-свопов {swrows}/{tot} ({100*swrows/max(1,tot):.1f}%) | медиана|Δx|={mdx:.1f}px")
        if mode == "resolve":
            draw = arr.copy()
            for k, nm in enumerate(names):
                if nm not in th:
                    continue
                col = COLS[k % len(COLS)]
                for y, x in th[nm]["pts"].items():
                    xi = int(x)
                    if 0 <= y < draw.shape[0] and 0 <= xi < draw.shape[1]:
                        draw[y, max(0, xi - 1):xi + 2] = col
            d = Image.fromarray(draw); dr = ImageDraw.Draw(d)
            sw = {nm: set() for nm in names}
            _ = metrics(th, gtd, ty, events, mode)
            for a, b in events:
                yc = (a + b) // 2
                dr.ellipse([x0 - 14, yc - 14, x0 + 14, yc + 14], outline=(255, 140, 0), width=4)
            for tag, yc in [("a", ty + (by - ty) // 4), ("b", (ty + by) // 2), ("c", ty + 3 * (by - ty) // 4)]:
                d.crop((max(0, x0 - 40), yc - 350, x1 + 40, yc + 350)).save(OUT / f"graph_{grp}_{tag}.png")
            print(f"  оверлей -> F:\\nds\\output\\graph_{grp}_[a,b,c].png (линии цветом, оранж=пересечение)")


if __name__ == "__main__":
    main()

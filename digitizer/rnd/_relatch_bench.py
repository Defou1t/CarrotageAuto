r"""_relatch_bench.py — КЭШИРУЮЩИЙ СТЕНД для перебора стратегий ПЕРЕЗАХВАТЫВАНИЯ (§6.19).

ЗАЧЕМ. `_latch_probe.py` на каждый прогон декодирует 10-21 Мпикс скан, строит маски цветов и
плотный GT — это минуты. Для ПЕРЕБОРА стратегий (а §6.19 требует именно перебора: «прыгать
только на ран, согласованный с направлением/шириной») такая цена неприемлема.

Дорогая часть выносится в кэш ОДИН РАЗ. Для каждой пары (кривая, цвет) сохраняются РАНЫ ПО
СТРОКАМ внутри оракульной полосы — в виде CSR-массивов (a, b, c + ptr по строкам). После этого
стратегия — чистая функция над ранами: полный прогон 24 кривых занимает секунды, а не минуты.

★ ГЕЙТ КОРРЕКТНОСТИ: стратегия `baseline` обязана воспроизвести `_latch_probe.py` — med покривой
сверяется с `output/taskS/latch.json`. Пока `--verify` не зелёный, любые цифры со стенда негодны.

★ МЕТРИКА (durable, §6.9): med БЕЗ cov не цитировать. Главное число — ЧЕСТНЫЕ: med<=3px И cov>=0.9.

  python _relatch_bench.py --build      # собрать кэш (один раз, дорого)
  python _relatch_bench.py --verify     # сверить baseline с latch.json
  python _relatch_bench.py              # прогнать зарегистрированные стратегии
"""
import sys, io, json, pickle, argparse, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np

ARCH = Path(r"F:\nds\projects\Archive")
OUT = Path(r"F:\nds\output\taskS")
CACHE = OUT / "bench"
MN = r"F:\nds\Auto\mnemonics.json"

# Те же 5 скважин / 24 кривые, что в _latch_probe.py — выборка не меняется, иначе цифры
# несравнимы с базовой линией (durable-урок: не обобщать по первым листам прогона).
SH = [r"BOGAT_011\wlg\BOGAT_011_BKZ, DS_2800-3190_200_1984-03-02_D_1_B_1.nlgx",
      r"BOGAT_015\wlg\BOGAT_015_BKZ, DS_3300-3700_200_1990-01-08_D_1.nlgx",
      r"LEVEN_023\wlg\LEVEN_023_BKZ, DS_1010-1500_200_1996-12-02_D_1.nlgx",
      r"BEZLUD_051\wlg\BEZLUD_051_RK, AK, DS_2688-3142_500_1998-08-05_D_1.nlgx",
      r"YULIIV_055\wlg\YULIIV_055_MK, MBK, MDS_2084-3060_200_1998-07-18_D_1.nlgx"]

SLMAX = 30.0
WIDE_RUN = 14
BAND_PAD = 8          # trace_line(band_pad=8) — полоса шире GT-полосы на столько
MAX_GAP = 25          # _extend_ends(max_gap=25)


# ───────────────────────────── сборка кэша ─────────────────────────────

def build(pad=20, cache=None, shift=0):
    """Один дорогой проход: скан → маски цветов → раны по строкам в полосе каждой кривой.

    pad — насколько полоса ШИРЕ размаха экспертной кривой (20 = оракульная полоса, как в
    `_latch_probe.py`). Прогон с бОльшим pad отвечает на вопрос «какую полосу обязан отдать U1»:
    оракул и прод отличаются в 62× (§6.28.3), а между ними не мерено ничего.
    cache — куда класть (по умолчанию штатный кэш; для других pad ОБЯЗАТЕЛЬНО свой каталог,
    иначе `--verify` начнёт сверять чужие числа)."""
    cache = Path(cache) if cache else CACHE
    import cv2
    from extract_nlgx import extract, NULL
    import dataset as ds
    from dataset_build import find_image
    from _multi_replica_probe import dense
    from auto import frame as F, understand as U, confidence as C, trace2d as T, meta as M
    from auto import imaging as im
    from auto.config import DEFAULT
    p = DEFAULT.cv
    cache.mkdir(parents=True, exist_ok=True)

    for rel in SH:
        n = ARCH / rel
        well = n.parent.parent.name
        img = find_image(n)
        if not img:
            print(f"!! {well}: нет картинки"); continue
        rgb = cv2.cvtColor(cv2.imread(str(img), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m = M.parse_filename(n.name, MN)
            fr = F.frame_from_nlgx(str(n), m, p, rgb=rgb)
            if getattr(fr, "row_shift", None) is not None:
                rgb = F.apply_row_shift(rgb, fr.row_shift)
            sheet = U.understand(rgb, fr, m, p)
            C.classify(sheet)
        colors = sorted({L.color for L in sheet.lines}) or ["black"]
        fgs = {c: T._color_fg(rgb, c, p) for c in colors}
        H, W = rgb.shape[:2]

        mo = extract(str(n))
        gts = [c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])]
        dens = {g["name"]: dense(g) for g in gts}

        sheet_rec = {"well": well, "colors": colors, "H": H, "W": W, "curves": [],
                     "gt": {k: (np.array(sorted(v), np.int32),
                               np.array([v[y] for y in sorted(v)], np.float32))
                            for k, v in dens.items()}}
        for g in gts:
            own = dens[g["name"]]
            if len(own) < 50:
                continue
            ys = sorted(own); xs = [own[y] for y in ys]
            # shift — полоса СМЕЩЕНА относительно кривой (модель отказа прода: U1 нашёл полосу,
            # но она стоит на соседе). base тоже съезжает: он берётся из полосы, а не из GT.
            lo = max(0, int(min(xs) - pad) - BAND_PAD + shift)
            hi = min(W, int(max(xs) + pad) + BAND_PAD + 1 + shift)
            rec = {"name": g["name"], "short": g["name"].split()[0],
                   "lo": lo, "hi": hi, "y0": min(ys), "y1": max(ys),
                   "base": float(np.median(xs)) + shift, "runs": {}}
            for c in colors:
                rec["runs"][c] = _csr(fgs[c], lo, hi, H, im)
            sheet_rec["curves"].append(rec)
            print(f"   {well} {rec['short']:<8} полоса {hi-lo:>5}px  строки {rec['y0']}..{rec['y1']}")
        with open(cache / f"{well}.pkl", "wb") as f:
            pickle.dump(sheet_rec, f, protocol=5)
        print(f"== {well}: {len(sheet_rec['curves'])} кривых, цвета {colors}")


def _csr(fg, lo, hi, H, im):
    """Раны по всем строкам в полосе [lo,hi) → CSR. Смещение +lo уже применено (абсолютный x).
    Кэшируются ВСЕ строки листа, а не только [y0,y1]: _extend_ends уходит за пределы кривой."""
    A, B, Cc, ptr = [], [], [], np.zeros(H + 1, np.int64)
    for y in range(H):
        for a, b, c in im.row_runs(fg[y, lo:hi]):
            A.append(a + lo); B.append(b + lo); Cc.append(c + lo)
        ptr[y + 1] = len(A)
    # ЦЕНТР рана — строго float64: row_runs отдаёт seg.mean() в float64, и округление до float32
    # сдвигало бы argmin на «ничьих» ⇒ стенд перестал бы быть бит-в-бит равен проду.
    return (np.array(A, np.int32), np.array(B, np.int32),
            np.array(Cc, np.float64), ptr)


def load(cache=None):
    cache = Path(cache) if cache else CACHE
    out = []
    for rel in SH:
        well = (ARCH / rel).parent.parent.name
        f = cache / f"{well}.pkl"
        if f.exists():
            with open(f, "rb") as fh:
                out.append(pickle.load(fh))
    return out


# ───────────────────────── воспроизведение trace_line ─────────────────────────

def _runs_at(csr, y):
    A, B, Cc, ptr = csr
    i, j = ptr[y], ptr[y + 1]
    return A[i:j], B[i:j], Cc[i:j]


def trace(rec, csr, H, chooser=None, slmax=SLMAX, wide_run=WIDE_RUN, probe=None):
    """БИТ-В-БИТ повтор auto.trace2d.trace_line над кэшем.

    chooser=None — штатное поведение (ближайший ран к предсказанию, БЕЗ ограничения расстояния).
    chooser(A,B,Cc, pred, x, v, st) -> индекс рана | None — подменяет ТОЛЬКО ветку else
    («нет рана под предсказанием»), т.е. ровно то место, куда бьёт §6.19. None = коастить.
    st — словарь состояния стратегии (живёт на всю кривую): ширины ранов, история и т.п.
    probe(y, A, B, Cc, pred, x, v, branch, k) — ТОЛЬКО наблюдение (диагностика), на решения не
    влияет: инструментируется ТА ЖЕ траектория, что и без него.
    """
    lo, hi, base = rec["lo"], rec["hi"], rec["base"]
    x = None; v = 0.0; tr = {}; st = {"lo": lo, "hi": hi, "base": base, "widths": []}
    for y in range(max(0, rec["y0"]), min(H, rec["y1"] + 1)):
        A, B, Cc = _runs_at(csr, y)
        if not len(A):
            if x is not None:
                x = x + float(np.clip(v, -slmax, slmax))
            if probe:
                probe(y, A, B, Cc, x, x, v, "empty", None)
            continue
        if x is None:
            k = int(np.argmin(np.abs(Cc - base)))
            x = float(Cc[k]); v = 0.0; tr[y] = x
            if probe:
                probe(y, A, B, Cc, base, x, v, "init", k)
            continue
        pred = x + float(np.clip(v, -slmax, slmax))
        cont = np.nonzero((A - 2 <= pred) & (pred <= B + 2))[0]
        if len(cont):
            k = int(cont[np.argmin(np.abs(Cc[cont] - pred))]); branch = "cont"
        elif chooser is None:
            k = int(np.argmin(np.abs(Cc - pred))); branch = "else"
        else:
            k = chooser(A, B, Cc, pred, x, v, st); branch = "else"
            if k is None:                       # стратегия отказалась — коаст по инерции
                if probe:
                    probe(y, A, B, Cc, pred, x, v, "coast", None)
                x = pred; continue
        if probe:
            probe(y, A, B, Cc, pred, x, v, branch, k)
        a, b, c = int(A[k]), int(B[k]), float(Cc[k])
        nx = (b if abs(b - base) >= abs(a - base) else a) if (b - a) >= wide_run else c
        v = 0.6 * v + 0.4 * (nx - x); x = float(nx); tr[y] = float(nx)
        st["widths"].append(b - a)
    _extend_ends(tr, csr, H, slmax)
    return tr


def _extend_ends(tr, csr, H, slmax, max_gap=MAX_GAP):
    if not tr:
        return
    for direction in (-1, +1):
        y0 = min(tr) if direction < 0 else max(tr)
        x = tr[y0]; gap = 0; y = y0 + direction
        while 0 <= y < H and gap <= max_gap:
            A, B, Cc = _runs_at(csr, y)
            if not len(A):
                gap += 1; y += direction; continue
            k = int(np.argmin(np.abs(Cc - x)))
            a, b, c = int(A[k]), int(B[k]), float(Cc[k])
            if abs(c - x) > slmax + (b - a):
                break
            x = c; tr[y] = float(c); gap = 0; y += direction


# ───────────────────────────── метрики ─────────────────────────────

def _gt_matrix(sheet):
    """Плотная матрица GT (кривая × строка), NaN где кривой на строке нет. Кэшируется на листе.
    Нужна, чтобы скоринг был векторным: наивный цикл «строка × все кривые листа» давал 4 млн
    вызовов numpy и съедал бОльшую часть прогона."""
    if "_mat" not in sheet:
        names = list(sheet["gt"].keys())
        mat = np.full((len(names), sheet["H"]), np.nan)
        for i, nm in enumerate(names):
            ys, xs = sheet["gt"][nm]
            mat[i, ys] = xs
        sheet["_mat"] = (names, mat)
    return sheet["_mat"]


def _unpack(tr):
    ys = np.fromiter(tr.keys(), np.int64, len(tr))
    xs = np.fromiter(tr.values(), np.float64, len(tr))
    o = np.argsort(ys)
    return ys[o], xs[o]


def score_curve(tr, rec, sheet, tol=10.0):
    """med/cov/<=3px + разбор ошибки (своя / ЛАТЧ на чужую / между кривыми).
    cov — доля строк ЭКСПЕРТНОЙ кривой, которые мы покрыли (без неё med врёт, §6.9)."""
    if not tr:
        return None
    names, mat = _gt_matrix(sheet)
    oi = names.index(rec["name"])
    ys, xs = _unpack(tr)
    own_x = mat[oi, ys]
    common = ~np.isnan(own_x)
    if int(common.sum()) < 30:
        return None
    d = np.abs(xs[common] - own_x[common])
    dist = np.abs(mat[:, ys] - xs)                      # (кривых, строк), NaN где кривой нет
    dist = np.where(np.isnan(dist), np.inf, dist)
    best_i = np.argmin(dist, axis=0)
    best_d = dist[best_i, np.arange(len(ys))]
    seen = np.isfinite(best_d)                          # строки, где ХОТЬ ОДНА кривая размечена
    near = seen & (best_d <= tol)
    cnt = {"своя": int((near & (best_i == oi)).sum()),
           "латч": int((near & (best_i != oi)).sum()),
           "между": int((seen & ~near).sum())}
    s = sum(cnt.values()) or 1
    return {"med": float(np.median(d)), "cov": int(common.sum()) / len(sheet["gt"][rec["name"]][0]),
            "le3": float((d <= 3).mean()),
            **{k: 100.0 * v / s for k, v in cnt.items()}}


def run_strategy(chooser=None, tracer=None, tol=10.0, cache=None, color="oracle"):
    """Прогнать стратегию по всем кривым.

    ⚠ color="oracle" (умолчание, унаследовано от `_latch_probe.py`) — цвет маски выбирается ПО
    ЛУЧШЕМУ med относительно эксперта. Это ОРАКУЛ: в проде цвет так выбрать нельзя. Замер 22.07:
    оракульный цвет совпадает с доступным без GT критерием лишь у 38% кривых (на 3 листах из 5
    оракул берёт black там, где «больше туши» указывает на red) ⇒ все числа стенда (§6.15, §6.20,
    §6.25-§6.29) получены ПРИ ОРАКУЛЬНОМ ЦВЕТЕ, и это надо цитировать вместе с ними.

    color="jitter" — честная альтернатива БЕЗ GT: берётся трасса с наименьшим типичным рывком
    |Δx| между соседними строками (своя кривая идёт гладко, чужая маска даёт рванину).
    """
    rows = []
    for sheet in load(cache):
        H = sheet["H"]
        names, mat = _gt_matrix(sheet)
        for rec in sheet["curves"]:
            oi = names.index(rec["name"])
            best, best_med = None, 1e9
            for c in sheet["colors"]:
                csr = rec["runs"][c]
                tr = (tracer(rec, csr, H) if tracer else trace(rec, csr, H, chooser))
                if not tr:
                    continue
                ys, xs = _unpack(tr)
                own_x = mat[oi, ys]
                m = ~np.isnan(own_x)
                if int(m.sum()) < 30:
                    continue
                if color == "jitter":
                    key = float(np.median(np.abs(np.diff(xs)))) if len(xs) > 1 else 1e9
                else:
                    key = float(np.median(np.abs(xs[m] - own_x[m])))
                if key < best_med:
                    best_med, best = key, tr
            if best is None:
                continue
            sc = score_curve(best, rec, sheet, tol)
            if sc:
                rows.append({"well": sheet["well"], "curve": rec["short"],
                             "band": rec["hi"] - rec["lo"], **sc})
    return rows


def report(name, rows):
    honest = [r for r in rows if r["med"] <= 3 and r["cov"] >= 0.9]
    med_med = float(np.median([r["med"] for r in rows])) if rows else float("nan")
    med_cov = float(np.median([r["cov"] for r in rows])) if rows else float("nan")
    print(f"\n=== {name} — {len(rows)} кривых ===")
    print(f"   med(med) {med_med:8.1f}px   med(cov) {med_cov:.2f}   "
          f"★ЧЕСТНЫХ (med<=3 И cov>=0.9) {len(honest)}/{len(rows)}")
    tot = {k: float(np.mean([r[k] for r in rows])) for k in ("своя", "латч", "между")} if rows else {}
    if tot:
        print(f"   своя {tot['своя']:.1f}%   ЛАТЧ {tot['латч']:.1f}%   между {tot['между']:.1f}%")
    return {"name": name, "med_med": med_med, "med_cov": med_cov,
            "honest": len(honest), "n": len(rows), "rows": rows}


def verify():
    """ГЕЙТ: baseline со стенда обязан совпасть с _latch_probe.py (output/taskS/latch.json)."""
    ref = json.loads((OUT / "latch.json").read_text(encoding="utf-8"))
    refmap = {(r["well"], r["curve"]): r["med"] for r in ref["per_curve"]}
    rows = run_strategy()
    bad = 0
    print(f"\n=== СВЕРКА baseline vs _latch_probe.py ===")
    for r in rows:
        k = (r["well"], r["curve"])
        if k not in refmap:
            print(f"   ?? {k} нет в эталоне"); bad += 1; continue
        if abs(refmap[k] - r["med"]) > 0.05:
            print(f"   ✗ {k[0]:<12}{k[1]:<8} стенд {r['med']:8.1f}  эталон {refmap[k]:8.1f}")
            bad += 1
    miss = set(refmap) - {(r["well"], r["curve"]) for r in rows}
    for k in sorted(miss):
        print(f"   ✗ пропала кривая {k}"); bad += 1
    print(f"   {'✓ ГЕЙТ ЗЕЛЁНЫЙ' if not bad else f'✗ РАСХОЖДЕНИЙ {bad}'} "
          f"({len(rows)} кривых сверено)")
    return bad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--pad", type=int, default=20, help="полоса шире размаха GT на столько px")
    ap.add_argument("--cache", default=None, help="каталог кэша (для pad != 20 — ОБЯЗАТЕЛЬНО свой)")
    ap.add_argument("--shift", type=int, default=0, help="сдвиг полосы вбок, px (модель отказа прода)")
    a = ap.parse_args()
    if a.build:
        build(a.pad, a.cache, a.shift)
    elif a.verify:
        sys.exit(0 if verify() else 1)
    else:
        report("baseline (как в проде)", run_strategy())

r"""
digitize_b3.py — СКВОЗНАЯ СБОРКА нового пайплайна (Фаза A+B, без U-Net/torch — чистый cv2/py3.14):
  скан + nlgx-шаблон → B1 инстансы → B3 трекер-идентичность → вписать трассы → сдаваемый _auto.nlgx(+bck).

Конвейер: extract(template) → track_identity (цвет→штрихи→сшивка) → для каждой кривой шаблона
АГРЕГИРУЕМ одноцветные линии, трассирующие её (перевынос = несколько линий-уровней) → заменяем
тег 35490 (X-по-строкам) → опц. A2b-снап Depth Grid (--grid4m) + патч скан-пути (--patch-scan) →
write nlgx(+bck) + верификация. Шаблон даёт калибровку/уровни/значения; трекер даёт ФОРМУ; трасса
шаблона нужна только для маппинга линий к кривым (как digitize.py).

  python digitize_b3.py <файл.nlgx | папка> [--image f.jpg] [--out DIR] [--regrid] [--patch-scan]
  (--regrid: пересобрать Depth Grid на НАТИВНОМ шаге шаблона; по умолчанию сетку шаблона НЕ трогаем)

Папка → батч по всем *.nlgx (кроме *_auto) + CSV-сводка. Результат открывается в NeuraLOG для QC.
"""
import sys, struct, csv
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from extract_nlgx import extract, NULL
from dataset_build import find_image
from write_nlgx import read_full, write_full, write_bck, set_tag, find_ifd
from set_depth_grid import regrid
import dataset as ds
import track_identity as ti
import extract_instances as ei
import behavior_priors as bp


def curve_pred(short):
    def is_this(tags):
        if 34768 not in tags or struct.unpack("<I", tags[34768][2][:4])[0] != 7:
            return False
        if 35470 not in tags:
            return False
        nm = tags[35470][2].split(b"\x00")[0].decode("latin1")
        return nm.startswith(short + " ") or nm == short
    return is_this


def aggregate_xs(lines, gd, gtc, n, top_y, tol=12, row_tol=25, chan=None, fill_gate=14,
                 occ_lines=None, fill_gate2=6, occ_max=20):
    """Собрать X-трассу кривой из одноцветных B3-линий, трассирующих её (медиана |Δx|<tol).
    Перевынос = несколько линий-уровней → берём ближайшую к шаблонной трассе на каждой строке.
    PER-ROW ГЕЙТ (row_tol): точку пишем ТОЛЬКО если она близка к эталону на ЭТОЙ строке —
    иначе на границе перевыноса штрих, совпавший по медиане, давал ЧУЖУЮ линию (x=529 вместо
    25×-линии слева, баг QC). Отсечённое = РАЗРЫВ (NULL), а не чужая линия (заодно анти-мост).
    GAP-FILL (chan): где B3-линии дали разрыв (same-color пересечение слило CC), но эталон есть —
    берём ближайшее ЧЕРНИЛО того же цвета в узком гейте у эталона (трасса из ИЗОБРАЖЕНИЯ, шаблон
    лишь указывает какая линия; ловит входящую backup-линию ×5/×25 у перехода).
    Возвращает (new_xs[n], n_заполнено, own_px)."""
    agg = {}
    for L in lines:
        if L["color"] != gtc:
            continue
        ov = [(y, gd[y], L["tr"][y]) for y in gd if y in L["tr"]]
        if len(ov) < 20:
            continue
        if np.median([abs(a - b) for _, a, b in ov]) < tol:
            for y, gx, bx in ov:
                if abs(bx - gx) > row_tol:           # грубое расхождение на строке → разрыв
                    continue
                if y not in agg or abs(bx - gx) < abs(agg[y] - gx):
                    agg[y] = bx
    # ПЛОТНЫЙ ink-follow: на КАЖДОЙ строке диапазона снапим к ближайшему чернилу СВОЕГО ЦВЕТА у
    # направляющей (интерполяция эталона). Ловит КОНЧИКИ ПИКОВ (на строке пика чернило = вершина),
    # а не срезает их прямой между разреженными точками эталона.
    def _snap(mask, y, gx, gate):
        H, W = mask.shape
        if not (0 <= y < H):
            return None
        x0 = max(0, int(gx) - gate); x1 = min(W, int(gx) + gate)
        idx = np.nonzero(mask[y, x0:x1])[0]
        if len(idx):
            xs = idx + x0
            return float(xs[np.argmin(np.abs(xs - gx))])
        return None
    gys = sorted(gd)
    if len(gys) >= 2 and chan is not None:
        ga = np.array(gys, float); xa = np.array([gd[y] for y in gys], float)
        for y in range(gys[0], gys[-1] + 1):
            if y in agg:
                continue
            v = _snap(chan, y, float(np.interp(y, ga, xa)), fill_gate)
            if v is not None:
                agg[y] = v
        # ОККЛЮЗИЯ через ИДЕНТИЧНОСТЬ B3: своего цвета нет в КОРОТКОМ разрыве, ОБРАМЛЁННОМ own-color
        # точками (вход/выход из-под другой линии) → ведём по ПЕРЕКРЫВАЮЩЕЙ B3-ЛИНИИ, стоящей у
        # направляющей бракета. Это РЕАЛЬНАЯ кривая (GZ1), а не грид/baseline — поэтому бледная MDS
        # (где у направляющей НЕТ B3-линии) НЕ заливается (фикс свопов широкого any-ink). Заливаем
        # лишь если перекрыватель НЕПРЕРЫВЕН (≥80% строк разрыва).
        if occ_lines:
            ays = sorted(agg)
            for i in range(1, len(ays)):
                ya, yb = ays[i - 1], ays[i]
                if not (1 < yb - ya <= occ_max):
                    continue
                xa2, xb2 = agg[ya], agg[yb]
                cand = {}
                for y in range(ya + 1, yb):
                    gx = xa2 + (xb2 - xa2) * (y - ya) / (yb - ya)
                    best, bd = None, fill_gate2
                    for L in occ_lines:
                        lx = L["tr"].get(y)
                        if lx is not None and abs(lx - gx) < bd:
                            bd = abs(lx - gx); best = lx
                    if best is not None:
                        cand[y] = best
                if len(cand) >= 0.8 * (yb - ya - 1):
                    agg.update(cand)
    new_xs = [int(round(agg[top_y + i])) if (top_y + i) in agg else NULL for i in range(n)]
    # own_px только по строкам, где есть И эталон (agg содержит и доплотнённые строки вне gd)
    dd = [abs(agg[y] - gd[y]) for y in agg if y in gd]
    own = float(np.median(dd)) if dd else None
    return new_xs, len(agg), own


def digitize_one(nlgx, image=None, out=None, do_regrid=False, patch_scan=False,
                 las=False, verbose=True):
    nlgx = str(nlgx)
    out = Path(out) if out else Path(r"F:\nds\output")
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(nlgx).stem
    res = {"stem": stem, "curves": 0, "written": 0, "verified": 0,
           "cover_pct": None, "own_px": None, "dst": None, "error": None}
    try:
        m = extract(nlgx)
        img = Path(image) if image else find_image(Path(nlgx))
        if not img or not Path(img).is_file():
            res["error"] = "нет картинки"
            if verbose: print(f"  {stem[:50]:<50} ПРОПУСК: нет картинки")
            return res
        rgb = np.asarray(Image.open(img).convert("RGB"))
        H, W = rgb.shape[:2]
        lines, masks = ti.track_identity(rgb, m)
        # цветовые каналы для gap-fill на ПОЛНОМ теле трека (без вычитания масок): A3 на 1:500
        # съедала кривую — и грид (exclude ~46%), и саму сжатую пиковую кривую как «текст».
        # Цветной грид/текст редок (red/blue/green безопасны); ЧЁРНЫЙ — минус весь exclude (грид чёрный).
        g = masks["geom"]; xl = max(0, g["x_left"] - 5); xr = min(W, (g["x_right"] or W) + 30)
        body = np.zeros((H, W), np.uint8); body[g["top_y"]:g["bottom_y"], xl:xr] = 1
        chans = ei.classify_ink(rgb, body)
        chans["black"] = (chans["black"] & (masks["exclude"] == 0)).astype(np.uint8)
        curves = ds.real_curves(m)
        res["curves"] = len(curves)

        ifds = read_full(open(nlgx, "rb").read())
        written = []; owns = []; covs = []
        for c in curves:
            short = c["name"].split()[0]
            ty = c["top_y"]; n = len(c["xs"])
            gd = {ty + i: x for i, x in enumerate(c["xs"]) if x != NULL and 0 <= ty + i < H}
            if len(gd) < 30:
                continue
            gtc = ei._gt_color(rgb, gd)
            # окклюзия через B3-идентичность ТОЛЬКО для РЕЗИСТИВНЫХ кривых (GZ/BK/... пересекаются;
            # caliper DS/MDS — одиночные, не окклюзируются → не свопаем). Перекрыватели = B3-линии
            # другого цвета (GZ1 для GZ2).
            occ = [L for L in lines if L["color"] != gtc] if bp.curve_class(short) == "RES" else None
            new_xs, nfill, own = aggregate_xs(lines, gd, gtc, n, ty,
                                              chan=chans.get(gtc), occ_lines=occ)
            if nfill < 20:
                continue
            idxs = find_ifd(ifds, curve_pred(short))
            if not idxs:
                continue
            set_tag(ifds, idxs[0], 35490, 4, new_xs)
            rng = max(gd) - min(gd) + 1          # покрытие = доля ГЛУБИННОГО диапазона (fill плотный)
            written.append(short); owns.append(own); covs.append(min(100.0, nfill / rng * 100))
        res["written"] = len(written)
        res["own_px"] = round(float(np.median(owns)), 1) if owns else None
        res["cover_pct"] = round(float(np.median(covs)), 0) if covs else None
        if do_regrid:
            # шаг — НАТИВНЫЙ из шаблона (BKZ=4м, BK=10м, 1:500=10м), НЕ хардкод 4м.
            # По умолчанию сетку шаблона не трогаем (она уже верна — экспертная).
            step = m.get("depth_grid", {}).get("step_m") or 4.0
            regrid(ifds, m, float(step))
        if patch_scan:
            for i in find_ifd(ifds, lambda tags: 34878 in tags):
                set_tag(ifds, i, 34878, 2, str(img))

        data = write_full(ifds)
        dst = out / f"{stem}_auto.nlgx"
        open(dst, "wb").write(data)
        open(out / f"{stem}_auto.bck", "wb").write(write_bck(data))
        res["dst"] = str(dst)
        if las:                                  # достроить сдачу: nlgx+bck+LAS
            import export_las
            export_las.export(str(dst), verbose=False)

        # верификация: переоткрыть, сверить что записанные xs совпали с агрегатом
        m2 = extract(str(dst)); ok = 0
        for c2 in ds.real_curves(m2):
            short = c2["name"].split()[0]
            if short not in written:
                continue
            if any(x != NULL for x in c2["xs"]):
                ok += 1
        res["verified"] = ok
        if verbose:
            print(f"  {stem[:48]:<48} кривых={res['curves']} вписано={res['written']} "
                  f"покрытие≈{res['cover_pct']}% own≈{res['own_px']}px verified={ok}")
    except Exception as e:
        res["error"] = repr(e)[:140]
        if verbose: print(f"  {stem[:48]:<48} ОШИБКА {res['error']}")
    return res


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__); return
    target = a[0]
    image = a[a.index("--image") + 1] if "--image" in a else None
    out = a[a.index("--out") + 1] if "--out" in a else None
    do_regrid = "--regrid" in a          # пересобрать сетку (нативный шаг); по умолч. НЕ трогаем
    las = "--las" in a                   # достроить сдачу LAS (значения)
    patch_scan = "--patch-scan" in a
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    p = Path(target)
    if p.is_dir():
        files = sorted(f for f in p.glob("*.nlgx") if "_auto" not in f.stem)
        print(f"БАТЧ: {len(files)} файлов в {p}")
        rows = []
        for f in files:
            rows.append(digitize_one(f, None, out, do_regrid, patch_scan, las))
        okn = sum(1 for r in rows if r["dst"])
        cov = [r["cover_pct"] for r in rows if r["cover_pct"] is not None]
        print(f"\nГОТОВО: {okn}/{len(files)} файлов; медиана покрытия "
              f"{np.median(cov) if cov else '-'}%")
        csvp = (Path(out) if out else Path(r"F:\nds\output")) / "digitize_b3_summary.csv"
        with open(csvp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"CSV -> {csvp}")
    else:
        r = digitize_one(target, image, out, do_regrid, patch_scan, las)
        print(f"-> {r.get('dst')}")


if __name__ == "__main__":
    main()

r"""Выгрузка сдачи MK: трассы (<stem>_traces.npz) + шаблон <stem>.nlgx → <stem>_auto.nlgx + .bck + .las.

Инъекция по мнемонике (MGZ→MGZ*, MPZ→MPZ*), теги per-кривая: 35490 X-по-строкам (NULL вне трассы),
сегменты 35492..35498 = один full-range сегмент level=0 (перевыносы — отдельный этап), bbox
35478..35484 (=min/max±1, семантика сверена по GT BEZLUD), плюс патч скан-пути 34878 на реальный
(шаблоны часто ссылаются на мёртвый путь — NeuraLOG не грузит скан; кодировка cp1251 — latin1
падает на кириллице). bck = ТОЧНАЯ КОПИЯ nlgx: наблюдение «флаг off=5489» было артефактом —
на Yatskivska этот байт лежит ВНУТРИ данных трассы MPZ1, write_bck молча портил кривую;
реальный .bck NeuraLOG — предыдущее сохранение, т.е. валидный nlgx (анализ 02.07).
LAS — через export_las (значения из scale-осей шаблона). Верификация: extract() выхода + сверка трасс.

  python mk_export.py <traces.npz> <template.nlgx> [--scan img.jpg] [--out DIR] [--no-las]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from write_nlgx import read_full, write_full, set_tag, find_ifd
from extract_nlgx import extract, NULL
import dataset as ds


def _load_traces(npz, hole_max=5):
    """Трассы из npz + интерполяция микро-дыр (<hole_max строк): меньше кликов эксперту;
    большие разрывы сохраняем (домен: перевыносы/отрывы = отдельные линии с разрывом)."""
    d = np.load(npz)
    out = []
    for pre in ("mgz", "mpz"):
        tr = {int(y): float(x) for y, x in zip(d[f"{pre}_y"], d[f"{pre}_x"])}
        ys = sorted(tr)
        for a, b in zip(ys, ys[1:]):
            if 1 < b - a <= hole_max:
                for y in range(a + 1, b):
                    tr[y] = tr[a] + (tr[b] - tr[a]) * (y - a) / (b - a)
        out.append(tr)
    return out


def _curve_ifd(ifds, model, mnem):
    """(idx IFD, curve-dict) кривой с мнемоникой mnem (MGZ/MPZ) — по имени 35470.
    min_pts=1: в шаблоне-рамке кривые — заглушки на десяток точек (real_curves их режет)."""
    import struct
    for c in ds.real_curves(model, min_pts=1):
        if ds.mnemonic(c["name"]).upper() != mnem:
            continue
        def is_this(tags, name=c["name"]):
            if 34768 not in tags or struct.unpack("<I", tags[34768][2][:4])[0] != 7:
                return False
            return 35470 in tags and tags[35470][2].split(b"\x00")[0].decode("latin1") == name
        idxs = find_ifd(ifds, is_this)
        if idxs:
            return idxs[0], c
    return None, None


def qc_render(nlgx_path, out_dir, n=4, half_h=130):
    """QC-кропы ИЗ ЗАПИСАННОГО nlgx поверх скана (проверяем сдаваемый файл, не трассы в памяти).
    Глубина робастно по top_y/bottom_y (span_px в свежих рамках = 0)."""
    from PIL import Image, ImageDraw, ImageFont
    Image.MAX_IMAGE_PIXELS = None
    m = extract(str(nlgx_path))
    da = m.get("depth_axis")
    ip = m.get("img_path")
    if not da or not ip or not Path(ip).is_file():
        return []
    img = np.asarray(Image.open(ip).convert("RGB")).copy()
    H, W = img.shape[:2]
    cols = {}; xs_all = []
    for c in m["curves"]:
        mn = ds.mnemonic(c["name"]).upper()
        if mn not in ("MGZ", "MPZ"):
            continue
        col = (220, 0, 0) if mn == "MGZ" else (0, 110, 230)
        ty = c["top_y"]
        for i, x in enumerate(c["xs"]):
            if x == NULL:
                continue
            y = ty + i
            if 0 <= y < H and 0 <= x < W:
                img[y, max(0, x - 1):x + 2] = col
                xs_all.append(x)
    if not xs_all:
        return []
    x0, x1 = max(0, int(np.percentile(xs_all, 2)) - 60), min(W, int(np.percentile(xs_all, 98)) + 90)
    for gy in (m.get("depth_grid") or {}).get("ys", []):         # записанная сетка — видно снап к линиям
        if 0 <= gy < H:
            img[gy, x0:x1] = (0, 150, 60)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    ty, by, td, bd = da["top_y"], da["bottom_y"], da["top_depth"], da["bottom_depth"]
    outs = []
    for d in np.linspace(td, bd, n + 2)[1:-1]:
        d = round(float(d))
        yc = int(ty + (d - td) * (by - ty) / max(bd - td, 1e-6))
        if not (half_h <= yc < H - half_h):
            continue
        crop = img[yc - half_h:yc + half_h, x0:x1]
        im_c = Image.fromarray(crop).resize(((x1 - x0) * 3, half_h * 6), Image.NEAREST)
        ImageDraw.Draw(im_c).text((6, 4), f"{d}м из _auto.nlgx  MGZ=красн MPZ=син", fill=(0, 0, 0), font=font)
        p = Path(out_dir) / f"{Path(nlgx_path).stem}_qc_{d}m.png"
        im_c.save(p); outs.append(str(p))
    return outs


def export_bundle(traces_npz, template_nlgx, scan_path=None, out_dir=None, write_las=True, log=print,
                  grid_step=4.0):
    """Главный вход (и для GUI): пишет <stem>_auto.nlgx/.bck/.las. Возврат: dict-отчёт или None.
    grid_step: шаг Depth Grid в метрах (4 = каждые 10 клеток при 1:200); 0/None — не трогать сетку."""
    traces_npz, template_nlgx = str(traces_npz), str(template_nlgx)
    stem = Path(template_nlgx).stem
    out = Path(out_dir) if out_dir else Path(traces_npz).parent
    out.mkdir(parents=True, exist_ok=True)
    mgz, mpz = _load_traces(traces_npz)
    if not mgz or not mpz:
        log(f"  ! пустые трассы в {Path(traces_npz).name} — пропуск")
        return None
    model = extract(template_nlgx)
    ifds = read_full(open(template_nlgx, "rb").read())
    rep = {"stem": stem, "curves": []}
    for mnem, tr in (("MGZ", mgz), ("MPZ", mpz)):
        idx, c = _curve_ifd(ifds, model, mnem)
        if idx is None:
            log(f"  ! в шаблоне нет кривой {mnem} — пропуск канала")
            continue
        ty, n = c["top_y"], len(c["xs"])
        new_xs = [max(0, int(round(tr[ty + i]))) if (ty + i) in tr else NULL for i in range(n)]
        rows = [i for i, x in enumerate(new_xs) if x != NULL]
        vals = [x for x in new_xs if x != NULL]
        set_tag(ifds, idx, 35490, 4, new_xs)
        set_tag(ifds, idx, 35492, 4, 1)                          # 1 сегмент level=0 на всю трассу
        set_tag(ifds, idx, 35494, 4, ty + rows[0])
        set_tag(ifds, idx, 35496, 4, ty + rows[-1])
        set_tag(ifds, idx, 35498, 4, 0)
        set_tag(ifds, idx, 35478, 4, max(0, min(vals) - 1))      # bbox (min/max±1, как у GT)
        set_tag(ifds, idx, 35480, 4, max(0, ty + rows[0] - 1))
        set_tag(ifds, idx, 35482, 4, max(vals) + 1)
        set_tag(ifds, idx, 35484, 4, ty + rows[-1] + 1)
        rep["curves"].append((c["name"], len(rows)))
    if not rep["curves"]:
        log("  ! ни одной кривой не вписано — выгрузка отменена")
        return None
    if scan_path:                                                # NeuraLOG грузит скан по этому пути
        doc = find_ifd(ifds, lambda t: 34878 in t)
        if doc:
            set_tag(ifds, doc[0], 34878, 2, str(scan_path).encode("cp1251") + b"\x00")
    # Depth Grid: горизонтали каждые 10 клеток (4 м при 1:200) со СНАПОМ к реальным линиям
    # скана — опорные точки между крестиками, учёт изгиба/растяжки бумаги (QC эксперта 03.07)
    img_for_grid = scan_path or model.get("img_path")
    if grid_step and img_for_grid and Path(str(img_for_grid)).is_file():
        try:
            from set_depth_grid import regrid
            from detect_calibration import detect_bold_hgrid_multiband
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            import numpy as _np
            g = _np.asarray(Image.open(str(img_for_grid)).convert("L"))
            geo = detect_bold_hgrid_multiband(g)            # жирные робастно к подписям (QC №4)
            res = regrid(ifds, model, step=grid_step, bold_geo=geo if geo[0] else None)
            if res:
                rep["grid_lines"] = len(res[0])
                log(f"  Depth Grid: {len(res[0])} линий шаг {grid_step}м (жирных {len(geo[0])}, "
                    f"шаг ~{geo[1]:.0f}px, мультиполосный наклон + якорь в ось)")
            else:
                log("  ! Depth Grid: нет тип-8 IFD в шаблоне — пропуск")
        except Exception as e:
            log(f"  ! Depth Grid: {e}")
    data = write_full(ifds)
    dst = out / f"{stem}_auto.nlgx"
    dst.write_bytes(data)
    (out / f"{stem}_auto.bck").write_bytes(data)                 # bck = точная копия (см. докстрок)
    # верификация round-trip: наши трассы читаются назад с точностью округления
    m2 = extract(str(dst))
    ok = True
    for mnem, tr in (("MGZ", mgz), ("MPZ", mpz)):
        c2 = next((c for c in ds.real_curves(m2) if ds.mnemonic(c["name"]).upper() == mnem), None)
        if not c2:
            continue
        errs = [abs(c2["xs"][y - c2["top_y"]] - x) for y, x in tr.items()
                if 0 <= y - c2["top_y"] < len(c2["xs"]) and c2["xs"][y - c2["top_y"]] != NULL]
        med = float(np.median(errs)) if errs else 99.0
        if med > 0.51 or not errs:
            ok = False
        rep[f"rt_{mnem}"] = round(med, 2)
    rep["verify"] = "OK" if ok and m2["depth_axis"] == model["depth_axis"] else "FAIL"
    for name, npts in rep["curves"]:
        log(f"  вписано {name}: {npts} точек")
    log(f"  верификация round-trip: {rep['verify']} (median |Δx| MGZ={rep.get('rt_MGZ')} MPZ={rep.get('rt_MPZ')})")
    log(f"  -> {dst}")
    if write_las:
        try:
            from export_las import export as las_export
            r = las_export(str(dst), verbose=False)
            rep["las"] = r["las"] if r else None
            log(f"  -> {rep['las']}" if rep.get("las") else "  ! LAS: нет восстановимых кривых (шкалы шаблона)")
        except Exception as e:
            rep["las"] = None
            log(f"  ! LAS ошибка: {e}")
    try:
        qc = qc_render(dst, out)
        if qc:
            log(f"  QC-кропы из nlgx: {len(qc)} шт -> {out}")
    except Exception as e:
        log(f"  ! QC-рендер: {e}")
    return rep


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        print(__doc__); return
    sys.stdout.reconfigure(encoding="utf-8")
    scan = a[a.index("--scan") + 1] if "--scan" in a else None
    outd = a[a.index("--out") + 1] if "--out" in a else None
    export_bundle(a[0], a[1], scan_path=scan, out_dir=outd, write_las="--no-las" not in a)


if __name__ == "__main__":
    main()

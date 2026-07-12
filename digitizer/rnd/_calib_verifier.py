# -*- coding: utf-8 -*-
"""Калибровка verifier+decode_rail на Archive GT — ВСЕ кривые (542 скана).
Быстрый путь: trace2d ЗАСЕИВАЕТСЯ по экспертной кривой (band из GT x-диапазона) — пропускаем
медленный understand (313с/планшет на 62k-высотных сканах). Меряем:
  • dx (px) наша-трасса vs эксперт по-строчно — качество trace2d (спайки/дёрганье/недотяг);
  • level-acc наши decode_rail-уровни vs GT (тег 35498) — только резист. с 5х-цепочкой;
  • verify_line флаги: целятся ли спайк-строки в реальные ошибки (dx@спайк >> общей dx).
Инкрементально в calib_result.json. Аргументы: --limit N --offset K."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
import dataset as ds, decode_levels as DL
from auto import imaging as im, meta as M, frame as F, trace2d as T, refine as R
from auto.understand import Line
from auto.config import Config

ARCH = Path(r"F:\nds\projects\Archive")
LIMIT = int(sys.argv[sys.argv.index("--limit")+1]) if "--limit" in sys.argv else 40
OFFSET = int(sys.argv[sys.argv.index("--offset")+1]) if "--offset" in sys.argv else 0
WIN = int(sys.argv[sys.argv.index("--win")+1]) if "--win" in sys.argv else 10000  # окно строк для скорости
cfg = Config(); p = cfg.cv

def expert_trace(c):
    ty = c["top_y"]
    return {ty+i: x for i, x in enumerate(c["xs"]) if x != NULL}

def gt_levels(c):
    out = {}
    for s, e, l in (c.get("segments") or []):
        for y in range(s, e+1): out[y] = l
    return out

def curve_color(name):
    info = M.curve_info(name, cfg.mnemonics)
    return info.get("color") or "black"

plates = [n for wlg in sorted(ARCH.glob("*/wlg")) for n in sorted(wlg.glob("*.nlgx"))
          if "_auto" not in n.stem and find_image(n)]
plates = plates[OFFSET:OFFSET+LIMIT]
print(f"планшетов: {len(plates)}", flush=True)

rows = []
out_json = Path(__file__).with_name("calib_result.json")
fg_cache = {}
t0 = time.time()
for pi, nlgx in enumerate(plates):
    try:
        model = extract(str(nlgx)); img = find_image(nlgx)
        rgb_full = im.load_rgb(str(img)); m = M.parse_filename(str(img), cfg.mnemonics)
        fr = F.frame_from_nlgx(str(nlgx), m, p, rgb=rgb_full)
        # ОКНО строк для скорости: от top_y на WIN строк (или весь, если короче)
        wy0 = fr.top_y; wy1 = min(rgb_full.shape[0], wy0 + WIN)
        rgb = rgb_full[wy0:wy1]
    except Exception as e:
        print("ERR", nlgx.stem[:28], repr(e)[:50]); continue
    fg_cache.clear()
    for c in ds.real_curves(model):
        et0 = expert_trace(c)
        et = {y - wy0: x for y, x in et0.items() if wy0 <= y < wy1}   # в координаты окна
        if len(et) < 100: continue
        ys = np.array(sorted(et)); xs = np.array([et[y] for y in ys])
        color = curve_color(c["name"])
        if color not in fg_cache:
            fg_cache[color] = T._color_fg(rgb, color, p)
        fg = fg_cache[color]
        # засев: Line по экспертному band (расширяем, чтобы наш трейсер шёл свободно)
        L = Line(track_index=0, color=color, x_center=float(np.median(xs)),
                 x_lo=float(xs.min())-10, x_hi=float(xs.max())+10,
                 y0=int(ys.min()), y1=int(ys.max()), thickness=3, rough_n=None,
                 behavior="peaky", n_strokes=1, density=1.0)
        tr = T.trace_line(fg, L, fr, p)
        if len(tr) < 100: continue
        tr_ds, nsp0 = R.despike(tr)          # деспайкнутая
        common = sorted(set(tr_ds) & set(et))
        if len(common) < 100: continue
        dx = np.array([abs(tr_ds[y]-et[y]) for y in common])
        dx_raw = np.array([abs(tr[y]-et[y]) for y in common if y in tr])
        fam = DL.build_family(model, c)
        is5x = DL.is_resistive(c["name"]) and len(fam) >= 2
        lacc = None
        if is5x:
            ratio = max(1.5, abs(fam[1]["v_right"]-fam[1]["v_left"])/(abs(fam[0]["v_right"]-fam[0]["v_left"]) or 1))
            lv = R.decode_rail(tr_ds, fam[0]["x_left"], fam[0]["x_right"], ratio=ratio, min_run=p.level_min_run)
            gl0 = gt_levels(c); gl = {y-wy0: l for y, l in gl0.items()}   # в координаты окна
            lvc = [y for y in common if y in gl]
            lacc = float(np.mean([lv.get(y,0)==gl[y] for y in lvc])) if lvc else None
        else:
            lv = {y:0 for y in tr_ds}
        v = R.verify_line(tr_ds, lv, min_run=p.level_min_run)
        sp = set(v["spike_rows"]) & set(common)
        dx_sp = float(np.median([abs(tr_ds[y]-et[y]) for y in sp])) if sp else None
        rows.append(dict(plate=nlgx.stem[:22], curve=c["name"].split()[0], cls=M.curve_class(c["name"]),
                         n=len(common), dx_med=float(np.median(dx)), dx_p90=float(np.percentile(dx,90)),
                         dx_raw_med=float(np.median(dx_raw)) if len(dx_raw) else None,
                         despiked=nsp0, is5x=is5x, lvl_acc=lacc, spikes=v["n_spikes"], dx_on_spike=dx_sp))
    out_json.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    el = time.time()-t0
    print(f"  [{pi+1}/{len(plates)}] {nlgx.stem[:20]:<22} кривых={len(rows)} {el:.0f}s ({el/(pi+1):.1f}/шт)", flush=True)

if rows:
    dxm = np.array([r["dx_med"] for r in rows]); dxr = np.array([r["dx_raw_med"] for r in rows if r["dx_raw_med"] is not None])
    la = np.array([r["lvl_acc"] for r in rows if r["lvl_acc"] is not None])
    dsp = [r["dx_on_spike"] for r in rows if r["dx_on_spike"] is not None]
    print(f"\n=== СВОДКА {len(rows)} кривых / {len(set(r['plate'] for r in rows))} планшетов")
    print(f"dx_med (деспайк): медиана={np.median(dxm):.1f}px ≤2px={np.mean(dxm<=2)*100:.0f}% ≤5px={np.mean(dxm<=5)*100:.0f}% p90={np.percentile(dxm,90):.1f}")
    print(f"dx_med (сырая): медиана={np.median(dxr):.1f}px  → деспайк улучшил на {np.median(dxr)-np.median(dxm):+.1f}px")
    by = {}
    for r in rows: by.setdefault(r["cls"], []).append(r["dx_med"])
    for cls,vv in sorted(by.items()): print(f"   {cls:<6}: n={len(vv)} dx_med медиана={np.median(vv):.1f}")
    if len(la): print(f"level-acc (decode_rail vs GT, резист.): медиана={np.median(la):.2f} ≥0.9={np.mean(la>=0.9)*100:.0f}% n={len(la)}")
    if dsp: print(f"dx@спайк-строки медиана={np.median(dsp):.1f}px vs общая {np.median(dxm):.1f}px — верификатор целит в ошибки: {np.median(dsp)>2*np.median(dxm)}")

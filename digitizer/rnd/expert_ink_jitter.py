"""
Этап A (продолжение, 04.07): дублей D1/D2 в эталоне НЕТ (это разные зонды одной
БКЗ-серии, подтверждено шапками) — потолок эксперт-vs-эксперт не измерить.
Альтернативный потолок: ЭКСПЕРТ-vs-ЧЕРНИЛА — насколько сама трасса эксперта
отклоняется от центра тёмного штриха. Это (а) собственный джиттер кликов/хорд
эксперта, (б) прямой ориентир для планки «наша трасса vs эталон ≤3px»: точнее
центра штриха эксперт сам не сидит.

Метод: для каждой валидной строки трассы берём горизонтальную полоску ±W px
вокруг экспертного x; если в ±SEED px от x есть тёмный пиксель (<120, как в
batch_validate.align_frac) — расширяем непрерывный тёмный ран и меряем
offset = x_expert - центр рана. Раны шире MAX_RUN px (слипание/клякса) и
строки мимо чернил считаем отдельно (miss), в джиттер не включаем.
"""
import sys, os, glob, json
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from extract_nlgx import extract, NULL
from dataset import real_curves

ARCHIVE = r"F:\nds\projects\Archive"
WELLS = ["Yatskivska_001", "Pn_Zavoda_001"]
OUT_DIR = r"F:\nds\output\expert_dup_ceiling"
THR = 120        # чернила: gray < THR (конвенция batch_validate)
W = 25           # полуширина полоски анализа
SEED = 2         # эксперт «на чернилах», если тёмное в ±SEED px
MAX_RUN = 22     # ран шире этого = слипание двух линий/клякса — не меряем
ROW_STEP = 4     # каждая 4-я строка (статистики хватает, время экономим)
MAX_ROWS = 4000  # на кривую


def find_image(nlgx_path):
    stem = os.path.splitext(os.path.basename(nlgx_path))[0]
    img_dir = os.path.join(os.path.dirname(os.path.dirname(nlgx_path)), "img")
    for ext in (".jpg", ".png", ".jpeg", ".tif"):
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def curve_offsets(gray, curve):
    """(offsets, n_meas, n_miss, n_wide) — offset подписанный, px."""
    H, Wimg = gray.shape
    ty = curve["top_y"]
    pts = [(ty + i, x) for i, x in enumerate(curve["xs"]) if x != NULL]
    pts = pts[::ROW_STEP][:MAX_ROWS]
    offs, miss, wide = [], 0, 0
    for y, x in pts:
        if not (0 <= y < H and W <= x < Wimg - W):
            continue
        strip = gray[y, x - W:x + W + 1] < THR
        c = W
        # сид: ближайший тёмный к центру в ±SEED
        seed = None
        for d in range(SEED + 1):
            if strip[c - d]: seed = c - d; break
            if strip[c + d]: seed = c + d; break
        if seed is None:
            miss += 1
            continue
        L = seed
        while L > 0 and strip[L - 1]:
            L -= 1
        R = seed
        while R < 2 * W and strip[R + 1]:
            R += 1
        if (R - L + 1) > MAX_RUN:
            wide += 1
            continue
        offs.append(((L + R) / 2.0) - c)
    return np.array(offs), len(pts), miss, wide


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    files = []
    # ⚠ §6.108: список планшетов — АРГУМЕНТОМ, а не зашитыми двумя скважинами. Джиттер эталона
    # нужен ПО СОРТАМ (A — проверено экспертом, B — «почти правильное»): без разреза по сорту
    # число описывает неизвестно чью разметку. Файл: путь к nlgx на строку.
    if len(sys.argv) > 1:
        for ln in open(sys.argv[1], encoding="utf-8"):
            ln = ln.strip().strip('"')
            if ln and "_auto" not in os.path.basename(ln).lower():
                files.append((os.path.basename(os.path.dirname(os.path.dirname(ln))), ln))
        print(f"список: {sys.argv[1]}")
    else:
        for well in WELLS:
            files += [(well, f) for f in sorted(glob.glob(os.path.join(ARCHIVE, well, "wlg", "*.nlgx")))
                      if "_auto" not in os.path.basename(f).lower()]
    print(f"планшетов: {len(files)}")
    for k, (well, nlgx) in enumerate(files, 1):
        stem = os.path.basename(nlgx)
        img_path = find_image(nlgx)
        if not img_path:
            print(f"[{k}/{len(files)}] {stem}: нет скана — пропуск")
            continue
        try:
            m = extract(nlgx)
            curves = real_curves(m)
            if not curves:
                print(f"[{k}/{len(files)}] {stem}: нет кривых")
                continue
            gray = np.asarray(Image.open(img_path).convert("L"))
        except Exception as e:
            print(f"[{k}/{len(files)}] {stem}: ERR {e}")
            continue
        for c in curves:
            offs, n, miss, wide = curve_offsets(gray, c)
            if len(offs) < 30:
                continue
            rows.append({
                "well": well, "plate": stem, "curve": c["name"].strip(),
                "n": int(n), "n_meas": int(len(offs)),
                "miss_frac": round(miss / n, 4), "wide_frac": round(wide / n, 4),
                "median_abs": round(float(np.median(np.abs(offs))), 3),
                "p90_abs": round(float(np.percentile(np.abs(offs), 90)), 3),
                "bias": round(float(np.median(offs)), 3),
            })
        if rows:
            last = [r for r in rows if r["plate"] == stem]
            s = " ".join(f"{r['curve'].split()[0]}:{r['median_abs']}" for r in last)
            print(f"[{k}/{len(files)}] {stem}: {s}")

    med = np.array([r["median_abs"] for r in rows])
    p90 = np.array([r["p90_abs"] for r in rows])
    miss = np.array([r["miss_frac"] for r in rows])
    print(f"\n=== ПОТОЛОК ЭКСПЕРТ-vs-ЧЕРНИЛА ({len(rows)} кривых) ===")
    print(f"median|off| по кривым: median={np.median(med):.2f}px  p90={np.percentile(med,90):.2f}px  max={med.max():.2f}px")
    print(f"p90|off| по кривым:    median={np.median(p90):.2f}px  p90={np.percentile(p90,90):.2f}px")
    print(f"miss_frac (мимо чернил ±{SEED}px): median={np.median(miss)*100:.1f}%  p90={np.percentile(miss,90)*100:.1f}%")

    out = {"thr": THR, "seed_px": SEED, "max_run": MAX_RUN, "row_step": ROW_STEP,
           "n_curves": len(rows), "curves": rows,
           "summary": {"median_of_medians_px": float(np.median(med)),
                       "p90_of_medians_px": float(np.percentile(med, 90)),
                       "median_of_p90_px": float(np.median(p90)),
                       "median_miss_frac": float(np.median(miss))}}
    out_path = os.path.join(OUT_DIR, "ink_jitter_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Отчёт: {out_path}")


if __name__ == "__main__":
    main()

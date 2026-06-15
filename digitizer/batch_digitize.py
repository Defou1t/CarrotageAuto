r"""
batch_digitize.py — БАТЧ template-пайплайна по скважине: модель грузится ОДИН раз,
для каждого шаблона nlgx → digitize_one (U-Net→track_multi→inject) + Depth Grid 4 м +
патч скан-пути → `_auto.nlgx`(+bck) в выходной папке + отчёт CSV. Цель — операционализировать
форму+сетку на десятках реальных файлов для QC эксперта и измерить робастность/планку правок.

ЗАПУСК (venv ComfyUI python):
  python batch_digitize.py <ckpt> --well BOGAT_011 [--mnem BKZ,MK,BK] [--limit N]
                           [--out DIR] [--lam 0.15] [--no-grid] [--no-patch]
"""
import sys, time, csv
from pathlib import Path
import numpy as np
import torch
from infer import load_model
from digitize import digitize_one

ARCHIVE = Path(r"F:\nds\projects\Archive")


def match_mnem(stem, mnems):
    if not mnems:
        return True
    return any(f"_{mm}_" in stem or f"_{mm}," in stem for mm in mnems)


def main():
    ckpt = sys.argv[1]; args = sys.argv[2:]
    well = None; mnems = ["BKZ"]; limit = 999; out = None; lam = 0.15
    grid = 4.0; patch = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--well": well = args[i+1]; i += 2
        elif a == "--mnem": mnems = [s.strip() for s in args[i+1].split(",")]; i += 2
        elif a == "--limit": limit = int(args[i+1]); i += 2
        elif a == "--out": out = args[i+1]; i += 2
        elif a == "--lam": lam = float(args[i+1]); i += 2
        elif a == "--no-grid": grid = None; i += 1
        elif a == "--no-patch": patch = False; i += 1
        elif a == "--device": device = args[i+1]; i += 2
        else: i += 1
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if not well:
        print("нужен --well <имя>"); return
    out = Path(out) if out else Path(rf"F:\nds\output\batch\{well}")
    out.mkdir(parents=True, exist_ok=True)
    wlg = ARCHIVE / well / "wlg"
    files = [n for n in sorted(wlg.glob("*.nlgx"))
             if "_auto" not in n.stem and match_mnem(n.stem, mnems)][:limit]

    net, ck = load_model(ckpt, device)
    print(f"model val_dice {ck.get('val_dice'):.3f} device={device}; "
          f"well={well} mnem={mnems} grid4m={grid is not None} patch={patch}; файлов: {len(files)}\n")
    rows = []; t0 = time.time()
    for n in files:
        ts = time.time()
        r = digitize_one(net, device, str(n), out=out, lam=lam,
                         grid_step=grid, patch_scan=patch, verbose=True)
        r["sec"] = round(time.time() - ts, 1)
        rows.append(r)

    okf = [r for r in rows if r["dst"] and not r["error"]]
    errf = [r for r in rows if r["error"]]
    tc = sum(r["curves"] for r in rows); tw = sum(r["written"] for r in rows)
    tv = sum(r["verified"] for r in rows)
    errs = [r["track_err"] for r in okf if r["track_err"] is not None]
    print(f"\n==== ИТОГО ({time.time()-t0:.0f}s) ====")
    print(f"файлов: {len(rows)}; успешно: {len(okf)}; ошибок/пропусков: {len(errf)}")
    print(f"кривых GT: {tc}; вписано трасс: {tw}; верифицировано (механика): {tv}/{tw}")
    if errs:
        print(f"track-ошибка vs эксперт (медиана |dx| на файл): мед={np.median(errs):.1f}px "
              f"макс={np.max(errs):.1f}px; файлов с >8px: {sum(1 for e in errs if e > 8)}")
    if errf:
        print("ОШИБКИ/ПРОПУСКИ:")
        for r in errf:
            print(f"  {r['stem'][:54]}: {r['error']}")
    csvp = out / "batch_report.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "curves", "written", "verified", "track_err", "sec", "error", "dst"])
        for r in rows:
            w.writerow([r["stem"], r["curves"], r["written"], r["verified"],
                        r["track_err"], r.get("sec"), r["error"], r["dst"]])
    print(f"CSV -> {csvp}\nвыход -> {out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Драйвер долгих прогонов: один процесс на задание, без bash-форков (15.09 цикл упал по fork).
Идемпотентен: готовые чекпойнты / дампы пропускает. Запуск: run_jobs.py train | honest"""
import os, re, subprocess, sys, time
from pathlib import Path

PY  = r"D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe"
RND = r"F:/nds/Auto/digitizer/rnd"
OUT = Path(r"F:/nds/output/taskS")
S   = Path(__file__).parent
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

def run(args, log, keep=None):
    """запуск с потоковой записью в лог (строки с UserWarning отбрасываются)"""
    log.write(f"$ {' '.join(args)}\n"); log.flush()
    p = subprocess.Popen([PY] + args, cwd=RND, env=ENV, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    out = []
    for line in p.stdout:
        if "UserWarning" in line or "warnings.warn" in line:
            continue
        out.append(line)
        if keep is None or re.search(keep, line):
            log.write(line); log.flush()
    p.wait()
    log.write(f"[exit {p.returncode}]\n"); log.flush()
    return "".join(out)

def evalfold(ck, thr, tag, f, log):
    if not (OUT / f"{tag}_0of1.pkl").exists():
        run(["_rowdec_eval.py", "--ident", "model", "--ckpt", ck, "--fold", str(f),
             "--peak-thr", str(thr), "--shard", "0/1", "--tag", tag], log, keep="⚠|ДЕРЖАННЫЙ|фолд по")
    else:
        log.write(f"  {tag}: дамп уже есть, пропуск\n")
    s = run(["_rowdec_eval.py", "--sum", "--tag", tag], log, keep="ЧЕСТНЫХ")
    m = re.search(r"ЧЕСТНЫХ: (\d+) из (\d+)", s)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

def sweep(ckdir, prefix, log, note):
    for thr in ("0.6", "0.2"):
        tot = n = 0
        for f in range(5):
            h, N = evalfold(f"{ckdir}/rowdec_of5_f{f}_s0.pt", thr, f"{prefix}{thr}_f{f}", f, log)
            log.write(f"{prefix} thr={thr} фолд {f}: {h} из {N}\n"); log.flush()
            tot += h; n += N
        log.write(f"★ {prefix.upper()} thr={thr} → {tot} из {n}   ({note})\n"); log.flush()

job = sys.argv[1]
if job == "train":
    ckdir = OUT / "rowdec_model" / "bg1"
    with open(S / "bg1.log", "a", encoding="utf-8") as log:
        for f in range(5):
            ck = ckdir / f"rowdec_of5_f{f}_s0.pt"
            if ck.exists():
                log.write(f"=== фолд {f}: чекпойнт уже есть, пропуск ===\n"); continue
            for tries in range(8):                 # партии по 6 ч (правило заказчика 19.09), пока фолд не дообучится
                if ck.exists():
                    break
                log.write(f"=== обучение bg=1 фолд {f}, заход {tries + 1} {time.strftime('%d.%m %H:%M')} (подхват со снимка, если есть; партия <= 6 ч) ===" + chr(10))
                run(["_rowdec_net.py", "--fold", str(f), "--folds", "5", "--epochs", "8", "--ch", "48",
                     "--sigma", "1.5", "--pos-weight", "10", "--lam", "1.0", "--emb", "8", "--seed", "0",
                     "--bg", "1", "--max-hours", "6", "--out", str(ckdir)], log, keep="эпоха|чекпойнт|ПОДХВАТ|ПАРТИЯ|⛔|Error|снимк")
        log.write(f"=== оценка bg1 (честный фолд, --fold-src train) {time.strftime('%d.%m %H:%M')} ===\n")
        sweep(str(ckdir).replace("\\", "/"), "evbg", log, "nobg честно — см. honest.log")
        log.write("=== ГОТОВО ===\n=== DONE ===\n")
elif job == "honest":
    with open(S / "honest.log", "a", encoding="utf-8") as log:
        log.write(f"=== честный пересчёт frozen_nobg {time.strftime('%d.%m %H:%M')} ===\n")
        sweep("F:/nds/output/taskS/rowdec_model/frozen_nobg", "evh", log, "утёкшие: 0.6 → 634, 0.2 → 702 из 1441")
        log.write("=== ГОТОВО ===\n=== DONE ===\n")

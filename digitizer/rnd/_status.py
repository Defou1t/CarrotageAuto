r"""
_status.py — ДАШБОРД статуса v3-пайплайна: что идёт / застряло / сломалось.
Запуск:  python F:\nds\output\_status.py        (разовый снимок)
         python F:\nds\output\_status.py watch  (обновление каждые 15с)
Логи процессов: prep -> prep.log, train -> train.log (в F:\nds\output\unet_v3\).
"""
import os, time, glob, json, re, subprocess, sys

V3 = r"F:\nds\output\unet_v3"
DATA = os.path.join(V3, "data")


def age_min(p):
    return (time.time() - os.path.getmtime(p)) / 60 if os.path.exists(p) else None


def tail(p, n=1):
    if not os.path.exists(p):
        return ""
    try:
        return "".join(open(p, encoding="utf-8", errors="replace").readlines()[-n:]).strip()
    except Exception:
        return ""


def py_procs():
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"],
                             capture_output=True, text=True, creationflags=0x08000000).stdout
        return out.lower().count("python.exe")
    except Exception:
        return "?"


def snapshot():
    L = []
    L.append("=" * 56)
    L.append("  СТАТУС v3-ПАЙПЛАЙНА  " + time.strftime("%H:%M:%S"))
    L.append("=" * 56)
    L.append(f"  python-процессов запущено: {py_procs()}")

    # --- PREP ---
    idx = os.path.join(DATA, "index.json")
    ntr = len(glob.glob(os.path.join(DATA, "train", "*.npz")))
    nva = len(glob.glob(os.path.join(DATA, "val", "*.npz")))
    plog = os.path.join(V3, "prep.log")
    if os.path.exists(idx):
        ix = json.load(open(idx, encoding="utf-8"))
        tiles = sum(r.get("n", 0) for r in ix)
        L.append(f"  [PREP ] ✓ ГОТОВО — {len(ix)} файлов, {tiles} тайлов (npz train {ntr}/val {nva})")
    else:
        prog = tail(plog, 1) or f"npz train {ntr}/val {nva}"
        a = age_min(plog)
        flag = "⚠ ЗАВИС?" if (a is not None and a > 8) else "… идёт"
        L.append(f"  [PREP ] {flag} — {prog}")

    # --- TRAIN ---
    tlog = os.path.join(V3, "train.log")
    model = os.path.join(V3, "unet_v3.pt")
    if os.path.exists(tlog):
        txt = open(tlog, encoding="utf-8", errors="replace").read()
        eps = re.findall(r"ep\s+(\d+)/(\d+)\s+loss=([\d.]+)\s+val_dice=([\d.]+)", txt)
        a = age_min(tlog)
        if "Traceback" in txt and not eps:
            L.append(f"  [TRAIN] ✗ ОШИБКА — {tail(tlog, 2)}")
        elif eps:
            e, tot, ls, vd = eps[-1]
            best = max(float(x[3]) for x in eps)
            flag = "⚠ ЗАВИС?" if (a is not None and a > 12) else "✓ идёт"
            L.append(f"  [TRAIN] {flag} — эпоха {e}/{tot} loss={ls} val_dice={vd} (best {best:.4f}), "
                     f"обновл. {a:.0f} мин назад")
            L.append(f"           (val_dice — внутр. метрика v3, маски шире старых; реальная проверка — оверлей на AK)")
        else:
            L.append(f"  [TRAIN] … стартует — {tail(tlog, 1)}")
    else:
        L.append(f"  [TRAIN] — не запущен (модель unet_v3.pt: {'есть' if os.path.exists(model) else 'нет'})")
    L.append("=" * 56)
    return "\n".join(L)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        try:
            while True:
                os.system("cls")
                print(snapshot()); print("\n(Ctrl+C — выход)")
                time.sleep(15)
        except KeyboardInterrupt:
            pass
    else:
        print(snapshot())


if __name__ == "__main__":
    main()

r"""_seq_band_spec.py — ТЗ ДЛЯ U1: НАСКОЛЬКО ТОЧНУЮ ПОЛОСУ ОБЯЗАН ОТДАТЬ U1.

§6.28.3 оставил ветку с двумя точками и пропастью между ними: на ОРАКУЛЬНОЙ полосе (GT ±20px)
оконный селектор даёт med 15.9px и 4/24 честных, в ПРОДЕ (полоса от U1) — 979.8px и 0/8, разрыв
62×. «Чинить U1» — это лозунг, пока не сказано, ЧТО ИМЕННО он должен отдать.

Здесь полоса портится КОНТРОЛИРУЕМО: кэш стенда пересобирается с pad = 20 / 60 / 150 / 400 px
вокруг размаха экспертной кривой. Получается кривая деградации «точность полосы → результат»,
т.е. численное требование к U1: до какого pad селектор ещё держит идентичность.

★ Сравнивается ВСЕГДА пара (база, селектор) на ОДНОМ И ТОМ ЖЕ кэше — иначе неясно, что именно
портится: трассировщик или сама задача.

  python _seq_band_spec.py --base                 # только база (быстро, без GPU)
  <ComfyUI>\python_embeded\python.exe _seq_band_spec.py --seq --ckpt seq_model_d45p.pt
"""
import sys, argparse, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
import _relatch_bench as BE

PADS = [(20, None),                                  # штатный кэш = оракульная полоса
        (60, r"F:\nds\output\taskS\bench_pad60"),
        (150, r"F:\nds\output\taskS\bench_pad150"),
        (400, r"F:\nds\output\taskS\bench_pad400")]

# ВТОРАЯ ОСЬ: полоса нужной ШИРИНЫ, но стоящая НЕ НА ТОЙ кривой. Замер базы показал, что она
# вреднее ширины (сдвиг 250px: своя 16.9% против 25.6% у полосы вдвое шире).
SHIFTS = [(0, None),
          (100, r"F:\nds\output\taskS\bench_sh100"),
          (250, r"F:\nds\output\taskS\bench_sh250")]


def summary(rows):
    honest = [r for r in rows if r["med"] <= 3 and r["cov"] >= 0.9]
    return (float(np.median([r["med"] for r in rows])), float(np.median([r["cov"] for r in rows])),
            len(honest), len(rows),
            float(np.mean([r["своя"] for r in rows])), float(np.mean([r["латч"] for r in rows])),
            float(np.median([r["band"] for r in rows])))


def line(tag, pad, rows):
    m, c, h, n, sv, lt, bw = summary(rows)
    print(f"{tag:<10}{pad:>6}{bw:>10.0f}{m:>10.1f}{c:>8.2f}{sv:>8.1f}{lt:>8.1f}{h:>6}/{n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--seq", action="store_true")
    ap.add_argument("--ckpt", default="seq_model_d45p.pt")
    ap.add_argument("--point", default="center")
    ap.add_argument("--axis", default="pad", choices=["pad", "shift"],
                    help="что портим: ШИРИНУ полосы или её ПОЛОЖЕНИЕ")
    a = ap.parse_args()

    tracer = None
    if a.seq:
        import torch
        from _decoder_seq import WindowSelector, make_tracer, OUT
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = WindowSelector().to(dev)
        net.load_state_dict(torch.load(OUT / a.ckpt, map_location=dev)["sd"]); net.eval()
        tracer = make_tracer(net, dev, point=a.point)
        print(f"модель {a.ckpt}, точка={a.point}, устройство {dev}")

    axis = SHIFTS if a.axis == "shift" else PADS
    print(f"\n{'':<10}{a.axis:>6}{'полоса':>10}{'med(med)':>10}{'cov':>8}{'своя%':>8}{'латч%':>8}{'честных':>9}")
    for pad, cache in axis:
        if cache and not Path(cache).is_dir():
            print(f"{'':<10}{pad:>6}   кэш не собран — пропуск"); continue
        if a.base or not a.seq:
            line("база", pad, BE.run_strategy(cache=cache))
        if a.seq:
            t0 = time.time()
            line("окно", pad, BE.run_strategy(tracer=tracer, cache=cache))
            print(f"           [{time.time()-t0:.0f}с]")
    print("\nЧитать: pad, на котором честных падает до базовых 3/24 (или med улетает в сотни px),")
    print("и есть ТРЕБОВАНИЕ К U1 — точнее полосу искать незачем, грубее нельзя.")

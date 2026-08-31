r"""_band_diag.py — РАЗБОР ПОЛОСЫ 3-10px (§6.116 → §6.119): почему «почти попали» не попадают.

§6.116 разметил впервые: у 565 кривых (9.6%) лучший кандидат пула ошибается на 3-10px — трасса идёт
ПО СВОЕЙ кривой, но мимо порога честности (med ≤ 3px). Группа сопоставима со всем резервом раскладки
(986), и ей не нужен ни новый вес, ни новый порог — только доведение до 3px. Причина малого промаха
там НЕ разбиралась. Здесь разбирается, тремя замерами в один проход по пулам:

  1. ФОРМА ОШИБКИ — смещение против разброса. Знаковые разности d = tr - gt раскладываются на
     b = median(d) и s = median(|d - b|). Если b велико при малом s — трасса идёт ПАРАЛЛЕЛЬНО, и
     механизм возможен (центрирование, полширины штриха). Если b≈0 при большом s — чинить нечего.
     ⚠ Вычитание b — ОРАКУЛ (b известно только по эталону): это замер ФОРМЫ, а не механизм.
  2. МЕХАНИЗМ — сглаживание скользящим средним по окну W, БЕЗ эталона, поэтому цифра сразу честная.
     Считаются ОБЕ стороны: сколько кривых перешло порог и сколько СЛОМАЛОСЬ из уже честных
     (урок §6.113: резерв, посчитанный по агрегату, обязан проверяться механизмом).
  3. ПОЧЕМУ — высокочастотное содержание СИГНАЛА против ОСТАТКА: median |gt[i+1]-gt[i]| против
     median |r[i+1]-r[i]|, r = tr-gt. Если сигнал ≥ остатка, низкочастотный фильтр обязан резать
     кривую, а не промах, — и тогда провал п.2 не неудача ручки, а свойство задачи.

  <ComfyUI>\python_embeded\python.exe _band_diag.py --shard 0/8      # дампы по шардам
  <ComfyUI>\python_embeded\python.exe _band_diag.py --sum            # сводка по дампам

⚠ Замер идёт по ЛУЧШЕМУ КАНДИДАТУ ПУЛА, а не по отгрузке. §6.115: раскладка берёт половину
доступного, поэтому любой выигрыш отсюда на выдачу переносится ОТДЕЛЬНЫМ прогоном (§6 правило 1).
"""
import sys, pickle, argparse, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import statistics as st
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--shard", default="0/8")
ap.add_argument("--wins", nargs="+", type=int, default=[11, 31, 101, 301])
ap.add_argument("--out", default=r"F:/nds/output/taskS")
ap.add_argument("--sum", action="store_true", help="собрать дампы шардов в таблицы")
a = ap.parse_args()
OUT = Path(a.out)


def movavg(x, w):
    """Скользящее среднее по окну w, края дополнены крайним значением (O(n) через cumsum)."""
    if w <= 1 or len(x) < w:
        return x
    p = w // 2
    xp = np.concatenate([np.full(p, x[0]), x, np.full(w - 1 - p, x[-1])])
    c = np.cumsum(np.insert(xp, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def collect(SH_I, SH_N):
    seen, files = set(), []
    for root in a.pools:
        for f in sorted(Path(root).glob("*.pkl")):
            if f.stem not in seen:
                seen.add(f.stem); files.append(f)
    mine = files[len(files)*SH_I//SH_N:len(files)*(SH_I+1)//SH_N]
    print(f"★ ШАРД {SH_I}/{SH_N}: листов {len(mine)} из {len(files)}, окна {a.wins}")
    rows, done, skip = [], 0, 0
    for f in mine:
        d = pickle.load(open(f, "rb"))
        done += 1
        slots = {s["name"]: s for s in d["slots"]}
        tl = {}
        for i, ln in enumerate(d["lines"]):
            tl.setdefault(ln["track"], []).append(i)
        for nm, gt in d["gts"].items():
            s = slots.get(nm)
            if s is None:
                skip += 1; continue
            best = None
            for i in tl.get(s["track"], []):
                tr = d["lines"][i]["tr"]
                com = sorted(y for y in tr if y in gt)
                if len(com) < 30 or len(com) / max(1, len(gt)) < 0.9:
                    continue
                xt = np.array([tr[y] for y in com], float)
                xg = np.array([gt[y] for y in com], float)
                m = float(np.median(np.abs(xt - xg)))
                if best is None or m < best[0]:
                    best = (m, xt, xg)
            if best is None:
                skip += 1; continue
            m, xt, xg = best
            r = xt - xg
            b = float(np.median(r))
            rec = dict(med0=m, n=len(xt), bias=b,
                       scat=float(np.median(np.abs(r - b))),
                       near=float(np.mean(np.abs(r) <= 3.0)),
                       sig_hf=float(np.median(np.abs(np.diff(xg)))),
                       res_hf=float(np.median(np.abs(np.diff(r)))))
            for w in a.wins:
                rec[f"med{w}"] = float(np.median(np.abs(movavg(xt, w) - xg)))
            rows.append(rec)
    p = OUT / f"band_diag_{SH_I}of{SH_N}.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    print(f"★ СВЕРКА: листов обработано {done} + пропущено 0 = {done} против длины списка "
          f"{len(mine)}   {'★ СОШЛОСЬ' if done == len(mine) else '⛔ НЕ СОШЛОСЬ'}")
    print(f"кривых с покрытием ≥0.9: {len(rows)} (кривых без кандидата/слота: {skip}) → {p}")


def summarise():
    files = sorted(OUT.glob("band_diag_*of*.json"))
    if not files:
        sys.exit("нет дампов — сперва прогнать по шардам")
    den = files[0].stem.split("of")[1]
    files = [f for f in files if f.stem.endswith("of" + den)]
    rows = [r for f in files for r in json.loads(f.read_text(encoding="utf-8"))]
    print(f"шардов {len(files)} из {den}" + ("  ⚠⚠ ПРОГОН НЕПОЛНЫЙ" if len(files) < int(den) else ""))
    print(f"кривых с покрытием ≥0.9: {len(rows)}")
    band = [r for r in rows if 3.0 < r["med0"] <= 10.0]
    hon0 = sum(1 for r in rows if r["med0"] <= 3.0)
    print(f"честных (лучший кандидат ≤3px): {hon0}; полоса 3-10px: {len(band)}; "
          f"медиана ошибки {st.median([r['med0'] for r in rows]):.2f}px\n")

    print("1. ФОРМА ОШИБКИ В ПОЛОСЕ")
    ab = [abs(r["bias"]) for r in band]; sc = [r["scat"] for r in band]
    par = sum(1 for r in band if abs(r["bias"]) > r["scat"])
    fix = sum(1 for r in band if r["scat"] <= 3.0)
    print(f"   |смещение| медиана {st.median(ab):.1f}px | разброс медиана {st.median(sc):.1f}px")
    print(f"   идут ПАРАЛЛЕЛЬНО (|b|>s): {par} из {len(band)} ({100*par/len(band):.0f}%)")
    print(f"   ★ потолок формы (убрать b ПО ЭТАЛОНУ ⇒ ≤3px): {fix} ({100*fix/len(band):.0f}%) — ОРАКУЛ")
    print(f"   доля строк уже в ±3px: медиана {100*st.median([r['near'] for r in band]):.0f}%\n")

    print("2. МЕХАНИЗМ: СГЛАЖИВАНИЕ (без эталона)")
    wins = sorted(int(k[3:]) for k in rows[0] if k.startswith("med") and k != "med0")
    print(f"   {'окно':>6}{'честных':>9}{'Δ':>7}{'взято':>7}{'сломано':>9}{'из полосы':>11}")
    for w in wins:
        k = f"med{w}"
        h = sum(1 for r in rows if r[k] <= 3.0)
        up = sum(1 for r in rows if r["med0"] > 3.0 >= r[k])
        dn = sum(1 for r in rows if r["med0"] <= 3.0 < r[k])
        fb = sum(1 for r in band if r[k] <= 3.0)
        print(f"   {w:>6}{h:>9}{h-hon0:>+7}{up:>7}{dn:>9}{fb:>7} / {len(band)}")

    print("\n3. ПОЧЕМУ: ВЫСОКАЯ ЧАСТОТА СИГНАЛА ПРОТИВ ОСТАТКА")
    grp = {"честные ≤3px": [r for r in rows if r["med0"] <= 3],
           "полоса 3-10px": band,
           "далёкие >10px": [r for r in rows if r["med0"] > 10]}
    print(f"   {'группа':<18}{'кривых':>8}{'сигнал HF':>11}{'остаток HF':>12}{'сигнал/остаток':>16}")
    for g, v in grp.items():
        if not v:
            continue
        s1 = st.median([r["sig_hf"] for r in v]); s2 = st.median([r["res_hf"] for r in v])
        print(f"   {g:<18}{len(v):>8}{s1:>11.2f}{s2:>12.2f}{s1/max(s2,1e-9):>16.2f}")
    if band:
        m = st.median([r["med0"] for r in band]); h = st.median([r["res_hf"] for r in band])
        print(f"\n   ★ остаток в полосе: |r| медиана {m:.1f}px при шаге {h:.2f}px за строку.")
        print(f"     Будь это белый шум, |r| было бы ≈ {h/1.41:.2f}px — в {m/(h/1.41):.0f} раз меньше.")
        print(f"     ⇒ промах — МЕДЛЕННЫЙ УВОД, а не дрожание; фильтр низких частот его ПРОПУСКАЕТ.")


if a.sum:
    summarise()
else:
    i, n = (int(v) for v in a.shard.split("/"))
    collect(i, n)

r"""_shape_probe.py — ОТЛИЧАЕТ ЛИ ФОРМА ВЕРНОЕ ПРОДОЛЖЕНИЕ НА ПЕРЕСЕЧЕНИИ (Задача 7).

ГИПОТЕЗА ЭДУАРДА (20.08): у MGZ/MPZ много зигзагов, у SP своя гладкость — высокочастотное
ПОВЕДЕНИЕ кривой не меняется от того, что она пересеклась с соседкой. Значит там, где положение
уже не различает линии (§6.131: 4.5% строк неразрешимы, §6.134: на этом правило теряет 69.4%),
локальная форма может сказать, какое продолжение чьё.

⚠⚠ МЕРИТЬ ДО ПОСТРОЙКИ ПРИЗНАКА — приём §6.129 (там силу «порядка зондов» померили по эталону
одним проходом и уберегли от переобучения ради половинчатого эффекта). Здесь так же: никакого
обучения, только эталон и арифметика.

КАК. Событие пересечения = строка, где две кривые трека сходятся ближе NEAR px. Вокруг неё берём
окно ДО (`[-W, -GAP]`) и ПОСЛЕ (`[+GAP, +W]`) и считаем по каждой кривой локальную ИЗВИЛИСТОСТЬ —
медиану |второй разности| (мера дрожания, не наклона: прямая наклонная даёт 0) — и среднюю ШИРИНУ
штриха.

⚠⚠ ФОРМА СЧИТАЕТСЯ ПО ЧЕРНИЛАМ, А НЕ ПО ЭТАЛОНУ, И ЭТО НЕ ДЕТАЛЬ. Первая редакция брала вторую
разность прямо от эталонного x — и контроль применимости показал РОВНО НОЛЬ пригодных событий из
1537. Причина: экспертная трасса — это ВЕРШИНЫ полилинии (§6.26: точек 4-14% строк, медианный шаг
6-23 строки), между ними `dense()` кладёт ЛИНЕЙНУЮ интерполяцию, у которой вторая разность
тождественно нулевая. Мерилась расстановка вершин эксперта, а не форма кривой. Теперь положение
берётся из РАНА под эталоном (центр рана в маске полосы) — то есть из того же сигнала, который
будет доступен декодеру на инференсе.
Вопрос ровно один: ближе ли форма «до» к своему продолжению, чем к чужому.

  доля верных = P(|w_до(A) − w_после(A)| + |w_до(B) − w_после(B)|
                 < |w_до(A) − w_после(B)| + |w_до(B) − w_после(A)|)
Случайность здесь — 0.5. Всё, что выше, — сигнал; насколько выше, столько и стоит признак.

⚠⚠ ПРОВЕРКА НА ТОЖДЕСТВО (§6.94, самая коварная ловушка ветки: «оракульный выбор рана L1 ≤ 3px»
оказался тождествен «тушь есть в ±3px», нарушений 0 из 1888, вывод пришлось отзывать). Здесь
тождества быть не должно: `w_до` и `w_после` считаются по РАЗНЫМ строкам, и ничто по построению
не заставляет их совпадать у одной кривой. Контроль печатается: доля событий, где форма «до» у A и
B РАЗЛИЧАЕТСЯ меньше чем на `MINDIFF` — на таких парах признак не применим в принципе, и их надо
исключать из счёта, а не засчитывать как успех.
★ И второй контроль — ПОЛОЖЕНИЕ: та же проверка, но по x вместо формы. Ожидание было «около
случайности», и оно оказалось НЕВЕРНЫМ в поучительную сторону: на смоуке положение даёт 35%, то
есть НИЖЕ случайности. Так и должно быть на НАСТОЯЩЕМ пересечении — кривые меняются местами, и
«продолжение — то, что ближе» систематически указывает на соседа. Это ровно механизм §6.116
(«у половины кривых лучшая трасса идёт по ДРУГОЙ кривой») в чистом виде, и он подтверждает, что
события отобраны как пересечения, а не как случайные строки.
⚠ НИЧЬИ (`same == swap`) НЕ ВЫБРАСЫВАТЬ МОЛЧА: их на смоуке 41%, и если считать долю только по
неничейным, признак выглядит сильнее, чем он есть. Печатается оба числа — строгое (ничьи = 0.5,
как и положено незнанию) и по неничейным.

  <ComfyUI>\python_embeded\python.exe _shape_probe.py --shard 0/4
  <ComfyUI>\python_embeded\python.exe _shape_probe.py --sum
"""
import sys, argparse, json, pickle, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"F:/nds/output/taskS/rowdec")
ap.add_argument("--out", default=r"F:/nds/output/taskS")
ap.add_argument("--tag", default="shapeprobe")
ap.add_argument("--shard", default="0/4")
ap.add_argument("--near", type=float, default=5.0, help="сближение, px — что считаем пересечением")
ap.add_argument("--win", type=int, default=64, help="окно формы, строк")
ap.add_argument("--gap", type=int, default=8, help="отступ от точки пересечения, строк")
ap.add_argument("--thr", type=int, default=90, help="порог темноты над бумагой (§6.133 DELTA=90)")
ap.add_argument("--mindiff", type=float, default=0.2,
                help="минимальная разница форм A и B, ниже которой признак неприменим (px)")
ap.add_argument("--sum", action="store_true")
a = ap.parse_args()
OUT = Path(a.out)


def wig(x):
    """Извилистость: медиана |второй разности|. Наклон не штрафуется, дрожание — да."""
    if len(x) < 8:
        return None
    return float(np.median(np.abs(np.diff(x, 2))))


def family(slot):
    """'GZ41 DA1 SA4' → 'GZ' (семейство по мнемонике, цифры и оси отброшены)."""
    s = re.sub(r"\s+DA\d+\s+SA\d+\s*$", "", slot).strip()
    return re.sub(r"\d+$", "", s) or s


def ink_series(band, ys, xcol, i0, i1, thr):
    """Центры и ширины рана ПОД эталоном на строках [i0,i1) → (centers, widths) с NaN там, где рана
    нет. Это и есть «форма, как её видит картинка»: положение эталона используется только чтобы
    выбрать ран, а сама величина берётся с чернил."""
    import auto.imaging as _im
    H, Wb = band.shape
    cs = np.full(i1 - i0, np.nan, np.float32)
    ws = np.full(i1 - i0, np.nan, np.float32)
    for t, i in enumerate(range(i0, i1)):
        y = int(ys[i])
        if not (0 <= y < H):
            continue
        x = float(xcol[i])
        for x0, x1, c in _im.row_runs(band[y] >= thr):
            if x0 - 3.0 <= x <= x1 + 3.0:
                cs[t] = c; ws[t] = x1 - x0 + 1
                break
    return cs, ws


def run_track(p, slots):
    d = np.load(p)
    ys, xs, band = d["ys"], d["xs"], d["band"]
    K = xs.shape[1]
    if K < 2 or len(ys) < 4 * a.win:
        return []
    # ⚠ строки в выборке идут подряд не всегда (эталон рвётся) — работаем по ИНДЕКСАМ массива,
    # но требуем, чтобы окно было непрерывным по y, иначе «форма до» посчитается через разрыв.
    out = []
    for j in range(K):
        for k in range(j + 1, K):
            dx = np.abs(xs[:, j] - xs[:, k])
            close = np.where(dx <= a.near)[0]
            if not len(close):
                continue
            # события = начала участков сближения (одно пересечение = одно событие)
            ev = [close[0]] + [c for c, prev in zip(close[1:], close[:-1]) if c - prev > a.win]
            for i in ev:
                lo0, lo1 = i - a.win - a.gap, i - a.gap
                hi0, hi1 = i + a.gap, i + a.win + a.gap
                if lo0 < 0 or hi1 >= len(ys):
                    continue
                if ys[lo1] - ys[lo0] != lo1 - lo0 or ys[hi1] - ys[hi0] != hi1 - hi0:
                    continue                       # разрыв строк внутри окна — событие не берём
                ca0, wda0 = ink_series(band, ys, xs[:, j], lo0, lo1, a.thr)
                cb0, wdb0 = ink_series(band, ys, xs[:, k], lo0, lo1, a.thr)
                ca1, wda1 = ink_series(band, ys, xs[:, j], hi0, hi1, a.thr)
                cb1, wdb1 = ink_series(band, ys, xs[:, k], hi0, hi1, a.thr)
                # ⚠ окно с дырявой тушью формы не даёт — такие события не берём (а не считаем нулём)
                if min(np.isfinite(v).mean() for v in (ca0, cb0, ca1, cb1)) < 0.6:
                    continue
                wa0, wb0 = wig(ca0[np.isfinite(ca0)]), wig(cb0[np.isfinite(cb0)])
                wa1, wb1 = wig(ca1[np.isfinite(ca1)]), wig(cb1[np.isfinite(cb1)])
                if None in (wa0, wb0, wa1, wb1):
                    continue
                da0, db0 = np.nanmean(wda0), np.nanmean(wdb0)
                da1, db1 = np.nanmean(wda1), np.nanmean(wdb1)
                wsame = abs(da0 - da1) + abs(db0 - db1)
                wswap = abs(da0 - db1) + abs(db0 - da1)
                same = abs(wa0 - wa1) + abs(wb0 - wb1)
                swap = abs(wa0 - wb1) + abs(wb0 - wa1)
                # контроль ПОЛОЖЕНИЕМ: на пересечении обязан быть около случайности
                xa0, xb0 = xs[lo1 - 1, j], xs[lo1 - 1, k]
                xa1, xb1 = xs[hi0, j], xs[hi0, k]
                psame = abs(xa0 - xa1) + abs(xb0 - xb1)
                pswap = abs(xa0 - xb1) + abs(xb0 - xa1)
                out.append(dict(
                    shape_ok=int(same < swap), shape_tie=int(same == swap),
                    width_ok=int(wsame < wswap), width_tie=int(wsame == wswap),
                    wsep=float(abs(da0 - db0)),
                    pos_ok=int(psame < pswap), pos_tie=int(psame == pswap),
                    sep=float(abs(wa0 - wb0)),           # насколько формы РАЗНЫЕ до события
                    fam_same=int(family(slots[j]) == family(slots[k])),
                    wa=wa0, wb=wb0))
    return out


def collect(i, n):
    man = []
    for f in sorted(Path(a.data).glob("manifest_*of*.json")):
        man += json.loads(f.read_text(encoding="utf-8"))
    man = [m for m in man if m["K"] >= 2]
    mine = man[len(man) * i // n:len(man) * (i + 1) // n]
    print(f"★ ШАРД {i}/{n}: треков {len(mine)} из {len(man)} (K≥2)")
    rows = []
    for c, m in enumerate(mine, 1):
        try:
            rows += run_track(Path(a.data) / m["file"], m["slots"])
        except Exception as e:
            print(f"  ⚠ {m['file'][:40]}: {type(e).__name__}: {str(e)[:50]}")
        if c % 20 == 0 or c == len(mine):
            print(f"  {c}/{len(mine)}  событий {len(rows)}")
    p = OUT / f"{a.tag}_{i}of{n}.pkl"
    pickle.dump(dict(rows=rows, tracks=len(mine)), open(p, "wb"))
    print(f"★ готово: событий {len(rows)} → {p}")


def summarise():
    fs = sorted(OUT.glob(f"{a.tag}_*of*.pkl"))
    if not fs:
        sys.exit("нет дампов")
    den = fs[0].stem.split("of")[1]
    fs = [f for f in fs if f.stem.endswith("of" + den)]
    rows, tracks = [], 0
    for f in fs:
        d = pickle.load(open(f, "rb"))
        rows += d["rows"]; tracks += d["tracks"]
    print(f"шардов {len(fs)} из {den}" + ("  ⚠⚠ НЕПОЛНЫЙ" if len(fs) < int(den) else ""))
    print(f"треков {tracks}, событий сближения (<{a.near:.0f}px) {len(rows)}")
    if not rows:
        return

    def rate(sel, key):
        """СТРОГИЙ счёт: ничья = 0.5 (незнание), и она в знаменателе. Возвращает также долю
        по неничейным — разница между двумя числами и есть цена ничьих."""
        if not sel:
            return 0.0, 0, 0.0
        strict = sum(0.5 if r[key + "_tie"] else r[key + "_ok"] for r in sel) / len(sel)
        nz = [r for r in sel if not r[key + "_tie"]]
        loose = sum(r[key + "_ok"] for r in nz) / len(nz) if nz else 0.0
        return 100 * strict, len(sel), 100 * loose

    приг = [r for r in rows if r["sep"] >= a.mindiff]
    print(f"\n⚠ КОНТРОЛЬ ПРИМЕНИМОСТИ: формы A и B различаются ≥{a.mindiff}px "
          f"у {len(приг)} событий ({100*len(приг)/len(rows):.0f}%); "
          f"на остальных признак неприменим В ПРИНЦИПЕ и в счёт не идёт.")
    sh, nsh, shl = rate(приг, "shape"); po, npo, pol = rate(приг, "pos")
    wd, nwd, wdl = rate([r for r in rows if r["wsep"] >= 0.5], "width")
    print(f"\n★ ФОРМА (извилистость): {sh:.1f}% строго (ничьи = 0.5), {shl:.1f}% по неничейным "
          f"(n={nsh}, случайность 50%)")
    print(f"★ ШИРИНА ШТРИХА (разделение ≥0.5px): {wd:.1f}% строго, {wdl:.1f}% по неничейным (n={nwd})")
    print(f"⚠ КОНТРОЛЬ ПОЛОЖЕНИЕМ: {po:.1f}% строго (n={npo}). НИЖЕ 50% — верно и ожидаемо: "
          f"на пересечении кривые меняются местами, и «ближе» указывает на соседа (§6.116).")

    print(f"\n{'разделение форм (px)':<24}{'событий':>9}{'ФОРМА, %':>11}{'положение, %':>14}")
    qs = [(a.mindiff, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 1e9)]
    for lo, hi in qs:
        sel = [r for r in rows if lo <= r["sep"] < hi]
        if len(sel) < 20:
            continue
        s, ns, _ = rate(sel, "shape"); p, _, _ = rate(sel, "pos")
        print(f"{f'{lo:g} … {hi:g}':<24}{ns:>9}{s:>10.1f}%{p:>13.1f}%")

    print(f"\n{'пара':<24}{'событий':>9}{'ФОРМА, %':>11}{'положение, %':>14}")
    for lab, sel in (("★ ОДНО семейство", [r for r in приг if r["fam_same"]]),
                     ("разные семейства", [r for r in приг if not r["fam_same"]])):
        if not sel:
            continue
        s, ns, _ = rate(sel, "shape"); p, _, _ = rate(sel, "pos")
        print(f"{lab:<24}{ns:>9}{s:>10.1f}%{p:>13.1f}%")
    print("\n⇒ на однородном пучке (одно семейство) сигнал ожидается слабее — свойства там одинаковы;")
    print("  именно это разделение и решает, годится ли признак как ОБЩИЙ или только как частный.")


if a.sum:
    summarise()
else:
    i, n = (int(v) for v in a.shard.split("/"))
    collect(i, n)

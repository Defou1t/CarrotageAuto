r"""_rowdec_base.py — КОНТРОЛЬ БЕЗ ОБУЧЕНИЯ НА ПОСТРОЧНОЙ ВЫБОРКЕ (Задача 9, шаг 2).

⚠⚠ ЗАЧЕМ. §6.78 стоил ветке отозванного вывода: «объяснил 91% ошибки масштабом», а тупая константа
объясняла 89%. У любой подгонки обязан быть контроль. Здесь контроль такой: СКОЛЬКО ПОТОЛКА §6.133
берёт правило БЕЗ ОБУЧЕНИЯ, работающее ровно в той же постановке — построчно, без состояния.

ПРАВИЛО-КОНТРОЛЬ: на строке взять раны полосы, отобрать K самых широких, отсортировать по x и
раздать K траекториям по порядку. Ни истории, ни модели — только порядок на строке.
★ Именно поэтому он и интересен: на пересечении порядок МЕНЯЕТСЯ МЕСТАМИ, и правило обязано там
терять личность. Разрыв «правило → потолок» и есть цена того, чему предстоит учиться.

СЧЁТ — БЕЗЫМЯННЫЙ (решение Эдуарда 20.08): траектории матчатся к эталонным кривым трека лучшим 1:1,
подпись не требуется. Кривая честная, если med|Δx| ≤ 3px И cov ≥ 0.9 — та же метрика ветки.
⚠ Сравнивать это надо с потолком §6.133, который тоже безымянный (V2 50.1%, V3ц 58.0% корпуса), а
НЕ с 989/14.4% прода: то число считалось с именем и по другому корпусу.

  <ComfyUI>\python_embeded\python.exe _rowdec_base.py --shard 0/4
  <ComfyUI>\python_embeded\python.exe _rowdec_base.py --sum
"""
import sys, argparse, json, pickle
from itertools import permutations
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict
import numpy as np
from auto import imaging as im

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"F:/nds/output/taskS/rowdec")
ap.add_argument("--out", default=r"F:/nds/output/taskS")
ap.add_argument("--tag", default="rowbase")
ap.add_argument("--shard", default="0/4")
ap.add_argument("--thr", type=int, default=90, help="порог темноты над бумагой (§6.133 DELTA=90)")
ap.add_argument("--max-rows", type=int, default=20000, help="строк на трек в замере (0 = все)")
ap.add_argument("--sum", action="store_true")
a = ap.parse_args()
OUT = Path(a.out)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def match_1to1(err):
    """Лучшее 1:1 «наша траектория → эталонная кривая» по сумме медиан. K мал (в корпусе ≤12):
    до 6 — перебор перестановок, дальше жадно. ⚠ Жадность на больших K может занижать контроль —
    это в его пользу как контроля, но в отчёте оговаривается."""
    K = err.shape[0]
    if K <= 6:
        best, bp = None, None
        for p in permutations(range(K)):
            s = sum(err[i, p[i]] for i in range(K))
            if best is None or s < best:
                best, bp = s, p
        return list(bp)
    order = np.dstack(np.unravel_index(np.argsort(err, axis=None), err.shape))[0]
    used_r, used_c, out = set(), set(), [None] * K
    for r, c in order:
        if r in used_r or c in used_c:
            continue
        used_r.add(int(r)); used_c.add(int(c)); out[int(r)] = int(c)
    return [c if c is not None else 0 for c in out]


def run_track(p):
    d = np.load(p)
    band, ys, xs = d["band"], d["ys"], d["xs"]
    K = xs.shape[1]
    if a.max_rows and len(ys) > a.max_rows:                 # равномерное прореживание, не блоком
        idx = np.linspace(0, len(ys) - 1, a.max_rows).astype(int)
        ys, xs = ys[idx], xs[idx]
    H, Wb = band.shape
    ours = np.full((len(ys), K), np.nan, np.float32)
    for i, y in enumerate(ys):
        if not (0 <= y < H):
            continue
        runs = im.row_runs(band[y] >= a.thr)
        if not runs:
            continue
        if len(runs) > K:                                   # K самых ШИРОКИХ — правило без истории
            runs = sorted(sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:K],
                          key=lambda r: r[2])
        cs = [r[2] for r in runs]
        for j in range(min(K, len(cs))):
            ours[i, j] = cs[j]
    # ── ошибка «наша траектория j → эталонная кривая k» ────────────────────────────────────
    err = np.full((K, K), 1e9, np.float32)
    for j in range(K):
        ok = ~np.isnan(ours[:, j])
        if ok.sum() < 30:
            continue
        for k in range(K):
            err[j, k] = float(np.median(np.abs(ours[ok, j] - xs[ok, k])))
    perm = match_1to1(err)
    out = []
    for j in range(K):
        k = perm[j]
        ok = ~np.isnan(ours[:, j])
        cov = float(ok.sum()) / max(1, len(ys))
        med = float(np.median(np.abs(ours[ok, j] - xs[ok, k]))) if ok.sum() >= 30 else None
        # ── КОНТРОЛЬ B: то же правило + ТОТ ЖЕ МОСТИК, что у потолка §6.136 ─────────────────
        # ⚠ Без него выигрыш прототипа неатрибутируем: мостик сам по себе поднял потолок с 58.0%
        # до 82.2%, и сравнение обученной модели с ГОЛЫМ правилом (18.2%) приписало бы мостику
        # заслугу личности. Это ровно форма §6.78 (подгонка без контроля).
        idx = np.where(ok)[0]
        if len(idx) >= 2:
            allr = np.arange(len(ys))
            iv = np.interp(allr, idx, ours[idx, j])
            inside = (allr >= idx[0]) & (allr <= idx[-1])   # за краями interp держит константу
            e = np.abs(iv[inside] - xs[inside, k])
            medb = float(np.median(e)) if len(e) >= 30 else None
            covb = float(inside.sum()) / max(1, len(ys))
            gapb = int(np.max(np.diff(idx)))
        else:
            medb, covb, gapb = None, 0.0, 0
        out.append((K, med, cov, medb, covb, gapb))
    return out


def collect(i, n):
    man = []
    for f in sorted(Path(a.data).glob("manifest_*of*.json")):
        man += json.loads(f.read_text(encoding="utf-8"))
    files = sorted({m["file"] for m in man})
    mine = files[len(files) * i // n:len(files) * (i + 1) // n]
    print(f"★ ШАРД {i}/{n}: треков {len(mine)} из {len(files)}")
    rows, bad = [], defaultdict(int)
    for k, fn in enumerate(mine, 1):
        try:
            rows += run_track(Path(a.data) / fn)
        except Exception as e:
            bad[f"{type(e).__name__}: {e}"[:60]] += 1
        if k % 20 == 0 or k == len(mine):
            print(f"  {k}/{len(mine)}  кривых {len(rows)}")
    p = OUT / f"{a.tag}_{i}of{n}.pkl"
    pickle.dump(dict(rows=rows, tracks=len(mine), bad=dict(bad)), open(p, "wb"))
    print(f"★ готово: кривых {len(rows)}, отказов {sum(bad.values())} → {p}")
    for s, c in bad.items():
        print(f"    ⚠ {s}: {c}")


def summarise():
    fs = sorted(OUT.glob(f"{a.tag}_*of*.pkl"))
    if not fs:
        sys.exit("нет дампов")
    den = fs[0].stem.split("of")[1]
    fs = [f for f in fs if f.stem.endswith("of" + den)]
    rows, tracks, bad = [], 0, defaultdict(int)
    for f in fs:
        d = pickle.load(open(f, "rb"))
        rows += d["rows"]; tracks += d["tracks"]
        for s, c in d["bad"].items():
            bad[s] += c
    print(f"шардов {len(fs)} из {den}" + ("  ⚠⚠ НЕПОЛНЫЙ" if len(fs) < int(den) else ""))
    N = max(1, len(rows))
    hon = sum(1 for K, m, c, mb, cb, g in rows if HON(m, c))
    honb = sum(1 for K, m, c, mb, cb, g in rows if HON(mb, cb))
    print(f"треков {tracks}, кривых {len(rows)}, отказов {sum(bad.values())}")
    print(f"\n★ КОНТРОЛЬ A (правило без мостика): {hon} из {len(rows)} = {100*hon/N:.1f}%")
    print(f"★★ КОНТРОЛЬ B (правило + МОСТИК):   {honb} из {len(rows)} = {100*honb/N:.1f}%")
    print(f"   ⇒ мостик сам по себе даёт {100*(honb-hon)/N:+.1f} пункта — это НЕ заслуга личности;")
    print(f"     прототип обязан сравниваться с B, иначе выигрыш неатрибутируем (§6.78).")
    nocov = sum(1 for K, m, c, mb, cb, g in rows if c < 0.9)
    badmed = sum(1 for K, m, c, mb, cb, g in rows if c >= 0.9 and not HON(m, c))
    print(f"   не берёт по покрытию: {nocov} ({100*nocov/N:.1f}%), по ошибке: {badmed} ({100*badmed/N:.1f}%)")
    print(f"\n{'K':<6}{'кривых':>9}{'A честных':>10}{'доля':>8}{'B честных':>10}{'доля':>8}")
    agg = defaultdict(lambda: [0, 0, 0])
    for K, m, c, mb, cb, g in rows:
        b = agg[min(K, 5)]; b[0] += 1; b[1] += int(HON(m, c)); b[2] += int(HON(mb, cb))
    for k in sorted(agg):
        v = agg[k]
        print(f"{('%d' % k) if k < 5 else '5+':<6}{v[0]:>9}{v[1]:>10}{100*v[1]/v[0]:>7.1f}%"
              f"{v[2]:>10}{100*v[2]/v[0]:>7.1f}%")
    print(f"\nдля сравнения — потолок §6.133 в том же безымянном счёте: V2 50.1%, V3ц 58.0% корпуса;")
    print(f"прод на отгрузке без имени — 30.6%, с правками §6.132 — 35.3% (1130 листов).")


if a.sum:
    summarise()
else:
    i, n = (int(v) for v in a.shard.split("/"))
    collect(i, n)

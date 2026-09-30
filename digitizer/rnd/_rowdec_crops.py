r"""_rowdec_crops.py — КЭШ КРОПОВ ДЛЯ ОБУЧЕНИЯ ПОСТРОЧНОГО ДЕКОДЕРА (Задача 9, шаг 3).

⚠⚠ ЗАЧЕМ ОТДЕЛЬНЫЙ КЭШ. Выборка `_rowdec_data` лежит в `savez_compressed`, и обращение к `band`
РАЗЖИМАЕТ ВСЮ ПОЛОСУ трека (у крупных лент это сотни МБ на один сэмпл). Читать её на каждом шаге
обучения нельзя — узкое место будет не в сети, а в zlib. Здесь один проход по выборке нарезает
кропы фиксированного размера в ПЛОСКИЙ memmap, который дальше читается случайным доступом даром.

ЧТО В КРОПЕ: ROWS×COLS uint8 полосы (насколько темнее бумаги СВОЕЙ строки — §6.133; не бинарь,
иначе бледная тушь теряется и потолок падает с 58% до 50% ещё до обучения) плюс таргеты: x каждой
из K кривых на каждой строке кропа, в координатах кропа, и маска валидности.

ОТБОР КРОПОВ — НЕ РАВНОМЕРНЫЙ, И ЭТО ВАЖНО:
  • половина центрируется на СЛУЧАЙНОЙ ПАРЕ кривых (на широкой полосе равномерный кроп часто
    пустой: медианная полоса ~1200px при K=3, и push-член лосса остаётся без пары);
  • не меньше четверти — на СОБЫТИЯХ СБЛИЖЕНИЯ (<5px): §6.135 намерил, что именно там теряется
    личность, а §6.134 — что личность стоит +36.5 пункта против +5.4 у переднего плана.
⚠ Смещение выборки к трудным местам ОБЯЗАНО быть записано: доля событийных кропов печатается и
кладётся в манифест, иначе обученная модель будет мериться на распределении, которого не видела.

  <ComfyUI>\python_embeded\python.exe _rowdec_crops.py --shard 0/4 --per-track 200
"""
import sys, argparse, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"F:/nds/output/taskS/rowdec")
ap.add_argument("--out", default=r"F:/nds/output/taskS/rowdec_crops")
ap.add_argument("--shard", default="0/4")
ap.add_argument("--rows", type=int, default=128)
ap.add_argument("--cols", type=int, default=512)
ap.add_argument("--maxk", type=int, default=6, help="кривых на кроп; трек с большим K берётся частями")
ap.add_argument("--per-track", type=int, default=200)
# ★ 30.09 (§6.249): квота кропов ПО K трека, "1:26,2:35,3:58,4:32,5:57,6+:44" — состав как у замороженного набора. Разбор
#   §6.248: при единой квоте 40 весь пуловый корпус сместил бюджет к K = 1 (кропов с ≥3 кривыми −39%), и личность
#   выросла, а обнаружение на плотных кропах просело. Пусто = единая `--per-track` (прежнее поведение побайтно).
ap.add_argument("--per-track-by-k", default="")
ap.add_argument("--near", type=float, default=5.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
R, C, MK = a.rows, a.cols, a.maxk


_PTK = {}
for _kv in filter(None, a.per_track_by_k.split(",")):
    _k, _v = _kv.split(":")
    _PTK[_k.strip()] = int(_v)


def per_track_of(K):
    """квота кропов трека с K кривыми: `--per-track-by-k` («6+» — K ≥ 6) или единая `--per-track`"""
    if not _PTK:
        return a.per_track
    if str(K) in _PTK:
        return _PTK[str(K)]
    plus = [(int(k[:-1]), v) for k, v in _PTK.items() if k.endswith("+") and K >= int(k[:-1])]
    if not plus:
        sys.exit(f"--per-track-by-k: нет квоты для K = {K}")
    return max(plus)[1]


def crops_of(f, rng):
    d = np.load(Path(a.data) / f)
    band, ys, xs = d["band"], d["ys"], d["xs"]
    H, Wb = band.shape
    n, K = xs.shape
    if n < R + 8:
        return None
    # ⚠ строки таргета идут не подряд (эталон рвётся) — кроп берём только там, где окно СПЛОШНОЕ
    ok_start = np.zeros(n - R, bool)
    dy = np.diff(ys.astype(np.int64))
    run = np.concatenate([[0], np.cumsum(dy != 1)])
    ok_start[:] = run[R:] == run[:n - R]

    ev = np.zeros(n, bool)
    for j in range(K):
        for k in range(j + 1, K):
            ev |= np.abs(xs[:, j] - xs[:, k]) <= a.near
    ev_idx = np.where(ev[:n - R] & ok_start)[0]
    all_idx = np.where(ok_start)[0]
    if not len(all_idx):
        return None

    pt = per_track_of(K)
    want_ev = min(len(ev_idx), pt // 4)
    picks = list(rng.choice(ev_idx, want_ev, replace=False)) if want_ev else []
    rest = pt - len(picks)
    picks += list(rng.choice(all_idx, min(rest, len(all_idx)), replace=False))

    X = np.zeros((len(picks), R, C), np.uint8)
    Y = np.full((len(picks), R, MK), -1.0, np.float32)
    for t, i0 in enumerate(picks):
        i0 = int(i0)
        sl = slice(i0, i0 + R)
        y0 = int(ys[i0])
        # центр кропа: на паре кривых (иначе на широкой полосе кроп часто пуст)
        ks = rng.permutation(K)[:2]
        cx = float(np.mean([xs[i0 + R // 2, k] for k in ks]))
        x0 = int(np.clip(round(cx - C / 2), 0, max(0, Wb - C)))
        w = min(C, Wb - x0)
        X[t, :, :w] = band[y0:y0 + R, x0:x0 + w]
        rel = xs[sl, :] - x0
        # ⚠ ПОРОГ 0.5 ВЫБРАСЫВАЛ КРИВУЮ ЦЕЛИКОМ, ОСТАВЛЯЯ ЕЁ ТУШЬ В ВХОДЕ: у 38.7% кропов хотя бы
        # одна кривая трека терялась так, и её пиксели уходили в лосс как ОТРИЦАТЕЛЬНЫЕ. Сеть учили
        # тому, что настоящая кривая — это фон. Теперь кривая остаётся, а вне окна стоит -1.
        keep = [k for k in range(K) if np.mean((rel[:, k] >= 0) & (rel[:, k] < C)) > 0.05]
        for s, k in enumerate(keep[:MK]):
            v = rel[:, k]
            Y[t, :, s] = np.where((v >= 0) & (v < C), v, -1.0)
    return X, Y, len(picks), want_ev


def main(i, n):
    man = []
    for f in sorted(Path(a.data).glob("manifest_*of*.json")):
        man += json.loads(f.read_text(encoding="utf-8"))
    files = sorted({m["file"] for m in man})
    meta = {m["file"]: m for m in man}
    mine = files[len(files) * i // n:len(files) * (i + 1) // n]
    rng = np.random.default_rng(a.seed + i)
    print(f"★ ШАРД {i}/{n}: треков {len(mine)}, кроп {R}×{C}, до "
          f"{a.per_track_by_k + ' по K' if a.per_track_by_k else a.per_track} на трек")

    # ★ 29.09 (§6.248): кропы пишутся потоком в сырые файлы шарда, а `.npy` собирается из них в конце через
    #   open_memmap. Прежняя редакция держала список кропов И их склейку (`np.concatenate`) в памяти — вдвое больше
    #   размера шарда (28 тыс. кропов 256×512 ≈ 7.4 ГБ пика на шард, 4 шарда разом ≈ 30 ГБ). Байты выхода те же
    #   (заголовок open_memmap = заголовок np.save), сверено со старой редакцией на одних треках.
    xraw, yraw = OUT / f"x_{i}of{n}.raw.tmp", OUT / f"y_{i}of{n}.raw.tmp"
    rows_man, ev_tot, tot = [], 0, 0
    with open(xraw, "wb") as fx, open(yraw, "wb") as fy:
        for c, f in enumerate(mine, 1):
            try:
                r = crops_of(f, rng)
            except Exception as e:
                print(f"  ⚠ {f[:44]}: {type(e).__name__}: {str(e)[:50]}"); continue
            if r is None:
                continue
            X, Y, cnt, nev = r
            fx.write(np.ascontiguousarray(X).tobytes()); fy.write(np.ascontiguousarray(Y).tobytes())
            tot += X.shape[0]; ev_tot += nev
            m = meta[f]
            rows_man.append(dict(file=f, well=m["well"], sheet=m["sheet"], K=m["K"], crops=cnt))
            if c % 20 == 0 or c == len(mine):
                print(f"  {c}/{len(mine)}  кропов {tot:,}")
    if not tot:
        sys.exit("нет кропов")
    for raw, name, dt, tail in ((xraw, f"x_{i}of{n}.npy", np.uint8, (R, C)), (yraw, f"y_{i}of{n}.npy", np.float32, (R, MK))):
        # ⚠ БЕЗ сжатия: кэш читается случайным доступом
        mm = np.lib.format.open_memmap(OUT / name, mode="w+", dtype=dt, shape=(tot,) + tail)
        per = int(np.prod(tail)) * np.dtype(dt).itemsize
        with open(raw, "rb") as fh:
            for s in range(0, tot, 1024):
                k = min(1024, tot - s)
                mm[s:s + k] = np.frombuffer(fh.read(per * k), dt).reshape((k,) + tail)
        mm.flush(); del mm
        raw.unlink()
    (OUT / f"man_{i}of{n}.json").write_text(json.dumps(dict(
        rows=R, cols=C, maxk=MK, crops=int(tot), event_crops=int(ev_tot),
        tracks=rows_man), ensure_ascii=False), encoding="utf-8")
    print(f"★ готово: кропов {tot:,} ({tot * R * C / 2**30:.2f} ГБ), "
          f"из них на событиях сближения {ev_tot:,} ({100*ev_tot/max(1,tot):.0f}%)")


i, n = (int(v) for v in a.shard.split("/"))
main(i, n)

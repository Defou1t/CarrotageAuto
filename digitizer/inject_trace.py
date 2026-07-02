r"""
inject_trace.py — capstone: вписать трассы НАШЕГО трекера в nlgx → сдаваемый _auto.nlgx.
Берёт кэш трекера (cache_tracks.py) + исходный nlgx (шаблон калибровки/сегментов),
сопоставляет линии трекера с кривыми (по близости к GT), заменяет тег 35490 (X по
строкам) каждой кривой на трассу трекера, пишет _auto.nlgx (+bck). Калибровку/уровни
берём из шаблона (трекер даёт форму; значения/перевыносы — отдельная задача).
Результат открывается в NeuraLOG для QC поверх скана (шлюз сдачи подтверждён).

python inject_trace.py <cache.pkl> [--out DIR]
"""
import sys, pickle, glob, struct
from pathlib import Path
import numpy as np
from write_nlgx import read_full, write_full, set_tag, find_ifd
from extract_nlgx import extract, NULL
import dataset as ds

ARCHIVE = Path(r"F:\nds\projects\Archive")


def find_nlgx(stem):
    hits = glob.glob(str(ARCHIVE / "*" / "wlg" / f"{stem}.nlgx"))
    return hits[0] if hits else None


def main():
    pkl = sys.argv[1]
    out = Path(r"F:\nds\output")
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out")+1])
    sys.stdout.reconfigure(encoding="utf-8")
    rec = pickle.load(open(pkl, "rb"))
    stem = rec["stem"]
    src = find_nlgx(stem)
    if not src:
        print(f"не найден nlgx для {stem}"); return
    print(f"шаблон: {src}")

    m = extract(src)
    curves = ds.real_curves(m)
    lines = [dict(tr) for tr in rec["lines"]]   # tracker lines: {row: x}
    gts = dict(rec["gts"])                       # name -> {row:x}

    # сопоставить каждую кривую с лучшей линией трекера (по близости к её GT)
    def best_line(gd):
        best = (1e9, None)
        for li, L in enumerate(lines):
            common = [(L[y]) for y in gd if y in L]
            if len(common) < 15:
                continue
            err = np.median([abs(L[y]-gd[y]) for y in gd if y in L])
            if err < best[0]:
                best = (err, li)
        return best

    ifds = read_full(open(src, "rb").read())
    written = []
    for c in curves:
        short = c["name"].split()[0]
        gd = gts.get(short)
        if not gd:
            continue
        err, li = best_line(gd)
        if li is None:
            continue
        L = lines[li]
        top_y = c["top_y"]; n = len(c["xs"])
        new_xs = [int(round(L[top_y+i])) if (top_y+i) in L else NULL for i in range(n)]
        # найти IFD этой кривой (type7, 35470 начинается с short)
        def is_this(tags, short=short):
            if 34768 not in tags or struct.unpack("<I", tags[34768][2][:4])[0] != 7:
                return False
            if 35470 not in tags:
                return False
            nm = tags[35470][2].split(b"\x00")[0].decode("latin1")
            return nm.startswith(short + " ") or nm == short
        idxs = find_ifd(ifds, is_this)
        if not idxs:
            continue
        set_tag(ifds, idxs[0], 35490, 4, new_xs)
        written.append((short, err, sum(1 for x in new_xs if x != NULL)))

    data = write_full(ifds)
    dst = out / f"{stem}_auto.nlgx"
    open(dst, "wb").write(data)
    # bck = точная копия nlgx: write_bck(off=5489) патчил байт ВНУТРИ данных тега 35490
    # (портил трассу молча); реальный .bck NeuraLOG — предыдущее сохранение (анализ 02.07)
    open(out / f"{stem}_auto.bck", "wb").write(data)
    print(f"\nвписано трасс трекера: {len(written)}")
    for short, err, npts in written:
        print(f"  {short:<9} (track err≈{err:.1f}px, {npts} точек)")
    # верификация: переоткрыть и сверить, что xs соответствуют трекеру
    m2 = extract(str(dst))
    ok = 0
    for c2 in ds.real_curves(m2):
        short = c2["name"].split()[0]
        gd = gts.get(short)
        if not gd: continue
        err, li = best_line(gd)
        if li is None: continue
        L = lines[li]
        common = [(c2["xs"][y - c2["top_y"]], L[y]) for y in gd
                  if y in L and 0 <= y-c2["top_y"] < len(c2["xs"]) and c2["xs"][y-c2["top_y"]] != NULL]
        if common and np.median([abs(a-b) for a, b in common]) < 1.5:
            ok += 1
    print(f"верификация: {ok}/{len(written)} кривых в _auto.nlgx соответствуют трассе трекера")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()

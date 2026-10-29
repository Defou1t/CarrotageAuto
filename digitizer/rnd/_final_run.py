r"""_final_run.py — ИТОГОВАЯ АВТОМАТИЧЕСКАЯ ОЦИФРОВКА АРХИВА ПРОДОМ (29.10).

Каждый лист архива (`--roots`/*/wlg/*.nlgx, скан — `dataset_build.find_image`) — `auto.pipeline.run` с конфигом по умолчанию (прод),
рамка — nlgx листа, + .las. Выдача — `--out/<лист: sheet_dir>/` (раскладка как у повтора `_trace_cache.py`, счёт —
`_name_cost_prod.py --dir <out/..> --mode <имя>`). Уже сделанные листы (есть `*_auto.nlgx`) пропускаются; падение листа — в журнал,
прогон идёт дальше. Журнал — `--out/_log_<шард>.tsv`: лист, секунд, статус, файлов.

  _final_run.py --out F:/nds/output/final_2026-10/FINAL --shard 0/3 --max-hours 5.5
"""
import sys, argparse, hashlib, time, io, contextlib, traceback
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path


def sheet_dir(n):
    return f"{n.stem[:40]}_{hashlib.md5(n.stem.encode('utf-8')).hexdigest()[:8]}"


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
    ap.add_argument("--sheets", default="", help="файл со списком листов (имена .nlgx); по умолчанию — все")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-hours", type=float, default=5.5)
    ap.add_argument("--no-las", action="store_true")
    a = ap.parse_args()
    from dataset_build import find_image
    from auto.config import Config
    from auto import pipeline
    si, sn = map(int, a.shard.split("/"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    src = sorted({q for r in a.roots for q in Path(r).glob("*/wlg/*.nlgx")}, key=lambda q: q.name)
    if a.sheets:
        want = {l.strip() for l in Path(a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()}
        src = [q for q in src if q.name in want]
    src = src[si::sn]
    log = out / f"_log_{si}.tsv"
    t0 = time.time(); done = skip = fail = noimg = 0
    for i, n in enumerate(src):
        if time.time() - t0 > a.max_hours * 3600:
            print(f"★ лимит {a.max_hours} ч — стоп на {i}/{len(src)}"); break
        d = out / sheet_dir(n)
        if d.is_dir() and any(d.glob("*_auto.nlgx")):
            skip += 1; continue
        img = find_image(n)
        if not img:
            noimg += 1
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"{n.name}\t0\tнет скана\t0\n")
            continue
        cfg = Config(); cfg.out = d; d.mkdir(parents=True, exist_ok=True)
        t = time.time(); st = "ок"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    pipeline.run(str(img), frame_nlgx=str(n), cfg=cfg, las=not a.no_las)
                except Exception:
                    if a.no_las:
                        raise
                    st = "ок без las"
                    pipeline.run(str(img), frame_nlgx=str(n), cfg=cfg, las=False)
            done += 1
        except Exception as e:
            fail += 1; st = f"ПАДЕНИЕ {type(e).__name__}: {str(e)[:150]}"
            (d / "_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"{n.name}\t{time.time() - t:.0f}\t{st}\t{len(list(d.glob('*')))}\n")
        if (done + fail) % 20 == 0:
            print(f"  {i + 1}/{len(src)}: сделано {done}, падений {fail}, без скана {noimg}, было {skip}, {time.time() - t0:.0f} с")
    print(f"★ шард {a.shard}: сделано {done}, падений {fail}, без скана {noimg}, было {skip} из {len(src)}")


if __name__ == "__main__":
    main()

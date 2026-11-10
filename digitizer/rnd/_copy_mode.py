r"""_copy_mode.py — ПЕРЕНОС ГОТОВЫХ ЛИСТОВ СТЕНДА В ПОЛНЫЙ ПОВТОР (10.11, §6.359): каталоги листов `rp_lvl/<src>` с меткой
`_replay_ok.txt` копируются в `rp_lvl/<dst>`, если там листа ещё нет. Метка хранит описание режима без имени — повтор `--resume` с тем
же описанием эти листы пропустит (выдача та же: повтор детерминирован, ND0 = NSE22 бит-в-бит). Копия — во временный каталог и rename.

  _copy_mode.py --src NDP3 --dst NSE25
"""
import sys, argparse, shutil
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

TS = Path(r"F:/nds/output/taskS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    S = TS / "rp_lvl" / a.src; D = TS / "rp_lvl" / a.dst
    D.mkdir(parents=True, exist_ok=True)
    n = skip = 0
    for d in sorted(S.iterdir()):
        if not d.is_dir() or not (d / "_replay_ok.txt").exists() or not any(d.glob("*_auto.nlgx")):
            continue
        t = D / d.name
        if t.exists():
            skip += 1; continue
        tmp = D / (d.name + ".copying")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(d, tmp); tmp.rename(t); n += 1
    print(f"★ {a.src} → {a.dst}: скопировано {n}, уже было {skip}")


if __name__ == "__main__":
    main()

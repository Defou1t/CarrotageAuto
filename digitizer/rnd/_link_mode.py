r"""_link_mode.py — НОВАЯ БАЗА БЕЗ КОПИЙ (13.11): каталоги листов `rp_lvl/<src>` с меткой `_replay_ok.txt` переносятся в `rp_lvl/<dst>`
ЖЁСТКИМИ ССЫЛКАМИ (тот же том — место не тратится), кроме листов из `--exclude` (их переигрывают). Метка хранит описание режима без
имени — повтор `--resume` с тем же описанием перенесённые листы пропустит. ⚠ Файлы общие с `<src>`: в `<dst>` их не править на месте
(повтор пишет новые файлы через .tmp + rename — ссылка рвётся, исходник цел).

  _link_mode.py --src NSE22 --dst NSE26 --exclude F:/nds/output/taskS/dup_token/all.txt
"""
import sys, os, argparse, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

TS = Path(r"F:/nds/output/taskS")


def sheet_dir(nm):
    st = Path(nm).stem
    return f"{st[:40]}_{hashlib.md5(st.encode('utf-8')).hexdigest()[:8]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--exclude", default="")
    a = ap.parse_args()
    S = Path(os.path.realpath(TS / "rp_lvl" / a.src)); D = Path(os.path.realpath(TS / "rp_lvl" / a.dst))
    D.mkdir(parents=True, exist_ok=True)
    ex = set()
    if a.exclude:
        ex = {sheet_dir(l.strip()) for l in Path(a.exclude).read_text(encoding="utf-8").splitlines() if l.strip()}
    n = skip = nx = 0
    for d in sorted(S.iterdir()):
        if not d.is_dir() or not (d / "_replay_ok.txt").exists() or not any(d.glob("*_auto.nlgx")):
            continue
        if d.name in ex:
            nx += 1; continue
        t = D / d.name
        if t.exists():
            skip += 1; continue
        tmp = D / (d.name + ".linking")
        tmp.mkdir(exist_ok=True)
        for f in d.iterdir():
            if f.is_file() and not (tmp / f.name).exists():
                os.link(f, tmp / f.name)
        tmp.rename(t); n += 1
    print(f"★ {a.src} → {a.dst}: ссылками {n}, уже было {skip}, исключено (переиграть) {nx}")


if __name__ == "__main__":
    main()

r"""_strict_splice.py — СТРОГАЯ МЕРА НОВОЙ БАЗЫ СКЛЕЙКОЙ (13.11, §6.364): `_strict_<base>.json` с заменой записей листов, пересчитанных
отдельно (`_strict_<fix>.json`). Годится, когда новая база отличается от старой только этими листами (NSE26 = NSE22 + 19 переигранных,
прочие каталоги — жёсткие ссылки на те же файлы). Записи — [лист, часть, кривые, ошибка].

  _strict_splice.py --base nse22 --fix nse26_fix19 --out nse26
"""
import sys, json, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

TS = Path(r"F:/nds/output/taskS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--fix", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = json.loads((TS / f"_strict_{a.base}.json").read_text(encoding="utf-8"))
    F = {r[0]: r for r in json.loads((TS / f"_strict_{a.fix}.json").read_text(encoding="utf-8"))}
    seen = set(); out = []
    for r in B:
        if r[0] in F:
            out.append(F[r[0]]); seen.add(r[0])
        else:
            out.append(r)
    miss = sorted(set(F) - seen)
    if miss:
        print(f"⛔ листов пересчёта нет в базе: {len(miss)} ({miss[:3]}…) — не склеиваю"); sys.exit(2)
    f = TS / f"_strict_{a.out}.json"; tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8"); tmp.replace(f)
    print(f"★ {f.name}: записей {len(out)}, заменено {len(seen)} из {len(F)}")


if __name__ == "__main__":
    main()

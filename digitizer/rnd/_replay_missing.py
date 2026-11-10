r"""_replay_missing.py — ЛИСТЫ, КОТОРЫХ НЕТ В ПОВТОРЕ (08.11, §6.355–6.356): повтор ловит падение листа (MemoryError и т. п.), пишет
«ПАДЕНИЕ» и выходит с кодом 0 — кусок помечается готовым, а лист молча выпадает из A/B (сравнение — по общим листам). Здесь — листы
поля (`tcache_sheets.txt`) и вне (`outside_sheets.txt`), у которых в `rp_lvl/<база>` выдача есть, а в `rp_lvl/<режим>` нет
`*_auto.nlgx` → списки для повторного повтора.

  _replay_missing.py --mode NSE22 [--base NSE19] --out-f <файл> --out-o <файл>     (код выхода 0; число листов — в выводе)
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--base", default="NSE19")
    ap.add_argument("--out-f", required=True)
    ap.add_argument("--out-o", required=True)
    ap.add_argument("--only", default="", help="(стенд на части листов) только листы из этого списка")
    a = ap.parse_args()
    import _sample_review as S
    has = lambda mode, nm: any((S.TS / "rp_lvl" / mode / S.sheet_dir(nm)).glob("*_auto.nlgx"))
    tot = 0
    for lst, out in (("tcache_sheets.txt", a.out_f), ("outside_sheets.txt", a.out_o)):
        names = [l.strip() for l in (S.TS / lst).read_text(encoding="utf-8").splitlines() if l.strip()]
        if a.only:
            keep = {l.strip() for l in Path(a.only).read_text(encoding="utf-8").splitlines() if l.strip()}
            names = [nm for nm in names if nm in keep]
        miss = [nm for nm in names if has(a.base, nm) and not has(a.mode, nm)]
        Path(out).write_text("\n".join(miss) + ("\n" if miss else ""), encoding="utf-8")
        tot += len(miss)
        print(f"★ {lst}: листов {len(names)}, выдача есть в {a.base} и нет в {a.mode}: {len(miss)}")
    print(f"★ НЕДОСТАЁТ {tot}")


if __name__ == "__main__":
    main()

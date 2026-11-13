r"""_dup_token_sheets.py — ЛИСТЫ, ЗАДЕТЫЕ ОШИБКОЙ ЗАПИСИ emit (13.11): в рамке (nlgx архива) есть кривые (не DA) с одинаковым первым
словом имени (NGK1 DA1 SA1 / NGK1 DA2 SA1 …). До правки `emit.ifd_of` трасса второго слота писалась в IFD первого. Выход — списки
листов (все / поле-кэш / вне / сорт A) в `taskS/dup_token/`.

  _dup_token_sheets.py
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
from collections import Counter
from extract_nlgx import extract
from auto import meta as M

TS = Path(r"F:/nds/output/taskS")
ARCH = Path(r"F:\nds\projects\Archive")
OUT = TS / "dup_token"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hit = {}
    for q in sorted(ARCH.glob("*/wlg/*.nlgx")):
        try:
            names = [c["name"] for c in extract(str(q))["curves"] if M.mnem_root(c["name"]) != "DA"]
        except Exception as e:
            print(f"  ⚠ {q.name}: {type(e).__name__}"); continue
        cnt = Counter(n.split()[0] for n in names)
        d = sorted(n for n in names if cnt[n.split()[0]] > 1)
        if d:
            hit[q.name] = d
    lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
    parts = {"tcache": lst("tcache_sheets.txt"), "outside": lst("outside_sheets.txt"), "holdoutA": lst("holdoutA_sheets.txt")}
    (OUT / "all.json").write_text(json.dumps(hit, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "all.txt").write_text("\n".join(sorted(hit)) + "\n", encoding="utf-8")
    print(f"★ листов архива с повтором первого слова имени: {len(hit)}, кривых {sum(len(v) for v in hit.values())}")
    for k, S in parts.items():
        L = sorted(n for n in hit if n in S)
        (OUT / f"{k}.txt").write_text("\n".join(L) + ("\n" if L else ""), encoding="utf-8")
        print(f"  {k}: {len(L)} листов")


if __name__ == "__main__":
    main()

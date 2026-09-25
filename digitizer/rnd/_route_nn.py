r"""_route_nn.py — ДЛИНА КАНДИДАТА ТАК, КАК ЕЁ ВИДИТ `emit` (18.09, к `_route_learn.py --ink`).

ЗАЧЕМ. Оринтированная длина пары (`o_len`, `_route_learn.py`) считалась по `dense()` — с мостами через
разрывы ≤ 200 строк. `emit` в момент выбора видит трассу раскладки `tr` и пишет `xs[i] = tr[top_y+i]`
либо NULL (`emit.py:456`), то есть ЧИСЛО НЕПУСТЫХ СТРОК выдачи = |{строки tr в окне слота}| — это
величина, которую `emit` может посчитать РОВНО так же. Ловушка §6.146 (признак стенда ≠ признак
механизма) закрывается тем, что стенд берёт ту же величину: непустые строки из выдачи.
Кэш → `route_nn_cache.pkl`: {(sheet, track, slot, tag): (nonnull, span, dense_len)}.
"""
import sys, pickle, hashlib, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--cachef", default="route_learn_cache.pkl")
ap.add_argument("--a", default="ab_wellmap/A")
ap.add_argument("--b", default="ab_rdhonest/B")
ap.add_argument("--out", default="route_nn_cache.pkl")
a = ap.parse_args()
TS = Path(a.ts)
TR = pickle.load(open(TS / a.cachef, "rb"))


def rd(root, sh):
    stem = Path(sh).stem
    pd = TS / root / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    if not got:
        return {}
    out = {}
    for c in extract(str(got))["curves"]:
        if M.mnem_root(c["name"]) == "DA":
            continue
        nn = sum(1 for x in c["xs"] if x != NULL)
        rows = [i for i, x in enumerate(c["xs"]) if x != NULL]
        span = (rows[-1] - rows[0] + 1) if rows else 0
        out[c["name"]] = (nn, span, len(dense(c)))
    return out


out, miss = {}, 0
sheets = sorted({r["sheet"] for r in TR})
by = {}
for r in TR:
    by.setdefault(r["sheet"], []).append(r)
for k, sh in enumerate(sheets, 1):
    WA, WB = rd(a.a, sh), rd(a.b, sh)
    for r in by[sh]:
        for s, tag in r["have"]:
            v = (WA if tag == "A" else WB).get(s)
            if v is None:
                miss += 1
                continue
            out[(sh, r["track"], s, tag)] = v
    if k % 200 == 0:
        print(f"  … {k}/{len(sheets)}")
pickle.dump(out, open(TS / a.out, "wb"))
print(f"★ ГОТОВО: кандидатов {len(out)}, без трассы {miss} → {TS / a.out}")

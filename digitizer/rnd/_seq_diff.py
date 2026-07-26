r"""seq_diff2.py — полистный разбор жадный/селектор ПО ДВУМ критериям сразу.

Свип показал: цифры §6.66 (22->39, 27->31, 17->32) воспроизводятся ТОЛЬКО критерием `oracle`
при мостике 20 — то есть БЕЗ отбора K из N. `oracle` — это потолок, он по докстрингу
_pick_gate.py:19 «в прод не годится». Поэтому считаем оба:
  oracle  — то, что цитирует §6.66 (качество ТРАССИРОВКИ, отбор отключён);
  nl+npts — то, что реально отдаёт прод (§6.49), то есть цифра для заказчика.
"""
import sys, pickle, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.argv = ["_pick_gate.py"]
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")

import _pick_gate as G
from pathlib import Path

ROOT = Path(r"F:\nds\output\taskS\pick_gate")
PAIRS = [("гейт", "cache", "seq_gate"), ("валидация", "holdout", "seq_hold"),
         ("третий", "wide", "seq_wide")]
G.a.bridge = 20
G.a.dedup_tol = 50

raw = {}
for _, dg, dsq in PAIRS:
    for d in (dg, dsq):
        raw.setdefault(d, {f.stem: pickle.load(open(f, "rb"))
                           for f in sorted((ROOT / d).glob("*.pkl"))})

report = {}
for pick in ("oracle", "nl+npts"):
    print(f"\n{'#'*104}\n### КРИТЕРИЙ {pick}"
          + ("   (потолок: отбора нет — ЭТО цитирует §6.66)" if pick == "oracle"
             else "   (ПРОД §6.49 — цифра для заказчика)"))
    tg = ts = 0
    for label, dg, dsq in PAIRS:
        names = sorted(set(raw[dg]) & set(raw[dsq]))
        rows = []
        for nm in names:
            r = {}
            for tag, d in (("greedy", dg), ("seq", dsq)):
                dd = raw[d][nm]
                cand = [G.bridge(t, 20) for t in dd["traces"] if len(t) >= G.MINPTS]
                K = max(1, dd["K"] or len(dd["GM"]))
                h, ncur, med = G.score(G.PICKERS[pick](cand, K, G.a), dd["GM"], dd["raw"])
                r[tag] = h
                r[tag + "_med"] = None if med != med else round(med, 1)
                r[tag + "_n"] = len(cand)
                r["curves"] = ncur
            r.update(sheet=nm, K=K, d=r["seq"] - r["greedy"])
            rows.append(r)
        sg, ss = sum(r["greedy"] for r in rows), sum(r["seq"] for r in rows)
        up = [r for r in rows if r["d"] > 0]
        dn = [r for r in rows if r["d"] < 0]
        tg += sg; ts += ss
        print(f"\n{label}: листов {len(rows)}, кривых {sum(r['curves'] for r in rows)}"
              f"   {sg} -> {ss} ({ss-sg:+d})   вверх {len(up)}, ВНИЗ {len(dn)}")
        for r in sorted(rows, key=lambda q: q["d"]):
            if r["d"] < 0:
                print(f"   ВНИЗ {r['d']:+d}  {r['greedy']}->{r['seq']} из {r['curves']}"
                      f"  med {r['greedy_med']}->{r['seq_med']}"
                      f"  трасс {r['greedy_n']}->{r['seq_n']}  K={r['K']}   {r['sheet']}")
        report[f"{pick}/{label}"] = dict(greedy=sg, seq=ss, up=len(up), down=len(dn),
                                         rows=rows)
    print(f"\n>>> ИТОГО {pick}: {tg} -> {ts} ({ts-tg:+d})")
    report[f"{pick}/ИТОГО"] = dict(greedy=tg, seq=ts)

Path(ROOT / "seq_diff2.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
print(f"\n{ROOT / 'seq_diff2.json'}")

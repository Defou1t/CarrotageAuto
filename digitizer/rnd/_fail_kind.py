r"""_fail_kind.py — ЧЕМ НЕЧЕСТНА ВЕРСИЯ СЛОТА (02.10, к §6.256): чужая кривая, сдвиг, мимо всех, короткая.

§6.245: на поле у 61% слотов с обеими версиями (прод / декодер) не честна ни одна. Контекст декодера (§6.256) лечит лишь
один вид провала — уход на ЧУЖУЮ кривую (личность на пересечении, следование). Здесь — доли видов провала, чтобы знать
предел опыта. Для каждой нечестной версии слота (окно слота, мост ≤ 200 — как запись и счёт):
  «нет трассы»  — общих с эталоном строк < 30;
  «короткая»    — медиана ≤ 3 px, но покрытие < 0.9 (ведёт верно, но не всю кривую);
  иначе по строкам с ошибкой > 3 px:
  «чужая»       — ≥ половины таких строк в 3 px от ДРУГОЙ эталонной кривой листа (ушла на соседку);
  «сдвиг»       — ≥ половины таких строк с ошибкой 3–8 px (идёт рядом: край штриха, параллельная линия);
  «мимо»        — прочее (сетка, текст, шум, бледная тушь; или неоцифрованная кривая — эталон знает не все).
Только чтение кэша и повтора; выдачу не меняет.

  _fail_kind.py --dir F:/nds/output/taskS/rp_vc/N --cache F:/nds/output/taskS/tcache
"""
import sys, argparse, pickle, hashlib, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dir", required=True)
ap.add_argument("--cache", required=True)
ap.add_argument("--dump", default="")
a = ap.parse_args()
TS = Path(a.ts)
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SETS = {"поле": lst("wellmap_sheets.txt"), "сорт A": lst("holdoutA_sheets.txt")}
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}


def un(t):
    return dict(zip(t[0].tolist(), t[1].tolist()))


def windowed(tr, y0, n):
    """как запись слота + `dense` счёта: строки окна, мост ≤ 200 строк (копия `_version_ceiling.windowed`)"""
    pts = sorted((y, x) for y, x in tr.items() if y0 <= y < y0 + n)
    out = {}
    for (ya, xa), (yb, xb) in zip(pts, pts[1:]):
        out[ya] = xa
        if 0 < yb - ya <= 200:
            for yy in range(ya + 1, yb):
                out[yy] = xa + (xb - xa) * (yy - ya) / (yb - ya)
    if pts:
        out[pts[-1][0]] = pts[-1][1]
    return out


ST = {}          # статистика последней версии: доля строк эталона в 3 px, медиана ошибки плохих строк, доля плохих > 30 px


def kind(tr, gt, others):
    ST.clear()
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return "нет трассы", 0.0
    err = np.array([abs(tr[y] - gt[y]) for y in com])
    cov = len(com) / max(1, len(gt))
    eb = err[err > 3.0]
    ST.update(good=float((err <= 3.0).sum()) / max(1, len(gt)), bad_med=float(np.median(eb)) if len(eb) else 0.0,
              bad_far=float(np.mean(eb > 30.0)) if len(eb) else 0.0)
    if np.median(err) <= 3.0:
        return ("честна" if cov >= 0.9 else "короткая"), cov
    bad = [y for y, e in zip(com, err) if e > 3.0]
    near = 0
    for y in bad:
        x = tr[y]
        if any(y in o and abs(o[y] - x) <= 3.0 for o in others):
            near += 1
    if near >= 0.5 * len(bad):
        return "чужая", cov
    if float(np.mean((err[err > 3.0] <= 8.0))) >= 0.5:
        return "сдвиг", cov
    return "мимо", cov


KINDS = ("нет трассы", "короткая", "чужая", "сдвиг", "мимо")
C = {(sn, ver, grp): Counter() for sn in SETS for ver in ("прод", "декодер") for grp in ("ни одна", "одна")}
ROWS = []
for sn, names in SETS.items():
    for sh in names:
        q = SRC.get(sh)
        if not q:
            continue
        stem = q.stem; key = f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
        pj = next((Path(a.dir) / key).glob("*_pick.json"), None)
        if not pj:
            continue
        pk = json.loads(pj.read_text(encoding="utf-8"))
        slots = [s for s in pk.get("slots", []) if s.get("pi") is not None and s.get("di") is not None]
        if not slots:
            continue
        v = pickle.load(open(Path(a.cache) / f"{key}.pkl", "rb"))
        fr = {c["name"]: c for c in extract(v["frame_nlgx"])["curves"]}
        G = {c["name"]: dense(c) for c in extract(str(q))["curves"]
             if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        for s in slots:
            nm = s["name"]
            if nm not in G or nm not in fr:
                continue
            y0, n = int(fr[nm]["top_y"]), len(fr[nm]["xs"])
            others = [g for k, g in G.items() if k != nm]
            kp, cp = kind(windowed(un(v["traces"][s["pi"]][1]), y0, n), G[nm], others); sp = dict(ST)
            kd, cd = kind(windowed(un(v["alt"][s["di"]][1]), y0, n), G[nm], others); sd = dict(ST)
            grp = "ни одна" if kp != "честна" and kd != "честна" else "одна" if (kp == "честна") != (kd == "честна") else "обе"
            if grp == "обе":
                continue
            if kp != "честна":
                C[(sn, "прод", grp)][kp] += 1
            if kd != "честна":
                C[(sn, "декодер", grp)][kd] += 1
            ROWS.append(dict(set=sn, sheet=sh, name=nm, kp=kp, kd=kd, cov_p=cp, cov_d=cd, grp=grp,
                             fam=M.mnem_root(nm), sp=sp, sd=sd))
if a.dump:
    pickle.dump(ROWS, open(a.dump, "wb"))
for (sn, ver, grp), c in C.items():
    tot = sum(c.values())
    if not tot:
        continue
    print(f"★ {sn} · {ver} · слоты, где честна {grp}: {tot}  " +
          "  ".join(f"{k} {c[k]} ({100 * c[k] / tot:.0f}%)" for k in KINDS))
print("\n★ Нечестные версии в слотах «ни одна»: доля строк эталона в 3 px (сколько кривой ведёт верно), медиана ошибки "
      "плохих строк, доля плохих > 30 px — по виду провала:")
for sn in SETS:
    for ver, kk, ss in (("декодер", "kd", "sd"), ("прод", "kp", "sp")):
        for kd in ("чужая", "мимо"):
            rr = [r[ss] for r in ROWS if r["set"] == sn and r["grp"] == "ни одна" and r[kk] == kd and r[ss]]
            if rr:
                g = np.array([r["good"] for r in rr]); bm = np.array([r["bad_med"] for r in rr])
                bf = np.array([r["bad_far"] for r in rr])
                print(f"   {sn} · {ver} · {kd}: {len(rr)}; ведёт верно — медиана {np.median(g):.2f} (≥ 0.5 у {np.mean(g >= 0.5):.0%}); "
                      f"ошибка плохих строк — медиана {np.median(bm):.0f} px (квартили {np.percentile(bm, 25):.0f}–"
                      f"{np.percentile(bm, 75):.0f}); доля плохих > 30 px — медиана {np.median(bf):.2f}")
for sn in SETS:
    pair = Counter((r["kp"], r["kd"]) for r in ROWS if r["set"] == sn and r["grp"] == "ни одна")
    tot = sum(pair.values())
    print(f"\n★ {sn}: пары видов (прод, декодер) в слотах «ни одна», {tot}:")
    for (kp, kd), m in pair.most_common(12):
        print(f"   {kp:11s} / {kd:11s}  {m:4d}  ({100 * m / tot:.0f}%)")

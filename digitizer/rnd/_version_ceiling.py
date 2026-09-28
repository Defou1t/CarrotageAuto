r"""_version_ceiling.py — ПОТОЛОК ВЫБОРА ВЕРСИИ СЛОТА (прод / декодер) В НЫНЕШНЕМ ПРОДЕ (§6.245, 28.09).

§6.229: из 470 кривых с честным кандидатом 94 терялись на ВЕРСИИ другого пути; вето §6.240 вернуло +8. Здесь по повтору
нынешнего прода (`_pick.json` слотов с индексами обеих версий `pi`/`di`, §6.245) и кэшу: для каждого слота, где есть обе версии, —
честна ли каждая под именем слота (трасса обрезается окном слота и мостится ≤ 200 строк — как запись и счёт), какая выбрана
(`src`). Потолок = слоты, где честна только НЕвыбранная. Признаки для правила/модели (уверенность декодера из сайдкара, длины в
окне, доли прыжков и пропусков обеих версий, их расхождение) и их AUC на слотах, где честна ровно одна версия.

  _version_ceiling.py --dir F:/nds/output/taskS/rp_vc/N --cache F:/nds/output/taskS/tcache --conf F:/nds/output/taskS/conf_tcache.pkl
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
ap.add_argument("--conf", default="")
ap.add_argument("--dump", default=r"F:/nds/output/taskS/version_rows.pkl")
a = ap.parse_args()
TS = Path(a.ts)
CONF = pickle.load(open(a.conf, "rb")) if a.conf else {}
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SETS = {"поле": lst("wellmap_sheets.txt"), "сорт A": lst("holdoutA_sheets.txt")}
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}


def un(t):
    return dict(zip(t[0].tolist(), t[1].tolist()))


def windowed(tr, y0, n):
    """как запись слота + `dense` счёта: строки окна, мост ≤ 200 строк"""
    pts = sorted((y, x) for y, x in tr.items() if y0 <= y < y0 + n)
    out = {}
    for (ya, xa), (yb, xb) in zip(pts, pts[1:]):
        out[ya] = xa
        if 0 < yb - ya <= 200:
            for yy in range(ya + 1, yb):
                out[yy] = xa + (xb - xa) * (yy - ya) / (yb - ya)
    if pts:
        out[pts[-1][0]] = pts[-1][1]
    return out, len(pts)


def hon(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return False, None
    med = float(np.median([abs(tr[y] - gt[y]) for y in com]))
    return med <= 3.0 and len(com) / max(1, len(gt)) >= 0.9, med


def tstats(tr, y0, n):
    ys = sorted(y for y in tr if y0 <= y < y0 + n)
    if len(ys) < 2:
        return 0.0, 1.0, 0.0
    xs = np.array([tr[y] for y in ys]); dy = np.diff(ys)
    jumps = float(np.mean(np.abs(np.diff(xs)) / np.maximum(1, dy) > 3))
    gaps = 1.0 - len(ys) / max(1, ys[-1] - ys[0] + 1)
    return len(ys) / max(1, n), gaps, jumps


def auc(sc, y):
    sc = np.asarray(sc, float); y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(sc)) + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


ROWS = []; C = {k: Counter() for k in SETS}
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
        conf = CONF.get(f"{key}.pkl") or v.get("alt_conf") or []
        for s in slots:
            nm = s["name"]
            if nm not in G or nm not in fr:
                continue
            y0, n = int(fr[nm]["top_y"]), len(fr[nm]["xs"])
            tp = un(v["traces"][s["pi"]][1]); td = un(v["alt"][s["di"]][1])
            wp, np_ = windowed(tp, y0, n); wd, nd = windowed(td, y0, n)
            hp, mp = hon(wp, G[nm]); hd, md = hon(wd, G[nm])
            com = [y for y in wp if y in wd]
            agree = float(np.mean([abs(wp[y] - wd[y]) <= 3 for y in com])) if com else 0.0
            cp, gp, jp = tstats(tp, y0, n); cd, gd, jd = tstats(td, y0, n)
            dc = conf[s["di"]][0] if s["di"] < len(conf) and conf[s["di"]] else None
            row = dict(set=sn, sheet=sh, name=nm, src=s["src"], base=s["base"], flip=s["flip"], veto=bool(s.get("veto")),
                       hp=hp, hd=hd, conf=dc, cov_p=cp, gap_p=gp, jmp_p=jp, cov_d=cd, gap_d=gd, jmp_d=jd, agree=agree,
                       nn_base=s["nn_base"], nn_alt=s["nn_alt"])
            ROWS.append(row)
            c = C[sn]; c["слотов с обеими версиями"] += 1
            c[("обе честны" if hp and hd else "честна только прод" if hp else "честна только декодер" if hd else "ни одна")] += 1
            chosen_h = hp if s["src"] == "prod" else hd if s["src"] == "dec" else False
            other_h = hd if s["src"] == "prod" else hp if s["src"] == "dec" else False
            if other_h and not chosen_h:
                c[f"ПОТОЛОК: честна только невыбранная (выбран {s['src']})"] += 1
pickle.dump(ROWS, open(a.dump, "wb"))
for sn, c in C.items():
    print(f"★ {sn}:")
    for k, vv in c.items():
        print(f"   {k}: {vv}")
one = [r for r in ROWS if r["hp"] != r["hd"] and r["conf"] is not None]
y = [r["hd"] for r in one]
print(f"\n★ СЛОТЫ, ГДЕ ЧЕСТНА РОВНО ОДНА ВЕРСИЯ: {len(one)} (декодер {sum(y)}, прод {len(y) - sum(y)}); AUC «честна версия декодера»:")
for nm_, f in (("уверенность декодера", lambda r: r["conf"]), ("покрытие окна декодера", lambda r: r["cov_d"]),
               ("−покрытие окна прода", lambda r: -r["cov_p"]), ("покрытие дек − прод", lambda r: r["cov_d"] - r["cov_p"]),
               ("−пропуски декодера", lambda r: -r["gap_d"]), ("пропуски прода", lambda r: r["gap_p"]),
               ("−прыжки декодера", lambda r: -r["jmp_d"]), ("прыжки прода", lambda r: r["jmp_p"]),
               ("согласие версий", lambda r: r["agree"]), ("лог-отношение длин", lambda r: np.log1p(r["nn_alt"]) - np.log1p(r["nn_base"]) if r["base"] == "prod" else np.log1p(r["nn_base"]) - np.log1p(r["nn_alt"]))):
    print(f"   {nm_}: {auc([f(r) for r in one], y):.3f}")

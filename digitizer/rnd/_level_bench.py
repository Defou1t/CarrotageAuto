r"""_level_bench.py — БЫСТРЫЙ СТЕНД РАСКЛАДКИ УРОВНЕЙ МАСШТАБА ПО ВЫДАЧЕ (03.10, §6.262).

Мера — именные честные В ЗНАЧЕНИЯХ (как `_name_cost_prod.py --levels`, то же имя): выдача по строкам пересчитывается по шкале
своего уровня в пиксели шкалы уровня эталона, честно — медиана ≤ 3 px и покрытие ≥ 0.9. Трассы не меняются, меняется
только уровень по строкам. Кэш (`build`) — все кривые выдачи `rp_vc/N` с тем же именем в эталоне (поле + сорт A); у кривых без
цепочки масштабов уровень не влияет (их честность фиксирована).

  build → `--cache`; затем варианты:
    prod      — уровни выдачи как есть (паритет со счётом `--levels`);
    oracle    — уровни эталона (потолок при нынешних трассах);
    dec_all   — decode_levels для всех кривых с цепочкой;
    dec_res   — decode для резистивных (с нормализацией приставок BKZ_), остальным уровень 0;
    zero      — всем уровень 0;
  параметры декодера: --lam, --min-run, --dxfrac.

  _level_bench.py build
  _level_bench.py run --variants prod oracle dec_res zero
"""
import sys, argparse, pickle, hashlib, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["build", "run"])
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_vc/N")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/level_bench.pkl")
ap.add_argument("--variants", nargs="+", default=["prod", "oracle", "dec_all", "dec_res", "zero"])
ap.add_argument("--lam", type=float, default=0.4)
ap.add_argument("--min-run", type=int, default=25)
ap.add_argument("--dxfrac", type=float, default=0.12)
ap.add_argument("--rail", type=float, default=0.0, help="rail_gate декодера: переход вверх дёшев, только если перо у правого края (доля ширины)")
ap.add_argument("--bias", type=float, default=0.0, help="level_bias: приор к 1×")
ap.add_argument("--down", type=float, default=1.0, help="down_mult: удорожание перехода вниз")
ap.add_argument("--med", type=int, default=0, help="медианный фильтр x трассы по строкам перед декодером (окно, 0 — нет)")
ap.add_argument("--tag", default="")
ap.add_argument("--fam-table", action="store_true")
ap.add_argument("--sheets", default="", help="(build) §6.296: свой список листов (файл в --ts), набор «вне» вместо поля и сорта A")
ap.add_argument("--cap", type=float, default=400.0)
ap.add_argument("--lamt", type=float, default=50.0)
ap.add_argument("--bias-px", type=float, default=0.01)
a = ap.parse_args()
TS = Path(a.ts)


def lv_rows(segs):
    out = {}
    for y0, y1, lv in segs or []:
        for y in range(int(y0), int(y1) + 1):
            out[y] = int(lv)
    return out


if a.cmd == "build":
    from extract_nlgx import extract, NULL
    from _multi_replica_probe import dense
    from auto import meta as M
    import decode_levels as DL
    lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    CUR = []
    for sn, f in ((("вне", a.sheets),) if a.sheets else (("поле", "wellmap_sheets.txt"), ("сорт A", "holdoutA_sheets.txt"))):
        for sh in lst(f):
            q = SRC.get(sh)
            if not q:
                continue
            stem = q.stem
            pd = Path(a.dir) / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
            got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
            if not got:
                continue
            mt = extract(str(q)); mw = {c["name"]: c for c in extract(str(got))["curves"]}
            for c in mt["curves"]:
                nm = c["name"]
                if M.mnem_root(nm) == "DA" or sum(1 for x in c["xs"] if x != NULL) < 50 or nm not in mw:
                    continue
                w = mw[nm]
                gd, td = dense(c), dense(w)
                gy = np.array(sorted(gd), np.int32); ty = np.array(sorted(td), np.int32)
                xsr = [(w["top_y"] + i, x) for i, x in enumerate(w["xs"]) if x != NULL]
                CUR.append(dict(set=sn, sheet=sh, name=nm, root=M.mnem_root(nm), res=DL.is_resistive(nm),
                                chain=DL.build_family(mt, c),
                                gy=gy, gx=np.array([gd[y] for y in gy], np.float32), lt=list(c.get("segments") or []),
                                ty=ty, tx=np.array([td[y] for y in ty], np.float32), lw=list(w.get("segments") or []),
                                xy=np.array([y for y, _ in xsr], np.int32), xx=np.array([x for _, x in xsr], np.float32)))
    pickle.dump(CUR, open(a.cache, "wb"))
    print(f"кривых в кэше {len(CUR)}")
    sys.exit()

import decode_levels as DL
from auto import refine
CUR = pickle.load(open(a.cache, "rb"))


def seg_levels(segs, rows):
    """уровни строк rows по сегментам (вне сегментов — 0)"""
    out = np.zeros(len(rows), np.int16)
    for y0, y1, lv in segs or []:
        if lv:
            out[(rows >= y0) & (rows <= y1)] = lv
    return out


def honest(cv, lo):
    """именная честность в значениях; lo — None (уровни выдачи как есть), {} (всё 0) или dict строка→уровень
    (строки без значения наследуют последний уровень)"""
    gy, gx, ty, tx, ch = cv["gy"], cv["gx"], cv["ty"], cv["tx"], cv["chain"]
    com, ig, it = np.intersect1d(gy, ty, return_indices=True)
    if len(com) < 30:
        return False
    x = tx[it].astype(np.float64); g = gx[ig].astype(np.float64)
    lt = seg_levels(cv["lt"], com)
    if lo is None:
        lw = seg_levels(cv["lw"], com)
    elif not lo:
        lw = np.zeros(len(com), np.int16)
    else:
        ks = np.array(sorted(lo), np.int64); vs = np.array([lo[k] for k in ks], np.int16)
        pos = np.searchsorted(ks, com, side="right") - 1
        lw = np.where(pos >= 0, vs[np.clip(pos, 0, len(vs) - 1)], 0).astype(np.int16)
    if len(ch) >= 2:
        diff = lw != lt
        for o in np.unique(lw[diff]):
            for t in np.unique(lt[diff & (lw == o)]):
                m = diff & (lw == o) & (lt == t)
                if o >= len(ch) or t >= len(ch):
                    x[m] = 1e4; continue
                so, sg = ch[o], ch[t]
                v = so["v_left"] + (x[m] - so["x_left"]) * (so["v_right"] - so["v_left"]) / ((so["x_right"] - so["x_left"]) or 1)
                x[m] = sg["x_left"] + (v - sg["v_left"]) * (sg["x_right"] - sg["x_left"]) / ((sg["v_right"] - sg["v_left"]) or 1)
    e = np.abs(x - g)
    return bool(np.median(e) <= 3.0 and len(com) / max(1, len(gy)) >= 0.9)


def norm_res(name):
    w = name.split()[0]
    w = re.sub(r"^BKZ_", "", w)
    return DL.is_resistive(w + " x")


def decode(cv):
    ys, v = cv["xy"], cv["xx"].astype(np.float64)
    if a.med > 1:
        h = a.med // 2
        v = np.array([np.median(v[max(0, i - h):i + h + 1]) for i in range(len(v))])
    xs = dict(zip(ys.tolist(), v.tolist()))
    raw = DL.decode(xs, cv["chain"], lam=a.lam, dxfrac=a.dxfrac, rail_gate=a.rail, level_bias=a.bias, down_mult=a.down)
    return refine.enforce_min_run(raw, a.min_run)


def decode_px(xs_by_row, family, cap=400.0, lam_t=50.0, bias=0.01):
    """★ 03.10 (§6.263): DP уровней с непрерывностью В ПИКСЕЛЯХ ПРЕЖНЕЙ ШКАЛЫ: шаг (kp, x_prev) → (k, x) стоит
    min((|inv_kp(v_k(x)) − x_prev| / dy)², cap) + lam_t·|k − kp| + bias·k. Годится и для цепочек ×5 (умножение), и для
    сдвиговых (термометрия 24–33.5 → 33.5–43); потолок cap — случайный скачок на чужую линию не «объясняется» сменой уровня."""
    K = len(family)
    rows = sorted(xs_by_row)
    if K <= 1 or len(rows) < 2:
        return {y: 0 for y in rows}
    A = []
    for s_ in family:
        k_ = (s_["v_right"] - s_["v_left"]) / ((s_["x_right"] - s_["x_left"]) or 1)
        A.append((s_["x_left"], s_["v_left"], k_ if k_ else 1e-9))
    xs = np.array([xs_by_row[y] for y in rows], np.float64); ys = np.array(rows, np.float64)
    V = np.array([[vl + (x - xl) * k_ for (xl, vl, k_) in A] for x in xs])           # значения точки на каждом уровне
    dp = np.array([bias * k for k in range(K)], np.float64)
    back = np.zeros((len(rows), K), np.int8)
    lev = np.arange(K)
    for i in range(1, len(rows)):
        dy = max(1.0, ys[i] - ys[i - 1])
        # куда легла бы новая точка уровня k на шкале уровня kp: x' = xl_kp + (V[i,k] − vl_kp)/k_kp
        xl = np.array([a_[0] for a_ in A]); vl = np.array([a_[1] for a_ in A]); kk = np.array([a_[2] for a_ in A])
        xp_on = xl[None, :] + (V[i][:, None] - vl[None, :]) / kk[None, :]                 # [k, kp]
        d = np.minimum(((xp_on - xs[i - 1]) / dy) ** 2, cap)
        tot = dp[None, :] + d + lam_t * np.abs(lev[:, None] - lev[None, :])
        back[i] = np.argmin(tot, axis=1)
        dp = tot[lev, back[i]] + bias * lev
    k = int(np.argmin(dp)); out = {rows[-1]: k}
    for i in range(len(rows) - 1, 0, -1):
        k = int(back[i][k]); out[rows[i - 1]] = k
    return out


def oracle_levels(cv):
    """уровни эталона как точки смены (сегменты могут перекрываться на граничной строке — как seg_levels: поздний сверху)"""
    if not cv["lt"]:
        return {}
    y0 = int(min(s[0] for s in cv["lt"])); y1 = int(max(s[1] for s in cv["lt"])) + 1
    rows = np.arange(y0, y1 + 1)
    lv = seg_levels(cv["lt"], rows)
    ch = np.flatnonzero(np.diff(np.concatenate([[-1], lv])) != 0)
    return {int(rows[i]): int(lv[i]) for i in ch}


VAR = {
    "prod": lambda cv: None,
    "oracle": lambda cv: oracle_levels(cv),
    "zero": lambda cv: {},
    "dec_all": lambda cv: decode(cv) if len(cv["chain"]) >= 2 else {},
    "dec_res": lambda cv: decode(cv) if len(cv["chain"]) >= 2 and norm_res(cv["name"]) else {},
    "px": lambda cv: refine.enforce_min_run(decode_px(dict(zip(cv["xy"].tolist(), cv["xx"].astype(float).tolist())), cv["chain"],
                                                      cap=a.cap, lam_t=a.lamt, bias=a.bias_px), a.min_run) if len(cv["chain"]) >= 2 else {},
}
res = {}
FAMH = {}
for v in a.variants:
    C = Counter()
    PER = Counter()
    FH = Counter()
    for cv in CUR:
        if len(cv["chain"]) < 2:
            h = honest(cv, None)
        else:
            h = honest(cv, VAR[v](cv))
            FH[(cv["set"], re.sub(r"^BKZ_", "", cv["root"]))] += h
        C[cv["set"]] += h
        PER[(cv["set"], cv["sheet"])] += h
    res[v] = PER; FAMH[v] = FH
    print(f"★ {v:8s} {a.tag} (lam {a.lam}, min_run {a.min_run}, rail {a.rail}, bias {a.bias}, down {a.down}, med {a.med}): именных честных в значениях — поле {C['поле']}, сорт A {C['сорт A']}")
if "prod" in res:
    rng = np.random.default_rng(0)
    for v in res:
        if v == "prod":
            continue
        for sn in ("поле", "сорт A"):
            keys = {k for k in list(res["prod"]) + list(res[v]) if k[0] == sn}
            d = np.array([res[v][k] - res["prod"][k] for k in keys]); nz = d[d != 0]
            p = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)) >= abs(d.sum()))) if len(nz) else 1.0
            print(f"   {v} против prod, {sn}: Δ {int(d.sum()):+d} (листов ↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {p:.4f})")

if a.fam_table and "prod" in FAMH:
    for sn in ("поле", "сорт A"):
        fams = sorted({k[1] for v in FAMH.values() for k in v if k[0] == sn}, key=lambda f: -FAMH["prod"].get((sn, f), 0))
        print(f"★ {sn}: именных честных в значениях по семействам (только кривые с цепочкой): " + "; ".join(
            f"{f} " + "/".join(str(FAMH[v].get((sn, f), 0)) for v in a.variants) for f in fams[:20]) + f"   [{' / '.join(a.variants)}]")

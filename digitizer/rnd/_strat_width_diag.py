r"""_strat_width_diag.py — ЕСТЬ ЛИ СИГНАЛ В ШИРИНЕ?

Прежде чем крутить параметры, честно меряем: в ветке else, на строках где СВОЙ ран
БЫЛ (избежный уход), отличается ли ширина СВОЕГО рана от ширины ВЫБРАННОГО (чужого)?
Если распределения совпадают — приор ширины бесполезен, и это надо признать.

Меряем:
  • |w_own - w_prior| vs |w_pick - w_prior|  (w_prior — бегущая медиана принятых);
  • доля строк, где cost-argmin по ширине выбрал бы СВОЙ ран (против базовых 45%);
  • сколько кандидатов вообще на строке (если 1 — выбирать не из чего).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
import numpy as np
import _relatch_bench as BE

TOL = 3.0
WIDE = BE.WIDE_RUN
WIN = 64

S = {"else_own": 0, "base_ok": 0, "w_ok": 0, "gap_ok": 0, "one_cand": 0,
     "n_cand": [], "dw_own": [], "dw_pick": [], "w_own": [], "w_pick": [],
     "gap_own": [], "gap_pick": [], "prior": []}
# ★ то же самое, но ТОЛЬКО в МОМЕНТ УХОДА (прошлая строка — своя, эта — чужая).
# Здесь приор ширины ЧИСТЫЙ (набран на своей кривой), а не загрязнён чужой.
D = {"n": 0, "w_ok": 0, "gap_ok": 0, "cost_ok": 0, "dw_own": [], "dw_pick": [],
     "gap_own": [], "gap_pick": [], "w_own": [], "w_pick": [], "prior": []}
DUMP = []       # (gaps, widths, w_prior, own_idx, base_idx) на каждый избежный уход


def run():
    for sheet in BE.load():
        H = sheet["H"]
        names, mat = BE._gt_matrix(sheet)
        for rec in sheet["curves"]:
            gys, gxs = sheet["gt"][rec["name"]]
            own = dict(zip(gys.tolist(), gxs.tolist()))
            best_c, best_med = None, 1e9
            for c in sheet["colors"]:
                tr = BE.trace(rec, rec["runs"][c], H)
                common = [y for y in tr if y in own]
                if len(common) < 30:
                    continue
                med = float(np.median([abs(tr[y] - own[y]) for y in common]))
                if med < best_med:
                    best_med, best_c = med, c
            if best_c is None:
                continue
            hist = []
            prev = {"own": False}

            def probe(y, A, B, Cc, pred, x, v, branch, k, hist=hist, own=own, prev=prev):
                if branch in ("empty", "coast"):
                    return
                if k is not None:
                    w = int(B[k] - A[k])
                    if w < WIDE:
                        hist.append(w)
                if y not in own or not hist:
                    prev["own"] = False
                    return
                ox = own[y]
                hit = np.nonzero((A - TOL <= ox) & (ox <= B + TOL))[0]
                oi = int(hit[np.argmin(np.abs(Cc[hit] - ox))]) if len(hit) else None
                was_own, prev["own"] = prev["own"], (oi is not None and k == oi)
                # ★ момент ухода: шли по своей, свой ран ЕСТЬ, но взяли чужой — ИЗБЕЖНЫЙ уход
                if was_own and oi is not None and k != oi:
                    wp = float(np.median(hist[-WIN:]))
                    wr = (B - A).astype(np.float64)
                    gp = np.maximum(np.maximum(A - pred, pred - B), 0.0).astype(np.float64)
                    D["n"] += 1
                    if int(np.argmin(np.abs(wr - wp))) == oi:
                        D["w_ok"] += 1
                    if int(np.argmin(gp)) == oi:
                        D["gap_ok"] += 1
                    if int(np.argmin(gp + 3.0 * np.abs(wr - wp))) == oi:
                        D["cost_ok"] += 1
                    DUMP.append((gp.astype(np.float32), wr.astype(np.float32),
                                 np.float32(wp), oi, k))
                    D["dw_own"].append(abs(wr[oi] - wp)); D["dw_pick"].append(abs(wr[k] - wp))
                    D["w_own"].append(wr[oi]); D["w_pick"].append(wr[k])
                    D["gap_own"].append(gp[oi]); D["gap_pick"].append(gp[k])
                    D["prior"].append(wp)
                if branch != "else" or oi is None:
                    return
                wp = float(np.median(hist[-WIN:]))
                wr = (B - A).astype(np.float64)
                gap = np.maximum(np.maximum(A - pred, pred - B), 0.0).astype(np.float64)
                S["else_own"] += 1
                S["n_cand"].append(len(A))
                if len(A) == 1:
                    S["one_cand"] += 1
                if k == oi:
                    S["base_ok"] += 1
                if int(np.argmin(np.abs(wr - wp))) == oi:
                    S["w_ok"] += 1
                if int(np.argmin(gap)) == oi:
                    S["gap_ok"] += 1
                S["dw_own"].append(abs(wr[oi] - wp)); S["dw_pick"].append(abs(wr[k] - wp))
                S["w_own"].append(wr[oi]); S["w_pick"].append(wr[k])
                S["gap_own"].append(gap[oi]); S["gap_pick"].append(gap[k])
                S["prior"].append(wp)

            BE.trace(rec, rec["runs"][best_c], H, probe=probe)
            print(f"   {sheet['well']:<12}{rec['short']:<8} med={best_med:7.1f}  "
                  f"накоплено else-строк со своим раном: {S['else_own']}")


run()
n = max(1, S["else_own"])
print(f"\n{'='*78}")
print(f"строк else, где СВОЙ ран БЫЛ: {S['else_own']}")
print(f"  кандидатов на строке: med {np.median(S['n_cand']):.0f}  "
      f"единственный кандидат {100.0*S['one_cand']/n:.1f}%")
print(f"\n  ДОЛЯ ПОПАДАНИЯ В СВОЙ РАН:")
print(f"    база (ближайший центр)      : {100.0*S['base_ok']/n:5.1f}%")
print(f"    только зазор до pred        : {100.0*S['gap_ok']/n:5.1f}%")
print(f"    ТОЛЬКО ширина (|w-wprior|)  : {100.0*S['w_ok']/n:5.1f}%")
for nm, a, b in (("|w - w_prior|", "dw_own", "dw_pick"),
                 ("ширина рана   ", "w_own", "w_pick"),
                 ("зазор до pred ", "gap_own", "gap_pick")):
    ao, ap_ = np.array(S[a]), np.array(S[b])
    print(f"\n  {nm}:  СВОЙ med {np.median(ao):6.1f} (p90 {np.percentile(ao,90):6.1f})   "
          f"ВЫБРАННЫЙ med {np.median(ap_):6.1f} (p90 {np.percentile(ap_,90):6.1f})")
print(f"\n  приор ширины w_prior: med {np.median(S['prior']):.1f}  "
      f"p10 {np.percentile(S['prior'],10):.1f}  p90 {np.percentile(S['prior'],90):.1f}")

print(f"\n{'='*78}\n★ ТОЛЬКО МОМЕНТЫ ИЗБЕЖНОГО УХОДА (шли по своей, свой ран был, взяли чужой)")
d = max(1, D["n"])
print(f"  таких моментов: {D['n']}    приор ширины med {np.median(D['prior']):.1f}")
print(f"  спас бы выбор по ширине     : {100.0*D['w_ok']/d:5.1f}%")
print(f"  спас бы выбор по зазору     : {100.0*D['gap_ok']/d:5.1f}%")
print(f"  спас бы зазор + 3*|dw|      : {100.0*D['cost_ok']/d:5.1f}%")
for nm, a, b in (("|w - w_prior|", "dw_own", "dw_pick"),
                 ("ширина рана   ", "w_own", "w_pick"),
                 ("зазор до pred ", "gap_own", "gap_pick")):
    ao, ap_ = np.array(D[a]), np.array(D[b])
    print(f"  {nm}:  СВОЙ med {np.median(ao):6.1f} (p90 {np.percentile(ao,90):6.1f})   "
          f"ВЗЯТЫЙ med {np.median(ap_):6.1f} (p90 {np.percentile(ap_,90):6.1f})")

# ─────────── оффлайн-грид по правилам выбора на моментах ухода ───────────
import pickle
with open(r"F:\nds\output\taskS\width_depart.pkl", "wb") as f:
    pickle.dump(DUMP, f, protocol=5)

def acc(rule):
    ok = 0
    for gp, wr, wp, oi, k in DUMP:
        if int(np.argmin(rule(gp, wr, float(wp)))) == oi:
            ok += 1
    return 100.0 * ok / max(1, len(DUMP))

print(f"\n★ ГРИД ПРАВИЛ на {len(DUMP)} моментах ухода (доля попаданий в СВОЙ ран):")
RULES = [("зазор (база-ish)",        lambda g, w, p: g),
         ("|w-p| симметрично",       lambda g, w, p: np.abs(w - p)),
         ("-ширина (шире=лучше)",     lambda g, w, p: -w),
         ("g + 1*max(0,p-w)",        lambda g, w, p: g + 1.0 * np.maximum(0, p - w)),
         ("g + 3*max(0,p-w)",        lambda g, w, p: g + 3.0 * np.maximum(0, p - w)),
         ("g + 6*max(0,p-w)",        lambda g, w, p: g + 6.0 * np.maximum(0, p - w)),
         ("g + 3*max(0,p-w) -0.5w",  lambda g, w, p: g + 3.0 * np.maximum(0, p - w) - 0.5 * w),
         ("g - 1.0*w",               lambda g, w, p: g - 1.0 * w),
         ("g - 2.0*w",               lambda g, w, p: g - 2.0 * w),
         ("g - 3.0*w",               lambda g, w, p: g - 3.0 * w),
         ("g/(1+w)",                 lambda g, w, p: g / (1.0 + w))]
for nm, r in RULES:
    print(f"   {nm:<26}{acc(r):5.1f}%")

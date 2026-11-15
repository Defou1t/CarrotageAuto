r"""_segpath_var.py — §6.372 СТЕНД: ВАРИАНТЫ ИЗВЛЕЧЕНИЯ ПУТЕЙ ПО КАРТЕ СЕТИ ТУШИ (15.11).

§6.371: сеть туши видит 88% линий, которых нет в пуле кандидатов, — значит, их теряет извлечение путей (`rowdec.trace_track` по карте сети,
§6.349): пики строки ≥ pthr · max строки, K путей жадно (взятые пики ±3 px исключаются), штраф скачка wjump · |Δx| / dy. Варианты:
  E — как §6.349 (pthr 0.6, K, wjump 0.15) — контроль;  A — pthr 0.3;  B — K + 2;  C — wjump 0.05;  D — pthr 0.3, K + 2, wjump 0.05.
Карта — сеть вне фолда (`ink_seg_f`: модель фолда скважины, иначе all.pt), эмбеддинги личности — декодер прода; кэш-источник — `tcache_inert`
(поле) / `tcache_out_inert` (вне). Счёт: у слотов без честного кандидата в прод-пуле (`line_choice_r` / `_out_r`) — есть ли честный путь
варианта (на плоскости ≤ 3 px, покрытие ≥ 0.9); число путей на лист (шум для выбора трассы).

  _segpath_var.py paths --sheets <список> [--vars E A B C D]    → taskS/segpath_var/<V>/<лист>.pkl (продолжаемо)
  _segpath_var.py ceil  --sheets <список>
"""
import sys, io, argparse, contextlib, pickle, hashlib, time, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
OUT = TS / "segpath_var"
SEG = TS / "ink_seg_f"
VARS = {"E": (0.6, 0, 0.15), "A": (0.3, 0, 0.15), "B": (0.6, 2, 0.15), "C": (0.6, 0, 0.05), "D": (0.3, 2, 0.05),
        "F": (0.6, 0, 0.5), "G": (0.6, 0, 1.5), "H": (0.6, 2, 0.5),           # F–H — §6.373: сильнее штраф скачка
        "I": (0.6, 0, 0.15), "J": (0.6, 2, 0.15),                             # I–J — §6.374: удержание (HOLD)
        "K": (0.6, 0, 0.15)}                                                  # K — §6.375: без исключения занятых пиков (EXCL)
EXCL = {"K": -1}                # §6.375: `cv.rowdec_excl` (3 = прод) — путь может идти по пику, взятому прежним путём
HOLD = {"I": 0.3, "J": 0.3}     # §6.374: `cv.rowdec_hold` — путь может не брать пик, если рядом доступного нет (§6.237), вместо прыжка
WM = str(TS / "rowdec_wellmap.json").replace("\\", "/")
TRACE_KNOBS = dict(row_decoder="auto5", rowdec_wellmap=WM, rowdec_slot_all=True, rowdec_slot_len=0.18, rowdec_ink_seg="")


def sheet_dir(n):
    return f"{n.stem[:40]}_{hashlib.md5(n.stem.encode('utf-8')).hexdigest()[:8]}"


def sheets_of(f):
    want = [l.strip() for l in Path(f).read_text(encoding="utf-8").splitlines() if l.strip()]
    src = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    return [src[w] for w in want if w in src]


def pack(tr):
    return (np.fromiter(tr.keys(), np.int64, len(tr)), np.fromiter(tr.values(), np.float64, len(tr)))


def cmd_paths(a):
    from types import SimpleNamespace
    from auto import rowdec as RD, imaging as im, frame as FR, ink_seg as IS
    im.EXIF_TRANSPOSE = False                       # кэши стендов — по скану без разворота EXIF
    from auto.config import Config
    folds = json.loads((SEG / "folds.json").read_text(encoding="utf-8"))
    orig = RD.trace_track
    T0 = time.time(); done = 0
    for n in sheets_of(a.sheets):
        todo = [v for v in a.vars if not (OUT / v / (sheet_dir(n) + ".pkl")).exists()]
        if not todo:
            continue
        src = next((TS / d / (sheet_dir(n) + ".pkl") for d in ("tcache_inert", "tcache_out_inert") if (TS / d / (sheet_dir(n) + ".pkl")).exists()), None)
        if src is None:
            print(f"  {n.stem[:44]}: нет кэша"); continue
        c = pickle.load(open(src, "rb"))
        knobs = dict(c["trace_knobs"]); knobs.update({k: v for k, v in TRACE_KNOBS.items() if k not in knobs or k == "rowdec_ink_seg"})
        fo = folds.get(n.parent.parent.name)
        sp = str(SEG / (f"f{fo}.pt" if fo is not None else "all.pt"))
        rgb = im.load_rgb(c["image"])
        if getattr(c["frame"], "row_shift", None) is not None:
            rgb = FR.apply_row_shift(rgb, c["frame"].row_shift)
        MAPS = {}

        def netmap(prob, y0_, x0_):
            key = (y0_, x0_, prob.shape)
            if key not in MAPS:
                sg = IS.prob_map(rgb, y0_, y0_ + prob.shape[0], x0_, x0_ + prob.shape[1], sp)
                if sg.ndim == 3:
                    sg = sg[0]
                if sg.shape != prob.shape:
                    s2 = np.zeros(prob.shape, np.float32); s2[:sg.shape[0], :sg.shape[1]] = sg; sg = s2
                MAPS.clear(); MAPS[key] = sg
            return MAPS[key]
        for v in todo:
            pthr, kplus, wj = VARS[v]
            cfg = Config()
            for k, val in knobs.items():
                setattr(cfg.cv, k, val)
            p = cfg.cv
            p.rowdec_ink_seg = ""; p.rowdec_extra_thr = 0.0; p.rowdec_extra_inert = ""; p.rowdec_extra_learn = ""
            p.rowdec_ink_segfam = ""; p.rowdec_peak_thr = pthr; p.rowdec_hold = HOLD.get(v, 0.0); p.rowdec_excl = EXCL.get(v, 3)
            sheet = SimpleNamespace(lines=c["lines"], frame=c["frame"], meta=c["meta"], k_slots=c["k_slots"])

            def tt(rgb_, track, k, p_, ck, y0, y1, conf_out=None, **kw):
                p_.rowdec_prob_mask = netmap
                try:
                    return orig(rgb_, track, k + kplus, p_, ck, y0, y1, wjump=wj, conf_out=conf_out)
                finally:
                    p_.rowdec_prob_mask = None
            RD.trace_track = tt
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    alt = RD.trace_auto(rgb, sheet, p)
            except Exception as e:
                print(f"  {n.stem[:44]} {v} ПАДЕНИЕ {type(e).__name__}: {e}"); alt = None
            finally:
                RD.trace_track = orig
            d = OUT / v; d.mkdir(parents=True, exist_ok=True)
            rec = dict(sheet=n.name, var=v, alt=None if alt is None else [(L, pack(tr)) for L, tr in alt if tr])
            tmp = d / (sheet_dir(n) + ".tmp"); pickle.dump(rec, open(tmp, "wb"), protocol=4); tmp.replace(d / (sheet_dir(n) + ".pkl"))
        done += 1
        del rgb
        if done % 5 == 0:
            print(f"  листов {done}, {time.time() - T0:.0f} с")
    print(f"★ PATHS: листов {done}")


def cmd_ceil(a):
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from _cand_ceiling import _tree, dense_tr
    from _line_choice_scan import honest_d
    lab = {}
    for d in ("line_choice_r", "line_choice_out_r"):
        for f in (TS / d).glob("*.pkl"):
            r = pickle.load(open(f, "rb"))
            lab[r["sheet"]] = {s["name"]: any(x.get("lab") and x["lab"][0] for x in s["cands"]) for s in r["slots"] if s.get("has_truth")}
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    res = {v: dict(nh=0, nh_hit=0, h=0, h_hit=0, paths=0, sheets=0) for v in a.vars}
    for n in sheets_of(a.sheets):
        if n.name not in lab:
            continue
        P = {}
        for v in a.vars:
            f = OUT / v / (sheet_dir(n) + ".pkl")
            if f.exists():
                r = pickle.load(open(f, "rb"))
                P[v] = [dense_tr(un(t)) for L, t in (r["alt"] or [])]
        if len(P) < len(a.vars):
            continue
        mt = extract(str(n))
        G = {c["name"]: dense(c) for c in mt["curves"]}
        TP = {v: [_tree(d) for d in P[v]] for v in P}
        for nm, had in lab[n.name].items():
            g = G.get(nm)
            if not g or len(g) < 50:
                continue
            tg = _tree(g)
            for v in a.vars:
                hit = any(honest_d(P[v][k], TP[v][k], g, tg)[0] for k in range(len(P[v])) if len(P[v][k]) >= 50)
                key = "h" if had else "nh"
                res[v][key] += 1; res[v][key + "_hit"] += int(hit)
        for v in a.vars:
            res[v]["paths"] += len(P[v]); res[v]["sheets"] += 1
    print("★ варианты путей по карте сети: у слотов без честного кандидата в пуле — доля с честным путём варианта")
    for v in a.vars:
        r = res[v]; pthr, kp, wj = VARS[v]
        print(f"   {v} (pthr {pthr}, K+{kp}, wjump {wj}{f', hold {HOLD[v]}' if v in HOLD else ''}{', без исключения' if v in EXCL else ''}): без честного {r['nh_hit']}/{r['nh']} = {r['nh_hit'] / max(r['nh'], 1):.1%}; "
              f"с честным {r['h_hit']}/{r['h']} = {r['h_hit'] / max(r['h'], 1):.1%}; путей на лист {r['paths'] / max(r['sheets'], 1):.1f}")


def cmd_diag(a):
    """почему путей нет: у слотов без честного кандидата — лучший одиночный путь (покрытие при медиане ≤ 3 / ≤ 6 px) и объединённое покрытие
    всех путей варианта (строки эталона, где какой-то путь в ≤ 3 px на плоскости)"""
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from _cand_ceiling import _tree, dense_tr
    from _line_choice_scan import honest_d
    lab = {}
    for d in ("line_choice_r", "line_choice_out_r"):
        for f in (TS / d).glob("*.pkl"):
            r = pickle.load(open(f, "rb"))
            lab[r["sheet"]] = {s["name"]: any(x.get("lab") and x["lab"][0] for x in s["cands"]) for s in r["slots"] if s.get("has_truth")}
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    R = {v: [] for v in a.vars}
    for n in sheets_of(a.sheets):
        if n.name not in lab or all(lab[n.name].values()):
            continue
        P = {}
        for v in a.vars:
            f = OUT / v / (sheet_dir(n) + ".pkl")
            if f.exists():
                P[v] = [dense_tr(un(t)) for L, t in (pickle.load(open(f, "rb"))["alt"] or [])]
        if len(P) < len(a.vars):
            continue
        G = {c["name"]: dense(c) for c in extract(str(n))["curves"]}
        for nm, had in lab[n.name].items():
            g = G.get(nm)
            if had or not g or len(g) < 50:
                continue
            tg = _tree(g); ys = np.array(sorted(g)); Q = np.array([[y, g[y]] for y in ys], float)
            for v in a.vars:
                c3 = c6 = 0.0; near = np.zeros(len(ys), bool)
                for p in P[v]:
                    if len(p) < 50:
                        continue
                    tp = _tree(p)
                    _, cov, m = honest_d(p, tp, g, tg)
                    if m <= 3: c3 = max(c3, cov)
                    if m <= 6: c6 = max(c6, cov)
                    d, _ = tp.query(Q); near |= d <= 3
                R[v].append((c3, c6, float(near.mean())))
    print("★ слоты без честного кандидата: лучший путь варианта и объединение путей")
    for v in a.vars:
        X = np.array(R[v]) if R[v] else np.zeros((0, 3))
        if not len(X):
            continue
        print(f"   {v}: n {len(X)}; одиночный ≤3 px: покр. ≥ 0.9 {np.mean(X[:, 0] >= 0.9):.0%}, ≥ 0.7 {np.mean(X[:, 0] >= 0.7):.0%}, медиана {np.median(X[:, 0]):.2f}"
              f" | ≤6 px: ≥ 0.9 {np.mean(X[:, 1] >= 0.9):.0%}, ≥ 0.7 {np.mean(X[:, 1] >= 0.7):.0%}"
              f" | объединение ≤3 px: ≥ 0.9 {np.mean(X[:, 2] >= 0.9):.0%}, ≥ 0.7 {np.mean(X[:, 2] >= 0.7):.0%}, медиана {np.median(X[:, 2]):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["paths", "ceil", "diag"])
    ap.add_argument("--sheets", required=True)
    ap.add_argument("--vars", nargs="+", default=list(VARS))
    a = ap.parse_args()
    {"paths": cmd_paths, "ceil": cmd_ceil, "diag": cmd_diag}[a.cmd](a)


if __name__ == "__main__":
    main()

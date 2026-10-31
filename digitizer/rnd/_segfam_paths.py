r"""_segfam_paths.py — ПУТИ ПО КАНАЛУ СВОЕГО СЕМЕЙСТВА СЕТИ ТУШИ ПО СЕМЕЙСТВАМ (31.10, §6.351, разведка).

§6.349: пути декодера по карте сети туши («чья-то оцифрованная кривая») дали честный путь половине слотов без честного кандидата.
Одна карта на все семейства ведёт путь по чужой кривой там, где кривые сходятся и пересекаются. Сеть по семействам (§6.350,
`ink_segfam_f`, канал 1 + k — семейство `FAMS[k]`) даёт карту на каждое семейство. Здесь для каждого семейства F среди кривых рамки
листа (имена `frame_nlgx` без DA — их знает и прод) — пути `rowdec.trace_track` по каналу F (карта сети вместо карты декодера, как
m < 0 у §6.349), K = min(K трека, число слотов F на листе + `--kplus`); эмбеддинги личности — декодера, привязка к линиям — прод
(`rowdec.trace_auto`). Сеть — вне фолда (скважина листа не в обучении). Прод-код не меняется: `trace_track` подменяется здесь.

  paths — кэш как у `redec` (`alt` — пути по каналам семейств, `alt_fam` — семейство каждого пути, `extra` = None).
  ceil  — потолок на листах `--sheets` против пула `line_choice_r`: у слотов без честного кандидата — есть ли честный путь (plane2
          ≤ 3 px, покрытие ≥ 0.9) среди путей кэшей `--new`; `--check` — кэши текущего пула (сверка счёта: у слотов «без честного»
          их пути честными быть не должны).

  _segfam_paths.py paths --sheets orc_sheets.txt --src F:/nds/output/taskS/tcache_inert --seg F:/nds/output/taskS/ink_segfam_f \
        --cache F:/nds/output/taskS/tcache_segfam --shard 0/2
  _segfam_paths.py ceil --sheets orc_sheets.txt --new "F:/nds/output/taskS/tcache_segfam|alt" \
        --check "F:/nds/output/taskS/tcache_seg|alt;F:/nds/output/taskS/tcache_seg|extra;F:/nds/output/taskS/tcache_segraw_f|extra"
"""
import sys, io, argparse, contextlib, pickle, hashlib, time, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["paths", "ceil"])
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--sheets", required=True)
ap.add_argument("--src", default=r"F:/nds/output/taskS/tcache_inert")
ap.add_argument("--seg", default=r"F:/nds/output/taskS/ink_segfam_f", help="каталог сетей по фолдам (folds.json, f*.pt, all.pt)")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache_segfam")
ap.add_argument("--kplus", type=int, default=0, help="путей на семейство сверх числа его слотов")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--max-hours", type=float, default=0.0)
ap.add_argument("--pool", default=r"F:/nds/output/taskS/line_choice_r")
ap.add_argument("--new", default="", help="ceil: кэши «каталог|поле;…» — новые пути")
ap.add_argument("--check", default="", help="ceil: кэши текущего пула — сверка счёта")
a = ap.parse_args()
TS = Path(a.ts); T0 = time.time()
WM = str(TS / "rowdec_wellmap.json").replace("\\", "/")
TRACE_KNOBS = dict(row_decoder="auto5", rowdec_wellmap=WM, rowdec_slot_all=True, rowdec_slot_len=0.18, rowdec_ink_seg="")   # как redec


def sheet_dir(n):
    return f"{n.stem[:40]}_{hashlib.md5(n.stem.encode('utf-8')).hexdigest()[:8]}"


def sheet_list():
    p = Path(a.sheets) if Path(a.sheets).is_absolute() else TS / a.sheets
    want = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    src = {}
    for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx"):
        src.setdefault(q.name, q)
    got = sorted([src[w] for w in want if w in src], key=lambda q: q.name)
    i, n = map(int, a.shard.split("/"))
    return got[i::n] if n > 1 else got


def pack(tr):
    return (np.fromiter(tr.keys(), np.int64, len(tr)), np.fromiter(tr.values(), np.float64, len(tr)))


def cmd_paths():
    from types import SimpleNamespace
    from auto import rowdec as RD, trace2d as T2, imaging as im, frame as FR, meta as Mm, ink_seg as IS
    from auto.config import Config
    from extract_nlgx import extract
    folds = json.loads((Path(a.seg) / "folds.json").read_text(encoding="utf-8"))
    CACHE = Path(a.cache); CACHE.mkdir(parents=True, exist_ok=True)
    orig = RD.trace_track
    done = skip = fail = 0
    for n in sheet_list():
        dst = CACHE / (sheet_dir(n) + ".pkl"); src = Path(a.src) / (sheet_dir(n) + ".pkl")
        if dst.exists():
            skip += 1; continue
        if not src.exists():
            fail += 1; print(f"  {n.stem[:44]} нет в исходном кэше"); continue
        c = pickle.load(open(src, "rb"))
        knobs = dict(c["trace_knobs"]); knobs.update({k: v for k, v in TRACE_KNOBS.items() if k not in (
            "row_decoder", "rowdec_wellmap", "rowdec_slot_all", "rowdec_slot_len") or k not in knobs})
        cfg = Config()
        for k, v in knobs.items():
            setattr(cfg.cv, k, v)
        p = cfg.cv
        p.rowdec_ink_seg = ""; p.rowdec_extra_thr = 0.0; p.rowdec_extra_inert = ""; p.rowdec_extra_learn = ""   # только пути семейств
        sheet = SimpleNamespace(lines=c["lines"], frame=c["frame"], meta=c["meta"], k_slots=c["k_slots"])
        fo = folds.get(n.parent.parent.name)
        sp = str(Path(a.seg) / (f"f{fo}.pt" if fo is not None else "all.pt"))
        try:
            FAM = Counter(IS.fam_index(Mm.mnem_root(cv["name"])) for cv in extract(str(c["frame_nlgx"]))["curves"]
                          if Mm.mnem_root(cv["name"]) != "DA")
            rgb = im.load_rgb(c["image"])
            if getattr(c["frame"], "row_shift", None) is not None:
                rgb = FR.apply_row_shift(rgb, c["frame"].row_shift)
            MAPS = {}; FID = {}

            def mk(F):
                def _m(prob, y0_, x0_):
                    key = (y0_, x0_, prob.shape)
                    if key not in MAPS:
                        sg = IS.prob_map(rgb, y0_, y0_ + prob.shape[0], x0_, x0_ + prob.shape[1], sp)
                        if sg.shape[1:] != prob.shape:
                            sg2 = np.zeros((sg.shape[0],) + prob.shape, np.float32); sg2[:, :sg.shape[1], :sg.shape[2]] = sg; sg = sg2
                        MAPS.clear(); MAPS[key] = {F_: sg[1 + F_].copy() for F_ in FAM}   # одна полоса трека за раз — память
                        del sg
                    return MAPS[key][F]
                return _m

            def tt(rgb_, track, k, p_, ck, y0, y1, conf_out=None, **kw):
                out = []
                for F, nF in sorted(FAM.items()):
                    p_.rowdec_prob_mask = mk(F)
                    for t in orig(rgb_, track, max(1, min(k, nF + a.kplus)), p_, ck, y0, y1, conf_out=conf_out):
                        FID[id(t)] = F; out.append(t)
                p_.rowdec_prob_mask = None
                return out

            RD.trace_track = tt
            alt = None
            if getattr(p, "row_decoder", "") and FAM:
                with contextlib.redirect_stdout(io.StringIO()):
                    alt = RD.trace_auto(rgb, sheet, p)
        except Exception as e:
            fail += 1; print(f"  {n.stem[:44]} ПАДЕНИЕ {type(e).__name__}: {e}"); continue
        finally:
            RD.trace_track = orig
        new = dict(c)
        new["alt"] = None if alt is None else [(L, pack(tr)) for L, tr in alt]
        new["alt_fam"] = None if alt is None else [FID.get(id(tr)) for L, tr in alt]
        new["alt_conf"] = getattr(alt, "conf", None) if alt is not None else None
        new["extra"] = None
        new["segfam"] = dict(seg=sp, fams=dict(FAM), kplus=a.kplus)
        tmp = dst.with_suffix(".tmp"); pickle.dump(new, open(tmp, "wb"), protocol=4); tmp.replace(dst)
        done += 1
        del rgb
        if done % 10 == 0:
            print(f"  сделано {done}, было {skip}, ошибок {fail}, {time.time() - T0:.0f} с")
        if a.max_hours and (time.time() - T0) / 3600 > a.max_hours:
            print("★ ПАРТИЯ ОКОНЧЕНА — выхожу кодом 75"); sys.exit(75)
    print(f"★ PATHS: сделано {done}, было {skip}, ошибок {fail}")
    if fail:
        sys.exit(3)


def cmd_ceil():
    import glob
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from _cand_ceiling import dense_tr
    from scipy.spatial import cKDTree
    from auto import ink_seg as IS
    globals()["cKDTree"] = cKDTree                                           # tree / err из lvl_loss.py ищут его в модуле
    src_ = open(Path(__file__).parent / "analysis_2026-10-20" / "lvl_loss.py", encoding="utf-8").read()
    exec(src_[src_.index("def tree"):src_.index("def lv_rows")], globals())  # tree, err
    HON = lambda e: e[0] <= 3.0 and e[1] >= 0.9
    SH = {q.name for q in sheet_list()}
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    sets = {nm: [x.rsplit("|", 1) if "|" in x else (x, "alt") for x in spec.split(";") if x.strip()]
            for nm, spec in (("new", a.new), ("check", a.check)) if spec}
    C = Counter(); FAMN = Counter(); SAME = Counter()
    for f in sorted(glob.glob(str(Path(a.pool) / "*.pkl"))):
        rec = pickle.load(open(f, "rb")); sh = rec["sheet"]
        if sh not in SH:
            continue
        q = SRC[sh]; key = sheet_dir(q)
        paths = {}; fams = {}
        try:
            for nm, lst in sets.items():
                paths[nm] = []; fams[nm] = []
                for d_, fld in lst:
                    c = pickle.load(open(Path(d_) / (key + ".pkl"), "rb"))
                    v = c.get(fld) or []
                    paths[nm] += [dense_tr(un(t)) for L, t in v]
                    fams[nm] += (c.get(fld + "_fam") or [None] * len(v))
        except FileNotFoundError:
            C["листов без кэша"] += 1; continue
        C["листов"] += 1
        gts = {cc["name"]: cc for cc in extract(str(q))["curves"]}
        for s in rec["slots"]:
            if not s["has_truth"] or s["name"] not in gts:
                continue
            g = dense(gts[s["name"]])
            had = any(sc["lab"] is not None and sc["lab"][0] for sc in s["cands"])
            grp = "с честным" if had else "без честного"
            C[(grp, "слотов")] += 1
            need = max(30, 0.9 * len(g))                  # тот же порог, что в err / HON: общих строк < 30 или покрытие < 0.9 — не честный
            for nm in paths:
                hon = [i for i, d in enumerate(paths[nm]) if d and sum(1 for y in d if y in g) >= need and HON(err(d, g))]
                C[(grp, nm)] += bool(hon)
                if nm == "new" and hon and not had:
                    FAMN[s["root"]] += 1
                    SAME["своего семейства" if any(fams[nm][i] == IS.fam_index(s["root"]) for i in hon) else "чужого"] += 1
    print(f"★ листов {C['листов']} (без кэша {C['листов без кэша']})")
    for grp in ("без честного", "с честным"):
        n0 = C[(grp, "слотов")]
        for nm in sets:
            print(f"   слоты {grp} кандидата в пуле ({n0}): пути «{nm}» — честный есть у {C[(grp, nm)]} ({C[(grp, nm)] / max(1, n0):.1%})")
    print("   новые честные по семействам:", dict(FAMN.most_common(12)))
    print("   канал пути:", dict(SAME))


{"paths": cmd_paths, "ceil": cmd_ceil}[a.cmd]()

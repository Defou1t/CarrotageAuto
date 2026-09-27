r"""_dec_joint.py — ВЫБОР K ПУТЕЙ ДЕКОДЕРА: ЖАДНЫЙ ПРОД ПРОТИВ ВАРИАНТОВ НА ТЕХ ЖЕ КАРТАХ (§6.237, 27.09).

`_dec_see.py`: карта декодера видит кривые без кандидата (пик в 3 px у 69% строк при 0.6·max, 78% при 0.3), а объединение
всех трасс держит 62% ⇒ предел — выбор путей. Прод (`rowdec.trace_track`): K путей ПО ОЧЕРЕДИ, Витерби по пикам, взятые
пики (±3 px) исключены, в каждой строке с доступным пиком путь ОБЯЗАН взять пик (пропуска нет) — на пересечении, занятом
первым путём, второй прыгает на чужую кривую. Здесь на одних и тех же картах трека (тот же чекпойнт по скважине):
  V0  — копия прод-декодирования (ПАРИТЕТ: трассы обязаны совпасть с декодером в кэше);
  H   — Витерби с состоянием УДЕРЖАНИЯ: путь может не брать пик строки (штраф `--wskip` за строку, не дольше `--gmax`
        строк подряд) и продолжить с той же x; строки удержания в трассу не пишутся (метрика мостит ≤ 30 строк);
  порог пиков 0.6 / 0.3.
Счёт: честные кривые трека 1:1 (медиана ≤ 3 px, покрытие ≥ 0.9 с мостом ≤ 30 строк) по вариантам; разрез — кривые, у
которых в кэше честного кандидата не было.

  _dec_joint.py --every 9 --wskip 0.5 --gmax 30
"""
import sys, argparse, pickle, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M, imaging as im, rowdec as RD
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--every", type=int, default=9)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--wskip", type=float, nargs="+", default=[0.5])
ap.add_argument("--gmax", type=int, default=30)
ap.add_argument("--rounds", type=int, default=0, help="кругов перекладки путей (0 — без варианта)")
ap.add_argument("--hjump", type=float, default=0.15, help="цена прыжка В ВАРИАНТАХ С УДЕРЖАНИЕМ (V0 — всегда прод 0.15)")
ap.add_argument("--kplus", type=int, default=0, help="вариант «удержание W + K+N путей» (0 — без варианта)")
ap.add_argument("--diag", action="store_true", help="по НЕвзятым вариантом кривым — медиана и покрытие лучшего пути")
ap.add_argument("--near", type=float, default=3.0, help="радиус «своя кривая есть»: удержание, если пиков нет в near·dy + near px")
ap.add_argument("--only", default="", help="считать только варианты, чьё имя содержит эту подстроку (V0 всегда)")
ap.add_argument("--dump", default=r"F:/nds/output/taskS/dec_joint.pkl")
ap.add_argument("--parity-cache", default="", help="сверка: кэш, собранный с `--knob rowdec_hold=W` (W = первый --wskip) — трассы декодера обязаны совпасть с вариантом «удержание W (0.6)»")
a = ap.parse_args()
TS = Path(a.ts); P = DEFAULT.cv
WMAP = json.loads((TS / "rowdec_wellmap.json").read_text(encoding="utf-8"))
SRC = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx"):
    SRC.setdefault(q.stem, q)
WJUMP, WEMB, EMB_MIN_K = 0.15, 1.0, 3


def un(t):
    return dict(zip(t[0].tolist(), t[1].tolist()))


def hon(tr, gt, bridge=30):
    com = [y for y in gt if y in tr]
    if len(com) < 30 or np.median([abs(tr[y] - gt[y]) for y in com]) > 3.0:
        return False
    rs = np.array(sorted(tr)); ok = 0
    for y in gt:
        if y in tr:
            ok += 1; continue
        i = np.searchsorted(rs, y)
        if 0 < i < len(rs) and rs[i] - rs[i - 1] <= bridge:
            ok += 1
    return ok / len(gt) >= 0.9


def mc(tr, gt, bridge=30):
    """(медиана |Δ| по общим строкам, покрытие с мостом) — для разбора промахов."""
    com = [y for y in gt if y in tr]
    if len(com) < 30:
        return None, 0.0
    med = float(np.median([abs(tr[y] - gt[y]) for y in com]))
    rs = np.array(sorted(tr)); ok = 0
    for y in gt:
        if y in tr:
            ok += 1; continue
        i = np.searchsorted(rs, y)
        if 0 < i < len(rs) and rs[i] - rs[i - 1] <= bridge:
            ok += 1
    return med, ok / len(gt)


def match(rows, cols, ok):
    pair = {}

    def try_(r, seen):
        for c in cols:
            if not ok.get((r, c)) or c in seen:
                continue
            seen.add(c)
            if c not in pair or try_(pair[c], seen):
                pair[c] = r
                return True
        return False
    for r in rows:
        try_(r, set())
    return {r: c for c, r in pair.items()}


def maps(rgb, track, ck, y0, y1):
    """prob и embs трека — копия прохода окнами `rowdec.trace_track`."""
    import torch
    net, dev = RD._load(ck)
    x0, x1 = int(track.x_left), int(track.x_right) + 1
    band = RD._band(rgb, P, x0, x1)
    H, Wb = band.shape
    y0 = max(0, int(y0)); y1 = min(H, int(y1))
    if y1 - y0 < 64 or Wb < 16:
        return None
    prob = np.zeros((y1 - y0, Wb), np.float32); embs = None
    STEP, OV, WIN = 512, 64, 512
    xcuts = [(c, min(Wb, c + WIN)) for c in range(0, max(1, Wb - 1), WIN - 64)]
    for (cx0, cx1) in xcuts:
        for gy in range(y0, y1, STEP - 2 * OV):
            gy2 = min(y1, gy + STEP)
            if gy2 - gy < 32:
                continue
            sub = band[gy:gy2, cx0:cx1].astype(np.float32) / 255.0
            with torch.no_grad():
                pp, ee = net(torch.from_numpy(sub)[None, None].to(dev))
                pp = torch.sigmoid(pp)[0, 0].float().cpu().numpy()
                ee = ee[0].float().cpu().numpy()
            if embs is None:
                embs = np.zeros((ee.shape[0], y1 - y0, Wb), np.float32)
            v0 = gy + (OV if gy > y0 else 0); v1 = gy2 - (OV if gy2 < y1 else 0)
            wx0 = cx0 + (32 if cx0 > 0 else 0); wx1 = cx1 - (32 if cx1 < Wb else 0)
            if wx1 <= wx0:
                wx0, wx1 = cx0, cx1
            prob[v0 - y0:v1 - y0, wx0:wx1] = pp[v0 - gy:v1 - gy, wx0 - cx0:wx1 - cx0]
            embs[:, v0 - y0:v1 - y0, wx0:wx1] = ee[:, v0 - gy:v1 - gy, wx0 - cx0:wx1 - cx0]
    if embs is None:
        return None
    return prob, embs, x0, y0


def peaks_of(prob, k, pthr):
    out = []
    for i in range(prob.shape[0]):
        row = prob[i]
        thr = pthr * float(row.max())
        idx = np.where((row >= thr) & (row >= np.roll(row, 1)) & (row >= np.roll(row, -1)))[0]
        if len(idx) > 8 * k:
            idx = np.sort(idx[np.argsort(row[idx])[-8 * k:]])
        out.append(idx)
    return out


def protos(prob, embs, peaks, k):
    if k < EMB_MIN_K:
        return None
    pts = [(i, x) for i in range(0, prob.shape[0], 7) for x in peaks[i]]
    if len(pts) < k * 8:
        return None
    Mx = np.stack([embs[:, i, x] for i, x in pts])
    rng = np.random.default_rng(0)
    mu = Mx[rng.choice(len(Mx), k, replace=False)]
    for _ in range(12):
        lab = ((Mx[:, None, :] - mu[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (lab == j).any():
                mu[j] = Mx[lab == j].mean(0)
    return mu


def decode(prob, embs, k, x0, y0, pthr, wskip=None, gmax=30, rounds=0):
    """wskip=None — ТОЧНАЯ копия прод-Витерби; иначе — с состоянием удержания."""
    peaks = peaks_of(prob, k, pthr)
    mu = protos(prob, embs, peaks, k)
    out = []
    taken = [set() for _ in range(prob.shape[0])]
    for j in range(k):
        out.append(_one_path(j, prob, embs, peaks, mu, taken, k, x0, y0, wskip, gmax))  # noqa
    # ★ ПЕРЕКЛАДКА (координатный спуск): путь по очереди освобождает свои пики и прокладывается заново при занятых
    #   остальных — жадный порядок перестаёт быть окончательным
    for _ in range(rounds):
        for j in range(k):
            for yy, xx in out[j].items():
                taken[yy - y0].discard(int(round(xx - x0)))
            out[j] = _one_path(j, prob, embs, peaks, mu, taken, k, x0, y0, wskip, gmax)
    return out


def _one_path(j, prob, embs, peaks, mu, taken, k, x0, y0, wskip, gmax):
    if True:
        rows, cands, locs = [], [], []
        for i in range(prob.shape[0]):
            idx = np.array([x for x in peaks[i] if not any(abs(x - t) <= 3 for t in taken[i])], int)
            if not len(idx):
                continue
            loc = -np.log(np.clip(prob[i, idx], 1e-6, 1.0))
            if mu is not None:
                E = np.stack([embs[:, i, x] for x in idx])
                loc = loc + WEMB * np.sqrt(((E - mu[j]) ** 2).sum(-1))
            rows.append(i); cands.append(idx.astype(float)); locs.append(loc)
        if len(rows) < 30:
            return {}
        if wskip is None:
            dp = [locs[0]]; bp = [np.full(len(cands[0]), -1, int)]
            for t in range(1, len(rows)):
                dy = max(1, rows[t] - rows[t - 1])
                jump = np.abs(cands[t][:, None] - cands[t - 1][None, :]) / dy
                tot = dp[t - 1][None, :] + WJUMP * jump
                arg = tot.argmin(1)
                dp.append(locs[t] + tot[np.arange(len(cands[t])), arg]); bp.append(arg)
            tr, s = {}, int(np.argmin(dp[-1]))
            for t in range(len(rows) - 1, -1, -1):
                x = int(cands[t][s])
                tr[rows[t] + y0] = float(x + x0); taken[rows[t]].add(x)
                s = int(bp[t][s])
                if s < 0:
                    break
            return tr
        # ── Витерби с удержанием: состояния строки = пики (n_t) + удержания (копии состояний прошлой строки с той же x).
        #    Удержание: +wskip за строку, счётчик ≤ gmax; удержаний держим не больше H лучших, чтобы не росло.
        Hmax = max(2 * k, 6)
        X = cands[0].copy(); D = locs[0].copy(); G = np.zeros(len(X), int); PK = np.ones(len(X), bool)
        hist = [(X, np.full(len(X), -1, int), PK)]
        for t in range(1, len(rows)):
            dy = max(1, rows[t] - rows[t - 1])
            jump = np.abs(cands[t][:, None] - X[None, :]) / dy
            tot = D[None, :] + a.hjump * jump
            arg = tot.argmin(1)
            Dp = locs[t] + tot[np.arange(len(cands[t])), arg]
            # удержание — только если своей кривой в строке нет (ни одного доступного пика в 3·dy + 3 px от позиции):
            #   слабый, но живой пик своей кривой обязан быть взят (иначе путь «удерживается» на бледных участках)
            near = np.abs(cands[t][None, :] - X[:, None]).min(1) if len(cands[t]) else np.full(len(X), np.inf)
            hold_ok = np.flatnonzero((G + dy <= gmax) & (near > a.near * dy + a.near))
            if len(hold_ok):
                hc = D[hold_ok] + wskip * dy
                keep = hold_ok[np.argsort(hc)[:Hmax]]
                Xh = X[keep]; Dh = D[keep] + wskip * dy; Gh = G[keep] + dy
            else:
                keep = np.zeros(0, int); Xh = np.zeros(0); Dh = np.zeros(0); Gh = np.zeros(0, int)
            X = np.concatenate([cands[t], Xh]); D = np.concatenate([Dp, Dh])
            G = np.concatenate([np.zeros(len(cands[t]), int), Gh])
            PK = np.concatenate([np.ones(len(cands[t]), bool), np.zeros(len(Xh), bool)])
            hist.append((X, np.concatenate([arg, keep]), PK))
        # конец пути — только в состоянии пика
        endc = np.flatnonzero(hist[-1][2])
        s = int(endc[np.argmin(D[endc])]) if len(endc) else int(np.argmin(D))
        tr = {}
        for t in range(len(rows) - 1, -1, -1):
            Xt, bpt, pkt = hist[t]
            if pkt[s]:
                x = int(Xt[s]); tr[rows[t] + y0] = float(x + x0); taken[rows[t]].add(x)
            s = int(bpt[s])
            if s < 0:
                break
        return tr


VARS = [("V0 прод (0.6, жадный)", 0.6, None, 0), ("порог 0.3", 0.3, None, 0)]
for w in a.wskip:
    VARS += [(f"удержание {w} (0.6)", 0.6, w, 0), (f"удержание {w} + порог 0.3", 0.3, w, 0)]
    if a.rounds:
        VARS += [(f"удержание {w} + перекладка {a.rounds}", 0.6, w, a.rounds)]
    if a.kplus:
        VARS += [(f"удержание {w} + K+{a.kplus}", 0.6, w, -a.kplus)]     # отрицательный rr = добавка к K
C = Counter(); PAR = Counter(); REC = []; DIAG = []
files = sorted(Path(a.cache).glob("*.pkl"))[a.offset::a.every]
if a.parity_cache:
    files = [Path(a.cache) / f.name for f in sorted(Path(a.parity_cache).glob("*.pkl"))]
T0 = time.time()
for fi, f in enumerate(files, 1):
    v = pickle.load(open(f, "rb"))
    stem = Path(v["frame_nlgx"]).stem
    q = SRC.get(stem)
    w = WMAP.get(Path(v["image"]).stem) or WMAP.get(v["stem"]) or WMAP.get(stem)
    if not q or w is None or not v.get("alt"):
        continue
    fold = RD.fold_of_well(w)
    ck = f"rowdec_of5_f{0 if fold is None else fold}_s0.pt"
    G = {c["name"]: dense(c) for c in extract(str(q))["curves"]
         if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    T = [un(t) for _, t in v["traces"]] + [un(t) for _, t in v["alt"]]
    tracks = v["frame"].tracks
    by_t = defaultdict(list)
    for g, gt in G.items():
        gx = float(np.median(list(gt.values())))
        ti = next((i for i, tk in enumerate(tracks) if tk.x_left <= gx <= tk.x_right), None)
        if ti is not None:
            by_t[ti].append(g)
    rgb = None
    for ti, gs in by_t.items():
        lines = [L for L, _ in v["traces"] if L.track_index == ti]
        alt = [un(t) for L, t in v["alt"] if L.track_index == ti]
        if not alt:
            continue
        K = max(len(lines), len(alt))
        if lines:
            y0 = min(L.y0 for L in lines); y1 = max(L.y1 for L in lines)
        else:
            y0, y1 = int(v["frame"].top_y), int(v["frame"].bottom_y)
        if rgb is None:
            rgb = im.load_rgb(v["image"])
        mp = maps(rgb, tracks[ti], ck, y0, y1)
        if mp is None:
            continue
        prob, embs, x0, py0 = mp
        cand_before = {g: any(hon(t, G[g]) for t in T) for g in gs}
        res = {}
        for name, pthr, ws, rr in VARS:
            if a.only and a.only not in name and not name.startswith("V0"):
                res[name] = set(); continue
            trs = [t for t in decode(prob, embs, K + max(0, -rr), x0, py0, pthr, ws, a.gmax, max(0, rr)) if len(t) >= 30]
            ok = {(g, i): hon(t, G[g]) for g in gs for i, t in enumerate(trs)}
            mt = match(gs, list(range(len(trs))), ok)
            res[name] = set(mt)
            if a.diag and ws is not None:
                for g in gs:
                    if g in mt:
                        continue
                    best = None
                    for t in trs:
                        m_, c_ = mc(t, G[g])
                        if m_ is None:
                            continue
                        # лучший — ближайший по медиане среди путей, державших ≥ 30 общих строк
                        if best is None or m_ < best[0]:
                            best = (m_, c_)
                    DIAG.append((name, cand_before[g], best))
            if a.parity_cache and ws == a.wskip[0] and pthr == 0.6 and rr == (-a.kplus if a.kplus else 0):
                pc = pickle.load(open(Path(a.parity_cache) / f.name, "rb"))
                palt = [un(t) for L, t in pc["alt"] if L.track_index == ti]
                PAR["трасс сверочного кэша"] += len(palt)
                PAR["совпали с удержанием"] += sum(1 for t in palt if any(t == u for u in trs))
            if name.startswith("V0"):
                # ПАРИТЕТ: каждая трасса декодера из кэша обязана найтись среди V0 побайтно
                PAR["трасс кэша"] += len(alt)
                PAR["совпали с V0"] += sum(1 for t in alt if any(t == u for u in trs))
        for name, _, _, _ in VARS:
            C[(name, "все")] += len(res[name])
            C[(name, "без кандидата")] += sum(1 for g in res[name] if not cand_before[g])
        C["кривых"] += len(gs); C["без кандидата"] += sum(1 for g in gs if not cand_before[g])
        REC.append((f.stem, ti, gs, {k2: sorted(v2) for k2, v2 in res.items()}, cand_before))
    del rgb
    if fi % 10 == 0:
        print(f"  … {fi}/{len(files)} [{time.time() - T0:.0f} с]", file=sys.stderr)
pickle.dump(REC, open(a.dump, "wb"))
print(f"★ ДЕКОДИРОВАНИЕ K ПУТЕЙ (каждый {a.every}-й лист кэша, сдвиг {a.offset}): кривых на треках с декодером {C['кривых']}, "
      f"из них без честного кандидата в кэше {C['без кандидата']}")
print(f"   ПАРИТЕТ V0 с декодером кэша: {PAR['совпали с V0']} из {PAR['трасс кэша']} трасс совпали побайтно")
if a.parity_cache:
    print(f"   ПАРИТЕТ прод-удержания со стендом: {PAR['совпали с удержанием']} из {PAR['трасс сверочного кэша']} трасс совпали побайтно")
base = C[(VARS[0][0], "все")]
for name, _, _, _ in VARS:
    if a.only and a.only not in name and not name.startswith("V0"):
        continue
    print(f"   {name:<32} честных 1:1 {C[(name, 'все')]:>5} ({C[(name, 'все')] - base:+d}); из кривых без кандидата — {C[(name, 'без кандидата')]}")
if a.diag:
    for name in sorted({d[0] for d in DIAG}):
        D = [d for d in DIAG if d[0] == name]
        nb = [d[2] for d in D if d[2] is not None]
        med = np.array([b[0] for b in nb]); cov = np.array([b[1] for b in nb])
        print(f"\n★ НЕ ВЗЯТЫ вариантом «{name}»: {len(D)} (без общего пути {len(D) - len(nb)})")
        if len(nb):
            print(f"   лучший путь: медиана ≤ 3 px, но покрытие < 0.9 — {int(np.sum((med <= 3) & (cov < 0.9)))}; "
                  f"медиана 3–10 px — {int(np.sum((med > 3) & (med <= 10)))}; 10–50 — {int(np.sum((med > 10) & (med <= 50)))}; "
                  f"> 50 — {int(np.sum(med > 50))}")
            m3 = (med <= 3) & (cov < 0.9)
            if m3.any():
                print(f"   у «рядом, но коротко»: покрытие — медиана {np.median(cov[m3]):.2f}")

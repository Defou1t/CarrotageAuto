r"""_rowdec_seam.py — ТЕРЯЕТ ЛИ ДЕКОДЕР КРИВУЮ НА ШВАХ ОКОН (§6.201 → B1b, первый кандидат).

ОТКУДА ВОПРОС. Агент-ревизор §6.200 нашёл в декодере расхождение обучения и инференса: обучающие
кропы 128×512 центрированы на паре эталонных кривых (`_rowdec_crops.py:80-83`), инференс режет
полосу трека равномерными окнами 512×512 с перекрытием 64 и сшивает, отрезая по 32 px с каждого
внутреннего края (`rowdec.py:228-247`). Треки в медиане 1185 px, 90% шире окна ⇒ швы по x есть
почти на каждом треке: `x_left + 480 + 448·k`. По y — `y0 + 448 + 384·k`. Поле зрения сети ±64 px
(дилатации 1..32), у пикселя на шве контекст с одной стороны обрезан; GroupNorm считает статистику
по всему окну, и окно при обучении вчетверо ниже.

ЧТО СЧИТАЕТ (по замороженным выдачам, счёта не тратит). Для каждой эталонной кривой честного поля
берётся её пара в выдаче по безымянному 1:1 (та же `match`, но с возвратом ПАР), и по каждой общей
строке считается попадание |dec − gt| ≤ 3px как функция расстояния от эталона до ближайшего шва по
x и по y. ★ КОНТРОЛЬ — прод (`ab_wellmap/A`): окон у него нет, и если его точность тоже проседает на
тех же местах, дело не в окнах, а в данных (шов совпал с линовкой, краем и т.п.).
⚠ Пары берутся ТОЛЬКО там, где 1:1 нашло пару (честную или нет — важно, что это та кривая):
кривые без пары в выдаче в счёт не идут, то есть стенд меряет ТОЧНОСТЬ ведения, не полноту.

  <ComfyUI>\python_embeded\python.exe _rowdec_seam.py
"""
import sys, argparse, pickle, hashlib, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--rows", default="pick_learn_honest2.pkl.dump")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--dec", default="ab_rdhonest/B")
ap.add_argument("--prod", default="ab_wellmap/A")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--cap", type=int, default=0)
a = ap.parse_args()
TS = Path(a.ts)
WIN, OVX, STEP, OVY = 512, 64, 512, 64          # ровно rowdec.py:228


def st(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    return float(np.median([abs(tr[y] - gt[y]) for y in com])), len(com) / max(1, len(gt))


def match_pairs(rows, cols, ok):
    """Та же механика, что `match` в `_name_cost_prod.py`, но возвращает ПАРЫ {эталон: колонка}.
    ⚠ Здесь `ok` — не «честна», а «есть ≥30 общих строк и med ≤ 50px»: нужна пара «та же кривая»,
    а не только честная, иначе точность у шва мерилась бы лишь на уже честных кривых."""
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


def _dir(root, sh):
    stem = Path(sh).stem
    return TS / root / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"


def read_out(root, sh):
    pd = _dir(root, sh)
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    if not got:
        return {}, None
    u = next(iter(sorted(pd.glob("*_understanding.json"))), None)
    fr = json.loads(u.read_text(encoding="utf-8"))["frame"] if u else None
    return ({c["name"]: dense(c) for c in extract(str(got))["curves"]
             if M.mnem_root(c["name"]) != "DA"}, fr)


trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]): int(r[3]["n_prod"]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q

BX = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 10**9)]
HIT = {"dec": defaultdict(Counter), "prod": defaultdict(Counter)}
HITY = {"dec": defaultdict(Counter), "prod": defaultdict(Counter)}
NTR = Counter()
sheets = sorted({r[0] for r in trk})
if a.cap:
    sheets = sheets[:a.cap]
done, miss = [], Counter()
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        miss["нет исходника"] += 1
        continue
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    bytr = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None:
            bytr[t].append(nm)
    WD, frD = read_out(a.dec, sh)
    WP, _ = read_out(a.prod, sh)
    if not WD or not WP or not frD:
        miss["⛔ пустая выдача/рамка"] += 1
        continue
    done.append(sh)
    tracks = frD["tracks"]
    for t, ns in bytr.items():
        if (sh, t) not in KOF or t >= len(tracks):
            continue
        xl, xr = int(tracks[t][0]), int(tracks[t][1])
        Wb = xr + 1 - xl
        # швы по x — ровно как их кладёт rowdec.py:229,243: окна с шагом 448, запись [c+32, c+480]
        seams_x = [xl + c + WIN - 32 for c in range(0, max(1, Wb - 1), WIN - OVX)
                   if c + WIN < Wb]
        # диапазон y декодера восстановить нельзя (y0 = min по линиям U1), поэтому швы по y
        # берутся от ВЕРХА выданной кривой: приближение, годное для сравнения с контролем
        for tag, W in (("dec", WD), ("prod", WP)):
            cols = [k for k in W if tm.get(k) == t]
            ok = {}
            for g in ns:
                for k in cols:
                    m, c = st(W[k], gts[g])
                    ok[(g, k)] = m is not None and m <= 50.0
            pairs = match_pairs(ns, cols, ok)
            NTR[tag] += 1
            NTR[tag + "_seam"] += bool(seams_x)
            for g, k in pairs.items():
                G, D = gts[g], W[k]
                ys = sorted(set(G) & set(D))
                if not ys:
                    continue
                ytop = min(D)
                seams_y = [ytop + STEP - OVY + j * (STEP - 2 * OVY) for j in range(0, 40)]
                for y in ys:
                    e = abs(D[y] - G[y])
                    dy = min(abs(y - s) for s in seams_y)
                    # ⚠ Треки УЖЕ окна швов не имеют; их строки в корзины по x не идут — иначе
                    #   «вдали от шва» смешивало бы узкие треки с широкими (смоук показал ×1.2).
                    if seams_x:
                        dx = min(abs(G[y] - s) for s in seams_x)
                        for lo, hi in BX:
                            if lo <= dx < hi:
                                HIT[tag][(lo, hi)]["n"] += 1
                                HIT[tag][(lo, hi)]["hit"] += (e <= 3)
                                break
                    for lo, hi in BX:
                        if lo <= dy < hi:
                            HITY[tag][(lo, hi)]["n"] += 1
                            HITY[tag][(lo, hi)]["hit"] += (e <= 3)
                            break
    if si % 200 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

_sk = sum(miss.values())
print(f"\nСВЕРКА СПИСКА: обработано {len(done)} + пропущено {_sk} = {len(done)+_sk} "
      f"против длины списка {len(sheets)}   {'★ СОШЛОСЬ' if len(done)+_sk == len(sheets) else '⛔ НЕ СОШЛОСЬ'}")
for k, v in miss.items():
    print(f"    «{k}»: {v}")
print(f"треков: декодер {NTR['dec']} (со швами по x {NTR['dec_seam']}), прод {NTR['prod']}")


def table(H, title, unit):
    print(f"\n★★ {title}")
    print(f"| расстояние до шва, {unit} | строк (дек) | ДЕКОДЕР попал ≤3px | строк (прод) | ПРОД попал ≤3px (контроль) |")
    print("|---|---|---|---|---|")
    for lo, hi in BX:
        d, p = H["dec"][(lo, hi)], H["prod"][(lo, hi)]
        lab = f"{lo}-{hi}" if hi < 10**9 else f"≥{lo}"
        print(f"| {lab} | {d['n']} | **{100*d['hit']/max(1,d['n']):.1f}%** | {p['n']} | "
              f"{100*p['hit']/max(1,p['n']):.1f}% |")


table(HIT, "ТОЧНОСТЬ ПОСТРОЧНО ПРОТИВ РАССТОЯНИЯ ДО ШВА ПО X (эталон → ближайший шов)", "px по x")
table(HITY, "ТО ЖЕ ПО Y (шов от верха выданной кривой — приближение)", "px по y")
d_near = sum(HIT["dec"][b]["hit"] for b in BX[:2]) / max(1, sum(HIT["dec"][b]["n"] for b in BX[:2]))
d_far = HIT["dec"][BX[-1]]["hit"] / max(1, HIT["dec"][BX[-1]]["n"])
p_near = sum(HIT["prod"][b]["hit"] for b in BX[:2]) / max(1, sum(HIT["prod"][b]["n"] for b in BX[:2]))
p_far = HIT["prod"][BX[-1]]["hit"] / max(1, HIT["prod"][BX[-1]]["n"])
print(f"\n★ ИТОГ ПО X: декодер у шва (<32px) {100*d_near:.1f}% против вдали (≥128px) {100*d_far:.1f}% "
      f"⇒ провал {100*(d_far-d_near):+.1f} п.п.; прод-контроль {100*p_near:.1f}% против {100*p_far:.1f}% "
      f"⇒ {100*(p_far-p_near):+.1f} п.п.")
print("   Разность разностей (декодер − контроль) — и есть цена шва:"
      f" {100*((d_far-d_near)-(p_far-p_near)):+.1f} п.п.")

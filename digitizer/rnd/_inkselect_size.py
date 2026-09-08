r"""_inkselect_size.py — ОТЛИЧИМА ЛИ ОЦИФРОВЫВАЕМАЯ ТУШЬ ОТ ЛИШНЕЙ ЛОКАЛЬНО (§6.192 → B1, шаг 1).

ОТКУДА ВОПРОС. §6.190: декодер ведёт настоящие линии, которых нет в эталоне. §6.192: пока эти линии
не отделены, любой сборщик тонет — 925 сегментов на кривую вместо десятков. ⇒ Первым шагом нужен
ОТБОР. Но прежде чем его строить, надо узнать, есть ли вообще сигнал: отличается ли тушь
оцифровываемой кривой от лишней ТАМ, ГДЕ РЕШЕНИЕ ПРИНИМАЕТСЯ.

★★ ГРАНУЛЯРНОСТЬ — РАН, а не штрих: ран и есть кандидат, из которых декодер выбирает шаг.
★★ ПРИЗНАКИ ТОЛЬКО ТЕ, ЧТО ДОСТУПНЫ В МОМЕНТ РЕШЕНИЯ (§6.146 — иначе замер обещает то, чего прод
не сможет повторить): ширина рана, темнота относительно бумаги СВОЕЙ строки, цвет, положение в
полосе трека, сколько ранов в строке, расстояние до ближайшего соседа, есть ли продолжение сверху
и снизу. Эталон используется ТОЛЬКО как метка, ни один признак его не видит.

МЕТКА: ран считается «оцифровываемым», если он ближе 3px к какой-нибудь эталонной кривой трека.

ЧТО ОТВЕЧАЕТ: AUC каждого признака и лучшей пары. AUC ≈ 0.5 — сигнала нет, отбор строить не из
чего, и B1 упирается в источник информации, а не в алгоритм. AUC ≥ 0.85 — отбор дешёв.

  <ComfyUI>\python_embeded\python.exe _inkselect_size.py --cap 20 --shard 0/4
  <ComfyUI>\python_embeded\python.exe _inkselect_size.py --sum
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from dataset_build import find_image
from auto import meta as M, imaging as im, trace2d as T
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--rows", default="pick_learn_honest2.pkl.dump")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--mink", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cap", type=int, default=20)
ap.add_argument("--step", type=int, default=10, help="брать каждую N-ю строку")
ap.add_argument("--maxw", type=int, default=120)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--tag", default="inksel")
ap.add_argument("--sum", action="store_true")
a = ap.parse_args()
TS = Path(a.ts)
COLORS = ["black", "red", "orange", "green", "blue"]
FEAT = ["ширина", "темнота", "чёрный", "цветной", "x в полосе", "ранов в строке",
        "до соседа", "есть сверху", "есть снизу", "длина опоры"]


def auc(x, y):
    """AUC = P(признак у положительного выше, чем у отрицательного). Считаем по рангам."""
    x = np.asarray(x, float); y = np.asarray(y, bool)
    if y.all() or not y.any():
        return 0.5
    r = np.argsort(np.argsort(x)) + 1.0
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


if a.sum:
    X, Y = [], []
    for p in sorted(TS.glob(f"{a.tag}_*of*.pkl")):
        d = pickle.load(open(p, "rb"))
        X.append(d["X"]); Y.append(d["Y"])
    X = np.vstack(X); Y = np.concatenate(Y)
    print(f"★★ ОТЛИЧИМА ЛИ ОЦИФРОВЫВАЕМАЯ ТУШЬ (ранов {len(Y)}, из них оцифрованных "
          f"{int(Y.sum())} = {100*Y.mean():.1f}%)")
    print("| признак | AUC | толкование |")
    aucs = []
    for j, nm in enumerate(FEAT):
        v = auc(X[:, j], Y)
        aucs.append((abs(v - 0.5), j, v))
        note = ("★ сильный" if abs(v - 0.5) > 0.35 else
                "★ заметный" if abs(v - 0.5) > 0.15 else
                "слабый" if abs(v - 0.5) > 0.05 else "шум")
        print(f"| {nm} | {v:.3f} | {note} |")
    aucs.sort(reverse=True)
    print(f"\n★ ЛУЧШИЙ ОДИНОЧНЫЙ: {FEAT[aucs[0][1]]} — AUC {aucs[0][2]:.3f}")
    # ★ лучшая ПАРА: сетка по двум сильнейшим, критерий — доля верных при равных весах классов
    j1, j2 = aucs[0][1], aucs[1][1]
    z = (X[:, j1] - X[:, j1].mean()) / (X[:, j1].std() + 1e-9)
    w = (X[:, j2] - X[:, j2].mean()) / (X[:, j2].std() + 1e-9)
    s1 = 1 if aucs[0][2] > 0.5 else -1
    s2 = 1 if aucs[1][2] > 0.5 else -1
    best = max(((auc(s1 * z + k * s2 * w, Y), k) for k in np.linspace(0, 2, 21)))
    print(f"★ ЛУЧШАЯ ПАРА: {FEAT[j1]} + {best[1]:.1f}×{FEAT[j2]} — AUC {best[0]:.3f}")
    # ★★★ РАБОЧЕЕ ЧИСЛО, А НЕ AUC. При 6.8% положительных AUC красив и бесполезен: решает, сколько
    # ЛИШНЕЙ туши уходит, если НУЖНУЮ почти не терять. Порог ставим по recall — терять
    # оцифровываемые раны нельзя, они и есть ответ; лишние же только создают развилки.
    sc = s1 * z + best[1] * s2 * w
    print(f"\n★★★ ЧТО ДАЁТ ОТБОР НА ПРАКТИКЕ (порог по recall, счёт «{FEAT[j1]} + "
          f"{best[1]:.1f}×{FEAT[j2]}»)")
    print("| сохраняем оцифрованных | отсеиваем лишних | ранов останется на строку | во сколько меньше |")
    base = len(Y) / max(1, len(Y) - int(Y.sum()))     # для доли; плотность берём из dens
    for rec in (0.99, 0.98, 0.95, 0.90):
        thr = np.quantile(sc[Y], 1 - rec)
        keep = sc >= thr
        neg_out = 1 - (keep & ~Y).sum() / max(1, (~Y).sum())
        left = keep.sum() / max(1, len(Y))
        print(f"| {100*rec:.0f}% | **{100*neg_out:.1f}%** | {100*left:.1f}% от всех | "
              f"**{1/max(1e-9,left):.1f}×** |")
    print("\n⚠ AUC говорит о РАЗДЕЛИМОСТИ, а не о готовом правиле: порог и цена ошибок выбираются "
          "отдельно, и §6.144 напоминает, что подмена признака при переносе в прод стоит кривых.")
    print("⚠ Признаки локальные: они сокращают множество кандидатов, но НЕ выбирают K линий. "
          "Выбор остаётся за сборкой — этот замер лишь говорит, во сколько раз ей станет легче.")
    sys.exit(0)

i, n = (int(x) for x in a.shard.split("/"))
trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]): int(r[3]["n_prod"]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q
_all = sorted({sh for (sh, t), K in KOF.items() if K >= a.mink})
cand = [_all[k] for k in np.random.default_rng(a.seed).permutation(len(_all))[:a.cap]]
mine = [s for k, s in enumerate(cand) if k % n == i]
print(f"★ ШАРД {i}/{n}: листов {len(mine)} из {len(cand)}")

X, Y = [], []
for si, sh in enumerate(mine, 1):
    q = SRC.get(sh)
    img = find_image(q) if q else None
    if not img:
        continue
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    tracks = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None and KOF.get((sh, t), 0) >= a.mink:
            tracks[t].append(nm)
    if not tracks:
        continue
    try:
        rgb = im.load_rgb(str(img))
    except Exception:
        continue
    H, W = rgb.shape[:2]
    p = DEFAULT.cv
    masks = {c: T._color_fg(rgb, c, p) for c in COLORS}
    fgm = np.zeros((H, W), bool)
    for c in COLORS:
        fgm |= masks[c]
    V = im.value_channel(rgb)
    paper = np.empty(H, np.float32)
    for y0 in range(0, H, 4096):
        y1 = min(H, y0 + 4096)
        paper[y0:y1] = np.percentile(V[y0:y1, ::4], 90, axis=1)

    for t, ns in tracks.items():
        allx = [x for nm in ns for x in gts[nm].values()]
        X0, X1 = max(0, int(min(allx)) - 40), min(W, int(max(allx)) + 40)
        band = max(1, X1 - X0)
        ys = sorted(set().union(*[set(gts[nm]) for nm in ns]))
        for y in range(ys[0], ys[-1] + 1, a.step):
            if not (1 <= y < H - 1):
                continue
            r = fgm[y, X0:X1]
            if not r.any():
                continue
            d = np.diff(r.astype(np.int8))
            st = list(np.flatnonzero(d == 1) + 1) + ([0] if r[0] else [])
            en = list(np.flatnonzero(d == -1) + 1) + ([len(r)] if r[-1] else [])
            st, en = sorted(st), sorted(en)
            runs = [(s, e) for s, e in zip(st, en) if e - s <= a.maxw]
            if not runs:
                continue
            cx = [X0 + (s + e) / 2.0 for s, e in runs]
            gx = [gts[nm][y] for nm in ns if y in gts[nm]]
            for k, (s, e) in enumerate(runs):
                x = cx[k]
                lo, hi = X0 + s, X0 + e
                seg = V[y, lo:hi]
                dark = float(paper[y] - seg.min()) if seg.size else 0.0
                col = max(COLORS, key=lambda c: masks[c][y, lo:hi].sum())
                near = min([abs(x - x2) for j2, x2 in enumerate(cx) if j2 != k], default=band)
                up = float(fgm[y - 1, max(0, int(x) - 3):int(x) + 4].any())
                dn = float(fgm[y + 1, max(0, int(x) - 3):int(x) + 4].any())
                # ★ «длина опоры»: сколько строк подряд вверх есть тушь под этим x (до 40) —
                #   локальная устойчивость, доступная в момент решения без всякой сборки.
                sup = 0
                for dy in range(1, 41):
                    if y - dy < 0 or not fgm[y - dy, max(0, int(x) - 3):int(x) + 4].any():
                        break
                    sup += 1
                X.append([e - s, dark, 1.0 if col == "black" else 0.0,
                          0.0 if col == "black" else 1.0, (x - X0) / band, len(runs),
                          near, up, dn, sup])
                Y.append(bool(gx) and min(abs(x - g) for g in gx) <= 3.0)
    if si % 3 == 0 or si == len(mine):
        print(f"  {si}/{len(mine)}  ранов {len(Y)}, оцифрованных {int(np.sum(Y))}")

out = TS / f"{a.tag}_{i}of{n}.pkl"
pickle.dump(dict(X=np.array(X, np.float32), Y=np.array(Y, bool)), open(out, "wb"))
print(f"★ ГОТОВО: ранов {len(Y)}, оцифрованных {int(np.sum(Y))} → {out.name}")

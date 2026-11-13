r"""_nodup_selfstyle.py — §6.362 РАЗВЕДКА (13.11): ИМЯ РАЗДВИНУТОЙ ПАРЫ ПО СТИЛЮ ЭТОГО ЖЕ СЛОТА НА ЕГО СОБСТВЕННЫХ СТРОКАХ.

§6.359 раздвигает дубли (безымянных +133), но кому из пары какая линия — монетка; модель стиля корпуса (§6.361) на этих парах хуже
монетки. Здесь — свидетельство ЭТОГО листа: до раздвигания (NSE22) у слотов пары есть строки, где их трассы НЕ совпадают (каждый на своей
линии); стиль линии там (цвет, толщина, штрих) сравнивается со стилем двух линий после раздвигания (NDS3): общей и новой.

Пара = слоты одного трека, чьи трассы NSE22 совпадали (≤ 3 px) на доле строк ≥ 0.3, а в NDS3 — < 0.3; «ушедший» X — тот, чья трасса
изменилась сильнее. Исход по эталону (мера приёмки, `_sample_review`): именных при раскладке как есть (`cur`) и при обмене трасс (`swp`).

  _nodup_selfstyle.py scan  [--base NSE22 --mode NDS3]   — по листам стенда → taskS/nodup_selfstyle/pairs.jsonl (продолжаемо)
  _nodup_selfstyle.py report                              — точность правила по запасу, выигрыш, потолок
"""
import sys, argparse, json, os
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
OUT = TS / "nodup_selfstyle"
KEYS = ("c1", "c2", "width", "dark", "gap")
SCALE = dict(c1=15.0, c2=15.0, width=1.0, dark=40.0, gap=0.15)


def _idle():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)   # IDLE_PRIORITY_CLASS
    except Exception:
        pass


def ovl(a, b, tol=3.0):
    """доля строк a, где b есть и ближе tol"""
    if not a:
        return 0.0
    return sum(1 for y, x in a.items() if y in b and abs(b[y] - x) <= tol) / len(a)


def style(rgb, gray, paper, rows, tr, NS):
    rows = sorted(rows)
    if len(rows) < 40:
        return None
    st = max(1, len(rows) // 600)
    rows = rows[::st]
    ys = np.array(rows, np.int64); xs = np.array([tr[y] for y in rows], float)
    s = NS.style_along(rgb, gray, paper, ys, xs)
    return None if s is None else {k: round(float(s[k]), 3) for k in KEYS + ("used",)} | {"n": len(rows)}


def scan(a):
    import _sample_review as SR
    from auto import name_style as NS, level_ink as LI
    from dataset_build import find_image
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    _idle()
    OUT.mkdir(parents=True, exist_ok=True)
    jf = OUT / f"pairs_{a.mode}.jsonl"
    done = set()
    if jf.exists():
        for l in jf.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(l)["sheet"])
            except Exception:
                pass
    sheets = [l.strip() for l in (TS / "nodup_stand" / "all.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    TRACK = SR.load_tracks(); SRC = {q.name: q for q in SR.ARCH.glob("*/wlg/*.nlgx")}
    n = 0
    for nm in sheets:
        if nm in done or nm not in SRC:
            continue
        d = SR.sheet_dir(nm)
        gb = next(iter(sorted((TS / "rp_lvl" / a.base / d).glob("*_auto.nlgx"))), None)
        gm = next(iter(sorted((TS / "rp_lvl" / a.mode / d).glob("*_auto.nlgx"))), None)
        rec = {"sheet": nm, "pairs": []}
        if gb is None or gm is None:
            rec["skip"] = "нет выдачи"
        else:
            tm = TRACK.get(nm, {})
            SR._TREES.clear()
            mt, mw, gts, wr = SR.curves(SRC[nm], gm)
            _, mwp, _, pre = SR.curves(SRC[nm], gb)
            LWp = {c["name"]: SR.lv_rows(c) for c in mwp["curves"] if c["name"] in pre}
            LG = {c["name"]: SR.lv_rows(c) for c in mt["curves"] if c["name"] in gts}
            CH = {c["name"]: SR.DL.build_family(mt, c) for c in mt["curves"] if c["name"] in gts}
            LW = {c["name"]: SR.lv_rows(c) for c in mw["curves"] if c["name"] in wr}
            E = {}

            def hon(w, g):
                if g not in gts or w not in wr:
                    return 0
                if (w, g) not in E:
                    E[(w, g)] = SR.err(SR.relevel(wr[w], LW.get(w, {}), LG.get(g, {}), CH.get(g, [])), gts[g])
                return int(SR.HON(*E[(w, g)]))
            Ep = {}

            def hon_pre(w, g):
                if g not in gts or w not in pre:
                    return 0
                if (w, g) not in Ep:
                    Ep[(w, g)] = SR.err(SR.relevel(pre[w], LWp.get(w, {}), LG.get(g, {}), CH.get(g, [])), gts[g])
                return int(SR.HON(*Ep[(w, g)]))
            rec["named_pre"] = sum(hon_pre(g, g) for g in gts); rec["named_post"] = sum(hon(g, g) for g in gts)
            pairs = []
            names = [k for k in pre if k in wr and tm.get(k) is not None]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    A, B = names[i], names[j]
                    if tm[A] != tm[B]:
                        continue
                    if max(ovl(pre[A], pre[B]), ovl(pre[B], pre[A])) < 0.3:
                        continue
                    if max(ovl(wr[A], wr[B]), ovl(wr[B], wr[A])) >= 0.3:
                        continue
                    sa, sb = ovl(pre[A], wr[A]), ovl(pre[B], wr[B])
                    X, Y = (A, B) if sa <= sb else (B, A)
                    pairs.append((X, Y, sa, sb))
            if pairs:
                img = find_image(SRC[nm])
                if img is None:
                    rec["skip"] = "нет скана"
                else:
                    with Image.open(img) as im:
                        rgb = np.asarray(im.convert("RGB")); gray = np.asarray(im.convert("L"))
                    paper = LI.paper_of(gray)
                    for X, Y, sa, sb in pairs:
                        pX, pY = pre[X], pre[Y]
                        shared = [y for y, x in pX.items() if y in pY and abs(pY[y] - x) <= 3]
                        own = lambda P, Q: [y for y, x in P.items() if y not in Q or abs(Q[y] - x) > 3]
                        qX, qY = wr[X], wr[Y]
                        r = {"X": X, "Y": Y, "track": str(tm[X]), "sim": [round(min(sa, sb), 3), round(max(sa, sb), 3)],
                             "n_shared": len(shared),
                             "cur": hon(X, X) + hon(Y, Y), "swp": hon(X, Y) + hon(Y, X),
                             "pre": [hon_pre(X, X), hon_pre(Y, Y)], "post": [hon(X, X), hon(Y, Y)],
                             "ov_pre": [round(ovl(pX, pY), 3), round(ovl(pY, pX), 3)],
                             "in_gt": [X in gts, Y in gts],
                             "own_X": style(rgb, gray, paper, own(pX, pY), pX, NS),
                             "own_Y": style(rgb, gray, paper, own(pY, pX), pY, NS),
                             "alt": style(rgb, gray, paper, own(qX, qY), qX, NS),
                             "sh": style(rgb, gray, paper, [y for y in shared if y in qY], qY, NS)}
                        rec["pairs"].append(r)
                    del rgb, gray
        with open(jf, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
        if n % 20 == 0:
            print(f"  {n} листов (всего готово {len(done) + n} из {len(sheets)})")
    print(f"★ scan готов: {len(done) + n} листов")


def dist(s, t, keys=KEYS):
    return sum(abs(s[k] - t[k]) / SCALE[k] for k in keys)


def report(a):
    R = [p for l in (OUT / f"pairs_{a.mode}.jsonl").read_text(encoding="utf-8").splitlines() for p in json.loads(l)["pairs"]]
    print(f"★ пар {len(R)}; именных как есть {sum(p['cur'] for p in R)}, при обмене всех {sum(p['swp'] for p in R)}, "
          f"потолок (лучшее в каждой паре) {sum(max(p['cur'], p['swp']) for p in R)}")
    dec = [p for p in R if p["cur"] != p["swp"]]
    print(f"★ решающих пар (эталон различает) {len(dec)}: верна раскладка как есть {sum(p['cur'] > p['swp'] for p in dec)}, "
          f"обмен {sum(p['swp'] > p['cur'] for p in dec)}")
    for keys in (KEYS, ("c1", "c2"), ("width",), ("c1", "c2", "width"), ("gap",), ("dark",)):
        print(f"── признаки {'+'.join(keys)}")
        for mode in ("both", "X", "Y"):
            sc = []
            for p in R:
                oX, oY, al, sh = p["own_X"], p["own_Y"], p["alt"], p["sh"]
                if al is None or sh is None:
                    continue
                m = 0.0; used = False
                if mode in ("both", "X") and oX is not None:
                    m += dist(oX, sh, keys) - dist(oX, al, keys); used = True       # > 0: X ближе к новой линии — как есть
                if mode in ("both", "Y") and oY is not None:
                    m += dist(oY, al, keys) - dist(oY, sh, keys); used = True       # > 0: Y ближе к общей — как есть
                if used:
                    sc.append((m, p))
            L = [f"  {mode:4s} пар с признаком {len(sc):4d}"]
            for thr in (0.0, 0.5, 1.0, 2.0, 3.0):
                sw = [p for m, p in sc if m < -thr]
                gain = sum(p["swp"] - p["cur"] for p in sw)
                dd = [(m, p) for m, p in sc if p["cur"] != p["swp"] and abs(m) > thr]
                acc = sum((m > 0) == (p["cur"] > p["swp"]) for m, p in dd) / max(len(dd), 1)
                L.append(f"запас>{thr}: обменов {len(sw)}, Δ {gain:+d}, точн. {acc:.2f} из {len(dd)}")
            print(" | ".join(L))


def decomp(a):
    """откуда Δ именных NDS3 − NSE22: пары по состоянию ДО (X — ушедший слот) и всё прочее"""
    from collections import defaultdict
    L = [json.loads(l) for l in (OUT / f"pairs_{a.mode}.jsonl").read_text(encoding="utf-8").splitlines()]
    L = [r for r in L if "named_pre" in r]
    tot = sum(r["named_post"] - r["named_pre"] for r in L)
    P = [p for r in L for p in r["pairs"]]
    dp = sum(sum(p["post"]) - sum(p["pre"]) for p in P)
    print(f"★ листов {len(L)}: именных {sum(r['named_pre'] for r in L)} → {sum(r['named_post'] for r in L)} (Δ {tot:+d}); "
          f"на раздвинутых парах Δ {dp:+d} ({len(P)} пар), прочее Δ {tot - dp:+d}")
    G = defaultdict(list)
    for p in P:
        G[tuple(p["pre"])].append(p)
    print("  ДО (X ушедший, Y оставшийся) | пар | ПОСЛЕ X / Y (сумма) | Δ | обмен дал бы | ov_pre X→Y медиана | Y→X | n_shared")
    for k in sorted(G):
        g = G[k]
        print(f"  {str(k):10s} {len(g):5d}  X {sum(p['post'][0] for p in g):4d} / Y {sum(p['post'][1] for p in g):4d}  "
              f"Δ {sum(sum(p['post']) - sum(p['pre']) for p in g):+5d}  обмен {sum(p['swp'] - sum(p['pre']) for p in g):+5d}  "
              f"{np.median([p['ov_pre'][0] for p in g]):.2f}  {np.median([p['ov_pre'][1] for p in g]):.2f}  "
              f"{int(np.median([p['n_shared'] for p in g]))}")
    best = sum(max(sum(p["pre"]), p["cur"], p["swp"]) - sum(p["pre"]) for p in P)
    print(f"★ потолок по парам (лучшее из: не раздвигать / как есть / обмен): Δ {best:+d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "report", "decomp"])
    ap.add_argument("--base", default="NSE22")
    ap.add_argument("--mode", default="NDS3")
    a = ap.parse_args()
    {"scan": scan, "report": report, "decomp": decomp}[a.cmd](a)


if __name__ == "__main__":
    main()

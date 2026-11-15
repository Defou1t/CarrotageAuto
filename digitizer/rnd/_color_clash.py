r"""_color_clash.py — §6.378 РАЗВЕДКА: «разные имена трека — разные цвета» на цветных листах (15.11).

Разбор глазами выборки 15.11 (35 неверных кривых): 15 связаны с цветом туши по легенде шапки (листы 01, 04, 06, 30); на 04 четыре наших
имени сидят на двух зелёных линиях, а чёрная и синяя линии трека не взяты никем. Без чтения легенды проверяемо: у эталона в треке разные
имена обычно разного цвета. Здесь (выдача NSE26 вне фолда, случайные листы поля + вне): цвет кривой = медиана RGB самого тёмного пикселя
±1 px вдоль трассы (каждая 4-я строка); трек «многоцветный», если у кривых эталона трека попарно есть расхождение ≥ DC по какому-то каналу.
Для каждого слота многоцветного трека (наша кривая того же имени): честна ли в пикселях (медиана |dx| ≤ 3 по общим строкам, покрытие ≥ 0.9);
«конфликт цвета» — наша кривая ближе SC к цвету другой нашей кривой того же трека; «сирота» — цвет эталона слота дальше DC от цветов ВСЕХ
наших кривых трека. Класс A (чинится правилом различия цветов) = неверна И конфликт И сирота. Ложные тревоги — верные с конфликтом.

  _color_clash.py [--n 400] [--seed 3]   → taskS/color_clash/rows.jsonl (продолжаемо), сводка
"""
import sys, argparse, json, random, hashlib, time, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
OUT = TS / "color_clash"


def sheet_dir(n):
    return f"{n.stem[:40]}_{hashlib.md5(n.stem.encode('utf-8')).hexdigest()[:8]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--mode", default="NSE26")
    ap.add_argument("--dc", type=float, default=50.0)
    ap.add_argument("--sc", type=float, default=25.0)
    a = ap.parse_args()
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from auto import meta as M, imaging as im
    im.EXIF_TRANSPOSE = False
    OUT.mkdir(parents=True, exist_ok=True)
    jf = OUT / "rows.jsonl"
    done = {json.loads(l)["sheet"] for l in jf.read_text(encoding="utf-8").splitlines()} if jf.exists() else set()
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    names = []
    for f in ("tcache_sheets.txt", "outside_sheets.txt"):
        names += [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    names = sorted(set(names)); random.Random(a.seed).shuffle(names)
    T0 = time.time(); nsh = len(done)
    for nm in names:
        if nsh >= a.n:
            break
        if nm in done or nm not in SRC:
            continue
        n = SRC[nm]
        od = TS / "rp_lvl" / a.mode / sheet_dir(n)
        fs = list(od.glob("*_auto.nlgx")) if od.exists() else []
        src = next((TS / d / (sheet_dir(n) + ".pkl") for d in ("tcache", "tcache_out") if (TS / d / (sheet_dir(n) + ".pkl")).exists()), None)
        if not fs or src is None:
            continue
        try:
            m = extract(str(n)); mo = extract(str(fs[0]))
        except Exception:
            continue
        T = [c for c in m["curves"] if M.mnem_root(c["name"]) != "DA" and len(c.get("points") or c.get("xs") or [1]) > 0]
        ours = {c["name"]: c for c in mo["curves"]}
        img = pickle.load(open(src, "rb"))["image"]
        rgb = im.load_rgb(img)                           # координаты nlgx — исходного скана (без row_shift)
        H, W = rgb.shape[:2]
        gray = rgb.mean(2)

        def color(tr):
            ys = sorted(tr)[::4]; px = []
            for y in ys:
                x = int(round(tr[y]))
                if 0 <= y < H and 1 <= x < W - 1:
                    g = gray[y, x - 1:x + 2]; j = int(np.argmin(g))
                    if g[j] < 200:
                        px.append(rgb[y, x - 1 + j])
            return np.median(np.array(px, float), 0).round(1).tolist() if len(px) >= 30 else None

        recs = []
        for c in T:
            gt = dense(c)
            if len(gt) < 200:
                continue
            oc = ours.get(c["name"])
            ot = dense(oc) if oc is not None else {}
            com = [y for y in gt if y in ot]
            hon = False
            if len(com) >= 50:
                hon = bool(np.median([abs(ot[y] - gt[y]) for y in com]) <= 3 and len(com) / len(gt) >= 0.9)
            recs.append(dict(curve=c["name"], track=(c["name"].split() + ["", ""])[1], ct=color(gt), co=color(ot) if len(ot) >= 200 else None, hon=hon,
                             has=oc is not None))
        with open(jf, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(sheet=nm, recs=recs), ensure_ascii=False) + "\n")
        nsh += 1
        del rgb, gray
        if nsh % 25 == 0:
            print(f"  листов {nsh}, {time.time() - T0:.0f} с")
    # ── сводка ──
    dist = lambda p, q: float(np.max(np.abs(np.array(p) - np.array(q))))
    tot = wrong = mc_slots = mc_wrong = A = FA = mc_tracks = 0; sheets_mc = set(); clash_wrong = orphan_wrong = 0
    nsheets = 0
    for l in jf.read_text(encoding="utf-8").splitlines():
        q = json.loads(l); nsheets += 1
        by = {}
        for r in q["recs"]:
            by.setdefault(r["track"], []).append(r)
        for tk, R in by.items():
            for r in R:
                tot += 1; wrong += not r["hon"]
            ct = [r["ct"] for r in R if r["ct"] is not None]
            if len(ct) < 2 or max(dist(p, q_) for i, p in enumerate(ct) for q_ in ct[i + 1:]) < a.dc:
                continue
            mc_tracks += 1; sheets_mc.add(q["sheet"])
            cos = [r["co"] for r in R if r["co"] is not None]
            for r in R:
                if not r["has"] or r["ct"] is None:
                    continue
                mc_slots += 1; mc_wrong += not r["hon"]
                clash = r["co"] is not None and any(dist(r["co"], o) < a.sc for o in cos if o is not r["co"]) and sum(1 for o in cos if dist(r["co"], o) < a.sc) >= 2
                orphan = all(dist(r["ct"], o) >= a.dc for o in cos) if cos else False
                if not r["hon"]:
                    clash_wrong += clash; orphan_wrong += orphan
                    A += clash and orphan
                elif clash:
                    FA += 1
    print(f"★ листов {nsheets}; слотов эталона {tot}, неверных в пикселях {wrong}")
    print(f"   многоцветных треков {mc_tracks} на {len(sheets_mc)} листах; слотов там {mc_slots}, неверных {mc_wrong}")
    print(f"   у неверных на многоцветных: конфликт цвета {clash_wrong}, сирота {orphan_wrong}, класс A (оба) {A}")
    print(f"   ложные тревоги (верные с конфликтом цвета): {FA}")
    sc = 2687 / max(nsheets, 1)
    print(f"   на весь стенд (× {sc:.1f}): класс A ≈ {A * sc:.0f}, ложных ≈ {FA * sc:.0f}")


if __name__ == "__main__":
    main()

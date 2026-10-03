"""Чистый тест именования глазами: окно листа без наложений, красная риска — строка R, сетка x подписана.
  python name_test.py render --dump ... --n 10 --seed 1 --out names2      → картинки + ключ (не смотреть до ответа)
  python name_test.py score names2/answers.json names2/key.json           → сверка: моё x имени в строке R против эталона
answers.json: {"00": {"GZ41 DA1 SA4": 512, ...}, ...}
"""
import sys, json, argparse, hashlib, pickle, random
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from extract_nlgx import extract, NULL

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["render", "score"])
ap.add_argument("a1", nargs="?"); ap.add_argument("a2", nargs="?")
ap.add_argument("--dump"); ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="names2")
a = ap.parse_args()
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
try:
    font = ImageFont.truetype("arial.ttf", 14)
except Exception:
    font = ImageFont.load_default()


def dense(c):
    pts = [(c["top_y"] + i, float(x)) for i, x in enumerate(c["xs"]) if x != NULL]
    out = {}
    for (y0, x0), (y1, x1) in zip(pts, pts[1:]):
        out[y0] = x0
        if 0 < y1 - y0 <= 200:
            for yy in range(y0 + 1, y1):
                out[yy] = x0 + (x1 - x0) * (yy - y0) / (y1 - y0)
    if pts:
        out[pts[-1][0]] = pts[-1][1]
    return out


if a.cmd == "render":
    OUT = Path(a.out); OUT.mkdir(exist_ok=True)
    IMGS = {}
    for q in Path(r"F:\nds\projects\Archive").glob("*/img/*"):
        IMGS.setdefault(q.stem, q)
    ROWS = pickle.load(open(a.dump, "rb"))
    sheets = sorted({r["sheet"] for r in ROWS})
    rng = random.Random(a.seed); rng.shuffle(sheets)
    KEY = {}
    for j, sh in enumerate(sheets[:a.n]):
        q = SRC[sh]; stem = q.stem
        G = {c["name"]: dense(c) for c in extract(str(q))["curves"]
             if not c["name"].startswith("DA") and sum(1 for x in c["xs"] if x != NULL) >= 50}
        img = Image.open(IMGS[stem]).convert("RGB")
        # строка R: где есть все кривые эталона (если нет — где больше всего)
        allys = sorted(set().union(*[set(g) for g in G.values()]))
        cand = [y for y in allys[::50] if sum(y in g for g in G.values()) == len(G)] or allys[::50]
        R = cand[len(cand) // 2]
        y0, y1 = R - 500, R + 500
        x0, x1 = 0, img.width
        cr = img.crop((x0, y0, x1, y1))
        sc = 1100 / cr.width
        cr = cr.resize((1100, int(cr.height * sc)), Image.LANCZOS)
        pad = 22
        can = Image.new("RGB", (cr.width, cr.height + pad), "white"); can.paste(cr, (0, pad))
        dr = ImageDraw.Draw(can, "RGBA")
        yy = pad + int((R - y0) * sc)
        dr.line([(0, yy), (can.width, yy)], fill=(255, 0, 0, 160), width=1)
        for x in range(0, img.width, 100):
            xx = int((x - x0) * sc)
            dr.line([(xx, pad), (xx, can.height)], fill=(0, 90, 255, 60 if x % 500 else 140), width=1)
            if x % 500 == 0:
                dr.text((xx + 2, 2), str(x), fill=(0, 0, 200), font=font)
        can.save(OUT / f"{j:02d}_window.png")
        KEY[f"{j:02d}"] = dict(sheet=sh, R=R, truth={k: g.get(R) for k, g in G.items()})
        print(f"{j:02d}: {stem}: строка R = {R}; имена: {', '.join(G)}")
    json.dump(KEY, open(OUT / "key.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
else:
    ans = json.load(open(a.a1, encoding="utf-8")); key = json.load(open(a.a2, encoding="utf-8"))
    ok = tot = 0
    for j, d in ans.items():
        tr = key[j]["truth"]
        for name, x in d.items():
            if tr.get(name) is None:
                continue
            tot += 1
            # верно, если ближайшая кривая эталона к моему x — того же имени и в 15 px
            near = min(((abs(v - x), k) for k, v in tr.items() if v is not None))
            good = near[1] == name and near[0] <= 25
            ok += good
            print(f"{j} {name:16s}: моё x {x:6.0f}, эталон {tr[name]:6.0f}  {'✓' if good else '✗ ближе всего ' + near[1]}")
    print(f"★ верно {ok} из {tot}")

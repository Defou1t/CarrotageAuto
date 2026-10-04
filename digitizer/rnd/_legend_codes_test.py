r"""_legend_codes_test.py — ЧИТАЕТ ЛИ ЛОКАЛЬНАЯ МОДЕЛЬ СО ЗРЕНИЕМ КОДЫ ЗОНДОВ В ШАПКЕ (04.10, к §6.282).

Заказчик (04.10): ГЗ разных зондов одного цвета и толщины различаются подписью над ними — коды зондов (таблица заказчика,
последний столбец): GZ1 A0.4M0.1N, GZ2 A1.0M0.1N, GZ3 A2.0M0.5N, GZ4 A4.0M0.5N, GZ5 A8.0M1.0N, OGZ N0.5M2.0A. Здесь — узкая проверка:
шапка листа (от верха скана до начала кривых) → LM Studio (localhost:1234) → JSON-список найденных кодов; сверка с ожидаемым набором
по именам ГЗ/ОГЗ эталона листа (полнота и лишние).

  _legend_codes_test.py --n 15 --model qwen2.5-vl-7b-instruct
"""
import sys, argparse, json, random, re, base64, io, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
import numpy as np
from PIL import Image
from extract_nlgx import extract, NULL

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=15)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--model", default="qwen2.5-vl-7b-instruct")
ap.add_argument("--url", default="http://localhost:1234/v1/chat/completions")
ap.add_argument("--max-side", type=int, default=1600)
ap.add_argument("--out", default=r"F:/nds/output/taskS/legend_codes_test.json")
a = ap.parse_args()
CODE = {"GZ1": "A0.4M0.1N", "GZ2": "A1.0M0.1N", "GZ3": "A2.0M0.5N", "GZ4": "A4.0M0.5N", "GZ5": "A8.0M1.0N", "OGZ": "N0.5M2.0A"}
ALT = {"A8.0M0.5N": "A8.0M1.0N"}
TS = Path(r"F:/nds/output/taskS")
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
IM = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/img/*"):
    IM.setdefault(q.stem, q)


def norm(s):
    s = str(s).upper().replace(" ", "").replace(",", ".")
    s = s.translate(str.maketrans("АМНОВСЕКРТХ", "AMHOBCEKPTX")).replace("H", "N")
    s = re.sub(r"(\d)(?=[AMN])", r"\1", s)
    return ALT.get(s, s)


sheets = [l.strip() for l in (TS / "wellmap_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
random.Random(a.seed).shuffle(sheets)
R = []
for sh in sheets:
    if len(R) >= a.n:
        break
    q = SRC.get(sh)
    if not q or q.stem not in IM:
        continue
    mt = extract(str(q))
    exp = set()
    starts = []
    for c in mt["curves"]:
        m = re.match(r"^(?:BKZ_)?(GZ\d|OGZ)", c["name"])
        pts = [(c["top_y"] + i) for i, x in enumerate(c["xs"]) if x != NULL]
        if m and len(pts) >= 50:
            exp.add(CODE.get(m.group(1), "?")); starts.append(pts[0])
    exp.discard("?")
    if len(exp) < 2:
        continue
    im = Image.open(IM[q.stem]).convert("RGB")
    y1 = min(starts) + 40
    head = im.crop((0, max(0, y1 - 2600), im.width, y1))
    s_ = a.max_side / max(head.size)
    if s_ < 1:
        head = head.resize((int(head.width * s_), int(head.height * s_)), Image.LANCZOS)
    buf = io.BytesIO(); head.save(buf, format="PNG")
    prompt = ("This is the header (legend and scales) of a scanned paper well-log chart. List every electrode probe code (gradient or "
              "potential sonde designation) that is printed or hand-written in it, such as A0.4M0.1N, A1.0M0.1N, A2.0M0.5N, A4.0M0.5N, "
              "A8.0M1.0N, N0.5M2.0A, N11M0.5A. Output ONLY a JSON list of the codes exactly as written, for example "
              "[\"A0.4M0.1N\", \"A2.0M0.5N\"]. If there are none, output [].")
    body = {"model": a.model, "temperature": 0, "max_tokens": 300,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                      {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}]}]}
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(a.url, data=json.dumps(body).encode(),
                                                                     headers={"Content-Type": "application/json"}), timeout=900).read())
        txt = r["choices"][0]["message"].get("content") or ""
    except Exception as e:
        txt = f"ERROR {e}"
    found = set()
    mm = re.findall(r"\[.*?\]", txt, flags=re.S)
    if mm:
        try:
            found = {norm(x) for x in json.loads(mm[-1])}
        except Exception:
            found = {norm(x) for x in re.findall(r"[AN][\d.,]+M[\d.,]+[AN]", txt)}
    hit = exp & found; miss = exp - found; extra = {f for f in found if f not in exp and re.match(r"^[AN][\d.]+M[\d.]+[AN]$", f)}
    R.append(dict(sheet=sh, expected=sorted(exp), found=sorted(found), hit=len(hit), miss=sorted(miss), extra=sorted(extra),
                  sec=round(time.time() - t0, 1), raw=txt[-300:]))
    print(f"  {sh[:50]:50s} ожидалось {sorted(exp)} найдено {sorted(found)} ({time.time() - t0:.0f} с)")
json.dump(R, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
tot = sum(len(r["expected"]) for r in R); hit = sum(r["hit"] for r in R); ext = sum(len(r["extra"]) for r in R)
full = sum(1 for r in R if not r["miss"])
print(f"★ {a.model}: листов {len(R)}; кодов ожидалось {tot}, найдено верно {hit} ({hit / max(1, tot):.0%}); лишних {ext}; "
      f"листов, где найдены все — {full}")

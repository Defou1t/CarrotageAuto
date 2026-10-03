r"""_vlm_names.py — ИМЕНА КРИВЫХ ПО ЛЕГЕНДЕ ШАПКИ ЛОКАЛЬНОЙ МОДЕЛЬЮ СО ЗРЕНИЕМ (03.10, §6.260/§6.261).

§6.260: на поле 350 из 1192 честно проведённых кривых выдачи стоят под чужим именем; Claude по шапке листа (легенда зондов,
жирная/тонкая линия, цвет туши, стрелки) назвал верно 20 из 20 на 6 листах. Здесь то же для локальных моделей LM Studio.

  build — набор: листы поля с перепутанными именами (`--swaps`) и контрольные (все честные кривые названы верно); на лист —
          шапка (`*_header.png`), окно с номерными метками кривых выдачи (`*_window.png`), `meta.json`: метки по трекам, имена-
          кандидаты трека, ИСТИНА метки (имя эталона, с которым кривая выдачи честна 1:1 при 3 px; нет пары — null).
  ask   — вопрос модели (`--model`), ответ — JSON {"#1": "имя", ...} последней строкой; `answers_<модель>.json`.
  score — верно ли имя по меткам с истиной: модель против прода, отдельно по перепутанным и контрольным листам.

  _vlm_names.py build --out F:/nds/output/vlm_names --n-swap 25 --n-ctrl 25 --seed 11
  _vlm_names.py ask --out F:/nds/output/vlm_names --model gemma-4-26b-a4b-it
  _vlm_names.py score --out F:/nds/output/vlm_names --model gemma-4-26b-a4b-it
"""
import sys, argparse, json, pickle, hashlib, random, base64, time, re, urllib.request
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["build", "ask", "score"])
ap.add_argument("--out", default=r"F:/nds/output/vlm_names")
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_vc/N")
ap.add_argument("--swaps", default=r"F:/nds/output/taskS/name_swaps_field.pkl")
ap.add_argument("--sheets", default="wellmap_sheets.txt")
ap.add_argument("--n-swap", type=int, default=25)
ap.add_argument("--n-ctrl", type=int, default=25)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--exclude", default="", help="листы, уже виденные глазами (json-ключ name_test), — не брать")
ap.add_argument("--model", default="")
ap.add_argument("--url", default="http://localhost:1234/v1/chat/completions")
ap.add_argument("--max-tokens", type=int, default=4000)
ap.add_argument("--only", default="", help="ask: только эти листы (через запятую, номера)")
a = ap.parse_args()
OUT = Path(a.out)
GLOSS = {"GZ": "gradient zond (ГЗ, БКЗ); GZ1..GZ5 = A0.4M0.1N, A1.0M0.1N, A2.0M0.5N, A4.0M0.5N, A8.0M1.0N",
         "OGZ": "inverse gradient zond (обращённый ГЗ), e.g. N0.5M2.0A / N0.5M4.0A", "PZ": "potential zond (ПЗ), e.g. N11M0.5A",
         "SP": "spontaneous potential ПС, mV", "DS": "caliper ДС / кавернометр", "CALI": "caliper ДС", "CAL": "profile caliper arm ПМ",
         "MDS": "micro-caliper", "GK": "gamma ray ГК", "NGK": "neutron gamma НГК", "NNK": "neutron-neutron ННК", "NNKM": "ННК",
         "BK": "laterolog БК", "MBK": "micro-laterolog БМК", "BMK": "micro-laterolog БМК", "IK": "induction ИК", "IKA": "induction ИК (active)",
         "IKR": "induction ИК (reactive)", "MGZ": "micro-gradient zond МГЗ", "MPZ": "micro-potential zond МПЗ", "MSP": "micro ПС",
         "T": "acoustic interval time T1/T2", "A": "acoustic amplitude A1/A2", "AK": "acoustic ΔT", "DT": "acoustic ΔT", "DTP": "acoustic ΔT (P)",
         "AL": "acoustic attenuation α", "AP": "acoustic attenuation/amplitude", "TM": "thermometer термометрия", "REZ": "resistivity",
         "ZAT": "ИННК decay / attenuation", "TAU": "ИННК neutron lifetime τ", "I": "ИННК count rate / intensity", "NGZ": "gradient zond (normal)"}
COLS = [(230, 0, 0), (0, 0, 230), (200, 0, 200), (0, 150, 150), (230, 120, 0), (0, 150, 0), (120, 60, 0), (90, 90, 90)]


def root_of(name):
    w = name.split()[0]
    r = re.match(r"([A-Z_]+?)(\d*)$", w)
    base = r.group(1) if r else w
    for pre in ("BKZ_",):
        if base.startswith(pre):
            base = base[len(pre):]
    return base, (r.group(2) if r else "")


def gloss(name):
    base, dig = root_of(name)
    g = GLOSS.get(base, "")
    if base == "GZ" and dig:
        g = f"gradient zond number {dig[0]} ({['A0.4M0.1N', 'A1.0M0.1N', 'A2.0M0.5N', 'A4.0M0.5N', 'A8.0M1.0N'][int(dig[0]) - 1] if dig[0] in '12345' else '?'})"
    return g


if a.cmd == "build":
    from PIL import Image, ImageDraw, ImageFont
    from extract_nlgx import extract, NULL
    from _multi_replica_probe import dense
    from auto import meta as M
    Image.MAX_IMAGE_PIXELS = None
    OUT.mkdir(parents=True, exist_ok=True)
    TS = Path(a.ts)
    smap = pickle.load(open(TS / "slotmap.pkl", "rb"))
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    IMGS = {}
    for q in Path(r"F:\nds\projects\Archive").glob("*/img/*"):
        IMGS.setdefault(q.stem, q)
    try:
        font = ImageFont.truetype("arialbd.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    def st(tr, gt):
        com = [y for y in tr if y in gt]
        if len(com) < 30:
            return None, 0.0
        return float(np.median([abs(tr[y] - gt[y]) for y in com])), len(com) / max(1, len(gt))

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

    def truth_of(sh):
        """→ (W, G, tm, {имя выдачи: имя эталона или None}) по 1:1 при 3 px в треках"""
        q = SRC[sh]; stem = q.stem
        pd = Path(a.dir) / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
        got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
        if not got:
            return None
        Wc = [c for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA" and any(x != NULL for x in c["xs"])]
        W = {c["name"]: dense(c) for c in Wc}
        G = {c["name"]: dense(c) for c in extract(str(q))["curves"]
             if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        tm = smap.get(sh, {})
        ident = {w: None for w in W}
        for t in {tm.get(g) for g in G if tm.get(g) is not None}:
            gs = [g for g in G if tm.get(g) == t]; ws = [k for k in W if tm.get(k) == t and W[k]]
            ok = {}
            for g in gs:
                for w in ws:
                    m, c = st(W[w], G[g])
                    ok[(g, w)] = m is not None and m <= 3.0 and c >= 0.9
            for g, w in match(gs, ws, ok).items():
                ident[w] = g
        return W, G, tm, ident

    swaps = pickle.load(open(a.swaps, "rb"))
    swap_sheets = sorted({r["sheet"] for r in swaps})
    excl = set()
    if a.exclude:
        excl = {v["sheet"] for v in json.load(open(a.exclude, encoding="utf-8")).values()}
    rng = random.Random(a.seed)
    cand_sw = [s for s in swap_sheets if s not in excl]
    rng.shuffle(cand_sw)
    allsh = [l.strip() for l in (Path(a.ts) / a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()]
    cand_ct = [s for s in allsh if s not in set(swap_sheets) and s not in excl]
    rng.shuffle(cand_ct)
    picked = []
    for kind, pool, need in (("перепутаны", cand_sw, a.n_swap), ("контроль", cand_ct, a.n_ctrl)):
        got_n = 0
        for sh in pool:
            if got_n >= need:
                break
            if sh not in SRC or SRC[sh].stem not in IMGS:
                continue
            r = truth_of(sh)
            if r is None:
                continue
            W, G, tm, ident = r
            tracks = {}
            for w in W:
                if tm.get(w) is not None:
                    tracks.setdefault(tm[w], []).append(w)
            multi = {t: ws for t, ws in tracks.items() if len(ws) >= 2}
            if not multi:
                continue
            if kind == "контроль" and not any(ident[w] is not None and ident[w] == w for ws in multi.values() for w in ws):
                continue                                  # контроль: в многокривом треке есть верно названная честная кривая
            picked.append((kind, sh, W, G, tm, ident, tracks))
            got_n += 1
    META = {}
    for j, (kind, sh, W, G, tm, ident, tracks) in enumerate(picked):
        stem = SRC[sh].stem
        img = Image.open(IMGS[stem]).convert("RGB")
        ys = [y for w in W.values() for y in w]
        top, bot = min(ys), max(ys)
        h0 = max(0, top - 6000)
        hd = img.crop((0, h0, img.width, min(img.height, top + 300)))
        s = min(1400 / hd.width, 2000 / hd.height)
        hd = hd.resize((max(1, int(hd.width * s)), max(1, int(hd.height * s))), Image.LANCZOS)
        hd.save(OUT / f"{j:02d}_header.png")
        # окно 1200 строк, где больше всего кривых выдачи
        best, by = -1, top
        for yc in range(top, max(top + 1, bot - 1200), 200):
            c = sum(sum(1 for y in range(yc, yc + 1200, 50) if y in w) for w in W.values())
            if c > best:
                best, by = c, yc
        y0, y1 = by, by + 1200
        cr = img.crop((0, y0, img.width, min(img.height, y1)))
        s = min(1400 / cr.width, 1.0)
        cr = cr.resize((max(1, int(cr.width * s)), max(1, int(cr.height * s))), Image.LANCZOS)
        dr = ImageDraw.Draw(cr)
        labels = {}
        order = sorted(W, key=lambda w: np.median([x for y, x in W[w].items() if y0 <= y < y1] or [1e9]))
        for i, w in enumerate(order):
            lab = f"#{i + 1}"; labels[lab] = w
            col = COLS[i % len(COLS)]
            pts = [(y, x) for y, x in sorted(W[w].items()) if y0 <= y < y1]
            if not pts:
                continue
            for frac in (0.15, 0.5, 0.85):
                y, x = pts[int(frac * (len(pts) - 1))]
                px, py = x * s, (y - y0) * s
                tx, ty = px + 30, py - 12
                dr.line([(px + 2, py), (tx, ty + 10)], fill=col, width=2)
                dr.rectangle([tx, ty, tx + 34, ty + 22], fill=(255, 255, 255), outline=col, width=2)
                dr.text((tx + 3, ty + 1), lab, fill=col, font=font)
        cr.save(OUT / f"{j:02d}_window.png")
        trk = {}
        for lab, w in labels.items():
            t = tm.get(w)
            trk.setdefault(str(t), {"labels": [], "names": []})
            trk[str(t)]["labels"].append(lab); trk[str(t)]["names"].append(w)
        META[f"{j:02d}"] = dict(kind=kind, sheet=sh, labels=labels, tracks=trk,
                                truth={lab: ident[w] for lab, w in labels.items()}, window=[y0, y1])
        print(f"{j:02d} [{kind}] {stem}: кривых {len(W)}, треков {len(trk)}, честных {sum(v is not None for v in ident.values())}")
    json.dump(META, open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

elif a.cmd == "ask":
    META = json.load(open(OUT / "meta.json", encoding="utf-8"))
    tag = re.sub(r"[^A-Za-z0-9._-]", "_", a.model)
    fa = OUT / f"answers_{tag}.json"
    ANS = json.load(open(fa, encoding="utf-8")) if fa.exists() else {}
    only = set(a.only.split(",")) if a.only else None
    t0 = time.time()
    for j, m in sorted(META.items()):
        if j in ANS or (only and j not in only):
            continue
        lines = []
        for t, d in m["tracks"].items():
            if t == "None":
                continue
            names = "; ".join(f"{n} ({gloss(n) or 'see legend'})" for n in d["names"])
            lines.append(f"Track {t}: tags {', '.join(d['labels'])} — choose among names: {names}")
        prompt = ("You are reading a scanned paper well-log chart (Soviet/Ukrainian, 1960–2010s). IMAGE 1 is the chart header and "
                  "legend: tool/zond captions (e.g. A0.4M0.1N, A2.0M0.5N, N0.5M2.0A, ПС, ДС, ГК, НГК, T1, T2, A1, ΔT, α), their scale "
                  "lines, and how each curve is drawn — ink colour, bold or thin or dashed line, left-to-right order of the scale "
                  "captions, sometimes arrows from a caption to its curve. IMAGE 2 is a section of the same chart; each drawn curve "
                  "is marked by a numbered tag (#1, #2, ...) with a short leader line touching the curve (tags are repeated at three "
                  "heights). For every tag decide which curve name it is, using ONLY the legend evidence (colour, line weight/style, "
                  "caption order and arrows, curve character). Each name may be used at most once within its track.\n"
                  + "\n".join(lines) +
                  "\nThink briefly, then give the final answer as ONE line of JSON mapping every tag to a name, e.g. "
                  "{\"#1\": \"<name>\", \"#2\": \"<name>\"}. Use the names exactly as written.")
        content = [{"type": "text", "text": prompt}]
        for part in ("header", "window"):
            b64 = base64.b64encode((OUT / f"{j}_{part}.png").read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        body = {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens, "messages": [{"role": "user", "content": content}]}
        req = urllib.request.Request(a.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t1 = time.time()
        try:
            msg = json.loads(urllib.request.urlopen(req, timeout=3600).read())["choices"][0]["message"]
            txt = (msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")[-1500:]
        except Exception as e:
            txt = f"ERROR {type(e).__name__}: {e}"
        js = None
        for mm in reversed(re.findall(r"\{[^{}]*\"#\d+\"[^{}]*\}", txt)):
            try:
                js = json.loads(mm); break
            except Exception:
                continue
        ANS[j] = dict(answer=js, text=txt[-1200:], sec=round(time.time() - t1, 1))
        json.dump(ANS, open(fa, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  {j} [{m['kind']}] {time.time() - t1:.0f} с: {js}")
    print(f"★ {a.model}: готово {len(ANS)} листов за {time.time() - t0:.0f} с")

else:
    META = json.load(open(OUT / "meta.json", encoding="utf-8"))
    tag = re.sub(r"[^A-Za-z0-9._-]", "_", a.model)
    ANS = json.load(open(OUT / f"answers_{tag}.json", encoding="utf-8"))
    C = Counter()
    for j, m in sorted(META.items()):
        ans = (ANS.get(j) or {}).get("answer") or {}
        for lab, tr in m["truth"].items():
            if tr is None or m["tracks"].get(str(None), {}).get("labels", []) and lab in m["tracks"]["None"]["labels"]:
                continue
            k = m["kind"]
            C[(k, "меток")] += 1
            C[(k, "прод")] += m["labels"][lab] == tr
            C[(k, "модель")] += ans.get(lab) == tr
            C[(k, "без ответа")] += lab not in ans
    for k in ("перепутаны", "контроль"):
        n = C[(k, "меток")]
        if n:
            print(f"★ {k}: меток с истиной {n}; прод верно {C[(k, 'прод')]} ({100 * C[(k, 'прод')] / n:.0f}%), "
                  f"{a.model} верно {C[(k, 'модель')]} ({100 * C[(k, 'модель')] / n:.0f}%), без ответа {C[(k, 'без ответа')]}")
    secs = [v.get("sec", 0) for v in ANS.values()]
    print(f"   время на лист: медиана {np.median(secs):.0f} с")

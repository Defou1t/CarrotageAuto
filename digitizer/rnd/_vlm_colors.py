r"""_vlm_colors.py — §6.385 ПИЛОТ: цвета кривых трека по шапке локальной моделью со зрением (20.11).

§6.383: у 16 из 24 цветных листов в шапке есть цветной знак кривой (подпись / шкала / образец линии), совпадение с кривой эталона 87%;
§6.384: простыми средствами (пиксели, горизонтальные штрихи) набор цветов трека не снимается (41 / 28%, 24 / 46%) — мешают синяя ручка,
печати, сетка. Здесь — модель со зрением в LM Studio (localhost:1234): шапка трека (до 3000 строк над началом кривых, полоса трека по шкалам
рамки ± 20 px), вопрос — какими цветами нарисованы подписи / шкалы / образцы линий кривых трека. Ответ → хроматические семейства
(как `_header_colors.FAM`; чёрный и серый не считаются) против семейств кривых эталона трека (`taskS/color_clash/rows.jsonl`).

  _vlm_colors.py [--model qwen2.5-vl-7b-instruct] [--index F:/nds/output/header_2026-11-16/index.tsv]   → <out>/answers.jsonl, сводка
"""
import sys, argparse, json, pickle, hashlib, base64, io, time, re, urllib.request
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
WORD = {"red": 0, "orange": 1, "brown": 1, "yellow": 2, "green": 3, "teal": 4, "cyan": 4, "turquoise": 4, "blue": 5,
        "purple": 6, "violet": 6, "pink": 7, "magenta": 7}
PROMPT = ("This image is the header of one track of a scanned well-log chart (the curves themselves start at the bottom edge). In such "
          "headers each curve of the track has a label (a probe name such as A2.0M0.5N, ПС, ГК, ГЗ1, or a delay value) and/or a scale bar "
          "with digits and/or a short line sample, drawn in the SAME ink colour as the curve. List the distinct ink colours used for these "
          "curve labels, scale bars and line samples. Ignore the printed form text, stamps, the grid and handwritten notes that are not curve "
          "labels. Answer with JSON only: {\"colors\": [...]} using words from: black, grey, red, orange, brown, yellow, green, teal, blue, "
          "purple, pink.")


def sheet_dir(n):
    return f"{n.stem[:40]}_{hashlib.md5(n.stem.encode('utf-8')).hexdigest()[:8]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-vl-7b-instruct")
    ap.add_argument("--url", default="http://localhost:1234/v1/chat/completions")
    ap.add_argument("--index", default=r"F:/nds/output/header_2026-11-16/index.tsv")
    ap.add_argument("--out", default=r"F:/nds/output/taskS/vlm_colors")
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--max-tokens", type=int, default=300)
    a = ap.parse_args()
    from PIL import Image
    from extract_nlgx import extract
    from auto import imaging as im
    from _header_colors import fam_of_hue, hue_deg
    im.EXIF_TRANSPOSE = False
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    jf = od / "answers.jsonl"
    done = {(d["sheet"], d["track"]) for d in map(json.loads, jf.read_text(encoding="utf-8").splitlines())} if jf.exists() else set()
    sheets = [l.split("\t")[1] for l in Path(a.index).read_text(encoding="utf-8").splitlines() if l.strip()]
    truth = {}
    for l in (TS / "color_clash" / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(l)
        if q["sheet"] in sheets:
            for r in q["recs"]:
                if r["ct"] is not None and (max(r["ct"]) - min(r["ct"])) >= 40:
                    truth.setdefault((q["sheet"], r["track"]), set()).add(fam_of_hue(float(hue_deg(np.array([r["ct"]]))[0])))
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    T0 = time.time()
    for nm in sheets:
        n = SRC[nm]
        src = next((TS / d / (sheet_dir(n) + ".pkl") for d in ("tcache", "tcache_out") if (TS / d / (sheet_dir(n) + ".pkl")).exists()), None)
        if src is None:
            continue
        m = extract(str(n))
        top = int(min(c["top_y"] for c in m["curves"] if c.get("top_y") is not None))
        tracks = {}
        for s in m["scale_axes"]:
            tk = (s["name"].split() + [""])[0]
            if tk.startswith("DA"):
                lo, hi = sorted((s["x_left"], s["x_right"])); a_, b_ = tracks.get(tk, (lo, hi)); tracks[tk] = (min(a_, lo), max(b_, hi))
        todo = [tk for tk in tracks if (nm, tk) not in done]
        if not todo:
            continue
        rgb = im.load_rgb(pickle.load(open(src, "rb"))["image"])
        H, W = rgb.shape[:2]
        for tk in todo:
            lo, hi = tracks[tk]
            y0 = max(0, top - a.rows); x0 = int(max(0, lo - 20)); x1 = int(min(W, hi + 20))
            crop = Image.fromarray(rgb[y0:top + 50, x0:x1])
            sc = a.max_side / max(crop.size)
            if sc < 1:
                crop = crop.resize((max(1, int(crop.width * sc)), max(1, int(crop.height * sc))), Image.LANCZOS)
            b = io.BytesIO(); crop.save(b, "PNG"); b64 = base64.b64encode(b.getvalue()).decode()
            body = {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens, "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
            t1 = time.time()
            try:
                req = urllib.request.Request(a.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                txt = json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["message"]["content"]
            except Exception as e:
                txt = f"ERROR {type(e).__name__}: {e}"
            js = re.findall(r'\{[^{}]*"colou?rs"[^{}]*\}', txt, flags=re.I)          # рассуждающая модель: берётся ПОСЛЕДНИЙ JSON
            words = re.findall(r"[a-z]+", (js[-1] if js else txt).lower())
            fams = sorted({WORD[w] for w in words if w in WORD})
            rec = dict(sheet=nm, track=tk, answer=txt[:400], fams=fams, tru=sorted(truth.get((nm, tk), set())), sec=round(time.time() - t1, 1))
            with open(jf, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  {nm[:40]} {tk}: {fams} / эталон {rec['tru']} ({rec['sec']} с)")
        del rgb
    tp = fn = fp = 0; trk_c = trk_b = fp_b = 0
    for d in map(json.loads, jf.read_text(encoding="utf-8").splitlines()):
        det, tru = set(d["fams"]), set(d["tru"])
        if tru:
            trk_c += 1; tp += len(det & tru); fn += len(tru - det); fp += len(det - tru)
        else:
            trk_b += 1; fp_b += bool(det)
    print(f"★ {a.model}: треков с хроматическими кривыми {trk_c} — полнота {tp / max(tp + fn, 1):.1%} ({tp}/{tp + fn}), точность "
          f"{tp / max(tp + fp, 1):.1%} ({tp}/{tp + fp}); «чёрных» треков {trk_b} — ложный цвет у {fp_b}; {time.time() - T0:.0f} с")


if __name__ == "__main__":
    main()

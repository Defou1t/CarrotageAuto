r"""detect_legend.py — читает ЛЕГЕНДУ из ШАПКИ бланка через VLM (LM Studio): для каждой кривой —
ШТРИХ (сплошной жирный/тонкий/пунктир) и МАСШТАБ в 1 см. Штрих = прямой сигнал ИДЕНТИЧНОСТИ
(идея Эдуарда: в шапке видно каким штрихом какая линия), масштаб = калибровка значений.

Шапка — верх скана (строки 0..top_y, над данными): таблица «Зонд/метод | Колір кривої | Масштаб».
Расширяет detect_scales (тот же VLM-вызов с json_schema constrained-decode).

  python detect_legend.py <scan> [--nlgx <f> | --topy N] [--model google/gemma-3-12b] [--save-crop]
"""
import sys, json, base64, io, urllib.request, re
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

BASE = "http://localhost:1234/v1"
PROMPT = (
    "На изображении — ШАПКА каротажного бланка. Найди таблицу-ЛЕГЕНДУ с колонками "
    "«Зонд /метод/», «Колір кривої» (ОБРАЗЕЦ штриха — как нарисована кривая) и «Масштаб в 1 см». "
    "Таких таблиц может быть ДВЕ рядом (левая и правая = два прохода) — прочитай ОБЕ.\n"
    "Для КАЖДОЙ строки верни: \n"
    "• zond — название зонда/метода (напр. GZ1, GZ2, GZ3, MDS, SP, ПС, A2.0K0.5N, K1.0C);\n"
    "• stroke — тип штриха по образцу: solid_bold (жирная сплошная), solid_thin (тонкая сплошная), "
    "dashed (пунктир ----), dash_dot (штрих-пунктир);\n"
    "• scale — масштаб в 1 см как написано (напр. '10 ом·м', '2 см', '125 мв').\n"
    "Игнорируй прочее (тип лаборатории/кабель/скорость/оператор/даты). ТОЛЬКО JSON, без пояснений."
)
SCHEMA = {"type": "object", "properties": {"curves": {"type": "array", "items": {
    "type": "object", "properties": {
        "zond": {"type": "string"},
        "stroke": {"type": "string", "enum": ["solid_bold", "solid_thin", "dashed", "dash_dot"]},
        "scale": {"type": "string"}},
    "required": ["zond", "stroke", "scale"]}}}, "required": ["curves"]}


def vlm_read(crop, model, timeout=150):
    im = Image.fromarray(crop)
    if im.width > 1100:
        im = im.resize((1100, int(im.height * 1100 / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": url}}]}],
        "temperature": 0.0, "max_tokens": 1800, "reasoning_effort": "low",
        "response_format": {"type": "json_schema",
            "json_schema": {"name": "legend", "strict": True, "schema": SCHEMA}}}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode())
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    cont = (resp.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    try:
        return json.loads(cont).get("curves", []), "ok"
    except Exception:
        m = re.search(r"\{.*\}", cont, re.DOTALL)
        return (json.loads(m.group()).get("curves", []) if m else None), cont[:200]


def main():
    a = sys.argv[1:]
    sys.stdout.reconfigure(encoding="utf-8")
    scan = a[0]
    model = a[a.index("--model") + 1] if "--model" in a else "google/gemma-3-12b"
    rgb = np.asarray(Image.open(scan).convert("RGB"))
    if "--nlgx" in a:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from extract_nlgx import extract
        top_y = int(extract(a[a.index("--nlgx") + 1])["depth_axis"]["top_y"])
    elif "--topy" in a:
        top_y = int(a[a.index("--topy") + 1])
    else:
        top_y = rgb.shape[0] // 13                       # дефолт: ~верхняя 1/13 (шапка над данными)
    crop = rgb[0:max(50, top_y)]
    if "--save-crop" in a:
        Image.fromarray(crop).save(Path(scan).with_suffix(".header.png"))
    curves, status = vlm_read(crop, model)
    print(f"шапка {Path(scan).stem[:40]} (0..{top_y}) status={status}")
    if curves:
        for c in curves:
            print(f"  {c.get('zond',''):<12} штрих={c.get('stroke',''):<10} масштаб={c.get('scale','')}")
    else:
        print("  легенда не прочитана (LM Studio запущен? модель vision?)")


if __name__ == "__main__":
    main()

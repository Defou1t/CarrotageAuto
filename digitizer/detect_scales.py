r"""
detect_scales.py — A1 (Фаза A роадмапа §6.6.7): детект полосы масштабной линейки на скане +
чтение VLM (LM Studio) → масштабы на трек (единицы + числа). Сверка с nlgx scale family (GT).
Линейка задаёт ДО трассировки: какие масштабы есть (1×/5×/25×), их диапазоны, тип трека
(ОМ·М=резистив, СМ=CALI, мВ=SP). Резистив = ×5-цепочка; число строк ОМ·М = глубина перевыносов.

python detect_scales.py --nlgx <f.nlgx> [--model google/gemma-4-26b-a4b] [--band 380] [--save-crop]
"""
import sys, json, re, base64, io, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image
from decode_levels import build_family

BASE = "http://localhost:1234/v1"
PROMPT = (
    "На изображении — МАСШТАБНАЯ ЛИНЕЙКА каротажной диаграммы: горизонтальные строки чисел шкалы "
    "(печатные/рукописные) с единицами, уложенные СТОПКОЙ. Прочитай КАЖДУЮ строку слева-направо.\n"
    "ВАЖНО: строк сопротивления (ОМ·М) обычно НЕСКОЛЬКО, одна под другой — базовая (например 0 5 10 15 20), "
    "затем ×5 (например 0 25 50 75 100), затем ×25 (например 0 125 250 ...). Прочитай ВСЕ такие строки, "
    "даже если видны только первые 1-2 числа (0 25 / 0 125). Также найди строку ПС в милливольтах (мВ, "
    "например -25 0 +25) и строку кавернометрии в сантиметрах (СМ, например 25 30 35 40 45). "
    "Игнорируй прочий текст бланка (скважина, даты, К=, М=, имена зондов H4.0M0.5N).\n"
    'Ответь СТРОГО ОДНИМ JSON-массивом, по объекту на строку сверху-вниз: '
    '[{"unit":"ОМ·М","numbers":[0,5,10,15,20]},{"unit":"ОМ·М","numbers":[0,25]}, ...]. '
    "unit — одно из: ОМ·М, СМ, мВ. Не разобрал строку — пропусти. "
    "ВЫВЕДИ ТОЛЬКО JSON — без рассуждений, пояснений и преамбулы."
)


def crop_ruler(rgb, top_y, band=380):
    y0 = max(0, top_y - band); y1 = min(rgb.shape[0], top_y + 20)
    return rgb[y0:y1]


def vlm_read(crop, model, timeout=180):
    im = Image.fromarray(crop)
    if im.width > 1000:   # меньше vision-токенов: gemma 4096-контекст переполняется на большом изображении
        s = 1000 / im.width
        im = im.resize((1000, int(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    # КОНСТРЕЙН-ДЕКОД через response_format json_schema — заставляет валидный JSON,
    # без преамбулы/рассуждений и без repetition-collapse (надёжнее префилла на reasoning-модели).
    schema = {"type": "object", "properties": {"rows": {"type": "array", "items": {
        "type": "object", "properties": {
            "unit": {"type": "string", "enum": ["ОМ·М", "СМ", "мВ"]},
            "numbers": {"type": "array", "items": {"type": "number"}}},
        "required": ["unit", "numbers"]}}}, "required": ["rows"]}
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": data_url}}]}],
        "temperature": 0.0, "max_tokens": 1500, "reasoning_effort": "low",
        "response_format": {"type": "json_schema",
            "json_schema": {"name": "scales", "strict": True, "schema": schema}}}
    req = urllib.request.Request(f"{BASE}/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode())
    msg = resp.get("choices", [{}])[0].get("message", {})
    cont = (msg.get("content", "") or "").strip()
    try:
        obj = json.loads(cont)
        return obj.get("rows", obj if isinstance(obj, list) else []), msg
    except Exception:
        m2 = re.search(r"\{.*\}", cont, re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group()).get("rows", []), msg
            except Exception:
                pass
    return None, msg


def infer_scales(rows):
    """VLM-строки → структура: резистивные ЦЕПОЧКИ (база + число ×5-уровней), caliper, sp.
    Новая цепочка начинается на 'базовой' строке (много меток, напр. 0 5 10 15 20)."""
    out = {"resistivity_chains": [], "caliper": None, "sp": None}
    cur = None
    for r in rows:
        u = r.get("unit"); nums = [n for n in r.get("numbers", []) if isinstance(n, (int, float))]
        if u == "ОМ·М":
            is_base = len(nums) >= 4
            if is_base or cur is None:
                cur = {"base_labeled": (min(nums), max(nums)) if nums else None, "levels": 1}
                out["resistivity_chains"].append(cur)
            else:
                cur["levels"] += 1
        elif u == "СМ":
            out["caliper"] = (min(nums), max(nums)) if nums else None
        elif u == "мВ":
            out["sp"] = (min(nums), max(nums)) if nums else None
    return out


def gt_families(m):
    """Сводка nlgx scale-семейств на кривую: (тип, число уровней, база v_left..v_right, units)."""
    out = []
    for c in ds.real_curves(m):
        fam = build_family(m, c)
        if not fam:
            continue
        base = fam[0]
        out.append({"curve": c["name"].split()[0], "levels": len(fam),
                    "base": (round(base["v_left"], 2), round(base["v_right"], 2)),
                    "units": base["units"]})
    return out


def main():
    a = sys.argv[1:]
    nlgx = a[a.index("--nlgx") + 1]
    model = a[a.index("--model") + 1] if "--model" in a else "google/gemma-4-26b-a4b"
    band = int(a[a.index("--band") + 1]) if "--band" in a else 380
    sys.stdout.reconfigure(encoding="utf-8")
    m = extract(nlgx)
    img = find_image(Path(nlgx))
    rgb = np.asarray(Image.open(img).convert("RGB"))
    top_y = m["depth_axis"]["top_y"]
    crop = crop_ruler(rgb, top_y, band)
    if "--save-crop" in a:
        p = rf"F:\nds\output\ruler_crop_{Path(nlgx).stem[:30]}.png"
        Image.fromarray(crop).save(p); print(f"crop -> {p} ({crop.shape[1]}x{crop.shape[0]})")
    rows, msg = vlm_read(crop, model)
    print(f"\nVLM ({model}) прочитал строк линейки:")
    if rows:
        for r in rows:
            print(f"   {r}")
    else:
        print("   НЕ распарсилось. raw:", str(msg.get('content') or msg.get('reasoning_content'))[:300])
    if rows:
        inf = infer_scales(rows)
        print("\nИНФЕРЕНС структуры:")
        for i, ch in enumerate(inf["resistivity_chains"]):
            print(f"   резистив-цепочка {i}: база {ch['base_labeled']} ОМ·М, ×5-уровней={ch['levels']}")
        if inf["caliper"]: print(f"   caliper (CALI/DS): {inf['caliper']} СМ")
        if inf["sp"]: print(f"   SP: {inf['sp']} мВ")
    print(f"\nnlgx GT scale-семейства:")
    for g in gt_families(m):
        print(f"   {g['curve']:<6} уровней={g['levels']} база={g['base']} {g['units']}")


if __name__ == "__main__":
    main()

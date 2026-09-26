r"""_vlm_local_test.py — ТЕСТ РАЗВИЛОК (`_vlm_sep_test.py`) НА ЛОКАЛЬНОЙ МОДЕЛИ СО ЗРЕНИЕМ ЧЕРЕЗ LM STUDIO (§6.232, 26.09).

§6.222: на 40 развилках, где прод-селектор ушёл на чужую линию, зрители (Claude) верны в 93%. Можно ли это на машине
заказчика, без облака: те же картинки и тот же вопрос — локальной модели через OpenAI-совместимый сервер LM Studio
(`localhost:1234`). Ключ ответов модели не передаётся; сверка — после.

  _vlm_local_test.py --model qwen2.5-vl-7b-instruct --key <sep_key.json>
"""
import sys, argparse, json, base64, time, re, urllib.request
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--dir", default=r"F:/nds/output/sep_test")
ap.add_argument("--url", default="http://localhost:1234/v1/chat/completions")
ap.add_argument("--out", default="")
ap.add_argument("--max-tokens", type=int, default=3000, help="рассуждающим моделям нужен запас: ответ идёт после рассуждения")
a = ap.parse_args()
PROMPT = ("The image has two halves. LEFT: a raw crop of a scanned paper well-log (several drawn curves, grid lines, text). "
          "RIGHT: the same crop with marks: small GREEN dots mark ONE drawn curve down to a point; the dots stop some rows ABOVE "
          "a junction zone. Below that zone two candidate continuations are marked with small hollow circles: RED circles "
          "labelled \"A\" and BLUE circles labelled \"B\". Exactly one candidate continues the SAME drawn ink line as the green "
          "dots; the other lies on a different line (another curve, a grid/frame line, or noise). Follow the ink through the "
          "unmarked zone (use the LEFT half to see the ink without marks) and decide. Reply with the first line exactly "
          "\"ANSWER: A\" or \"ANSWER: B\", then one short sentence of reasoning.")
key = json.load(open(a.key, encoding="utf-8"))
res = {}; ok = n = 0; t0 = time.time()
for c in sorted(key):
    img = Path(a.dir) / f"{c}.png"
    b64 = base64.b64encode(img.read_bytes()).decode()
    body = {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens,
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
    req = urllib.request.Request(a.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        _msg = json.loads(urllib.request.urlopen(req, timeout=1800).read())["choices"][0]["message"]
        txt = _msg.get("content") or ""
        if not re.search(r"ANSWER:\s*[AB]", txt):        # рассуждающая модель: ответ мог остаться в рассуждении
            txt = (txt + "\n" + (_msg.get("reasoning_content") or "")[-800:]).strip()
    except Exception as e:
        txt = f"ERROR {type(e).__name__}: {e}"
    _all = re.findall(r"ANSWER:\s*([AB])", txt or "")
    ans = _all[-1] if _all else "?"
    good = ans == key[c]["answer"]; ok += good; n += 1
    res[c] = dict(answer=ans, truth=key[c]["answer"], text=(txt or "")[:300])
    print(f"  {c}: {ans} (верно {key[c]['answer']}) {'+' if good else '-'}  {(txt or '').strip().splitlines()[-1][:110] if txt else ''}")
print(f"★ {a.model}: верно {ok} из {n} ({100*ok/max(1,n):.0f}%) за {time.time()-t0:.0f} с")
if a.out:
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

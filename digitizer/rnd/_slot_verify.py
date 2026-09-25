r"""_slot_verify.py — СОВПАДАЕТ ЛИ МЕХАНИЗМ §6.213 СО СТЕНДОМ НА ПОЛЕ (18.09, после A/B `_slot_ab_run.ps1`).

Ловушка §6.146: признак стенда ≠ признак механизма. Здесь по всем листам A/B сверяются:
  1. `nn_base` / `nn_alt` из `<лист>_pick.json` («slots») режима RA — с `route_nn_cache.pkl` стенда
     (непустые строки замороженных выдач A/B). Ожидание: равенство там, где кандидаты те же
     (A ≈ прод, B ≈ декодер без kslots); расхождения — на листах, затронутых kslots (125) и шум прогона.
  2. решение «перевернуть» механизма — с правилом стенда `o_nn ≥ 0.18` на СТЕНДОВЫХ nn.
  3. сколько переворотов всего, из них на гейтованных / за предгейтом, и по K трека.
Печатает доли совпадений; расхождения — списком первых 10.

  <ComfyUI>\python_embeded\python.exe _slot_verify.py --mode RA
"""
import sys, argparse, pickle, json, hashlib, math
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dir", default="ab_slot")
ap.add_argument("--mode", default="RA")
ap.add_argument("--thr", type=float, default=0.18)
ap.add_argument("--sheets", default="wellmap_sheets.txt")
ap.add_argument("--gated", default="kslots_gated.txt")
a = ap.parse_args()
TS = Path(a.ts)
NN = pickle.load(open(TS / "route_nn_cache.pkl", "rb"))
sheets = [l.strip() for l in (TS / a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()]
gated = {l.strip() for l in (TS / a.gated).read_text(encoding="utf-8").splitlines() if l.strip()}
TAG = {"prod": "A", "dec": "B"}

n_sheets = n_json = 0
c = Counter(); mism = []
for sh in sheets:
    stem = Path(sh).stem
    pd = TS / a.dir / a.mode / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    if not pd.is_dir():
        continue
    n_sheets += 1
    pj = next(iter(pd.glob("*_pick.json")), None)
    if not pj:
        c["без _pick.json (декодер не считался)"] += 1
        continue
    n_json += 1
    d = json.loads(pj.read_text(encoding="utf-8"))
    if not d.get("gated", True):
        c["листов за предгейтом с декодером"] += 1
    for q in d.get("slots", []):
        c["слотов"] += 1
        base_tag, alt_tag = TAG[q["base"]], TAG["dec" if q["base"] == "prod" else "prod"]
        sb = NN.get((sh, q["track"], q["name"], base_tag))
        sa = NN.get((sh, q["track"], q["name"], alt_tag))
        if sb is None and sa is None:
            c["слот не в стенде"] += 1
            continue
        nb = sb[0] if sb else 0
        na = sa[0] if sa else 0
        eq_b = nb == q["nn_base"]; eq_a = na == q["nn_alt"]
        c["nn_base совпал"] += eq_b; c["nn_alt совпал"] += eq_a
        c["оба nn совпали"] += eq_b and eq_a
        pred = na >= 30 and (math.log1p(na) - math.log1p(nb)) >= a.thr
        c["решение совпало"] += (pred == q["flip"])
        c["перевороты механизма"] += q["flip"]
        c["перевороты стенда (на его nn)"] += pred
        if q["flip"]:
            c["перевороты: гейт" if sh in gated else "перевороты: за предгейтом"] += 1
        if pred != q["flip"] and len(mism) < 10:
            mism.append((sh[:36], q["name"], q["base"], q["nn_base"], q["nn_alt"], nb, na, q["flip"]))

print(f"листов с выдачей {n_sheets} из {len(sheets)}; с _pick.json {n_json}; "
      f"за предгейтом с декодером {c['листов за предгейтом с декодером']}")
S = c["слотов"]
if S:
    print(f"слотов {S}: не в стенде {c['слот не в стенде']}; nn_base совпал {c['nn_base совпал']} "
          f"({100*c['nn_base совпал']/S:.1f}%), nn_alt совпал {c['nn_alt совпал']} ({100*c['nn_alt совпал']/S:.1f}%), "
          f"оба {c['оба nn совпали']} ({100*c['оба nn совпали']/S:.1f}%)")
    print(f"решение «перевернуть» совпало со стендом на {c['решение совпало']} из {S} ({100*c['решение совпало']/S:.1f}%); "
          f"переворотов механизма {c['перевороты механизма']} (гейт {c['перевороты: гейт']}, за гейтом "
          f"{c['перевороты: за предгейтом']}), стенда на его nn {c['перевороты стенда (на его nn)']}")
    if mism:
        print("первые расхождения решений (лист, слот, база, nn_base/alt механизма, nn стенда, flip):")
        for m in mism:
            print("  ", m)

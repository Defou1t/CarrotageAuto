r"""_prod_gate.py — ШИРОКИЙ ПРОД-ГЕЙТ: что пайплайн реально пишет ПОКРИВОЙ после работы дня.

Правила проекта: оракульный успех обязан подтвердиться в проде (§6.16); честных цитировать
только med<=3px И cov>=0.9 (§6.9); проверять утечку эталона auto.xs != gt.xs (§4). Этот гейт —
полный пайплайн (understand+trace+emit) + гейт покривой с ТАКСОНОМИЕЙ ОТКАЗА, чтобы видеть НЕ
среднее, а ПОЧЕМУ слот не взят:
  ЧЕСТНАЯ  — med<=3 И cov>=0.9;   ТОЧНА_НО_НЕПОЛНА — med<=3, cov<0.9;
  ИДЕНТ    — записана, но med>10 (ушла на соседа/латч);   НЕ_ЗАПИСАНА — слот пуст;
  LEAK     — в nlgx осталась экспертная трасса (ловушка §4).

  python _prod_gate.py --set semeguniv       # 30 листов Semeguniv_001
  python _prod_gate.py --set multi --tag base
"""
import sys, io, json, time, argparse, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image
from auto.pipeline import run as pipe_run
from auto.config import Config
from _u1_gate import sheets_for       # та же выборка, что U1-гейт

OUT = Path(r"F:\nds\output\taskS\prod")
ARCH = Path(r"F:\nds\projects\Archive")


def archive_sample(per_well, seed_shift=0):
    """Стратифицированная выборка АРХИВА: до per_well листов с каждой скважины, round-robin по
    скважинам (не по алфавиту — иначе первые скважины перевешивают). Даёт широту по 46 скважинам
    и по классам сразу. Детерминизм без random: берём каждый k-й лист скважины."""
    wells = sorted([d for d in ARCH.iterdir() if d.is_dir()])
    picks = []
    for w in wells:
        sheets = sorted((w / "wlg").glob("*.nlgx")) if (w / "wlg").is_dir() else sorted(w.glob("*.nlgx"))
        if not sheets:
            continue
        step = max(1, len(sheets) // per_well)
        sel = sheets[seed_shift % max(1, step)::step][:per_well]
        picks.append((w.name, sel))
    # round-robin: по одному листу с каждой скважины за круг
    out, i = [], 0
    while True:
        added = False
        for _, sel in picks:
            if i < len(sel):
                out.append(sel[i]); added = True
        if not added:
            break
        i += 1
    return out


def gate_sheet(n):
    img = find_image(n)
    if not img:
        return {"sheet": n.stem, "err": "нет картинки"}
    cfg = Config(); cfg.out = OUT / "nlgx"
    cfg.out.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            sheet, traces, res = pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
    except Exception as e:
        return {"sheet": n.stem, "err": f"{type(e).__name__}: {e}"}
    mo = extract(str(n))
    gts = [c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])]
    aup = Path(res.get("nlgx", ""))
    byname = {}
    if aup.is_file():
        byname = {c["name"]: c for c in extract(str(aup)).get("curves", [])}
    rec = {"sheet": n.stem, "n_lines": len(sheet.lines), "n_curves": len(gts),
           "sec": round(time.time() - t0, 1), "curves": []}
    for g in gts:
        nm = g["name"].split()[0]
        gxs = {g["top_y"] + i: x for i, x in enumerate(g["xs"]) if x != NULL}
        ac = byname.get(g["name"])
        if ac is None:
            rec["curves"].append({"name": nm, "state": "НЕ_ЗАПИСАНА"}); continue
        if ac["xs"] == g["xs"]:
            rec["curves"].append({"name": nm, "state": "LEAK"}); continue
        axs = {ac["top_y"] + i: x for i, x in enumerate(ac["xs"]) if x != NULL}
        common = [y for y in axs if y in gxs]
        if not common:
            rec["curves"].append({"name": nm, "state": "НЕ_ЗАПИСАНА"}); continue
        d = np.array([abs(axs[y] - gxs[y]) for y in common], float)
        med = float(np.median(d)); cov = len(common) / max(1, len(gxs))
        if med <= 3 and cov >= 0.9:
            state = "ЧЕСТНАЯ"
        elif med <= 3:
            state = "ТОЧНА_НО_НЕПОЛНА"
        elif med > 10:
            state = "ИДЕНТ"
        else:
            state = "БЛИЗКО"
        rec["curves"].append({"name": nm, "state": state, "med": round(med, 1), "cov": round(cov, 2)})
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--per-well", type=int, default=4, help="листов на скважину для --set archive")
    ap.add_argument("--max-mpx", type=float, default=120.0, help="пропускать сканы крупнее (по nlgx-габаритам нельзя, по картинке)")
    a = ap.parse_args()
    from collections import Counter
    import cv2
    sheets = archive_sample(a.per_well) if a.set == "archive" else sheets_for(a.set)
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"prod_{a.set}_{a.tag}.json"
    rows = []
    tot = Counter()
    skipped_big = []
    for n in sheets:
        img = find_image(n)
        if img and a.max_mpx:                       # ГАРД РАЗМЕРА: не топить прогон одним 300-Мпикс листом
            hdr = cv2.imread(str(img), cv2.IMREAD_REDUCED_COLOR_8)
            if hdr is not None and (hdr.shape[0] * hdr.shape[1] * 64) / 1e6 > a.max_mpx:
                skipped_big.append(n.stem); continue
        r = gate_sheet(n)
        rows.append(r)
        if "err" in r:
            print(f"  !! {r['sheet'][:50]:<52} {r['err']}")
        else:
            st = Counter(c["state"] for c in r["curves"])
            tot.update(st)
            print(f"  {r['sheet'][:50]:<52} {st.get('ЧЕСТНАЯ',0)}/{r['n_curves']} честных  "
                  f"[{dict(st)}]  ({r['sec']}s)")
        fp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")  # инкрементально
    ncur = sum(len(r.get("curves", [])) for r in rows if "err" not in r)
    nsheet = sum(1 for r in rows if "err" not in r)
    print(f"\n=== ИТОГО {nsheet} листов, {ncur} кривых ===")
    print(f"  ★ ЧЕСТНЫХ (med<=3 И cov>=0.9): {tot.get('ЧЕСТНАЯ',0)}/{ncur} = "
          f"{100.0*tot.get('ЧЕСТНАЯ',0)/max(1,ncur):.1f}%")
    print(f"  таксономия отказа: {dict(tot)}")
    if skipped_big:
        print(f"  ⚠ ПРОПУЩЕНО по размеру (>{a.max_mpx:.0f}Мпикс): {len(skipped_big)} листов "
              f"— НЕ входят в проценты: {', '.join(skipped_big[:6])}{'...' if len(skipped_big)>6 else ''}")
    print(f"  -> {fp}")

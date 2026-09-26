r"""_slot_cache_data.py — ВЫБОРКА ДЛЯ РАСКЛАДКИ ИЗ КЭША ТРАСС: ВСЕ ПАРЫ (СЛОТ × ЛИНИЯ ТРЕКА) С ЦЕЛЬЮ (§6.230, 26.09).

§6.228–§6.229: у 470 кривых честная трасса уже есть среди кандидатов, но раскладка отдаёт слот другой линии; правила без
обучения (дубли, добор) не помогли. Прод-раскладка — обученная (`auto/slot_model.py`, `slot_model_g250.npz`: бустинг по
14 признакам пары, жадное 1:1), но обучена давно — на пулах старой конфигурации и только по трассам прод-пути.
Здесь — выборка из КЭША (1434 листа, трассы обоих путей ровно те, что получает `emit`): признаки — ПРОД-функцией
`slot_model.rows` (с запретом геометрии `slot_geom`, как в проде), цель — та же, что у действующей модели
(`log1p(медиана px) + 3·max(0, 0.9 − покрытие)`), метка скважины — для честного разбиения по фолдам.

  _slot_cache_data.py --cache F:/nds/output/taskS/tcache --out F:/nds/output/taskS/slotcv/slotdata.npz
"""
import sys, argparse, pickle, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M, emit as E, slot_model as SM, slot_geom
from auto.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--out", default=r"F:/nds/output/taskS/slotcv/slotdata.npz")
a = ap.parse_args()
TS = Path(a.ts)
wm = json.load(open(TS / "rowdec_wellmap.json", encoding="utf-8"))
cfg = Config(); cv = cfg.cv; MN = cfg.mnemonics


def unpack(t):
    rows, xs = t
    return dict(zip(rows.tolist(), xs.tolist()))


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    return float(np.median([abs(tr[y] - gt[y]) for y in com])), len(com) / max(1, len(gt))


XA, YA, SA, WA, PA, HA = [], [], [], [], [], []
files = sorted(Path(a.cache).glob("*.pkl"))
for i, f in enumerate(files, 1):
    v = pickle.load(open(f, "rb"))
    fn = v["frame_nlgx"]; frame = v["frame"]
    model = extract(str(fn))
    gts = {c["name"]: dense(c) for c in model["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    slots = []
    for c in model.get("curves", []):
        nm = c.get("name", "")
        if M.mnem_root(nm) != "DA":
            info = M.curve_info(nm, MN)
            slots.append(dict(name=nm, track=E._slot_track(model, c, frame), color=info["color"], cls=info["class"]))
    if not slots:
        continue
    forbid = slot_geom.make_forbid(model, cv)
    well = wm.get(Path(fn).stem, Path(fn).parent.parent.name)
    for path, lst in (("prod", v["traces"]), ("dec", v["alt"] or [])):
        by_track = {}
        for L, t in lst:
            by_track.setdefault(L.track_index, []).append((L, unpack(t)))
        X, pairs = SM.rows(slots, by_track, forbid)
        for k, (nm, L, tr) in enumerate(pairs):
            if nm not in gts:
                continue
            m, c = err(tr, gts[nm])
            y = np.log1p(m if m is not None else 5000.0) + 3.0 * max(0.0, 0.9 - c)
            XA.append(X[k]); YA.append(y); SA.append(Path(fn).name); WA.append(well); PA.append(path)
            HA.append(1 if (m is not None and m <= 3.0 and c >= 0.9) else 0)
    if i % 100 == 0:
        print(f"  … {i}/{len(files)}, пар {len(YA)}", flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out, X=np.array(XA, np.float64), y=np.array(YA), sheet=np.array(SA), well=np.array(WA),
                    path=np.array(PA), hon=np.array(HA, np.int8))
print(f"★ пар {len(YA)}, листов {len(set(SA))}, скважин {len(set(WA))}, честных пар {int(sum(HA))} → {a.out}")

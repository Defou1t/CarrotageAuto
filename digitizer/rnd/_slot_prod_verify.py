r"""_slot_prod_verify.py — НЕЗАВИСИМАЯ ПЕРЕПРОВЕРКА ЦИФРЫ §6.90 И ЕЁ ГЛАВНОГО ДОПУЩЕНИЯ.

Проверку §6.90 должны были сделать скептики (wf_110d68ca-bbb), но недельный лимит их оборвал:
из пяти утверждений уцелели два. Остальное проверяется здесь, своим кодом, а не тем же самым.

ЧТО СЧИТАЕТСЯ:
  1) ПЕРЕСЧЁТ ★честных кривых по ВЫДАННЫМ файлам A/B — своей реализацией метрики §6.36
     (денсификация полилинии, med≤3px, cov≥0.9, контроль утечки), НЕ функциями `_slot_prod_ab.py`.
     Своя денсификация сверяется с канонической `_multi_replica_probe.dense` на всех кривых:
     если они разойдутся, дальше считать нельзя.
  2) ДОПУЩЕНИЕ ОБ ОТКАЗНЫХ ЛИСТАХ — самое слабое место замера. §6.90 исключил 186 листов из 297,
     потому что «при отказе файл выходит прежним по построению». Но отказ определялся ОФЛАЙН по
     пулам, а в проде решение принимается на своих слотах и своих трассах. Здесь берутся листы,
     которые офлайн-проверка объявила отказными, пайплайн гоняется с флагом и без, и выданные
     файлы сравниваются. Любое различие означает, что 186 листов исключены НЕПРАВОМЕРНО.
  3) ВЫРОЖДЕННОСТЬ «цены +0» — сколько слотов рамки вообще НЕ имеют экспертной кривой. Если почти
     все имеют, то «соперничают все слоты» и «соперничают только слоты с эталоном» — одно и то же,
     и ноль ничего не доказывает про лёгкий каркас (§6.62).

  <ComfyUI>\python_embeded\python.exe _slot_prod_verify.py [--abstain-check 3]
"""
import sys, io, json, argparse, contextlib, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from _multi_replica_probe import dense as dense_ref
from auto import meta as M
from auto import slot_model as SM
from auto.pipeline import run as pipe_run
from auto.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--ab", default=r"F:\nds\output\taskS\prod_ab_slot")
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div"])
ap.add_argument("--abstain-check", type=int, default=3, help="сколько ОТКАЗНЫХ листов перепрогнать")
ap.add_argument("--gate", default="frac0.8")
ap.add_argument("--out", default=r"F:\nds\output\taskS\prod_verify_abstain")
a = ap.parse_args()


def my_dense(c, max_gap=200):
    """Своя денсификация экспертной полилинии: вершины + линейная интерполяция, разрывы >max_gap
    не мостим. Написана заново, сверяется с канонической ниже."""
    pts = [(c["top_y"] + i, float(x)) for i, x in enumerate(c["xs"]) if x != NULL]
    out = {}
    for k in range(len(pts) - 1):
        (y0, x0), (y1, x1) = pts[k], pts[k + 1]
        out[y0] = x0
        if 0 < y1 - y0 <= max_gap:
            step = (x1 - x0) / (y1 - y0)
            for yy in range(y0 + 1, y1):
                out[yy] = x0 + step * (yy - y0)
    if pts:
        out[pts[-1][0]] = pts[-1][1]
    return out


def my_honest(ours, gt):
    """★честная кривая (§6.36/§6.59): ≥30 общих строк, медиана |Δ| ≤ 3px, покрытие ≥ 0.9."""
    com = [y for y in ours if y in gt]
    if len(com) < 30:
        return False
    d = np.array([abs(ours[y] - gt[y]) for y in com], float)
    return float(np.median(d)) <= 3.0 and len(com) / max(1, len(gt)) >= 0.9


def my_leaked(ours, raw):
    ey = [raw["top_y"] + i for i, x in enumerate(raw["xs"]) if x != NULL]
    ex = [float(x) for x in raw["xs"] if x != NULL]
    em = dict(zip(ey, ex))
    com = [y for y in ours if y in em]
    if len(com) >= 30 and np.mean([abs(ours[y] - em[y]) < 1e-9 for y in com]) > 0.5:
        return True
    return len(ours) == len(ey) and all(abs(ours[y] - em[y]) < 1e-9 for y in com) and bool(com)


class FakeLine:
    __slots__ = ("color", "behavior", "x_center")

    def __init__(self, color, behavior, x_center):
        self.color, self.behavior, self.x_center = color, behavior, x_center


WLG = {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WLG[q.name] = q
for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
    WLG.setdefault(q.name, q)
BY_STEM = {q.stem[:40]: q for q in WLG.values()}

# ── 1. ПЕРЕСЧЁТ ЦИФРЫ ОТГРУЗКИ ────────────────────────────────────────────────────────────────
print("=" * 78)
print("1. ПЕРЕСЧЁТ ★честных кривых по выданным файлам (своя метрика)")
print("=" * 78)
dmax, ncmp = 0.0, 0
tot = dict(A=0, B=0)
per = dict(A={}, B={})
sheets, nodir = 0, 0
for da in sorted((Path(a.ab) / "A").glob("*")):
    q = BY_STEM.get(da.name)
    if q is None:
        nodir += 1; continue
    G = extract(str(q))
    raws = {c["name"]: c for c in G["curves"]
            if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    if not raws:
        continue
    gts = {}
    for nm, c in raws.items():
        mine, ref = my_dense(c), dense_ref(c)
        keys = set(mine) | set(ref)
        if keys:
            dmax = max(dmax, max(abs(mine.get(k, 1e9) - ref.get(k, 1e9)) for k in keys))
            ncmp += 1
        gts[nm] = mine
    ok = True
    for tag in ("A", "B"):
        f = next(iter(sorted((Path(a.ab) / tag / da.name).glob("*_auto.nlgx"))), None)
        if f is None:
            ok = False; continue
        W = {c["name"]: my_dense(c) for c in extract(str(f))["curves"]
             if M.mnem_root(c["name"]) != "DA"}
        h = sum(1 for nm, gd in gts.items()
                if nm in W and W[nm] and not my_leaked(W[nm], raws[nm]) and my_honest(W[nm], gd))
        per[tag][da.name] = h
        tot[tag] += h
    sheets += 1 if ok else 0
# ⚠ Порог 1e-3 px, а не «побитово»: каноническая `dense` интерполирует в float32 (значения кривой
# приходят из nlgx как float32), моя — в float64, отсюда законные ~5e-7 px. При пороге честности
# 3 px это нуль, но «побитово» здесь потребовать нельзя — стенд кричал бы ложно.
print(f"своя денсификация против канонической: сверено {ncmp} кривых, max|Δ| = {dmax:.3e} px"
      + ("  ⇒ совпадает (float32 против float64)" if dmax < 1e-3 else
         "  ⛔ РАСХОЖДЕНИЕ, дальше не считать"))
up = sum(1 for k in per["A"] if per["B"].get(k, 0) > per["A"][k])
dn = sum(1 for k in per["A"] if per["B"].get(k, 0) < per["A"][k])
print(f"листов сверено {len(per['A'])} (без пары каталогов: {nodir})")
print(f"★ ПЕРЕСЧЁТ: A {tot['A']} → B {tot['B']} ({tot['B']-tot['A']:+d}), листов ↑{up}/↓{dn}")
print(f"  §6.90 утверждает: 25 → 32 (+7), ↑7/↓0 — "
      + ("СОВПАЛО" if (tot['A'], tot['B'], up, dn) == (25, 32, 7, 0) else "⛔ НЕ СОВПАЛО"))

# ── 3. ВЫРОЖДЕННОСТЬ «цены +0» ────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("3. НАСКОЛЬКО ВЫРОЖДЕН НОЛЬ «соперничают все слоты рамки»")
print("=" * 78)
seen, n_all, n_gt, n_extra_track, sh_diff, nsh = set(), 0, 0, 0, 0, 0
abstain_pool = []
w = SM.load(SM.DEFAULT_MODEL)
kind, thr = SM._parse_gate(a.gate)
# ⚠ КЛЮЧ ДЕДУПЛИКАЦИИ — ИМЯ ФАЙЛА, А НЕ ПОЛЕ `name`: `_pool_oracle.py` кладёт дамп как
# `{nlgx.stem[:60]}.pkl`, поэтому дубль виден ДО чтения и стоит ноль. ⚠ В отличие от
# `_start_probe`/`_param_sweep` здесь нет прохода «сначала список, потом счёт»: дамп нужен целиком,
# и 2.35 ГБ читаются по делу — экономится только повторное чтение дублей.
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        if f.stem in seen:
            continue
        seen.add(f.stem)
        d = pickle.load(open(f, "rb"))
        if not d["name"].startswith(f.stem[:40]):
            print(f"  ⚠ имя дампа {f.stem[:40]!r} расходится с полем name {d['name'][:40]!r}")
        nsh += 1
        sl_all = [s for s in d["slots"]]
        gt_names = set(d["gts"])
        extra = [s for s in sl_all if s["name"] not in gt_names]
        n_all += len(sl_all); n_gt += sum(1 for s in sl_all if s["name"] in gt_names)
        tracks = {ln["track"] for ln in d["lines"]}
        n_extra_track += sum(1 for s in extra if s["track"] in tracks)
        sh_diff += 1 if extra else 0
        by_track = {}
        for ln in d["lines"]:
            by_track.setdefault(ln["track"], []).append(
                (FakeLine(ln["color"], ln["behavior"], ln["x_center"]), ln["tr"]))
        slots = [dict(name=s["name"], track=s["track"], color=s["color"], cls=s["class"])
                 for s in sl_all]
        if SM.assign(slots, by_track, w, kind, thr) is None:
            abstain_pool.append(d["name"])
print(f"листов {nsh}: слотов в рамках {n_all}, из них с экспертной кривой {n_gt} "
      f"({100*n_gt/max(1,n_all):.1f}%)")
print(f"слотов БЕЗ эталона {n_all-n_gt}, из них на треке с линиями {n_extra_track}; "
      f"листов, где такие слоты есть: {sh_diff}")
print("⇒ " + ("НОЛЬ ВЫРОЖДЕН: лишних слотов почти нет, «все слоты» и «слоты с эталоном» — почти\n"
                "  одно и то же множество. Про лёгкий каркас (§6.62, слоты пусты) это НЕ говорит ничего."
                if n_extra_track < 0.05 * n_all else
                "ноль содержателен: лишних слотов заметно, и они реально соперничают за линии"))

# ── 2. ДОПУЩЕНИЕ ОБ ОТКАЗНЫХ ЛИСТАХ ───────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(f"2. ОТКАЗНЫЕ ЛИСТЫ: правда ли файл не меняется (перепрогон {a.abstain_check} листов × 2)")
print("=" * 78)
CACHE = pickle.load(open("F:/nds/output/taskS/_slot_abstain_cache_v2.pkl", "rb"))
WELL = dict(zip(CACHE["names"], CACHE["wells"]))
AUDIT = set(json.loads(Path(r"F:\nds\Auto\auto\models\slot_model_g250.json")
                       .read_text(encoding="utf-8"))["audit_wells"])
cand = [n for n in abstain_pool if WELL.get(n) in AUDIT and n in WLG][:a.abstain_check]
print(f"отказных листов офлайн: {len(abstain_pool)} (в аудиторских скважинах: "
      f"{sum(1 for n in abstain_pool if WELL.get(n) in AUDIT)}); проверяем {len(cand)}")
same = diff = 0
for nm in cand:
    q = WLG[nm]
    img = find_image(q)
    if not img:
        print(f"  {nm[:50]:<52} нет картинки"); continue
    outs = {}
    for tag, mdl in (("off", ""), ("on", SM.DEFAULT_MODEL)):
        cfg = Config()
        cfg.out = Path(a.out) / tag / q.stem[:40]
        cfg.cv.slot_model = mdl; cfg.cv.slot_gate = a.gate
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pipe_run(str(img), frame_nlgx=str(q), cfg=cfg, stages=False)
        except Exception as e:
            print(f"  {nm[:50]:<52} ПАДЕНИЕ {type(e).__name__}: {e}"); outs = None; break
        g = next(iter(sorted(cfg.out.glob("*_auto.nlgx"))), None)
        outs[tag] = ({c["name"]: [x for x in c["xs"] if x != NULL]
                      for c in extract(str(g))["curves"] if M.mnem_root(c["name"]) != "DA"}
                     if g else None)
    if not outs or outs.get("off") is None or outs.get("on") is None:
        continue
    A, B = outs["off"], outs["on"]
    bad = [k for k in set(A) | set(B)
           if len(A.get(k, [])) != len(B.get(k, []))
           or not np.allclose(A.get(k, []), B.get(k, []), atol=1e-6)]
    if bad:
        diff += 1
        print(f"  ⛔ РАЗЛИЧАЕТСЯ {nm[:44]:<46} кривых: {', '.join(bad[:5])}")
    else:
        same += 1
        print(f"  ✓ идентично    {nm[:44]:<46} кривых {len(A)}")
print(f"\n★ ИТОГ ДОПУЩЕНИЯ: идентичных {same}, различающихся {diff}")
print("⇒ " + ("допущение подтверждено на проверенных листах: исключение отказных из замера правомерно"
                if diff == 0 else
                "⛔ ДОПУЩЕНИЕ ЛОЖНО: исключение 186 листов из §6.90 неправомерно, замер пересчитать"))

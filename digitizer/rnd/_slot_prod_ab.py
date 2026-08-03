r"""_slot_prod_ab.py — A/B ОБУЧЕННОЙ РАСКЛАДКИ НА ОТГРУЖАЕМОМ ПУТИ (§6.90). По ВЫДАННЫМ файлам.

⚠⚠ ЗАЧЕМ. §6.88-§6.89 померили механизм на ПУЛАХ: честность считалась у трассы, назначенной слоту.
В файл же уходит кривая после пост-обработки (уровни-перевыносы, NULL-разрывы, ветка масштаба
§6.77/§6.78). Память проекта: стенд и отгрузка расходятся в разы, и цитировать надо отгрузку.
Здесь пайплайн прогоняется ДВАЖДЫ на одних листах — `slot_model=""` и `slot_model=<вес>` — и
честные кривые считаются в `<stem>_auto.nlgx`, а не в пуле.

★ ЛИСТЫ, НА КОТОРЫХ МОДЕЛЬ ОТКАЗЫВАЕТСЯ, НЕ ГОНЯЮТСЯ ВОВСЕ: при отказе `map_lines` возвращает
None, раскладка идёт тем же правилом, и файл выходит бит-в-бит прежним. Отказ определяется офлайн
по пулам (`slot_model.assign`), поэтому прогон сокращается втрое. ⚠ Это верно ровно пока пулы
собраны ТЕМ ЖЕ кодом трассировки — иначе список отказов будет не тот (§6.71).

★ СКВАЖИНЫ ВЗЯТЫ АУДИТОРСКИЕ: вес обучен БЕЗ них (`slot_model_g250.json::audit_wells`).

  <ComfyUI>\python_embeded\python.exe _slot_prod_ab.py [--cap 13] [--gate frac0.8]
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
from _multi_replica_probe import dense
from auto import meta as M
from auto import slot_model as SM
from auto.pipeline import run as pipe_run
from auto.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div"])
ap.add_argument("--gate", default="frac0.8")
ap.add_argument("--model", default=SM.DEFAULT_MODEL)
ap.add_argument("--cap", type=int, default=13, help="листов на скважину (0 = все)")
ap.add_argument("--out", default=r"F:\nds\output\taskS\prod_ab_slot")
# ★ ШАРД (правило Эдуарда 28.07: что можно — считать на ПК, и не одним потоком). Полный прогон
# 111 листов × 2 режима — 2.5 часа последовательно; 8 шардами на 16 ядрах — 20 минут. Каталоги
# выдачи у шардов не пересекаются (имя листа), поэтому агрегировать можно тем же `_slot_prod_ab`
# с --shard 0/1 либо `_slot_prod_verify.py`, который читает выдачу с диска.
ap.add_argument("--shard", default="0/1", help="i/N — взять i-й БЛОК из N (не каждый N-й: блок "
                                               "сохраняет локальность чтения сканов)")
a = ap.parse_args()
SH_I, SH_N = (int(v) for v in a.shard.split("/"))

HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(ours, gt):
    com = [y for y in ours if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(ours[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


def leaked(tr, raw):
    """Контроль копирования эксперта (§6.36)."""
    er = np.array([raw["top_y"] + i for i, x in enumerate(raw["xs"]) if x != NULL])
    ex = np.array([x for x in raw["xs"] if x != NULL], float)
    oy = np.array(sorted(tr)); ox = np.array([tr[y] for y in oy], float)
    if len(oy) == len(er) and np.array_equal(oy, er) and np.allclose(ox, ex):
        return True
    com = np.intersect1d(oy, er)
    if len(com) >= 30:
        om = dict(zip(oy.tolist(), ox.tolist())); em = dict(zip(er.tolist(), ex.tolist()))
        if np.mean([abs(om[y] - em[y]) < 1e-9 for y in com]) > 0.5:
            return True
    return False


class FakeLine:
    __slots__ = ("color", "behavior", "x_center")

    def __init__(self, color, behavior, x_center):
        self.color, self.behavior, self.x_center = color, behavior, x_center


AUDIT = set(json.loads(Path(r"F:\nds\Auto\auto\models\slot_model_g250.json")
                       .read_text(encoding="utf-8"))["audit_wells"])
CACHE = pickle.load(open("F:/nds/output/taskS/_slot_abstain_cache_v2.pkl", "rb"))
WELL = dict(zip(CACHE["names"], CACHE["wells"]))
w = SM.load(a.model)
kind, thr = SM._parse_gate(a.gate)

# ── какие листы модель ТРОГАЕТ (на прочих файл не меняется) ────────────────────────────────────
WLG = {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WLG[q.name] = q
for e in (r"F:\nds\projects\Semeguniv_001\wlg",):
    for q in Path(e).glob("*.nlgx"):
        WLG.setdefault(q.name, q)

# ⚠⚠ СКВАЖИНА ЛИСТА ОПРЕДЕЛЯЕТСЯ ПО ИМЕНИ ФАЙЛА, ДО `pickle.load`. Прежний цикл читал ВСЕ пять
# каталогов (702 дампа, 2.35 ГБ) и лишь потом выяснял, что лист не из аудиторской скважины и не
# нужен; на шардах это умножалось на число процессов. `_pool_oracle.py` кладёт дамп как
# `{nlgx.stem[:60]}.pkl`, а имена листов в кэше — те же самые, поэтому имя файла и есть ключ: с
# диска читаются только аудиторские дампы. Совпадение ключа с `d["name"]` проверяется при чтении.
BY_STEM = {Path(n).stem[:60]: n for n in CACHE["names"]}
touched, per_well, skipped = [], {}, 0
seen = set()
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        sheet = BY_STEM.get(f.stem)
        if f.stem in seen or WELL.get(sheet) not in AUDIT:
            continue
        seen.add(f.stem)
        d = pickle.load(open(f, "rb"))
        if d["name"] != sheet:
            print(f"  ⚠ имя дампа {f.stem[:40]!r} расходится с полем name {d['name'][:40]!r}")
        by_track = {}
        for ln in d["lines"]:
            by_track.setdefault(ln["track"], []).append(
                (FakeLine(ln["color"], ln["behavior"], ln["x_center"]), ln["tr"]))
        slots = [dict(name=s["name"], track=s["track"], color=s["color"], cls=s["class"])
                 for s in d["slots"]]
        if SM.assign(slots, by_track, w, kind, thr) is None:
            skipped += 1
            continue
        wl = WELL[d["name"]]
        if a.cap and len(per_well.get(wl, [])) >= a.cap:
            continue
        q = WLG.get(d["name"])
        if q is None:
            continue
        per_well.setdefault(wl, []).append(d["name"])
        touched.append(q)
print(f"аудиторских скважин {len(AUDIT)}, листов в пулах {len(seen)}; модель ОТКАЗЫВАЕТСЯ на "
      f"{skipped} (файл не меняется, не гоняем)")
print(f"★ затронуто {len(touched)} листов: " +
      ", ".join(f"{k}:{len(v)}" for k, v in sorted(per_well.items())))
if SH_N > 1:
    # ★ БЛОКОМ, А НЕ ЧЕРЕЗ ШАГ: сканы одной скважины лежат в одном каталоге, а имена листов
    # начинаются со скважины ⇒ блок оставляет шарду локальность чтения, шаг `i % N` раскидывал
    # каждый процесс по всему архиву и гонял головку HDD впустую.
    touched.sort(key=lambda p: p.name)
    lo = len(touched) * SH_I // SH_N; hi = len(touched) * (SH_I + 1) // SH_N
    touched = touched[lo:hi]
    print(f"★ ШАРД {SH_I}/{SH_N}: мой кусок — {len(touched)} листов × 2 режима")

# ── два прогона ───────────────────────────────────────────────────────────────────────────────
res, FP = {}, {"A прод (правило)": {}, "B обученная раскладка": {}}
for tag, mdl in (("A прод (правило)", ""), ("B обученная раскладка", a.model)):
    tot = dict(hon=0, curves=0, sheets=0, leak=0)
    per = {}
    print(f"\n{'='*78}\n{tag}: slot_model={mdl!r}\n{'='*78}")
    for n in touched:
        img = find_image(n)
        if not img:
            print(f"  {n.stem[:44]:<46} нет картинки"); continue
        cfg = Config()
        cfg.out = Path(a.out) / ("A" if not mdl else "B") / n.stem[:40]
        cfg.cv.slot_model = mdl                    # §6.71: режим ПИННИТСЯ стендом, не наследуется
        cfg.cv.slot_gate = a.gate
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
        except Exception as e:
            print(f"  {n.stem[:44]:<46} ПАДЕНИЕ {type(e).__name__}: {e}"); continue
        G = extract(str(n))
        raws = {c["name"]: c for c in G["curves"]
                if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        gts = {nm: dense(c) for nm, c in raws.items()}
        got = next(iter(sorted(cfg.out.glob("*_auto.nlgx"))), None)
        if got is None:
            print(f"  {n.stem[:44]:<46} файл не выдан"); continue
        W = {c["name"]: dense(c) for c in extract(str(got))["curves"]
             if M.mnem_root(c["name"]) != "DA"}
        # ⚠ отпечаток ВЫДАЧИ: равные счётчики честности НЕ доказывают, что флаг сработал
        FP[tag][n.name] = tuple(sorted((nm, len(v), round(float(np.median(list(v.values()))), 3))
                                      for nm, v in W.items() if v))
        h = lk = 0
        for nm, gd in gts.items():
            if nm not in W or not W[nm]:
                continue
            if leaked(W[nm], raws[nm]):
                lk += 1; continue
            h += HON(*err(W[nm], gd))
        per[n.name] = h
        tot["hon"] += h; tot["curves"] += len(gts); tot["sheets"] += 1; tot["leak"] += lk
        print(f"  {n.stem[:42]:<44} кривых {len(gts):>2}  ЧЕСТНЫХ {h}")
    res[tag] = (tot, per)
    print(f"ИТОГО {tag}: листов {tot['sheets']}, кривых {tot['curves']}, "
          f"★честных {tot['hon']}, утечек {tot['leak']}")

(ta, pa), (tb, pb) = res["A прод (правило)"], res["B обученная раскладка"]
up = sum(1 for k in pa if pb.get(k, 0) > pa[k]); dn = sum(1 for k in pa if pb.get(k, 0) < pa[k])
fa, fb = FP["A прод (правило)"], FP["B обученная раскладка"]
same = [k for k in fa if k in fb and fa[k] == fb[k]]
print(f"\n★ ФЛАГ СРАБОТАЛ? выдача различается на {len(fa) - len(same)} листах из {len(fa)}"
      + ("  ⛔ НИ НА ОДНОМ — модель не включилась, замер недействителен" if not fa or
         len(same) == len(fa) else ""))
print(f"\n{'='*78}\n★★ ОТГРУЖАЕМЫЙ ПУТЬ: {ta['hon']} → {tb['hon']} "
      f"({tb['hon']-ta['hon']:+d}), листов ↑{up}/↓{dn} из {len(pa)}\n{'='*78}")
for k in sorted(pa, key=lambda k: pb.get(k, 0) - pa[k]):
    if pb.get(k, 0) != pa[k]:
        print(f"  {k[:52]:<54} {pa[k]} → {pb.get(k, 0)}")
print("⚠ листы, где модель отказывается, в этот замер не входят: их файл не меняется по построению")

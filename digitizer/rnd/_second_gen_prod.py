r"""_second_gen_prod.py — ВТОРОЙ ГЕНЕРАТОР ВНУТРИ ПАЙПЛАЙНА, НА НАСТОЯЩИХ ЛИНИЯХ (§6.101, решающая проверка).

⚠⚠ ЗАЧЕМ ОТДЕЛЬНЫЙ СТЕНД, ЕСЛИ ЕСТЬ `_second_gen.py`. У того стенда размах линии восстановлен ИЗ ЕЁ ЖЕ
ПРОД-ТРАССЫ (в пуле нет `line.x_lo/x_hi`), то есть окно поиска центрировано на том, что прод уже нашёл.
Это подсказка, которой у генератора в проде нет, и она поднимает его результат: узкое окно там сильнее
широкого (pad=8 → 419, pad=150 → 402), хотя у честного генератора должно быть наоборот. Пока эта
подсказка не убрана, «+126 кривых» цитировать НЕЛЬЗЯ.

КАК УБРАНА. Перехватывается `trace2d.trace_auto` — точка, где у прода УЖЕ есть настоящие объекты
`Line` (со своими `x_lo/x_hi/y0/y1/x_center/color`, полученными детекцией, а не трассировкой) и есть
`rgb`. Оригинал вызывается как есть, а затем к его результату ДОБАВЛЯЮТСЯ трассы второго генератора,
построенные на тех же линиях и той же маске цвета. Дальше пайплайн не трогается вовсе: раскладка
`emit._map_lines_to_slots` получает пул побольше и выбирает сама — ровно то, что предлагал §6.101
(«не переписывать трассировщик, а добавить +1 источник трасс»).

★ ЧТО МЕРИТСЯ. Честные кривые в ВЫДАННОМ `<stem>_auto.nlgx` (не в пуле), два режима на одних листах:
  A — пайплайн как есть;
  B — пайплайн с добавленным генератором.
⚠ Путь трассировки пиннится флагом `--seq` (по умолчанию пусто = жадный `trace2d`): умолчание
`CVParams.seq_model` зависит от наличия torch, и наследовать его нельзя (§6.71, урок §6.102).
⚠ Контроль включения — отпечаток выдачи: если файлы совпали бит-в-бит, замер недействителен.

  <ComfyUI>\python_embeded\python.exe _second_gen_prod.py [--pad 150] [--cap 3] [--shard 0/12]
"""
import sys, io, argparse, contextlib, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from _multi_replica_probe import dense
from auto import imaging as im, meta as M, trace2d as T
from auto.pipeline import run as pipe_run
from auto.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div"])
ap.add_argument("--pad", type=int, default=150, help="окно генератора вокруг размаха ЛИНИИ")
ap.add_argument("--seq", default="", help="путь трассировки; пусто = жадный trace2d")
ap.add_argument("--cap", type=int, default=3)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--out", default=r"F:\nds\output\taskS\second_gen_prod")
a = ap.parse_args()
SH_I, SH_N = (int(v) for v in a.shard.split("/"))
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9
_orig_auto = T.trace_auto
ADD = {"on": False, "n": 0}


def trace_simple(fg, L, pad):
    """Ближайший по центру ран к своему предыдущему выбору. Без скорости, вершины и коаста."""
    H, W = fg.shape
    lo = max(0, int(L.x_lo) - pad); hi = min(W, int(L.x_hi) + pad + 1)
    if hi <= lo:
        return {}
    base = float(L.x_center); x = None; tr = {}
    for y in range(max(0, int(L.y0)), min(H, int(L.y1) + 1)):
        rr = im.row_runs(fg[y, lo:hi])
        if not rr:
            continue
        rr = [(p + lo, q + lo, c + lo) for p, q, c in rr]
        ref = base if x is None else x
        r = min(rr, key=lambda t: 0.0 if t[0] <= ref <= t[1]
                else min(abs(t[0] - ref), abs(t[1] - ref)))
        x = float(r[2]); tr[y] = x
    return tr


def trace_auto_plus(rgb, sheet, p=None):
    """Оригинал + трассы второго генератора на ТЕХ ЖЕ настоящих линиях.

    ⚠⚠ КАЖДОМУ КАНДИДАТУ — СВОЯ КОПИЯ `Line`, И БЕЗ ЭТОГО ЗАМЕР МЁРТВ. `emit._map_lines_to_slots`
    помечает использованной саму ЛИНИЮ (`used.add(id(L))`), а не трассу. Если добавить вторую трассу
    с тем же объектом линии, она не может быть выбрана НИКОГДА: прод-трасса идёт в списке первой,
    занимает слот и закрывает линию. Первый прогон именно так и вышел — 25 трасс добавлено, выдача
    бит-в-бит прежняя.
    ⚠ ЦЕНА ЭТОГО РЕШЕНИЯ, КОТОРУЮ НАДО ЦИТИРОВАТЬ: с копиями исключительность становится
    ПОКАНДИДАТНОЙ, а не полинейной — одна физическая линия теперь может занять ДВА слота. Для
    домена это неверно (линия нарисована одна), поэтому число надо читать как ПОТОЛОК добавления
    источника, а следующий шаг — вернуть исключительность по происхождению.
    """
    out = _orig_auto(rgb, sheet, p)
    if not ADD["on"]:
        return out
    import copy
    from auto.config import DEFAULT
    p = p or DEFAULT.cv
    fgc = {}
    extra = []
    for L in sheet.lines:
        if L.confidence != "AUTO" and not getattr(p, "trace_flagged", False):
            continue
        if L.color not in fgc:
            try:
                fgc[L.color] = T._color_fg(rgb, L.color, p)
            except Exception:
                continue
        tr = trace_simple(fgc[L.color], L, a.pad)
        if len(tr) >= 30:
            extra.append((copy.copy(L), tr))
    ADD["n"] += len(extra)
    return out + extra


T.trace_auto = trace_auto_plus


def err(o, g):
    c = [y for y in o if y in g]
    if len(c) < 30:
        return None, 0.0
    d = np.array([abs(o[y] - g[y]) for y in c], float)
    return float(np.median(d)), len(c) / max(1, len(g))


def leaked(tr, raw):
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


WELL, WLG = {}, {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WELL[q.name] = wlg.parent.name; WLG[q.name] = q
for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
    WELL.setdefault(q.name, "Semeguniv"); WLG.setdefault(q.name, q)
BY_STEM = {q.stem[:60]: q for q in WLG.values()}

seen, per_well, sheets = set(), {}, []
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        q = BY_STEM.get(f.stem)
        if f.stem in seen or q is None:
            continue
        seen.add(f.stem)
        wl = WELL.get(q.name, "?")
        if a.cap and len(per_well.get(wl, [])) >= a.cap:
            continue
        per_well.setdefault(wl, []).append(q.name); sheets.append(q)
Path(a.out).mkdir(parents=True, exist_ok=True)
print(f"взято {len(sheets)} листов со {len(per_well)} скважин, окно pad={a.pad}, seq={a.seq or '—'}")
if SH_N > 1:
    sheets = sorted(sheets, key=lambda p: p.name)
    sheets = sheets[len(sheets) * SH_I // SH_N: len(sheets) * (SH_I + 1) // SH_N]
    print(f"★ ШАРД {SH_I}/{SH_N}: {len(sheets)} листов × 2 режима")

res, FP = {}, {}
for tag, on in (("A прод", False), ("B +генератор", True)):
    ADD["on"] = on; ADD["n"] = 0
    tot = dict(hon=0, curves=0, sheets=0, leak=0)
    per, FP[tag] = {}, {}
    print(f"\n{'='*78}\n{tag}: второй генератор {'ВКЛ' if on else 'выкл'}\n{'='*78}")
    for n in sheets:
        img = find_image(n)
        if not img:
            continue
        cfg = Config(); cfg.cv.seq_model = a.seq
        cfg.out = Path(a.out) / tag.split()[0] / n.stem[:40]
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
        FP[tag][n.name] = tuple(sorted((k, len(v), round(float(np.median(list(v.values()))), 3))
                                       for k, v in W.items() if v))
        h = lk = 0
        for k, gd in gts.items():
            if k not in W or not W[k]:
                continue
            if leaked(W[k], raws[k]):
                lk += 1; continue
            h += HON(*err(W[k], gd))
        per[n.name] = h
        tot["hon"] += h; tot["curves"] += len(gts); tot["sheets"] += 1; tot["leak"] += lk
        print(f"  {n.stem[:42]:<44} кривых {len(gts):>2}  ЧЕСТНЫХ {h}")
    res[tag] = (tot, per)
    print(f"ИТОГО {tag}: листов {tot['sheets']}, кривых {tot['curves']}, ★честных {tot['hon']}, "
          f"утечек {tot['leak']}, добавлено трасс {ADD['n']}")
T.trace_auto = _orig_auto

pickle.dump({"res": res, "fp": FP, "pad": a.pad, "seq": a.seq},
            open(Path(a.out) / f"ab_{SH_I}of{SH_N}.pkl", "wb"))
(ta, pa), (tb, pb) = res["A прод"], res["B +генератор"]
up = sum(1 for k in pa if pb.get(k, 0) > pa[k]); dn = sum(1 for k in pa if pb.get(k, 0) < pa[k])
same = sum(1 for k in FP["A прод"] if FP["B +генератор"].get(k) == FP["A прод"][k])
print(f"\n{'='*78}\n★★ ОТГРУЖАЕМЫЙ ПУТЬ: {ta['hon']} → {tb['hon']} ({tb['hon']-ta['hon']:+d}), "
      f"листов ↑{up}/↓{dn} из {len(pa)}; выдача различается на "
      f"{len(FP['A прод'])-same}/{len(FP['A прод'])} листах"
      + ("  ⛔ НЕ ИЗМЕНИЛАСЬ — генератор не включился" if same == len(FP["A прод"]) else ""))

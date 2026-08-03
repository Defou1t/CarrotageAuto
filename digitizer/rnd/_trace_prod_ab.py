r"""_trace_prod_ab.py — A/B ПАРАМЕТРОВ ТРАССИРОВЩИКА НА ОТГРУЖАЕМОМ ПУТИ (§6.101 шаг 1, часть 2).

⚠⚠ ЗАЧЕМ ОТДЕЛЬНЫЙ СТЕНД. `_param_sweep.py` меряет ГОЛУЮ `trace_line` на восстановленной линии.
В файл же уходит кривая после `refine_trace` (эскалация полосы до трека, деспайк, границы Вороного),
раскладки по слотам и пост-обработки уровней. Durable-память ветки: **стенд и отгрузка расходятся
в разы, цитировать надо отгрузку** (§6.69, §6.75, §6.48 — правка, принятая по стенду, была откачена).

КАК. Пайплайн гоняется по разу на каждый режим на ОДНИХ И ТЕХ ЖЕ листах. Режим = набор умолчаний
`trace_line`, наложенный обёрткой:
  `kw.setdefault(...)` — то есть ЯВНО переданные вызывающим значения не трогаются. Это важно:
  `refine_trace` сам передаёт `x_range` при эскалации полосы и `wide_run=10**6` для `prefer_body`
  (свипующее перо MBK), и оба этих решения обязаны выжить.
Честность считается в выданном `<stem>_auto.nlgx`, а не в пуле; контроль утечки эксперта — как
в `_pool_oracle.py`.

★ ОТПЕЧАТОК ВЫДАЧИ обязателен: равные счётчики честности сами по себе НЕ доказывают, что режим
включился (§6.71). Если файлы совпали бит-в-бит — замер недействителен, и стенд это говорит вслух.

⚠⚠ ПУТЬ ТРАССИРОВКИ ЗАДАЁТСЯ РЕЖИМОМ (`seq=`), И ЭТО НЕ ФОРМАЛЬНОСТЬ. По умолчанию
`CVParams.seq_model = "seq_model_d45p.pt"`, и при доступном torch линии ведёт ОБУЧЕННЫЙ СЕЛЕКТОР
`trace_seq`, у которого правила вершины выноса НЕТ ВООБЩЕ (`nx = C[k]`, всегда центр). Первый заход
этого стенда наследовал умолчание и намерил «правка ничего не меняет» — файлы вышли бит-в-бит
одинаковыми на 19 листах, потому что патч `trace2d.trace_line` не вызывался ни разу.

⚠⚠ РАСКЛАДКА (`slot=`) ПИННИТСЯ ТАК ЖЕ, И ЭТО НЕ ИЗБЫТОЧНО. `_slot_prod_ab.py` (§6.90, число
42 → 50) задавал `cfg.cv.slot_model`, но `seq_model` НЕ пиннил — значит наследовал умолчание
`seq_model_d45p.pt` и под ComfyUI-питоном мерил СЕЛЕКТОРНЫЙ путь, нигде этого не объявив. Хуже
того, он пропускал листы, на которых модель отказывается, определяя отказ ОФЛАЙН ПО ПУЛАМ, — а пулы
собраны ЖАДНЫМ прогоном (§6.104, побитово 4/4 против 0/4). Условие из его же шапки («верно ровно
пока пулы собраны ТЕМ ЖЕ кодом трассировки») при этом нарушено. Здесь оба режима гоняются на ВСЕХ
листах выборки, без офлайн-отсева.

  <ComfyUI>\python_embeded\python.exe _trace_prod_ab.py \
      --mode "A жадный:seq=,slot=" --mode "B центр:seq=,slot=,wide_run=1000000" \
      --mode "C селектор:seq=seq_model_d45p.pt,slot=" \
      --mode "D сел+раскладка:seq=seq_model_d45p.pt,slot=slot_model_g250.npz" \
      --mode "E сел+медиана:seq=seq_model_d45p.pt,slot=,order=med_x" [--cap 3] [--shard 0/12]

⚠ Имена режимов БЕЗ ПРОБЕЛОВ, если стенд запускается через `Start-Process -ArgumentList`: массив
склеивается пробелом БЕЗ квотирования, и «A сел.база:…» доезжает до argparse двумя аргументами.
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
from auto import meta as M, trace2d as T
from auto.pipeline import run as pipe_run
from auto.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div"])
ap.add_argument("--mode", action="append", required=True,
                help='"имя:seq=<чекпойнт|пусто>,slot=<вес|пусто>,k=v,…"; k — параметры trace_line')
ap.add_argument("--slot-gate", default="frac0.8", help="мера уверенности обученной раскладки")
ap.add_argument("--cap", type=int, default=3, help="листов на скважину (0 = все)")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--out", default=r"F:\nds\output\taskS\prod_ab_trace")
a = ap.parse_args()
SH_I, SH_N = (int(v) for v in a.shard.split("/"))
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def parse_mode(s):
    """`имя:seq=<чекпойнт|пусто>,slot=<вес|пусто>,wide_run=…` → (имя, seq_model, slot_model, kwargs).
    ⚠⚠ `seq` И `slot` ОБЯЗАТЕЛЬНЫ И ПИННЯТСЯ ЯВНО. В `CVParams` по умолчанию стоит
    `seq_model_d45p.pt`, и при доступном torch `trace_auto` ведёт линии ОБУЧЕННЫМ СЕЛЕКТОРОМ
    `trace_seq`, а не `trace2d.trace_line`, — тогда любая правка констант `trace_line` до кода
    НЕ ДОХОДИТ (проверено: выдача выходит бит-в-бит одинаковой на 19 листах). Наследовать режим из
    умолчаний нельзя: он зависит от интерпретатора (torch есть только в embedded-python ComfyUI),
    то есть один и тот же стенд давал бы РАЗНЫЕ пути на разных машинах — ровно запрет §6.71.
    ⚠ `slot` от пути трассировки НЕ зависит (скор считает numpy) и действует на обоих."""
    nm, _, tail = s.partition(":")
    kw, seq, slot, order = {}, None, None, "x_center"
    for part in filter(None, tail.split(",")):
        k, _, v = part.partition("=")
        if k == "seq":
            seq = v
        elif k == "slot":
            slot = v
        elif k == "order":
            order = v or "x_center"
        else:
            kw[k] = None if v in ("None", "") else (float(v) if k == "slmax" else int(v))
    if seq is None:
        sys.exit(f"режим {nm!r}: не задан seq= (пусто = жадный путь, иначе имя чекпойнта)")
    if slot is None:
        sys.exit(f"режим {nm!r}: не задан slot= (пусто = раскладка ПРАВИЛОМ, иначе имя веса)")
    # ⚠ `order=` НЕ обязателен, в отличие от `seq`/`slot`, и это осознанно. Пиннить требуется то,
    # что молча разъезжается МЕЖДУ МАШИНАМИ: `seq` зависит от наличия torch, `slot` — от наличия
    # веса. `slot_order` же берётся из конфига одинаково везде, поэтому умолчание тут честное.
    if order not in ("x_center", "med_x", "rough_n"):
        sys.exit(f"режим {nm!r}: order={order!r} — ждали x_center | med_x | rough_n")
    if seq and kw:
        print(f"⚠ режим {nm!r}: при включённом селекторе параметры {list(kw)} НЕ действуют — "
              f"`trace_seq` строит свой трассировщик и правила вершины у него нет вовсе")
    return nm, seq, slot, order, kw


MODES = [parse_mode(s) for s in a.mode]
_orig_trace = T.trace_line


def make(kw):
    def wrapped(fg, line, frame, p, **rest):
        for k, v in kw.items():
            rest.setdefault(k, v)          # ⚠ setdefault: явные x_range/wide_run refine'а живут
        return _orig_trace(fg, line, frame, p, **rest)
    return wrapped


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


WELL, WLG = {}, {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WELL[q.name] = wlg.parent.name; WLG[q.name] = q
for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
    WELL.setdefault(q.name, "Semeguniv"); WLG.setdefault(q.name, q)

# ── выборка: до --cap листов на СКВАЖИНУ, чтобы плотные скважины не решали за всех ────────────
# ⚠ Список строится ПО ИМЕНАМ ФАЙЛОВ. Унаследованный цикл `pickle.load` по всем пулам читал 2.35 ГБ
# ради одного поля `name` — на шардах это десятки гигабайт до начала работы (та же ловушка, что
# в `_param_sweep.py`). `_pool_oracle.py` кладёт дамп как `{nlgx.stem[:60]}.pkl`.
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
print(f"листов в пулах {len(seen)}, взято {len(sheets)} со {len(per_well)} скважин "
      f"(cap {a.cap or '—'}), режимов {len(MODES)}")
Path(a.out).mkdir(parents=True, exist_ok=True)
if SH_N > 1:
    sheets = sorted(sheets, key=lambda p: p.name)
    lo = len(sheets) * SH_I // SH_N; hi = len(sheets) * (SH_I + 1) // SH_N
    sheets = sheets[lo:hi]          # блоком, а не через шаг: сканы скважины лежат рядом
    print(f"★ ШАРД {SH_I}/{SH_N}: {len(sheets)} листов × {len(MODES)} режима")

res, FP = {}, {}
for nm_mode, seq, slot, order, kw in MODES:
    T.trace_line = make(kw) if kw else _orig_trace
    tot = dict(hon=0, curves=0, sheets=0, leak=0)
    per, FP[nm_mode] = {}, {}
    print(f"\n{'='*78}\n{nm_mode}: seq={seq or '— (жадный trace2d)'}, "
          f"slot={slot or '— (раскладка правилом)'}, order={order}, "
          f"{kw or 'константы trace_line по умолчанию'}\n{'='*78}")
    for n in sheets:
        img = find_image(n)
        if not img:
            continue
        cfg = Config()
        cfg.cv.seq_model = seq            # §6.71: путь трассировки задаёт стенд, а не умолчания
        cfg.cv.slot_model = slot          # §6.71: и раскладку тоже — см. шапку про `_slot_prod_ab`
        cfg.cv.slot_gate = a.slot_gate
        cfg.cv.slot_order = order         # §6.105: ключ порядка в раскладке ПРАВИЛОМ
        cfg.out = Path(a.out) / nm_mode.split()[0] / n.stem[:40]
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
        FP[nm_mode][n.name] = tuple(sorted(
            (k, len(v), round(float(np.median(list(v.values()))), 3)) for k, v in W.items() if v))
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
    res[nm_mode] = (tot, per)
    print(f"ИТОГО {nm_mode}: листов {tot['sheets']}, кривых {tot['curves']}, "
          f"★честных {tot['hon']}, утечек {tot['leak']}")
T.trace_line = _orig_trace

pickle.dump({"res": res, "fp": FP, "modes": [tuple(m) for m in MODES]},
            open(Path(a.out) / f"ab_{SH_I}of{SH_N}.pkl", "wb"))
base_nm = MODES[0][0]
(ta, pa) = res[base_nm]
print(f"\n{'='*78}\n★★ ОТГРУЖАЕМЫЙ ПУТЬ (база = {base_nm}: {ta['hon']} честных / "
      f"{ta['curves']} кривых / {ta['sheets']} листов)\n{'='*78}")
for nm_mode, _seq, _slot, _order, _kw in MODES[1:]:
    tb, pb = res[nm_mode]
    up = sum(1 for k in pa if pb.get(k, 0) > pa[k]); dn = sum(1 for k in pa if pb.get(k, 0) < pa[k])
    same = sum(1 for k in FP[base_nm] if FP[nm_mode].get(k) == FP[base_nm][k])
    flag = "" if same < len(FP[base_nm]) else "  ⛔ ВЫДАЧА НЕ ИЗМЕНИЛАСЬ — режим не включился"
    print(f"{nm_mode:<24} {ta['hon']} → {tb['hon']} ({tb['hon']-ta['hon']:+d}), "
          f"листов ↑{up}/↓{dn} из {len(pa)}; выдача различается на "
          f"{len(FP[base_nm])-same}/{len(FP[base_nm])} листах{flag}")

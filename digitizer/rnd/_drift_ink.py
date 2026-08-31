r"""_drift_ink.py — ЧТО ЛЕЖИТ ПОД УВЕДЁННОЙ ТРАССОЙ В ПОЛОСЕ 3-10px: ЧУЖАЯ ТУШЬ ИЛИ БУМАГА (§6.148 → §6.149)

ОТКУДА ВОПРОС. §6.148 вскрыл самую большую корзину ветки — 1396 кривых, которых не берёт ни один
путь, — и показала она вот что: покрытие там ни при чём (медиана cov = 1.00), а **31.3% корзины
(436 кривых, 15.8% всей выборки) промахивается на 3-10px**. Это крупнейший ОДНОРОДНЫЙ кусок
недостижимого, и против него закрыт ровно ОДИН класс механизмов: §6.119 доказал, что постфильтры
не помогают (сглаживание даёт −283 честных, сигнал и ошибка в одной полосе частот, промах —
МЕДЛЕННЫЙ УВОД, а не дрожание). Там же прямо сказано: «работа с изображением (увод — это сход с
туши) не проверялась и остаётся открытой». Это она.

ЧТО МЕРЯЕТ СТЕНД. По ГОТОВОЙ выдаче (пайплайн не запускается) для каждой кривой полосы 3-10px
смотрит, ЧТО ПОД ТРАССОЙ на каждой пробной строке:
    своя тушь   — под трассой чернила, и ближайшая экспертная кривая — ЭТА ЖЕ;
    чужая тушь  — под трассой чернила, но ближайшая экспертная кривая ДРУГАЯ (сошла на соседа);
    бумага      — под трассой чернил нет вовсе (трасса идёт по пустому месту).
⇒ Это разделяет две совершенно разные болезни, которые в полосе 3-10px выглядят одинаково:
«трасса держится своей линии, но систематически смещена» лечится калибровкой под тушь, а «трасса
сползла на соседа» — только идентичностью, то есть тем же рычагом, что §6.144.

⚠⚠ ЭТО ДИАГНОЗ, А НЕ ПОТОЛОК. Стенд не обещает ни одной кривой: он говорит, какой механизм вообще
может здесь сработать. §6.113 durable: оценка резерва обязана проверяться механизмом, идущим по
строкам, — но чтобы его писать, надо сперва знать, за что цепляться.

  python _drift_ink.py --list                       # отобрать полосу (без картинок, быстро)
  <ComfyUI>\python_embeded\python.exe _drift_ink.py --shard 0/4 --out F:/nds/output/taskS/driftink
  python _drift_ink.py --sum --out F:/nds/output/taskS/driftink
"""
import sys, argparse, pickle, hashlib, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:/nds/output/taskS/ab_rowdec_pair")
ap.add_argument("--modes", nargs="+", default=["A", "B"])
ap.add_argument("--out", default=r"F:/nds/output/taskS/driftink")
ap.add_argument("--band", nargs=2, type=float, default=[3.0, 10.0])
ap.add_argument("--shard", default="0/1")
ap.add_argument("--step", type=int, default=16, help="каждая N-я строка трассы")
ap.add_argument("--halfw", type=int, default=2, help="полуокно по x, px")
ap.add_argument("--near", type=float, default=6.0, help="ближе этого к экспертной = «её тушь»")
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--list", action="store_true")
ap.add_argument("--sum", action="store_true")
# ★★★ §6.149 → §6.151: ЕДИНСТВЕННАЯ ЖИВАЯ ГИПОТЕЗА ПО ПОЛОСЕ 3-10px. Если трасса лежит на
# СВОЕЙ туши и не смещена, а промах 3-10px, то вопрос не «где линия», а «какую точку ШИРОКОГО
# штриха брать за значение». Проверяется прямо: ширина рана под трассой у кривых полосы
# против ширины под ЧЕСТНЫМИ кривыми ТЕХ ЖЕ ЛИСТОВ. Контроль обязателен — без него «медиана
# 7px» не значит ничего: может, на этих бланках вся тушь такая.
ap.add_argument("--width", action="store_true", help="мерить ШИРИНУ рана под трассой + контроль")
# ★★★ §6.151 → СЛЕДУЮЩИЙ ЗАМЕР. §6.151 доказал: штрих 12px, эксперт в его ЦЕНТРЕ, трасса у КРАЯ.
# Прод при этом НОМИНАЛЬНО берёт центр рана (`trace2d.py:88`, `nx = c`, правило вершины выключено
# при ширине 12 < `trace_wide_run` = 50). ⇒ Вопрос сузился до одного: между выбором рана и
# записанным значением точка уезжает — ГДЕ? Замер сравнивает записанное значение с центром рана
# ТРЕМЯ способами (см. `run_variants`), и именно РАЗНИЦА между способами показывает виновника.
ap.add_argument("--center", action="store_true",
                help="сравнить ЗАПИСАННОЕ значение с ЦЕНТРОМ рана, в котором оно лежит (§6.151)")
a = ap.parse_args()
SH_I, SH_N = (int(z) for z in a.shard.split("/"))
LIST = Path(a.out) / "band.tsv"
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


# ═══ СВОДКА ШИРИНЫ РАНА ═══════════════════════════════════════════════════════════════════════
if a.sum and a.width:
    parts = sorted(Path(a.out).glob("width_*of*.pkl"))
    want, rows, have = None, [], set()
    for f in parts:
        d = pickle.load(open(f, "rb"))
        want = want or d["n_shards"]; have.add(d["shard"]); rows += d["rows"]
    print(f"шардов {len(have)} из {want}"
          f"{'' if want and len(have) == want else '   ⚠⚠ ПРОГОН НЕПОЛНЫЙ'}")
    band_w = [r["w"] for r in rows if r["kind"] == "полоса" and r["w"] is not None]
    ctrl_w = [r["w"] for r in rows if r["kind"] == "честная" and r["w"] is not None]
    print(f"кривых: полоса 3-10px {len(band_w)}, контроль (ЧЕСТНЫЕ те же листы) {len(ctrl_w)}")
    if not band_w or not ctrl_w:
        sys.exit("нет одной из групп — сравнивать не с чем")
    b = np.array(band_w, float); c = np.array(ctrl_w, float)
    print("\n★★ ШИРИНА РАНА ПОД ТРАССОЙ (медиана по кривой, px)")
    print(f"   полоса 3-10px : медиана {np.median(b):5.1f}, p25 {np.percentile(b,25):.1f}, "
          f"p75 {np.percentile(b,75):.1f}, доля >6px {100*(b>6).mean():.0f}%")
    print(f"   ★ КОНТРОЛЬ    : медиана {np.median(c):5.1f}, p25 {np.percentile(c,25):.1f}, "
          f"p75 {np.percentile(c,75):.1f}, доля >6px {100*(c>6).mean():.0f}%")
    # ⚠ Перестановочный тест: без него «7 против 5» — это разговор, а не замер (§6.118).
    rng = np.random.default_rng(20260827)
    obs = float(np.median(b) - np.median(c))
    pool = np.concatenate([b, c]); nb = len(b)
    null = np.array([float(np.median(z[:nb]) - np.median(z[nb:]))
                     for z in (rng.permutation(pool) for _ in range(5000))])
    pp = float((np.abs(null) >= abs(obs)).mean())
    print(f"   разница медиан {obs:+.1f}px; перестановочный p = {pp:.4f} (5000 перестановок)")
    # ★★ ПРИРОДА ШИРОКОГО РАНА: одна кривая или слипшиеся штрихи
    for kind in ("полоса", "честная"):
        nn = [r.get("n_in") for r in rows if r["kind"] == kind and r.get("n_in") is not None]
        gp = [r.get("gt_pos") for r in rows if r["kind"] == kind and r.get("gt_pos") is not None]
        if nn:
            nn = np.array(nn, float)
            print(f"\n   {kind}: экспертных кривых ВНУТРИ рана — медиана {np.median(nn):.1f}, "
                  f"доля с ≥2 {100*(nn >= 2).mean():.0f}%")
        if gp:
            gp = np.array(gp, float)
            print(f"   {kind}: эталон внутри рана — медиана {np.median(gp):.2f} "
                  f"(0 = левый край, 0.5 = центр, 1 = правый); |откл. от центра| "
                  f"{np.median(np.abs(gp - 0.5)):.2f}")
    print(f"   ⇒ {'★★ ТУШЬ ПОД ПОЛОСОЙ ШИРЕ — гипотеза §6.149 подтверждена' if obs > 0 and pp < 0.05 else '⛔ ШИРИНА НЕ РАЗЛИЧАЕТ — гипотеза §6.149 ОТПАЛА, полосу лечит не выбор точки в штрихе'}")
    sys.exit()


if a.sum and a.center:
    parts = sorted(Path(a.out).glob("center_*of*.pkl"))
    want, rows, have = None, [], set()
    for f in parts:
        d = pickle.load(open(f, "rb"))
        want = want or d["n_shards"]; have.add(d["shard"]); rows += d["rows"]
    print(f"шардов {len(have)} из {want}"
          f"{'' if want and len(have) == want else '   ⚠⚠ ПРОГОН НЕПОЛНЫЙ'}")
    B = [r for r in rows if r["kind"] == "полоса"]
    C = [r for r in rows if r["kind"] == "честная"]
    print(f"кривых: полоса 3-10px {len(B)}, контроль (ЧЕСТНЫЕ те же листы) {len(C)}")
    if not B or not C:
        sys.exit("нет одной из групп — сравнивать не с чем")
    take = lambda rs, k: np.array([r[k] for r in rs if r.get(k) is not None], float)

    print("\n★★★ ГДЕ ЛЕЖИТ ЗАПИСАННОЕ ЗНАЧЕНИЕ ОТНОСИТЕЛЬНО ЦЕНТРА РАНА (px, медиана по кривой)")
    print(f"   {'величина':<46}{'полоса':>10}{'контроль':>10}")
    for k, name in (("ad_prod", "|значение − центр рана ПРОДА| (row_runs)"),
                    ("ad_col", "|значение − центр рана своего цвета|"),
                    ("ad_any", "|значение − центр штриха (любая тушь)|"),
                    ("ad_gt", "★ КОНТРОЛЬ: |эталон − центр штриха|"),
                    ("ashift", "★ |центр рана ПРОДА − центр штриха|")):
        b, c = take(B, k), take(C, k)
        if len(b) and len(c):
            print(f"   {name:<46}{np.median(b):>10.2f}{np.median(c):>10.2f}")
    print("\n   ЗНАКОВЫЕ (сдвиг или разброс — §6.119/§6.149 показали, что это разные болезни):")
    for k, name in (("d_prod", "значение − центр рана ПРОДА"),
                    ("d_any", "значение − центр штриха"),
                    ("shift", "центр рана ПРОДА − центр штриха")):
        b, c = take(B, k), take(C, k)
        if len(b) and len(c):
            print(f"   {name:<46}{np.median(b):>+10.2f}{np.median(c):>+10.2f}")
    print("\n   ШИРИНА РАНА, медиана (сверка с §6.151: полоса 12.0, контроль 8.0):")
    for k, name in (("w_any", "любая тушь — штрих целиком"), ("w_col", "свой цвет")):
        b, c = take(B, k), take(C, k)
        if len(b) and len(c):
            print(f"   {name:<46}{np.median(b):>10.1f}{np.median(c):>10.1f}")

    # ★★★ СВЕРКА С §6.151 И ПРЯМАЯ ПРОВЕРКА ЕГО ФОРМУЛИРОВКИ. Читать ПЕРВЫМ: если `gt_pos` не
    # воспроизводит 0.53/0.57, стенд восстановил ран не так, как прошлый замер, и всё остальное
    # в этой сводке недействительно (§6 правило 2 — сверка объёма и контрольных чисел).
    print("\n★★ ПОЛОЖЕНИЕ В РАНЕ (0 = левый край, 0.5 = ЦЕНТР, 1 = правый):")
    for k, name, ref in (("gt_pos", "★ СВЕРКА эталон в ране (§6.151: 0.53 / 0.57)", True),
                         ("tr_pos", "★ ТРАССА в ране — «у края» или в центре?", False)):
        b, c = take(B, k), take(C, k)
        if len(b) and len(c):
            print(f"   {name:<46}{np.median(b):>10.2f}{np.median(c):>10.2f}")
            print(f"   {'  |отклонение от центра|':<46}"
                  f"{np.median(np.abs(b - 0.5)):>10.2f}{np.median(np.abs(c - 0.5)):>10.2f}")
    gp = take(B, "gt_pos")
    if len(gp):
        ok = abs(float(np.median(gp)) - 0.53) <= 0.05
        print(f"   ⇒ сверка с §6.151: {'★ СОШЛАСЬ' if ok else '⛔ НЕ СОШЛАСЬ — числа ниже НЕДЕЙСТВИТЕЛЬНЫ'}")

    # ★★★ ГДЕ ЖИВЁТ ПРОМАХ. Трасса в центре своего рана, эксперт в центре своего; если ран ОДИН,
    # промаха быть не может. ⇒ вся полоса обязана сидеть на строках, где эксперт ВНЕ рана трассы.
    print("\n★★★ РАЗЛОЖЕНИЕ ПРОМАХА ПО СТРОКАМ (эксперт внутри рана под трассой или вне его):")
    print(f"   {'величина':<46}{'полоса':>10}{'контроль':>10}")
    for k, name in (("gt_in", "доля строк, где эксперт ВНУТРИ рана трассы"),
                    ("e_in", "★ |трасса − эталон| на этих строках, px"),
                    ("e_out", "★ |трасса − эталон| на ОСТАЛЬНЫХ строках, px")):
        b, c = take(B, k), take(C, k)
        if len(b) and len(c):
            print(f"   {name:<46}{np.median(b):>10.2f}{np.median(c):>10.2f}")
    ein, eout = take(B, "e_in"), take(B, "e_out")
    if len(ein) and len(eout):
        print(f"   ⇒ на строках ОДНОГО рана промах {np.median(ein):.2f}px, на разных "
              f"{np.median(eout):.2f}px — разница {np.median(eout) / max(0.01, np.median(ein)):.0f}×")
        print("   ⇒ полоса 3-10px — это НЕ «какую точку штриха взять», а «на КАКОМ ране стоять»,")
        print("     то есть ИДЕНТИЧНОСТЬ (тот же рычаг, что §6.144/§6.148), а не выбор точки.")

    # ⚠ Без перестановочного теста «0.9 против 0.4» — разговор, а не замер (§6 правило 3).
    rng = np.random.default_rng(20260828)
    b, c = take(B, "ad_prod"), take(C, "ad_prod")
    obs = float(np.median(b) - np.median(c))
    pool = np.concatenate([b, c]); nb = len(b)
    null = np.array([float(np.median(z[:nb]) - np.median(z[nb:]))
                     for z in (rng.permutation(pool) for _ in range(5000))])
    pp = float((np.abs(null) >= abs(obs)).mean())
    print(f"\n   |значение − центр рана прода|: разница медиан {obs:+.2f}px, "
          f"перестановочный p = {pp:.4f} (5000 перестановок)")

    # ★★★ ПРАВИЛО ЧТЕНИЯ ЗАФИКСИРОВАНО ДО ПРОГОНА (§6 правило 1: менять его, увидев число, —
    # ровно та ошибка, на которой ветка горела четырежды). Пороги: «мало» < 1.5px (меньше
    # полуширины пера), «велико» ≥ 2.0px (нижний край полосы 3-10px минус шум эталона 1.0px).
    ap_, as_ = float(np.median(take(B, "ad_prod"))), float(np.median(take(B, "ashift")))
    print("\n★★★ ЧТЕНИЕ (правило зафиксировано ДО прогона):")
    if ap_ < 1.5 and as_ >= 2.0:
        print(f"   |значение − центр рана прода| = {ap_:.2f}px ⇒ прод БЕРЁТ центр того рана,")
        print(f"   что видит; но сам этот ран смещён от штриха на {as_:.2f}px")
        print("   ⇒ ⛔ ВИНОВАТ БИНАРЬ (`_color_fg` / `dark_v`), а не то, что после выбора рана.")
    elif ap_ >= 2.0:
        print(f"   |значение − центр рана прода| = {ap_:.2f}px ⇒ ⛔ ТОЧКА УЕЗЖАЕТ ПОСЛЕ ВЫБОРА")
        print("   РАНА (`refine` либо селектор `trace_seq`) — сам выбор рана НЕ виноват.")
    else:
        print(f"   |значение − центр рана прода| = {ap_:.2f}px при смещении рана {as_:.2f}px —")
        print("   ⚠ НИ ОДНА из двух версий не выделена: полоса 3-10px этим замером НЕ объясняется,")
        print("   и разница с контролем важнее абсолютных величин (см. таблицу выше).")
    sys.exit()


# ═══ СВОДКА ═══════════════════════════════════════════════════════════════════════════════════
if a.sum:
    parts = sorted(Path(a.out).glob("drift_*of*.pkl"))
    want, rows, have = None, [], set()
    for f in parts:
        d = pickle.load(open(f, "rb"))
        want = want or d["n_shards"]; have.add(d["shard"]); rows += d["rows"]
    print(f"шардов {len(have)} из {want}"
          f"{'' if want and len(have) == want else '   ⚠⚠ ПРОГОН НЕПОЛНЫЙ'}")
    print(f"кривых полосы разобрано: {len(rows)}")
    if not rows:
        sys.exit("нет данных")
    agg = Counter()
    frac_own, frac_other, frac_paper = [], [], []
    for r in rows:
        n = max(1, r["own"] + r["other"] + r["paper"])
        frac_own.append(r["own"] / n); frac_other.append(r["other"] / n)
        frac_paper.append(r["paper"] / n)
        top = max(("своя тушь", r["own"]), ("чужая тушь", r["other"]), ("бумага", r["paper"]),
                  key=lambda z: z[1])[0]
        agg[top] += 1
    print(f"\n★★ ЧТО ПОД ТРАССОЙ (по преобладанию строк):")
    for k, v in agg.most_common():
        print(f"   {k:<12} {v:>5}  ({100*v/len(rows):4.1f}%)")
    print(f"\n   доля строк, медиана по кривым: своя {np.median(frac_own):.2f}, "
          f"чужая {np.median(frac_other):.2f}, бумага {np.median(frac_paper):.2f}")
    # ★ Насколько «своя» трасса смещена системно: медиана ЗНАКОВОЙ разности
    sh = np.array([r["signed"] for r in rows if r["signed"] is not None], float)
    if len(sh):
        own = np.array([r["own"] / max(1, r["own"] + r["other"] + r["paper"]) for r in rows
                        if r["signed"] is not None])
        m = own >= 0.5
        print(f"\n★★ ЗНАКОВОЕ СМЕЩЕНИЕ (трасса минус эталон, px):")
        print(f"   на кривых СО СВОЕЙ тушью ({int(m.sum())}): медиана {np.median(sh[m]):+.1f}, "
              f"|медиана| {abs(np.median(sh[m])):.1f} против медианы модуля "
              f"{np.median(np.abs(sh[m])):.1f}")
        print(f"   ⇒ {'★ ЭТО СИСТЕМНЫЙ СДВИГ — калибруется' if abs(np.median(sh[m])) > 0.5*np.median(np.abs(sh[m])) else '⚠ это РАЗБРОС, а не сдвиг — калибровать нечего (как §6.119)'}")
    sys.exit()

# ═══ ОТБОР ПОЛОСЫ (без картинок) ══════════════════════════════════════════════════════════════
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC.setdefault(q.name, q)
TRACK, seen = {}, set()
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        if f.stem in seen:
            continue
        seen.add(f.stem)
        d = pickle.load(open(f, "rb"))
        TRACK[d["name"]] = {s["name"]: s["track"] for s in d["slots"]}
BYDIR = {f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}": nm
         for nm, q in SRC.items()}


def curves(p):
    return {c["name"]: dense(c) for c in extract(str(p))["curves"] if M.mnem_root(c["name"]) != "DA"}


if a.list or not LIST.is_file():
    Path(a.out).mkdir(parents=True, exist_ok=True)
    lo, hi = a.band
    out, seen_sheets = [], 0
    base = Path(a.dir) / a.modes[0]
    for d in sorted((x for x in base.iterdir() if x.is_dir()), key=lambda p: p.name):
        nm = BYDIR.get(d.name)
        if nm is None:
            continue
        wr = {}
        for mode in a.modes:
            f = next(iter(sorted((Path(a.dir) / mode / d.name).glob("*_auto.nlgx"))), None)
            if f:
                wr[mode] = curves(f)
        if len(wr) < len(a.modes):
            continue
        gts = {c["name"]: dense(c) for c in extract(str(SRC[nm]))["curves"]
               if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        if not gts:
            continue
        tmap = TRACK.get(nm, {}); seen_sheets += 1
        for g, gt in gts.items():
            ti = tmap.get(g)
            best = None
            for mode, w in wr.items():
                for k in (w if ti is None else [k for k in w if tmap.get(k) == ti]):
                    m, c = err(w[k], gt)
                    if m is None:
                        continue
                    if best is None or m < best[0]:
                        best = (m, c, mode, k)
            if best is None or HON(best[0], best[1]):
                continue
            if lo < best[0] <= hi:
                out.append((nm, g, best[2], best[3], f"{best[0]:.2f}", f"{best[1]:.3f}"))
    LIST.write_text("\n".join("\t".join(map(str, r)) for r in out) + "\n", encoding="utf-8")
    print(f"★ ПОЛОСА {lo}-{hi}px: {len(out)} кривых на {len({r[0] for r in out})} листах "
          f"(разобрано {seen_sheets} листов) → {LIST}")
    if a.list:
        sys.exit()

# ═══ ЗАМЕР: ЧТО ПОД ТРАССОЙ ═══════════════════════════════════════════════════════════════════
from dataset_build import find_image
from auto import imaging as im
from auto.config import Config
from auto import rowdec

P = Config().cv
band = [ln.split("\t") for ln in LIST.read_text(encoding="utf-8").splitlines() if ln.strip()]
sheets = sorted({r[0] for r in band})
if SH_N > 1:
    sheets = sheets[len(sheets) * SH_I // SH_N: len(sheets) * (SH_I + 1) // SH_N]
    print(f"★ ШАРД {SH_I}/{SH_N}: {len(sheets)} листов")
by_sheet = {}
for r in band:
    by_sheet.setdefault(r[0], []).append(r)

def run_span(x0i, cmap_row):
    """→ (левый, правый) края непрерывного рана туши под колонкой x0i; None если туши нет."""
    W = len(cmap_row)
    if not (0 <= x0i < W) or not cmap_row[x0i]:
        return None
    L = x0i
    while L > 0 and cmap_row[L - 1]:
        L -= 1
    R = x0i
    while R + 1 < W and cmap_row[R + 1]:
        R += 1
    return L, R


def run_variants(x0i, cmap_row, gap=4):
    """Три РАЗНЫХ рана под одной колонкой — и разница МЕЖДУ ними и есть предмет замера (§6.151).

    ⚠⚠ ЗАЧЕМ ТРИ, А НЕ ОДИН. §6.149/§6.151 мерили ран ЛЮБОЙ туши подряд (`run_span`), а прод
    ведёт по маске ОДНОГО цвета (`trace2d._color_fg`), склеивает раны через зазор
    (`imaging.row_runs`, gap=4) и центром берёт СРЕДНЕЕ ПИКСЕЛЕЙ (`float(seg.mean())`), а не
    середину (x0+x1)/2. Это ТРИ независимых отличия, каждое из которых способно увести точку от
    центра штриха, и порознь они не видны — видна только их сумма, те самые 3-10px.

    → dict или None:
        any_c   середина рана ЛЮБОЙ туши          — «центр штриха», взгляд §6.151
        col_c   середина рана СВОЕГО цвета        — что от штриха оставляет прод-бинарь
        prod_c  СРЕДНЕЕ пикселей своего цвета в ране со склейкой — буквально `c` прода
        w_any / w_col  ширины соответствующих ранов
    ⚠ Цвет берётся ИЗ КАРТЫ под самой точкой, а не из выдачи: в `*_auto.nlgx` цвета линии нет.
    `_colmap` строится из ТЕХ ЖЕ масок, что `_color_fg` (`rowdec.py:163`), поэтому «свой цвет»
    здесь — та же величина, что у прода, с точностью до разрешения перекрытий.
    ★ `prod_c` считается ВЫЗОВОМ `im.row_runs` — той же функции, что зовёт `trace_line`, а не
    её пересказом: пересказ разошёлся бы с продом ровно там, где замер и должен быть точен."""
    W = len(cmap_row)
    if not (0 <= x0i < W):
        return None
    col = int(cmap_row[x0i])
    if col == 0:
        return None
    sp = run_span(x0i, cmap_row)
    if sp is None:
        return None
    aL, aR = sp
    L = x0i
    while L > 0 and cmap_row[L - 1] == col:
        L -= 1
    R = x0i
    while R + 1 < W and cmap_row[R + 1] == col:
        R += 1
    prod_c = None
    for x0, x1, c in im.row_runs(cmap_row == col, gap=gap):
        if x0 <= x0i <= x1:
            prod_c = c
            break
    if prod_c is None:
        return None
    return dict(any_c=(aL + aR) / 2.0, col_c=(L + R) / 2.0, prod_c=prod_c,
                w_any=aR - aL + 1, w_col=R - L + 1, any_L=aL, any_R=aR)


if a.width:
    # ★★ ЗАМЕР ШИРИНЫ + КОНТРОЛЬ. На КАЖДОМ листе полосы меряются две группы: кривые полосы и
    # ЧЕСТНЫЕ кривые того же листа. ⚠ Контроль на ТЕХ ЖЕ листах обязателен: бланки различаются
    # пером и разрешением скана, и «медиана 7px» без контроля не значит ничего.
    from dataset_build import find_image
    from auto import imaging as im
    from auto.config import Config
    from auto import rowdec
    P = Config().cv
    band = [ln.split("	") for ln in LIST.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sheets = sorted({r[0] for r in band})
    if SH_N > 1:
        sheets = sheets[len(sheets) * SH_I // SH_N: len(sheets) * (SH_I + 1) // SH_N]
        print(f"★ ШАРД {SH_I}/{SH_N}: {len(sheets)} листов")
    by_sheet = {}
    for r in band:
        by_sheet.setdefault(r[0], []).append(r)
    rows, done, skip = [], 0, Counter()
    for nm in sheets:
        q = SRC.get(nm)
        img = find_image(q) if q else None
        if not img:
            skip["нет картинки"] += 1; continue
        stem = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
        gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
               if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        tmap = TRACK.get(nm, {})
        wr = {}
        for mode in a.modes:
            f = next(iter(sorted((Path(a.dir) / mode / stem).glob("*_auto.nlgx"))), None)
            if f:
                wr[mode] = curves(f)
        if not wr or not gts:
            skip["нет выдачи"] += 1; continue
        try:
            rgb = im.load_rgb(str(img))
        except Exception:
            skip["скан не читается"] += 1; continue
        cmap = rowdec._colmap(rgb, P)
        H, W = cmap.shape
        want = {(r[1], r[2], r[3]) for r in by_sheet[nm]}      # (эталон, режим, имя выдачи)

        def widths(tr):
            """→ (ширина рана, СКОЛЬКО экспертных кривых внутри рана, положение эталона в ране).

            ⚠⚠ ТРИ ВЕЛИЧИНЫ, А НЕ ОДНА, И ЭТО ГЛАВНОЕ В ЗАМЕРЕ. Широкий ран бывает двух совершенно
            разных природ: ТОЛСТОЕ ПЕРО (одна кривая, 12px штрих) и СЛИПШИЕСЯ ШТРИХИ (две кривые
            коснулись). Лечение у них противоположное: у первого — выбрать точку в штрихе, у
            второго — разделить кривые, то есть опять идентичность. Ширина одна их не различает.
            Положение эталона внутри рана (0 = левый край, 1 = правый) говорит третье: если
            эксперт сидит по центру, а трасса нет, вопрос решается правилом выбора точки."""
            ws, ns, pos = [], [], []
            for y in sorted(tr)[::a.step]:
                y = int(y)
                if not (0 <= y < H):
                    continue
                sp = run_span(int(round(tr[y])), cmap[y])
                if sp is None:
                    continue
                L, R = sp
                ws.append(R - L + 1)
                ns.append(sum(1 for gv in gts.values() if y in gv and L <= gv[y] <= R))
            return ((float(np.median(ws)) if len(ws) >= 10 else None),
                    (float(np.median(ns)) if len(ns) >= 10 else None))

        def where_gt(tr, gt):
            """→ медиана относительного положения ЭТАЛОНА внутри рана под трассой (0..1)."""
            pos = []
            for y in sorted(tr)[::a.step]:
                y = int(y)
                if not (0 <= y < H) or y not in gt:
                    continue
                sp = run_span(int(round(tr[y])), cmap[y])
                if sp is None:
                    continue
                L, R = sp
                if R > L and L <= gt[y] <= R:
                    pos.append((gt[y] - L) / (R - L))
            return float(np.median(pos)) if len(pos) >= 10 else None

        for (g, mode, k) in want:
            tr = wr.get(mode, {}).get(k)
            if tr:
                w, n = widths(tr)
                rows.append(dict(sheet=nm, curve=g, kind="полоса", w=w, n_in=n,
                                 gt_pos=where_gt(tr, gts.get(g, {}))))
        # контроль: ЧЕСТНЫЕ кривые этого же листа (по лучшему из режимов)
        for g, gt in gts.items():
            ti = tmap.get(g)
            best = None
            for mode, w in wr.items():
                for k in (w if ti is None else [k2 for k2 in w if tmap.get(k2) == ti]):
                    m, c = err(w[k], gt)
                    if m is None:
                        continue
                    if best is None or m < best[0]:
                        best = (m, c, mode, k)
            if best and HON(best[0], best[1]):
                w, n = widths(wr[best[2]][best[3]])
                rows.append(dict(sheet=nm, curve=g, kind="честная", w=w, n_in=n,
                                 gt_pos=where_gt(wr[best[2]][best[3]], gt)))
        done += 1
        del rgb, cmap
        if done % 10 == 0:
            print(f"  {done}/{len(sheets)} листов, замеров {len(rows)}")
    tot = done + sum(skip.values())
    print(f"\n★ СВЕРКА: разобрано {done} + пропущено {sum(skip.values())} = {tot} против "
          f"{len(sheets)} листов шарда   {'★ СОШЛОСЬ' if tot == len(sheets) else '⛔ НЕ СОШЛОСЬ'}")
    for k, v in skip.items():
        print(f"    пропущено «{k}»: {v}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pickle.dump({"rows": rows, "shard": SH_I, "n_shards": SH_N},
                open(Path(a.out) / f"width_{SH_I}of{SH_N}.pkl", "wb"))
    print(f"дамп: width_{SH_I}of{SH_N}.pkl  ({len(rows)} замеров)")
    sys.exit()


if a.center:
    # ★★★ §6.151: ЗАПИСАННОЕ ЗНАЧЕНИЕ ПРОТИВ ЦЕНТРА РАНА, В КОТОРОМ ОНО ЛЕЖИТ.
    # Известно: штрих 12px, эксперт в центре, трасса на 3-10px в стороне, а прод НОМИНАЛЬНО берёт
    # центр рана (`trace2d.py:88`). Значит, точка уезжает либо ДО выбора рана (бинарь показал
    # прода не тот ран), либо ПОСЛЕ (`refine`/селектор). Замер разделяет эти два случая одним
    # проходом по ГОТОВОЙ выдаче: если |значение − центр рана ПРОДА| мало, а сам этот ран смещён
    # от штриха — виноват бинарь; если велико — виновато то, что после выбора.
    # ⚠ Контроль на ТЕХ ЖЕ листах обязателен по той же причине, что в `--width`: перо и
    # разрешение скана меняются от партии к партии.
    from dataset_build import find_image
    from auto import imaging as im
    from auto.config import Config
    from auto import rowdec
    P = Config().cv
    band = [ln.split("\t") for ln in LIST.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sheets = sorted({r[0] for r in band})
    if SH_N > 1:
        sheets = sheets[len(sheets) * SH_I // SH_N: len(sheets) * (SH_I + 1) // SH_N]
        print(f"★ ШАРД {SH_I}/{SH_N}: {len(sheets)} листов")
    by_sheet = {}
    for r in band:
        by_sheet.setdefault(r[0], []).append(r)
    rows, done, skip = [], 0, Counter()
    for nm in sheets:
        q = SRC.get(nm)
        img = find_image(q) if q else None
        if not img:
            skip["нет картинки"] += 1; continue
        stem = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
        gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
               if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        tmap = TRACK.get(nm, {})
        wr = {}
        for mode in a.modes:
            f = next(iter(sorted((Path(a.dir) / mode / stem).glob("*_auto.nlgx"))), None)
            if f:
                wr[mode] = curves(f)
        if not wr or not gts:
            skip["нет выдачи"] += 1; continue
        try:
            rgb = im.load_rgb(str(img))
        except Exception:
            skip["скан не читается"] += 1; continue
        cmap = rowdec._colmap(rgb, P)
        H, W = cmap.shape

        def centers(tr, gt):
            """→ dict медиан по кривой: куда уехало значение от каждого из трёх центров.

            ⚠ Все величины считаются на ОДНИХ И ТЕХ ЖЕ пробных строках — иначе разность медиан
            сравнивала бы разные подвыборки строк (та же ловушка, что §6.143: два счёта одного
            прогона обязаны сниматься одним проходом)."""
            d_prod, d_col, d_any, shift, d_gt, w_any, w_col = [], [], [], [], [], [], []
            # ★★ ДВА КОНТРОЛЯ, БЕЗ КОТОРЫХ ЗАМЕР НЕ ЧИТАЕТСЯ.
            # `gt_pos` — относительное положение ЭТАЛОНА в ране, считается ТОЧНО как `where_gt`
            # в `--width` (только строки, где эталон ВНУТРИ рана). Это сверка с §6.151: там
            # намерено 0.53 на полосе и 0.57 в контроле. Не сойдётся — значит стенд восстановил
            # ран не так, как прошлый замер, и остальные числа читать нельзя.
            # `tr_pos` — то же самое для ТРАССЫ. Это прямая проверка формулировки §6.151
            # «эксперт в центре, трасса у КРАЯ»: если trace тоже 0.5, формулировка неверна.
            gt_pos, tr_pos = [], []
            inside, e_in, e_out = [], [], []
            for y in sorted(tr)[::a.step]:
                y = int(y)
                if not (0 <= y < H):
                    continue
                rv = run_variants(int(round(tr[y])), cmap[y])
                if rv is None:
                    continue
                d_prod.append(tr[y] - rv["prod_c"])
                d_col.append(tr[y] - rv["col_c"])
                d_any.append(tr[y] - rv["any_c"])
                shift.append(rv["prod_c"] - rv["any_c"])
                w_any.append(rv["w_any"]); w_col.append(rv["w_col"])
                aL, aR = rv["any_L"], rv["any_R"]
                if aR > aL:
                    tr_pos.append((tr[y] - aL) / (aR - aL))
                    if y in gt and aL <= gt[y] <= aR:
                        gt_pos.append((gt[y] - aL) / (aR - aL))
                if y in gt:
                    d_gt.append(gt[y] - rv["any_c"])
                    # ★★★ РЕШАЮЩЕЕ РАЗЛОЖЕНИЕ. Трасса сидит в ЦЕНТРЕ своего рана (tr_pos = 0.50),
                    # эксперт — в центре СВОЕГО (gt_pos = 0.53). Если это ОДИН ран, промаха быть
                    # не может; значит вся полоса 3-10px обязана сидеть на строках, где эксперт
                    # ВНЕ рана под трассой. Проверяется прямо: доля таких строк и промах отдельно
                    # внутри и снаружи. Без этого замер говорит только «не эти трое».
                    inside.append(1.0 if aL <= gt[y] <= aR else 0.0)
                    (e_in if aL <= gt[y] <= aR else e_out).append(abs(tr[y] - gt[y]))
            if len(d_prod) < 10:
                return None
            med = lambda z: float(np.median(z)) if len(z) else None
            amed = lambda z: float(np.median(np.abs(z))) if len(z) else None
            return dict(n=len(d_prod),
                        d_prod=med(d_prod), ad_prod=amed(d_prod),
                        d_col=med(d_col), ad_col=amed(d_col),
                        d_any=med(d_any), ad_any=amed(d_any),
                        shift=med(shift), ashift=amed(shift),
                        d_gt=med(d_gt) if len(d_gt) >= 10 else None,
                        ad_gt=amed(d_gt) if len(d_gt) >= 10 else None,
                        gt_pos=med(gt_pos) if len(gt_pos) >= 10 else None,
                        tr_pos=med(tr_pos) if len(tr_pos) >= 10 else None,
                        gt_in=(float(np.mean(inside)) if len(inside) >= 10 else None),
                        e_in=med(e_in) if len(e_in) >= 10 else None,
                        e_out=med(e_out) if len(e_out) >= 10 else None,
                        w_any=med(w_any), w_col=med(w_col))

        want = {(r[1], r[2], r[3]) for r in by_sheet[nm]}      # (эталон, режим, имя выдачи)
        for (g, mode, k) in want:
            tr = wr.get(mode, {}).get(k)
            if tr:
                c = centers(tr, gts.get(g, {}))
                if c:
                    rows.append(dict(sheet=nm, curve=g, kind="полоса", **c))
        # контроль: ЧЕСТНЫЕ кривые этого же листа (по лучшему из режимов) — та же выборка, что
        # в `--width`, чтобы обе таблицы §6.151 читались на одних и тех же кривых
        for g, gt in gts.items():
            ti = tmap.get(g)
            best = None
            for mode, w in wr.items():
                for k in (w if ti is None else [k2 for k2 in w if tmap.get(k2) == ti]):
                    m, cv = err(w[k], gt)
                    if m is None:
                        continue
                    if best is None or m < best[0]:
                        best = (m, cv, mode, k)
            if best and HON(best[0], best[1]):
                c = centers(wr[best[2]][best[3]], gt)
                if c:
                    rows.append(dict(sheet=nm, curve=g, kind="честная", **c))
        done += 1
        del rgb, cmap
        if done % 10 == 0:
            print(f"  {done}/{len(sheets)} листов, замеров {len(rows)}")
    tot = done + sum(skip.values())
    print(f"\n★ СВЕРКА: разобрано {done} + пропущено {sum(skip.values())} = {tot} против "
          f"{len(sheets)} листов шарда   {'★ СОШЛОСЬ' if tot == len(sheets) else '⛔ НЕ СОШЛОСЬ'}")
    for k, v in skip.items():
        print(f"    пропущено «{k}»: {v}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pickle.dump({"rows": rows, "shard": SH_I, "n_shards": SH_N},
                open(Path(a.out) / f"center_{SH_I}of{SH_N}.pkl", "wb"))
    print(f"дамп: center_{SH_I}of{SH_N}.pkl  ({len(rows)} замеров)")
    sys.exit()


rows, done, skip = [], 0, Counter()
for nm in sheets:
    q = SRC.get(nm)
    img = find_image(q) if q else None
    if not img:
        skip["нет картинки"] += 1; continue
    stem = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    try:
        rgb = im.load_rgb(str(img))
    except Exception:
        skip["скан не читается"] += 1; continue
    cmap = rowdec._colmap(rgb, P)              # та же карта туши, что у прода (§6.142)
    H, W = cmap.shape
    for (_nm, g, mode, k, med, cov) in by_sheet[nm]:
        f = next(iter(sorted((Path(a.dir) / mode / stem).glob("*_auto.nlgx"))), None)
        if not f:
            skip["нет выдачи"] += 1; continue
        tr = curves(f).get(k)
        gt = gts.get(g)
        if not tr or not gt:
            skip["нет трассы"] += 1; continue
        own = other = paper = 0
        signed = []
        for y in sorted(tr)[::a.step]:
            y = int(y)
            if not (0 <= y < H) or y not in gt:
                continue
            x = int(round(tr[y]))
            x0, x1 = max(0, x - a.halfw), min(W, x + a.halfw + 1)
            if x1 <= x0 or not cmap[y, x0:x1].any():
                paper += 1; continue
            # чья это тушь: чья экспертная кривая ближе к этой точке
            near, dist = None, 1e9
            for gg, gv in gts.items():
                if y in gv:
                    dd = abs(gv[y] - x)
                    if dd < dist:
                        near, dist = gg, dd
            if near == g and dist <= a.near:
                own += 1
            elif near is not None and dist <= a.near:
                other += 1
            else:
                other += 1                      # тушь есть, но ничья ближняя — считаем не своей
            signed.append(tr[y] - gt[y])
        rows.append(dict(sheet=nm, curve=g, mode=mode, med=float(med), cov=float(cov),
                         own=own, other=other, paper=paper,
                         signed=float(np.median(signed)) if signed else None))
    done += 1
    del rgb, cmap
    if done % 10 == 0:
        print(f"  {done}/{len(sheets)} листов, кривых {len(rows)}")

tot = done + sum(skip.values())
print(f"\n★ СВЕРКА: разобрано {done} + пропущено {sum(skip.values())} = {tot} против "
      f"{len(sheets)} листов шарда   {'★ СОШЛОСЬ' if tot == len(sheets) else '⛔ НЕ СОШЛОСЬ'}")
for k, v in skip.items():
    print(f"    пропущено «{k}»: {v}")
Path(a.out).mkdir(parents=True, exist_ok=True)
pickle.dump({"rows": rows, "shard": SH_I, "n_shards": SH_N},
            open(Path(a.out) / f"drift_{SH_I}of{SH_N}.pkl", "wb"))
print(f"дамп: {Path(a.out) / f'drift_{SH_I}of{SH_N}.pkl'}  ({len(rows)} кривых)")

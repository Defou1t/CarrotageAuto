r"""_cov_cause.py — ВСКРЫТИЕ КОРЗИНЫ «НЕДОСТУПНА (cov<0.9)»: единственной крупной, не вскрытой ни разу.

§6.104 разложил экспертные кривые отгружаемой конфигурации по причинам: ведение 50.0%, отбор 18.9%,
★ ПОКРЫТИЕ 17.1% (76 кривых из 444). Отбор с тех пор разобран трижды (§6.105, §6.109), ведение —
вся ветка §6.95-§6.101, а корзина покрытия не открывалась НИ РАЗУ на отгружаемом пути.

⚠ ЧТО УЖЕ ЕСТЬ ПО ПОКРЫТИЮ И ПОЧЕМУ ЭТО НЕ ОТВЕТ (не переоткрытие закрытого, §6.28.1):
  §6.23 (20.07): 19 кривых ОДНОЙ скважины, прод-гейт, ЖАДНЫЙ путь — 46% пропущенных строк
                 восстановимы по туши, 53% ловушка §6.2. Выборка — одна скважина.
  §6.28.1 (21.07): те же строки на ОРАКУЛЬНОЙ полосе — 0.0% восстановимо. Сам §6.28.1 и предупреждает:
                 «не путать с §6.23 — там мерился ПРОД, где полосу ищет U1». Закрыт СТЕНД, не прод.
Обе — июль, до селектора ранов (§6.102), до обученной раскладки (§6.108) и на корпусе втрое меньше.

ВОПРОСЫ. Всё считается ОФЛАЙН по сохранённым пулам, картинки не читаются (проверка «есть ли тушь
под пропущенной строкой» — следующий шаг и стоит прогона по изображениям):
  1. НАСКОЛЬКО не хватает: распределение лучшего покрытия. 0.85 — вопрос порога и добивки,
     0.2 — ведение не пошло вовсе, и корзина на самом деле не про покрытие;
  2. трасса вообще НА ТОЙ кривой? med лучшего-по-покрытию кандидата: ≤3px — «точна, но неполна»
     (пул §6.23, один шаг от честной), >3px — корзина про идентичность;
  3. ГДЕ теряется: верх / низ / дыры внутри пролёта трассы. Концы лечит окно и добивка, дыры — max_gap;
  4. ЧИНИТ ЛИ СШИВКА: покрывает ли ОБЪЕДИНЕНИЕ точных кандидатов трека ≥0.9 строк GT. Если да —
     строки уже прослежены, но разными линиями, и это задача склейки, а не ведения;
  5. ЦЕНА: сколько кривых стали бы честными, будь покрытие закрыто (верхняя граница выигрыша);
  6. КОНЦЕНТРАЦИЯ по мнемоникам и скважинам — корзина КЛАСС оказалась на 59% из двух мнемоник (§6.109).

КОНТРОЛИ:
  (1) разложение §6.104 воспроизводится ТЕМ ЖЕ кодом (`err`/`HON` скопированы из `_accuracy_status.py`
      дословно) и печатается рядом — если корзины разошлись, читать нечего;
  (2) сверка «прочитано + дублей пропущено = найдено файлов» (обязательный минимум §6.106);
  (3) сумма подкорзин печатается против размера корзины;
  (4) ⚠ порог `len(com) < 30` самой метрики считается ОТДЕЛЬНО: кривая короче 34 строк не может стать
      честной ПО ОПРЕДЕЛЕНИЮ (cov≥0.9 от <34 строк — это <30 общих), и такие обязаны быть видны.

  python _cov_cause.py [--pools ...] [--limit N] [--examples 15]
"""
import sys, argparse, collections, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
# ⚠ ТОТ ЖЕ СПИСОК И ТОТ ЖЕ ПОРЯДОК, что у прогона §6.109 и у идущего A/B (`ab_w202`): дедупликация
# берёт ПЕРВЫЙ каталог, в котором встретилось имя листа, поэтому порядок — часть определения выборки.
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--limit", type=int, default=0, help="взять первые N листов (0 = все) — для прогрева")
ap.add_argument("--examples", type=int, default=15)
a = ap.parse_args()

HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9
MINCOM = 30  # порог самой метрики (`_accuracy_status.err`), вынесен, чтобы его можно было считать


def err(tr, gt):
    """Дословно из `_accuracy_status.py`. Менять нельзя: на нём стоит разложение §6.104."""
    com = [y for y in tr if y in gt]
    if len(com) < MINCOM:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


def raw(tr, gt):
    """Диагностическая пара БЕЗ порога 30 строк: (med, cov, общих строк).

    ⚠ Нужна отдельно от `err`, потому что `err` возвращает cov=0.0 при <30 общих строк — и короткая
    трасса выглядела бы «непокрывающей», хотя она может кроить свою кривую целиком. Для приговора
    берётся `err`, для разбора — `raw`; расхождение между ними и есть корзина «порог метрики»."""
    com = [y for y in tr if y in gt]
    if not com:
        return None, 0.0, 0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt)), len(com)


WELL = {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WELL[q.name] = wlg.parent.name

CAUSE = ["1 нет слота", "2 нет кандидатов", "3 недоступна (cov<0.9)", "4 далеко (>3px)",
         "5 ★ есть в пуле, не выбрана", "6 ★★ записана честно"]
cnt = {c: 0 for c in CAUSE}

# подкорзины корзины 3
sub = collections.Counter()
sub_b_traced = 0      # из подкорзины B: строки прослежены, но ЧУЖОЙ линией (cov_all ≥ 0.9)
sub_b_other = 0       # из подкорзины B: точная трасса ЕСТЬ, но на ДРУГОМ треке листа (вина полосы)
cd_rows = []          # выгрузка корзин C+D для следующего шага (проверка туши по изображениям)
covs, covs_acc, meds, miss_top, miss_bot, miss_hole, gtlen = [], [], [], [], [], [], []
by_mnem, by_well = collections.Counter(), collections.Counter()
tot_by_mnem, tot_by_well = collections.Counter(), collections.Counter()
examples = []

# ★ ВТОРОЙ СРЕЗ, НА УРОВНЕ ВЫДАННОЙ КРИВОЙ. Корзины §6.104 считаются по КАНДИДАТАМ в пуле, а §6.23
# мерил другое — какой получилась ЗАПИСАННАЯ кривая. «Точна, но неполна» на выдаче размазана у
# §6.104 по корзинам 3 и 5, поэтому её надо считать отдельно, иначе её просто не видно.
# ⚠ Считается по `written` САМИХ ПУЛОВ (тот же объект, что раскладывал §6.104), не по свежей отгрузке.
wcase = collections.Counter()
w_in = collections.Counter()   # куда попадают «точна, но неполна» из среза выдачи
ws_rows = []                   # выгрузка b+c: записана точно, но коротко — адрес покрытия

nfile = nsheet = ndup = 0
seen = set()
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        nfile += 1
        if f.stem in seen:
            ndup += 1
            continue
        seen.add(f.stem)
        if a.limit and nsheet >= a.limit:
            continue
        d = pickle.load(open(f, "rb"))
        nsheet += 1
        slots = {s["name"]: s for s in d["slots"]}
        tl = {}
        for i, ln in enumerate(d["lines"]):
            tl.setdefault(ln["track"], []).append(i)
        wl = WELL.get(d["name"], "?")
        for nm, gt in d["gts"].items():
            mnem = nm.split()[0] if nm.split() else nm
            tot_by_mnem[mnem] += 1
            tot_by_well[wl] += 1
            # срез по ВЫДАННОЙ кривой (независим от корзин, считается по всем кривым)
            ww = d["written"].get(nm)
            if ww is None:
                wcase["1 не записана вовсе"] += 1
            else:
                wm, wc, wn = raw(ww, gt)
                if wn < MINCOM:
                    wcase["2 записана, но общих строк <30 — метрика не считает"] += 1
                elif wm <= 3.0 and wc >= 0.9:
                    wcase["6 ★★ честная (med≤3, cov≥0.9)"] += 1
                elif wm <= 3.0:
                    wcase["5 ★ ТОЧНА, НО НЕПОЛНА (med≤3, cov<0.9) — пул §6.23"] += 1
                elif wc >= 0.9:
                    wcase["4 полна, но не та (cov≥0.9, med>3) — идентичность"] += 1
                else:
                    wcase["3 и неточна, и неполна"] += 1
            s = slots.get(nm)
            if s is None:
                cnt[CAUSE[0]] += 1
                continue
            cand = tl.get(s["track"], [])
            if not cand:
                cnt[CAUSE[1]] += 1
                continue
            ms = [err(d["lines"][i]["tr"], gt) for i in cand]
            cov_ok = [(m, c) for m, c in ms if c >= 0.9 and m is not None]
            w = d["written"].get(nm)
            wr = err(w, gt) if w is not None else (None, 0.0)
            # ★ КУДА ПОПАДАЮТ «ТОЧНА, НО НЕПОЛНА» ИЗ СРЕЗА ВЫДАЧИ. Догадка «раз записана короткая
            # точная, значит длинной не было» неверна: полностью покрывающий кандидат мог БЫТЬ —
            # точный (тогда это ошибка порядка) или неточный (тогда записана лучшая из имеющихся).
            if ww is not None and wm is not None and wm <= 3.0 and wc < 0.9 and wn >= MINCOM:
                # ⚠ Ярлык — по тому, ЧЕМ это чинится, а не по тому, что лежало в пуле: полный, но
                # НЕточный кандидат делу не помогает, значит случаи b и c одинаково упираются в
                # удлинение уже записанной трассы, и только a чинится порядком.
                if any(m is not None and m <= 3.0 and c >= 0.9 for m, c in ms):
                    w_in["a полный ТОЧНЫЙ кандидат в пуле БЫЛ ⇒ чинится ОТБОРОМ"] += 1
                else:
                    w_in["b полный был, но неточный ⇒ чинится только удлинением ⇒ ПОКРЫТИЕ"
                         if any(c >= 0.9 for m, c in ms) else
                         "c полного не было вовсе ⇒ ПОКРЫТИЕ"] += 1
                    # ★ ЭТО И ЕСТЬ АДРЕС ПОКРЫТИЯ: кривая УЖЕ записана и УЖЕ точна, отбор её проходить
                    # не надо — не хватает только строк. Выгружаем поимённо вместе с тем, ГДЕ именно
                    # их не хватает у САМОЙ ЗАПИСАННОЙ трассы (вход для проверки туши по §6.23).
                    g = np.fromiter(gt.keys(), int)
                    lo, hi = min(ww), max(ww)
                    ng = max(1, len(g))
                    have = np.fromiter((y in ww for y in g), bool, len(g))
                    ws_rows.append((d["name"], nm, wl, f"{wc:.3f}", f"{wm:.2f}", len(gt),
                                    f"{float((g < lo).sum())/ng:.3f}",
                                    f"{float((g > hi).sum())/ng:.3f}",
                                    f"{float(((g >= lo) & (g <= hi) & ~have).sum())/ng:.3f}"))
            if HON(*wr):
                cnt[CAUSE[5]] += 1
                continue
            if not cov_ok:
                # ================= КОРЗИНА 3: РАЗБОР =================
                cnt[CAUSE[2]] += 1
                by_mnem[mnem] += 1
                by_well[wl] += 1
                gtlen.append(len(gt))
                rs = [raw(d["lines"][i]["tr"], gt) for i in cand]
                # ⚠⚠ ОПОРНЫЙ КАНДИДАТ — СНАЧАЛА ТОЧНЫЙ, ПОТОМ САМЫЙ ПОКРЫВАЮЩИЙ. Первая редакция
                # брала argmax(cov) и спрашивала «на той ли он кривой» — но тогда кривая, у которой
                # ЕСТЬ точная трасса на 0.85 и есть чужая на 0.88, уезжала в «ведение», хотя это
                # ровно пул «точна, но неполна» (§6.23). Приговор корзины от этого сдвигался в
                # сторону ведения, то есть в сторону уже известного ответа.
                accur = [k for k, r in enumerate(rs)
                         if r[0] is not None and r[0] <= 3.0 and r[2] >= MINCOM]
                pool_i = accur if accur else list(range(len(rs)))
                bi = max(pool_i, key=lambda k: rs[k][1])
                bmed, bcov, bcom = rs[bi]
                covs.append(bcov)
                meds.append(bmed if bmed is not None else np.inf)
                # ★ СШИВКА: объединение строк ТОЧНЫХ кусков трека. Порог общих строк здесь НИЖЕ
                # (5, не 30): смысл сшивки как раз в коротких кусках, которые поодиночке метрике
                # не видны. Верхняя граница `cov_all` — объединение ВСЕХ линий трека без разбора.
                acc5 = [i for i, r in zip(cand, rs) if r[0] is not None and r[0] <= 3.0 and r[2] >= 5]
                uni = set()
                for i in acc5:
                    uni |= d["lines"][i]["tr"].keys() & gt.keys()
                cov_uni = len(uni) / max(1, len(gt))
                uni_all = set()
                for i in cand:
                    uni_all |= d["lines"][i]["tr"].keys() & gt.keys()
                cov_all = len(uni_all) / max(1, len(gt))
                # ВЗАИМНО ИСКЛЮЧАЮЩИЕ ПОДКОРЗИНЫ, порядок = приоритет объяснения.
                # ⚠ Развилка «покрытие или ведение» решается НАЛИЧИЕМ ТОЧНОЙ ТРАССЫ (`accur`),
                # а не свойствами самой покрывающей: пока на треке нет ни одной линии, идущей по
                # СВОЕЙ кривой, речь не о недоборе покрытия, а о том, что вести не за чем.
                # ⚠ C ПРОВЕРЯЕТСЯ РАНЬШЕ B: «сшивка закрывает» — утверждение более сильное, чем
                # «длинной точной трассы нет». Кривая, собираемая из коротких точных кусков (каждый
                # короче порога метрики в 30 строк), иначе молча уехала бы в ведение.
                if len(gt) < 34:
                    sub["A порог метрики: кривая короче 34 строк"] += 1
                elif cov_uni >= 0.9:
                    sub["C ★ СШИВКА ЗАКРЫВАЕТ: точные куски трека кроют ≥0.9 строк"] += 1
                elif not accur:
                    sub["B ТОЧНОЙ трассы на треке нет вовсе — это ведение, не покрытие"] += 1
                    if cov_all >= 0.9:
                        sub_b_traced += 1
                    # ⚠ РАЗВИЛКА ВНУТРИ B: «вести не за чем» или «оно прослежено, но НА ЧУЖОМ ТРЕКЕ»?
                    # Раскладка смотрит только трек слота (`auto/emit.py`), поэтому для КОРЗИНЫ это
                    # одно и то же, а для ПЛАНА — совсем нет: второе чинится полосой (U1), а не
                    # ведением. Считается по всем линиям листа, стоит одного лишнего прохода.
                    elsewhere = any(
                        (r[0] is not None and r[0] <= 3.0 and r[2] >= MINCOM)
                        for r in (raw(ln["tr"], gt) for j, ln in enumerate(d["lines"])
                                  if j not in cand))
                    if elsewhere:
                        sub_b_other += 1
                else:
                    sub["D ★ ТОЧНА, НО НЕПОЛНА: своя трасса есть, строк не хватает"] += 1
                # «где теряется» и распределение покрытия — ТОЛЬКО там, где опорная трасса СВОЯ:
                # у чужой линии вопрос «где потеряли строки» смысла не имеет.
                if accur:
                    covs_acc.append(bcov)
                    tr = d["lines"][cand[bi]]["tr"]
                    mt = mb = mh = 0.0
                    if tr:
                        lo, hi = min(tr), max(tr)
                        g = np.fromiter(gt.keys(), int)
                        ng = max(1, len(g))
                        inside = (g >= lo) & (g <= hi)
                        have = np.fromiter((y in tr for y in g), bool, len(g))
                        mt = float((g < lo).sum()) / ng
                        mb = float((g > hi).sum()) / ng
                        mh = float((inside & ~have).sum()) / ng
                        miss_top.append(mt); miss_bot.append(mb); miss_hole.append(mh)
                    # ★ ВЫГРУЗКА КОРЗИН C+D ПОИМЁННО: следующий шаг (есть ли тушь под пропущенными
                    # строками, §6.23) идёт по изображениям и стоит отдельного прогона — список
                    # кривых он должен получить готовым, а не пересчитывать 2677 дампов заново.
                    cd_rows.append((d["name"], nm, wl, f"{bcov:.3f}", f"{bmed:.2f}",
                                    f"{cov_uni:.3f}", f"{cov_all:.3f}", len(gt),
                                    f"{mt:.3f}", f"{mb:.3f}", f"{mh:.3f}"))
                    if len(examples) < a.examples:
                        examples.append((d["name"][:44], nm[:14], bcov, bmed, cov_uni, cov_all,
                                         len(gt), wl[:12]))
                continue
            bm = min(m for m, _ in cov_ok)
            if bm <= 3.0:
                cnt[CAUSE[4]] += 1
            else:
                cnt[CAUSE[3]] += 1

W = 96
tot = sum(cnt.values())
print(f"{'='*W}")
print(f"ВЫБОРКА: файлов найдено {nfile}, прочитано листов {nsheet}, дублей пропущено {ndup}"
      + (f"  ⚠ ОГРАНИЧЕНО --limit {a.limit}" if a.limit else ""))
ok = (nsheet + ndup == nfile) or bool(a.limit)
print(f"  СВЕРКА СПИСКА: {nsheet} + {ndup} = {nsheet+ndup} против найденных {nfile}   "
      + ("★ СОШЛОСЬ" if ok else "⛔ РАСХОЖДЕНИЕ"))
print(f"  экспертных кривых {tot}, скважин {len(tot_by_well)}")
print(f"{'='*W}")

print("\nКОНТРОЛЬ: ПОЛНОЕ РАЗЛОЖЕНИЕ ТЕМ ЖЕ КОДОМ (сверять с §6.104: 3.8/17.1/50.0/18.9/10.1%)")
for c in CAUSE:
    print(f"  {c:<34}{cnt[c]:>8}{100*cnt[c]/max(1,tot):>7.1f}%")

print(f"\nСРЕЗ ПО ВЫДАННОЙ КРИВОЙ (не по кандидатам): всего {sum(wcase.values())}   "
      + ("★ СОШЛОСЬ" if sum(wcase.values()) == tot else "⛔ РАСХОЖДЕНИЕ с разложением"))
for k, v in sorted(wcase.items()):
    print(f"  {k:<54}{v:>8}{100*v/max(1,tot):>7.1f}%")

n5 = wcase.get("5 ★ ТОЧНА, НО НЕПОЛНА (med≤3, cov<0.9) — пул §6.23", 0)
if n5:
    print(f"\n  КУДА ПОПАДАЮТ ЭТИ {n5} «ТОЧНА, НО НЕПОЛНА» (был ли на треке полный кандидат):")
    for k, v in sorted(w_in.items()):
        print(f"    {k:<62}{v:>6}{100*v/max(1,n5):>7.1f}%")
    print(f"    сумма {sum(w_in.values())} против {n5}   "
          + ("★ СОШЛОСЬ" if sum(w_in.values()) == n5 else
             f"⚠ разница {n5-sum(w_in.values())} — кривые без слота/кандидатов"))
    if ws_rows:
        mh = np.array([float(r[8]) for r in ws_rows])
        me = np.array([float(r[6]) + float(r[7]) for r in ws_rows])
        print(f"  ★ b+c = {len(ws_rows)} кривых ({100*len(ws_rows)/max(1,tot):.1f}% выборки) записаны "
              f"ТОЧНО, но коротко ⇒ честными их делает только удлинение, отбор им проходить НЕ НАДО")
        print(f"    где не хватает у САМОЙ записанной: дыры {mh.mean():.1%} длины против "
              f"{me.mean():.1%} на концах; дыры больше концов у {int((mh > me).sum())} из {len(mh)}")
        dump2 = Path("F:/nds/output/taskS/cov_written_short.tsv")
        with open(dump2, "w", encoding="utf-8") as fh:
            fh.write("лист\tкривая\tскважина\tcov\tmed\tстрокGT\tверх\tниз\tдыры\n")
            for r in ws_rows:
                fh.write("\t".join(str(x) for x in r) + "\n")
        print(f"    выгружено поимённо → {dump2}")

n3 = cnt[CAUSE[2]]
print(f"\n{'='*W}\nКОРЗИНА «НЕДОСТУПНА (cov<0.9)»: {n3} кривых = {100*n3/max(1,tot):.1f}% выборки\n{'='*W}")
print(f"СУММА ПОДКОРЗИН {sum(sub.values())} против размера корзины {n3}   "
      + ("★ СОШЛОСЬ" if sum(sub.values()) == n3 else "⛔ РАСХОЖДЕНИЕ"))
nB = 0
for k, v in sorted(sub.items()):
    print(f"  {k:<62}{v:>7}{100*v/max(1,n3):>7.1f}%")
    if k.startswith("B"):
        nB = v
if nB:
    print(f"     из них строки всё же прослежены — но ЧУЖОЙ линией (объединение трека ≥0.9): "
          f"{sub_b_traced} из {nB}")
    print(f"     ★ из них ТОЧНАЯ трасса есть, но НА ДРУГОМ ТРЕКЕ листа (вина полосы, не ведения): "
          f"{sub_b_other} из {nB} = {100*sub_b_other/nB:.0f}%")
nCD = sub.get("C ★ СШИВКА ЗАКРЫВАЕТ: точные куски трека кроют ≥0.9 строк", 0) \
    + sub.get("D ★ ТОЧНА, НО НЕПОЛНА: своя трасса есть, строк не хватает", 0)
print(f"\n★ ЦЕНА КОРЗИНЫ = C+D: {nCD} кривых ({100*nCD/max(1,n3):.1f}% корзины, "
      f"{100*nCD/max(1,tot):.1f}% выборки) стоят НА СВОЕЙ кривой, и не хватает только покрытия.")
print(f"  Остальное ({n3-nCD}) — не покрытие: своей трассы на треке нет вовсе.")

if covs_acc:
    cv = np.array(covs_acc)
    print(f"\nПОКРЫТИЕ СВОЕЙ (точной) ТРАССЫ, где она есть — {len(cv)} кривых, порог честности 0.9;"
          f" медиана {np.median(cv):.2f}")
    for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .8), (.8, .9), (.9, 1.01)):
        n = int(((cv > lo) & (cv <= hi)).sum())
        print(f"  {lo:.1f}-{hi:.1f}{n:>8}{100*n/len(cv):>7.1f}%  " + "█" * int(46 * n / len(cv)))
    near = int((cv > 0.8).sum())
    print(f"⇒ у {near} кривых ({100*near/len(cv):.1f}% из них) не хватает менее 20% строк")

if covs:
    md = np.array(meds, float)
    print(f"\nMED ОПОРНОГО КАНДИДАТА ПО ВСЕЙ КОРЗИНЕ (точный, если он есть; иначе покрывающий)")
    for lo, hi in ((0, 1), (1, 3), (3, 10), (10, 100), (100, 1e9)):
        n = int(((md > lo) & (md <= hi)).sum())
        lab = f">{lo:g}px" if hi > 1e8 else f"{lo:g}-{hi:g}px"
        print(f"  {lab:<10}{n:>8}{100*n/len(md):>7.1f}%  " + "█" * int(46 * n / len(md)))
    print(f"  медиана {np.median(md[np.isfinite(md)]):.1f}px по {int(np.isfinite(md).sum())} "
          f"считаемым из {len(md)}")

if miss_top:
    mt, mb, mh = np.array(miss_top), np.array(miss_bot), np.array(miss_hole)
    print(f"\nГДЕ ТЕРЯЕТСЯ ПОКРЫТИЕ (доли длины кривой, среднее по {len(mt)} кривым C+D)")
    print(f"  верх (выше начала трассы)   {mt.mean():>6.1%}")
    print(f"  низ  (ниже конца трассы)    {mb.mean():>6.1%}")
    print(f"  дыры внутри пролёта         {mh.mean():>6.1%}")
    ends = mt + mb
    print(f"⇒ концы против дыр: {ends.mean():.1%} против {mh.mean():.1%}; "
          f"кривых, где дыры больше концов: {int((mh > ends).sum())} из {len(mh)}")

if by_mnem:
    print(f"\nКОНЦЕНТРАЦИЯ ПО МНЕМОНИКАМ (доля — сколько кривых ЭТОЙ мнемоники попало в корзину)")
    print(f"  {'мнемоника':<14}{'в корзине':>10}{'всего':>8}{'доля':>8}")
    for m, v in by_mnem.most_common(12):
        print(f"  {m[:13]:<14}{v:>10}{tot_by_mnem[m]:>8}{100*v/max(1,tot_by_mnem[m]):>7.0f}%")
    top2 = sum(v for _, v in by_mnem.most_common(2))
    print(f"⇒ две верхние мнемоники — {100*top2/max(1,n3):.0f}% корзины (у корзины КЛАСС было 59%)")

    WMIN, WTOP = 20, 10   # порог «скважина крупная» и длина списка — не текстом, а из счётчика
    rows = [(w, v, tot_by_well[w]) for w, v in by_well.items() if tot_by_well[w] >= WMIN]
    print(f"\nХУДШИЕ СКВАЖИНЫ ПО ВСЕЙ КОРЗИНЕ (B+C+D): {len(rows)} скважин от {WMIN} кривых, "
          f"верхние {min(WTOP, len(rows))}")
    print(f"  {'скважина':<26}{'в корзине':>10}{'всего':>8}{'доля':>8}")
    for w, v, t in sorted(rows, key=lambda q: -q[1] / max(1, q[2]))[:WTOP]:
        print(f"  {w[:25]:<26}{v:>10}{t:>8}{100*v/max(1,t):>7.0f}%")

if examples:
    print(f"\nПРИМЕРЫ C+D (своя трасса есть): cov — её покрытие, uni — сшивка точных, all — все линии")
    print(f"  {'лист':<46}{'кривая':<15}{'cov':>6}{'med':>7}{'uni':>7}{'all':>7}{'строк':>8}"
          f"  скважина")
    for nmz, cz, bc, bm2, cu, ca, lg, wz in examples:
        print(f"  {nmz:<46}{cz:<15}{bc:>6.2f}{bm2:>7.1f}{cu:>7.2f}{ca:>7.2f}{lg:>8}  {wz}")

if cd_rows:
    dump = Path("F:/nds/output/taskS/cov_cd.tsv")
    with open(dump, "w", encoding="utf-8") as fh:
        fh.write("лист\tкривая\tскважина\tcov\tmed\tcov_uni\tcov_all\tстрокGT\tверх\tниз\tдыры\n")
        for r in cd_rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"\n★ ВЫГРУЖЕНО {len(cd_rows)} кривых C+D → {dump} (вход для проверки туши по изображениям)")

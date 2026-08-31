r"""_rowdec_color.py — НЕСЁТ ЛИ ТРАЕКТОРИЯ ДЕКОДЕРА ЦВЕТ, ПО КОТОРОМУ ЕЁ МОЖНО УЗНАТЬ (§6.141 → задача 1)

ЗАЧЕМ. §6.141 намерил на отгрузке: построчный декодер даёт **+61 кривую БЕЗ ИМЕНИ** и **−41 С
ИМЕНЕМ**. Знак подписи устойчив во всех трёх прогонах (−7, −31, −41), и ни привязка 1:1, ни
передача решения раскладке его не чинят ⇒ причина не в том, КОМУ отдать решение, а в том, что
траектория декодера не несёт признаков, по которым линия узнаётся. Прод ведёт каждую линию по
маске ЕЁ ЦВЕТА (`trace2d._color_fg`) — то есть цвет у него в построении; декодер работает по
одному серому «насколько темнее бумаги» (`rowdec._band`) и цвета не видит вовсе.
⇒ Кандидат: считать цвет ПОД траекторией и отдать его раскладке вместе с ней.

ЧТО МЕРЯЕТ ЭТОТ СТЕНД (пайплайн НЕ запускается, читается ГОТОВАЯ выдача прогона):
  1. **МЕРИТСЯ ЛИ ЦВЕТ** — контроль на кривых, названных ВЕРНО: совпадает ли цвет под выданной
     траекторией с цветом под экспертной кривой того же имени. Если нет — рычага нет, и дальше
     читать нечего (та же дисциплина, что нуль-контроль §6.20.3).
  2. **РАЗЛИЧАЕТ ЛИ ЦВЕТ ЛИНИИ ТРЕКА** — доля кривых, чей цвет УНИКАЛЕН внутри трека. Это верхняя
     граница: там, где две кривые трека одного цвета, цвет не разведёт их никогда.
  3. * **ПОТОЛОК РЫЧАГА** — сколько кривых, у которых геометрия ЧЕСТНАЯ, но имя ЧУЖОЕ, цвет
     указал бы верно. Это и есть та часть −41, которую механизм может вернуть.

⚠⚠ ЭТО ПОТОЛОК, А НЕ ПРИРОСТ. §6.113 durable: оценка резерва по агрегату обязана проверяться
механизмом. Здесь цвет под траекторией сравнивается с ЭТАЛОННЫМ, то есть со знанием, которого на
выдаче нет. Механизм получит меньше.
⚠ И §6.13: цвет в `mnemonics.json` — НЕ доменный факт (SP нарисована чёрной на 17 листах из 20).
Поэтому эталонный цвет берётся ИЗ ТУШИ под экспертной кривой, а не из словаря.

  <ComfyUI>\python_embeded\python.exe _rowdec_color.py --dir F:/nds/output/taskS/ab_rowdec_pair \
      --mode B --shard 0/8 --out F:/nds/output/taskS/rdcolor
  ...потом:  python _rowdec_color.py --sum --out F:/nds/output/taskS/rdcolor
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:/nds/output/taskS/ab_rowdec_pair")
ap.add_argument("--mode", default="B", help="подкаталог режима, чью подпись разбираем")
ap.add_argument("--out", default=r"F:/nds/output/taskS/rdcolor")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--step", type=int, default=8, help="каждая N-я строка траектории")
ap.add_argument("--halfw", type=int, default=3, help="полуокно по x вокруг траектории, px")
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--sum", action="store_true")
a = ap.parse_args()
SH_I, SH_N = (int(z) for z in a.shard.split("/"))
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


def dom(cnt, floor=20):
    """→ (цвет, чистота) по счётчику; (None, 0) если счётчик пуст или слишком мал.

    ⚠ Считается ДВАЖДЫ и по-разному, потому что ответы расходятся: по СУММЕ ПИКСЕЛЕЙ пересечение
    с соседкой другого цвета вносит целый сгусток и перебивает всю кривую (замер: под экспертной
    `AK1` 865 чёрных против 300 красных — красные принесены одним пересечением). По ГОЛОСАМ СТРОК
    пересечение стоит ровно столько строк, сколько занимает. Ведущим считается голос строк."""
    t = sum(cnt.values())
    if t < floor:
        return None, 0.0
    c = max(cnt, key=lambda k: cnt[k])
    return c, cnt[c] / t


def vdom(votes):
    """→ (цвет, доля строк) по голосам строк; порог — 8 строк с тушью."""
    return dom(votes, floor=8)


# ═══ СВОДКА ═══════════════════════════════════════════════════════════════════════════════════
if a.sum:
    parts = sorted(Path(a.out).glob("rdcolor_*of*.pkl"))
    want, rows, sheets, have = None, [], 0, set()
    for f in parts:
        d = pickle.load(open(f, "rb"))
        want = want or d["n_shards"]
        have.add(d["shard"]); rows += d["rows"]; sheets += d["sheets"]
    print(f"шардов {len(have)} из {want}"
          f"{'' if want and len(have) == want else '   ⚠⚠ ПРОГОН НЕПОЛНЫЙ'}")
    print(f"листов {sheets}, разобранных кривых выдачи: {len(rows)}")
    if not rows:
        sys.exit("нет данных")

    # ⚠ ВСЕ ТРИ ВОПРОСА СЧИТАЮТСЯ ДВАЖДЫ — по сумме пикселей и по голосам строк (см. `dom`).
    # Расхождение двух прочтений само по себе ответ: если они спорят, цвет вынимается неустойчиво.
    for READ, KW, KG, F in (("ГОЛОСА СТРОК", "vot_w", "vot_gt", vdom),
                            ("сумма пикселей", "col_w", "col_gt", dom)):
        print(f"\n{'='*78}\n>>> ПРОЧТЕНИЕ: {READ}\n{'='*78}")
        # 1. МЕРИТСЯ ЛИ ЦВЕТ: на ВЕРНО названных — цвет под выдачей против цвета под эталоном
        ok = [r for r in rows if r["hon_named"]]
        agree = nomeas = 0
        pure, conf = [], Counter()
        for r in ok:
            cw, pw = F(r[KW]); cg, pg = F(r[KG].get(r["name"], {}))
            if cw is None or cg is None:
                nomeas += 1; continue
            agree += (cw == cg); pure.append(min(pw, pg)); conf[(cg, cw)] += 1
        n1 = len(ok) - nomeas
        print(f"\n* 1. МЕРИТСЯ ЛИ ЦВЕТ (контроль на ВЕРНО названных, {len(ok)} кривых)")
        print(f"   тушь под траекторией не найдена:  {nomeas}")
        print(f"   цвет выдачи = цвет эталона:       {agree} из {n1} ({100*agree/max(1,n1):.1f}%)")
        print(f"   медиана чистоты цвета:            {np.median(pure) if pure else 0:.2f}")
        bad_conf = [(k, n) for k, n in conf.most_common() if k[0] != k[1]]
        if bad_conf:
            print("   путаница (эталон → выдача), верх: "
                  + ", ".join(f"{cg}→{cw} {n}" for (cg, cw), n in bad_conf[:6]))

        # 2. РАЗЛИЧАЕТ ЛИ ЦВЕТ ЛИНИИ ТРЕКА (по туши под ЭКСПЕРТНЫМИ кривыми трека)
        uq = tt = 0
        for r in rows:
            cs = [F(v)[0] for v in r[KG].values()]
            cs = [c for c in cs if c]
            if len(cs) < 2 or not r["true"]:
                continue
            me = F(r[KG].get(r["true"], {}))[0]
            if me:
                tt += 1; uq += (Counter(cs)[me] == 1)
        print(f"\n* 2. РАЗЛИЧАЕТ ЛИ ЦВЕТ ЛИНИИ ТРЕКА (кривых на треках, где кривых ≥2: {tt})")
        print(f"   цвет ВЕРНОЙ кривой уникален на треке: {uq} ({100*uq/max(1,tt):.1f}%)")

        # 3. ПОТОЛОК: честная геометрия, чужое имя — укажет ли цвет верную?
        bad = [r for r in rows if (not r["hon_named"]) and r["hon_free"] and r["true"]]
        fix = amb = wrong = nom = 0
        for r in bad:
            cw = F(r[KW])[0]
            ct = F(r[KG].get(r["true"], {}))[0]
            if cw is None or ct is None:
                nom += 1; continue
            others = [F(r[KG][k])[0] for k in r[KG] if k != r["true"]]
            if cw != ct:
                wrong += 1
            elif cw in others:
                amb += 1
            else:
                fix += 1
        print(f"\n** 3. ПОТОЛОК РЫЧАГА (геометрия честная, имя ЧУЖОЕ: {len(bad)} кривых)")
        print(f"   тушь не померилась:                  {nom}")
        print(f"   цвет выдачи ≠ цвету верной кривой:   {wrong}")
        print(f"   цвет верный, но на треке НЕ один:    {amb}")
        print(f"   * ЦВЕТ УКАЗЫВАЕТ ВЕРНУЮ ОДНОЗНАЧНО:  {fix}"
              f"  ({100*fix/max(1,len(bad)):.1f}% ошибок подписи)")
    print(f"\n⚠ Это ПОТОЛОК: эталонный цвет на выдаче недоступен, механизм получит меньше.")

    # ── 4. МЕХАНИЗМ, А НЕ АГРЕГАТ: ПЕРЕНАЗНАЧИТЬ ИМЕНА ВНУТРИ ТРЕКА ПО ЦВЕТУ ────────────────
    # ⚠⚠ §6.113 durable: оценка резерва по агрегату обязана проверяться механизмом. Пункт 3 считает
    # «скольким цвет указал бы верную» поштучно и не спрашивает, СОГЛАСОВАНЫ ли эти указания внутри
    # трека. Здесь — согласованы: внутри каждого трека делается взаимно однозначное назначение
    # «траектория → имя» со стоимостью 0/1 по совпадению цвета, а ничьи разрешаются в пользу
    # НЫНЕШНЕГО имени (то есть цвет только переставляет то, что он различает, и ничего не портит
    # там, где молчит). Считаем честных С ИМЕНЕМ до и после.
    # ⚠ Это всё ещё ОЦЕНКА СВЕРХУ для механизма в `rowdec`: там цвет — слагаемое стоимости рядом с
    # положением, а здесь он решает единолично. Но это уже НАЗНАЧЕНИЕ, а не подсчёт указаний.
    by_track = {}
    for r in rows:
        by_track.setdefault((r["sheet"], r["track"]), []).append(r)

    def assign(cost, n, m):
        """Точное назначение перебором по маске (n ≤ m ≤ 12), иначе жадно по стоимости.
        ⚠ Траекторий может быть БОЛЬШЕ, чем экспертных кривых трека (декодер выдаёт k по числу
        ЛИНИЙ U1, а эксперт мог разметить меньше) — тогда часть остаётся без имени, и перебор по
        маске такого случая не покрывает: жадный путь."""
        if n > m or n > 12 or m > 12:
            used, res = set(), [-1] * n
            for i, j in sorted(((i, j) for i in range(n) for j in range(m)),
                               key=lambda z: cost[z[0]][z[1]]):
                if res[i] >= 0 or j in used:
                    continue
                res[i] = j; used.add(j)
            return res
        INF = float("inf")
        dp = [INF] * (1 << m); dp[0] = 0.0; ch = [None] * (1 << m)
        order = []
        for i in range(n):
            nxt = [INF] * (1 << m); nch = [None] * (1 << m)
            for msk in range(1 << m):
                if dp[msk] == INF or bin(msk).count("1") != i:
                    continue
                for j in range(m):
                    if msk >> j & 1:
                        continue
                    v = dp[msk] + cost[i][j]
                    if v < nxt[msk | 1 << j]:
                        nxt[msk | 1 << j] = v; nch[msk | 1 << j] = (msk, j)
            dp, ch = nxt, nch
            order.append(ch)
        best, bm = INF, None
        for msk in range(1 << m):
            if bin(msk).count("1") == n and dp[msk] < best:
                best, bm = dp[msk], msk
        res = [-1] * n
        for i in range(n - 1, -1, -1):
            prev, j = order[i][bm]
            res[i] = j; bm = prev
        return res

    was = aft = 0
    for (sh, ti), rs in by_track.items():
        names = sorted({k for r in rs for k in r["vot_gt"]})
        if not names:
            continue
        gcol = {g: vdom(rs[0]["vot_gt"].get(g, {}))[0] for g in names}
        cost = []
        for r in rs:
            c = vdom(r["vot_w"])[0]
            cost.append([(0.0 if (c is not None and c == gcol[g]) else 1.0)
                         + (0.0 if g == r["name"] else 0.01) for g in names])
        res = assign(cost, len(rs), len(names))
        for r, j in zip(rs, res):
            was += bool(r["hon_named"])
            aft += bool(r["hon_free"] and j >= 0 and names[j] == r["true"])
    print(f"\n{'='*78}\n** 4. ПЕРЕНАЗНАЧЕНИЕ ИМЁН ВНУТРИ ТРЕКА ПО ЦВЕТУ (механизм, не агрегат)\n{'='*78}")
    print(f"   честных С ИМЕНЕМ было:   {was}")
    print(f"   стало после цвета:       {aft}   ({aft-was:+d})")
    print(f"⚠ Цвет здесь решает ЕДИНОЛИЧНО, ничьи — в пользу нынешнего имени ⇒ это оценка СВЕРХУ "
          f"для слагаемого стоимости в `rowdec.trace_auto`.")
    sys.exit()

# ═══ ЗАМЕР ════════════════════════════════════════════════════════════════════════════════════
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from dataset_build import find_image
from auto import meta as M, imaging as im
from auto.config import Config

P = Config().cv

SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC.setdefault(q.name, q)
print(f"разметок найдено: {len(SRC)}")

TRACK, seen = {}, set()
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        if f.stem in seen:
            continue
        seen.add(f.stem)
        d = pickle.load(open(f, "rb"))
        TRACK[d["name"]] = {s["name"]: s["track"] for s in d["slots"]}
print(f"треков загружено для {len(TRACK)} листов")

BYDIR = {f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}": nm
         for nm, q in SRC.items()}
MD = Path(a.dir) / a.mode
dirs = sorted((d for d in MD.iterdir() if d.is_dir()), key=lambda p: p.name) if MD.is_dir() else []
print(f"каталогов выдачи в {MD}: {len(dirs)}")
if SH_N > 1:
    dirs = dirs[len(dirs) * SH_I // SH_N: len(dirs) * (SH_I + 1) // SH_N]
    print(f"* ШАРД {SH_I}/{SH_N}: {len(dirs)} листов")


def masks(rgb):
    """Маски цветов ПРОД-КОДОМ: те же, по которым прод ведёт линию (`trace2d._color_fg`)."""
    cm = im.color_channels(rgb, P)
    st = im.structure_mask(rgb, P)
    out = {c: cm[c] & ~st for c in ("red", "orange", "green", "blue")}
    blk = im.dark_mask(rgb, P) & ~st
    for c in out.values():
        blk = blk & ~c
    out["black"] = blk
    return out


def under(tr, mk, H, W):
    """→ ({цвет: пикселей}, {цвет: СТРОК}) под траекторией (каждая --step строка, окно ±--halfw)."""
    cnt, vote = Counter(), Counter()
    for y in sorted(tr)[::a.step]:
        if not (0 <= y < H):
            continue
        x = int(round(tr[y]))
        x0, x1 = max(0, x - a.halfw), min(W, x + a.halfw + 1)
        if x1 <= x0:
            continue
        best, bn = None, 0
        for c, m in mk.items():
            v = int(m[y, x0:x1].sum())
            if v:
                cnt[c] += v
                if v > bn:
                    best, bn = c, v
        if best:
            vote[best] += 1
    return dict(cnt), dict(vote)


rows, done, skip = [], 0, Counter()
for d in dirs:
    nm = BYDIR.get(d.name)
    if nm is None:
        skip["каталог не опознан"] += 1; continue
    got = next(iter(sorted(d.glob("*_auto.nlgx"))), None)
    if got is None:
        skip["нет выдачи"] += 1; continue
    src = SRC[nm]
    img = find_image(src)
    if not img:
        skip["нет картинки"] += 1; continue
    gts = {c["name"]: dense(c) for c in extract(str(src))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    wr = {c["name"]: dense(c) for c in extract(str(got))["curves"]
          if M.mnem_root(c["name"]) != "DA"}
    wr = {k: v for k, v in wr.items() if len(v) >= 30}
    if not gts or not wr:
        skip["пусто"] += 1; continue
    tmap = TRACK.get(nm, {})
    try:
        rgb = im.load_rgb(str(img))
    except Exception:
        skip["скан не читается"] += 1; continue
    H, W = rgb.shape[:2]
    mk = masks(rgb)
    _g = {k: under(g, mk, H, W) for k, g in gts.items()}
    colg = {k: v[0] for k, v in _g.items()}
    votg = {k: v[1] for k, v in _g.items()}
    for k, t in wr.items():
        ti = tmap.get(k)
        pool = list(gts) if ti is None else [g for g in gts if tmap.get(g) == ti]
        if not pool:
            pool = list(gts)
        best, bm = None, None
        for g in pool:
            m, c = err(t, gts[g])
            if m is None:
                continue
            if bm is None or m < bm:
                best, bm = (g if HON(m, c) else None), m
        hn = k in gts and HON(*err(t, gts[k]))
        cw, vw = under(t, mk, H, W)
        rows.append(dict(sheet=nm, name=k, track=ti, hon_named=bool(hn),
                         hon_free=best is not None, true=best,
                         col_w=cw, vot_w=vw,
                         col_gt={g: colg[g] for g in pool},
                         vot_gt={g: votg[g] for g in pool}))
    done += 1
    del rgb, mk
    if done % 10 == 0:
        print(f"  {done}/{len(dirs)} листов, кривых {len(rows)}")

tot = done + sum(skip.values())
print(f"\n* СВЕРКА: разобрано {done} + пропущено {sum(skip.values())} = {tot} "
      f"против {len(dirs)} каталогов   {'* СОШЛОСЬ' if tot == len(dirs) else 'НЕ СОШЛОСЬ'}")
for k, v in skip.items():
    print(f"    пропущено «{k}»: {v}")
Path(a.out).mkdir(parents=True, exist_ok=True)
outf = Path(a.out) / f"rdcolor_{SH_I}of{SH_N}.pkl"
pickle.dump({"rows": rows, "sheets": done, "shard": SH_I, "n_shards": SH_N,
             "mode": a.mode, "dir": a.dir}, open(outf, "wb"))
print(f"дамп: {outf}  ({len(rows)} кривых)")

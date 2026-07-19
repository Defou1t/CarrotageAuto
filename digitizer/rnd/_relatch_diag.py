r"""_relatch_diag.py — МЕХАНИКА трассировщика: есть ли вообще запас у «умного перезахватывания»?

Контекст. §6.19 закрыл правку «запретить прыжок на соседа» (cov рушится до 0-3%) и записал
durable-вывод: «ветка ближайший-ран — ОСНОВНАЯ рабочая, ран РЕДКО перекрывает предсказание».
Следующий шаг роадмапа — «прыгать только на ран, согласованный с направлением/шириной».

Но прежде чем перебирать стратегии выбора, надо ответить на ОДИН вопрос, который решает, есть ли
у них потолок вообще:

  ★ Когда трассировщик прыгает — БЫЛ ЛИ на этой строке ран СВОЕЙ кривой?

  • если своего рана на строке НЕТ → никакой чооser не поможет, узкое место в маске (recall),
    и «умное перезахватывание» упирается в потолок ещё до старта;
  • если свой ран ЕСТЬ, а мы берём чужой → запас реален, и он ровно такого размера,
    сколько таких строк.

Плюс проверяется гипотеза о ПРИРОДЕ КОАСТА (важно для трактовки §6.19): в отвергнутой правке
коаст шёл с НЕИЗМЕННОЙ скоростью и без перезахвата, т.е. x убегал по инерции и уже не мог
вернуться. Тогда обвал cov до 1-3% — артефакт реализации коаста, а НЕ доказательство того, что
ран редко перекрывает предсказание. Меряем долю ветки `cont` прямо.

  python _relatch_diag.py [--tol 3]
"""
import sys, json, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
import _relatch_bench as BE

ap = argparse.ArgumentParser()
ap.add_argument("--tol", type=float, default=3.0, help="допуск «ран накрывает точку GT»")
a = ap.parse_args()

rows_out = []
TOT = {"cont": 0, "else": 0, "empty": 0, "init": 0,
       "own_run_есть": 0, "строк_с_GT": 0,
       "else_свой_ран_есть": 0, "else_взяли_свой": 0, "else_взяли_чужой": 0,
       "cont_свой_ран_есть": 0, "cont_взяли_свой": 0,
       "уходов": 0, "уходов_свой_был": 0, "на_своей": 0}
jumps = []
own_gaps = []

for sheet in BE.load():
    H = sheet["H"]
    print(f"\n### {sheet['well']}  цвета {sheet['colors']}")
    for rec in sheet["curves"]:
        gys, gxs = sheet["gt"][rec["name"]]
        own = dict(zip(gys.tolist(), gxs.tolist()))

        # цвет выбираем ТОТ ЖЕ, что и стенд (лучший по med) — иначе диагностируем другую траекторию
        best_c, best_med = None, 1e9
        for c in sheet["colors"]:
            tr = BE.trace(rec, rec["runs"][c], H)
            common = [y for y in tr if y in own]
            if len(common) < 30:
                continue
            med = float(np.median([abs(tr[y] - own[y]) for y in common]))
            if med < best_med:
                best_med, best_c = med, c
        if best_c is None:
            continue

        st = {"cont": 0, "else": 0, "empty": 0, "init": 0, "own_есть": 0, "с_GT": 0,
              "else_свой_есть": 0, "else_свой_взят": 0, "cont_свой_есть": 0, "cont_свой_взят": 0,
              "уходов": 0, "уходов_свой_был": 0, "на_своей": 0}
        loc_jumps = []
        prev = {"own": False}

        def probe(y, A, B, Cc, pred, x, v, branch, k, st=st, own=own, loc_jumps=loc_jumps,
                  prev=prev):
            st[branch] = st.get(branch, 0) + 1
            if branch in ("empty", "init", "coast") or y not in own:
                return
            st["с_GT"] += 1
            ox = own[y]
            # индекс рана, накрывающего точку эксперта (свой ран)
            hit = np.nonzero((A - a.tol <= ox) & (ox <= B + a.tol))[0]
            oi = int(hit[np.argmin(np.abs(Cc[hit] - ox))]) if len(hit) else None
            is_own = oi is not None and k == oi
            if oi is not None:
                st["own_есть"] += 1
                if branch == "else":
                    st["else_свой_есть"] += 1
                    loc_jumps.append(abs(float(Cc[k]) - pred))
                    if is_own:
                        st["else_свой_взят"] += 1
                else:
                    st["cont_свой_есть"] += 1
                    if is_own:
                        st["cont_свой_взят"] += 1
            if is_own:
                st["на_своей"] += 1
            # ★ МОМЕНТ УХОДА: шли по своей кривой, а на этой строке взяли ЧУЖОЙ ран.
            # Только здесь вопрос «был ли свой ран доступен» означает «можно ли было НЕ уйти».
            # На строках, где мы УЖЕ латчнуты, свой ран «есть» почти всегда — но это следствие,
            # а не запас: вернуться туда стоит такого же прыжка.
            if prev["own"] and not is_own:
                st["уходов"] += 1
                if oi is not None:
                    st["уходов_свой_был"] += 1
            prev["own"] = is_own

        BE.trace(rec, rec["runs"][best_c], H, probe=probe)
        n = st["cont"] + st["else"] + st["empty"] + st["init"]
        if not st["с_GT"]:
            continue
        p_cont = 100.0 * st["cont"] / max(1, st["cont"] + st["else"])
        p_own = 100.0 * st["own_есть"] / st["с_GT"]
        p_pick = 100.0 * st["else_свой_взят"] / max(1, st["else_свой_есть"])
        rows_out.append({"well": sheet["well"], "curve": rec["short"],
                         "band": rec["hi"] - rec["lo"], "med": round(best_med, 1),
                         "cont%": round(p_cont, 1), "свой_ран_есть%": round(p_own, 1),
                         "else_свой_взят%": round(p_pick, 1),
                         "else_строк": st["else"], "else_свой_есть": st["else_свой_есть"]})
        for k2, kk in (("cont", "cont"), ("else", "else"), ("empty", "empty"), ("init", "init")):
            TOT[kk] += st[k2]
        TOT["строк_с_GT"] += st["с_GT"]; TOT["own_run_есть"] += st["own_есть"]
        TOT["else_свой_ран_есть"] += st["else_свой_есть"]
        TOT["else_взяли_свой"] += st["else_свой_взят"]
        TOT["else_взяли_чужой"] += st["else_свой_есть"] - st["else_свой_взят"]
        TOT["cont_свой_ран_есть"] += st["cont_свой_есть"]
        TOT["cont_взяли_свой"] += st["cont_свой_взят"]
        TOT["уходов"] += st["уходов"]; TOT["уходов_свой_был"] += st["уходов_свой_был"]
        TOT["на_своей"] += st["на_своей"]
        jumps += loc_jumps
        print(f"   {rec['short']:<8} med={best_med:7.1f} полоса{rec['hi']-rec['lo']:>5}  "
              f"cont {p_cont:5.1f}%  свой ран есть {p_own:5.1f}%  "
              f"из них в else взяли свой {p_pick:5.1f}%")

print(f"\n{'='*78}\n=== ИТОГО по {len(rows_out)} кривым ===")
tot_rows = TOT["cont"] + TOT["else"]
print(f"  ветка cont (ран перекрывает предсказание) : {100.0*TOT['cont']/max(1,tot_rows):5.1f}%"
      f"   ({TOT['cont']} строк)")
print(f"  ветка else (прыжок на ближайший)          : {100.0*TOT['else']/max(1,tot_rows):5.1f}%"
      f"   ({TOT['else']} строк)")
print(f"  строк, где под точкой эксперта ЕСТЬ ран   : "
      f"{100.0*TOT['own_run_есть']/max(1,TOT['строк_с_GT']):5.1f}%   "
      f"({TOT['own_run_есть']} из {TOT['строк_с_GT']})")
print(f"\n  ★ ЗАПАС «умного перезахватывания» — строки else, где СВОЙ ран БЫЛ:")
print(f"     таких строк                : {TOT['else_свой_ран_есть']}")
print(f"     из них взяли СВОЙ ран      : {TOT['else_взяли_свой']:>9}  "
      f"({100.0*TOT['else_взяли_свой']/max(1,TOT['else_свой_ран_есть']):.1f}%)")
print(f"     из них взяли ЧУЖОЙ (запас) : {TOT['else_взяли_чужой']:>9}  "
      f"({100.0*TOT['else_взяли_чужой']/max(1,TOT['else_свой_ран_есть']):.1f}%)")
if jumps:
    j = np.array(jumps)
    print(f"\n  дистанция прыжка в else (там, где свой ран был): "
          f"med {np.median(j):.1f}px  p90 {np.percentile(j,90):.1f}px  >30px {100.0*(j>30).mean():.1f}%")

print(f"\n  ★★ МОМЕНТ УХОДА СО СВОЕЙ КРИВОЙ (единственный честный замер запаса):")
print(f"     строк, где шли по СВОЕЙ                : {TOT['на_своей']}")
print(f"     уходов со своей на чужую               : {TOT['уходов']}")
print(f"     из них свой ран БЫЛ на этой строке     : {TOT['уходов_свой_был']:>8}  "
      f"({100.0*TOT['уходов_свой_был']/max(1,TOT['уходов']):.1f}%)  ← ИЗБЕЖНЫЕ уходы")
print(f"     из них своего рана НЕ БЫЛО (обрыв туши): "
      f"{TOT['уходов']-TOT['уходов_свой_был']:>8}  "
      f"({100.0*(TOT['уходов']-TOT['уходов_свой_был'])/max(1,TOT['уходов']):.1f}%)  ← нужен КОАСТ, а не выбор")

print(f"\nЧитать так:")
print(f"  • много ИЗБЕЖНЫХ уходов  -> выигрывает УМНЫЙ ВЫБОР рана (ширина/направление);")
print(f"  • много уходов БЕЗ своего рана -> выбирать не из чего, нужен КОАСТ через обрыв")
print(f"    с перезахватом (и тогда обвал cov в §6.19 — артефакт коаста без затухания, а не закон).")

OUT = Path(r"F:\nds\output\taskS")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "relatch_diag.json").write_text(
    json.dumps({"per_curve": rows_out, "total": TOT,
                "jump_med": float(np.median(jumps)) if jumps else None},
               ensure_ascii=False, indent=1), encoding="utf-8")

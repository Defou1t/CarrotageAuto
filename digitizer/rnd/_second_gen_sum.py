r"""_second_gen_sum.py — сводка `_second_gen.py`: что даёт второй генератор трасс в пуле (§6.101).

⚠⚠ ЗДЕСЬ СКЛЕИВАЮТСЯ ДВА РАЗБИЕНИЯ, И ЭТО СДЕЛАНО НАМЕРЕННО. Шард `3/4` (последняя четверть списка —
YULIIV и соседи, до 35 линий на лист) шёл ~100 с на лист и за полтора часа не дошёл до первой
контрольной точки. Его блок пересчитан восемью подшардами `24..31 / 32`. Блочное шардирование
пропорционально, поэтому куски совпадают ТОЧНО: `702*3//4 == 702*24//32 == 526`, `702*32//32 == 702`.
Перекрытий нет, дыр нет — и это проверяется здесь заново, по именам листов, а не на слово.
⚠ Урок §6.102 (дамп дымового прогона, попавший в сводку) применён: листы дедуплицируются ПО ИМЕНИ,
поэтому случайное пересечение разбиений не может посчитаться дважды — оно будет НАЗВАНО.

ЧТО ЗНАЧАТ КОЛОНКИ:
  потолок     — лучшая трасса пула на каждую кривую, без ограничений (что вообще есть в пуле);
  оптимум 1:1 — лучшее взаимно-однозначное назначение (что достижимо при контракте «одна трасса —
                один слот»). ⚠ Память проекта: гибрид ОБЯЗАН уважать 1:1, иначе рост потолка
                ничего не стоит.

  <ComfyUI>\python_embeded\python.exe _second_gen_sum.py
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:\nds\output\taskS\second_gen")
a = ap.parse_args()

seen, rows, pad, dn, tl, files = {}, [], None, 0, 0, []
dup = 0
for f in sorted(Path(a.dir).glob("rows_*of*.pkl")):
    z = pickle.load(open(f, "rb"))
    pad = pad or z["pad"]
    if z["pad"] != pad:
        print(f"⚠ {f.name}: другой набор окон {z['pad']} — пропущен"); continue
    files.append(f.name); dn += z.get("done", 0); tl += z.get("total", 0)
    for r in z["rows"]:
        if r["sheet"] in seen:
            dup += 1; continue
        seen[r["sheet"]] = r; rows.append(r)
if not rows:
    sys.exit("нет данных")
print(f"дампов {len(files)}: {', '.join(sorted(files))}")
if dup:
    print(f"⚠⚠ ПЕРЕСЕЧЕНИЕ РАЗБИЕНИЙ: {dup} листов встретились дважды и учтены ОДИН раз")
print(f"листов уникальных {len(rows)}, обработано шардами {dn} (из {tl} назначенных)")

t = lambda k: sum(r.get(k, 0) for r in rows)
nc, pc, po = t("curves"), t("prod_cap"), t("prod_opt")
print(f"\nкривых {nc}, трасс прода в пулах {t('nprod')}")
print(f"{'вариант':<26}{'потолок':>9}{'Δ':>6}{'оптимум 1:1':>13}{'Δ':>6}{'трасс':>8}")
print(f"{'прод (как есть)':<26}{pc:>9}{'':>6}{po:>13}{'':>6}{t('nprod'):>8}")
for p in pad:
    print(f"{'моё ведение, pad=' + str(p):<26}{t(f'mine{p}_cap'):>9}"
          f"{t(f'mine{p}_cap')-pc:>+6}{t(f'mine{p}_opt'):>13}{t(f'mine{p}_opt')-po:>+6}"
          f"{t(f'n{p}'):>8}")
for p in pad:
    print(f"{'★ ОБЪЕДИНЕНИЕ, pad=' + str(p):<26}{t(f'uni{p}_cap'):>9}"
          f"{t(f'uni{p}_cap')-pc:>+6}{t(f'uni{p}_opt'):>13}{t(f'uni{p}_opt')-po:>+6}"
          f"{t('nprod')+t(f'n{p}'):>8}")
print(f"\n⚠ «оптимум 1:1» — потолок ОТБОРА, а не выдача. Реальная раскладка ниже его "
      f"(§6.90: цена отбора K из N); цитировать как задел, а не как точность.")

best = max(pad, key=lambda p: t(f"uni{p}_opt"))
up = sum(1 for r in rows if r.get(f"uni{best}_opt", 0) > r.get("prod_opt", 0))
dnn = sum(1 for r in rows if r.get(f"uni{best}_opt", 0) < r.get("prod_opt", 0))
print(f"★ лучшее окно pad={best}: листов вверх {up}, вниз {dnn} из {len(rows)}")

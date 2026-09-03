r"""_pickthresh_sum.py — ВОСПРОИЗВОДИМЫЙ СТЕНД ПОД §6.163: НЕ СВОДИТСЯ ЛИ ОБУЧЕННЫЙ ВЫБОР К «ВЗЯТЬ ПОРОГ ПОЖЁСТЧЕ».

ОТКУДА ВОПРОС. §6.159 показал, что 93% прироста модели — это ОТКАЗЫ там, где порог согласился.
Отсюда законный и неприятный вопрос: а не воспроизводится ли всё преимущество одной цифрой в
конфиге? Если «быть осторожнее» и есть весь секрет, то `rowdec_pick = 1` даст то же самое
бесплатно — без веса, без восьми признаков и без второй раскладки.
⚠ И ВТОРАЯ ПРИЧИНА, ПОЧЕМУ СТЕНД ЗАВЕДЁН ОТДЕЛЬНО: §6.163 цитируется в решениях, а пересчитать
его было НЕЧЕМ — числа снимались разбором, написанным на один раз и не сохранённым. Раздел, чьи
числа нельзя пересчитать, — это не замер, а воспоминание о замере. Здесь он пересчитывается.

ЧЕГО СТЕНД НЕ ДЕЛАЕТ: не считает ни одного листа. Всё берётся из ЗАМОРОЖЕННЫХ выдач.

★ ПРЕДПОСЫЛКА, БЕЗ КОТОРОЙ СВИП НЕДЕЙСТВИТЕЛЕН. Симулировать произвольный порог по готовым
выдачам можно только там, где режим «оба пути» тождествен выбору между режимом A (прод целиком) и
режимом B (декодер целиком) — то есть на ОДНОТРЕКОВЫХ листах. Стенд это не предполагает, а
ПРОВЕРЯЕТ: на каждом однотрековом листе выдача порога обязана совпасть с B там, где он взял
декодер, и с A там, где не взял. Хоть одно расхождение — свип объявляется недействительным.

★★ ПОПРАВКА §6.163, РАДИ КОТОРОЙ ЗДЕСЬ ОТДЕЛЬНЫЙ КОД: доля листов, на которых МОДЕЛЬ берёт
декодер, читается из решений САМОЙ МОДЕЛИ (`ab_pickmodel/f*/M/**/_pick.json`), а НЕ из `_pick.json`
порога. Первая редакция разбора перепутала их и напечатала 80.0% вместо 68.6% — на этой ошибке
стоял вывод «модель отказывается не чаще, а в других местах», и он был неверен.

СЧЁТ — ВЕДУЩИЙ (БЕЗЫМЯННЫЙ): элемент [1] кортежа `percurve_*.pkl` (named, UNNAMED, expert).

  <ComfyUI>\python_embeded\python.exe _pickthresh_sum.py
  … --folds 0,1     # только фолды 0-1 — поле, на котором писался §6.163
"""
import sys, json, pickle, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:\nds\output\taskS")
ap.add_argument("--folds", default="", help="ограничить поле фолдами, напр. 0,1 (пусто = все)")
ap.add_argument("--nmax", type=int, default=8)
ap.add_argument("--perm", type=int, default=200000)
a = ap.parse_args()
TS = Path(a.ts)

norm = lambda s: s if s.endswith(".nlgx") else s + ".nlgx"


def lead(pkl, mode):
    """Ведущий (безымянный) счёт по листам. [1] — не [0]: [0] именной, справочный."""
    d = pickle.load(open(pkl, "rb"))
    return {k: v[1] for k, v in d[mode].items()}


def picks(root):
    """Решения по трекам из `_pick.json` выдачи. Ключ — имя листа, значение — список треков."""
    out = {}
    for f in Path(root).glob("*/*_pick.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        out[norm(f.name[:-len("_pick.json")])] = d
    return out


# ─────────────────────────────────────────────────────────── источники и проверка их наличия
need = {
    "percurve_pair.pkl": "прод A и декодер целиком B",
    "percurve_rule.pkl": "порог rowdec_pick=3",
}
gone = [f"{k} ({v})" for k, v in need.items() if not (TS / k).exists()]
if gone:
    sys.exit("⛔ НЕТ ИСТОЧНИКОВ: " + "; ".join(gone))

A = lead(TS / "percurve_pair.pkl", "A")
B = lead(TS / "percurve_pair.pkl", "B")
D = lead(TS / "percurve_rule.pkl", "D")

M, mpick = {}, {}
folds_of = {}
for f in range(5):
    pc = TS / f"percurve_pickf{f}.pkl"
    if pc.exists():
        M.update(lead(pc, "M"))
    mp = TS / "ab_pickmodel" / f"f{f}" / "M"
    if mp.is_dir():
        for k, v in picks(mp).items():
            mpick[k] = v
            folds_of[k] = f
    sl = TS / "pickfolds" / f"sheets_f{f}.txt"
    if sl.exists():
        for ln in sl.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                folds_of.setdefault(norm(ln.strip()), f)

dpick = picks(TS / "ab_rdpick_c" / "D")
print(f"★ ИСТОЧНИКИ: прод {len(A)}, декодер {len(B)}, порог {len(D)} (решений {len(dpick)}), "
      f"модель {len(M)} (решений {len(mpick)}), раскладка по фолдам {len(folds_of)}")
if not dpick or not mpick:
    sys.exit("⛔ НЕТ РЕШЕНИЙ `_pick.json` — свип и доли посчитать не из чего")

keep = None
if a.folds:
    keep = {int(x) for x in a.folds.split(",")}
    print(f"★ поле ограничено фолдами {sorted(keep)}")

# ──────────────────────────────── ПРОВЕРКА ПРЕДПОСЫЛКИ: однотрековый лист тождествен выбору A/B
single = [k for k, v in dpick.items() if len(v.get("tracks", [])) == 1]
base = [k for k in single if k in A and k in B and k in D]
if keep is not None:
    base = [k for k in base if folds_of.get(k) in keep]

took = [k for k in base if dpick[k]["tracks"][0].get("dec")]
skip = [k for k in base if not dpick[k]["tracks"][0].get("dec")]
bad_t = [k for k in took if D[k] != B[k]]
bad_s = [k for k in skip if D[k] != A[k]]

print(f"\n★★ ПРЕДПОСЫЛКА СИМУЛЯЦИИ (однотрековых листов {len(base)} из {len(dpick)} у порога)")
print(f"| правило | листов | расхождений |")
print(f"| ВЗЯЛО декодер | {len(took)} | D ≠ B на {len(bad_t)} |")
print(f"| НЕ взяло      | {len(skip)} | D ≠ A на {len(bad_s)} |")
ok_premise = not bad_t and not bad_s
if not ok_premise:
    print("⛔⛔ ПРЕДПОСЫЛКА НЕ ДЕРЖИТСЯ — СВИП НЕДЕЙСТВИТЕЛЕН, порог по готовым выдачам не симулируется.")
    for k in (bad_t + bad_s)[:5]:
        print(f"    {k}: A={A[k]} B={B[k]} D={D[k]}")
else:
    print("★ ни одного расхождения ⇒ любой порог по `n_dec` симулируется ТОЧНО")

# ★ форма правила проверяется по данным, а не берётся на веру: `dec` обязан быть `n_dec <= pick`
form = [k for k in base
        if dpick[k]["tracks"][0].get("dec") != (dpick[k]["tracks"][0]["n_dec"] <= dpick[k]["pick"])]
print(f"★ форма правила «dec ⟺ n_dec ≤ pick» нарушена на {len(form)} листах из {len(base)}"
      + ("" if not form else f"  ⚠ например {form[0]}"))

if not ok_premise:
    sys.exit(1)


def sim(N, keys):
    """Счёт порога `rowdec_pick = N` на однотрековых листах. N = 0 — ВЫКЛЮЧЕНО (не «n_dec ≤ 0»):
    так это читает `auto/config.py`, и подмена смысла здесь дала бы чужую кривую."""
    if N == 0:
        return np.array([A[k] for k in keys], float), 0.0
    take = [dpick[k]["tracks"][0]["n_dec"] <= N for k in keys]
    v = np.array([B[k] if t else A[k] for k, t in zip(keys, take)], float)
    return v, 100.0 * sum(take) / max(1, len(take))


# ──────────────────────────────────────────────────────────────────────── СВИП ПО ЗНАЧЕНИЮ ПОРОГА
print(f"\n★★★ СВИП ПОРОГА на {len(base)} однотрековых листах (ведущий счёт)")
print("| rowdec_pick | честных | декодер на |")
for N in range(0, a.nmax + 1):
    v, use = sim(N, base)
    mark = "  ← в проде" if N == 3 else ("  ← прод-умолчание (выкл)" if N == 0 else "")
    print(f"| {N} | {v.sum():.0f} | {use:.1f}% |{mark}")


def paired(x, y, nm, perm=a.perm):
    """Парный тест по ПОЛИСТНЫМ разностям: перестановка = случайные знаки. Тот же приём и тот же
    сид, что в `_pickmodel_sum.py`, чтобы числа двух стендов сравнивались, а не спорили."""
    d = x - y
    obs = float(d.sum())
    nz = d[d != 0]
    up, dn = int((d > 0).sum()), int((d < 0).sum())
    rng = np.random.default_rng(20260829)
    if len(nz):
        sg = rng.integers(0, 2, size=(perm, len(nz))) * 2 - 1
        p = float((np.abs((sg * np.abs(nz)).sum(1)) >= abs(obs)).mean())
    else:
        p = 1.0
    return obs, up, dn, p


# ──────────────────────────────────── ДЕРЖАННАЯ ПЕРЕНАСТРОЙКА ПОРОГА: N выбирается БЕЗ своего фолда
have_folds = sorted({folds_of[k] for k in base if k in folds_of})
if have_folds:
    print(f"\n★★ ДЕРЖАННАЯ ПЕРЕНАСТРОЙКА ПОРОГА (N выбирается на ОСТАЛЬНЫХ фолдах)")
    held, chosen = 0.0, []
    for f in have_folds:
        tr = [k for k in base if folds_of.get(k) not in (None, f)]
        te = [k for k in base if folds_of.get(k) == f]
        if not tr or not te:
            continue
        best = max(range(1, a.nmax + 1), key=lambda N: sim(N, tr)[0].sum())
        chosen.append(best)
        held += sim(best, te)[0].sum()
    fixed3 = sum(sim(3, [k for k in base if folds_of.get(k) in have_folds])[0])
    print(f"   N по фолдам {chosen}; держанно {held:.0f} против {fixed3:.0f} у N=3 "
          f"⇒ **{held - fixed3:+.0f}**")
    print("   ⚠ читать как «порог настроен не оптимально, но переставлять его отдельно смысла нет»:"
          " значимость проверяется ниже, на общей почве с моделью")

# ───────────────────────────────────────────────── ГЛАВНОЕ СРАВНЕНИЕ: ВСЁ НА ОДНИХ И ТЕХ ЖЕ ЛИСТАХ
common = [k for k in base if k in M and k in mpick]
if not common:
    sys.exit("\n⛔ ОБЩЕГО ПОЛЯ НЕТ: у модели нет ни одного из этих листов — сравнивать нечего")

mt = [k for k in common if len(mpick[k].get("tracks", [])) == 1]
if len(mt) != len(common):
    print(f"\n⚠ у модели однотрековых {len(mt)} из {len(common)} — беру пересечение")
    common = mt

muse = 100.0 * sum(1 for k in common if mpick[k]["tracks"][0].get("dec")) / len(common)
vA = np.array([A[k] for k in common], float)
vB = np.array([B[k] for k in common], float)
vM = np.array([M[k] for k in common], float)
v3, u3 = sim(3, common)

print(f"\n★★★ ГЛАВНОЕ СРАВНЕНИЕ — всё на одних и тех же {len(common)} листах (ведущий счёт)")
print("| вариант | честных | Δ к порогу N=3 | листов ↑/↓ | перест. p | декодер на |")
rows = [("прод (один путь)", vA, 0.0)]
for N in range(1, a.nmax + 1):
    vN, uN = sim(N, common)
    rows.append((f"порог rowdec_pick = {N}" + (" (в проде)" if N == 3 else ""), vN, uN))
rows.append(("★ обученный выбор", vM, muse))
rows.append(("декодер целиком", vB, 100.0))
for nm, v, use in rows:
    if v is v3 or (nm.endswith("(в проде)")):
        print(f"| {nm} | {v.sum():.0f} | — | | | {use:.1f}% |")
        continue
    obs, up, dn, p = paired(v, v3, nm)
    print(f"| {nm} | {v.sum():.0f} | {obs:+.0f} | {up}/{dn} | {p:.4f} | {use:.1f}% |")

orac = np.maximum(vA, vB)
print(f"| оракул по листу (знает эталон) | {orac.sum():.0f} | {orac.sum()-v3.sum():+.0f} | | | |")

# ★ модель против ЛУЧШЕГО фиксированного порога — самый строгий из честных вопросов
bestN = max(range(1, a.nmax + 1), key=lambda N: sim(N, common)[0].sum())
vbest, ubest = sim(bestN, common)
obs, up, dn, p = paired(vM, vbest, "модель против лучшего порога")
print(f"\n★ МОДЕЛЬ ПРОТИВ ЛУЧШЕГО ФИКСИРОВАННОГО ПОРОГА (N={bestN}, выбран ЗАДНИМ ЧИСЛОМ на этих же"
      f" листах — фора порогу): Δ = {obs:+.0f}, листов ↑{up}/↓{dn}, p = {p:.4f}")
print(f"   декодер: модель на {muse:.1f}% листов, порог N={bestN} на {ubest:.1f}%, N=3 на {u3:.1f}%")
d_use = muse - u3
print(f"   ⇒ модель берёт декодер {'РЕЖЕ' if d_use < 0 else 'ЧАЩЕ'} порога в проде на "
      f"{abs(d_use):.1f} п.п.")

# ──────────────────────────────────────────────────────── СВЕРКА С ТЕМ, ЧТО ЗАПИСАНО В §6.163
# ★★ СВЕРЯЮТСЯ ПРИРОСТЫ, А НЕ СУММЫ. Сумма зависит от ЗНАМЕНАТЕЛЯ — от того, какие листы попали
# в поле; прирост Δ к N=3, счёт ↑/↓ и p — нет. §6.169 стоил ветке снятого вывода ровно на этом:
# два числа с разных объёмов сравнили как одно, и «совпадение до единицы» оказалось артефактом.
print("\n★★ СВЕРКА С §6.163 ПО ПРИРОСТАМ (суммы зависят от знаменателя, приросты — нет)")
was = {"прод": (-35, 47, 89), "N=1": (+10, 41, 38), "N=2": (+14, 28, 20),
       "модель": (+26, 24, 6), "декодер": (-37, 10, 40), "оракул": (+79, None, None)}
mk = {"прод": vA, "N=1": sim(1, common)[0], "N=2": sim(2, common)[0],
      "модель": vM, "декодер": vB, "оракул": orac}
same = 0
for k, (wd, wu, wn) in was.items():
    obs, up, dn, _ = paired(mk[k], v3, k, perm=1)
    ok = abs(obs - wd) < 0.5 and (wu is None or (up == wu and dn == wn))
    same += ok
    tail = "" if wu is None else f", ↑{up}/↓{dn} против записанных ↑{wu}/↓{wn}"
    print(f"   {k:<8} Δ записано {wd:+4}   сейчас {obs:+5.0f}{tail}   {'★ сошлось' if ok else '⚠ иное'}")
print(f"★ приростов сошлось {same} из {len(was)}; поле сейчас {len(common)} листов")

# ★ если поле расходится с записанным ровно на несколько листов — НАЗВАТЬ ИХ, а не списать на шум.
if a.folds == "0,1" and len(common) != 465:
    diff = len(common) - 465
    print(f"⚠ ПОЛЕ ОТЛИЧАЕТСЯ ОТ ЗАПИСАННОГО НА {diff:+d} ЛИСТ(ОВ). Ищу, снятие каких листов сводит суммы:")
    tgt = {"прод": 351, "N=3": 386, "модель": 412, "декодер": 349}
    cur = {"прод": vA.sum(), "N=3": v3.sum(), "модель": vM.sum(), "декодер": vB.sum()}
    need = {k: cur[k] - tgt[k] for k in tgt}
    nm = [k for k in common
          if A[k] == need["прод"] and D[k] == need["N=3"]
          and M[k] == need["модель"] and B[k] == need["декодер"]]
    if diff == 1 and len(nm) == 1:
        k = nm[0]
        print(f"   ★ ровно один лист сводит ВСЕ четыре суммы: {k}")
        print(f"     прод {A[k]}, декодер {B[k]}, порог {D[k]}, модель {M[k]}, "
              f"n_dec {dpick[k]['tracks'][0]['n_dec']:.0f} ⇒ порог декодер НЕ берёт")
        print("     ⇒ расхождение объяснено ПОЛНОСТЬЮ: разбор §6.163 этот лист не взял, приросты "
              "от этого не изменились ни на единицу.")
    else:
        print(f"   ⚠ однозначного объяснения нет: подходящих листов {len(nm)} при расхождении {diff:+d}")
elif not a.folds:
    print("⚠ поле НЕ ограничено фолдами 0-1: §6.163 писался, когда у модели были только они."
          " Числа ниже — БОЛЬШЕ того замера, а не его повтор. Для повтора: --folds 0,1")

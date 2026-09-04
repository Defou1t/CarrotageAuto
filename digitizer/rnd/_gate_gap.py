r"""_gate_gap.py — ГДЕ ЛЕЖАТ НЕДОБРАННЫЕ КРИВЫЕ: РАЗЛОЖЕНИЕ ЗАЗОРА ДО ОРАКУЛА (§6.179).

ОТКУДА ВОПРОС. §6.178 закрыл выбор конфигурации: 965 → 1109 при декодере на 64.4% листов. Но
оракул по листу — **1252**, то есть ~143 кривые не добраны, и направление живое. Прежде чем
искать новый признак, надо понять, ЧЕГО именно не хватает, а не гадать.

ЧТО СЧИТАЕТ (по замороженным выдачам, счёта не тратит):
  1. **Потолок САМОГО ГЕЙТА** — `sum(max(A, H))` по листам. Это лучшее, что может дать ЛЮБОЕ
     решение по листу при нынешнем втором пути. Отличать его от оракула `max(A, B)` обязательно:
     второй меряет потолок выбора между ДВУМЯ ЧИСТЫМИ путями, а гейт выбирает между продом и
     «прод + порог по треку», и это разные потолки.
  2. **Разложение потери гейта на две ПРОТИВОПОЛОЖНЫЕ ошибки:**
     * НЕ ПУСТИЛ, а второй путь помог бы (упущенное);
     * ПУСТИЛ, а второй путь навредил (напрасное).
     Лечатся они разным: первое — более мягким порогом, второе — более жёстким. Если обе велики,
     порогом их не развести вовсе, и нужен ДРУГОЙ признак.
  3. **Сосредоточен ли остаток.** Если половина зазора лежит на десятке листов — это задача про
     конкретные бланки; если он размазан по сотням — про признак.

  <ComfyUI>\python_embeded\python.exe _gate_gap.py
  … --thr 0.8
"""
import sys, argparse, json, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--prod", default=r"F:/nds/output/taskS/percurve_wellmap.pkl")
ap.add_argument("--prod-mode", default="A")
ap.add_argument("--rule", default=r"F:/nds/output/taskS/percurve_wellmap.pkl")
ap.add_argument("--rule-mode", default="H")
ap.add_argument("--dec", default=r"F:/nds/output/taskS/percurve_rdhonest.pkl")
ap.add_argument("--dec-mode", default="B")
ap.add_argument("--model", nargs="*", default=[f"F:/nds/output/taskS/percurve_hpickf{f}.pkl"
                                               for f in range(5)])
ap.add_argument("--model-mode", default="M")
ap.add_argument("--dumps", nargs="+", default=[r"F:/nds/output/taskS/ab_wellmap/H"])
ap.add_argument("--thr", type=float, default=0.80)
a = ap.parse_args()

lead = lambda f, m: {k: v[1] for k, v in pickle.load(open(f, "rb"))[m].items()}
A, H, B = lead(a.prod, a.prod_mode), lead(a.rule, a.rule_mode), lead(a.dec, a.dec_mode)
M = {}
for f in a.model:
    M.update(lead(f, a.model_mode))

src = {}
for root in a.dumps:
    for d in Path(root).iterdir():
        if not d.is_dir():
            continue
        u = next(iter(d.glob("*_understanding.json")), None)
        if not u:
            continue
        L = (json.loads(u.read_text(encoding="utf-8")).get("lines") or [])
        if not L:
            continue
        c = Counter(str(x.get("color")) for x in L)
        src[u.name[:-len("_understanding.json")] + ".nlgx"] = max(c.values()) / len(L)

K = sorted(set(A) & set(H) & set(B) & set(M) & set(src))
if len(K) < 100:
    print(f"⛔ общих листов {len(K)} — разложение бессмысленно")
    sys.exit(1)
va = np.array([A[k] for k in K], float)
vh = np.array([H[k] for k in K], float)
vb = np.array([B[k] for k in K], float)
vm = np.array([M[k] for k in K], float)
ct = np.array([src[k] for k in K])
take = ct >= a.thr
gate = np.where(take, vh, va)

print(f"★ листов {len(K)}; гейт `color_top_frac >= {a.thr}` пускает {100*take.mean():.1f}%")
print("\n★★ ПОТОЛКИ — ИХ ТРИ, И ПУТАТЬ ИХ НЕЛЬЗЯ")
print(f"| потолок | что значит | кривых |")
print(f"| гейт по листу над порогом | лучшее РЕШЕНИЕ ПО ЛИСТУ при нынешнем втором пути | "
      f"{np.maximum(va, vh).sum():.0f} |")
print(f"| выбор по листу прод/декодер | лучшее из двух ЧИСТЫХ путей | {np.maximum(va, vb).sum():.0f} |")
print(f"| выбор по листу прод/порог/декодер | всё, что достижимо решением по листу | "
      f"{np.maximum(np.maximum(va, vb), vh).sum():.0f} |")
print(f"\n★ достигнуто: прод {va.sum():.0f} · порог {vh.sum():.0f} · обученный выбор {vm.sum():.0f} "
      f"· ★ ГЕЙТ+ПОРОГ {gate.sum():.0f}")

# ── две ПРОТИВОПОЛОЖНЫЕ ошибки гейта ───────────────────────────────────────────────────────────
miss = (~take) & (vh > va)          # не пустил, а помогло бы
waste = take & (vh < va)            # пустил, а навредило
gain_miss = float((vh - va)[miss].sum())
loss_waste = float((va - vh)[waste].sum())
ceil_gate = float(np.maximum(va, vh).sum())
print(f"\n★★ РАЗЛОЖЕНИЕ ЗАЗОРА ДО ПОТОЛКА ГЕЙТА ({ceil_gate:.0f} − {gate.sum():.0f} = "
      f"{ceil_gate - gate.sum():.0f} кривых)")
print(f"| ошибка | листов | кривых | лечится |")
print(f"| НЕ ПУСТИЛ, а помогло бы | {int(miss.sum())} | −{gain_miss:.0f} | порогом МЯГЧЕ |")
print(f"| ПУСТИЛ, а навредило | {int(waste.sum())} | −{loss_waste:.0f} | порогом ЖЁСТЧЕ |")
if gain_miss and loss_waste:
    r = max(gain_miss, loss_waste) / max(1e-9, min(gain_miss, loss_waste))
    print(f"⇒ обе ошибки одного порядка (отношение {r:.1f}) ⇒ ОДНИМ ПОРОГОМ их не развести: "
          f"сдвиг лечит одну и ровно настолько же портит другую. Нужен ДРУГОЙ признак."
          if r < 2.0 else
          f"⇒ одна ошибка вдвое+ крупнее другой (отношение {r:.1f}) ⇒ сдвиг порога ещё что-то даёт.")

# ── насколько остаток сосредоточен ─────────────────────────────────────────────────────────────
resid = np.maximum(va, vh) - gate
nz = np.sort(resid[resid > 0])[::-1]
if len(nz):
    tot = nz.sum()
    for q in (10, 25, 50):
        n = max(1, int(round(len(nz) * q / 100)))
        print(f"   верхние {q}% листов с потерей ({n} из {int((resid>0).sum())}) держат "
              f"{100*nz[:n].sum()/tot:.0f}% зазора")
print(f"   потеря на лист, где она есть: медиана {np.median(nz):.1f}, максимум {nz.max():.0f}"
      if len(nz) else "   потерь нет")

# ── что дал бы ИДЕАЛЬНЫЙ выбор пути ПО ТРЕКУ вдобавок к идеальному гейту ───────────────────────
print(f"\n★ ЧТО ОСТАЁТСЯ ЗА ПРЕДЕЛАМИ РЕШЕНИЯ ПО ЛИСТУ: потолок по листу "
      f"{np.maximum(np.maximum(va, vb), vh).sum():.0f}, а потрековый оракул §6.177 — 1257. "
      f"Разница и есть то, что решением по листу не берётся ПРИНЦИПИАЛЬНО.")

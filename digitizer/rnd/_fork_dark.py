r"""_fork_dark.py — РАЗЛИЧИМЫ ЛИ НА РАЗВИЛКЕ ВЕРНЫЙ И ВЫБРАННЫЙ РАН ПО ТЕМНОТЕ И ШИРИНЕ (§6.235, 27.09).

Дамп `_seq_onpolicy.py --dark-dump` (прод-селектор ведёт сам, без сбросов): на каждом решении с верным раном среди
кандидатов — темнота кандидатов (p90 «бумага строки − V» в окне ±4 строки × ран), ширина, метка, выбор модели и
собственная темнота кривой (медиана по эталону — то, что мог бы помнить канал истории). Разрез: трасса на своей линии
(УДЕРЖАНИЕ — момент ухода) / не на своей (ВОЗВРАТ). Правила-оракулы «ближе к своей темноте» и «ближе к своей ширине»:
сколько ошибок чинят и сколько верных ломают — если оба около половины, признак не несёт сигнала сверх того, что у сети есть.

  _fork_dark.py --dump F:/nds/output/taskS/fork_dark.pkl
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dump", default=r"F:/nds/output/taskS/fork_dark.pkl")
a = ap.parse_args()
R = pickle.load(open(a.dump, "rb"))


def corr(r):
    return next(i for i, c in enumerate(r[4]) if c[2])


def own_pick(r):
    return int(np.argmin([abs(c[0] - r[2]) for c in r[4]]))


print(f"★ РЕШЕНИЙ {len(R)} (с верным раном среди кандидатов)")
for name, sel in (("УДЕРЖАНИЕ (трасса на своей линии — момент ухода)", lambda r: r[1]),
                  ("ВОЗВРАТ (трасса не на своей)", lambda r: not r[1])):
    S = [r for r in R if sel(r)]
    E = [r for r in S if not r[0]]; OK = [r for r in S if r[0]]
    if not E:
        continue
    d = np.array([abs(r[4][r[3]][0] - r[4][corr(r)][0]) for r in E])
    dc = np.array([abs(r[4][corr(r)][0] - r[2]) for r in E])
    dw = np.array([abs(r[4][r[3]][0] - r[2]) for r in E])
    fix = sum(1 for r in E if r[4][own_pick(r)][2]); brk = sum(1 for r in OK if not r[4][own_pick(r)][2])
    wd = np.array([abs(r[4][r[3]][1] - r[4][corr(r)][1]) for r in E])
    nc = Counter(min(len(r[4]), 4) for r in E)
    print(f"\n{name}: решений {len(S)}, ошибок {len(E)} ({100*len(E)/len(S):.2f}%)")
    print(f"   |темнота выбранного − верного|: медиана {np.median(d):.0f}, > 20 у {100*np.mean(d > 20):.0f}%, > 40 у {100*np.mean(d > 40):.0f}%")
    print(f"   верный ближе к своей темноте, чем выбранный, — {100*np.mean(dc < dw):.0f}% ошибок; "
          f"правило «ближе к своей темноте»: чинит {fix} ({100*fix/len(E):.0f}%), ломает {brk} ({100*brk/max(1, len(OK)):.1f}% верных)")
    print(f"   |ширина выбранного − верного|: медиана {np.median(wd):.0f} px, > 3 px у {100*np.mean(wd > 3):.0f}%")
    print("   кандидатов в ошибке: " + ", ".join(f"{k if k < 4 else '4+'}: {100*v/len(E):.0f}%" for k, v in sorted(nc.items())))

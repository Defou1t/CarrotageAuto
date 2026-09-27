r"""_replay_sweep.py — ПЕРЕБОР РУЧЕК РАСКЛАДКИ И ВЫБОРА ПУТИ ПОВТОРОМ С КЭША ТРАСС (§6.224, 26.09).

§6.222: повтор `emit` с кэша побайтно равен отгрузке на всём поле ⇒ всё, что действует ПОСЛЕ ведения (выбор пути по треку и
по слоту, раскладка), меряется за минуты. Здесь — база (нынешний прод) и варианты, каждый меняет ОДНУ ручку; повтор всех
режимов одним вызовом `_trace_cache.py replay`, счёт каждого — `_name_cost_prod.py` (по файлам), сравнение с базой — по листам.

КРИТЕРИЙ (задаётся ДО прогона в ROADMAP; здесь — исполнение): вариант ОТБИРАЕТСЯ, если на поле безымянных Δ > 0 при
p < 0.05 / k (k — число вариантов, поправка Бонферрони) и именных Δ ≥ −5; ПРИНИМАЕТСЯ, если вдобавок на держанном сорте A
безымянных Δ ≥ 0 и именных Δ ≥ 0. Из нескольких принятых — с наибольшим Δ на поле; комбинации — отдельным прогоном.

  _replay_sweep.py --base "rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1" --var P1:rdpick=1 --var S10:slotlen=0.10 ...
"""
import sys, argparse, subprocess, pickle, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--sheets", default="tcache_sheets.txt")
ap.add_argument("--out", default=r"F:/nds/output/taskS/rp_sweep")
ap.add_argument("--tag", default="sweep")
ap.add_argument("--base", required=True, help="ключи повтора базы (нынешний прод)")
ap.add_argument("--var", action="append", default=[], help="ИМЯ:ключ=зн[,ключ=зн] — ОТЛИЧИЯ от базы")
ap.add_argument("--perm", type=int, default=200000)
ap.add_argument("--skip-replay", action="store_true")
ap.add_argument("--conf", default="", help="§6.240: сайдкар уверенности декодера, передаётся повтору")
a = ap.parse_args()
TS = Path(a.ts); PY = sys.executable; RND = Path(__file__).resolve().parent


def merge(base, diff):
    kv = dict(x.split("=", 1) for x in base.split(",") if x)
    kv.update(dict(x.split("=", 1) for x in diff.split(",") if x))
    return ",".join(f"{k}={v}" for k, v in kv.items())


modes = [("B", a.base)] + [(v.split(":", 1)[0], merge(a.base, v.split(":", 1)[1])) for v in a.var]
k = len(modes) - 1
print(f"★ ПЕРЕБОР: база + {k} вариантов; порог отбора на поле p < {0.05 / max(1, k):.4f}")
for nm, spec in modes:
    print(f"   {nm}: {spec}")
if not a.skip_replay:
    cmd = [PY, str(RND / "_trace_cache.py"), "replay", "--sheets", a.sheets, "--cache", a.cache, "--out", a.out]
    if a.conf:
        cmd += ["--conf", a.conf]
    for nm, spec in modes:
        cmd += ["--mode", f"{nm}:{spec}"]
    r = subprocess.run(cmd, cwd=str(RND), capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-2000:]); sys.exit("⛔ повтор упал")
R = {}
for nm, _ in modes:
    pc = TS / f"percurve_{a.tag}_{nm}.pkl"
    if not pc.exists():
        r = subprocess.run([PY, str(RND / "_name_cost_prod.py"), "--dir", a.out, "--mode", nm, "--dump", str(pc)],
                           cwd=str(RND), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 or not pc.exists():
            print(r.stderr[-1500:]); sys.exit(f"⛔ счёт режима {nm} не выгружен")
    d = pickle.load(open(pc, "rb"))
    R[nm] = d[nm] if nm in d else next(iter(d.values()))
    print(f"  счёт {nm} готов", flush=True)

wm = json.load(open(TS / "rowdec_wellmap.json", encoding="utf-8"))
lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
SETS = {"поле": lst("wellmap_sheets.txt"), "сорт A": lst("holdoutA_sheets.txt")}


def perm_p(dv):
    nz = dv[dv != 0]
    if not len(nz):
        return 1.0
    rng = np.random.default_rng(0)
    sims = (rng.choice([-1, 1], size=(a.perm, len(nz))) * np.abs(nz)).sum(1)
    return float((np.abs(sims) >= abs(dv.sum()) - 1e-9).mean())


B = R["B"]
res = {}
print("\n| вариант | набор | безымянных база → вар (Δ) | листов ↑/↓ | p | именных Δ |")
print("|---|---|---|---|---|---|")
for nm, spec in modes[1:]:
    V = R[nm]; res[nm] = {}
    for sn, sh in SETS.items():
        com = sorted(s for s in sh if s in B and s in V)
        du = np.array([V[s][1] - B[s][1] for s in com], float)
        dn = np.array([V[s][0] - B[s][0] for s in com], float)
        p = perm_p(du)
        res[nm][sn] = dict(du=int(du.sum()), dn=int(dn.sum()), p=p, up=int((du > 0).sum()), dn_=int((du < 0).sum()), n=len(com))
        print(f"| {nm} | {sn} ({len(com)}) | {int(sum(B[s][1] for s in com))} → {int(sum(V[s][1] for s in com))} "
              f"({int(du.sum()):+d}) | {int((du > 0).sum())}/{int((du < 0).sum())} | {p:.4f} | {int(dn.sum()):+d} |")
thr = 0.05 / max(1, k)
sel = [nm for nm in res if res[nm]["поле"]["du"] > 0 and res[nm]["поле"]["p"] < thr and res[nm]["поле"]["dn"] >= -5]
acc = [nm for nm in sel if res[nm]["сорт A"]["du"] >= 0 and res[nm]["сорт A"]["dn"] >= 0]
print(f"\n★ ОТОБРАНО на поле (Δ > 0, p < {thr:.4f}, именных ≥ −5): {sel or '—'}")
print(f"★★ ПРИНЯТО (и сорт A: безымянных ≥ 0, именных ≥ 0): {acc or '—'}")
if acc:
    best = max(acc, key=lambda n: res[n]["поле"]["du"])
    print(f"★★ ЛУЧШИЙ: {best} — {dict(modes)[best]}")
json.dump(res, open(TS / f"{a.tag}_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

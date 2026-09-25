r"""_seqbig_verdict.py — ПРИГОВОР СЕЛЕКТОРУ НА ПОЛНОМ КОРПУСЕ (§6.220). Критерий задан ДО прогона (ROADMAP §6.220).

Сравнение двух повторов режима «нынешний прод» (kslots + выбор по слоту + имена) на кэшах трасс, собранных со старым
(`seq_model_d45p.pt`, 25 листов) и новым (`seq_model_big.pt`) селектором. Счёт — `_name_cost_prod.py` по файлам.
  ПОЛЕ (1123 листа; скважины поля в обучение нового НЕ входили): безымянных Δ ≥ +20 при p < 0.01 и именных Δ ≥ −15;
  ДЕРЖАННЫЙ СОРТ A (311 листов): безымянных Δ > 0 и НЕ ⛔ (Δ < 0 при p < 0.05).
Оба условия ⇒ ★ ПРИНЯТЬ (код 0); иначе ⚠/⛔ (код 2). p — парная перестановка знаков по листам.

  _seqbig_verdict.py --old percurve_rpold_N.pkl --new percurve_rpbig_N.pkl
"""
import sys, argparse, pickle, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--old", required=True)
ap.add_argument("--new", required=True)
ap.add_argument("--mode", default="N")
ap.add_argument("--perm", type=int, default=200000)
a = ap.parse_args()
TS = Path(a.ts)
wm = json.load(open(TS / "rowdec_wellmap.json", encoding="utf-8"))


def load(p):
    d = pickle.load(open(TS / p, "rb"))
    return d[a.mode] if a.mode in d else next(iter(d.values()))     # файл счёта помечен своим режимом


def lst(n):
    return {l.strip() for l in (TS / n).read_text(encoding="utf-8").splitlines() if l.strip()}


def perm_p(d):
    nz = d[d != 0]
    if not len(nz):
        return 1.0
    rng = np.random.default_rng(0)
    sims = (rng.choice([-1, 1], size=(a.perm, len(nz))) * np.abs(nz)).sum(1)
    return float((np.abs(sims) >= abs(d.sum()) - 1e-9).mean())


O, N = load(a.old), load(a.new)
ok_all = True
for name, sheets in (("ПОЛЕ", lst("wellmap_sheets.txt")), ("ДЕРЖАННЫЙ СОРТ A", lst("holdoutA_sheets.txt"))):
    com = sorted(s for s in sheets if s in O and s in N)
    # ⛔ 26.09 (аудит): критерий задан на ВСЁ поле / весь сорт A — неполное покрытие (> 0.5% листов нет хотя бы в одном
    #   счёте) приговором не считается: код 3, драйвер не пишет маркер конца.
    if len(com) < 0.995 * len(sheets):
        print(f"⛔ {name}: НЕПОЛНО — в обоих счётах {len(com)} листов из {len(sheets)}; приговор не выносится")
        sys.exit(3)
    du = np.array([N[s][1] - O[s][1] for s in com], float)
    dn = np.array([N[s][0] - O[s][0] for s in com], float)
    pu = perm_p(du)
    wells = defaultdict(float)
    for s, v in zip(com, du):
        wells[wm.get(Path(s).stem, "?")] += v
    wp = sum(1 for v in wells.values() if v > 0); wn = sum(1 for v in wells.values() if v < 0)
    if name == "ПОЛЕ":
        ok = du.sum() >= 20 and pu < 0.01 and dn.sum() >= -15
    else:
        ok = du.sum() > 0 and not (du.sum() < 0 and pu < 0.05)
    ok_all &= ok
    print(f"{name}: листов {len(com)} (из {len(sheets)}), кривых {sum(O[s][2] for s in com)}; "
          f"безымянных {int(sum(O[s][1] for s in com))} → {int(sum(N[s][1] for s in com))} (Δ {int(du.sum()):+d}, "
          f"листов ↑{int((du > 0).sum())}/↓{int((du < 0).sum())}, p = {pu:.4f}, скважин +{wp}/−{wn}); "
          f"именных {int(sum(O[s][0] for s in com))} → {int(sum(N[s][0] for s in com))} (Δ {int(dn.sum()):+d})   "
          f"{'★ условие выполнено' if ok else '⛔ условие НЕ выполнено'}")
print("★★ ПРИНЯТЬ новый селектор" if ok_all else "⛔ НЕ ПРИНИМАТЬ новый селектор")
sys.exit(0 if ok_all else 2)

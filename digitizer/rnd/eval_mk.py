r"""Трек 2 / объективный ГЕЙТ — px-ошибка + RATE СВОПОВ идентичности MGZ/MPZ vs экспертный GT (nlgx).
Заменяет «на глаз по кропам» на числа. Метрики:
  • swap_rate — доля строк, где pred-MGZ ближе к GT-MPZ (идентичность перепутана). ~0 хорошо;
    ~100% = глобально инвертировано (чинится свопом каналов); ~50% = рандомные свопы (плохо).
  • px_aligned — медиана |Δx| после ЛУЧШЕЙ пары pred↔GT (качество СЛЕДОВАНИЯ кривой, без идентичности);
  • px_labeled — |pred_MGZ − GT_MGZ| как есть (с ошибками идентичности); coverage — доля покрытых строк GT.

  python eval_mk.py <pred_traces.npz> <gt.nlgx>          # оценка предсказания
  python eval_mk.py --sanity <gt.nlgx>                   # GT vs GT (должно дать 0/0)
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_nlgx import extract, NULL
import dataset as ds


def gt_traces(nlgx):
    m = extract(str(nlgx))
    cm = {ds.mnemonic(c["name"]).upper(): c for c in ds.real_curves(m)}
    if "MGZ" not in cm or "MPZ" not in cm:
        return None, None
    def tr(c):
        ty = c["top_y"]; return {ty + i: float(x) for i, x in enumerate(c["xs"]) if x != NULL}
    return tr(cm["MGZ"]), tr(cm["MPZ"])


def evaluate(pred_mgz, pred_mpz, gt_mgz, gt_mpz):
    rows = sorted(set(gt_mgz) & set(gt_mpz))
    swaps = 0; n = 0; eA_m = []; eA_p = []; eL_m = []; eL_p = []
    for y in rows:
        if y not in pred_mgz or y not in pred_mpz:
            continue
        pm, pp = pred_mgz[y], pred_mpz[y]; gm, gp = gt_mgz[y], gt_mpz[y]
        eL_m.append(abs(pm - gm)); eL_p.append(abs(pp - gp))          # как размечено (с идентичностью)
        straight = abs(pm - gm) + abs(pp - gp); swapped = abs(pm - gp) + abs(pp - gm)
        if swapped < straight:                                        # pred-пара ближе в СВОПнутом виде
            swaps += 1; eA_m.append(abs(pm - gp)); eA_p.append(abs(pp - gm))
        else:
            eA_m.append(abs(pm - gm)); eA_p.append(abs(pp - gp))
        n += 1
    if not n:
        return {"n": 0}
    return {"n": n, "coverage": round(n / max(1, len(rows)), 3),
            "swap_rate": round(swaps / n, 3),
            "mgz_px_aligned": round(float(np.median(eA_m)), 1), "mpz_px_aligned": round(float(np.median(eA_p)), 1),
            "mgz_px_labeled": round(float(np.median(eL_m)), 1), "mpz_px_labeled": round(float(np.median(eL_p)), 1),
            "aligned_le2px": round(float(np.mean(np.array(eA_m) <= 2)), 2)}


def main():
    a = sys.argv[1:]
    sys.stdout.reconfigure(encoding="utf-8")
    if a and a[0] == "--sanity":
        gm, gp = gt_traces(a[1])
        print("GT найден:", gm is not None, "| sanity GT-vs-GT:", evaluate(gm, gp, gm, gp))
        return
    pred, nlgx = a[0], a[1]
    d = np.load(pred)
    pmgz = {int(y): float(x) for y, x in zip(d["mgz_y"], d["mgz_x"])}
    pmpz = {int(y): float(x) for y, x in zip(d["mpz_y"], d["mpz_x"])}
    gm, gp = gt_traces(nlgx)
    if gm is None:
        print("НЕТ GT MGZ/MPZ в", nlgx); return
    print(f"{Path(nlgx).stem[:40]}: {evaluate(pmgz, pmpz, gm, gp)}")


if __name__ == "__main__":
    main()

r"""_decoder_seq_diag.py — РАЗБОР ОСТАТОЧНОЙ ОШИБКИ оконного селектора (§6.25-§6.27).

Отвечает на два вопроса сразу, по ВСЕМ 24 кривым стенда:

1. ИЗ ЧЕГО СОСТОИТ px. Каждая строка трассы: наш ран накрывает точку эксперта (ран верный, px
   тратится на точку ВНУТРИ рана) / накрывает ЧУЖУЮ кривую (латч, §6.15) / ничью. Плюс контрфакт
   `med*` — ошибка при идеальном выборе точки внутри верного рана. Именно этот разбор дал §6.26
   (у 7 из 9 med* = 0 ⇒ виновато правило точки, а не контекст) и §6.27 (голова не окупилась).

2. ИЗ ЧЕГО СОСТОИТ НЕДОБОР COV (§6.23). Строки эксперта, которых в трассе НЕТ, делятся:
   ЕСТЬ ТУШЬ (трасса оборвалась на живой кривой — восстановимо) / НЕТ ТУШИ (GT на пустом:
   пунктир/выцвет — ★ловушка §6.20.3, добивать нельзя: cov растёт, правильность не проверяется).

  <ComfyUI>\python_embeded\python.exe _decoder_seq_diag.py --ckpt seq_model_d45p.pt [--point center]
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
import torch
import _relatch_bench as BE
from _decoder_seq import WindowSelector, make_tracer, OUT

INK_TOL = 4          # ±px, как в `_coverage_diag.py` (§6.23): «под строкой GT есть тушь»


def diag(tracer):
    print(f"{'скважина':<12}{'кривая':<8}{'полоса':>7}{'med':>7}{'med*':>7}"
          f"{'свой ран%':>10}{'чужой%':>8}{'ничей%':>8}{'полушир':>8}"
          f"{'cov':>6}{'проп.ЕСТЬ туш%':>15}{'проп.НЕТ туши%':>15}")
    agg = {"mine": 0, "alien": 0, "none": 0, "ink": 0, "noink": 0}
    for sheet in BE.load():
        H = sheet["H"]
        names, mat = BE._gt_matrix(sheet)
        for rec in sheet["curves"]:
            oi = names.index(rec["name"])
            best, best_med, best_c = None, 1e9, None
            for c in sheet["colors"]:                    # цвет выбирается как в гейте: лучший med
                tr = tracer(rec, rec["runs"][c], H)
                if not tr:
                    continue
                ys = np.fromiter(tr.keys(), np.int64, len(tr))
                xs = np.fromiter(tr.values(), np.float64, len(tr))
                m = ~np.isnan(mat[oi, ys])
                if int(m.sum()) < 30:
                    continue
                med = float(np.median(np.abs(xs[m] - mat[oi, ys][m])))
                if med < best_med:
                    best_med, best, best_c = med, tr, c
            if best is None:
                continue
            csr = rec["runs"][best_c]
            own = mine = alien = none = 0
            err, err_ideal, halfw = [], [], []
            for y, x in best.items():
                gt = mat[oi, y]
                if np.isnan(gt):
                    continue
                own += 1
                A, B, C = BE._runs_at(csr, y)
                k = np.nonzero((A - 1 <= x) & (x <= B + 1))[0]   # ран, в котором стоит НАША точка
                if not len(k):
                    none += 1; err.append(abs(x - gt)); err_ideal.append(abs(x - gt)); continue
                k = int(k[0]); a, b = float(A[k]), float(B[k])
                err.append(abs(x - gt))
                if a - 3 <= gt <= b + 3:
                    mine += 1; err_ideal.append(0.0); halfw.append((b - a) / 2)
                else:
                    d = np.abs(mat[:, y] - (a + b) / 2)          # накрывает ли он ЧУЖУЮ кривую
                    d = np.where(np.isnan(d), np.inf, d)
                    j = int(np.argmin(d))
                    if j != oi and a - 3 <= mat[j, y] <= b + 3:
                        alien += 1
                    else:
                        none += 1
                    err_ideal.append(abs(x - gt))
            # ── НЕДОБОР COV: что под пропущенными строками эксперта
            gys, gxs = sheet["gt"][rec["name"]]
            miss = [(int(y), float(xx)) for y, xx in zip(gys, gxs) if int(y) not in best]
            ink = 0
            for y, xx in miss:
                if not (0 <= y < H):
                    continue
                A, B, _ = BE._runs_at(csr, y)
                if len(A) and np.any((A - INK_TOL <= xx) & (xx <= B + INK_TOL)):
                    ink += 1
            nm = max(1, len(miss))
            s = max(1, own)
            cov = own / max(1, len(gys))
            agg["mine"] += mine; agg["alien"] += alien; agg["none"] += none
            agg["ink"] += ink; agg["noink"] += len(miss) - ink
            print(f"{sheet['well']:<12}{rec['short']:<8}{rec['hi']-rec['lo']:>7}"
                  f"{np.median(err):>7.1f}{np.median(err_ideal):>7.1f}"
                  f"{100*mine/s:>10.1f}{100*alien/s:>8.1f}{100*none/s:>8.1f}"
                  f"{(np.median(halfw) if halfw else float('nan')):>8.1f}"
                  f"{cov:>6.2f}{100*ink/nm:>15.1f}{100*(len(miss)-ink)/nm:>15.1f}")
    t = max(1, agg["mine"] + agg["alien"] + agg["none"])
    tm = max(1, agg["ink"] + agg["noink"])
    print(f"\nВСЕГО строк трассы: свой ран {100*agg['mine']/t:.1f}%  чужой {100*agg['alien']/t:.1f}%  "
          f"ничей {100*agg['none']/t:.1f}%")
    print(f"ВСЕГО пропущено строк GT: {agg['ink']+agg['noink']}  из них ЕСТЬ ТУШЬ "
          f"{100*agg['ink']/tm:.1f}% (восстановимо)  НЕТ ТУШИ {100*agg['noink']/tm:.1f}% "
          f"(ловушка §6.20.3 — добивать нельзя)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="seq_model_d45p.pt")
    ap.add_argument("--point", default="center", choices=["rule", "center", "pred", "head"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = WindowSelector().to(dev)
    net.load_state_dict(torch.load(OUT / a.ckpt, map_location=dev)["sd"]); net.eval()
    print(f"модель {a.ckpt}, точка={a.point}, устройство {dev}")
    print("med* = ошибка при идеальном выборе точки ВНУТРИ верного рана (§6.26)\n")
    diag(make_tracer(net, dev, point=a.point))

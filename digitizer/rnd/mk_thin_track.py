r"""Толщинный пост-трекер «ТОНКАЯ-ВОЗЛЕ-ЖИРНОЙ» (домен-ответ Эдуарда 04.07 по 3853):
на части планшетов одна кривая ЖИРНАЯ зигзагит, вторая — ТОНКАЯ линия рядом с
редкими выходами. Провалы 3752/3853 (le3 23-34%) — не «наложение», а качели
толщинной идентичности каналов модели.

★ ВЕРДИКТ ГЕЙТА (04.07, на ночных ens3-трассах): к 3752/3853 НЕ применим.
3853 — роли толщиной не различимы (обе НАШИ трассы сидят на жгуте, ширину
настоящей тонкой по ним не измерить); 3752 — «толщина» 72px = жгут целиком,
якорь-жирная сама битая (med 12.8px), тонкая от неё 12.6→26.9px РЕГРЕСС.
Диагноз по картинке (export_qc\..._diag_3860_3880.png): 3752 = СПЛОШНОЙ
зигзаг-режим (штрихи почти горизонтальны, ~300px за 1-3 строки, клики эксперта
на вершинах) — per-row экстракция принципиально не описывает кривую, нужен
СКЕЛЕТНЫЙ обход пути пера (этап B п.3, отдельная сессия). Пасс оставлен: может
пригодиться на планшетах с настоящей структурой «тонкая возле жирной», где
роли различимы (гейт применимости THICK_RATIO отсекает остальное).

Пасс (image-first, prob как tiebreak):
  1. Роли per-planshet: жирная = трасса с большей медианной шириной рана;
     если ширины близки (ratio < THICK_RATIO) — пасс НЕ применим, выходим как есть.
  2. Жирная не трогается (канальный peak её ведёт хорошо).
  3. Тонкая per-row: кандидаты = тёмные раны шириной ≤ THIN_FRAC*толщина жирной
     в окне ±SEARCH от жирной, НЕ совпадающие с раном жирной; выбор по
     непрерывности (ближе к prev) + бонус за prob своего канала. Нет кандидата →
     тонкая «прячется» на жирной (слипание, sliplis-норма) → x жирной.

  python mk_thin_track.py <traces.npz> <scan.jpg> <etalon.nlgx>   # гейт до/после
"""
import sys, os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from extract_nlgx import extract, NULL
from mk_refine import _stroke_width

THICK_RATIO = 1.35   # жирная/тонкая: ниже — ролей нет, пасс выключен
THIN_FRAC = 0.75     # кандидат тонкой: ширина ≤ THIN_FRAC * ширины жирной в строке
SEARCH = 70          # окно поиска тонкой вокруг жирной, px
JUMP = 28            # непрерывность: макс. скачок тонкой между соседними её строками


def _runs(row, x0, x1, gap=1):
    """Тёмные раны [a,b] в окне столбцов [x0,x1) с допуском gap светлых px."""
    runs = []
    a = None
    g = 0
    for x in range(x0, x1):
        if row[x]:
            if a is None:
                a = x
            g = 0
        elif a is not None:
            if g < gap:
                g += 1
            else:
                runs.append((a, x - 1 - g))
                a = None; g = 0
    if a is not None:
        runs.append((a, x1 - 1 - g))
    return runs


def thin_near_thick(dark, thick, thin, prob_thin=None):
    """Перестройка тонкой трассы. Возврат (новая_тонкая, stats)."""
    H, W = dark.shape
    w_thick = _stroke_width(dark, thick)
    out = {}
    prev = None
    n_cand = n_slip = n_keep = 0
    for y in sorted(thick):
        xt = thick[y]
        xi = int(round(xt))
        x0, x1 = max(0, xi - SEARCH), min(W, xi + SEARCH + 1)
        runs = _runs(dark[y], x0, x1)
        # ран жирной = содержащий xt (или ближайший)
        troom = None
        for a, b in runs:
            if a - 2 <= xt <= b + 2:
                troom = (a, b); break
        cands = []
        for a, b in runs:
            if troom and not (b < troom[0] or a > troom[1]):
                continue                       # это ран жирной
            wr = b - a + 1
            if wr > THIN_FRAC * max(w_thick, 4.0):
                continue                       # слишком толстый — чужая жирная/клякса
            c = (a + b) / 2.0
            score = 0.0
            if prev is not None:
                score -= abs(c - prev) / JUMP  # непрерывность
            if prob_thin is not None:
                lo, hi = max(0, int(c) - 2), min(W, int(c) + 3)
                score += 2.0 * float(prob_thin[y, lo:hi].max())
            cands.append((score, c))
        old = thin.get(y)
        if cands:
            cands.sort(reverse=True)
            c = cands[0][1]
            if prev is not None and abs(c - prev) > JUMP and old is not None \
                    and abs(old - (prev if prev is not None else old)) < abs(c - prev):
                out[y] = old                   # резкий скок и старая ближе — не рвём
                n_keep += 1
            else:
                out[y] = c
                n_cand += 1
        else:
            out[y] = xt                        # тонкая прячется на жирной
            n_slip += 1
        prev = out[y]
    return out, {"w_thick": round(w_thick, 1), "cand": n_cand, "slip": n_slip, "keep": n_keep}


def apply(dark, mgz, mpz, prob=None):
    """Роли по толщине; возврат (mgz, mpz, stats|None). None = пасс не применим."""
    wm = _stroke_width(dark, mgz)
    wp = _stroke_width(dark, mpz)
    if max(wm, wp) < THICK_RATIO * min(wm, wp):
        return mgz, mpz, None
    if wm >= wp:                               # MGZ жирная (3853 по Эдуарду)
        thick, thin, pi = mgz, mpz, 1
    else:
        thick, thin, pi = mpz, mgz, 0
    pt = prob[pi] if prob is not None else None
    new_thin, st = thin_near_thick(dark, thick, thin, prob_thin=pt)
    st["roles"] = f"thick={'MGZ' if thick is mgz else 'MPZ'} (w {max(wm,wp):.0f}/{min(wm,wp):.0f})"
    if thick is mgz:
        return mgz, new_thin, st
    return new_thin, mpz, st


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    from night_gate import gate_traces
    tr_path, scan, et_path = sys.argv[1], sys.argv[2], sys.argv[3]
    d = np.load(tr_path)
    mgz = {int(y): float(x) for y, x in zip(d["mgz_y"], d["mgz_x"])}
    mpz = {int(y): float(x) for y, x in zip(d["mpz_y"], d["mpz_x"])}
    rgb = np.asarray(Image.open(scan).convert("RGB"))
    dark = rgb.max(2) < 110
    pr = os.path.join(os.path.dirname(tr_path), "mk_prob2.npy")
    prob = np.load(pr).astype(np.float32) if os.path.exists(pr) else None
    me = extract(et_path)
    def show(tag, a, b):
        r = gate_traces({"MGZ1": a, "MPZ1": b}, me)
        print(f"--- {tag} ---")
        for k, v in r.items():
            print(f"  [{k}] med {v['med']}px | ≤3px {v['le3']*100:.0f}% | miss {v['miss_frac']*100:.0f}%")
        return r
    show("до", mgz, mpz)
    m2, p2, st = apply(dark, mgz, mpz, prob=prob)
    if st is None:
        print("роли толщиной не различимы — пасс не применим")
        return
    print(f"stats: {st}")
    show("после thin-near-thick", m2, p2)


if __name__ == "__main__":
    main()

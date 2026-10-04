r"""name_style.py — ИМЕНА КРИВЫХ ТРЕКА ПО СТИЛЮ ЛИНИИ: ЦВЕТ, ТОЛЩИНА, ШТРИХ (§6.273, 04.10).

ЗАЧЕМ. Заказчик различает кривые трека «по цвету, толщине и штриху» (ГЗ1 толще ГЗ2, ПС красная, ОГЗ синяя/тоньше…). Раскладка
(`emit`) этого не видит: на поле из 1303 честных кривых 392 названы чужим именем, а перепутанные пары в 81% различимы стилем.

ЧТО ДЕЛАЕТ. До записи трасс в рамку: для каждой кривой — стиль вдоль её трассы (толщина по нормали, цвет самого тёмного пикселя
рана, затемнение, доля разрывов); для пары кривых одного трека под именами A и B парная модель (градиентный бустинг, учена по
эталону) даёт P(порядок верен). P < 1 − τ ⇒ трассы меняются местами между слотами (жадно, самые уверенные первыми, каждая —
в одной перестановке). Уровни масштаба потом считаются уже для нового слота.

ЗАМЕРЕНО ДО ВНЕСЕНИЯ (§6.273, стенд `_name_style.py`, пары 1:1 мерой приёмки): τ = 0.9 — поле +25 именных (p = 0.0048, вне
фолда по скважинам), сорт A +2.

⚠ Замер стиля и признаки ПОВТОРЯЮТ `digitizer/rnd/_style_scan.style_along` и `_name_style.feat` в точности (§6.90): правка —
в обоих местах. Без sklearn или без файла модели — раскладка не меняется, и об этом говорится вслух.
"""
import json
import re
from pathlib import Path

import numpy as np

_MODELS = Path(__file__).resolve().parent / "models"
_CACHE = {}
_SAID = set()


def _announce(msg):
    if msg not in _SAID:
        _SAID.add(msg)
        print(f"  {msg}")


def style_along(rgb, gray, paper, ys, xs, thr=60):
    """стиль линии по точкам (ys, xs) — ТО ЖЕ, что `_style_scan.style_along`"""
    H, W = gray.shape
    ys = ys.astype(np.int64); xs = np.round(xs).astype(np.int64)
    ok = (ys >= 1) & (ys < H - 1) & (xs >= 45) & (xs < W - 45)
    ys, xs = ys[ok], xs[ok]
    n = len(ys)
    if n < 20:
        return None
    half = 44; NC = 2 * half + 1; c = half
    cols = xs[:, None] + np.arange(-half, half + 1)[None, :]
    D = paper[ys][:, None] - gray[ys[:, None], cols].astype(np.int16)
    ink = D > thr
    gap = ~ink[:, c - 2:c + 3].any(1)
    j0 = np.full(n, -1)
    for d in (0, -1, 1, -2, 2, -3, 3):
        m = (j0 < 0) & ink[:, c + d]
        j0[m] = c + d
    has = j0 >= 0
    ar = np.arange(NC)[None, :]
    leftF = np.maximum.accumulate(np.where(~ink, ar, -1), axis=1)
    rightF = np.minimum.accumulate(np.where(~ink, ar, NC)[:, ::-1], axis=1)[:, ::-1]
    jj = np.clip(j0, 0, NC - 1)
    L = leftF[np.arange(n), jj] + 1; R = rightF[np.arange(n), jj] - 1
    wl = (R - L + 1).astype(float)
    good = has & (L > 0) & (R < NC - 1) & (wl <= 40)
    Dm = np.where((ar >= L[:, None]) & (ar <= R[:, None]), D, -999)
    k = Dm.argmax(1)
    px = rgb[ys, cols[np.arange(n), k]].astype(np.int32)
    c1 = (px[:, 0] - px[:, 2]).astype(float); c2 = (px[:, 1] - (px[:, 0] + px[:, 2]) / 2).astype(float)
    dk = D[np.arange(n), k].astype(float)
    s_ = np.gradient(xs.astype(float), ys.astype(float)) if n > 2 else np.zeros(n)
    wn = wl / np.sqrt(1.0 + s_ * s_)
    if good.sum() < 10:
        return None
    g = good
    return dict(n=int(n), width=float(np.median(wn[g])), width_h=float(np.median(wl[g])),
                c1=float(np.median(c1[g])), c2=float(np.median(c2[g])), dark=float(np.median(dk[g])),
                gap=float(gap.mean()), used=float(g.mean()))


def _probe(name):
    m = re.match(r"^(?:BKZ_)?[A-Z]+?(\d)", name.split()[0])
    return int(m.group(1)) if m else 0


def _decade(sheet):
    m = re.search(r"(19[5-9]\d|20[0-2]\d)", sheet)
    return (int(m.group(1)) // 10 * 10) if m else 0


def _hue(t):
    c1, c2 = t["c1"], t["c2"]
    if max(abs(c1), abs(c2)) < 30:
        return 0
    if c2 > 30 and c2 >= abs(c1):
        return 3
    if c1 > 30:
        return 1
    if c1 < -30:
        return 2
    return 4


def feat(tx, ty, A, B, sheet, fams, root_of):
    """признаки упорядоченной пары — ТО ЖЕ, что `_name_style.feat` (семейство — код из словаря модели; новое — новый код)"""
    def fcode(name):
        f = re.sub(r"^BKZ_", "", root_of(name))
        return fams.setdefault(f, len(fams))
    return [tx["width"], ty["width"], tx["width"] - ty["width"], tx["c1"], tx["c2"], ty["c1"], ty["c2"],
            tx["c1"] - ty["c1"], tx["c2"] - ty["c2"], tx["dark"], ty["dark"], tx["gap"], ty["gap"],
            _hue(tx), _hue(ty), fcode(A), fcode(B), _probe(A), _probe(B), _probe(A) - _probe(B), _decade(sheet)]


def resolve(spec, sheet=None):
    """spec — файл модели в `auto/models` (прод) или `oof:<каталог>` (замер: лист поля — модель фолда его скважины)"""
    if not spec:
        return None
    if spec.startswith("oof:"):
        d = Path(spec[4:])
        mp = d / "name_style_folds.json"
        if str(mp) not in _CACHE:
            _CACHE[str(mp)] = json.loads(mp.read_text(encoding="utf-8"))
        k = _CACHE[str(mp)].get(sheet)
        return d / (f"name_style_f{k}.pkl" if k is not None else "name_style_all.pkl")
    p = Path(spec)
    if not p.is_absolute():
        p = _MODELS / spec
    return p if p.is_file() else None


def available(spec):
    try:
        import sklearn                                          # noqa: F401
    except Exception:
        return False
    if not spec:
        return False
    if spec.startswith("oof:"):
        return (Path(spec[4:]) / "name_style_all.pkl").is_file()
    return resolve(spec) is not None


def _load(path):
    key = str(path)
    if key not in _CACHE:
        import pickle
        _CACHE[key] = pickle.load(open(key, "rb"))
    return _CACHE[key]


def swaps(entries, sheet, path, root_of, tau=None):
    """entries — [(имя, трек, стиль)] (стиль None — кривая не участвует) → [(A, B, P)] перестановок имён"""
    ck = _load(path)
    model, fams = ck["model"], dict(ck["fams"])
    tau = ck["tau"] if tau is None else tau
    by = {}
    for nm, t, st in entries:
        if st is not None:
            by.setdefault(t, []).append((nm, st))
    out = []
    for t, L in by.items():
        cand = []
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                A, sa = L[i]; B, sb = L[j]
                p = float(model.predict_proba(np.array([feat(sa, sb, A, B, sheet, fams, root_of)], float))[0, 1])
                if p < 1 - tau:
                    cand.append((p, A, B))
        used = set()
        for p, A, B in sorted(cand):
            if A in used or B in used:
                continue
            used |= {A, B}
            out.append((A, B, p))
    return out

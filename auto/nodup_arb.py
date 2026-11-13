"""§6.363 — арбитр раздвигания дублей выбора трассы (`line_choice._nodup`, правило "arb").

Конфликт — два слота трека, чьи трассы совпадают на доле строк ≥ θ. Правило «sum» выбирает, кого переносить (lo — тот, у кого
вариант с большей суммой P пары); арбитр решает, что делать с этим выбором: «не раздвигать» (keep), «как есть» (cur — lo уходит на свою
альтернативу) или «обмен» (swp — lo остаётся на общей линии, его альтернативу берёт hi). Модель — две регрессии HGB
(Δ именных «как есть» − «не раздвигать» и «обмен» − «не раздвигать»), учена вне фолда по скважинам на стенде §6.359
(`digitizer/rnd/_nodup_arb.py`). Признаки — только из записи конфликта (`rec`, та же, что пишет журнал `NDS_NODUP_LOG`).

spec — файл в `auto/models` (прод) или `oof:<каталог>` (замер: лист — модель фолда его скважины, `nodup_arb_folds.json`).
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np

_MODELS = Path(__file__).resolve().parent / "models"
_CACHE = {}
FEATS = ["p_lo", "p_hi", "palt_lo", "phi_on_alt", "has_alt_hi", "palt_hi", "plo_on_althi", "frac_lo", "frac_hi",
         "rx_sh", "rx_alt", "rx_althi", "own_lo", "own_hi", "sum_lo", "sum_hi", "dsa", "same_fam",
         "chg_lo", "chg_hi", "wov_lo", "wov_hi", "rank_lo", "rank_hi", "ncand_lo", "ncand_hi", "pmax_lo", "pmax_hi"]


def _fam(n):
    return re.sub(r"\d+$", "", n.split()[0])


def _sa(n):
    m = re.search(r"SA(\d+)", n)
    return int(m.group(1)) if m else 0


def feat_dict(rec):
    """признаки конфликта; rec["moved"] — lo (кого переносит «sum»)"""
    A, B = rec["pair"]; lo = rec["moved"]; hi = B if lo == A else A
    il, ih = (0, 1) if lo == A else (1, 0)
    al = rec["alts"].get(lo); ah = rec["alts"].get(hi)
    sl = rec["slot"][il] if len(rec["slot"][il]) >= 2 else [0.0, 1.0]
    xl, xr = sl[0], sl[1]
    rx = lambda x: (x - xl) / ((xr - xl) or 1.0)
    nz = lambda v: np.nan if v is None else float(v)
    f = dict(p_lo=rec["p"][il], p_hi=rec["p"][ih], palt_lo=al[1] if al else np.nan, phi_on_alt=nz(al[2]) if al else np.nan,
             has_alt_hi=float(ah is not None), palt_hi=ah[1] if ah else np.nan, plo_on_althi=nz(ah[2]) if ah else np.nan,
             frac_lo=rec["frac"][il], frac_hi=rec["frac"][ih], rx_sh=rx(rec["x"][ih]), rx_alt=rx(al[4]) if al else np.nan,
             rx_althi=rx(ah[4]) if ah else np.nan, own_lo=float(rec["own"][il]), own_hi=float(rec["own"][ih]),
             sum_lo=(al[1] + rec["p"][ih]) if al else np.nan, sum_hi=(ah[1] + rec["p"][il]) if ah else np.nan,
             dsa=_sa(lo) - _sa(hi), same_fam=float(_fam(lo) == _fam(hi)))
    for k in ("chg", "wov", "rank", "ncand", "pmax"):
        v = rec.get(k) or [None, None]
        f[k + "_lo"] = nz(v[il]); f[k + "_hi"] = nz(v[ih])
    return f


def vec(rec, feats=FEATS):
    f = feat_dict(rec)
    return [f.get(k, np.nan) for k in feats]


def resolve(spec, sheet=None):
    if not spec:
        return None
    if spec.startswith("oof:"):
        d = Path(spec[4:])
        mp = d / "nodup_arb_folds.json"
        if str(mp) not in _CACHE:
            _CACHE[str(mp)] = json.loads(mp.read_text(encoding="utf-8"))
        k = _CACHE[str(mp)].get(sheet)
        return d / (f"nodup_arb_f{k}.pkl" if k is not None else "nodup_arb_all.pkl")
    p = Path(spec)
    if not p.is_absolute():
        p = _MODELS / spec
    return p if p.is_file() else None


def load(spec, sheet=None):
    p = resolve(spec, sheet)
    if p is None or not Path(p).is_file():
        return None
    if str(p) not in _CACHE:
        with open(p, "rb") as f:
            _CACHE[str(p)] = pickle.load(f)
    return _CACHE[str(p)]


def decide(rec, ck, margin=None, can_swp=True):
    """→ ("keep" | "cur" | "swp", r̂ как есть, r̂ обмен); can_swp=False (у hi нет трассы-кандидата) — выбор только keep / cur"""
    x = np.array([vec(rec, ck["feats"])], float)
    rc = float(ck["mc"].predict(x)[0]); rs = float(ck["ms"].predict(x)[0])
    m = ck.get("margin", 0.0) if margin is None or margin < 0 else margin
    if can_swp and rs > rc + m and rs > m:
        return "swp", rc, rs
    if rc < -m:
        return "keep", rc, rs
    return "cur", rc, rs

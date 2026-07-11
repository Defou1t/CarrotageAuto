r"""
meta.py — приор «ЧТО» из ИМЕНИ ФАЙЛА + mnemonics.json (durable, ROADMAP §5/§6.2).

Имя планшета: `Well_CURVES_from_to_scale_part.jpg`
  напр. `Semeguniv_20_BK+MBK_3080_3520_200_D1.jpg`
        → скважина Semeguniv_20, кривые BK+MBK, интервал 3080..3520 м, масштаб 1:200, часть D1.
Картинке остаётся найти «ГДЕ», а не «ЧТО»: ожидаемые мнемоники/единицы/группа/тип-трека —
из mnemonics.json (`filename_hints` тип→набор, `curves` мнемоника→единицы/группа, `aliases`).

Это ЧИСТЫЙ Python (re/json) — тестируется без изображений.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# класс поведения по мнемонике (дублирует digitizer/behavior_priors, но без зависимости от cv)
CALI = {"DS", "MDS", "DN", "DC"}
RES = {"GZ", "PZ", "BK", "MBK", "MGZ", "MPZ", "OGZ", "REZ", "IK", "IKA", "IKR", "KS"}

# ожидаемый «сильный» цвет кривой по мнемонике (эмпирика PLAN §6.6.7/§6.6.13);
# None = чёрная/неопределённая (берём по изображению). SP часто УНИКАЛЬНО красная.
EXPECTED_COLOR = {"SP": "red"}


@dataclass
class FileMeta:
    well: str
    curves_token: str                 # «BK+MBK», «BKZ», «STK+DS» как в имени
    top_depth: Optional[float]
    bottom_depth: Optional[float]
    scale: Optional[int]              # 200 = 1:200, 500 = 1:500 (НЕ dpi)
    part: str                         # «D1», «D2», …
    expected_curves: list = field(default_factory=list)   # мнемоники из filename_hints
    raw_stem: str = ""

    @property
    def depth_span(self):
        if self.top_depth is None or self.bottom_depth is None:
            return None
        return abs(self.bottom_depth - self.top_depth)


_NUM = r"-?\d+(?:\.\d+)?"


def _depth_marker(toks):
    """Найти маркер интервала глубин: вернуть (top, bot, curves_end_idx, after_idx).
    Поддержаны ОБА формата: пара соседних чисел `..._3080_3520_...` (delivery) и
    один дефисный токен `..._3630-3896_...` (archive). curves_end_idx — индекс токена-кривых
    (прямо перед маркером); after_idx — индекс первого токена ПОСЛЕ маркера (для масштаба)."""
    def ok(a, b):
        return 0 < a < 12000 and 0 < b < 12000 and abs(b - a) >= 1
    # 1) дефисный токен num-num (но НЕ дата yyyy-mm-dd = 3 части)
    for i, t in enumerate(toks):
        mm = re.fullmatch(r"(\d{2,5})-(\d{2,5})", t)
        if mm:
            a, b = float(mm.group(1)), float(mm.group(2))
            if ok(a, b):
                return a, b, i, i + 1
    # 2) пара соседних чисел
    nums = [i for i, t in enumerate(toks) if re.fullmatch(_NUM, t)]
    for a, b in zip(nums, nums[1:]):
        if b == a + 1 and ok(float(toks[a]), float(toks[b])):
            return float(toks[a]), float(toks[b]), a, b + 1
    return None


def parse_filename(path, mnemonics_path=None) -> FileMeta:
    """Разобрать имя планшета `Well[_NN]_CURVES_from[-_]to_scale[_part]`. Терпимо к хвостам.
    mnemonics_path — раскрыть тип-планшета (BKZ→GZ1..OGZ) через filename_hints."""
    stem = Path(path).stem
    toks = stem.split("_")
    dm = _depth_marker(toks)
    if not dm:
        return FileMeta(well=toks[0] if toks else stem, curves_token="",
                        top_depth=None, bottom_depth=None, scale=None, part="", raw_stem=stem)
    top, bot, cidx, after = dm
    curves_token = toks[cidx - 1] if cidx >= 1 else ""
    well = "_".join(toks[:cidx - 1]) if cidx >= 2 else (toks[0] if toks else stem)
    scale = int(float(toks[after])) if after < len(toks) and re.fullmatch(_NUM, toks[after]) else None
    part = ""
    if scale is not None and after + 1 < len(toks) and re.fullmatch(r"[A-Za-zД]+\d*", toks[after + 1]):
        part = toks[after + 1]                     # D1/D2-подобный токен (не дата)
    m = FileMeta(well=well, curves_token=curves_token, top_depth=top, bottom_depth=bot,
                 scale=scale, part=part, raw_stem=stem)
    m.expected_curves = expected_curves(curves_token, mnemonics_path)
    return m


_mnem_cache = {}


def load_mnemonics(path) -> dict:
    path = str(path)
    if path not in _mnem_cache:
        with open(path, encoding="utf-8") as f:
            _mnem_cache[path] = json.load(f)
    return _mnem_cache[path]


def expected_curves(curves_token: str, mnemonics_path=None) -> list:
    """Тип в имени (BKZ/BK+MBK/SK/MK/…) → ожидаемые мнемоники (filename_hints)."""
    if not curves_token:
        return []
    hints = {}
    if mnemonics_path:
        hints = load_mnemonics(mnemonics_path).get("filename_hints", {})
    out, seen = [], set()
    for part in re.split(r"[+\-_]", curves_token):
        p = part.strip().upper()
        if not p:
            continue
        if p in hints:                       # тип-планшета → набор зондов
            for c in hints[p]:
                if c not in seen:
                    seen.add(c); out.append(c)
        else:                                 # отдельная мнемоника
            if p not in seen:
                seen.add(p); out.append(p)
    return out


def mnem_root(name: str) -> str:
    """'GZ31' / 'BK1 DA1 SA1' → 'GZ'/'BK' (первый токен без хвостовых цифр)."""
    return re.sub(r"\d+$", "", str(name).split()[0]) if name else ""


def curve_class(name: str) -> str:
    mm = mnem_root(name).upper()
    if mm == "SP":
        return "SP"
    if mm in CALI:
        return "CALI"
    if mm in RES:
        return "RES"
    return "OTHER"


def expected_color(name: str) -> Optional[str]:
    return EXPECTED_COLOR.get(mnem_root(name).upper())


def curve_info(name: str, mnemonics_path) -> dict:
    """Сводка приоров на мнемонику: класс поведения, ожид. цвет, единицы/группа из словаря.
    Корень — ДЛИННЕЙШЕЕ совпадение со словарём (слот NeuraLOG 'SP21' = SP2 №1, а не SP №21;
    голое отбрасывание цифр давало SP и теряло оранжевую SP2)."""
    d = load_mnemonics(mnemonics_path)
    aliases = d.get("aliases", {})
    curves = d.get("curves", {})
    base = str(name).split()[0].upper() if name else ""
    m_dig = re.search(r"\d+$", base)
    root = None
    if m_dig:
        # слот NeuraLOG = <мнемоника><№экземпляра>: цифру экземпляра отбрасываем ОБЯЗАТЕЛЬНО,
        # затем длиннейший префикс из словаря ('SP21'→SP2, 'GZ31'→GZ3, 'GZ1'→GZ — иначе слот
        # 'GZ1' на STK-каркасе ловил словарную мнемонику GZ1 (зонд БКЗ) и терял цвет green)
        digs = len(m_dig.group())
        for k in range(1, digs + 1):
            cand = aliases.get(base[:-k], base[:-k])
            if cand in curves:
                root = cand; break
    if root is None:
        cand = aliases.get(base, base)
        root = cand if cand in curves else aliases.get(mnem_root(name).upper(),
                                                       mnem_root(name).upper())
    info = curves.get(root, {})
    # цвет: словарь (экспертное знание, напр. STK: PZ чёрная/GZ зелёная/SP красная) > встроенный приор
    return {"root": root, "class": curve_class(root), "color": info.get("color") or expected_color(root),
            "unit": info.get("unit"), "group": info.get("group"), "comment": info.get("comment")}

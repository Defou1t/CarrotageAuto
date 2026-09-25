r"""slot_geom.py — ГЕОМЕТРИЯ ШКАЛЫ СЛОТА ИЗ ШАБЛОНА КАК ОГРАНИЧЕНИЕ РАСКЛАДКИ (§6.215, 21.09).

⚠⚠ ВЫКЛЮЧЕНО ПО УМОЛЧАНИЮ (`CVParams.slot_template_geom = False` = прод бит-в-бит). ОТКАТ = False.

ЗАЧЕМ. §6.214: 375 кривых лежат в выдаче геометрически верно и подписаны неверно. Раскладка (правило и
обученная модель) не знает двух вещей, которые шаблон сообщает про каждый слот:
  1. ПОЛОСА его шкалы `[x_left, x_right]` — на листах с двумя осями глубины (BEZLUD_051: DA1 326…1800,
     DA2 1801…2974) слот левой половины получал кривую из правой;
  2. НОЛЬ его шкалы: у сдвинутой шкалы (`v_left < 0`, например GZ5 на BKZ2: −11…8 Ohmm) ноль стоит внутри
     трека (x ≈ 1002), а сопротивление/проводимость отрицательными не бывают ⇒ кривая слота обязана лежать
     правее нуля. Раскладка отдавала GZ5 кривую с x ≈ 350 (это OGZ), а OGZ — кривую с x ≈ 1180.
Эталон эти законы соблюдает: в полосе 2733 из 2739 кривых, правее нуля 594 из 607 (§6.214). На замороженном
поле запрет пар-нарушителей возвращает **+44 именных (732 → 776, листов ↑30/↓3, скважин +18/−1)** при
неизменной геометрии (`_name_constrain.py`).

ЧТО ДЕЛАЕТ. `geom(model, slot_name)` → (x_left, x_right, zero_x | None) по scale-оси с ПОЛНЫМ суффиксом
имени слота (как `emit._slot_track`); `forbidden(tr, g, pad, viol)` → True, если медиана x трассы вне
полосы ±pad, либо у слота есть сдвинутый ноль и больше `viol` доли строк трассы лежат левее нуля − pad.
Знаковые величины (mV, м, мкс, у.е., …) нуля не имеют — только полоса.
"""
import re
import numpy as np

# единицы, у которых отрицательные значения законны — ноль шкалы для них не ограничение
SIGNED_UNITS = {"MV", "M", "US", "US/M", "DB/M", "UE", "G/CM3", "MKS", "MKS/M", "%", "MM"}
_SUF = re.compile(r"(DA\d+\s+SA\d+)\s*$")


def geom(model, slot_name):
    """→ (x_left, x_right, zero_x или None) для слота, либо None, если оси с таким суффиксом в шаблоне нет."""
    m = _SUF.search(slot_name or "")
    if not m:
        return None
    suf = m.group(1)
    for s in model.get("scale_axes", []):
        if str(s.get("name", "")).strip() != suf:
            continue
        xl, xr = s.get("x_left"), s.get("x_right")
        if xl is None or xr is None or xr <= xl:
            return None
        zx = None
        u = str(s.get("units", "")).upper()
        vl, vr = s.get("v_left"), s.get("v_right")
        if u not in SIGNED_UNITS and vl is not None and vr is not None and vr != vl and vl < 0:
            zx = float(xl) + (0.0 - vl) / (vr - vl) * (xr - xl)
        return (float(xl), float(xr), zx)
    return None


def forbidden(tr, g, pad=15.0, viol=0.5):
    """Нарушает ли трасса `tr` ({row: x}) геометрию слота `g` (см. `geom`)."""
    if g is None or not tr:
        return False
    xs = np.fromiter(tr.values(), float)
    xl, xr, zx = g
    med = float(np.median(xs))
    if not (xl - pad <= med <= xr + pad):
        return True
    if zx is not None and float(np.mean(xs < zx - pad)) > viol:
        return True
    return False


def make_forbid(model, cv):
    """→ callable(slot_name, tr) → bool, либо None, если ручка выключена."""
    if not bool(getattr(cv, "slot_template_geom", False)):
        return None
    pad = float(getattr(cv, "slot_geom_pad", 15.0) or 15.0)
    viol = float(getattr(cv, "slot_geom_viol", 0.5) or 0.5)
    cache = {}

    def fb(slot_name, tr):
        if slot_name not in cache:
            cache[slot_name] = geom(model, slot_name)
        return forbidden(tr, cache[slot_name], pad, viol)
    return fb

r"""_ab_decide.py — ПРЕДРЕГИСТРИРОВАННЫЕ КРИТЕРИИ ПО ЛОГАМ A/B (08.11, §6.355–6.356): разбирает вывод `merge_ab3.py` (`_ab_<тег>.log`)
и `_strict_ab.py` (`_strict_ab_<тег>.log`) и проверяет пункты критерия; код выхода 0 — взят, 1 — нет, 2 — нет логов / не разобрать.

  --rule detour       §6.355 (вычистка §6.353 (2)): приёмка поле+вне именных Δ ≥ 0, безымянных Δ ≥ 0, сорт A именных Δ ≥ 0; строгая
                      поле+вне — чистых ≤ 5% Δ > 0 при p < 0.01, чистых ≤ 1% Δ ≥ 0, лишних строк ↓, пропусков не ↑ больше 5%
                      (и справка: обычное правило отгрузки — именных Δ ≥ +15 при p < 0.01)
  --rule spike-stand  §6.356 стенд: строгая поле+вне — верных Δ ≥ 0, чистых ≤ 1% Δ ≥ 0, ≤ 5% Δ ≥ 0, лишних ↓, пропусков не ↑;
                      сорт A — верных Δ ≥ 0, чистых ≤ 5% Δ ≥ 0
  --rule ship         обычное правило отгрузки: приёмка поле+вне именных Δ ≥ +15 при p < 0.01, безымянных Δ ≥ 0, сорт A именных
                      Δ ≥ 0 (строгая мера — справкой)
  --rule spike-ab     §6.356 A/B: то же, что spike-stand, по строгой мере повтора + приёмка поле+вне именных Δ ≥ 0, безымянных Δ ≥ 0,
                      сорт A именных Δ ≥ 0

  _ab_decide.py --rule detour --merge F:/nds/output/taskS/_ab_nse22.log --strict F:/nds/output/taskS/_strict_ab_nse22.log
"""
import sys, re, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

NUM = r"(-?\d+)"


def read(p):
    """текст лога (UTF-8 / UTF-8 с BOM / UTF-16 от PowerShell)"""
    b = Path(p).read_bytes()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            t = b.decode(enc)
            if "★" in t or "##" in t:
                return t
        except UnicodeDecodeError:
            pass
    return b.decode("utf-8", "replace")


def merge(p):
    """{(часть, именных|безымянных): (a, b, Δ, p)}, {часть: листов}"""
    out, n = {}, {}
    for m in re.finditer(r"★ (поле\+вне|сорт A|поле|вне) \((\d+) листов\), (именных|безымянных) честных в значениях: \S+ " + NUM +
                         r" → \S+ " + NUM + r" \(Δ ([+-]?\d+), листов ↑\d+/↓\d+, p = ([\d.]+)\)", read(p)):
        out[(m.group(1), m.group(3))] = (int(m.group(4)), int(m.group(5)), int(m.group(6)), float(m.group(7)))
        n[m.group(1)] = int(m.group(2))
    return out, n


def strict(p):
    """{часть: {верных|чистых1|чистых5: (a, b, Δ, p), лишних: (a, b), пропусков: (a, b), листов: N}, «только»: (в A, в B)}"""
    out, cur = {}, None
    for ln in read(p).splitlines():
        m = re.search(r"общих листов \d+ \(только в \S+ (\d+), только в \S+ (\d+)\)", ln)
        if m:
            out["только"] = (int(m.group(1)), int(m.group(2))); continue
        m = re.match(r"## (поле\+вне|сорт A) \((\d+) листов\)", ln.strip())
        if m:
            cur = out.setdefault(m.group(1), {}); cur["листов"] = int(m.group(2)); continue
        if cur is None:
            continue
        m = re.search(r"(верных приёмкой|чистых ≤ 1%|чистых ≤ 5%)\s*: " + NUM + r" → " + NUM + r" \(([+-]?\d+), листов ↑\d+/↓\d+, p = ([\d.]+)\)", ln)
        if m:
            k = {"верных приёмкой": "верных", "чистых ≤ 1%": "чистых1", "чистых ≤ 5%": "чистых5"}[m.group(1)]
            cur[k] = (int(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5))); continue
        m = re.search(r"лишних строк\s*: " + NUM + r" → " + NUM + r" .*пропусков " + NUM + r" → " + NUM, ln)
        if m:
            cur["лишних"] = (int(m.group(1)), int(m.group(2))); cur["пропусков"] = (int(m.group(3)), int(m.group(4)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True, choices=["detour", "spike-stand", "spike-ab", "ship"])
    ap.add_argument("--merge", default="")
    ap.add_argument("--strict", required=True)
    ap.add_argument("--n-fo", type=int, default=2062, help="листов поле+вне в полном сравнении (база NSE19)")
    ap.add_argument("--n-a", type=int, default=311, help="листов сорта A")
    a = ap.parse_args()
    try:
        S = strict(a.strict); M_, MN = merge(a.merge) if a.merge else ({}, {})
    except FileNotFoundError as e:
        print(f"⛔ нет лога: {e}"); sys.exit(2)
    # покрытие (08.11): сравнение идёт по ОБЩИМ листам — выпавший лист (падение повтора) не должен молча сузить выборку
    cov = []
    if S.get("только", (0, 0)) != (0, 0) or "только" not in S:
        cov.append(f"строгая: листы только в одном из повторов {S.get('только')}")
    for part, n in (("поле+вне", a.n_fo), ("сорт A", a.n_a)):
        if S.get(part, {}).get("листов") != n:
            cov.append(f"строгая {part}: листов {S.get(part, {}).get('листов')} вместо {n}")
        if a.merge and MN.get(part) != n:
            cov.append(f"приёмка {part}: листов {MN.get(part)} вместо {n}")
    if cov:
        print("⛔ неполное сравнение — критерий не проверяю: " + "; ".join(cov)); sys.exit(2)
    checks = []
    def need(name, ok):
        checks.append((name, bool(ok)))
    try:
        F, A = S["поле+вне"], S["сорт A"]
        if a.rule == "ship":                              # обычное правило отгрузки; строгая мера — справкой
            d = M_[("поле+вне", "именных")]
            need(f"приёмка поле+вне именных Δ ≥ +15 при p < 0.01 ({d[2]:+d}, p = {d[3]})", d[2] >= 15 and d[3] < 0.01)
            need(f"приёмка поле+вне безымянных Δ ≥ 0 ({M_[('поле+вне', 'безымянных')][2]:+d})", M_[("поле+вне", "безымянных")][2] >= 0)
            need(f"приёмка сорт A именных Δ ≥ 0 ({M_[('сорт A', 'именных')][2]:+d})", M_[("сорт A", "именных")][2] >= 0)
            print(f"  справка (строгая): чистых ≤ 5% {F['чистых5'][2]:+d} (p = {F['чистых5'][3]}), ≤ 1% {F['чистых1'][2]:+d}, лишних строк "
                  f"{F['лишних'][0]} → {F['лишних'][1]}; сорт A чистых ≤ 5% {A['чистых5'][2]:+d}")
        if a.rule in ("detour", "spike-ab"):
            need(f"приёмка поле+вне именных Δ ≥ 0 ({M_[('поле+вне', 'именных')][2]:+d})", M_[("поле+вне", "именных")][2] >= 0)
            need(f"приёмка поле+вне безымянных Δ ≥ 0 ({M_[('поле+вне', 'безымянных')][2]:+d})", M_[("поле+вне", "безымянных")][2] >= 0)
            need(f"приёмка сорт A именных Δ ≥ 0 ({M_[('сорт A', 'именных')][2]:+d})", M_[("сорт A", "именных")][2] >= 0)
        if a.rule == "detour":
            need(f"строгая чистых ≤ 5% Δ > 0 при p < 0.01 ({F['чистых5'][2]:+d}, p = {F['чистых5'][3]})", F["чистых5"][2] > 0 and F["чистых5"][3] < 0.01)
            need(f"строгая чистых ≤ 1% Δ ≥ 0 ({F['чистых1'][2]:+d})", F["чистых1"][2] >= 0)
            need(f"лишних строк ↓ ({F['лишних'][0]} → {F['лишних'][1]})", F["лишних"][1] < F["лишних"][0])
            need(f"пропусков не ↑ больше 5% ({F['пропусков'][0]} → {F['пропусков'][1]})", F["пропусков"][1] <= 1.05 * F["пропусков"][0])
        elif a.rule in ("spike-stand", "spike-ab"):
            need(f"поле+вне верных Δ ≥ 0 ({F['верных'][2]:+d})", F["верных"][2] >= 0)
            need(f"поле+вне чистых ≤ 1% Δ ≥ 0 ({F['чистых1'][2]:+d})", F["чистых1"][2] >= 0)
            need(f"поле+вне чистых ≤ 5% Δ ≥ 0 ({F['чистых5'][2]:+d})", F["чистых5"][2] >= 0)
            need(f"лишних строк ↓ ({F['лишних'][0]} → {F['лишних'][1]})", F["лишних"][1] < F["лишних"][0])
            need(f"пропусков не ↑ ({F['пропусков'][0]} → {F['пропусков'][1]})", F["пропусков"][1] <= F["пропусков"][0])
            need(f"сорт A верных Δ ≥ 0 ({A['верных'][2]:+d})", A["верных"][2] >= 0)
            need(f"сорт A чистых ≤ 5% Δ ≥ 0 ({A['чистых5'][2]:+d})", A["чистых5"][2] >= 0)
    except KeyError as e:
        print(f"⛔ лог не разобран, нет пункта {e}"); sys.exit(2)
    for name, ok in checks:
        print(f"  {'✓' if ok else '✗'} {name}")
    if a.rule == "detour" and ("поле+вне", "именных") in M_:
        d = M_[("поле+вне", "именных")]
        print(f"  справка — обычное правило отгрузки (именных Δ ≥ +15 при p < 0.01): {'взято' if d[2] >= 15 and d[3] < 0.01 else 'нет'}")
    ok = all(c for _, c in checks)
    print(f"★ критерий {a.rule}: {'ВЗЯТ' if ok else 'НЕ ВЗЯТ'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

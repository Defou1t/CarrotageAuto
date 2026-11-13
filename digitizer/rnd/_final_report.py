r"""_final_report.py — сводка итоговой оцифровки архива (_final_run.py): листы, падения, кривые по мнемоникам, по скважинам, объём,
счёт мерой приёмки (строки ★ лога `_name_cost_prod.py`). Пишет `<out>/../ОТЧЁТ_ОЦИФРОВКИ.md`.

  _final_report.py --out F:/nds/output/final_2026-10/FINAL --score F:/nds/output/taskS/_final_score.log
"""
import sys, re, argparse, hashlib, os
from collections import Counter, defaultdict
from pathlib import Path


# дефекты исходных данных (§6.352): падение листа — не ошибка оцифровки
KNOWN = {"Melykh_034_1980_01_22_KM_0144_1264_500_D_1.nlgx": "скан — 92 байта `desktop.ini` во всех копиях приёмки; нужен настоящий скан",
         "БВп29_1987.06.25_ПМ.nlgx": "TIFF недописан: нет IFD, данных 2710 строк из ~9500; нужен целый оригинал"}


def sheet_dir(n):
    return f"{n.stem[:40]}_{hashlib.md5(n.stem.encode('utf-8')).hexdigest()[:8]}"


def las_curves(p):
    out, on = [], False
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("~"):
            on = line.upper().startswith("~C")
            if line.upper().startswith("~A"):
                break
            continue
        if on and line.strip() and not line.lstrip().startswith("#"):
            m = line.split(".")[0].strip()
            if m.upper() != "DEPT":
                out.append(m)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
    ap.add_argument("--score", default="")
    a = ap.parse_args()
    out = Path(a.out)
    src = sorted({q for r in a.roots for q in Path(r).glob("*/wlg/*.nlgx")}, key=lambda q: q.name)
    auto = [q.name for q in src if re.search(r"_auto\d*$", q.stem)]       # выдачи автомата в папках архива — не листы
    src = [q for q in src if q.name not in auto]
    st, sec = {}, {}
    for lg in sorted(out.glob("_log_*.tsv")):
        for line in lg.read_text(encoding="utf-8").splitlines():
            p = line.split("\t")
            if len(p) >= 3:
                st[p[0]] = p[2]; sec[p[0]] = float(p[1] or 0)      # последний круг — главный
    cnt = Counter(); mn = Counter(); by_well = defaultdict(lambda: [0, 0, 0]); fails = []; size = 0; ncurv = 0
    for q in src:
        d = out / sheet_dir(q); well = q.parent.parent.name
        by_well[well][0] += 1
        done = d.is_dir() and any(d.glob("*_auto.nlgx"))
        s = st.get(q.name, "")
        if done:
            cnt["ок без las" if s == "ок без las" else "ок"] += 1; by_well[well][1] += 1
            for f in d.iterdir():
                size += f.stat().st_size
            las = next(d.glob("*_auto.las"), None)
            if las:
                c = las_curves(las); ncurv += len(c); by_well[well][2] += len(c)
                mn.update(m.rstrip("0123456789") or m for m in c)
        elif s == "нет скана":
            cnt["нет скана"] += 1
        elif s.startswith("ПАДЕНИЕ"):
            cnt["падение"] += 1; fails.append((q.name, s))
        else:
            cnt["не обработан"] += 1
    L = [f"# Итоговая автоматическая оцифровка архива — {len(src)} листов", ""]
    L += ["| итог | листов |", "|---|---|"] + [f"| {k} | {v} |" for k, v in cnt.most_common()]
    if auto:
        L += ["", f"Не листы, в счёт не входят: {len(auto)} `*_auto*.nlgx` — прежние выдачи автомата в папках wlg архива "
              f"({', '.join(auto)}); не трогались."]
    L += ["", "Сканы с тегом EXIF Orientation (12) и TIFF Бильче-Волыци с незаписанными первыми полосами (37) переделаны 13.11 нынешним "
          "продом с разворотом и починкой чтения везде (§6.352): прежде первые читались неразвёрнутыми, у вторых падали проходы после "
          "ведения. 20 листов с повтором первого слова имени кривой (NGK1 DA1 / NGK1 DA2 …) переделаны после правки записи (§6.364)."]
    ok = cnt["ок"] + cnt["ок без las"]
    t = [sec[k] for k in sec if not st[k].startswith("нет")]
    L += ["", f"Кривых в .las: **{ncurv}** (в среднем {ncurv / max(ok, 1):.1f} на лист). Выдача: {size / 2**30:.1f} ГБ "
          f"(nlgx + bck + las + наложение + json на лист). Время: {sum(t) / 3600:.1f} ч счёта, {sum(t) / max(len(t), 1):.0f} с на лист.", ""]
    L += ["## Мнемоники (по числу кривых)", "", "| мнемоника | кривых |", "|---|---|"] + [f"| {k} | {v} |" for k, v in mn.most_common(25)]
    L += ["", "## По скважинам", "", "| скважина | листов | оцифровано | кривых |", "|---|---|---|---|"]
    L += [f"| {w} | {v[0]} | {v[1]} | {v[2]} |" for w, v in sorted(by_well.items())]
    if fails:
        L += ["", "## Падения", ""] + [f"- `{n}` — {s}" + (f" — **дефект данных:** {KNOWN[n]}" if n in KNOWN else "") for n, s in fails]
    if a.score and Path(a.score).exists():
        sl = [l for l in Path(a.score).read_text(encoding="utf-8", errors="replace").splitlines() if l.startswith("★")]
        L += ["", "## Счёт мерой приёмки (с эталоном архива)", "", "⚠ Модели прода учены на всём корпусе — по листам корпуса число завышено; "
              "честная оценка — A/B вне фолда (база NSE19: поле+вне 3110 именных, сорт A 435).", "", "```"] + sl + ["```"]
    dup = out.parent / "ДУБЛИ_ДЛЯ_ПРОВЕРКИ.tsv"                # §6.364: `_final_dups.py` — флаг ручной проверки
    if dup.exists():
        dr = [l.split("	") for l in dup.read_text(encoding="utf-8").splitlines()[1:] if l.strip()]
        L += ["", "## Дубли трасс — проверить вручную", "",
              f"Пар кривых, чьи трассы совпадают (≤ 3 px) на ≥ 30% строк: **{len(dr)}** на **{len({r[0] for r in dr})}** листах — "
              f"`{dup.name}` (лист, пара, доли строк, глубины). По двум случайным выборкам это главная ошибка автомата: одна кривая пары "
              "легла на линию другой (имя или линия неверны у одной из двух); реже кривые и правда совпадают. Какая из пары неверна, "
              "автомат надёжно не определяет — при проверке смотреть обе."]
    rep = out.parent / "ОТЧЁТ_ОЦИФРОВКИ.md"
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"★ отчёт: {rep}; " + ", ".join(f"{k} {v}" for k, v in cnt.most_common()) + f"; кривых {ncurv}")


if __name__ == "__main__":
    main()

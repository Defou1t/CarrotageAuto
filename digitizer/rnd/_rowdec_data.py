r"""_rowdec_data.py — ВЫБОРКА ДЛЯ ПОСТРОЧНОГО ДЕКОДЕРА (Задача 9, шаг 1; §6.131 + §6.133).

ЧТО ЭТО И ЧЕМ ОТЛИЧАЕТСЯ ОТ `_decoder_seq_data.py`. Тот собирает решения ВЕДЕНИЯ: «трасса стоит
здесь, какой ран взять следующим» — с teacher-forcing по экспертной трассе и инъекцией дрейфа,
потому что учит СЕЛЕКТОР РАНА внутри ведения. Здесь постановка другая и состояния нет вовсе:
на строке трека предсказать x ВСЕХ K кривых сразу. Ошибка такого решения локальна, поэтому
медиана её переживает — §6.133 намерил это в кривых (V0 = 6862 из 6862).

ЧТО КЛАДЁТСЯ НА ДИСК (по листу, .npz):
  band     uint8 [H × Wb]  — «насколько темнее бумаги СВОЕЙ строки», 0..255, полоса трека.
                            ⚠ Не бинарь. §6.133: потолок держит ПОКРЫТИЕ, и прод-бинарь
                            (`dark_v=110`) теряет бледную тушь (V≈123 при бумаге 243). Декодер
                            обязан видеть серое, иначе упрётся в 50% ещё до обучения.
  rgbmed   uint8 [H × Wb × 3] опц. — не пишем: цвет прототипу не нужен (V3 мерился бесцветным).
  ys       int32 [n]        — строки, где эталон есть у ВСЕХ слотов трека (иначе таргет неполон)
  xs       float32 [n × K]  — x каждого слота в координатах ПОЛОСЫ (band), не листа
  slots    список имён слотов трека в порядке рамки

⚠⚠ ПОЛОСА БЕРЁТСЯ ИЗ РАМКИ, А НЕ ИЗ ЭТАЛОНА. `_decoder_seq_data` строит полосу по размаху
экспертной трассы (`xsr.min()-pad … xsr.max()+pad`) — для селектора рана это допустимо, там полоса
на инференсе приходит от линии. Здесь так делать НЕЛЬЗЯ: на инференсе эталона нет, и выборка,
собранная по нему, дала бы обученному декодеру геометрию, которой в проде не существует. Берём
`scale_axes[*].x_left/x_right` — ту же рамку nlgx, которой пользуется прод.

⚠ РАЗБИЕНИЕ ПО СКВАЖИНАМ (§6.87): листы одной скважины делят бланк, почерк и цвета, и модель учит
их, а не правило (замер: по листам +31%, по скважинам +17%). Скважина пишется в манифест, деление
делает обучение, а не этот стенд.
⚠ РАЗМЕР ВЫБОРКИ ПИШЕТСЯ В ЛОГ ПЕРВОЙ СТРОКОЙ (§6.70): «набор собран отдельно» не означает
«модель его не видела», и число листов потом берут ИЗ ЛОГА, а не из функции, которая вернёт что угодно.

  <ComfyUI>\python_embeded\python.exe _rowdec_data.py --shard 0/8 --sheets 0
  <ComfyUI>\python_embeded\python.exe _rowdec_data.py --manifest
"""
import sys, argparse, pickle, json, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict
import numpy as np
from extract_nlgx import extract
from dataset_build import find_image
from auto import imaging as im
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--wlg-roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--out", default=r"F:/nds/output/taskS/rowdec")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--sheets", type=int, default=0, help="ограничить число листов (0 = все); для смоука")
# ★ §6.87: РАЗНООБРАЗИЕ СКВАЖИН ВАЖНЕЕ ОБЪЁМА (300 листов с 37 скважин дали 172 честные пары
# против 79 у 300 листов с 4 скважин). Корпус даёт 2815 треков и 56 млн строк на 200 скважинах —
# это 70+ ГБ растров; для прототипа берём по нескольку листов со скважины, а не всё подряд.
ap.add_argument("--cap-well", type=int, default=0, help="листов на скважину (0 = без ограничения)")
ap.add_argument("--pad", type=int, default=12, help="запас вокруг полосы трека, px")
ap.add_argument("--row-step", type=int, default=1, help="прореживание строк таргета")
ap.add_argument("--manifest", action="store_true")
# ★ §6.210: ПОЛОСА КАК НА ОТГРУЗКЕ. Кэш по осям эталона (`axes`) — вырезка вокруг экспертных слотов;
#   прод режет весь трек U0 `[x_left, x_right]` (`rowdec.py:219`), и там чужой туши больше. Порог
#   пиков 0.2 дал на кэше `axes` +68, на отгрузке −37 — стенд переоценивал расширение кандидатов.
#   `--band u0` берёт границы трека из замороженной `_understanding.json` (`frame.tracks`).
ap.add_argument("--band", default="axes", choices=["axes", "u0"],
                help="axes = объединение scale-осей слотов ± pad (кэш §6.140); u0 = весь трек U0, как на отгрузке")
ap.add_argument("--u0-from", default=r"F:/nds/output/taskS/ab_pregate/G",
                help="каталог замороженной выдачи с `_understanding.json` (frame.tracks) для --band u0")
ap.add_argument("--like", default="", help="взять ТЕ ЖЕ листы, что в манифестах этого кэша (сопоставимость)")
ap.add_argument("--wells-from", default="", help="§6.248: только скважины из манифестов кропов этого каталога (раздача фолдов "
                "`_rowdec_net` — по отсортированному списку скважин; лишняя скважина сдвинула бы фолды и дала утечку)")
a = ap.parse_args()
OUT = Path(a.out)
DELTA = 90.0            # §6.133: тот же относительный порог, что дал V3 (бумага строки − 90)


def sheet_index():
    WELL, WLG = {}, {}
    for root in a.wlg_roots:
        for wlg in Path(root).glob("*/wlg"):
            for q in wlg.glob("*.nlgx"):
                WELL.setdefault(q.name, wlg.parent.name); WLG.setdefault(q.name, q)
    for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
        WELL.setdefault(q.name, "Semeguniv"); WLG.setdefault(q.name, q)
    BY = {q.stem[:60]: q for q in WLG.values()}
    seen, out = set(), []
    for root in a.pools:
        for f in sorted(Path(root).glob("*.pkl")):
            if f.stem in seen or f.stem not in BY:
                continue
            seen.add(f.stem); out.append((f, BY[f.stem], WELL.get(BY[f.stem].name, "?")))
    return out


def sa_suffix(slot_name):
    """'GZ11 DA1 SA1' → 'DA1 SA1' (имя scale-axis в рамке)."""
    m = re.search(r"(DA\d+\s+SA\d+)\s*$", slot_name)
    return m.group(1) if m else None


def u0_tracks(nlgx, img=None, rgb=None):
    """→ [[x_left, x_right], …] трека U0: из замороженной `_understanding.json`, а если листа там
    нет (кэш шире поля 1123) — той же рамкой, что строит прод при `--frame`: `frame_from_nlgx`."""
    import hashlib
    stem = Path(nlgx).stem
    pd = Path(a.u0_from) / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    u = next(iter(sorted(pd.glob("*_understanding.json"))), None) if pd.is_dir() else None
    if u:
        return json.loads(u.read_text(encoding="utf-8"))["frame"]["tracks"]
    if img is None:
        return None
    from auto import frame as frame_mod, meta as meta_mod
    from auto.config import Config
    cfg = Config()
    m = meta_mod.parse_filename(str(img), cfg.mnemonics)
    fr = frame_mod.frame_from_nlgx(str(nlgx), m, cfg.cv, rgb=rgb)
    return [[int(t.x_left), int(t.x_right)] for t in fr.tracks]


def build_sheet(dump, nlgx, img):
    """→ list[dict] по ТРЕКАМ листа."""
    mo = extract(str(nlgx))
    ax = {}
    for s in mo.get("scale_axes", []):
        ax[str(s.get("name", "")).strip()] = (int(s["x_left"]), int(s["x_right"]))
    track = {s["name"]: s["track"] for s in dump["slots"]}
    gts = {nm: g for nm, g in dump["gts"].items() if g and len(g) >= 30}
    if not gts:
        return []

    rgb = im.load_rgb(str(img))
    H, W = rgb.shape[:2]
    u0 = u0_tracks(nlgx, img, rgb) if a.band == "u0" else None
    if a.band == "u0" and not u0:
        return []                                   # рамки нет — лист пропускаем честно
    V = im.value_channel(rgb)
    paper = np.empty(H, np.float32)
    for y0 in range(0, H, 4096):                    # ⚠ память: см. _row_ceiling
        y1 = min(H, y0 + 4096)
        paper[y0:y1] = np.percentile(V[y0:y1, ::4], 90, axis=1)
    # «насколько темнее бумаги своей строки», 0..255 — вход декодера
    # ⚠⚠ СТРУКТУРА ВЫЧИТАЕТСЯ, И ЭТО ИСПРАВЛЕНИЕ. Первая редакция её оставляла, а потолок §6.133
    # считался по маске БЕЗ структуры — то есть модель училась на другом входе, чем тот, на котором
    # получены 86.4%, и сравнивать их было нельзя. Хуже: сетка и рамка попадали в обучение как тушь,
    # и голова вероятности выучила «где вообще тёмное» (замер: верный пик top-1 лишь у 23-32% строк,
    # p(верный) 0.61 против p(лучшего чужого) 0.68).
    dark = np.clip((paper[:, None] - V.astype(np.float32)), 0, 255).astype(np.uint8)
    dark[im.structure_mask(rgb, DEFAULT.cv)] = 0

    by_track = defaultdict(list)
    for nm in gts:
        by_track[track.get(nm)].append(nm)

    out = []
    for t, names in sorted(by_track.items(), key=lambda z: (z[0] is None, z[0])):
        # ── ПОЛОСА ИЗ РАМКИ: объединение scale-axis всех слотов трека ───────────────────────
        spans = [ax[sa_suffix(nm)] for nm in names
                 if sa_suffix(nm) and sa_suffix(nm) in ax]
        if not spans:
            continue                                # рамка не дала полосу — лист пропускаем честно
        if a.band == "u0":
            if t is None or t >= len(u0):
                continue
            lo, hi = max(0, int(u0[t][0])), min(W, int(u0[t][1]) + 1)   # ровно rowdec.py:219
        else:
            lo = max(0, min(s[0] for s in spans) - a.pad)
            hi = min(W, max(s[1] for s in spans) + a.pad + 1)
        if hi - lo < 16:
            continue
        names = sorted(names)                       # порядок слотов — из рамки, а не из картинки
        common = set.intersection(*[set(gts[nm]) for nm in names])
        ys = np.array(sorted(y for y in common if 0 <= y < H)[::a.row_step], np.int32)
        if len(ys) < 100:
            continue
        xs = np.stack([[gts[nm][int(y)] - lo for y in ys] for nm in names], 1).astype(np.float32)
        # ⚠ таргет ВНЕ полосы = рамка и эталон не согласны; такие строки не учим и считаем отдельно
        good = np.all((xs >= 0) & (xs <= hi - lo - 1), axis=1)
        out.append(dict(track=int(t) if t is not None else -1, lo=int(lo), hi=int(hi),
                        band=dark[:, lo:hi], ys=ys[good], xs=xs[good],
                        slots=names, oof=int((~good).sum())))
    return out


def collect(i, n):
    sheets = sheet_index()
    if a.like:
        keep = set()
        for f in sorted(Path(a.like).glob("manifest_*of*.json")):
            keep |= {m["sheet"] for m in json.loads(f.read_text(encoding="utf-8"))}
        sheets = [x for x in sheets if x[1].name in keep]
        print(f"★ ТЕ ЖЕ ЛИСТЫ, ЧТО В {a.like}: {len(sheets)} из {len(keep)} в манифестах")
    if a.wells_from:
        W = set()
        for f in sorted(Path(a.wells_from).glob("man_*of*.json")):
            W |= {t["well"] for t in json.loads(f.read_text(encoding="utf-8"))["tracks"]}
        n0 = len(sheets)
        sheets = [x for x in sheets if x[2] in W]
        print(f"★ ТОЛЬКО СКВАЖИНЫ {a.wells_from} ({len(W)}): листов {len(sheets)} из {n0}")
    if a.cap_well:
        per, keep = defaultdict(int), []
        for f, q, well in sheets:
            if per[well] >= a.cap_well:
                continue
            per[well] += 1; keep.append((f, q, well))
        print(f"★ ОТБОР ПО СКВАЖИНАМ: {len(keep)} листов из {len(sheets)} "
              f"({len(per)} скважин, до {a.cap_well} листов с каждой)")
        sheets = keep
    mine = sheets[len(sheets) * i // n:len(sheets) * (i + 1) // n]
    if a.sheets:
        mine = mine[:a.sheets]
    # ★ §6.70: РАЗМЕР ВЫБОРКИ — ПЕРВОЙ СТРОКОЙ ЛОГА.
    print(f"★ ШАРД {i}/{n}: ЛИСТОВ В ВЫБОРКЕ {len(mine)} (всего в пулах {len(sheets)})")
    OUT.mkdir(parents=True, exist_ok=True)
    man, rows_tot, oof_tot = [], 0, 0
    # ★ §6.248: промежуточный манифест раз в 10 листов и подхват — снятый шард не теряет сделанное
    part = OUT / f"manifest_{i}of{n}.part.json"
    done_sheets = set()
    if part.exists():
        man = json.loads(part.read_text(encoding="utf-8"))
        done_sheets = {m["sheet"] for m in man}
        rows_tot = sum(m["rows"] for m in man); oof_tot = sum(m["oof"] for m in man)
        print(f"★ ПОДХВАТ: листов уже {len(done_sheets)}, треков {len(man)}")
    for k, (f, q, well) in enumerate(mine, 1):
        if q.name in done_sheets:
            continue
        img = find_image(q)
        if not img:
            continue
        try:
            d = pickle.load(open(f, "rb"))
            tracks = build_sheet(d, q, img)
        except Exception as e:
            print(f"  ⚠ {q.name[:50]}: {type(e).__name__}: {str(e)[:60]}")
            continue
        for j, tr in enumerate(tracks):
            p = OUT / f"{q.stem[:60]}__t{tr['track']}.npz"
            np.savez_compressed(p, band=tr["band"], ys=tr["ys"], xs=tr["xs"],
                                lo=tr["lo"], hi=tr["hi"])
            man.append(dict(file=p.name, sheet=q.name, well=well, track=tr["track"],
                            slots=tr["slots"], K=len(tr["slots"]), rows=int(len(tr["ys"])),
                            oof=tr["oof"], lo=tr["lo"], hi=tr["hi"]))
            rows_tot += len(tr["ys"]); oof_tot += tr["oof"]
        if k % 10 == 0 or k == len(mine):
            print(f"  {k}/{len(mine)}  треков {len(man)}, строк {rows_tot:,}")
            tmp = part.with_suffix(".tmp"); tmp.write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8"); tmp.replace(part)
    p = OUT / f"manifest_{i}of{n}.json"
    p.write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
    part.unlink(missing_ok=True)
    print(f"★ готово: треков {len(man)}, строк {rows_tot:,}, вне полосы {oof_tot:,} → {p}")


def manifest():
    fs = sorted(OUT.glob("manifest_*of*.json"))
    if not fs:
        sys.exit("нет манифестов")
    den = fs[0].stem.split("of")[1]
    fs = [f for f in fs if f.stem.endswith("of" + den)]
    man = []
    for f in fs:
        man += json.loads(f.read_text(encoding="utf-8"))
    print(f"шардов {len(fs)} из {den}" + ("  ⚠ НЕПОЛНЫЙ" if len(fs) < int(den) else ""))
    wells = defaultdict(int); kdist = defaultdict(int); rows = 0
    for m in man:
        wells[m["well"]] += 1; kdist[m["K"]] += 1; rows += m["rows"]
    print(f"треков {len(man)}, листов {len({m['sheet'] for m in man})}, "
          f"скважин {len(wells)}, строк-таргетов {rows:,}")
    print("K: " + ", ".join(f"{k}→{v}" for k, v in sorted(kdist.items())))
    oof = sum(m["oof"] for m in man)
    print(f"строк вне полосы рамки (эталон и рамка не согласны): {oof:,} "
          f"({100*oof/max(1,rows+oof):.2f}%)")


if a.manifest:
    manifest()
else:
    i, n = (int(v) for v in a.shard.split("/"))
    collect(i, n)

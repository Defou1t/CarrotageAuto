r"""_emit_traces.py — ЭМИССИЯ K РАЗЛИЧЁННЫХ ТРАСС БЕЗ ЗАДАЧИ ИМЕНОВАНИЯ (постановка §6.33).

Эдуард (22.07): имена определять НЕ нужно — на входе рамка с ПРАВИЛЬНЫМИ именами кривых и
заготовленными цепочками масштабов. Нужно отличать линии между собой и понимать переходы.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ `emit.emit_into_frame` (и зачем отдельный путь):
 1. **Слоты НЕ подбираются по классу/цвету/порядку.** `_map_lines_to_slots` выбирает лучшего
    кандидата в 12% случаев (§6.31) и на выдачу влияет разрушительно. Здесь трассы кладутся в
    слоты ПО ПОРЯДКУ СЛЕВА НАПРАВО — детерминированно, без эвристики. Имя проставит эксперт.
 2. **ВСЕ слоты сначала ОЧИЩАЮТСЯ.** Сегодня выход = побайтовая копия рамки, и у незаполненного
    слота остаётся ЭКСПЕРТНАЯ трасса (ловушка §4) плюс её сегменты 35494/35496/35498 и bbox.
    Здесь чистятся 35490/35492/35494/35496/35498 у КАЖДОЙ kind=7 записи, включая те, что
    `emit` пропускает (корень DA*, второй IFD с тем же токеном).
 3. **Окно строк — НАШЕ.** `emit` берёт `top_y`/`n_rows` ЭКСПЕРТНОЙ кривой (emit.py:269) и режет
    наши точки по чужой разметке (найдено аудитом §6.36). Здесь пишутся 35474/35476/35488 по
    нашей трассе.
 4. **Уровни — нашим декодером** с упором из трассы (§6.41).

★ САМОПРОВЕРКА: после записи файл перечитывается и КАЖДАЯ кривая сверяется с экспертной —
совпадение сетки строк или >50% точных совпадений x помечает результат негодным.

  <ComfyUI>\python_embeded\python.exe _emit_traces.py [--sheet <nlgx>] [--seq]
"""
import sys, io, struct, argparse, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from write_nlgx import read_full, write_full, set_tag, find_ifd
import dataset as ds
import decode_levels as DL
from dataset_build import find_image
from _multi_replica_probe import dense
from _levels_decode2 import decode2

DEF_SHEET = r"F:\nds\projects\Semeguniv_001\wlg\Semeguniv_1_BKZ_3400_3640_200_D1.nlgx"
OUT = Path(r"F:\nds\output\taskS\emit_traces")
LAM, DXF, GW = 0.05, 0.01, 8.0


def _tags(ifd):
    """ifd['entries'] — список [tag,typ,count,raw]; приводим к dict как в write_nlgx.find_ifd."""
    return {t: (typ, c, raw) for t, typ, c, raw in ifd["entries"]}


def curve_ifds(ifds):
    """Индексы ВСЕХ записей kind=7 (Curve trace) + их имя из тега 35470."""
    out = []
    for i, ifd in enumerate(ifds):
        tg = _tags(ifd)
        if 34768 not in tg:
            continue
        try:
            if struct.unpack("<I", tg[34768][2][:4])[0] != 7:
                continue
        except Exception:
            continue
        nm = ""
        if 35470 in tg:
            nm = tg[35470][2].split(b"\x00")[0].decode("latin1")
        out.append((i, nm))
    return out


def emit_traces(frame_nlgx, traces, out_dir, model=None, image=None, stem=None):
    """traces — список dict{row: x}, наши. Кладутся в слоты ПО ПОРЯДКУ СЛЕВА НАПРАВО.
    Возвращает (путь, список (имя_слота, число точек))."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = model or extract(str(frame_nlgx))
    ifds = read_full(open(frame_nlgx, "rb").read())
    if image:
        for i in find_ifd(ifds, lambda tags: 34878 in tags):
            set_tag(ifds, i, 34878, 2, str(image))

    # 1) ОЧИСТКА ВСЕХ kind=7: ни одна экспертная трасса не должна пережить запись
    cif = curve_ifds(ifds)
    for i, nm in cif:
        n_rows = 0
        tg = _tags(ifds[i])
        if 35488 in tg:
            try:
                n_rows = int(struct.unpack("<I", tg[35488][2][:4])[0])
            except Exception:
                n_rows = 0
        set_tag(ifds, i, 35490, 4, [NULL] * max(1, n_rows))
        set_tag(ifds, i, 35492, 4, [0])
        for t in (35494, 35496, 35498):
            set_tag(ifds, i, t, 4, [0])

    # 2) ОТБОР K ТРАСС ИЗ N — критерий `nl+npts` (§6.49). Стенд `_pick_gate.py`,
    # ★ ТРИ НЕЗАВИСИМЫХ НАБОРА ЛИСТОВ, 106 листов / 417 кривых:
    #      набор       листов  кривых  потолок  охват(было)  nl+npts(стало)
    #      гейт            26      90       19        10          16
    #      валидация       20     100       16        12          12
    #      новые 60        60     227       40        24          27
    #      ИТОГО          106     417       75        46          55   (61% → 73% потолка)
    # Улучшено 10 листов, ДЕГРАДИРОВАЛО 2 из 106.
    #
    # ⚠ ПОЧЕМУ НЕ ОХВАТ ГЛУБИНЫ (было до 24.07): латченая трасса ДЛИННЕЕ настоящей ПО ПОСТРОЕНИЮ —
    # она идёт по одной кривой, уходит на соседнюю, и её охват есть сумма двух. «Самая длинная»
    # систематически предпочитает латч.
    #
    # ЧТО ОТЛИЧАЕТ ЛАТЧ ВНУТРЕННЕ (GT не нужен): у него ДВА разных ЧАСТИЧНЫХ согласия с другими
    # кандидатами — по куску с каждой из двух кривых. У настоящей трассы либо один почти полный
    # двойник (норма: линий U1 больше, чем кривых), либо ничего. Берётся ВТОРОЕ по величине
    # согласие среди НЕ-дубликатов; разделённые уходят в конец очереди, внутри групп — по числу
    # точек. ⚠ Сам фильтр в одиночку НЕ помогает (`nolatch` = уровень прода) — работает только
    # связка «фильтр латча + порядок по массе точек».
    #
    # ⚠⚠ ИСТОРИЯ, ЧТОБЫ НЕ ПОВТОРИТЬ: на ОДНОМ гейте (26 листов) этот же критерий давал +7, а на
    # первых 20 свежих листах −1 (§6.48) — правка была откачена и принята заново только после
    # третьего набора. Порядок критериев ЗАВИСИТ ОТ ПОПУЛЯЦИИ ЛИСТОВ; любой кандидат обязан
    # пройти все три набора `_pick_gate.py --cv`.
    slots = [(i, nm) for i, nm in cif if not nm.split()[0].upper().startswith("DA")]
    cand = [t for t in traces if len(t) >= 30]
    med_of = {id(t): float(np.median(list(t.values()))) for t in cand}
    # ⚠ ДВА РАЗНЫХ ПОРОГА, НЕ СКЛЕИВАТЬ: «идут вместе» (согласие) — 20px, «это одна и та же
    # кривая» (слияние) — 50px. Диагностика Pn_Zavoda (`--diag`): три трассы, разъехавшиеся на
    # 150-200px, суть ОДНА кривая, залатченная по-разному, и при пороге 20 они занимали три слота
    # из пяти. Устойчивость на 106 листах: слияние 20/30/40/50/70 → 53/54/54/55/54 честных
    # (охват — 44..46 на всём диапазоне); порог разделённости 0.10/0.15/0.20/0.25/0.35 →
    # 52/54/55/51/51. Выигрыш держится всюду, взяты вершины плато.
    DUP_TOL, MERGE_TOL, SPLIT_THR, DUP_AGREE = 20.0, 50.0, 0.20, 0.85

    def agree(t1, t2):
        """Доля общих строк, где трассы совпадают в пределах DUP_TOL (ЧАСТИЧНОЕ согласие)."""
        com = [y for y in t1 if y in t2]
        if len(com) < 50:
            return 0.0
        return float(np.mean([abs(t1[y] - t2[y]) < DUP_TOL for y in com]))

    def same_curve(t1, t2, tol=MERGE_TOL):
        # ⚠ Не заменять на расстояние медиан: правая тройка листа стоит плотно (1980/2046/2089),
        # и порог по медианам выбрасывал настоящие кривые.
        com = [y for y in t1 if y in t2]
        if len(com) < 50:
            return False
        return float(np.median([abs(t1[y] - t2[y]) for y in com])) < tol

    def split_score(t):
        ps = sorted((p for q in cand if q is not t
                     for p in (agree(t, q),) if p < DUP_AGREE), reverse=True)
        return ps[1] if len(ps) > 1 else 0.0

    sp = {id(t): split_score(t) for t in cand}
    cand.sort(key=lambda t: (sp[id(t)] >= SPLIT_THR, -len(t)))
    picked = []
    for t in cand:
        if len(picked) >= len(slots):
            break
        if not any(same_curve(t, q) for q in picked):
            picked.append(t)
    for t in cand:                          # добор, если различных не хватило
        if len(picked) >= len(slots):
            break
        if all(t is not q for q in picked):
            picked.append(t)
    ours = sorted(picked, key=lambda t: med_of[id(t)])
    fam_by_name = {c["name"]: DL.build_family(model, c) for c in model.get("curves", [])}
    written = []
    for k, tr in enumerate(ours):
        if k >= len(slots):
            break
        i, nm = slots[k]
        rows = sorted(tr)
        y0, y1 = rows[0], rows[-1]
        n = y1 - y0 + 1
        xs = [int(round(tr[y])) if y in tr else NULL for y in range(y0, y1 + 1)]
        # 3) ОКНО СТРОК — НАШЕ (а не экспертное top_y/n_rows, emit.py:269)
        set_tag(ifds, i, 35474, 4, [int(y0)])
        set_tag(ifds, i, 35476, 4, [int(y1)])
        set_tag(ifds, i, 35488, 4, [int(n)])
        set_tag(ifds, i, 35490, 4, xs)
        # 4) уровни нашим декодером; упор из трассы (§6.41)
        fam = fam_by_name.get(nm) or []
        if len(fam) > 1:
            lv = decode2({y: float(tr[y]) for y in rows}, fam, LAM, DXF, GW,
                         auto=True, rail_from_trace=True)
            segs = []
            cur = lv.get(rows[0], 0); s = rows[0]
            for y in rows[1:]:
                k2 = lv.get(y, 0)
                if k2 != cur:
                    segs.append((s, y - 1, cur)); cur = k2; s = y
            segs.append((s, rows[-1], cur))
        else:
            segs = [(y0, y1, 0)]
        set_tag(ifds, i, 35492, 4, [len(segs)])
        set_tag(ifds, i, 35494, 4, [int(a) for a, _, _ in segs])
        set_tag(ifds, i, 35496, 4, [int(b) for _, b, _ in segs])
        set_tag(ifds, i, 35498, 4, [int(c) for _, _, c in segs])
        vx = [x for x in xs if x != NULL]
        if vx:
            set_tag(ifds, i, 35478, 4, [int(min(vx))]); set_tag(ifds, i, 35480, 4, [int(y0)])
            set_tag(ifds, i, 35482, 4, [int(max(vx))]); set_tag(ifds, i, 35484, 4, [int(y1)])
        written.append((nm, len(rows), len(segs)))
    stem = stem or Path(frame_nlgx).stem
    p = out_dir / f"{stem}_traces.nlgx"
    blob = write_full(ifds)                 # write_full возвращает БАЙТЫ, не пишет файл
    p.write_bytes(blob)
    # .bck = точная копия nlgx (write_bck устарел и портит трассу — см. его докстринг)
    (out_dir / f"{stem}_traces.bck").write_bytes(blob)
    return p, written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=DEF_SHEET)
    ap.add_argument("--seq", action="store_true")
    ap.add_argument("--ckpt", default="seq_model_d45p.pt")
    a = ap.parse_args()
    SHEET = Path(a.sheet)

    from auto import confidence as CM, emit as EM, refine as RF, trace2d as T
    _c = CM.classify

    def allauto(sheet, *ar, **kw):
        r = _c(sheet, *ar, **kw)
        for L in sheet.lines:
            L.confidence = "AUTO"
        return r
    CM.classify = allauto
    TR = {}
    _m = EM._map_lines_to_slots

    def cap(t, mo, fr, mn):
        TR["all"] = list(t); return _m(t, mo, fr, mn)
    EM._map_lines_to_slots = cap

    if a.seq:
        import torch
        from _decoder_core import features
        from _decoder_seq import WindowSelector, OUT as MODELS
        from _decoder_seq_data import MAXC, patch
        from auto import imaging as im
        DEV = "cuda" if torch.cuda.is_available() else "cpu"
        NET = WindowSelector().to(DEV)
        NET.load_state_dict(torch.load(MODELS / a.ckpt, map_location=DEV)["sd"]); NET.eval()

        def patched(fg, line, frame, p, band_pad=8, slmax=30.0, wide_run=14, x_range=None,
                    jump_limit=None):
            H, W = fg.shape
            lo = max(0, int(x_range[0])) if x_range is not None else max(0, int(line.x_lo) - band_pad)
            hi = min(W, int(x_range[1]) + 1) if x_range is not None else min(W, int(line.x_hi) + band_pad + 1)
            base = line.x_center
            band = np.ascontiguousarray(fg[:, lo:hi] > 0)
            x = None; v = 0.0; tr = {}
            with torch.no_grad():
                for y in range(max(0, line.y0), min(H, line.y1 + 1)):
                    runs = im.row_runs(fg[y, lo:hi])
                    if not runs:
                        if x is not None:
                            x = x + float(np.clip(v, -slmax, slmax))
                        continue
                    A = np.array([r[0] + lo for r in runs]); B = np.array([r[1] + lo for r in runs])
                    C = np.array([r[2] + lo for r in runs], float)
                    if x is None:
                        k = int(np.argmin(np.abs(C - base))); x = float(C[k]); v = 0.0; tr[y] = x; continue
                    pred = x + float(np.clip(v, -slmax, slmax))
                    idx, X = features(A, B, C, pred, x, v, base, MAXC)
                    if len(idx) == 1:
                        k = int(idx[0])
                    else:
                        ink, val = patch(band, lo, y, pred)
                        pt = torch.from_numpy(np.stack([ink, val]).astype(np.float32))[None].to(DEV)
                        f = np.zeros((1, MAXC, 10), np.float32); f[0, :len(idx)] = X
                        mm = np.zeros((1, MAXC), np.float32); mm[0, :len(idx)] = 1
                        sc = NET(pt, torch.from_numpy(f).to(DEV), torch.from_numpy(mm).to(DEV))
                        k = int(idx[int(sc[0].argmax().item())])
                    nx = float(C[k]); v = 0.6 * v + 0.4 * (nx - x); x = nx; tr[y] = float(nx)
            T._extend_ends(tr, fg, lo, hi, slmax)
            return tr
        T.trace_line = patched

    from auto.pipeline import run as pipe_run
    from auto.config import Config
    img = find_image(SHEET)
    cfg = Config(); cfg.out = OUT / "pipe"
    with contextlib.redirect_stdout(io.StringIO()):
        pipe_run(str(img), frame_nlgx=str(SHEET), cfg=cfg, stages=False)
    ours = [tr for _, tr in TR.get("all", [])]
    print(f"лист: {SHEET.name}\nнаших трасс: {len(ours)}")

    model = extract(str(SHEET))
    p, written = emit_traces(SHEET, ours, OUT, model=model, image=img)
    print(f"записано слотов: {len(written)} -> {p}")
    for nm, npt, nseg in written:
        print(f"   {nm:<18} точек {npt:>6}  сегментов уровня {nseg}")

    # ★ САМОПРОВЕРКА: перечитать и сверить с экспертом
    A = extract(str(p))
    gts = {c["name"]: c for c in ds.real_curves(model) if any(x != NULL for x in c["xs"])}
    bad = 0; kept = 0
    print(f"\n{'слот':<18}{'наших точек':>12}  контроль")
    for c in A.get("curves", []):
        nm = c["name"]
        axs = {c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL}
        if not axs:
            print(f"{nm:<18}{0:>12}  пусто (слот очищен)"); continue
        kept += 1
        g = gts.get(nm)
        note = "✓ наша"
        if g:
            er = np.array([g["top_y"] + i for i, x in enumerate(g["xs"]) if x != NULL])
            ex = np.array([x for x in g["xs"] if x != NULL], float)
            oy = np.array(sorted(axs)); ox = np.array([axs[y] for y in oy], float)
            if len(oy) == len(er) and np.array_equal(oy, er):
                note = "✗ СЕТКА = ЭКСПЕРТ"; bad += 1
            else:
                com = np.intersect1d(oy, er)
                if len(com) >= 30:
                    em = dict(zip(er.tolist(), ex.tolist())); om = dict(zip(oy.tolist(), ox.tolist()))
                    s = np.mean([abs(om[y] - em[y]) < 1e-9 for y in com])
                    if s > 0.5:
                        note = f"✗ совпало x {100*s:.0f}%"; bad += 1
        print(f"{nm:<18}{len(axs):>12}  {note}")
    print(f"\nслотов с данными: {kept}, помечено утечкой: {bad}")
    print("★ ЧИСТО — в файле нет экспертных трасс" if bad == 0 else "⚠⚠ УТЕЧКА")

    # ★ КАЧЕСТВО ВЫДАННОГО ФАЙЛА (эксперт только для СРАВНЕНИЯ): назначение 1-к-1, как §6.36.
    # Имена не проверяются (постановка §6.33) — важно, что K трасс различены и точны.
    GM = {}
    for nm, g in gts.items():
        d = dense(g)
        if len(d) >= 50:
            GM[nm] = {y: d[y] for y in sorted(d)}
    got_tr = {}
    for c in A.get("curves", []):
        axs = {c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL}
        if len(axs) >= 30:
            got_tr[c["name"]] = axs
    pairs = []
    for nm, gm in GM.items():
        for sn, tr in got_tr.items():
            com = [y for y in tr if y in gm]
            if len(com) < 30:
                continue
            dd = np.array([abs(tr[y] - gm[y]) for y in com])
            pairs.append((float(np.median(dd)), len(com) / len(gm), nm, sn))
    pairs.sort(key=lambda q: (q[1] < 0.9, q[0]))
    asg, used = {}, set()
    for md, cv, nm, sn in pairs:
        if nm in asg or sn in used:
            continue
        asg[nm] = (md, cv, sn); used.add(sn)
    print(f"\nКАЧЕСТВО (эксперт только для сравнения, назначение 1-к-1):")
    print(f"{'кривая GT':<18}{'med':>8}{'cov':>7}  легла в слот")
    for nm in GM:
        if nm not in asg:
            print(f"{nm.split()[0]:<18}   не найдена"); continue
        md, cv, sn = asg[nm]
        print(f"{nm.split()[0]:<18}{md:>8.1f}{cv:>7.2f}  {sn.split()[0]}")
    if asg:
        h = sum(1 for md, cv, _ in asg.values() if md <= 3 and cv >= 0.9)
        print(f"★ЧЕСТНЫХ {h}/{len(GM)}   med(med) "
              f"{np.median([v[0] for v in asg.values()]):.1f}px")

r"""_trace_cache.py — КЭШ ТРАСС: полный пайплайн ОДИН раз на лист, дальше правки раскладки/выбора пути — повтором `emit` (§6.218, 25.09).

ЗАЧЕМ (аудит §6.217). Профиль листа: `emit` — 2% времени (2 с из 95), 62% — построчные вызовы селектора. A/B правок,
меняющих только раскладку или выбор пути (§6.206, §6.213 частично, §6.215 целиком), гоняли весь пайплайн ~40 ч.
Кэш хранит ровно то, что `emit_into_frame` получает от ведения: трассы прод-пути, трассы декодера (`.alt`), флаг
предгейта (`.alt_gated`), рамку. Повтор `emit` на кэше — секунды на лист. И второе: это СТЕНД, ТОЖДЕСТВЕННЫЙ
ОТГРУЗКЕ ниже точки кэша — расхождения «пул против файла» (§6.123, §6.141, §6.210) здесь невозможны по построению.

  build  — собрать кэш: пайплайн прод-конфигурации с ДЕКОДЕРОМ НА ВСЕХ ЛИСТАХ (`rowdec_slot_all`, `rowdec_slot_len`
           > 0, чтобы `.alt` был и за предгейтом); `emit` перехвачен и не пишет выдачу. Шарды, партии ≤ 6 ч,
           пропуск готовых (`<кэш>/<каталог листа>.pkl`).
  replay — повторить `emit_into_frame` с ручками режима (`--mode ИМЯ:ключ=зн,…`, ключи ниже) → `<out>/<ИМЯ>/<каталог
           листа>/<скан>_auto.nlgx`, как у `_trace_prod_ab.py` (счёт — тем же `_name_cost_prod.py`).
Ключи режима повтора (только то, что действует ПОСЛЕ ведения): rdpick, rdmodel, kspick, slotlen, slotall, slotgeom,
sib, gate, slot (модель раскладки), order. ⛔ Ключи ведения (kslots, rdpeak, rddir, seq, softfg, …) требуют НОВОГО кэша —
повтор отказывается.

  <ComfyUI>\python_embeded\python.exe _trace_cache.py build --sheets wellmap_sheets.txt --cache F:/nds/output/taskS/tcache --shard 0/4
  <ComfyUI>\python_embeded\python.exe _trace_cache.py replay --sheets wellmap_sheets.txt --cache F:/nds/output/taskS/tcache \
        --out F:/nds/output/taskS/rp_names --mode G: --mode N:slotgeom=1
"""
import sys, io, argparse, contextlib, pickle, hashlib, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from dataset_build import find_image

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["build", "replay"])
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--sheets", required=True, help="список листов (имена nlgx), относительно --ts или абсолютный")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--out", default="")
ap.add_argument("--mode", action="append", default=[])
ap.add_argument("--shard", default="0/1")
ap.add_argument("--max-hours", type=float, default=0.0)
ap.add_argument("--fast", action="store_true", help="сборка с ускоренным селектором `_seq_fast` (выдача побайтно та же, §6.218)")
ap.add_argument("--seq", default="", help="чекпойнт селектора для сборки (пусто = прод `seq_model`); путь с каталогом — как есть (§6.220)")
ap.add_argument("--wlg-roots", nargs="+", default=[r"F:\nds\projects\Archive"])
a = ap.parse_args()
TS = Path(a.ts)
CACHE = Path(a.cache)
T0 = time.time()

# ── конфигурация ВЕДЕНИЯ, под которую собирается кэш (прод + декодер на всех листах) ────────────
WM = str(TS / "rowdec_wellmap.json").replace("\\", "/")
TRACE_KNOBS = dict(row_decoder="auto5", rowdec_wellmap=WM, rowdec_slot_all=True, rowdec_slot_len=0.18)
if a.seq:                                  # ★ §6.220: кэш под ДРУГИМ селектором (ключ ведения, пишется в кэш)
    TRACE_KNOBS["seq_model"] = a.seq


def sheet_dir(n):
    return f"{n.stem[:40]}_{hashlib.md5(n.stem.encode('utf-8')).hexdigest()[:8]}"


def sheet_list():
    p = Path(a.sheets) if Path(a.sheets).is_absolute() else TS / a.sheets
    want = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    src = {}
    for r in a.wlg_roots:
        for q in Path(r).glob("*/wlg/*.nlgx"):
            src.setdefault(q.name, q)
    got = [src[w] for w in want if w in src]
    if len(got) < len(want):
        print(f"⚠ разметки не найдено для {len(want) - len(got)} листов из {len(want)}")
    i, n = map(int, a.shard.split("/"))
    got = sorted(got, key=lambda q: q.name)
    return got[i::n] if n > 1 else got


def pack(tr):
    """{row: x} → (int64 rows в порядке ВСТАВКИ, float64 xs) — порядок и значения восстанавливаются точно"""
    return (np.fromiter(tr.keys(), np.int64, len(tr)), np.fromiter(tr.values(), np.float64, len(tr)))


def unpack(t):
    rows, xs = t
    return dict(zip(rows.tolist(), xs.tolist()))


def cmd_build():
    from auto import pipeline as P, emit as E
    from auto.config import Config
    CACHE.mkdir(parents=True, exist_ok=True)
    grab = {}

    def fake_emit(sheet, traces, out, stem, rgb=None, frame_nlgx=None, mnemonics_path=None, image=None, las=False, cv=None):
        grab["v"] = dict(
            traces=[(L, pack(tr)) for L, tr in traces],
            alt=None if getattr(traces, "alt", None) is None else [(L, pack(tr)) for L, tr in traces.alt],
            alt_gated=bool(getattr(traces, "alt_gated", True)),
            frame=sheet.frame, stem=stem, image=str(image), frame_nlgx=str(frame_nlgx),
            trace_knobs=dict(TRACE_KNOBS))
        return {}
    E.emit = fake_emit                       # pipeline зовёт emit_mod.emit — перехват без правки прод-кода
    if a.fast:
        import _seq_fast
        _seq_fast.install()
    sheets = sheet_list()
    done = skip = fail = perm = 0
    # ⛔ 26.09 (аудит): прежде упавший лист молча выпадал — `build` выходил кодом 0, супервизор считал шард готовым, и лист
    #   не пересчитывался никогда (кэш 1433 из 1434). Теперь: падение пишется в `_failed.txt`; лист, упавший уже ДВАЖДЫ, дальше
    #   пропускается (постоянный); при новых падениях — код 3, супервизор перезапустит, и лист получит вторую попытку.
    FAILED = CACHE / "_failed.txt"
    prev = [l.split("\t")[0] for l in FAILED.read_text(encoding="utf-8").splitlines()] if FAILED.exists() else []
    print(f"★ СБОРКА КЭША: листов в шарде {len(sheets)}, конфигурация ведения {TRACE_KNOBS}")
    for n in sheets:
        dst = CACHE / (sheet_dir(n) + ".pkl")
        if dst.exists():
            skip += 1
            continue
        if prev.count(n.name) >= 2:
            perm += 1
            continue
        img = find_image(n)
        if not img:
            fail += 1; print(f"  {n.stem[:44]} нет скана")
            with open(FAILED, "a", encoding="utf-8") as fh:
                fh.write(f"{n.name}\tнет скана\n")
            continue
        cfg = Config()
        for k, v in TRACE_KNOBS.items():
            setattr(cfg.cv, k, v)
        cfg.out = CACHE / "_scratch"
        grab.clear()
        t = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                P.run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
        except Exception as e:
            fail += 1; print(f"  {n.stem[:44]} ПАДЕНИЕ {type(e).__name__}: {e}")
            with open(FAILED, "a", encoding="utf-8") as fh:
                fh.write(f"{n.name}\t{type(e).__name__}: {str(e)[:200]}\n")
            continue
        tmp = dst.with_suffix(".tmp")
        pickle.dump(grab["v"], open(tmp, "wb"), protocol=4)
        tmp.replace(dst)
        done += 1
        print(f"  {n.stem[:44]:<46} {time.time() - t:5.0f} с  трасс {len(grab['v']['traces'])}, "
              f"декодер {len(grab['v']['alt']) if grab['v']['alt'] is not None else '—'}, гейт {grab['v']['alt_gated']}")
        if a.max_hours and (time.time() - T0) / 3600 > a.max_hours:
            print(f"★ ПАРТИЯ ОКОНЧЕНА ({(time.time() - T0) / 3600:.1f} ч) — выхожу кодом 75"); sys.exit(75)
    print(f"★ ГОТОВО: собрано {done}, уже было {skip}, падений {fail}, постоянных (≥ 2 попыток) {perm}")
    if fail:
        sys.exit(3)


REPLAY_KEYS = {"rdpick": ("rowdec_pick", int), "rdmodel": ("rowdec_pick_model", str), "kspick": ("rowdec_k_slots_pick", str),
               "slotlen": ("rowdec_slot_len", float), "slotall": ("rowdec_slot_all", lambda v: bool(int(v))),
               "slotfill": ("rowdec_slot_fill", lambda v: bool(int(v))),
               "slotgeom": ("slot_template_geom", lambda v: bool(int(v))), "sib": ("slot_sib", float),
               "gate": ("slot_gate", str), "slot": ("slot_model", str), "order": ("slot_order", str)}


def cmd_replay():
    from auto import emit as E, trace2d as T
    from auto.config import Config
    modes = []
    for spec in a.mode:
        nm, _, tail = spec.partition(":")
        kw = {}
        for part in filter(None, tail.split(",")):
            k, _, v = part.partition("=")
            if k not in REPLAY_KEYS:
                sys.exit(f"⛔ режим {nm!r}: ключ {k!r} меняет ВЕДЕНИЕ или не известен — нужен новый кэш, повтор не годится")
            kw[k] = v
        modes.append((nm, kw))
    sheets = sheet_list()
    stat = {nm: [0, 0] for nm, _ in modes}
    for n in sheets:
        src = CACHE / (sheet_dir(n) + ".pkl")
        if not src.exists():
            for nm, _ in modes:
                stat[nm][1] += 1
            continue
        c = pickle.load(open(src, "rb"))
        prod = [(L, unpack(t)) for L, t in c["traces"]]
        alt0 = None if c["alt"] is None else [(L, unpack(t)) for L, t in c["alt"]]
        for nm, kw in modes:
            cfg = Config()
            for k, v in c["trace_knobs"].items():
                setattr(cfg.cv, k, v)
            cfg.cv.rowdec_slot_len, cfg.cv.rowdec_slot_all = 0.0, False      # умолчание режима повтора = прод до §6.213
            cfg.cv.slot_template_geom = False     # ★ 25.09: имена §6.215 включены в прод; повтор без `slotgeom=1` — как A/B до них
            for k, v in kw.items():
                attr, conv = REPLAY_KEYS[k]
                setattr(cfg.cv, attr, conv(v))
            # ── эмуляция решений `trace2d.trace_auto` по кэшу (см. trace2d.py: gated / slot_all / want_both) ──
            p = cfg.cv
            slot_len = float(getattr(p, "rowdec_slot_len", 0.0) or 0.0)
            gated = c["alt_gated"]
            alt = alt0 if (gated or (slot_len > 0 and getattr(p, "rowdec_slot_all", False))) else None
            pick = int(getattr(p, "rowdec_pick", 0) or 0)
            want_both = pick > 0 or bool(getattr(p, "rowdec_pick_model", "") or "") or slot_len > 0
            if alt is not None and not want_both:
                traces = alt
            elif alt is None:
                traces = prod
            else:
                traces = T._WithAlt(prod); traces.alt = alt; traces.alt_gated = gated
            out = Path(a.out) / nm / sheet_dir(n)
            out.mkdir(parents=True, exist_ok=True)
            for old in out.glob("*_auto.nlgx"):
                old.unlink()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    E.emit_into_frame(traces, c["frame_nlgx"], c["frame"], out, c["stem"], cfg.mnemonics,
                                      image=c["image"], las=False, cv=cfg.cv)
                stat[nm][0] += 1
            except Exception as e:
                stat[nm][1] += 1; print(f"  {n.stem[:44]} {nm} ПАДЕНИЕ {type(e).__name__}: {e}")
    for nm, (ok, bad) in stat.items():
        print(f"★ ПОВТОР {nm}: выдано {ok}, пропущено/упало {bad} из {len(sheets)}")


cmd_build() if a.cmd == "build" else cmd_replay()

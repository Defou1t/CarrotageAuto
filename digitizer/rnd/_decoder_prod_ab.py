r"""_decoder_prod_ab.py — ПЕРЕНОСИТСЯ ЛИ ОКОННЫЙ СЕЛЕКТОР (§6.25-§6.27) В ПРОД.

На оракульной полосе стенда bench связка «окно + точка=центр» даёт med(med) 68.3→15.9px,
своя 34.9→45.2%, честных 3→4/24. Но §6.16/§6.20.5 durable: **оракульный успех в прод не
переносится сам по себе** — там полосу обязан найти U1, и на «узкополосном» транше это уже
обнуляло результат (0 честных из 8), а перенос правки ширины дал 882px против 12.9px на оракуле
(разрыв 68×). Поэтому селектор не имеет права идти в `auto/` до этого замера.

Метод — ПРЯМОЙ A/B монкипатчем `trace2d.trace_line`, пакет `auto/` не трогается (как §6.15,
§6.20.5). Проверка утечки эталона (`auto.xs == gt.xs`) обязательна: без неё «идеальные 0.0px»
на незаписанном слоте читаются как успех (ловушка §4). Гейт и метрика — те же, что в
`_relatch_prod_ab.py`, откуда `measure()` и перенесена, чтобы цифры были сравнимы напрямую.

  <ComfyUI>\python_embeded\python.exe _decoder_prod_ab.py [--n 5] [--ckpt seq_model_d45p.pt]
"""
import sys, io, json, argparse, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
import torch
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image
from auto.pipeline import run as pipe_run
from auto.config import Config
from auto import meta as M
from auto import imaging as im
from auto import trace2d as T
from _decoder_core import features
from _decoder_seq import WindowSelector, OUT as MODELS
from _decoder_seq_data import MAXC, patch

MN = r"F:\nds\Auto\mnemonics.json"
ARCH = Path(r"F:\nds\projects\Archive")
OUT = Path(r"F:\nds\output\taskS\prod_ab_seq")
NF = 10

# те же 5 листов, что на стенде bench — иначе перенос не с чем сравнивать
SH = [r"BOGAT_011\wlg\BOGAT_011_BKZ, DS_2800-3190_200_1984-03-02_D_1_B_1.nlgx",
      r"BOGAT_015\wlg\BOGAT_015_BKZ, DS_3300-3700_200_1990-01-08_D_1.nlgx",
      r"LEVEN_023\wlg\LEVEN_023_BKZ, DS_1010-1500_200_1996-12-02_D_1.nlgx",
      r"BEZLUD_051\wlg\BEZLUD_051_RK, AK, DS_2688-3142_500_1998-08-05_D_1.nlgx",
      r"YULIIV_055\wlg\YULIIV_055_MK, MBK, MDS_2084-3060_200_1998-07-18_D_1.nlgx"]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--files", nargs="*", default=None)
ap.add_argument("--ckpt", default="seq_model_d45p.pt")
ap.add_argument("--no-escalate", action="store_true",
                help="заглушить эскалацию полосы до ТРЕКА в refine.refine_trace (refine.py:292-299). "
                     "По §6.29 расширение полосы на сотни px — главный убийца идентичности, а "
                     "эскалация делает ровно это, отменяя годную полосу U1 (§6.30)")
ap.add_argument("--force-auto", action="store_true",
                help="снять FLAG-гейт (§6.15): все линии в AUTO, трассируются все. §6.30: у 96% "
                     "кривых лучшая по полосе линия помечена FLAG и до трассировщика не доходит")
a = ap.parse_args()

ORIG = T.trace_line

if a.no_escalate:
    # Минимальный refine_trace: тот же жёсткий предел и prefer_body, тот же деспайк/транзиты,
    # но БЕЗ ветки «недотяг → перетрасс полосой во весь трек».
    from auto import refine as refine_mod
    _despike, _drop = refine_mod.despike, refine_mod.drop_transits

    def refine_no_escalate(fg, line, frame, p, trace_line_fn, track):
        hlo = getattr(line, "x_hard_lo", None); hhi = getattr(line, "x_hard_hi", None)
        hard = hlo is not None and hhi is not None
        kw = {"wide_run": 10 ** 6} if getattr(line, "prefer_body", False) else {}
        tr = (trace_line_fn(fg, line, frame, p, x_range=(int(hlo), int(hhi)), **kw) if hard
              else trace_line_fn(fg, line, frame, p, **kw))
        if len(tr) < 30:
            return tr
        tr, _ = _despike(tr, win=p.despike_win, k=p.despike_k, min_jump=p.despike_min_jump)
        if getattr(line, "prefer_body", False):
            tr, _ = _drop(tr, min_span=p.transit_min_span)
        return tr
    refine_mod.refine_trace = refine_no_escalate
    print("★ ЭСКАЛАЦИЯ ПОЛОСЫ ДО ТРЕКА ЗАГЛУШЕНА")

if a.force_auto:
    # §6.15 делал ровно это базовым трассировщиком (28/39 слотов, НОЛЬ честных, 20 ложных
    # идентичностей). Связка «без FLAG + ОКОННЫЙ селектор» не проверялась никогда.
    from auto import confidence as confidence_mod
    _classify = confidence_mod.classify

    def classify_all_auto(sheet, *args, **kw):
        r = _classify(sheet, *args, **kw)
        for L in sheet.lines:
            L.confidence = "AUTO"
        return r
    confidence_mod.classify = classify_all_auto
    print("★ FLAG-гейт СНЯТ: все линии в AUTO")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NET = WindowSelector().to(DEV)
NET.load_state_dict(torch.load(MODELS / a.ckpt, map_location=DEV)["sd"])
NET.eval()


def patched(fg, line, frame, p, band_pad=8, slmax=30.0, wide_run=14, x_range=None,
            jump_limit=None):
    """trace_line, где выбор рана делает ОКОННЫЙ СЕЛЕКТОР, а точка в ране = ЦЕНТР (§6.26).
    Тело скопировано с оригинала; отличия помечены ★. Патч и стенд считают признаки и окно
    ОДНИМИ И ТЕМИ ЖЕ функциями (`_decoder_core.features`, `_decoder_seq_data.patch`)."""
    H, W = fg.shape
    if x_range is not None:
        lo = max(0, int(x_range[0])); hi = min(W, int(x_range[1]) + 1)
    else:
        lo = max(0, int(line.x_lo) - band_pad); hi = min(W, int(line.x_hi) + band_pad + 1)
    base = line.x_center
    band = np.ascontiguousarray(fg[:, lo:hi] > 0)      # ★ растр полосы для окна
    x = None; v = 0.0
    tr = {}
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
                k = int(np.argmin(np.abs(C - base)))
                x = float(C[k]); v = 0.0; tr[y] = x; continue
            pred = x + float(np.clip(v, -slmax, slmax))
            idx, X = features(A, B, C, pred, x, v, base, MAXC)
            if len(idx) == 1:
                k = int(idx[0])
            else:
                ink, val = patch(band, lo, y, pred)      # ★ окно ±64 строки
                pt = torch.from_numpy(np.stack([ink, val]).astype(np.float32))[None].to(DEV)
                f = np.zeros((1, MAXC, NF), np.float32); f[0, :len(idx)] = X
                m = np.zeros((1, MAXC), np.float32); m[0, :len(idx)] = 1
                sc = NET(pt, torch.from_numpy(f).to(DEV), torch.from_numpy(m).to(DEV))
                k = int(idx[int(sc[0].argmax().item())])
            nx = float(C[k])                             # ★ точка = ЦЕНТР рана (§6.26)
            v = 0.6 * v + 0.4 * (nx - x); x = nx; tr[y] = float(nx)
    T._extend_ends(tr, fg, lo, hi, slmax)
    return tr


def measure(tag, sheets):
    """ПЕРЕНЕСЕНО ИЗ `_relatch_prod_ab.py` без изменений — метрика обязана совпадать."""
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    rows, per_sheet = [], []
    for well, n, img, m, gts in sheets:
        cfg = Config(); cfg.out = out
        # ⚠⚠ РЕЖИМ ЗАДАЁТ СТЕНД, А НЕ ПРОД-УМОЛЧАНИЕ (§6.71). С 26.07 `seq_model` включён по
        # умолчанию, поэтому `Config()` пустил бы селектор и в ветку A — A/B показал бы ноль
        # разницы по причине, не имеющей отношения к делу. Здесь селектор включается ТОЛЬКО
        # монкипатчем `T.trace_line` ниже.
        cfg.cv.seq_model = ""
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                sheet, traces, res = pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
        except Exception as e:
            print(f"  {well:<12} ERR {type(e).__name__}: {e}")
            per_sheet.append({"well": well, "err": str(e)}); continue
        aup = Path(res.get("nlgx", ""))
        rec = {"well": well, "n_lines": len(sheet.lines), "n_gt": len(gts),
               "written": res.get("written"), "curves": []}
        if not aup.is_file():
            rec["err"] = "нет nlgx"; per_sheet.append(rec); continue
        A = extract(str(aup))
        byname = {c["name"]: c for c in A["curves"]}
        for g in gts:
            gxs = {g["top_y"] + i: xx for i, xx in enumerate(g["xs"]) if xx != NULL}
            nm = g["name"].split()[0]
            ac = byname.get(g["name"])
            if ac is None:
                rec["curves"].append({"name": nm, "state": "missing"}); continue
            if ac["xs"] == g["xs"]:          # ЛОВУШКА §4: эталон не перезаписан
                rec["curves"].append({"name": nm, "state": "leak"}); continue
            axs = {ac["top_y"] + i: xx for i, xx in enumerate(ac["xs"]) if xx != NULL}
            common = [yy for yy in axs if yy in gxs]
            if not common:
                rec["curves"].append({"name": nm, "state": "cov0"}); continue
            d = np.array([abs(axs[yy] - gxs[yy]) for yy in common], float)
            r = {"name": nm, "state": "ok", "cov": round(len(common) / max(1, len(gxs)), 3),
                 "med": round(float(np.median(d)), 1), "p3": round(float((d <= 3).mean() * 100))}
            rec["curves"].append(r); rows.append(r)
        per_sheet.append(rec)
        st = " ".join(f"{c['name']}:{c.get('med','-')}" for c in rec["curves"])
        print(f"  {well:<12} lines={rec['n_lines']:<4} written={rec['written']}  {st}")
    ok = [r for r in rows if r["state"] == "ok"]
    honest = [r for r in ok if r["med"] <= 3 and r["cov"] >= 0.9]
    allc = sum(len(s.get("curves", [])) for s in per_sheet)
    miss = sum(1 for s in per_sheet for c in s.get("curves", []) if c["state"] == "missing")
    leak = sum(1 for s in per_sheet for c in s.get("curves", []) if c["state"] == "leak")
    summ = {"tag": tag, "slots": allc, "measured": len(ok), "missing": miss, "leak": leak,
            "med_med": float(np.median([r["med"] for r in ok])) if ok else None,
            "med_cov": float(np.median([r["cov"] for r in ok])) if ok else None,
            "honest": len(honest), "sheets": per_sheet}
    print(f"  --- {tag}: слотов {allc}, измерено {len(ok)}, не записано {miss}, leak {leak}")
    if ok:
        print(f"      med(med) {summ['med_med']:.1f}px  med(cov) {summ['med_cov']:.2f}  "
              f"★ЧЕСТНЫХ (med<=3 И cov>=0.9) {len(honest)}/{len(ok)}")
    return summ


files = [Path(f) for f in a.files] if a.files else [ARCH / s for s in SH][:a.n]
sheets = []
for n in files:
    img = find_image(n)
    if not img:
        print(f"нет картинки: {n.name}"); continue
    m = M.parse_filename(n.name, MN)
    mo = extract(str(n))
    gts = [c for c in ds.real_curves(mo) if any(xx != NULL for xx in c["xs"])]
    sheets.append((n.parent.parent.name, n, img, m, gts))
print(f"листов: {len(sheets)}, модель {a.ckpt}, устройство {DEV}\n")

print("=== A) БАЗА (прод как есть) ===")
A_ = measure("A_base", sheets)
print("\n=== B) ОКОННЫЙ СЕЛЕКТОР + точка=центр ===")
T.trace_line = patched
B_ = measure("B_seq", sheets)
T.trace_line = ORIG

print("\n" + "=" * 78)
print(f"{'':<24}{'слотов':>8}{'измер':>7}{'не зап':>8}{'leak':>6}{'med(med)':>10}{'med(cov)':>10}{'ЧЕСТНЫХ':>9}")
for s in (A_, B_):
    print(f"{s['tag']:<24}{s['slots']:>8}{s['measured']:>7}{s['missing']:>8}{s['leak']:>6}"
          f"{(s['med_med'] if s['med_med'] is not None else float('nan')):>10.1f}"
          f"{(s['med_cov'] if s['med_cov'] is not None else float('nan')):>10.2f}"
          f"{s['honest']:>9}")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "prod_ab_seq.json").write_text(json.dumps({"A": A_, "B": B_}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
print("\nЧитать так: если ЧЕСТНЫХ и «не записано» не сдвинулись — оракульный выигрыш НЕ")
print("переносится (как §6.16/§6.20.5), и узкое место выше по конвейеру: U1 не даёт полосу.")

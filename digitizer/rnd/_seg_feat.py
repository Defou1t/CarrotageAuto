r"""_seg_feat.py — СЕТЬ ТУШИ КАК ПРИЗНАК КАНДИДАТА ВЫБОРА ТРАССЫ (29.10, §6.348).

Для каждого листа скана `_line_choice_scan.py` (`--data`) — кандидаты в том же порядке, что `_trace_patch.py` (прод `traces`, декодер
`alt`, доп. пути `--extra` по порядку). Карта сети туши (`auto.ink_seg`, вне фолда: `--seg` — каталог с `folds.json` и `f{k}.pt`,
скважина листа не в обучении; `all.pt` — для скважин вне фолдов) по рамке всех кандидатов; скан — с `row_shift`, как `redec`.
По кандидату (строки через `--step`): v = max карты в ±1 px от x; признаки — среднее v и доля строк с v > 0.5.
Пишет `--out` — {(лист, k): (среднее, доля)}. §6.350 `--fam` (сеть по семействам, `_seg_net.py --fam`): {(лист, k): float16 [1 + K]} —
среднее v по каждому каналу (тушь, затем семейства `_seg_patch.FAMS`); слот выбирает свой канал в стенде.

  _seg_feat.py --data F:/nds/output/taskS/line_choice_s --cache F:/nds/output/taskS/tcache
      --extra "…|alt;…|extra" --seg F:/nds/output/taskS/ink_seg_f --out F:/nds/output/taskS/segfeat_f.pkl --shard 0/2
"""
import sys, argparse, pickle, json, time
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np



def _save(R, outp):
    """08.11 (отключения питания): запись через временный файл и подмену — оборванная запись не портит готовое"""
    tmp = outp.with_name(outp.name + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(R, fh, protocol=4)
    tmp.replace(outp)

def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--seg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--fam", action="store_true")
    a = ap.parse_args()
    import hashlib
    from _cand_ceiling import dense_tr
    from auto import imaging as im, frame as FR, ink_seg as IS
    si, sn = map(int, a.shard.split("/"))
    FO = json.loads((Path(a.seg) / "folds.json").read_text(encoding="utf-8"))
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    files = sorted(Path(a.data).glob("*.pkl"))[si::sn]
    outp = Path(a.out)
    R = {}
    if outp.exists():
        try:
            R = pickle.load(open(outp, "rb"))
        except Exception:                                 # 08.11: оборван отключением питания — начать заново, файл не терять
            outp.replace(outp.with_name(outp.name + f".bad_{int(time.time())}"))
    done = {sh for sh, _ in R}
    t0 = time.time(); n = 0
    for f in files:
        rec = pickle.load(open(f, "rb")); sh = rec["sheet"]
        n += 1
        if sh in done or sh not in SRC:
            continue
        q = SRC[sh]
        key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
        tc = Path(a.cache) / (key + ".pkl")
        if not tc.exists():
            continue
        c = pickle.load(open(tc, "rb"))
        xsrc = []
        for spec in a.extra.split(";") if a.extra else []:
            dd, _, fld = spec.partition("|")
            xp = Path(dd) / tc.name
            if xp.exists():
                xsrc += pickle.load(open(xp, "rb")).get(fld or "alt") or []
        DD = []
        try:
            for cs in rec["cands"]:
                t = c["traces"][cs["idx"]][1] if cs["src"] == "prod" else (c["alt"][cs["idx"]][1] if cs["idx"] < 1000 else xsrc[cs["idx"] - 1000][1])
                DD.append(dense_tr(un(t)))
            rgb = im.load_rgb(c["image"])
        except (OSError, IndexError) as e:
            print(f"  ПРОПУСК {sh[:50]}: {type(e).__name__}"); continue
        if getattr(c["frame"], "row_shift", None) is not None:
            rgb = FR.apply_row_shift(rgb, c["frame"].row_shift)
        k_ = FO.get(Path(c["frame_nlgx"]).parent.parent.name)
        net = str(Path(a.seg) / (f"f{k_}.pt" if k_ is not None else "all.pt"))
        ys_all = [y for d in DD for y in d]; xs_all = [x for d in DD for x in d.values()]
        if not ys_all:
            continue
        y0, y1 = int(min(ys_all)), int(max(ys_all)) + 1
        x0, x1 = int(min(xs_all)) - 4, int(max(xs_all)) + 5
        H, W = rgb.shape[:2]
        y0, x0 = max(0, y0), max(0, x0); y1, x1 = min(H, y1), min(W, x1)
        # ★ 03.11: карта — полосами по сетке плиток prob_map (шаг плитки × 8 строк): плитки независимы, значения бит-в-бит, а
        #   память ∝ полосе. Целиком карта по семействам [1 + K, h, w] float32 на листе в 100+ тыс. строк — > 22 ГБ (ro0 §6.350).
        PR = []
        for d in DD:
            ys = np.array(sorted(d))[::a.step]
            ys = ys[(ys >= y0) & (ys < y1)]
            PR.append((ys, np.round([d[int(y)] for y in ys]).astype(int) - x0 if len(ys) else np.zeros(0, int)))
        VAL = [[] for _ in DD]; C = None; band = (1024 - 2 * 64) * 8
        for s0 in range(y0, y1, band):
            S = IS.prob_map(rgb, s0, min(y1, s0 + band), x0, x1, net)
            C = S.shape[0] if a.fam else None
            for k, (ys, xs) in enumerate(PR):
                m = (ys >= s0) & (ys < s0 + band)
                if len(ys) < 10 or not m.any():
                    continue
                yy, xx = ys[m] - s0, xs[m]
                if a.fam:                                 # [1 + K, h, w]: max в ±1 px — по каждому каналу
                    VAL[k].append(np.max(np.stack([S[:, yy, np.clip(xx + o, 0, S.shape[2] - 1)] for o in (-1, 0, 1)]), 0))
                else:
                    VAL[k].append(np.max(np.stack([S[yy, np.clip(xx + o, 0, S.shape[1] - 1)] for o in (-1, 0, 1)]), 0))
            del S
        for k, (ys, xs) in enumerate(PR):
            if len(ys) < 10:
                R[(sh, k)] = np.full(C, np.nan, np.float16) if a.fam else (np.nan, np.nan); continue
            v = np.concatenate(VAL[k], axis=-1)
            if a.fam:                                     # среднее по строкам — по каждому каналу
                R[(sh, k)] = v.mean(1).astype(np.float16); continue
            R[(sh, k)] = (float(v.mean()), float((v > 0.5).mean()))
        if n % 50 == 0:
            _save(R, outp)
            print(f"  {n}/{len(files)} листов, {time.time() - t0:.0f} с")
    _save(R, outp)
    print(f"★ готово: {len(files)} листов, кандидатов {len(R)} → {outp}")


if __name__ == "__main__":
    main()

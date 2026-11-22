r"""_trk.py — §6.387 ОБУЧАЕМЫЙ ТРЕКЕР ЛИНИИ («веди эту линию») (22.11).

§6.372–6.382: у слотов без честного кандидата линия видна сетью туши, но ни Витерби по её карте, ни сшивание пиков её не проводят
(K 10.9%, потолок пиков 16.8%). Здесь — сеть, которая по кадру скана вокруг текущей точки и уже пройденному пути предсказывает x линии на
16 строк вперёд. Кадр: строки [y − 32, y + 32) × столбцы [x − 96, x + 96) (RGB) + канал пройденного пути (строки ≤ y); цель — столбец кадра
линии эксперта в строках y + 1 … y + 16 (192 класса). Обучение — все листы стенда (поле + вне) КРОМЕ скважин оценочных листов
(`taskS/segpath_sheets60.txt`, §6.372); координаты — nlgx исходного скана (без row_shift, без разворота EXIF).

  _trk.py data  [--per 60]          → taskS/trk/data_NN.npz (продолжаемо по листам)
  _trk.py train [--epochs 6]        → taskS/trk/trk.pt
  _trk.py eval                      → слоты оценочных листов: честный ли путь трекера с затравкой у начала линии эксперта
"""
import sys, argparse, pickle, json, hashlib, time, random
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
OUT = TS / "trk"
UP, DN, HW, FUT = 32, 32, 96, 16           # кадр: 32 строки вверх, 32 вниз, ±96 столбцов; 16 строк вперёд


def sheet_dir(n):
    return f"{n.stem[:40]}_{hashlib.md5(n.stem.encode('utf-8')).hexdigest()[:8]}"


def src_of(n):
    return next((TS / d / (sheet_dir(n) + ".pkl") for d in ("tcache", "tcache_out") if (TS / d / (sheet_dir(n) + ".pkl")).exists()), None)


def crop(rgb, path, y, x):
    """→ (uint8 [64, 192, 4]) — RGB + пройденный путь (строки ≤ y) в кадре вокруг (y, x); вне скана — белое"""
    H, W = rgb.shape[:2]
    out = np.full((UP + DN, 2 * HW, 4), 255, np.uint8); out[..., 3] = 0
    y0, x0 = y - UP, int(round(x)) - HW
    a0, a1 = max(0, y0), min(H, y0 + UP + DN); b0, b1 = max(0, x0), min(W, x0 + 2 * HW)
    if a1 > a0 and b1 > b0:
        out[a0 - y0:a1 - y0, b0 - x0:b1 - x0, :3] = rgb[a0:a1, b0:b1]
    for yy in range(y0, y + 1):
        if yy in path:
            c = int(round(path[yy])) - x0
            if 0 <= c < 2 * HW:
                out[yy - y0, c, 3] = 255
    return out


def eval_wells():
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    ev = [l.strip() for l in (TS / "segpath_sheets60.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {SRC[s].parent.parent.name for s in ev if s in SRC}, ev, SRC


def cmd_data(a):
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from auto import imaging as im
    im.EXIF_TRANSPOSE = False
    DD = OUT / a.data_dir if a.data_dir else OUT
    DD.mkdir(parents=True, exist_ok=True)
    wells, _, SRC = eval_wells()
    names = []
    for f in ("tcache_sheets.txt", "outside_sheets.txt"):
        names += [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    names = sorted(set(names)); random.Random(1).shuffle(names)
    names = [s for s in names if s in SRC and SRC[s].parent.parent.name not in wells]
    done_f = DD / "data_done.txt"
    done = set(done_f.read_text(encoding="utf-8").splitlines()) if done_f.exists() else set()
    rng = np.random.default_rng(7)
    X, T, buf_sheets, shard = [], [], [], len(list(DD.glob("data_*.npz")))
    T0 = time.time()

    def flush():
        nonlocal X, T, buf_sheets, shard
        if not X:
            return
        tmp = DD / f"data_{shard:03d}.tmp.npz"
        np.savez(tmp, X=np.stack(X), T=np.stack(T))
        tmp.replace(DD / f"data_{shard:03d}.npz")
        with open(done_f, "a", encoding="utf-8") as fh:
            fh.write("\n".join(buf_sheets) + "\n")
        shard += 1; X, T, buf_sheets = [], [], []

    nproc = 0
    for nm in names:
        if nm in done:
            continue
        if a.limit and nproc >= a.limit:
            break
        nproc += 1
        src = src_of(SRC[nm])
        if src is None:
            continue
        try:
            m = extract(str(SRC[nm]))
            rgb = im.load_rgb(pickle.load(open(src, "rb"))["image"])
        except Exception as e:
            print(f"  ПРОПУСК {nm[:40]}: {type(e).__name__}"); continue
        H = rgb.shape[0]
        cand = []
        for c in m["curves"]:
            tr = dense(c)
            if len(tr) < 500:
                continue
            ys = np.array(sorted(tr))
            ys = ys[(ys - UP >= ys[0]) & (ys + FUT <= ys[-1]) & (ys + DN < H)]
            cand += [(tr, int(y)) for y in ys[::40]]
        if cand:
            for i in rng.choice(len(cand), size=min(a.per, len(cand)), replace=False):
                tr, y = cand[i]
                if not all((y + t) in tr for t in range(1, FUT + 1)):
                    continue
                x = tr[y]
                X.append(crop(rgb, tr, y, x))
                T.append(np.array([int(np.clip(round(tr[y + t] - x) + HW, 0, 2 * HW - 1)) for t in range(1, FUT + 1)], np.int16))
        buf_sheets.append(nm)
        del rgb
        if len(X) >= 6000:
            flush()
            print(f"  шард {shard}, листов в этом запуске {nproc}, {time.time() - T0:.0f} с")
    flush()
    print(f"★ DATA: шардов {shard}, {time.time() - T0:.0f} с")


def net_def():
    import torch, torch.nn as nn

    class Trk(nn.Module):
        def __init__(s, ch=48):
            super().__init__()
            def blk(i, o, st=(1, 1)):
                return nn.Sequential(nn.Conv2d(i, o, 3, st, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
            s.enc = nn.Sequential(blk(4, ch), blk(ch, ch), blk(ch, ch, (2, 1)), blk(ch, 2 * ch), blk(2 * ch, 2 * ch, (2, 1)),
                                  blk(2 * ch, 2 * ch), nn.Conv2d(2 * ch, 2 * ch, (1, 9), 1, (0, 4)), nn.ReLU(inplace=True),
                                  nn.Conv2d(2 * ch, 2 * ch, (1, 9), 1, (0, 8), dilation=(1, 2)), nn.ReLU(inplace=True))
            s.head = nn.Conv2d(2 * ch, 1, 1)

        def forward(s, x):                       # x [B, 4, 64, 192] → logits [B, 16, 192]
            return s.head(s.enc(x))[:, 0]
    return Trk


def cmd_train(a):
    import torch, torch.nn.functional as F
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted((OUT / a.data_dir).glob("data_[0-9][0-9][0-9].npz")) if a.data_dir else sorted(OUT.glob("data_[0-9][0-9][0-9].npz"))
    D = [np.load(f) for f in files]
    X = np.concatenate([d["X"] for d in D]); T = np.concatenate([d["T"] for d in D])
    print(f"  примеров {len(X)} из {len(files)} шардов")
    net = net_def()().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = a.epochs * (len(X) // a.batch)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(0); T0 = time.time(); run = None
    for st in range(steps):
        idx = rng.integers(0, len(X), a.batch)
        xb = torch.from_numpy(X[idx]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        if rng.random() < 0.5:                   # отражение по x — цель отражается тоже
            xb = torch.flip(xb, dims=[3]); tb = torch.from_numpy(2 * HW - 1 - T[idx].astype(np.int64)).to(dev)
        else:
            tb = torch.from_numpy(T[idx].astype(np.int64)).to(dev)
        if a.jitter:                             # §6.387 (2): против дрейфа — кадр не по центру линии и сдвинутая история
            d1 = int(rng.integers(-a.jitter, a.jitter + 1))
            if d1:
                xb = torch.roll(xb, d1, dims=3); tb = (tb + d1).clamp(0, 2 * HW - 1)
            if rng.random() < 0.5:
                d2 = int(rng.integers(-a.jitter, a.jitter + 1))
                xb[:, 3] = torch.roll(xb[:, 3], d2, dims=2)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lo = net(xb)
        loss = F.cross_entropy(lo.float().reshape(-1, 2 * HW), tb.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        run = float(loss) if run is None else 0.98 * run + 0.02 * float(loss)
        if (st + 1) % 2000 == 0:
            acc = float((lo.argmax(-1) - tb).abs().le(3).float().mean())
            print(f"  шаг {st + 1}/{steps}: потеря {run:.3f}, |Δ| ≤ 3 px {acc:.1%}, {time.time() - T0:.0f} с")
            torch.save(net.state_dict(), OUT / f"trk{a.tag}.tmp.pt"); (OUT / f"trk{a.tag}.tmp.pt").replace(OUT / f"trk{a.tag}.pt")
    torch.save(net.state_dict(), OUT / f"trk{a.tag}.tmp.pt"); (OUT / f"trk{a.tag}.tmp.pt").replace(OUT / f"trk{a.tag}.pt")
    print(f"★ TRAIN: {steps} шагов, {time.time() - T0:.0f} с")


def cmd_eval(a):
    import torch
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from _cand_ceiling import _tree, dense_tr
    from _line_choice_scan import honest_d
    from auto import imaging as im
    im.EXIF_TRANSPOSE = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net_def()().to(dev); net.load_state_dict(torch.load(OUT / f"trk{a.tag}.pt", map_location=dev)); net.eval()
    _, ev, SRC = eval_wells()
    lab = {}
    for d in ("line_choice_r", "line_choice_out_r"):
        for f in (TS / d).glob("*.pkl"):
            r = pickle.load(open(f, "rb"))
            lab[r["sheet"]] = {s["name"]: any(x.get("lab") and x["lab"][0] for x in s["cands"]) for s in r["slots"] if s.get("has_truth")}
    res = {False: [0, 0], True: [0, 0]}; T0 = time.time()
    jf = OUT / f"eval{a.tag}.jsonl"; jf.unlink(missing_ok=True)
    for nm in ev:
        if nm not in lab or nm not in SRC:
            continue
        src = src_of(SRC[nm])
        if src is None:
            continue
        rgb = im.load_rgb(pickle.load(open(src, "rb"))["image"])
        C = {c["name"]: c for c in extract(str(SRC[nm]))["curves"]}
        for cn, has in lab[nm].items():
            if cn not in C:
                continue
            g = dense(C[cn])
            if len(g) < 200:
                continue
            ys = sorted(g); y = ys[0] + UP; yend = ys[-1]
            path = {yy: g[yy] for yy in range(ys[0], y + 1) if yy in g}            # затравка: первые 32 строки эксперта
            if not path:
                continue
            y = max(path)                                                         # у эксперта бывают разрывы строк
            with torch.no_grad():
                while y < yend:
                    xb = torch.from_numpy(crop(rgb, path, y, path[y])[None]).to(dev).permute(0, 3, 1, 2).float() / 255.0
                    pr = net(xb)[0].argmax(-1).cpu().numpy()
                    x0 = int(round(path[y])) - HW
                    for t in range(1, a.step + 1):
                        if y + t > yend:
                            break
                        path[y + t] = float(x0 + pr[t - 1])
                    y += a.step
            P = dense_tr(path)
            hit, cov, md = honest_d(P, _tree(P), g, _tree(g))
            res[has][0] += 1; res[has][1] += int(hit)
            with open(jf, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(sheet=nm, curve=cn, has=has, hit=bool(hit), cov=round(cov, 3), med=round(md, 2)), ensure_ascii=False) + "\n")
        del rgb
    for has, nmh in ((False, "без честного кандидата"), (True, "с кандидатом")):
        n, h = res[has]
        print(f"★ слоты {nmh}: трекер (затравка — 32 строки эксперта, шаг {a.step}) честен у {h}/{n} = {h / max(n, 1):.1%}")
    print(f"  {time.time() - T0:.0f} с")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "train", "eval"])
    ap.add_argument("--per", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jitter", type=int, default=0, help="§6.387 (2): сдвиги кадра / истории при обучении, px")
    ap.add_argument("--tag", default="", help="суффикс файлов модели и оценки")
    ap.add_argument("--data-dir", default="", help="подкаталог trk с данными")
    a = ap.parse_args()
    {"data": cmd_data, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()

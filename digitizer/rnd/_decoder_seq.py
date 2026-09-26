r"""_decoder_seq.py — ОКОННЫЙ СЕЛЕКТОР РАНОВ на torch + ЧЕСТНЫЙ ГЕЙТ на держанных скважинах.

Шаг после §6.24. Там селектор видел ОДНУ строку (10 признаков + poly) и упёрся: med 34.5,
честных 3/24, все три — узкополосные. Здесь решение на строке опирается на ОКНО ±64 строки:
CNN сворачивает патч маски чернил в ПОКОЛОНОЧНОЕ представление «идёт ли через эту колонку
согласованный ход», кандидат берёт эмбеддинг СВОЕЙ колонки и добавляет к нему те же 10
признаков строки. Тем самым добавлен ровно тот сигнал, которого §6.24 не хватало, и ничего
из работавшего не выброшено.

Обучение — listwise: softmax по кандидатам решения, цель — ран, накрывший точку эксперта
(мульти-позитив: logsumexp по правильным). Это точнее бинарной логистики §6.24: гейт на
инференсе — argmax внутри строки, обучение теперь той же формы.

★ ГЕЙТ НЕ МЕНЯЛСЯ: `_relatch_bench` (5 держанных скважин, 24 кривые), метрика — ЧЕСТНЫЕ
(med<=3px И cov>=0.9), med(med), своя%. База прод 3/24 / 68.3 / 34.9; §6.24 poly 3/24 / 34.5 / 36.4.

  <ComfyUI>\python_embeded\python.exe _decoder_seq.py --data seq_train.npz --epochs 6
  ...                                  _decoder_seq.py --gate-only        # только прогон гейта
"""
import sys, argparse, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import _relatch_bench as BE
from _decoder_core import features
from _decoder_seq_data import (NROW, NCOL, MAXC, R_COLS, COL_STEP, patch, unpack, PBYTES)

OUT = Path(r"F:\nds\output\taskS\decoder")
NF = 10
CKPT = OUT / "seq_model.pt"          # переопределяется --ckpt (варианты дрейфа обучаются параллельно)


# ───────────────────────────── модель ─────────────────────────────

class WindowSelector(nn.Module):
    """Патч (2×NROW×NCOL) → поколоночный эмбеддинг → скор кандидата в его колонке + 10 признаков.

    Свёртки НЕ пулят по колонкам: колонка = координата x, её разрешение и есть то, что
    отличает свою кривую от соседней. Пулинг только по строкам (окно сворачивается в
    «согласованность хода»)."""

    def __init__(self, ch=32, emb=48, ctx=3):
        super().__init__()
        self.ctx = ctx                                   # сколько соседних колонок берёт кандидат
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, ch, 3, stride=(2, 1), padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, stride=(2, 1), padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, stride=(2, 1), padding=1), nn.ReLU(),
        )
        self.col = nn.Sequential(nn.Conv1d(ch, emb, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(emb, emb, 3, padding=1), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(emb * (2 * ctx + 1) + NF, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU())
        self.head = nn.Linear(64, 1)                     # скор кандидата
        # ГОЛОВА ТОЧКИ (§6.26): смещение эксперта от центра рана в долях полуширины. Общий ствол —
        # признак «где внутри рана идёт ход» тот же, что отличает свой ран от чужого.
        self.head_off = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, P, F, M, want_off=False):
        """P (B,2,NROW,NCOL) float; F (B,MAXC,NF); M (B,MAXC) — маска живых кандидатов.
        Возвращает скоры (B,MAXC) с -inf на мёртвых; при want_off — ещё и смещение точки."""
        B = P.shape[0]
        z = self.cnn(P).mean(dim=2)                      # (B, ch, NCOL) — свёртка окна по строкам
        e = self.col(z)                                  # (B, emb, NCOL)
        # колонка кандидата: signed_off = (c-pred)/50 → px → индекс колонки патча
        cidx = torch.round(F[:, :, 1] * 50.0 / COL_STEP).long() + (NCOL // 2)
        off = torch.arange(-self.ctx, self.ctx + 1, device=P.device)
        g = (cidx.unsqueeze(-1) + off).clamp(0, NCOL - 1)          # (B,MAXC,2ctx+1)
        idx = g.reshape(B, 1, -1).expand(-1, e.shape[1], -1)
        gathered = torch.gather(e, 2, idx).reshape(B, e.shape[1], MAXC, -1)
        gathered = gathered.permute(0, 2, 1, 3).reshape(B, MAXC, -1)
        h = self.trunk(torch.cat([gathered, F], dim=-1))
        s = self.head(h).squeeze(-1).masked_fill(M == 0, -1e9)
        return (s, self.head_off(h).squeeze(-1)) if want_off else s


def listwise_loss(score, L, M):
    """Мульти-позитивная CE: -log( Σ_pos exp(s) / Σ_all exp(s) ). Форма обучения = форме
    инференса (argmax внутри решения), в отличие от бинарной логистики §6.24."""
    all_lse = torch.logsumexp(score, dim=1)
    pos = score.masked_fill(L == 0, -1e9)
    pos_lse = torch.logsumexp(pos, dim=1)
    ok = (L.sum(1) > 0)
    return (all_lse[ok] - pos_lse[ok]).mean()


# ───────────────────────────── обучение ─────────────────────────────

def offset_loss(off, D, L, F=None):
    """L1 по смещению точки — ТОЛЬКО на кандидатах, чей ран накрывает эксперта (на чужих ранах
    «правильной точки» не существует, учить там нечему).

    F не None → вес пропорционален ПОЛУШИРИНЕ рана. Причина (§6.27): медианная полуширина 1.0px,
    поэтому равновесная L1 почти целиком обучается на ранах, где выбирать нечего, а весь остаток
    ошибки живёт в хвосте широких ранов (p90 7.5px) — там же, где стоят кривые у порога."""
    w = L if F is None else L * (F[:, :, 2] * 20.0 / 2).clamp(min=1.0)
    return ((off - D).abs() * w).sum() / w.sum().clamp(min=1)


def train(data, epochs, bs, lr, dev, lam_off=1.0, off_w="flat"):
    d = np.load(OUT / data, allow_pickle=False)
    P, F, L, M = d["P"], d["F"], d["L"], d["M"]
    D = d["D"] if "D" in d.files else np.zeros_like(L, np.float32)
    has_off = "D" in d.files
    wells = d["wells"]
    uw = sorted(set(wells.tolist()))
    # СПЛИТ ПО СКВАЖИНАМ (не по строкам): соседние решения одной кривой почти дубликаты,
    # построчный сплит завысил бы валидацию.
    val_w = set(uw[::5])
    va = np.isin(wells, list(val_w)); tr = ~va
    print(f"решений {len(P)}: train {int(tr.sum())} / val {int(va.sum())} "
          f"(валидационные скважины {sorted(val_w)})")

    net = WindowSelector().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    ntr = int(tr.sum())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, epochs * (ntr // bs)))
    Pt, Ft, Lt, Mt, Dt = P[tr], F[tr], L[tr], M[tr], D[tr]
    Pv, Fv, Lv, Mv, Dv = P[va], F[va], L[va], M[va], D[va]
    rng = np.random.default_rng(0)

    def batch(Pb, Fb, Lb, Mb, Db, ii):
        p = np.stack([unpack(Pb[i]) for i in ii])
        return (torch.from_numpy(p).to(dev),
                torch.from_numpy(Fb[ii]).to(dev),
                torch.from_numpy(Lb[ii].astype(np.float32)).to(dev),
                torch.from_numpy(Mb[ii].astype(np.float32)).to(dev),
                torch.from_numpy(Db[ii].astype(np.float32)).to(dev))

    def evaluate():
        net.eval(); accm = accn = tot = 0; oe = []
        with torch.no_grad():
            for s in range(0, len(Pv), 4096):
                ii = np.arange(s, min(s + 4096, len(Pv)))
                p, f, l, m, dd = batch(Pv, Fv, Lv, Mv, Dv, ii)
                sc, off = net(p, f, m, want_off=True)
                k = sc.argmax(1)
                near = (f[:, :, 7] * m).argmax(1)         # is_nearest = что взяла бы база
                good = l.sum(1) > 0
                accm += l[torch.arange(len(k)), k][good].sum().item()
                accn += l[torch.arange(len(k)), near][good].sum().item()
                tot += int(good.sum().item())
                # ошибка ТОЧКИ в px: смещение × полуширина рана (width/20 → px)
                hw = (f[:, :, 2] * 20.0 / 2).clamp(min=1.0)
                oe.append((((off - dd).abs() * hw) * l).sum().item() / max(1e-9, l.sum().item()))
        net.train()
        return 100 * accm / max(1, tot), 100 * accn / max(1, tot), tot, float(np.mean(oe))

    for ep in range(epochs):
        idx = rng.permutation(ntr); t0 = time.time(); run = 0.0; nb = 0
        for s in range(0, ntr - bs + 1, bs):
            ii = idx[s:s + bs]
            p, f, l, m, dd = batch(Pt, Ft, Lt, Mt, Dt, ii)
            sc, off = net(p, f, m, want_off=True)
            loss = listwise_loss(sc, l, m) + (
                lam_off * offset_loss(off, dd, l, f if off_w == "width" else None) if has_off else 0.0)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            run += loss.item(); nb += 1
        am, an, tot, oe = evaluate()
        print(f"эпоха {ep+1}/{epochs}  loss {run/max(1,nb):.4f}  "
              f"ВАЛ(держанные скважины, {tot} решений): модель {am:.1f}%  база(ближайший) {an:.1f}%"
              + (f"  точка {oe:.2f}px" if has_off else "") + f"  [{time.time()-t0:.0f}с]")
    from _decoder_seq_data import R_ROWS as _RR, ROW_STEP as _RS
    torch.save({"sd": net.state_dict(), "geom": [NROW, NCOL, MAXC, COL_STEP],
                "geom_full": [_RR, _RS, R_COLS, COL_STEP]}, CKPT)   # ★ 26.09: полная геометрия окна (§6.225)
    print(f"-> {CKPT}")
    return net


# ───────────────────────────── гейт на bench ─────────────────────────────

def csr_band(csr, lo, hi, H):
    """Растр полосы (H × (hi-lo), bool) из CSR-ранов кэша bench — вход для той же patch()."""
    A, B, _, ptr = csr
    band = np.zeros((H, hi - lo), bool)
    for y in range(H):
        for i in range(ptr[y], ptr[y + 1]):
            band[y, max(0, A[i] - lo):max(0, B[i] - lo) + 1] = True
    return band


def make_tracer(net, dev, slmax=30.0, wide_run=14, point="rule"):
    """Трассировщик для bench: структура (коаст, вершина широкого рана, extend) — как в базе и в
    §6.24, меняется ТОЛЬКО правило выбора рана. Растр полосы строится один раз на (кривая,цвет)."""
    cache = {}

    def tracer(rec, csr, H):
        key = (id(csr), rec["lo"], rec["hi"])
        if key not in cache:
            cache.clear()
            cache[key] = csr_band(csr, rec["lo"], rec["hi"], H)
        band = cache[key]
        base = rec["base"]; lo = rec["lo"]
        x = None; v = 0.0; tr = {}
        with torch.no_grad():
            for y in range(max(0, rec["y0"]), min(H, rec["y1"] + 1)):
                A, B, C = BE._runs_at(csr, y)
                if not len(A):
                    if x is not None:
                        x = x + float(np.clip(v, -slmax, slmax))
                    continue
                if x is None:
                    k = int(np.argmin(np.abs(C - base)))
                    x = float(C[k]); v = 0.0; tr[y] = x; continue
                pred = x + float(np.clip(v, -slmax, slmax))
                idx, X = features(A, B, C, pred, x, v, base, MAXC)
                dlt = None
                if len(idx) == 1 and point != "head":
                    k = int(idx[0])
                else:
                    ink, val = patch(band, lo, y, pred)
                    p = torch.from_numpy(np.stack([ink, val]).astype(np.float32))[None].to(dev)
                    f = np.zeros((1, MAXC, NF), np.float32); f[0, :len(idx)] = X
                    m = np.zeros((1, MAXC), np.float32); m[0, :len(idx)] = 1
                    out = net(p, torch.from_numpy(f).to(dev), torch.from_numpy(m).to(dev),
                              want_off=(point == "head"))
                    sc, off = out if point == "head" else (out, None)
                    j = int(sc[0].argmax().item()); k = int(idx[j])
                    if off is not None:
                        dlt = float(off[0, j].item())
                a, b, c = int(A[k]), int(B[k]), float(C[k])
                # ТОЧКА ВНУТРИ РАНА (§6.25-диагностика: у 7 кривых из 9 med* = 0, т.е. весь
                # остаток медианной ошибки — здесь, а не в выборе рана).
                if point == "head" and dlt is not None:
                    nx = min(max(c + dlt * max((b - a) / 2, 1.0), a), b)
                elif point == "center":
                    nx = c
                elif point == "pred":
                    nx = min(max(pred, a), b)         # непрерывность: не двигаться без нужды
                else:
                    nx = (b if abs(b - base) >= abs(a - base) else a) if (b - a) >= wide_run else c
                v = 0.6 * v + 0.4 * (nx - x); x = float(nx); tr[y] = float(nx)
        BE._extend_ends(tr, csr, H, slmax)
        return tr
    return tracer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="seq_train.npz")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--point", default="rule", choices=["rule", "center", "pred", "head"],
                    help="точка внутри рана: правило прода / центр / clamp(предсказание) / голова регрессии")
    ap.add_argument("--lam-off", type=float, default=1.0, help="вес L1-лосса головы точки")
    ap.add_argument("--off-w", default="flat", choices=["flat", "width"],
                    help="взвешивать L1 головы точки полушириной рана (§6.27: медиана 1.0px)")
    a = ap.parse_args()
    if a.ckpt:
        CKPT = OUT / a.ckpt
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"устройство: {dev} ({torch.cuda.get_device_name(0) if dev=='cuda' else ''})")

    if a.gate_only:
        net = WindowSelector().to(dev)
        net.load_state_dict(torch.load(CKPT, map_location=dev)["sd"]); net.eval()
    else:
        net = train(a.data, a.epochs, a.bs, a.lr, dev, a.lam_off, a.off_w); net.eval()

    if not a.no_gate:
        # §6.106: объём — ИЗ СЧЁТЧИКА. «24 кривые» стояли строкой; смени bench состав — и шапка
        # продолжала бы утверждать 24 (ровно ловушка `_slot_cause.py`).
        _base_rows = BE.run_strategy()
        print(f"\n{'='*66}\n=== ГЕЙТ на держанных скважинах (bench, {len(_base_rows)} кривых) ===")
        rb = BE.report("БАЗА (прод)", _base_rows)
        t0 = time.time()
        rm = BE.report(f"ОКОННЫЙ селектор (torch), точка={a.point}",
                       BE.run_strategy(tracer=make_tracer(net, dev, point=a.point)))
        print(f"[прогон гейта {time.time()-t0:.0f}с]")
        # ПОКРИВОЙ, по ШИРИНЕ ПОЛОСЫ: §6.16/§6.24 утверждают, что порог честной гейтится
        # шириной, а не качеством селектора — таблица либо подтверждает это, либо опровергает.
        base = {(r["well"], r["curve"]): r for r in rb["rows"]}
        print(f"\n{'скважина':<12}{'кривая':<8}{'полоса':>7}{'med база':>10}{'med окно':>10}"
              f"{'cov':>6}{'своя%':>7}")
        for r in sorted(rm["rows"], key=lambda r: r["band"]):
            b = base.get((r["well"], r["curve"]), {})
            mark = " ★" if r["med"] <= 3 and r["cov"] >= 0.9 else ""
            print(f"{r['well']:<12}{r['curve']:<8}{r['band']:>7}{b.get('med', float('nan')):>10.1f}"
                  f"{r['med']:>10.1f}{r['cov']:>6.2f}{r['своя']:>7.1f}{mark}")
        print("\nЧитать: честных > 3 при своя% > 36.4 и med(cov) без обвала -> окно даёт то, чего")
        print("не хватало §6.24 (там честных 3/24, med 34.5, своя 36.4).")

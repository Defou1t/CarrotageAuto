r"""_line_choice_set.py — ВЫБОР ТРАССЫ С ОБЗОРОМ ЛИСТА: ПОПРАВКА P БУСТИНГА ВНИМАНИЕМ ПО СЛОТУ И ПО КАНДИДАТУ (24.10, §6.344).

Для `_line_choice_assign.py --setmodel`: токен — строка (слот, кандидат) листа: logit P второго яруса (вне фолда), признаки первого яруса
(z-оценки по обучению, NaN → 0 + флаг). Блок = внимание внутри слота → внутри кластера кандидата → по всему листу → MLP; выход —
поправка к logit P (голова с нулевым началом — старт с P бустинга). Учится вне фолда по скважинам: фолд k — на листах прочих фолдов
(и листах обучения без фолда); сорт A — модель на всём. Ничего не исполняет при импорте.
"""
import numpy as np


def refine(P, X, sheet, slot, clus, lab, fold_of, hold, nf, epochs=6, seeds=1, d=64, blocks=2, lr=1e-3, log=print):
    """P [N] вероятности бустинга; X [N, F]; sheet / slot / clus — [N] (лист, слот, кластер кандидата); lab [N] 1/0/−1;
    fold_of(лист) → фолд, None (только обучение) или −1 (не учить: скважины сорта A); hold — листы, которые мерит модель на всём.
    → новый P (только у листов фолдов и hold; прочие — как было)."""
    import torch, torch.nn as nn
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(P)
    by = {}
    for i, s in enumerate(sheet):
        by.setdefault(s, []).append(i)
    sheets = sorted(by)
    lg = np.log(np.clip(P, 1e-4, 1 - 1e-4) / (1 - np.clip(P, 1e-4, 1 - 1e-4)))
    lg = np.where(np.isfinite(P), lg, 0.0)

    class Blk(nn.Module):
        def __init__(s):
            super().__init__()
            s.at = nn.ModuleList([nn.MultiheadAttention(d, 4, batch_first=True) for _ in range(3)])
            s.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
            s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def forward(s, h, masks):
            for j in range(3):
                z = s.ln[j](h)
                h = h + s.at[j](z, z, z, attn_mask=masks[j], need_weights=False)[0]
            return h + s.ff(s.ln[3](h))

    class Net(nn.Module):
        def __init__(s, fin):
            super().__init__()
            s.inp = nn.Sequential(nn.Linear(fin, d), nn.GELU(), nn.Linear(d, d))
            s.bl = nn.ModuleList([Blk() for _ in range(blocks)])
            s.out = nn.Linear(d, 1); nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)

        def forward(s, x, masks):
            h = s.inp(x)
            for b in s.bl:
                h = b(h, masks)
            return s.out(h)[..., 0]

    def tensors(sh, mu, sd):
        idx = np.array(by[sh])
        x = (X[idx] - mu) / sd
        nanm = ~np.isfinite(x)
        x = np.where(nanm, 0.0, np.clip(x, -6, 6))
        f = np.concatenate([x, nanm.astype(float), lg[idx, None], P[idx, None]], 1)
        f = np.where(np.isfinite(f), f, 0.0)
        sl = np.asarray(slot)[idx]; cl = np.asarray(clus)[idx]
        ms = torch.from_numpy(sl[:, None] != sl[None, :]); mc = torch.from_numpy(cl[:, None] != cl[None, :])
        mc = mc & ~torch.eye(len(idx), dtype=torch.bool)
        ma = torch.zeros(len(idx), len(idx), dtype=torch.bool)
        return (idx, torch.tensor(f, dtype=torch.float32, device=dev)[None], [m.to(dev) for m in (ms, mc, ma)],
                torch.tensor(lab[idx], dtype=torch.float32, device=dev), torch.tensor(lg[idx], dtype=torch.float32, device=dev))

    def fit(train):
        tr_idx = np.concatenate([by[s] for s in train])
        mu = np.nanmean(X[tr_idx], 0); sd = np.nanstd(X[tr_idx], 0)
        mu = np.where(np.isfinite(mu), mu, 0.0); sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
        T = [tensors(s, mu, sd) for s in train if (lab[by[s]] >= 0).any()]
        nets = []
        for seed in range(seeds):
            torch.manual_seed(seed); rs = np.random.RandomState(seed)
            net = Net(T[0][1].shape[2]).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
            steps = epochs * len(T); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
            run = 0.0; st = 0
            for ep in range(epochs):
                for j in rs.permutation(len(T)):
                    _, f, ms, y, l0 = T[j]
                    o = net(f, ms)[0] + l0
                    m = y >= 0
                    loss = nn.functional.binary_cross_entropy_with_logits(o[m], y[m])
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
                    run = 0.99 * run + 0.01 * float(loss.detach()) if st else float(loss.detach()); st += 1
                log(f"    набор из {len(T)} листов, зерно {seed}, эпоха {ep + 1}/{epochs}: потеря {run:.4f}")
            nets.append(net.eval())
        return nets, mu, sd

    @torch.no_grad()
    def score(model, test, out):
        nets, mu, sd = model
        for s in test:
            idx, f, ms, _, l0 = tensors(s, mu, sd)
            o = sum(torch.sigmoid(n(f, ms)[0] + l0) for n in nets) / len(nets)
            out[idx] = o.cpu().numpy()

    out = P.copy()
    fo = {s: fold_of(s) for s in sheets}
    for k in range(nf):
        te = [s for s in sheets if fo[s] == k]
        if te:
            score(fit([s for s in sheets if fo[s] != k and fo[s] != -1]), te, out)
            log(f"  фолд {k}: {len(te)} листов")
    ts = sorted(s for s in hold if s in by)
    if ts:
        score(fit([s for s in sheets if fo[s] != -1]), ts, out)
        log(f"  сорт A: {len(ts)} листов")
    return out

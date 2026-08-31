r"""_rowdec_net.py — ПРОТОТИП ПОСТРОЧНОГО ДЕКОДЕРА (Задача 9, шаг 4).

ПОСТАНОВКА (§6.131 → §6.136). На строке трека предсказать положение ВСЕХ кривых сразу, без
состояния. Ошибка такого решения ЛОКАЛЬНА — медиана её переживает (§6.133: V0 = 100%), тогда как
нынешний трассировщик, потеряв личность на пересечении, уводит всю оставшуюся кривую (§6.116).

АРХИТЕКТУРА — ДВЕ ГОЛОВЫ, СЛОТОВ НЕТ:
  p     [1×R×C]  «здесь центр кривой» (BCE по мягкой цели σ=2px; жёсткая цель была бы враньём —
                 эксперт сам сидит в 1.0px медианы и 2.5px p90, §6.131);
  emb   [E×R×C]  эмбеддинг ЛИЧНОСТИ: точки одной кривой похожи, разных — различны.
⇒ Выход перестановочно-инвариантен по построению, что и требует безымянный счёт (решение Эдуарда
20.08, §6.134): траектории матчатся к эталону 1:1, мнемоника не нужна.

ПОЧЕМУ ИМЕННО ЭМБЕДДИНГ, А НЕ «x НА СЛОТ». Перестановка, посчитанная ПОСТРОЧНО по x, тождественна
контролю без обучения (сортировка по x) — а он берёт 18.2% при потолке 86.4%. Личность обязана
держаться признаком, а не порядком; §6.135 намерил, чем именно: ширина штриха 67.7%, извилистость
64.9%, положение 37.5% (НИЖЕ случайности — на пересечении «ближе» указывает на соседа).

ГЕОМЕТРИЯ. Дилатации по СТРОКАМ 1,2,4,8,16,32 (рецептивное поле ±63 строки — ровно окно, на
котором §6.135 мерил форму), по колонкам дилатация 1 и НИКАКОГО пулинга: колонка = x, и её
разрешение — это и есть то, чем своя кривая отличается от соседки.

  <ComfyUI>\python_embeded\python.exe _rowdec_net.py --fold 0 --epochs 8
  <ComfyUI>\python_embeded\python.exe _rowdec_net.py --overfit 4        # смоук проводки (G1)
"""
import sys, argparse, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--crops", default=r"F:/nds/output/taskS/rowdec_crops")
ap.add_argument("--out", default=r"F:/nds/output/taskS/rowdec_model")
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--emb", type=int, default=8)
ap.add_argument("--ch", type=int, default=32, help="каналов; 56k параметров при 32 недоучиваются")
ap.add_argument("--sigma", type=float, default=2.0)
ap.add_argument("--lam", type=float, default=1.0, help="вес pull/push против BCE")
ap.add_argument("--pos-weight", type=float, default=50.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--overfit", type=int, default=0, help="смоук: переобучить на N кропах")
# ★★ ФЛАГ АБЛЯЦИИ ФОНОВОГО ЧЛЕНА (§6.74: без флага вклад неизмерим).
# ⚠⚠ ЗАЧЕМ ЗАВЕДЁН И ПОЧЕМУ УМОЛЧАНИЕ 0. До 24.08 фоновый член считался БЕЗУСЛОВНО, а
# замороженный набор `frozen_nobg`, на котором стоят все числа §6.141, обучен БЕЗ него ⇒ кодом
# этого файла воспроизвести замороженный набор было НЕЛЬЗЯ. Умолчание обязано совпадать с тем,
# что стоит на пути замера (§6 правило 3), поэтому 0; вариант с фоном просят явно.
ap.add_argument("--bg", type=int, default=0,
                help="1 = чужая тушь отталкивается от прототипов (третий член emb_loss)")
ap.add_argument("--force", action="store_true",
                help="перезаписать существующий чекпойнт (по умолчанию ОТКАЗ, см. §6.141)")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    """Полносвёрточный оконный декодер. ⚠ Ни одного слоя, смешивающего колонки на большом
    расстоянии: сеть обязана решать ЛОКАЛЬНО, иначе она выучит расположение кривых конкретных
    бланков (та же ловушка, что §6.87 — разбиение по листам вместо скважин)."""

    def __init__(self, emb=8, ch=32):
        super().__init__()
        self.stem = nn.Conv2d(1, ch, 3, padding=1)
        blocks = []
        # ⚠ ДИЛАТАЦИЯ ПО ОБЕИМ ОСЯМ. Первая редакция дилатировала только строки (`dilation=(d,1)`),
        # и поле по КОЛОНКАМ выходило ±7px — соседняя кривая в 20px лежала ВНЕ поля зрения, то есть
        # эмбеддинг личности физически не мог отличить её от своей. Замер разбора: ±64 строки, но
        # ±7 колонок при обещанных в докстринге «±63».
        for d in (1, 2, 4, 8, 16, 32):
            blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=(d, d), dilation=(d, d)),
                nn.GroupNorm(4, ch), nn.ReLU(inplace=True)))
        self.blocks = nn.ModuleList(blocks)
        self.head_p = nn.Conv2d(ch, 1, 1)
        self.head_e = nn.Conv2d(ch, emb, 1)

    def forward(self, x):
        h = F.relu(self.stem(x))
        for b in self.blocks:
            h = h + b(h)
        return self.head_p(h), self.head_e(h)


def targets(y, C, sigma):
    """y [B,R,K] позиции (-1 = нет) → мягкая карта [B,1,R,C] и та же y для pull/push."""
    B, R, K = y.shape
    xs = torch.arange(C, device=y.device).view(1, 1, 1, C)
    yy = y.unsqueeze(-1)                                   # B,R,K,1
    m = (yy >= 0).float()
    g = torch.exp(-((xs - yy) ** 2) / (2 * sigma ** 2)) * m
    return g.max(dim=2).values.unsqueeze(1)                # B,1,R,C — максимум по кривым


def emb_loss(emb, y, x=None, delta=1.5, n_bg=64):
    """pull: точки одной кривой к своему центру; push: центры кривых друг от друга; ★ bg: ЧУЖАЯ
    ТУШЬ отталкивается от ВСЕХ прототипов.
    ⚠ Считается ПО КРОПУ, а не по батчу: кривые разных листов ничем не связаны.

    ⚠⚠ ЗАЧЕМ ТРЕТИЙ ЧЛЕН. Первая редакция учила эмбеддинг ТОЛЬКО на пикселях самих кривых — это
    0.4% поля (2 кривые × 250 строк из 256×512). А на инференсе прототипы набираются k-means по
    ПИКАМ, среди которых есть чужая тушь: сетка, соседний трек, рукописные метки. Их эмбеддинг
    не обучался ничему, поэтому расстояние до прототипа у них случайное — и декодер спокойно
    садился на чужую линию. Это train/test-скью в чистом виде (§6.90: своя реализация обязана
    считать то же, что обучающая; здесь — те же ТОЧКИ).
    Фон берётся там, где тушь ЕСТЬ (x > порога), но дальше 5px от любой размеченной кривой."""
    B, E, R, C = emb.shape
    K = y.shape[2]
    pull = emb.new_zeros(()); push = emb.new_zeros(()); bg = emb.new_zeros(())
    n_pull = 0; n_push = 0; n_bg = 0
    for b in range(B):
        mus = []
        for k in range(K):
            v = y[b, :, k]
            ok = v >= 0
            if ok.sum() < 8:
                continue
            rows = torch.nonzero(ok).squeeze(1)
            cols = v[ok].round().long().clamp(0, C - 1)
            e = emb[b, :, rows, cols]                      # E×n
            mu = e.mean(dim=1, keepdim=True)
            pull = pull + ((e - mu) ** 2).sum(0).mean(); n_pull += 1
            mus.append(mu.squeeze(1))
        for i in range(len(mus)):
            for j in range(i + 1, len(mus)):
                d = torch.norm(mus[i] - mus[j])
                push = push + F.relu(delta - d) ** 2; n_push += 1
        # ── ЧУЖАЯ ТУШЬ: далеко от КАЖДОГО прототипа ────────────────────────────────────────
        if x is not None and mus:
            ink = (x[b, 0] > 0.35)                       # тушь заметно темнее бумаги строки
            if ink.any():
                far = torch.ones_like(ink)
                for k in range(K):
                    v = y[b, :, k]; ok = v >= 0
                    if ok.sum() < 8:
                        continue
                    cols = v.round().long().clamp(0, C - 1)
                    for dxp in range(-5, 6):             # ±5px вокруг размеченной кривой — не фон
                        cc = (cols + dxp).clamp(0, C - 1)
                        far[torch.arange(R, device=far.device)[ok], cc[ok]] = False
                cand = torch.nonzero(ink & far)
                if len(cand):
                    sel = cand[torch.randperm(len(cand), device=cand.device)[:64]]
                    eb = emb[b, :, sel[:, 0], sel[:, 1]]          # E×n
                    for mu in mus:
                        d = torch.norm(eb - mu[:, None], dim=0)
                        bg = bg + (F.relu(delta - d) ** 2).mean(); n_bg += 1
    return (pull / max(1, n_pull)), (push / max(1, n_push)), (bg / max(1, n_bg))


def load(fold, folds):
    """Кропы + разбиение ПО СКВАЖИНАМ (§6.87: по листам модель учит бланк, а не правило)."""
    xs, ys, wells = [], [], []
    for xf in sorted(Path(a.crops).glob("x_*of*.npy")):
        tag = xf.stem[2:]
        X = np.load(xf, mmap_mode="r"); Y = np.load(Path(a.crops) / f"y_{tag}.npy", mmap_mode="r")
        man = json.loads((Path(a.crops) / f"man_{tag}.json").read_text(encoding="utf-8"))
        w = []
        for t in man["tracks"]:
            w += [t["well"]] * t["crops"]
        assert len(w) == X.shape[0], f"{tag}: манифест {len(w)} против кропов {X.shape[0]}"
        xs.append(X); ys.append(Y); wells += w
    X = np.concatenate([np.asarray(v) for v in xs]); Y = np.concatenate([np.asarray(v) for v in ys])
    wells = np.array(wells)
    uw = sorted(set(wells.tolist()))
    hold = {w for i, w in enumerate(uw) if i % folds == fold}
    te = np.array([w in hold for w in wells])
    print(f"кропов {X.shape[0]:,}; скважин {len(uw)}, в держанном фолде {len(hold)}; "
          f"обучение {int((~te).sum()):,} / проверка {int(te.sum()):,}")
    return X, Y, ~te, te


def main():
    # ⚠⚠ ОТКАЗ ПЕРЕЗАПИСАТЬ — ДО ОБУЧЕНИЯ, А НЕ ПОСЛЕ. §6.141: обучение и замер писали в одни
    # имена, и во время A/B два фолда из пяти были подменены другим вариантом лосса; замеры
    # редакций 1-3 сравнивались на СМЕШАННОМ наборе. Три часа счёта — не повод узнавать об этом
    # в конце. Новый вариант обучения — в СВОЙ каталог (`--out`), а не поверх.
    _ck = OUT / (f"rowdec_of{a.folds}_f{a.fold}_s{a.seed}.pt" if not a.overfit
                 else "rowdec_overfit.pt")
    if _ck.exists() and not a.force:
        sys.exit(f"⛔ {_ck} УЖЕ ЕСТЬ. Обучение остановлено, чтобы не подменить чекпойнт под чужим "
                 f"замером (§6.141). Свой каталог: --out <новый>; поверх осознанно: --force")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    X, Y, tr, te = load(a.fold, a.folds)
    if a.overfit:
        # ⚠ БРАТЬ КРОПЫ С ПАРОЙ КРИВЫХ. Первая редакция брала первые попавшиеся, и push-член вышел
        # тождественным нулём (в них была одна кривая) — то есть смоук проверял ПОЛОВИНУ лосса и
        # молчал об этом. Ровно форма §6.94: величина, равная нулю по построению, ничего не проверяет.
        pair = np.array([(Y[i, :, 0] >= 0).any() and (Y[i, :, 1] >= 0).any()
                         for i in range(len(Y))])
        cand = np.where(tr & pair)[0]
        print(f"кропов с ≥2 кривыми: {int(pair.sum()):,} из {len(pair):,}")
        idx = cand[:a.overfit]
        tr = np.zeros(len(tr), bool); tr[idx] = True; te = tr.copy()
        print(f"⚠ СМОУК ПРОВОДКИ: переобучение на {a.overfit} кропах (G1)")
    net = Net(a.emb, a.ch).to(DEV)
    print(f"параметров {sum(p.numel() for p in net.parameters()):,}, устройство {DEV}; "
          f"★ ВАРИАНТ ЛОССА: bg={a.bg} ({'с фоновым членом' if a.bg else 'БЕЗ фонового члена — '
          'как замороженный набор frozen_nobg'}), эпох {a.epochs}, каталог {OUT}")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    idx_tr = np.where(tr)[0]
    steps = max(1, len(idx_tr) // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * steps)
    pw = torch.tensor(a.pos_weight, device=DEV)
    for ep in range(a.epochs):
        net.train(); np.random.shuffle(idx_tr); t0 = time.time(); tot = tp = tu = ts = 0.0
        for s in range(steps):
            b = idx_tr[s * a.batch:(s + 1) * a.batch]
            xb = torch.from_numpy(X[b].astype(np.float32) / 255.0).unsqueeze(1).to(DEV)
            yb = torch.from_numpy(Y[b]).to(DEV)
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=(DEV == "cuda")):
                p, e = net(xb)
                g = targets(yb, xb.shape[-1], a.sigma)
                lp = F.binary_cross_entropy_with_logits(p.float(), g.float(), pos_weight=pw)
                pull, push, bgl = emb_loss(e.float(), yb, xb.float() if a.bg else None)
                loss = lp + a.lam * (pull + push + bgl)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += float(loss); tp += float(lp); tu += float(pull); ts += float(push + bgl)
        # ⚠ ДЕРЖАННЫЙ ЛОСС: маска `te` раньше не использовалась НИГДЕ, и «проверка 20,600» в логе
        # создавала впечатление проверки, которой не было. Без него переобучение невидимо.
        net.eval(); vb = 0.0; vn = 0
        with torch.no_grad():
            vi = np.where(te)[0][:: max(1, int(te.sum()) // (20 * a.batch))]
            for s2 in range(0, len(vi), a.batch):
                b = vi[s2:s2 + a.batch]
                if len(b) < 2:
                    continue
                xb = torch.from_numpy(X[b].astype(np.float32) / 255.0).unsqueeze(1).to(DEV)
                yb = torch.from_numpy(Y[b]).to(DEV)
                p, e = net(xb)
                g = targets(yb, xb.shape[-1], a.sigma)
                vb += float(F.binary_cross_entropy_with_logits(p.float(), g.float(), pos_weight=pw))
                vn += 1
        print(f"эпоха {ep+1}/{a.epochs}  loss {tot/steps:.4f} (bce {tp/steps:.4f}, "
              f"pull {tu/steps:.4f}, push {ts/steps:.4f})  ДЕРЖАННЫЙ bce {vb/max(1,vn):.4f}  "
              f"{time.time()-t0:.0f}с")
    ck = OUT / (f"rowdec_of{a.folds}_f{a.fold}_s{a.seed}.pt" if not a.overfit else "rowdec_overfit.pt")
    # ⚠⚠ ПРОИСХОЖДЕНИЕ — В САМ ЧЕКПОЙНТ. 24.08 выяснилось, что вариант лосса и число эпох у
    # замороженного набора восстанавливаются ТОЛЬКО по меткам времени и по строке предупреждения
    # в логе обучения (`ts += float(push)` против `push + bgl`), а логи переписывались одним
    # именем. То есть «этот чекпойнт обучен тем же кодом» было НЕПРОВЕРЯЕМЫМ утверждением —
    # ровно §6.71, только про веса, а не про кэши.
    torch.save(dict(sd=net.state_dict(), emb=a.emb, ch=a.ch, sigma=a.sigma, fold=a.fold,
                    folds=a.folds, seed=a.seed, pos_weight=a.pos_weight,
                    bg=int(a.bg), epochs=a.epochs, lam=a.lam, lr=a.lr, batch=a.batch,
                    crops=str(a.crops)), ck)
    print(f"★ чекпойнт → {ck}   (bg={a.bg}, эпох {a.epochs})")


if __name__ == "__main__":
    # ⚠⚠⚠ БЕЗ ЭТОЙ СТРОКИ ЛЮБОЙ ИМПОРТ ФАЙЛА РАДИ КЛАССА `Net` ЗАПУСКАЛ ОБУЧЕНИЕ И СОХРАНЯЛ
    # ЧЕКПОЙНТ. `_rowdec_eval.load_net()` импортирует этот модуль, подставив `--epochs 0`, — и
    # тем самым КЛАЛ СЛУЧАЙНЫЕ ВЕСА ПОВЕРХ ОБУЧЕННЫХ, после чего сам же их и загружал. Все замеры
    # модели после первого прогона оценки мерили необученную сеть; видно это было только по mtime
    # чекпойнта (05:40 — время оценки, а не обучения) и по строке «★ чекпойнт →» в выводе оценки.
    main()

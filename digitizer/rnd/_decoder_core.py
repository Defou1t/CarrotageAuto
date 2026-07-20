r"""_decoder_core.py — ЯДРО ОБУЧАЕМОГО СЕЛЕКТОРА РАНОВ (P2, первый ML-шаг, §6.8/§6.22).

Задача (§6.23): 42% кривых записаны, но ЛАТЧ на соседа (ИДЕНТ). §6.22 доказал: рамка
идентичность GZ не выводит — значит выбор рана должен УЧИТЬСЯ из траектории. Здесь — самый
дешёвый честный ML: на КАЖДОЙ строке модель скорит все раны-кандидаты в полосе и берёт лучший,
вместо жёсткого «ближайший к pred». Обучение — teacher-forcing по экспертным трассам: где
кривая реально пошла, тот ран и есть правильный. Чистый numpy (в базовом env нет sklearn/torch).

★ ЧЕСТНОСТЬ: признаки для обучения и для инференса считает ОДНА функция `features()` — иначе
train/test-скью. Модель — логистическая регрессия «этот кандидат — правильное продолжение»,
на инференсе argmax вероятности среди кандидатов строки.

★ ГЕЙТ (durable трек-2): мерить по СВОПАМ/честным на ДЕРЖАННЫХ скважинах, не по dice. Тестовые
5 скважин (кэш bench) в обучение НЕ входят — обучаемся на прочем архиве.
"""
import numpy as np

# порядок признаков фиксирован — им пользуются и экстрактор, и трассировщик
FEAT_NAMES = [
    "gap",          # зазор до предсказания (0 если ран накрывает pred)
    "signed_off",   # (центр - pred) / 50, знаковое смещение
    "width",        # ширина рана / 20
    "dist_x",       # |центр - текущий x| / 50
    "dist_base",    # |центр - базлайн| / 200
    "overlap",      # 1 если ран накрывает pred (ветка cont)
    "is_widest",    # 1 если это самый широкий ран строки
    "is_nearest",   # 1 если ближайший к pred по центру (что взяла бы база)
    "n_cands",      # число кандидатов / 5
    "vsmooth",      # |нов.скорость - тек.скорость| при взятии рана / 30
]
NF = len(FEAT_NAMES)


def features(A, B, C, pred, x, v, base, n_pick=6):
    """Признаки для КАНДИДАТОВ строки. Возвращает (idx, X): idx — индексы взятых кандидатов
    (n_pick ближайших к pred по центру, чтобы не скорить сотни ранов), X — (len(idx), NF).
    ИДЕНТИЧНА на train и inference."""
    if len(A) == 0:
        return np.array([], int), np.zeros((0, NF))
    order = np.argsort(np.abs(C - pred))[:n_pick]
    a = A[order].astype(np.float64); b = B[order].astype(np.float64); c = C[order]
    gap = np.maximum(np.maximum(a - pred, pred - b), 0.0)
    width = b - a
    overlap = ((a - 2 <= pred) & (pred <= b + 2)).astype(np.float64)
    nx = c                                             # упрощ.: центр (вершина широкого — отдельно в трассере)
    nv = 0.6 * v + 0.4 * (nx - x)
    widest = np.zeros(len(order)); widest[np.argmax(width)] = 1.0
    nearest = np.zeros(len(order)); nearest[np.argmin(np.abs(c - pred))] = 1.0
    X = np.stack([
        gap / 50.0,
        (c - pred) / 50.0,
        width / 20.0,
        np.abs(c - x) / 50.0,
        np.abs(c - base) / 200.0,
        overlap,
        widest,
        nearest,
        np.full(len(order), len(A) / 5.0),
        np.abs(nv - v) / 30.0,
    ], axis=1)
    return order, X


def expand(X):
    """Полиномиальное расширение 10 базовых признаков взаимодействиями (тест нелинейности БЕЗ
    переизвлечения: считается из уже сохранённой X, ИДЕНТИЧНО на train и inference). Индексы:
    0 gap,1 signed_off,2 width,3 dist_x,4 dist_base,5 overlap,6 is_widest,7 is_nearest,8 n_cands,9 vsmooth."""
    if X.shape[0] == 0:
        return X
    g, so, w, dx, db, ov, iw, inr, nc, vs = [X[:, i] for i in range(10)]
    extra = np.stack([
        g * inr,        # ближайший, но далёкий = неоднозначно
        w * iw,         # самый широкий И широкий = сильный сигнал своей кривой
        dx * vs,        # смещение × рывок скорости
        g * ov,         # зазор при перекрытии (≈0)
        so * so,        # квадрат смещения
        dx * dx,
        w * dx,         # широкий далёкий ран = чужой
        ov * inr,       # перекрывает И ближайший = обычный cont
        iw * inr,       # самый широкий И ближайший
        vs * inr,       # рывок при взятии ближайшего
    ], axis=1)
    return np.concatenate([X, extra], axis=1)


class Logistic:
    """Бинарная логистическая регрессия (numpy), L2, полный-батч градиент. Класс 1 = «этот
    кандидат — правильное продолжение экспертной кривой»."""
    def __init__(self, w=None, b=0.0):
        self.w = w; self.b = b

    def fit(self, X, y, l2=1e-3, lr=0.5, iters=400):
        n, d = X.shape
        # нормировка не нужна: признаки уже масштабированы в features()
        self.w = np.zeros(d); self.b = 0.0
        pos = max(1, y.sum()); neg = max(1, n - y.sum())
        wpos = n / (2 * pos); wneg = n / (2 * neg)     # баланс классов (правильных ранов мало)
        sw = np.where(y == 1, wpos, wneg)
        for _ in range(iters):
            z = X @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = (p - y) * sw
            gw = X.T @ g / n + l2 * self.w
            gb = g.mean()
            self.w -= lr * gw; self.b -= lr * gb
        return self

    def score(self, X):
        if X.shape[0] == 0:
            return np.zeros(0)
        return X @ self.w + self.b

    def to_dict(self):
        return {"w": self.w.tolist(), "b": float(self.b), "feat": FEAT_NAMES}

    @staticmethod
    def from_dict(d):
        return Logistic(np.array(d["w"], float), float(d["b"]))


def make_tracer(model, slmax=30.0, wide_run=14, n_pick=6, poly=False):
    """Обучаемый трассировщик для стенда bench: на каждой строке скорит кандидатов моделью,
    берёт argmax. Структура (коаст пустых строк, вершина широкого рана, extend) — как в базе,
    меняется ТОЛЬКО правило выбора рана. Плагается как BE.run_strategy(tracer=make_tracer(m)).
    poly=True — расширить признаки взаимодействиями (модель обучена на expand(X))."""
    import _relatch_bench as BE

    def tracer(rec, csr, H):
        base = rec["base"]
        x = None; v = 0.0; tr = {}
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
            idx, X = features(A, B, C, pred, x, v, base, n_pick)
            if poly:
                X = expand(X)
            k = int(idx[int(np.argmax(model.score(X)))])
            a, b, c = int(A[k]), int(B[k]), float(C[k])
            nx = (b if abs(b - base) >= abs(a - base) else a) if (b - a) >= wide_run else c
            v = 0.6 * v + 0.4 * (nx - x); x = float(nx); tr[y] = float(nx)
        BE._extend_ends(tr, csr, H, slmax)
        return tr
    return tracer

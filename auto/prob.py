r"""
prob.py — provider карты ПЕРЕДНЕГО ПЛАНА из обученной модели (опциональный torch).

Это ответ на «нужна ли наша нейронка»: ДА, но как HIGH-RECALL ГЕЙТ переднего плана, НЕ как
сегментатор идентичности. Durable (PLAN §6.6.10/§6.6.13):
  • U-Net бинарная маска как РАЗДЕЛИТЕЛЬ линий — ЗАКРЫТО (блобит, идентичность теряет);
  • recall-ретрейн (fade-аугментация + Tversky β>α) — РАБОТАЕТ: выцветший GZ recall 21→84%.
Поэтому модель подключается как `config.prob_provider`: даёт prob[HxW] в [0,1], которым
imaging.ink_foreground гасит сетку/пятна и ПОДНИМАЕТ бледные линии (потолок чистых правил).
Пайплайн работает и БЕЗ модели (provider=None) — тогда бледные цветные остаются потолком.

Переиспользует digitizer/infer (load_model + predict_prob), без дублирования инференса.
Запуск инференса — интерпретатором venv ComfyUI (torch); чистый пайплайн от него НЕ зависит.

  python -m auto.prob <ckpt> <image> [--out DIR]   # сохранить prob-хитмап (визуальная проверка)
"""
import sys
from pathlib import Path


def make_prob_provider(ckpt, device=None, tile=256, ov=96):
    """Собрать callable rgb->prob[HxW float32 0..1] из чекпойнта. Ленивая загрузка torch."""
    import torch                       # lazy: чистый пайплайн без torch не падает
    from infer import load_model, predict_prob
    # ⛔ 26.09 (аудит): `is_available()` при CUDA_VISIBLE_DEVICES="" даёт True при 0 устройств — та же ловушка, что уже
    #   исправлена в trace_seq.py; выбор — по числу устройств
    dev = device or ("cuda" if torch.cuda.device_count() > 0 else "cpu")
    net, ck = load_model(str(ckpt), dev)

    def provider(rgb):
        return predict_prob(net, rgb, dev, tile=tile, ov=ov)

    provider.meta = {"ckpt": str(ckpt), "epoch": ck.get("epoch"),
                     "val_dice": ck.get("val_dice"), "device": dev}
    return provider


def attach(cfg, ckpt, **kw):
    """Подключить модель к Config как prob_provider (для пайплайна/UI)."""
    cfg.prob_provider = make_prob_provider(ckpt, **kw)
    return cfg


def prob_from_npy(npy_path):
    """Провайдер из СОХРАНЁННОЙ prob-карты (.npy). Главный мост: тяжёлый torch-инференс гоним РАЗ
    интерпретатором venv ComfyUI (`python -m auto.prob ...` сохраняет _prob.npy), а чистый пайплайн
    py3.14 (без torch) грузит кэш и пользует как gate. rgb→prob[HxW float32 0..1]."""
    import numpy as np
    arr = np.load(str(npy_path)).astype("float32")

    def provider(rgb):
        h, w = rgb.shape[:2]
        if arr.shape == (h, w):
            return arr
        import cv2
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)

    provider.meta = {"npy": str(npy_path), "shape": list(arr.shape)}
    return provider


def attach_npy(cfg, npy_path):
    """Подключить СОХРАНЁННУЮ prob-карту к Config (без torch)."""
    cfg.prob_provider = prob_from_npy(npy_path)
    return cfg


def save_prob_overlay(rgb, prob, out, stem):
    """Хитмап prob поверх скана — визуально проверить, что модель ВИДИТ бледные линии."""
    import numpy as np
    from PIL import Image
    H, W = prob.shape
    heat = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    ov = rgb.copy()
    m = prob > 0.4
    ov[m] = (0.4 * ov[m] + 0.6 * np.stack([heat, np.zeros_like(heat), 255 - heat], -1)[m]).astype(np.uint8)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = out / f"{stem}_prob.png"
    Image.fromarray(ov).resize((max(1, W // 5), max(1, H // 5)), Image.LANCZOS).save(p)
    return str(p)


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        print(__doc__); return 1
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import auto  # noqa: F401  (bootstrap ../digitizer на путь)
    from auto import imaging
    ckpt, image = a[0], a[1]
    out = a[a.index("--out") + 1] if "--out" in a else "output"
    prov = make_prob_provider(ckpt)
    print(f"модель: {prov.meta}")
    rgb = imaging.load_rgb(image)
    prob = prov(rgb)
    import numpy as np
    print(f"prob: shape={prob.shape} max={prob.max():.3f} >0.4 покрытие={float((prob>0.4).mean())*100:.2f}%")
    Path(out).mkdir(parents=True, exist_ok=True)
    npy = Path(out) / f"{Path(image).stem}_prob.npy"
    np.save(str(npy), prob.astype("float32"))            # raw-карта для пайплайна (attach_npy)
    print(f"raw prob -> {npy}")
    p = save_prob_overlay(rgb, prob, out, Path(image).stem[:40])
    print(f"хитмап -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

r"""Пакетная выгонка prob-карт recall-модели (ML НА ФОН) интерпретатором venv ComfyUI.
Embeddable-python ComfyUI НЕ добавляет cwd в sys.path (._pth), поэтому пути прописаны явно —
`python -m auto.prob` там не заводится.

  <ComfyUI>\python_embeded\python.exe digitizer\rnd\_prob_batch.py <ckpt> <img> [<img> ...] --out DIR

Кладёт <stem>_prob.npy (сырая карта для auto.prob.attach_npy) + хитмап. Пайплайн py3.14 потом
цепляет карту БЕЗ torch: `python -m auto.pipeline <img> --frame <n.nlgx> --prob <карта.npy>`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent          # F:\nds\Auto
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "digitizer"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from auto import imaging
from auto.prob import make_prob_provider, save_prob_overlay

a = sys.argv[1:]
oi = a.index("--out") if "--out" in a else None
out = Path(a[oi + 1]) if oi is not None else ROOT / "output" / "prob"
skip = {oi, oi + 1} if oi is not None else set()
imgs = [x for i, x in enumerate(a) if i >= 1 and i not in skip and not x.startswith("--")]
out.mkdir(parents=True, exist_ok=True)
prov = make_prob_provider(a[0])
print("модель:", prov.meta)
for ip in imgs:
    stem = Path(ip).stem
    dst = out / f"{stem}_prob.npy"
    if dst.is_file():
        print(f"  есть: {dst.name}"); continue
    rgb = imaging.load_rgb(ip)
    prob = prov(rgb)
    np.save(str(dst), prob.astype("float32"))
    save_prob_overlay(rgb, prob, out, stem[:40])
    print(f"  {stem[:46]:<46} {rgb.shape[1]}x{rgb.shape[0]} prob>0.4 {float((prob>0.4).mean())*100:5.2f}% → {dst.name}")

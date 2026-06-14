"""
detect_calibration.py — детект геометрии планшета из ИЗОБРАЖЕНИЯ (без nlgx):
  • горизонтальные линии depth-сетки (жирные 4 м) -> depth<->y;
  • вертикальные края трека / деления шкалы -> value<->x.
Валидируется против nlgx ground-truth (extract_nlgx): точные Y сетки (35596),
x_left/x_right Scale Axis, top_y/bottom_y Depth Axis.

python detect_calibration.py <nlgx> <image>      # детект + сверка с nlgx
"""
import sys
import numpy as np
from PIL import Image
from extract_nlgx import extract
Image.MAX_IMAGE_PIXELS = None


def _profile_peaks(score, min_dist, prominence):
    """Простой поиск локальных максимумов с мин. расстоянием и prominence."""
    peaks = []
    n = len(score)
    i = 1
    while i < n-1:
        if score[i] >= score[i-1] and score[i] > score[i+1] and score[i] >= prominence:
            # уточнить как локальный максимум в окне min_dist
            lo = max(0, i-min_dist); hi = min(n, i+min_dist+1)
            if score[i] >= score[lo:hi].max():
                peaks.append(i)
                i += min_dist
                continue
        i += 1
    return np.array(peaks)


def detect_depth_grid(gray, thr=140):
    """
    Детект горизонтальных линий сетки. Возвращает (ys, spacing_fine, ys_bold).
    Идея: профиль числа тёмных пикселей по строкам -> пики = горизонтали.
    Жирные (4 м) — самые выраженные; шаг сетки — по автокорреляции профиля.
    """
    H, W = gray.shape
    # центральная x-полоса, чтобы горизонтали (через весь лист) набирали счёт,
    # а краевые подписи/шум не мешали
    x0, x1 = int(0.12*W), int(0.95*W)
    band = x1 - x0
    ink = (gray[:, x0:x1] < thr)
    row = ink.sum(axis=1).astype(np.float64)
    # вычесть медленный тренд (затемнение/пятна)
    k = 51
    base = np.convolve(row, np.ones(k)/k, mode="same")
    sig = np.clip(row - base, 0, None)

    # все горизонтали (тонкая сетка): пики с малым min_dist
    all_peaks = _profile_peaks(sig, min_dist=3, prominence=np.percentile(sig, 75))
    # шаг тонкой сетки = мода малых межпиковых интервалов
    fine = 12
    if len(all_peaks) > 5:
        diffs = np.diff(all_peaks)
        small = diffs[(diffs >= 4) & (diffs <= 40)]
        if len(small):
            fine = int(np.median(small))

    # ЖИРНЫЕ 4-м линии = длинные горизонтали (покрывают почти всю ширину) и
    # выраженные; ищем с min_dist ~ 5*fine, prominence высокий
    longline = (row >= 0.5 * band)           # линия через >=50% ширины
    sig_bold = sig * longline
    bold = _profile_peaks(sig_bold, min_dist=max(20, int(4.5*fine)),
                          prominence=np.percentile(sig_bold[sig_bold > 0], 50) if (sig_bold > 0).any() else 1)
    # шаг жирных
    bold_sp = float(np.median(np.diff(bold))) if len(bold) > 2 else None
    return all_peaks, fine, bold, bold_sp


def detect_track_edges(gray, thr=140):
    """Вертикальные края трека: профиль тёмных пикселей по столбцам -> пики."""
    H, W = gray.shape
    y0, y1 = int(0.2*H), int(0.8*H)
    ink = (gray[y0:y1, :] < thr)
    col = ink.sum(axis=0).astype(np.float64)
    k = 31; base = np.convolve(col, np.ones(k)/k, mode="same")
    sig = np.clip(col - base, 0, None)
    peaks = _profile_peaks(sig, min_dist=15, prominence=np.percentile(sig, 90))
    return peaks, sig


def validate(nlgx, image):
    m = extract(nlgx)
    g = np.asarray(Image.open(image).convert("L"))
    H, W = g.shape
    da = m["depth_axis"]; dg = m.get("depth_grid", {})
    true_ys = np.array(dg.get("ys", []))
    print(f"image {W}x{H}; nlgx depth axis y {da['top_y']}..{da['bottom_y']}, "
          f"grid lines={len(true_ys)} spacing~{(true_ys[-1]-true_ys[0])/(len(true_ys)-1):.1f}" if len(true_ys)>1 else "")

    peaks, fine, bold, bold_sp = detect_depth_grid(g)
    print(f"\nDEPTH GRID detect: {len(peaks)} h-lines, fine-spacing={fine}px, "
          f"bold(4m)={len(bold)} bold_spacing={bold_sp}")
    if len(true_ys) > 1:
        true_sp = (true_ys[-1]-true_ys[0])/(len(true_ys)-1)
        # истинные линии -> ближайшая ЖИРНАЯ детектированная (это и есть 4-м сетка)
        for label, det in [("all-peaks", peaks), ("bold", bold)]:
            if len(det):
                d = np.abs(true_ys[:, None] - det[None, :]).min(axis=1)
                print(f"  true grid -> nearest {label:9}: median={np.median(d):.1f}px "
                      f"within2={np.mean(d<=2)*100:.0f}% within5={np.mean(d<=5)*100:.0f}% (n_det={len(det)})")
        print(f"  true 4m spacing={true_sp:.2f}px ; detected bold_spacing={bold_sp}")

    edges, esig = detect_track_edges(g)
    print(f"\nTRACK EDGES detect: {len(edges)} v-lines at x={list(edges[:12])}")
    xs_true = sorted({s["x_left"] for s in m["scale_axes"]} | {s["x_right"] for s in m["scale_axes"]})
    print(f"  nlgx scale x_left/x_right set: {xs_true}")
    if len(edges):
        for xt in xs_true:
            dd = np.abs(edges - xt).min()
            print(f"    x={xt}: nearest detected edge {edges[np.abs(edges-xt).argmin()]} (|d|={dd}px)")


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])

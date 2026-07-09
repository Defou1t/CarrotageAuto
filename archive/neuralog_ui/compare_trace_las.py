"""
compare_trace_las.py - сравнение собственной трассы Блока 2 с эталонным LAS.

Эталон — ручная оцифровка в NeuraLOG, экспортированная в LAS 2.0.
Сравнивает trace.samples каждой кривой из *_analysis.json с колонкой LAS:
корреляция, смещение по глубине, RMSE, систематика амплитуды.

Использование:
    python compare_trace_las.py <image_or_analysis.json> <file.las>
    python compare_trace_las.py ... --no-plot
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEBUG_DIR = r"F:\nds\logs\analysis_debug"


def parse_las(las_path: str) -> dict:
    """Простой парсер LAS 2.0: возвращает {'DEPT': [...], 'BK': [...], ...}."""
    names: list[str] = []
    null_value = -999.25
    data_lines = []
    section = ""
    for line in Path(las_path).read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("~"):
            section = stripped[1].upper()
            continue
        if section == "C":
            mnem = re.match(r"\s*([A-Za-z0-9_]+)\s*\.", line)
            if mnem:
                names.append(mnem.group(1).upper())
        elif section == "W":
            null_match = re.match(r"\s*NULL\s*\.\s*(-?\d+(?:\.\d+)?)", line)
            if null_match:
                null_value = float(null_match.group(1))
        elif section == "A":
            data_lines.append(stripped)

    columns: dict[str, list[float]] = {name: [] for name in names}
    for line in data_lines:
        parts = line.split()
        if len(parts) != len(names):
            continue
        for name, token in zip(names, parts):
            try:
                value = float(token)
            except ValueError:
                value = null_value
            columns[name].append(value)
    return {"columns": columns, "null": null_value, "names": names}


def compare_curve(curve: dict, las: dict, max_shift_m: float = 3.0):
    """Метрики совпадения trace.samples кривой с колонкой LAS того же имени."""
    import numpy as np

    name = curve.get("name", "")
    samples = (curve.get("trace") or {}).get("samples") or []
    if len(samples) < 50:
        return {"curve": name, "status": "no_trace_samples"}
    las_name = next((n for n in las["names"] if n.upper() == name.upper()), None)
    depth_key = next((n for n in las["names"] if n.upper() in {"DEPT", "DEPTH"}), None)
    if las_name is None or depth_key is None:
        return {"curve": name, "status": "curve_not_in_las"}

    ref_d = np.asarray(las["columns"][depth_key], dtype=float)
    ref_v = np.asarray(las["columns"][las_name], dtype=float)
    good = ref_v != las["null"]
    ref_d, ref_v = ref_d[good], ref_v[good]

    my = np.asarray(samples, dtype=float)
    my_d, my_v = my[:, 0], my[:, 1]
    order = np.argsort(my_d)
    my_d, my_v = my_d[order], my_v[order]

    lo = max(ref_d.min(), my_d.min())
    hi = min(ref_d.max(), my_d.max())
    if hi - lo < 10:
        return {"curve": name, "status": "no_depth_overlap"}

    grid = np.arange(lo, hi, 0.1)
    ref_i = np.interp(grid, ref_d, ref_v)
    my_i = np.interp(grid, my_d, my_v)

    def corr(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    # Поиск сдвига по глубине, максимизирующего корреляцию.
    best_shift, best_corr = 0.0, corr(ref_i, my_i)
    for shift in np.arange(-max_shift_m, max_shift_m + 0.001, 0.1):
        my_shifted = np.interp(grid, my_d + shift, my_v)
        c = corr(ref_i, my_shifted)
        if c > best_corr:
            best_corr, best_shift = c, float(shift)

    my_best = np.interp(grid, my_d + best_shift, my_v)
    err = my_best - ref_i
    # Линейная регрессия my ~ a*ref + b: систематика амплитуды.
    a, b = np.polyfit(ref_i, my_best, 1)
    return {
        "curve": name,
        "status": "ok",
        "overlap_m": round(float(hi - lo), 1),
        "corr_raw": round(corr(ref_i, my_i), 4),
        "best_shift_m": round(best_shift, 2),
        "corr_shifted": round(best_corr, 4),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 4),
        "bias": round(float(err.mean()), 4),
        "p95_abs_err": round(float(np.percentile(np.abs(err), 95)), 4),
        "amp_gain_my_vs_ref": round(float(a), 4),
        "amp_offset": round(float(b), 4),
        "ref_range": [round(float(ref_i.min()), 3), round(float(ref_i.max()), 3)],
        "my_range": [round(float(my_best.min()), 3), round(float(my_best.max()), 3)],
        "_plot_data": (grid, ref_i, my_best),
    }


def plot_comparison(metrics: list[dict], out_path: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    panels = [m for m in metrics if m.get("status") == "ok"]
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 14), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, m in zip(axes, panels):
        grid, ref_i, my_i = m["_plot_data"]
        ax.plot(ref_i, grid, "k-", lw=0.7, label="LAS (ручная оцифровка)")
        ax.plot(my_i, grid, "r-", lw=0.7, alpha=0.8,
                label=f"trace (shift {m['best_shift_m']} m)")
        ax.set_title(f"{m['curve']}  corr={m['corr_shifted']}  rmse={m['rmse']}")
        ax.set_xlabel(m["curve"])
        ax.invert_yaxis()
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Depth, m")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Сравнение трассы Блока 2 с эталонным LAS")
    parser.add_argument("analysis", help="*_analysis.json или путь к изображению")
    parser.add_argument("las", help="Эталонный LAS 2.0")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    analysis_path = Path(args.analysis)
    if analysis_path.suffix.lower() != ".json":
        analysis_path = analysis_path.with_name(analysis_path.stem + "_analysis.json")
    result = json.loads(analysis_path.read_text(encoding="utf-8"))
    las = parse_las(args.las)

    metrics = [compare_curve(curve, las) for curve in result.get("curves", [])]

    if not args.no_plot:
        out_path = str(Path(DEBUG_DIR) / (analysis_path.stem.replace("_analysis", "") + "_compare.png"))
        if plot_comparison(metrics, out_path):
            print(f"plot: {out_path}", file=sys.stderr)

    for m in metrics:
        m.pop("_plot_data", None)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

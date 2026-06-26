r"""
config.py — централизованная конфигурация. УБИРАЕТ хардкод `F:\nds` (durable риск-санитария
ROADMAP §7: 45 файлов старого кода хардкодят пути). Пути берутся из переменных окружения, с
вменяемыми относительными дефолтами; CV-параметры — в одном месте.

ENV:
  CARROTAGE_DATA   корень данных (projects/<well>/{img,wlg,las}); дефолт — ./data
  CARROTAGE_OUT    папка вывода (_auto.nlgx/las/overlay/understanding.json); дефолт — ./output
  CARROTAGE_CORPUS корпус проверенных эталонов (для priors); дефолт — ./corpus
  LMSTUDIO_BASE    база LM Studio для A1-счёта строк линейки; дефолт http://localhost:1234/v1
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


@dataclass
class CVParams:
    """Параметры разбора изображения (см. карту разделимости ROADMAP §5 / PLAN §6.6.13).

    Сетка СВЕТЛАЯ (V~166), чёрная кривая V<110 → порог по АБСОЛЮТНОЙ темноте, НЕ adaptiveThreshold
    (он ловит и сетку). Цветная — по разнице каналов."""
    dark_v: int = 110          # абсолютная темнота чёрной кривой (V<dark_v)
    grid_v_lo: int = 150       # светло-серая полоса сетки [grid_v_lo, grid_v_hi)
    grid_v_hi: int = 205
    sat_thr: int = 28          # порог насыщенности для «цветной»
    rg_thr: int = 25           # |R-G| для красной SP / зелёной GZ
    struct_open_len: int = 120 # длина морф-открытия для ПРЯМЫХ структур (рамка/деления)
    min_line_h_frac: float = 0.04   # минимальная высота линии-инстанса (доля высоты трека)
    density_smooth: int = 9    # сглаживание столбцовой плотности при счёте линий
    valley_ratio: float = 0.40 # глубина долины плотности для сплита разнесённых (консервативно)


@dataclass
class Config:
    data: Path = field(default_factory=lambda: _env_path("CARROTAGE_DATA", _REPO / "data"))
    out: Path = field(default_factory=lambda: _env_path("CARROTAGE_OUT", _REPO / "output"))
    corpus: Path = field(default_factory=lambda: _env_path("CARROTAGE_CORPUS", _REPO / "corpus"))
    mnemonics: Path = field(default_factory=lambda: _REPO / "mnemonics.json")
    lmstudio_base: str = field(default_factory=lambda: os.environ.get("LMSTUDIO_BASE",
                                                                      "http://localhost:1234/v1"))
    # Опциональный provider карты переднего плана (recall-модель, fade-aug — PLAN §6.6.13).
    # Сигнатура: prob_provider(rgb: np.ndarray) -> np.ndarray[float32 HxW] в [0,1].
    # None → пайплайн работает БЕЗ модели (чистые правила; бледные линии — потолок).
    prob_provider = None
    cv: CVParams = field(default_factory=CVParams)

    def ensure_out(self) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        return self.out


# Дефолтный singleton для простых запусков; CLI может собрать свой Config.
DEFAULT = Config()

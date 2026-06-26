r"""
auto/ — АВТОНОМНЫЙ image-first векторизатор каротажа (курс v2, см. ../ROADMAP_v2.md).

Принцип (durable, решение заказчика 2026-06-26): НЕ «проход по файлу со сглаживанием»
экспертной трассы. Скрипт САМ анализирует снимок (рамка → сколько линий и переходов
масштаба → карта AUTO/FLAG → 2D-обход штриха) и векторизует в nlgx + las. Эксперт ТОЛЬКО
проверяет финал и точечно разрешает помеченные (FLAG) узлы.

Конвейер:
    скан + имя файла + mnemonics.json
      → frame.detect_frame        (U0) границы треков / верх-низ / сетка — рамка ≠ линии
      → understand.understand     (U1) сколько линий, их цвет/толщина/поведение/start-end,
                                        сколько переходов масштаба у каждой
      → confidence.classify       (U2) по зонам: AUTO (разрешимо) / FLAG (эксперту)
      → trace2d.trace             2D-обход штриха для AUTO-бакета → трасса ~1px
      → emit.emit                 инъекция трасс в лёгкую рамку NeuraLOG → nlgx+bck (+las)

Пакет ПЕРЕИСПОЛЬЗУЕТ проверенные модули из ../digitizer (формат nlgx, QC, декодер уровней,
поведенческие приоры, маски структуры) — не дублируя их. Никакого torch/scipy/skimage:
чистый Python 3.14 + OpenCV/NumPy/Pillow (recall-модель — опциональный provider, см. config).
"""
import sys as _sys
from pathlib import Path as _Path

__version__ = "0.1.0"

# Bootstrap: проверенный фундамент лежит в соседнем ../digitizer как ПЛОСКИЕ модули
# (extract_nlgx, write_nlgx, dataset, decode_levels, behavior_priors, detect_masks, ...),
# которые импортируются по имени. Кладём их каталог на путь один раз (как делает ../run.py).
_DIGITIZER = _Path(__file__).resolve().parent.parent / "digitizer"
if _DIGITIZER.is_dir() and str(_DIGITIZER) not in _sys.path:
    _sys.path.insert(0, str(_DIGITIZER))

del _sys, _Path

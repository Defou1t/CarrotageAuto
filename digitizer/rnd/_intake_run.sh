#!/bin/bash
# _intake_run.sh — ПРИЁМКА ДАННЫХ БЕЗ ПРИСМОТРА: качает оба источника подряд, потом раскладывает.
#
# ⚠ ПЕРЕЗАПУСКАЕМО. Каждый шаг пропускает уже сделанное (файл совпал по имени и размеру, архив уже
# распакован), поэтому скрипт можно оборвать и запустить снова — он продолжит, а не начнёт заново.
# ⚠ СТОРОЖ МЕСТА встроен в загрузчик: ниже 30 ГБ свободного он останавливается сам и пишет об этом.
#
#   bash _intake_run.sh
cd "$(dirname "$0")" || exit 1
LOG=E:/Carrotagki_auto/intake
mkdir -p "$LOG"

echo "[$(date +%H:%M)] === 1/3 чат с Богданом (69 ГБ) ==="
python _tg_fetch.py --chat 416502302 --dst "E:/Carrotagki_auto/from Bodya" \
    --download --jobs 4 --status "$LOG/progress_bodya.txt" >> "$LOG/log_bodya.txt" 2>&1
echo "[$(date +%H:%M)] Богдан: $(ls 'E:/Carrotagki_auto/from Bodya' | wc -l) файлов"

echo "[$(date +%H:%M)] === 2/3 группа (81 ГБ) ==="
python _tg_fetch.py --chat -1002422636760 --dst "E:/Carrotagki_auto/from group" \
    --download --jobs 4 --status "$LOG/progress_group.txt" >> "$LOG/log_group.txt" 2>&1
echo "[$(date +%H:%M)] группа: $(ls 'E:/Carrotagki_auto/from group' 2>/dev/null | wc -l) файлов"

echo "[$(date +%H:%M)] === 3/3 разбор и аудит ==="
"D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe" _intake_sort.py \
    --src "E:/Carrotagki_auto/from Bodya" "E:/Carrotagki_auto/from group" \
          "E:/Carrotagki_auto/from telegram" \
    --report "$LOG/missing.txt" >> "$LOG/log_sort.txt" 2>&1
tail -24 "$LOG/log_sort.txt"
echo "[$(date +%H:%M)] === ГОТОВО. Отчёт: $LOG/log_sort.txt, недостающее: $LOG/missing.txt"

# Бэкап CarrotageAuto в облако через rclone (Google Drive).
# Разовая настройка: winget install Rclone.Rclone ; rclone config  (создать remote "gdrive")
# Запуск:  powershell -File tools\backup_cloud.ps1            # критичное (~1 ГБ)
#          powershell -File tools\backup_cloud.ps1 -Full      # + датасеты v7/v8 (~9 ГБ)
param([switch]$Full)
$R = "gdrive:CarrotageAuto_backup"
$stamp = Get-Date -Format "yyyy-MM-dd"

# 1. Код — уже в GitHub (git push), дублируем bundle одним файлом
git -C F:\nds\Auto bundle create "$env:TEMP\CarrotageAuto_$stamp.bundle" --all
rclone copy "$env:TEMP\CarrotageAuto_$stamp.bundle" "$R/git/"

# 2. Чекпойнты моделей (~350 МБ) — невосстановимы без переобучения
rclone copy F:\nds\output\mk_data "$R/checkpoints/" --include "mk_sep*.pt" -P

# 3. Эталоны (свежие экспертные nlgx+сканы, 0.6 ГБ) — главная ценность
rclone sync F:\nds\projects\Archive\Yatskivska_001 "$R/etalon/Yatskivska_001" -P
rclone sync F:\nds\projects\Archive\Pn_Zavoda_001  "$R/etalon/Pn_Zavoda_001" -P

# 4. Выгрузки qc (лёгкие)
rclone sync F:\nds\output\mk_data\export_qc "$R/export_qc" -P

# 5. Датасеты тайлов (пересобираемы prep_mk из эталонов — только при -Full)
if ($Full) {
  rclone sync F:\nds\output\mk_data_v7 "$R/datasets/mk_data_v7" -P
  rclone sync F:\nds\output\mk_data_v8 "$R/datasets/mk_data_v8" -P
}
Write-Output "Бэкап завершён -> $R"

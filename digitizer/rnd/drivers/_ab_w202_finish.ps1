# Ждёт, пока идущий прогон `_trace_prod_ab.py --shard 11/12` (PID 5604, запущен 12.08 20:25)
# допишет `ab_11of12.pkl`, и сразу собирает сводку по всем 12 дампам в `ab_w202_final.log`.
# Ждать до суток; если процесс оборвётся, дампа не будет и сторож просто истечёт.
# Запускать ИЗ ОБЫЧНОГО ОКНА (не из сессии агента — там дочерние процессы убиваются):
#   powershell -NoProfile -File F:\nds\output\taskS\_ab_w202_finish.ps1
$pkl = 'F:\nds\output\taskS\ab_w202\ab_11of12.pkl'
$log = 'F:\nds\output\taskS\ab_w202_final.log'
$py  = 'D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
for ($i = 0; $i -lt 288 -and -not (Test-Path $pkl); $i++) { Start-Sleep -Seconds 300 }
if (-not (Test-Path $pkl)) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm') дампа так и нет — шард оборвался, нужен перезапуск --shard 11/12" |
        Out-File $log -Encoding utf8
    exit 1
}
Start-Sleep -Seconds 20          # дать дописать файл до конца
Set-Location 'F:\nds\Auto\digitizer\rnd'
& $py _trace_ab_sum.py --dir 'F:\nds\output\taskS\ab_w202' 2>&1 | Out-File $log -Encoding utf8
"" | Out-File $log -Append -Encoding utf8
"★ собрано $(Get-Date -Format 'yyyy-MM-dd HH:mm'); строку «те же скважины, ВЫДАЧА» в HANDOFF §3A обновить по этой сводке" |
    Out-File $log -Append -Encoding utf8

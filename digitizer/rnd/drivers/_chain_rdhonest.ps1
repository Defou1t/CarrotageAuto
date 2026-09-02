[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# ★★★ ЗАЩИТА ОТ КОНСОЛЬНОГО СИГНАЛА (01.09, §6.164). Прошлый запуск умер с кодом 0xC000013A =
# STATUS_CONTROL_C_EXIT: снос сессии агента разослал консольное событие, и задача планировщика
# погибла вместе с ним, хотя правило 8 HANDOFF считало планировщик безопасным путём. Шарды к тому
# моменту досчитали и уцелели, погиб именно ДРАЙВЕР — и фолды 3-4 не начались.
# SetConsoleCtrlHandler(NULL, TRUE) отключает реакцию ЭТОГО процесса на Ctrl+C/Ctrl+Break.
# ⚠ Значит Ctrl+C прогон больше не остановит. Остановка — только явно:
#     Stop-ScheduledTask -TaskName <имя>   (и, если нужно, Get-Process python | Stop-Process)
Add-Type -Namespace NdsWin -Name Ctrl -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError=true)] public static extern bool SetConsoleCtrlHandler(IntPtr handler, bool add);
'@
[void][NdsWin.Ctrl]::SetConsoleCtrlHandler([IntPtr]::Zero, $true)
# Chain: wait for nds_pickmodel_f1f4 to finish, verify it really completed, then run the honest
# decoder-alone pass. Runs as its own scheduled task, so it does not die with the agent session
# (HANDOFF rule 8). Never starts a second run alongside the first - that is the §6.106 pathology.
$log = 'F:\nds\output\taskS\_rdhonest_chain.log'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m |
    Out-File -FilePath $log -Encoding utf8 -Append }
Say 'chain started, waiting for nds_pickmodel_f1f4'

$deadline = (Get-Date).AddHours(20)
while ((Get-Date) -lt $deadline) {
    $t = Get-ScheduledTask -TaskName 'nds_pickmodel_f1f4' -ErrorAction SilentlyContinue
    if (-not $t -or $t.State -ne 'Running') { break }
    Start-Sleep -Seconds 300
}

# Completion is judged by DUMPS, not by the task state: a task can end without finishing the work.
$missing = @()
foreach ($f in 0..4) {
    $n = @(Get-ChildItem "F:\nds\output\taskS\ab_pickmodel\f$f\ab_*of8.pkl" -ErrorAction SilentlyContinue).Count
    if ($n -ne 8) { $missing += "f${f}:$n/8" }
}
if ($missing.Count -gt 0) {
    Say ("pickmodel NOT complete ({0}) - honest run NOT started, machine left alone" -f ($missing -join ' '))
    exit
}
Say 'pickmodel complete (5 folds x 8 dumps)'

$busy = @(Get-Process python -ErrorAction SilentlyContinue).Count
if ($busy -gt 0) {
    Say "still $busy python processes running - honest run NOT started"
    exit
}
Say 'machine free, starting _rdhonest_run.ps1'
& 'F:\nds\output\taskS\_rdhonest_run.ps1' *>&1 | Out-File -FilePath $log -Encoding utf8 -Append
Say 'honest run finished'

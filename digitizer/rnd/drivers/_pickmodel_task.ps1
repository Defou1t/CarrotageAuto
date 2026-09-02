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
# Wrapper for Task Scheduler: runs the resumable pickmodel driver and tees its output to a log.
# Launched via schtasks so it survives teardown of the agent session (HANDOFF rule 8).
$log = 'F:\nds\output\taskS\_pickmodel_f1f4.log'
"START {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Out-File -FilePath $log -Encoding utf8
& 'F:\nds\output\taskS\_pickmodel_run.ps1' *>&1 | Out-File -FilePath $log -Encoding utf8 -Append
"DONE {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Out-File -FilePath $log -Encoding utf8 -Append

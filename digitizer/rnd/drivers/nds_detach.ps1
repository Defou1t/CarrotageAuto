# nds_detach.ps1 — запустить .ps1-драйвер задачей планировщика, ВНЕ дерева процессов Claude Code (20.09).
#
# ЗАЧЕМ. Драйверы и всё, что они порождают (шарды, обучение), запущенные из инструмента Claude Code через
# Start-Process, живут в его job-объекте и гибнут при перезапуске приложения: 20.09 00:17 так умерли
# обучение bg1 и три драйвера; 18.09 09:17 / 09:44 / 10:21 — те же «молчаливые смерти» без кода выхода.
# Задача планировщика запускается службой, дерево отдельное; wscript прячет окно (_run_hidden.vbs).
# Использование: powershell -File nds_detach.ps1 -Name nds_slot_ab -Script F:\nds\output\taskS\_slot_ab_run.ps1
#   -Name   — имя задачи (пересоздаётся, если есть)
#   -Script — полный путь к драйверу
# ⚠ UTF-8 С BOM.
param([Parameter(Mandatory=$true)][string]$Name, [Parameter(Mandatory=$true)][string]$Script)
$vbs = 'F:\nds\output\taskS\_run_hidden.vbs'
$action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument ('"' + $vbs + '" "' + $Script + '"') -WorkingDirectory 'F:\nds\output\taskS'
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddYears(10)     # триггер формальный: запускаем вручную
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null
Start-ScheduledTask -TaskName $Name
Start-Sleep -Seconds 3
$i = Get-ScheduledTaskInfo -TaskName $Name
"{0}: state={1} last={2}" -f $Name, (Get-ScheduledTask -TaskName $Name).State, $i.LastRunTime

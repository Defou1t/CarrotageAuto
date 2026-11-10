# _shutdown_when_done.ps1 — ПО ПРЯМОЙ ПРОСЬБЕ ВЛАДЕЛЬЦА (10.11 ~05:25): «Выключи пк как закончатся все задачи тут и в диалоге с PHD».
# Ждёт: (1) конца A/B §6.357 (`=== SEGVOTE AB DONE ===` в _segvote_ab.log); (2) метки диалога CarrotageAuto `_session_done.txt`
# (итоги записаны) — не дольше 4 ч после (1); (3) тяжёлых процессов CarrotageAuto нет; (4) диалог «Структура репозиторію диплому та PhD»
# (F:\Base, транскрипт f661d5e2…) молчит ≥ 30 мин и его процессов (python / powershell / bash со скретчем cb1bf0a637fe807b) нет;
# bash без python после 3 ч тишины не держит. Затем `shutdown /s /t 300` (отмена — `shutdown /a`). Однократный: НЕ в nds_keepalive —
# после перезагрузки не оживает. ⚠ UTF-8 С BOM.
$ts = 'F:/nds/output/taskS'
$log = "$ts/_shutdown.log"
$phd = 'C:/Users/Defou1t/.claude/projects/F--Base/f661d5e2-7c9a-4911-a0de-99e357159768.jsonl'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Has($f, $pat) { (Test-Path $f) -and (Select-String -Path $f -Pattern $pat -SimpleMatch -Quiet) }
Say "сторож выключения запущен (PID $PID): жду A/B §6.357, итоги диалога CarrotageAuto и тишину диалога PhD"
while (-not (Has "$ts/_segvote_ab.log" '=== SEGVOTE AB DONE ===')) { Start-Sleep -Seconds 120 }
$t0 = Get-Date
Say "A/B §6.357 закончен; жду метку _session_done.txt (не дольше 4 ч)"
while (-not (Test-Path "$ts/_session_done.txt") -and ((Get-Date) - $t0).TotalHours -lt 4) { Start-Sleep -Seconds 60 }
Say ("метка итогов: {0}" -f $(if (Test-Path "$ts/_session_done.txt") { 'есть' } else { 'нет — 4 ч прошло' }))
$lastT = (Get-Date).AddHours(-1)
while ($true) {
  $P = @(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine })
  $nds = @($P | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match '_trace_cache\.py|_strict_stand\.py|_name_cost_prod\.py|_strict_ab\.py|merge_ab3\.py|_ab_decide\.py|_sample_review\.py|_lvl_kind\.py|_replay_missing\.py' }).Count
  $mine = @($P | Where-Object { $_.Name -ne 'claude.exe' -and $_.CommandLine -match 'cb1bf0a637fe807b|f661d5e2' })
  $py = @($mine | Where-Object { $_.Name -ne 'bash.exe' }).Count
  $sh = @($mine | Where-Object { $_.Name -eq 'bash.exe' }).Count
  $idle = if (Test-Path $phd) { ((Get-Date) - (Get-Item $phd).LastWriteTime).TotalMinutes } else { 9999 }
  $st = "CarrotageAuto тяжёлых {0}; PhD: тишина {1:N0} мин, процессов {2} (+ bash {3})" -f $nds, $idle, $py, $sh
  if (((Get-Date) - $lastT).TotalMinutes -ge 30) { Say $st; $lastT = Get-Date }
  if ($nds -eq 0 -and $idle -ge 30 -and $py -eq 0 -and ($sh -eq 0 -or $idle -ge 180)) { break }
  Start-Sleep -Seconds 120
}
Say "все задачи закончены — выключение через 5 мин (отмена: shutdown /a)"
& shutdown.exe /s /t 300 /c "Задачи CarrotageAuto и PhD закончены - выключение через 5 минут (отмена: shutdown /a)"
Say ("shutdown: код {0}" -f $LASTEXITCODE)

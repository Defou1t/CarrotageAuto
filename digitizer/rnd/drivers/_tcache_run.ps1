# _tcache_run.ps1 — сборка КЭША ТРАСС на поле 1123 + держанном сорте A 311 (§6.218), затем побайтная сверка повтора с A/B.
#
# Очередь: после проверки сорта A (=== HOLDA DONE ===). Сборка — `_trace_cache.py build --fast` (ускоренный селектор,
# выдача побайтно та же) в 4 шардах, партии <= 6 ч, пропуск готовых листов; супервизор перезапускает шард, пока тот не
# завершится кодом 0. Затем повтор G / RA / N на поле и сверка с ab_slot/G, ab_slot/RA, ab_names/N по всем 1123 листам —
# это одновременно проверка кэша и быстрого селектора на всём поле. ⚠ UTF-8 С BOM, шаблоны ожидания — ASCII.
# ★ 25.09 11:59: драйвер заменён на ходу — ранний дескриптор процесса (иначе ExitCode пуст и готовый шард
#   перезапускался бы до 40 раз) и подхват живых шардов прежнего драйвера.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log='F:\nds\output\taskS\_tcache.log'
$env:OMP_NUM_THREADS='4'
$N=4
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
# ★ 25.09: ОДИН ЭКЗЕМПЛЯР. 12:24 сторож перезапустил задачу в окне замены драйвера, и два драйвера кэша с 18:00 держали по
#   комплекту шардов — дубли считали одни листы и дрались за один .tmp (PermissionError, шард падал кодом 1).
#   Второй экземпляр не запускает ничего: ждёт конца первого и выходит, если тот дописал маркер конца.
$self = Split-Path -Leaf $MyInvocation.MyCommand.Path
function Others { @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like "*$self*" -and $_.CommandLine -notlike "*nds_detach*" }) }
if ((Others).Count -gt 0) {
  Say "⚠ уже работает экземпляр $((Others)[0].ProcessId) этого драйвера — второй не запускаю, жду его конца"
  while ((Others).Count -gt 0) { Start-Sleep -Seconds 60 }
  if ((Test-Path $log) -and (Select-String -Path $log -Pattern '=== TCACHE DONE ===' -SimpleMatch -CaseSensitive -Quiet)) { exit 0 }
  Say "первый экземпляр завершился без маркера конца — продолжаю сам"
}
Say "жду конца проверки сорта A (_holdA_ab.log: === HOLDA DONE ===)"
while (-not ((Test-Path 'F:\nds\output\taskS\_holdA_ab.log') -and (Select-String -Path 'F:\nds\output\taskS\_holdA_ab.log' -Pattern '=== HOLDA DONE ===' -SimpleMatch -CaseSensitive -Quiet))) { Start-Sleep -Seconds 300 }
Say "старт сборки кэша: 1434 листа, $N шардов"
New-Item -ItemType Directory -Force -Path "$ts/tcache/logs" | Out-Null
$live=@{}; $runs=@{}; $done=@{}; $adopted=@{}
while ($done.Count -lt $N) {
  foreach ($i in 0..($N-1)) {
    if ($done.ContainsKey($i)) { continue }
    $p=$live[$i]
    if ($p -and -not $p.HasExited) { continue }
    if ($p) {
      if ($adopted[$i]) { $adopted.Remove($i); Say "кэш: подхваченный шард $i завершился (код неизвестен) — контрольный перезапуск (готовые листы пропустит)" }
      elseif ($p.ExitCode -eq 0) { $done[$i]=$true; Say "кэш: шард $i готов"; continue }
      else { Say "кэш: шард $i вышел кодом $($p.ExitCode) — перезапуск" }
    } else {
      # ★ 25.09: ЧУЖОЙ ЖИВОЙ ШАРД (драйвер заменён, шард его пережил) — не дублировать, ждать
      $orphan = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*_trace_cache.py build*" -and $_.CommandLine -like "*--shard $i/$N*" -and $_.CommandLine -like "*$ts/tcache *" })
      if ($orphan.Count -gt 0) { $live[$i] = Get-Process -Id $orphan[0].ProcessId; $adopted[$i]=$true; Say "кэш: шард $i уже считается процессом $($orphan[0].ProcessId) — жду его"; continue }
    }
    if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
    if ($runs[$i] -ge 40) { $done[$i]=$true; Say "⛔ кэш: шард $i — 40 запусков исчерпаны"; continue }
    $runs[$i]++; $r=$runs[$i]
    $ar=@('_trace_cache.py','build','--sheets','tcache_sheets.txt','--cache',"$ts/tcache",'--shard',"$i/$N",'--max-hours','6','--fast')
    $live[$i] = Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$ts/tcache/logs/sh$i.r$r.log" -RedirectStandardError "$ts/tcache/logs/sh$i.r$r.err" -PassThru
    $null = $live[$i].Handle     # ★ 25.09: без раннего дескриптора PS 5.1 при перенаправлении вывода отдаёт ПУСТОЙ ExitCode
    Say "кэш: шард $i — запуск $r"
  }
  Start-Sleep -Seconds 60
}
$n = @(Get-ChildItem "$ts/tcache/*.pkl").Count
Say "кэш собран: файлов $n"
Say "повтор G / RA / N на поле"
& $py "$rnd/_trace_cache.py" 'replay' '--sheets' 'wellmap_sheets.txt' '--cache' "$ts/tcache" '--out' "$ts/rp_field" '--mode' 'G:rdpick=3' '--mode' 'RA:rdpick=3,slotlen=0.18,slotall=1' '--mode' 'N:rdpick=3,slotlen=0.18,slotall=1,slotgeom=1' 2>&1 | Out-File -FilePath "$ts/tcache_replay.txt" -Encoding utf8
& $py "$rnd/_tcache_parity.py" '--out' "$ts/rp_field" 'G=ab_slot/G' 'RA=ab_slot/RA' 'N=ab_names/N' 2>&1 | Out-File -FilePath "$ts/tcache_parity.txt" -Encoding utf8
if ($LASTEXITCODE -eq 0) { Say "★★ СВЕРКА: повтор с кэша побайтно равен A/B на всём поле (tcache_parity.txt)" } else { Say "⛔ СВЕРКА: есть различия — см. tcache_parity.txt" }
Say "=== TCACHE DONE ==="

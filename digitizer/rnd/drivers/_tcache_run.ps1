# _tcache_run.ps1 — сборка КЭША ТРАСС на поле 1123 + держанном сорте A 311 (§6.218), затем побайтная сверка повтора с A/B.
#
# Очередь: после проверки сорта A (=== HOLDA DONE ===). Сборка — `_trace_cache.py build --fast` (ускоренный селектор,
# выдача побайтно та же) в 4 шардах, партии <= 6 ч, пропуск готовых листов; супервизор перезапускает шард, пока тот не
# завершится кодом 0. Затем повтор G / RA / N на поле и сверка с ab_slot/G, ab_slot/RA, ab_names/N по всем 1123 листам —
# это одновременно проверка кэша и быстрого селектора на всём поле. ⚠ UTF-8 С BOM, шаблоны ожидания — ASCII.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log='F:\nds\output\taskS\_tcache.log'
$env:OMP_NUM_THREADS='4'
$N=4
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
Say "жду конца проверки сорта A (_holdA_ab.log: === HOLDA DONE ===)"
while (-not ((Test-Path 'F:\nds\output\taskS\_holdA_ab.log') -and (Select-String -Path 'F:\nds\output\taskS\_holdA_ab.log' -Pattern '=== HOLDA DONE ===' -SimpleMatch -CaseSensitive -Quiet))) { Start-Sleep -Seconds 300 }
Say "старт сборки кэша: 1434 листа, $N шардов"
New-Item -ItemType Directory -Force -Path "$ts/tcache/logs" | Out-Null
$live=@{}; $runs=@{}; $done=@{}
while ($done.Count -lt $N) {
  foreach ($i in 0..($N-1)) {
    if ($done.ContainsKey($i)) { continue }
    $p=$live[$i]
    if ($p -and -not $p.HasExited) { continue }
    if ($p) {
      if ($p.ExitCode -eq 0) { $done[$i]=$true; Say "кэш: шард $i готов"; continue }
      Say "кэш: шард $i вышел кодом $($p.ExitCode) — перезапуск"
    }
    if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
    if ($runs[$i] -ge 40) { $done[$i]=$true; Say "⛔ кэш: шард $i — 40 запусков исчерпаны"; continue }
    $runs[$i]++; $r=$runs[$i]
    $ar=@('_trace_cache.py','build','--sheets','tcache_sheets.txt','--cache',"$ts/tcache",'--shard',"$i/$N",'--max-hours','6','--fast')
    $live[$i] = Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$ts/tcache/logs/sh$i.r$r.log" -RedirectStandardError "$ts/tcache/logs/sh$i.r$r.err" -PassThru
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

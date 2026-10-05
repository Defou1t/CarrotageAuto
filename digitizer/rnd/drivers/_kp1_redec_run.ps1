# _kp1_redec_run.ps1 — к §6.292: декодер с путями K+1 на трек (`rowdec_k_plus=1`, основной порог; основной
# путь отличается от tcache на один путь) → кэш tcache_kp1 (поле alt). 4 шарда, партии ≤ 6 ч, подхват. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_kp1_redec.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
Say "старт redec kp1"
for ($round = 0; $round -lt 6; $round++) {
  $ps=@()
  foreach ($i in 0..3) {
    $a=@('-I','_trace_cache.py','redec','--src',"$ts/tcache_v2",'--cache',"$ts/tcache_kp1",'--sheets','tcache_sheets.txt',
         '--knob','rowdec_k_plus=1','--shard',"$i/4",'--max-hours','6')
    $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$ts/_kp1_redec_$i.log" -RedirectStandardError "$ts/_kp1_redec_$i.err" -PassThru
  }
  $codes=@()
  foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit(); $codes += $p.ExitCode }
  Say ("redec круг {0}: коды {1}" -f $round, ($codes -join ','))
  if (-not ($codes -contains 75)) { break }
}
$n = (Get-ChildItem "$ts/tcache_kp1" -Filter *.pkl).Count
Say "кэш tcache_kp1: $n листов"
Say "=== KP1 REDEC DONE ==="

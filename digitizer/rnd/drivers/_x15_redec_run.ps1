# _x15_redec_run.ps1 — к §6.292: декодер с дополнительными путями при пороге пиков 0.15 (`rowdec_extra_thr=0.15`, та же карта; основной
# путь — как в tcache) → кэш tcache_x15 (поле extra). 4 шарда, партии ≤ 6 ч, подхват. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_x15_redec.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
Say "старт redec x15"
for ($round = 0; $round -lt 6; $round++) {
  $ps=@()
  foreach ($i in 0..3) {
    $a=@('-I','_trace_cache.py','redec','--src',"$ts/tcache_v2",'--cache',"$ts/tcache_x15",'--sheets','tcache_sheets.txt',
         '--knob','rowdec_extra_thr=0.15','--shard',"$i/4",'--max-hours','6')
    $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$ts/_x15_redec_$i.log" -RedirectStandardError "$ts/_x15_redec_$i.err" -PassThru
  }
  $codes=@()
  foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit(); $codes += $p.ExitCode }
  Say ("redec круг {0}: коды {1}" -f $round, ($codes -join ','))
  if (-not ($codes -contains 75)) { break }
}
$n = (Get-ChildItem "$ts/tcache_x15" -Filter *.pkl).Count
Say "кэш tcache_x15: $n листов"
Say "=== X15 REDEC DONE ==="

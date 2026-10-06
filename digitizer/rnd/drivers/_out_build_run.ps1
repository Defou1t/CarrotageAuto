# _out_build_run.ps1 — §6.295: данные выбора трассы на листах ВНЕ поля и сорта A (1299 листов, 119 новых скважин). Сборка кэша
# трасс (прод-ведение + декодер фолда, державшего скважину, + пути второго порога) → повтор NS → скан пар слот × кандидат.
# Каждый шаг < 6 ч, партии с подхватом. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_out_build.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
Say "старт сборки кэша tcache_out"
for ($round = 0; $round -lt 8; $round++) {
  $ps=@()
  foreach ($i in 0..3) {
    $a=@('-I','_trace_cache.py','build','--sheets','outside_sheets.txt','--cache',"$ts/tcache_out",'--fast','--shard',"$i/4",'--max-hours','6')
    $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$ts/_out_build_$i.log" -RedirectStandardError "$ts/_out_build_$i.err" -PassThru
  }
  $codes=@()
  foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit(); $codes += $p.ExitCode }
  Say ("сборка круг {0}: коды {1}" -f $round, ($codes -join ','))
  if (-not (($codes -contains 75) -or ($codes -contains 3))) { break }
}
$n = (Get-ChildItem "$ts/tcache_out" -Filter *.pkl).Count
Say "кэш tcache_out: $n листов"
$base='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v2r,levelshift=1,namestyle=oof:F:/nds/output/taskS/name_style_v1'
$ps=@()
foreach ($i in 0..3) {
  $a=@('-I','_trace_cache.py','replay','--sheets','outside_sheets.txt','--cache',"$ts/tcache_out",'--out',"$ts/rp_lvl",
       '--conf',"$ts/conf_tcache.pkl",'--shard',"$i/4",'--mode',"NS:$base")
  $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_out_rp_NS_$i.log" -RedirectStandardError "$ts/_out_rp_NS_$i.err" -PassThru
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "повтор NS вне поля готов"
$a=@('-I','_line_choice_scan.py','--sheets','outside_sheets.txt','--cache',"$ts/tcache_out",'--dir',"$ts/rp_lvl/NS",'--out',"$ts/line_choice_out",'--workers','3')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
  -RedirectStandardOutput "$ts/_out_scan.log" -RedirectStandardError "$ts/_out_scan.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
Say ("скан: код {0}" -f $p.ExitCode)
Say "=== OUT BUILD DONE ==="

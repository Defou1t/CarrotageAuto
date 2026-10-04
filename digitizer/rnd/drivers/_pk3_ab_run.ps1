# _pk3_ab_run.ps1 — §6.274: порог пиков декодера 0.3 через отгрузку новой мерой. redec (4 шарда, партии ≤ 6 ч, подхват) →
# повтор NSP из tcache_pk3 (4 шарда) → счёт NS против NSP. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_pk3_ab.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
Say "старт redec"
for ($round = 0; $round -lt 6; $round++) {
  $ps=@()
  foreach ($i in 0..3) {
    $a=@('-I','_trace_cache.py','redec','--src',"$ts/tcache_v2",'--cache',"$ts/tcache_pk3",'--sheets','tcache_sheets.txt',
         '--knob','rowdec_peak_thr=0.3','--shard',"$i/4",'--max-hours','6')
    $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$ts/_pk3_redec_$i.log" -RedirectStandardError "$ts/_pk3_redec_$i.err" -PassThru
  }
  $codes=@()
  foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit(); $codes += $p.ExitCode }
  Say ("redec круг {0}: коды {1}" -f $round, ($codes -join ','))
  if (-not ($codes -contains 75)) { break }
}
$n = (Get-ChildItem "$ts/tcache_pk3" -Filter *.pkl).Count
Say "кэш tcache_pk3: $n листов"
$base='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=level_ink_v2.pt,levelshift=1,namestyle=name_style_v1.pkl'
$ps=@()
foreach ($i in 0..3) {
  $a=@('-I','_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/tcache_pk3",'--out',"$ts/rp_lvl",
       '--conf',"$ts/conf_tcache.pkl",'--shard',"$i/4",'--mode',"NSP:$base")
  $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_rp_lvl_NSP_$i.log" -RedirectStandardError "$ts/_rp_lvl_NSP_$i.err" -PassThru
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "повтор NSP готов"
$ps=@()
foreach ($i in 0..3) {
  $a=@('-I','_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/tcache",'--out',"$ts/rp_lvl",
       '--conf',"$ts/conf_tcache.pkl",'--shard',"$i/4",'--mode',"NSB:$base")
  $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_rp_lvl_NSB_$i.log" -RedirectStandardError "$ts/_rp_lvl_NSB_$i.err" -PassThru
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "повтор базы NSB (тот же прод из tcache, модели прода) готов"
$a=@('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NSB','--mode2','NSP','--dump',"$ts/rp_lvl_NSB_NSP.pkl")
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
  -RedirectStandardOutput "$ts/_pk3_ab_score.log" -RedirectStandardError "$ts/_pk3_ab_score.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
Say "=== PK3 AB DONE ==="

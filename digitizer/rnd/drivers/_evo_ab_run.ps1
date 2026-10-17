# _evo_ab_run.ps1 — §6.308: A/B на поле + листах вне поля (протокол --eval-out). NSE6 — v6e на пуле прода; NSE7 — v7e + пути инерции.
# Повторы поля (tcache) и вне поля (tcache_out) по 3 шарда; счёт NS против каждого — параллельно. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_evo_ab.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$common='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v2r,levelshift=1,namestyle=oof:F:/nds/output/taskS/name_style_v1'
function Rep($mode, $lc, $sheets, $cache, $extras) {
  $ps=@()
  foreach ($i in 0..2) {
    $a=@('-I','_trace_cache.py','replay','--sheets',$sheets,'--cache',$cache,'--extra-cache') + $extras + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard',"$i/3",'--mode',"${mode}:$common,linechoice=oof:$lc")
    $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$ts/_rp_${mode}_$($sheets.Substring(0,4))_$i.log" -RedirectStandardError "$ts/_rp_${mode}_$($sheets.Substring(0,4))_$i.err" -PassThru
    Start-Sleep -Seconds 5
  }
  foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
  Say ("повтор {0} {1}: {2} листов в каталоге" -f $mode, $sheets, (Get-ChildItem "$ts/rp_lvl/$mode" -Directory).Count)
}
Say "старт"
Rep 'NSE6' "$ts/line_choice_v6e" 'tcache_sheets.txt' "$ts/tcache" @("$ts/tcache_pk3")
Rep 'NSE6' "$ts/line_choice_v6e" 'outside_sheets.txt' "$ts/tcache_out" @("$ts/tcache_out|extra")
Rep 'NSE7' "$ts/line_choice_v7e" 'tcache_sheets.txt' "$ts/tcache" @("$ts/tcache_pk3","$ts/tcache_inert|extra")
Rep 'NSE7' "$ts/line_choice_v7e" 'outside_sheets.txt' "$ts/tcache_out" @("$ts/tcache_out|extra","$ts/tcache_out_inert|extra")
$s6 = Start-Process -FilePath $py -ArgumentList @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE6','--dump',"$ts/rp_lvl_NS_NSE6.pkl") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_nse6_score.log" -RedirectStandardError "$ts/_nse6_score.err" -PassThru
$s7 = Start-Process -FilePath $py -ArgumentList @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE7','--dump',"$ts/rp_lvl_NS_NSE7.pkl") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_nse7_score.log" -RedirectStandardError "$ts/_nse7_score.err" -PassThru
$null = $s6.Handle; $null = $s7.Handle; $s6.WaitForExit(); $s7.WaitForExit()
Say ("счёт: NSE6 код {0}, NSE7 код {1}" -f $s6.ExitCode, $s7.ExitCode)
Say "=== EVO AB DONE ==="

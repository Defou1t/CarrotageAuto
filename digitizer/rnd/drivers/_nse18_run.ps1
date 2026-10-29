# _nse18_run.ps1 — §6.348: повтор NSE18 (NSE16 + признак сети туши в выборе трассы v18e, сеть вне фолда) → счёт; v18p для прода. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_nse18.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
$common='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v7e,levelcut=oof:F:/nds/output/taskS/cut_cnn_w2,levelshift=1,namestyle=oof:F:/nds/output/taskS/name_style_v1'
function Rep($mode, $lc, $sheets, $cache, $extras) {
  $ps=@()
  foreach ($i in 0..2) {
    $a=@('-I','_trace_cache.py','replay','--sheets',$sheets,'--cache',$cache,'--extra-cache') + $extras + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard',"$i/3",'--mode',"${mode}:$common,linechoice=oof:$lc,lcseg=oof:$ts/ink_seg_f")
    $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$ts/_rp_${mode}_$($sheets.Substring(0,4))_$i.log" -RedirectStandardError "$ts/_rp_${mode}_$($sheets.Substring(0,4))_$i.err" -PassThru
    Start-Sleep -Seconds 5
  }
  foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
  Say ("повтор {0} {1}: {2} листов в каталоге" -f $mode, $sheets, (Get-ChildItem "$ts/rp_lvl/$mode" -Directory).Count)
}
Say "старт"
$sf="$ts/segfeat_f0.pkl;$ts/segfeat_f1.pkl;$ts/segfeat_o0.pkl;$ts/segfeat_o1.pkl"
$p18 = Start-Process -FilePath $py -ArgumentList @('-I','_line_choice_assign.py','--data',"$ts/line_choice_s",'--data-extra',"$ts/line_choice_out_s",'--feat','stack','--extra','--grid','hung:0.05:0.01','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn_s.pkl",'--segf',$sf,'--prod-all','--save-dir',"$ts/line_choice_v18p",'--dump',"$ts/lca_v18p.pkl") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_lca_v18p.log" -RedirectStandardError "$ts/_lca_v18p.err" -PassThru
Rep 'NSE18' "$ts/line_choice_v18e" 'tcache_sheets.txt' "$ts/tcache" @("$ts/tcache_pk3","$ts/tcache_inert|extra","$ts/tcache_seg|alt","$ts/tcache_seg|extra")
Rep 'NSE18' "$ts/line_choice_v18e" 'outside_sheets.txt' "$ts/tcache_out" @("$ts/tcache_out|extra","$ts/tcache_out_inert|extra","$ts/tcache_out_seg|alt","$ts/tcache_out_seg|extra")
Run 'nse18_score' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE18','--dump',"$ts/rp_lvl_NS_NSE18.pkl")
$null = $p18.Handle; $p18.WaitForExit(); Say ("lca_v18p: код {0}" -f $p18.ExitCode)
Say "=== NSE18 DONE ==="

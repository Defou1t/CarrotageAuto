$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_nse14.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
$common='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v5e,levelshift=1,namestyle=oof:F:/nds/output/taskS/name_style_v1'
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
Run 'tpatch_il_f' @('-I','_trace_patch.py','--data',"$ts/line_choice_il",'--cache',"$ts/tcache",'--extra',"$ts/tcache_pk3|alt;$ts/tcache_il|extra",'--out',"$ts/tpatch_il",'--workers','4')
Run 'tpatch_il_o' @('-I','_trace_patch.py','--data',"$ts/line_choice_out_il",'--cache',"$ts/tcache_out",'--extra',"$ts/tcache_out|extra;$ts/tcache_out_il|extra",'--out',"$ts/tpatch_il",'--workers','4')
New-Item -ItemType Directory -Force "$ts/line_choice_v14p" | Out-Null
Run 'trace_cnn_il' @('-I','_trace_cnn.py','--epochs','8','--data',"$ts/line_choice_il","$ts/line_choice_out_il",'--patch',"$ts/tpatch_il",'--save-dir',"$ts/line_choice_v14e",'--prod-out',"$ts/line_choice_v14p/trace_cnn_all.pt",'--out',"$ts/trace_cnn_il.pkl")
$base=@('-I','_line_choice_assign.py','--data',"$ts/line_choice_il",'--data-extra',"$ts/line_choice_out_il",'--feat','stack','--extra','--grid','hung:0.07:0.03','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn_il.pkl")
Run 'lca_v14e' ($base + @('--save-dir',"$ts/line_choice_v14e",'--dump',"$ts/lca_v14e.pkl"))
Run 'lca_v14p' ($base + @('--prod-all','--save-dir',"$ts/line_choice_v14p",'--dump',"$ts/lca_v14p.pkl"))
Rep 'NSE14' "$ts/line_choice_v14e" 'tcache_sheets.txt' "$ts/tcache" @("$ts/tcache_pk3","$ts/tcache_il|extra")
Rep 'NSE14' "$ts/line_choice_v14e" 'outside_sheets.txt' "$ts/tcache_out" @("$ts/tcache_out|extra","$ts/tcache_out_il|extra")
Run 'nse14_score' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE14','--dump',"$ts/rp_lvl_NS_NSE14.pkl")
Say "=== NSE14 DONE ==="
